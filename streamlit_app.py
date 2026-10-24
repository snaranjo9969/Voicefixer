"""VoiceFixer: interfaz web con Streamlit.

Coloca este archivo en la RAÍZ del repositorio (junto a setup.py y la carpeta
voicefixer/). Así Python importa el paquete local `voicefixer` sin instalarlo.

Local con GPU NVIDIA:  pip install -r requirements-cuda.txt
Local o nube sin GPU:  pip install -r requirements.txt
Arrancar:              streamlit run streamlit_app.py
"""

import ctypes
import gc
import hashlib
import http.client
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch

SAMPLE_RATE = 44100
CHUNK_SECONDS_GPU = 30  # trozo nativo del modelo; una GPU con mucha VRAM lo procesa de una vez
CHUNK_SECONDS_CPU = 5   # trozos pequeños para no agotar la RAM (p. ej. ~2,7 GB en Streamlit Cloud)
OVERLAP_SECONDS = 0.25  # fundido cruzado entre trozos para que no se noten los cortes

MODE_LABELS = {
    0: "Modo 0 · original",
    1: "Modo 1 · preprocesado",
    2: "Modo 2 · train",
}

# Mismas rutas que usa la librería. MD5 oficiales de https://zenodo.org/records/5600188
_CACHE = Path(os.path.expanduser("~")) / ".cache" / "voicefixer"
WEIGHTS = [
    {
        "label": "vf.ckpt (análisis, ~490 MB)",
        "path": _CACHE / "analysis_module" / "checkpoints" / "vf.ckpt",
        "md5": "1f479dabb9fa5724de5e20f2164cbc7a",
        "urls": [
            "https://zenodo.org/records/5600188/files/vf.ckpt?download=1",
            "https://huggingface.co/teamup-tech/VoiceFixer-weights/resolve/main/vf.ckpt",
        ],
    },
    {
        "label": "model.ckpt-1490000_trimed.pt (vocoder, ~136 MB)",
        "path": _CACHE / "synthesis_module" / "44100" / "model.ckpt-1490000_trimed.pt",
        "md5": "ac6de16561dd6042daf8bb3a806dc818",
        "urls": [
            "https://zenodo.org/records/5600188/files/model.ckpt-1490000_trimed.pt?download=1",
            "https://huggingface.co/teamup-tech/VoiceFixer-weights/resolve/main/model.ckpt-1490000_trimed.pt",
        ],
    },
]

st.set_page_config(page_title="VoiceFixer", page_icon="🎙️", layout="wide")


# --------------------------------------------------------------------------- pesos

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def download(url, part, bar, label):
    """Descarga a `part`, reanudando si ya hay un trozo. Devuelve True si reanudó."""
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "voicefixer-streamlit/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # el .part ya estaba completo
            return True
        raise

    with resp:
        resumed = bool(have) and resp.status == 206
        if not resumed:
            have = 0
        total = have + int(resp.headers.get("Content-Length") or 0)
        done, last = have, 0.0
        with open(part, "ab" if resumed else "wb") as f:
            while block := resp.read(1 << 20):
                f.write(block)
                done += len(block)
                if time.time() - last > 0.25:
                    last = time.time()
                    frac = min(done / total, 1.0) if total else 0.0
                    bar.progress(frac, text=f"Descargando {label}: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    if total and done < total:
        # Si el servidor corta la conexión, read() termina sin error: hay que comprobarlo.
        raise ConnectionError(f"descarga cortada en {done / 1e6:.0f} de {total / 1e6:.0f} MB")
    return resumed


def fetch(w, bar):
    """Descarga un checkpoint probando cada URL y lo valida por MD5 antes de usarlo."""
    dest = w["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in w["urls"]:
        host = urllib.parse.urlparse(url).netloc
        for attempt in range(3):
            try:
                resumed = download(url, part, bar, w["label"])
            except urllib.error.HTTPError as e:
                errors.append(f"{host}: HTTP {e.code}")
                if e.code in (408, 429) or e.code >= 500:
                    time.sleep(3 * (attempt + 1))
                    continue  # error temporal del servidor: reintentar
                break  # 404, 403…: probar el siguiente servidor
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                errors.append(f"{host}: {e}")
                time.sleep(3 * (attempt + 1))
                continue  # reintenta; el .part se reanuda
            bar.progress(1.0, text=f"Verificando {w['label']}…")
            if md5_file(part) == w["md5"]:
                os.replace(part, dest)
                return
            part.unlink()
            errors.append(f"{host}: el archivo descargado no coincide con el MD5 oficial")
            if not resumed:
                break  # contenido distinto en este servidor: probar el siguiente
    raise RuntimeError("\n".join(errors[-4:]))


@st.cache_resource
def _weights_state():
    return {"ok": False, "lock": threading.Lock()}


def ensure_weights():
    """Comprueba (una vez por arranque) que los pesos están completos; si no, los repara."""
    state = _weights_state()
    if state["ok"]:
        return
    with state["lock"]:  # dos pestañas no deben descargar el mismo archivo a la vez
        if state["ok"]:
            return
        status = st.status("Comprobando los pesos del modelo…", expanded=True)
        bar = status.progress(0.0)
        for w in WEIGHTS:
            if w["path"].exists():
                bar.progress(0.0, text=f"Verificando {w['label']}…")
                if md5_file(w["path"]) == w["md5"]:
                    continue
                status.warning(f"{w['label']} estaba incompleto o dañado. Se descarga de nuevo.")
                w["path"].unlink()
            try:
                fetch(w, bar)
            except RuntimeError as e:
                status.update(label="No se pudieron descargar los pesos", state="error")
                st.error(
                    f"No se pudo descargar **{w['label']}**.\n\n```\n{e}\n```\n\n"
                    f"Descárgalo a mano desde [Zenodo](https://zenodo.org/records/5600188) "
                    f"y guárdalo en:\n\n`{w['path']}`\n\nLuego recarga la página."
                )
                st.stop()
        status.update(label="Pesos del modelo verificados", state="complete", expanded=False)
        state["ok"] = True


# --------------------------------------------------------------------------- modelo

@st.cache_resource(show_spinner="Cargando el modelo en memoria…")
def load_model():
    # torch >= 2.6 usa weights_only=True por defecto y los checkpoints de VoiceFixer
    # (2021) pueden no cargar así. Los pesos vienen del Zenodo oficial del autor.
    _torch_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _load

    # ensure_weights() ya dejó los pesos verificados, así que la librería no descarga nada.
    from voicefixer import VoiceFixer

    model = VoiceFixer()

    # El modo 2 pone el modelo en train(); con momentum=0 la BatchNorm usa las
    # estadísticas del lote pero no altera las guardadas, que usan los modos 0 y 1.
    for m in model._model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0

    # Devuelve al sistema la RAM temporal usada al leer los checkpoints (Linux).
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    return model


@st.cache_resource
def model_lock():
    # Un solo modelo compartido entre pestañas/usuarios: inferencias de una en una.
    return threading.Lock()


def gpu_supported(major, minor):
    """¿Trae este build de PyTorch kernels para la arquitectura de la GPU?"""
    for arch in torch.cuda.get_arch_list():
        kind, _, num = arch.partition("_")
        if not num.isdigit():
            continue
        a_major, a_minor = int(num[:-1]), int(num[-1])
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True
    return False


def restore(model, audio, mode, cuda, chunk_seconds, on_chunk):
    """Restaura `audio` por trozos con fundido cruzado. Sin límite de duración."""
    chunk = int(chunk_seconds * SAMPLE_RATE)
    overlap = int(OVERLAP_SECONDS * SAMPLE_RATE)
    hop = chunk - overlap
    starts = list(range(0, max(len(audio) - overlap, 1), hop))
    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    out = np.zeros(len(audio), dtype=np.float32)

    for i, start in enumerate(starts):
        seg = audio[start : start + chunk]
        # restore_inmem mueve el modelo a GPU o CPU según `cuda`
        y = model.restore_inmem(seg, mode=mode, cuda=cuda).reshape(-1).astype(np.float32)
        y = librosa.util.fix_length(y, size=len(seg))
        if start > 0:
            n = min(overlap, len(seg))
            y[:n] *= fade_in[:n]
            out[start : start + n] *= 1.0 - fade_in[:n]
        out[start : start + len(seg)] += y
        on_chunk(i + 1, len(starts))

    peak = np.abs(out).max()
    return out / peak if peak > 1.0 else out


def output_name(file_name, mode):
    return f"{Path(file_name).stem}_voicefixer_modo{mode}.wav"


def build_zip(results, names):
    buf = BytesIO()
    used = set()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for r in results:
            name = output_name(names[r["file_idx"]], r["mode"])
            if name in used:  # dos archivos subidos con el mismo nombre
                name = f"{r['file_idx'] + 1:02d}_{name}"
            used.add(name)
            zf.writestr(name, r["wav"])
    return buf.getvalue()


def fmt_secs(s):
    m, s = divmod(int(round(s)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


# --------------------------------------------------------------------------- barra lateral

with st.sidebar:
    st.header("Dispositivo")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        st.success(f"**{props.name}**  \n{props.total_memory / 1024**3:.0f} GB VRAM · sm_{major}{minor}")
        st.caption(f"PyTorch {torch.__version__} · CUDA {torch.version.cuda}")
        if gpu_supported(major, minor):
            use_gpu = st.toggle("Usar GPU", value=True)
        else:
            st.error(
                f"Esta versión de PyTorch no trae soporte para sm_{major}{minor}. "
                "Reinstala con `pip install -r requirements-cuda.txt` (CUDA 13.0). "
                "Mientras tanto se usará la CPU."
            )
            use_gpu = False
    else:
        st.info("No hay GPU CUDA disponible: se usará la CPU (bastante más lento).")
        if torch.version.cuda is None:
            st.caption(
                f"PyTorch {torch.__version__} es la versión solo CPU. Si tienes una GPU "
                "NVIDIA, instala `requirements-cuda.txt`."
            )
        use_gpu = False

    chunk_seconds = CHUNK_SECONDS_GPU if use_gpu else CHUNK_SECONDS_CPU
    st.caption(f"Se procesa en trozos de {chunk_seconds} s, sin límite de duración.")


# --------------------------------------------------------------------------- página

st.title("🎙️ VoiceFixer")
st.caption(
    "Restaura voz degradada (ruido, reverberación, baja resolución, clipping). "
    "Basado en [haoheliu/voicefixer](https://github.com/haoheliu/voicefixer)."
)

ensure_weights()
voice_fixer = load_model()

files = st.file_uploader(
    "Sube uno o varios audios con voz",
    type=["wav", "flac", "mp3", "ogg"],
    accept_multiple_files=True,
)

mode_choice = st.radio(
    "Modo",
    options=[0, 1, 2, "todos"],
    format_func=lambda m: {
        0: "0 · Original (recomendado)",
        1: "1 · Preprocesado (quita altas frecuencias)",
        2: "2 · Train (a veces mejor en voz muy dañada)",
        "todos": "Todos · comparar 0, 1 y 2",
    }[m],
    horizontal=True,
)
modes = [0, 1, 2] if mode_choice == "todos" else [mode_choice]

names = [f.name for f in files]
signature = tuple((f.name, f.size) for f in files)
has_results = bool(files) and st.session_state.get("signature") == signature

if files and not has_results:
    with st.expander(f"Escuchar originales ({len(files)})", expanded=len(files) == 1):
        for f in files:
            st.caption(f.name)
            st.audio(f)

if files:
    label = "Restaurar audio" if len(files) == 1 else f"Restaurar {len(files)} audios"
    if len(modes) > 1:
        label += " en los 3 modos"
    if st.button(label, type="primary"):
        results = []
        total_units = len(files) * len(modes)
        bar = st.progress(0.0, text="En cola…")
        t_all = time.time()

        with model_lock(), torch.no_grad():  # no_grad es por hilo: va aquí
            for fi, f in enumerate(files):
                bar.progress(fi * len(modes) / total_units, text=f"Leyendo {f.name}…")
                f.seek(0)
                audio, _ = librosa.load(f, sr=SAMPLE_RATE, mono=True)

                for mi, m in enumerate(modes):
                    unit = fi * len(modes) + mi

                    def on_chunk(i, n, unit=unit, name=f.name, m=m):
                        bar.progress(
                            min((unit + i / n) / total_units, 1.0),
                            text=f"{name} · {MODE_LABELS[m]} · trozo {i}/{n}",
                        )

                    t0 = time.time()
                    y = restore(voice_fixer, audio, m, use_gpu, chunk_seconds, on_chunk)
                    wav = BytesIO()
                    sf.write(wav, y, samplerate=SAMPLE_RATE, format="WAV")
                    results.append({
                        "file_idx": fi,
                        "mode": m,
                        "wav": wav.getvalue(),
                        "elapsed": time.time() - t0,
                        "duration": len(audio) / SAMPLE_RATE,
                    })
                del audio

        if use_gpu:
            torch.cuda.empty_cache()
        bar.empty()

        st.session_state.update(
            signature=signature,
            results=results,
            zip=build_zip(results, names) if len(results) > 1 else None,
            total_time=time.time() - t_all,
        )
        has_results = True

# --------------------------------------------------------------------------- resultados

if has_results:
    results = st.session_state["results"]
    st.subheader("Resultados")

    head = st.columns([3, 1])
    head[0].caption(
        f"{len(files)} archivo(s) · {len(results)} salida(s) · "
        f"procesado en {fmt_secs(st.session_state['total_time'])}"
    )
    if st.session_state["zip"]:
        head[1].download_button(
            "Descargar todo (ZIP)",
            data=st.session_state["zip"],
            file_name="voicefixer_resultados.zip",
            mime="application/zip",
            type="primary",
            width="stretch",
        )

    for fi, f in enumerate(files):
        file_results = [r for r in results if r["file_idx"] == fi]
        if not file_results:
            continue
        with st.expander(
            f"{f.name} · {fmt_secs(file_results[0]['duration'])}",
            expanded=len(files) <= 3,
        ):
            cols = st.columns(1 + len(file_results))
            with cols[0]:
                st.markdown("**Original**")
                st.audio(f)
            for col, r in zip(cols[1:], file_results):
                with col:
                    st.markdown(f"**{MODE_LABELS[r['mode']]}**")
                    st.audio(r["wav"], format="audio/wav")
                    st.caption(f"Tiempo: {r['elapsed']:.1f} s")
                    st.download_button(
                        "Descargar WAV",
                        data=r["wav"],
                        file_name=output_name(f.name, r["mode"]),
                        mime="audio/wav",
                        key=f"dl_{fi}_{r['mode']}",
                    )
