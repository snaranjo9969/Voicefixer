"""VoiceFixer en Streamlit Community Cloud.

Coloca este archivo en la RAÍZ del repositorio (junto a setup.py y la carpeta
voicefixer/). Así Python importa el paquete local `voicefixer` sin instalarlo.
"""

import ctypes
import gc
import threading
import time
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch

SAMPLE_RATE = 44100
MAX_SECONDS = 60      # en la CPU gratuita cada minuto de audio tarda varios minutos
CHUNK_SECONDS = 5     # trozos pequeños: mantienen la RAM por debajo del límite (~2,7 GB)
OVERLAP_SECONDS = 0.25

st.set_page_config(page_title="VoiceFixer", page_icon="🎙️")
st.title("🎙️ VoiceFixer")
st.caption(
    "Restaura voz degradada (ruido, reverberación, baja resolución, clipping). "
    "Basado en [haoheliu/voicefixer](https://github.com/haoheliu/voicefixer)."
)


@st.cache_resource(show_spinner="Cargando modelos… la primera vez se descargan (puede tardar unos minutos).")
def load_model():
    # torch >= 2.6 usa weights_only=True por defecto y los checkpoints de VoiceFixer
    # (2021) pueden no cargar así. Los pesos vienen del Zenodo oficial del autor.
    _torch_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _load

    # La importación descarga los pesos desde Zenodo a ~/.cache/voicefixer si no existen.
    from voicefixer import VoiceFixer

    model = VoiceFixer()

    # El modo 2 pone el modelo en train(); con momentum=0 la BatchNorm usa las
    # estadísticas del lote pero no altera las guardadas, que comparten todos los usuarios.
    for m in model._model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0

    # Devuelve al sistema la RAM temporal usada al leer los checkpoints.
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    return model


@st.cache_resource
def model_lock():
    # Un solo modelo compartido entre usuarios: evitamos inferencias simultáneas.
    return threading.Lock()


def restore(model, audio, mode, progress):
    """Procesa en trozos con un pequeño fundido cruzado entre ellos."""
    chunk = CHUNK_SECONDS * SAMPLE_RATE
    overlap = int(OVERLAP_SECONDS * SAMPLE_RATE)
    hop = chunk - overlap
    starts = list(range(0, max(len(audio) - overlap, 1), hop))
    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    out = np.zeros(len(audio), dtype=np.float32)

    for i, start in enumerate(starts):
        seg = audio[start : start + chunk]
        y = model.restore_inmem(seg, mode=mode, cuda=False).reshape(-1).astype(np.float32)
        y = librosa.util.fix_length(y, size=len(seg))
        if start > 0:
            n = min(overlap, len(seg))
            y[:n] *= fade_in[:n]
            out[start : start + n] *= 1.0 - fade_in[:n]
        out[start : start + len(seg)] += y
        progress.progress((i + 1) / len(starts), text=f"Procesando… {i + 1}/{len(starts)}")

    peak = np.abs(out).max()
    return out / peak if peak > 1.0 else out


voice_fixer = load_model()

uploaded = st.file_uploader("Sube un audio con voz", type=["wav", "flac", "mp3", "ogg"])

mode = st.radio(
    "Modo",
    options=[0, 1, 2],
    format_func=lambda m: {
        0: "0 · Modelo original (recomendado)",
        1: "1 · Con preprocesado (quita altas frecuencias)",
        2: "2 · Modo train (a veces mejor en voz muy dañada)",
    }[m],
)

if uploaded is not None:
    st.subheader("Original")
    st.audio(uploaded)

    if st.button("Restaurar audio", type="primary"):
        uploaded.seek(0)
        audio, _ = librosa.load(uploaded, sr=SAMPLE_RATE, mono=True)
        seconds = len(audio) / SAMPLE_RATE

        if seconds > MAX_SECONDS:
            st.warning(
                f"El audio dura {seconds:.0f} s. Se procesarán solo los primeros "
                f"{MAX_SECONDS} s para que no tarde demasiado en el servidor gratuito."
            )
            audio = audio[: MAX_SECONDS * SAMPLE_RATE]

        progress = st.progress(0.0, text="En cola…")
        t0 = time.time()
        with model_lock(), torch.no_grad():  # no_grad es por hilo: va aquí, no al cargar
            restored = restore(voice_fixer, audio, mode, progress)
        elapsed = time.time() - t0
        progress.empty()

        buf = BytesIO()
        sf.write(buf, restored, samplerate=SAMPLE_RATE, format="WAV")
        st.session_state["result"] = (buf.getvalue(), elapsed, uploaded.name)

    if "result" in st.session_state and st.session_state["result"][2] == uploaded.name:
        wav_bytes, elapsed, name = st.session_state["result"]
        st.subheader("Restaurado")
        st.audio(wav_bytes, format="audio/wav")
        st.caption(f"Tiempo de proceso: {elapsed:.1f} s")
        st.download_button(
            "Descargar WAV",
            data=wav_bytes,
            file_name=name.rsplit(".", 1)[0] + "_voicefixer.wav",
            mime="audio/wav",
        )
