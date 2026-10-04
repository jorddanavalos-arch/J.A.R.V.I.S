"""Fase 2 — test de INTEGRACIÓN real: Piper sintetiza voz -> faster-whisper la transcribe.

Cierra el lazo TTS->STT sin micrófono. Lento (carga modelos); se salta si faltan.
Correr con:  .venv\\Scripts\\python -m pytest -m integration
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
VOZ = ROOT / "models" / "piper" / "es" / "es_ES" / "davefx" / "medium" / "es_ES-davefx-medium.onnx"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not VOZ.exists(), reason="voz Piper no descargada (scripts/descargar_voz.py)")
def test_piper_genera_y_whisper_transcribe():
    from audio.stt import STTEngine
    from audio.tts import TTSEngine

    tts = TTSEngine({"tts": {}})
    audio_i16, sr = tts.synth_to_array("hola jarvis esto es una prueba de voz")
    assert audio_i16.size > 0 and sr > 0
    audio_f = audio_i16.astype(np.float32) / 32768.0

    STTEngine._model = None  # forzar modelo pequeño y rápido para el test
    stt = STTEngine({"stt": {"modelo": "tiny", "download_root": "models/whisper"}})
    texto = stt.transcribe_array(audio_f, sr).lower()
    assert ("jarvis" in texto) or ("prueba" in texto), f"transcripción inesperada: {texto!r}"


@pytest.mark.skipif(not VOZ.exists(), reason="voz Piper no descargada (scripts/descargar_voz.py)")
def test_loudness_uniforme_entre_frases():
    """Multi-frase: Piper NO debe normalizar cada frase a fondo de escala por separado.

    Piper emite UN chunk por frase. Con normalize_audio=True (su default) reescala cada chunk
    a su pico máximo -> todas las frases acaban con pico == 32767 (fondo de escala), variando la
    ganancia entre frases. Con el fix (normalize_audio=False) los picos reflejan el audio crudo
    (~93-98% del fondo) y no quedan clavados al techo: la ganancia es uniforme entre frases.
    """
    from audio.tts import TTSEngine

    eng = TTSEngine({"tts": {}})
    voice = eng._ensure_voice()
    texto = "Hola, señor. Esta es la primera frase de prueba. Y aquí va una segunda frase distinta."
    chunks = list(voice.synthesize(texto, eng._syn_config()))
    assert len(chunks) >= 2, f"se esperaban varias frases; hubo {len(chunks)} chunk(s)"

    FS = 32767  # fondo de escala int16 tras la conversión de Piper (clip a ±32767)
    picos = [int(np.max(np.abs(c.audio_int16_array.astype(np.int32)))) for c in chunks]
    assert not all(p == FS for p in picos), (
        f"cada frase sigue normalizada a fondo de escala (normalize_audio activo): picos={picos}"
    )
