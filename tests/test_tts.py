"""Fase 2 — ciclo de vida del OutputStream en TTS.speak() (sin audio real).

Inyecta una voz FALSA y un OutputStream FALSO para verificar que el stream se cierra SIEMPRE
(también si write() falla a mitad) y que la falta de voz no rompe el bucle.
"""
from __future__ import annotations

import numpy as np

import audio.tts as tts_mod
from audio.tts import TTSEngine


class _Chunk:
    def __init__(self):
        self.audio_int16_array = np.zeros(160, dtype=np.int16)
        self.sample_rate = 22050


class _FakeVoice:
    def synthesize(self, text, cfg):
        return [_Chunk(), _Chunk()]


def _stream_factory(contador, fail_write=False):
    class _FakeStream:
        def __init__(self, **kw):
            pass

        def start(self):
            contador["start"] += 1

        def write(self, arr):
            contador["write"] += 1
            if fail_write:
                raise RuntimeError("write falló a mitad")

        def stop(self):
            contador["stop"] += 1

        def close(self):
            contador["close"] += 1

    return lambda **kw: _FakeStream(**kw)


def teardown_function(_):
    TTSEngine._voice = None  # limpia el singleton de proceso
    TTSEngine._voice_unavailable = False


def test_speak_cierra_outputstream(monkeypatch):
    contador = {"start": 0, "write": 0, "stop": 0, "close": 0}
    monkeypatch.setattr(tts_mod.sd, "OutputStream", _stream_factory(contador))
    eng = TTSEngine({"tts": {}})
    TTSEngine._voice = _FakeVoice()
    eng.speak("hola, señor")
    assert contador["write"] == 1  # clip completo en una sola escritura (reproducción fluida)
    assert contador["stop"] == 1 and contador["close"] == 1  # cerrado exactamente una vez


def test_speak_cierra_aunque_write_falle(monkeypatch):
    contador = {"start": 0, "write": 0, "stop": 0, "close": 0}
    monkeypatch.setattr(tts_mod.sd, "OutputStream", _stream_factory(contador, fail_write=True))
    eng = TTSEngine({"tts": {}})
    TTSEngine._voice = _FakeVoice()
    eng.speak("hola")  # no debe propagar la excepción
    assert contador["close"] == 1  # el finally cierra pese al error de write


def test_speak_sin_voz_no_revienta(monkeypatch):
    eng = TTSEngine({"tts": {}})

    def _boom():
        raise RuntimeError("voz no disponible")

    monkeypatch.setattr(eng, "_ensure_voice", _boom)
    eng.speak("hola")  # cae en el fallback de consola, sin excepción


def test_syn_config_desactiva_normalizacion_por_frase():
    # Piper, con normalize_audio=True (su default), normaliza CADA frase a fondo de escala
    # por separado: en respuestas multi-frase eso reescala la ganancia frase a frase y hace
    # que 'volume' actúe sobre audio ya normalizado (impredecible). Lo desactivamos para que
    # el nivel sea uniforme entre frases y 'volume' module el audio crudo.
    eng = TTSEngine({"tts": {}})
    cfg = eng._syn_config()
    assert cfg is not None, "SynthesisConfig debería construirse (piper instalado)"
    assert cfg.normalize_audio is False
