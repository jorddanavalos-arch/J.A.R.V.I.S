"""Tests del detector de palabra de activación (buffer/umbral; motores Porcupine y openWakeWord
simulados, sin cargar las librerías nativas)."""
from __future__ import annotations

import numpy as np

from audio.wake_word import WakeWord


class _FakeOww:
    """openWakeWord simulado: devuelve score alto en la llamada nº `dispara_en`."""

    def __init__(self, dispara_en=None):
        self.calls = 0
        self.dispara_en = dispara_en

    def predict(self, frame):
        self.calls += 1
        return {"hey_jarvis": 0.9 if self.calls == self.dispara_en else 0.0}

    def reset(self):
        pass


class _FakePorcupine:
    """Porcupine simulado: process() devuelve 0 (detectado) en la llamada nº `dispara_en`, si no -1."""

    def __init__(self, dispara_en=None, frame_length=512):
        self.calls = 0
        self.dispara_en = dispara_en
        self.frame_length = frame_length

    def process(self, frame):
        self.calls += 1
        return 0 if self.calls == self.dispara_en else -1


def _ww_oww(model):
    w = WakeWord({"umbral": 0.5})
    w._engine, w._oww, w._frame, w.loaded = "openwakeword", model, 1280, True
    return w


def _ww_pp(pp):
    w = WakeWord({"motor": "porcupine"})
    w._engine, w._pp, w._frame, w.loaded = "porcupine", pp, pp.frame_length, True
    return w


def test_oww_dispara_con_score_alto():
    assert _ww_oww(_FakeOww(dispara_en=1)).process(np.zeros(1280, dtype=np.int16)) is True


def test_oww_no_dispara_con_score_bajo():
    assert _ww_oww(_FakeOww(dispara_en=None)).process(np.zeros(1280, dtype=np.int16)) is False


def test_oww_acumula_hasta_frame_completo():
    fm = _FakeOww(dispara_en=None)
    w = _ww_oww(fm)
    assert w.process(np.zeros(600, dtype=np.int16)) is False  # <1280 -> sin predict
    assert fm.calls == 0
    assert w.process(np.zeros(800, dtype=np.int16)) is False  # 1400 -> 1 predict
    assert fm.calls == 1


def test_porcupine_dispara_en_indice_no_negativo():
    pp = _FakePorcupine(dispara_en=1, frame_length=512)
    assert _ww_pp(pp).process(np.zeros(512, dtype=np.int16)) is True


def test_porcupine_no_dispara_si_indice_negativo():
    pp = _FakePorcupine(dispara_en=None, frame_length=512)
    w = _ww_pp(pp)
    assert w.process(np.zeros(1024, dtype=np.int16)) is False  # 2 frames de 512, ninguno dispara
    assert pp.calls == 2


def test_sin_modelo_no_falla():
    assert WakeWord({}).process(np.zeros(1280, dtype=np.int16)) is False


def test_reset_limpia_el_buffer():
    fm = _FakeOww(dispara_en=None)
    w = _ww_oww(fm)
    w.process(np.zeros(600, dtype=np.int16))  # deja 600 en buffer
    w.reset()
    w.process(np.zeros(800, dtype=np.int16))  # 800 < 1280 (buffer limpio) -> sin predict
    assert fm.calls == 0
