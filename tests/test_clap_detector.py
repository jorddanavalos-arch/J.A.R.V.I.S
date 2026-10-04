"""Fase 1/3 — re-armado ROBUSTO del detector de palmada (watchdog + auto-reapertura).

Cubre el bug "solo funciona una vez": en Windows/MME el stream se reabre pero a veces deja de
entregar audio; el watchdog debe reabrirlo en vez de colgarse. Usa FakeStream/FakeCore: prueba
la lógica de E/S real (lección de Fase 1) sin micrófono.
"""
from __future__ import annotations

import numpy as np

import audio.clap_detector as cd
import audio.mic_capture as mc
from audio.clap_core import ClapEvent


class _FakeCore:
    """Núcleo de palmada de mentira: detecta SIEMPRE que recibe un bloque de audio."""

    noise_floor = 0.001
    piso_absoluto = 1e-4

    def __init__(self):
        self.hop = 160

    def reset(self):
        pass

    def push(self, block):
        return ClapEvent("doble_palmada", 0.0, 200.0, 200.0)


class _FakeStream:
    def __init__(self, callback, sig):
        self._cb = callback
        self._sig = sig

    def start(self):
        if self._sig is not None:  # entrega UN bloque -> dispara el callback (y la detección)
            self._cb(self._sig.reshape(-1, 1), self._sig.size, None, None)

    def stop(self):
        pass

    def close(self):
        pass


def _settings(**over):
    audio = {"sample_rate": 16000, "mic_settle_ms": 1, "mic_watchdog_ms": 200,
             "mic_max_reopen": 4, "palmada": {}}
    audio.update(over)
    return {"audio": audio, "logs": {"nivel": "INFO"}}


def _detector(**over):
    det = cd.ClapDetector(_settings(**over))
    det.core = _FakeCore()
    return det


def _factory(monkeypatch, sigs):
    """Cada apertura sucesiva del InputStream entrega la siguiente señal de ``sigs`` (o None)."""
    estado = {"n": 0}

    def make(**kw):
        i = estado["n"]
        estado["n"] += 1
        sig = sigs[i] if i < len(sigs) else None
        return _FakeStream(kw.get("callback"), sig)

    monkeypatch.setattr(cd.sd, "InputStream", make)
    return estado


def test_detecta_a_la_primera(monkeypatch):
    block = np.full(160, 0.5, dtype=np.float32)
    estado = _factory(monkeypatch, [block])
    assert _detector().wait_for_double_clap() is True
    assert estado["n"] == 1  # no hizo falta reabrir
    assert not mc._MIC_LOCK.locked()


def test_se_reabre_si_el_micro_no_entrega_audio(monkeypatch):
    block = np.full(160, 0.5, dtype=np.float32)
    estado = _factory(monkeypatch, [None, block])  # 1º mudo, 2º con palmada
    assert _detector().wait_for_double_clap() is True
    assert estado["n"] == 2  # se auto-reabrió tras el watchdog
    assert not mc._MIC_LOCK.locked()


def test_se_rinde_tras_max_reopen(monkeypatch):
    estado = _factory(monkeypatch, [None, None, None, None, None])
    det = _detector(mic_max_reopen=2)
    assert det.wait_for_double_clap() is False
    assert det.errored() is True
    assert estado["n"] == 2  # exactamente mic_max_reopen aperturas
    assert not mc._MIC_LOCK.locked()


def test_stop_durante_escucha_cancela(monkeypatch):
    det = _detector()

    class StopStream:
        def __init__(self, callback, sig):
            pass

        def start(self):
            det.stop()  # parada concurrente simulada (como el apagado del orquestador)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(cd.sd, "InputStream", lambda **kw: StopStream(kw.get("callback"), None))
    assert det.wait_for_double_clap() is False
    assert det.cancelled() is True
    assert not mc._MIC_LOCK.locked()
