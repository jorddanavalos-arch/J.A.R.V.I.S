"""Fase 2 — tests de la captura de voz (capa IO) con FakeStream (sin micrófono real)."""
from __future__ import annotations

import numpy as np

SR = 16000


def _noise(ms, amp, seed):
    n = int(SR * ms / 1000.0)
    return (np.random.default_rng(seed).standard_normal(n) * amp).astype(np.float32)


def silence(ms, seed=1):
    return _noise(ms, 0.001, seed)


def voice(ms, seed=2):
    return _noise(ms, 0.05, seed)


def _fake_factory(sig, raise_on_start=False):
    holder = {}

    class FakeStream:
        def __init__(self, callback=None, blocksize=320, **kw):
            self.cb = callback
            self.bs = blocksize
            self.stops = 0
            self.closes = 0
            holder["stream"] = self

        def start(self):
            if raise_on_start:
                raise RuntimeError("fallo simulado del stream")
            for i in range(0, sig.size, self.bs):
                self.cb(sig[i:i + self.bs].reshape(-1, 1), self.bs, None, None)

        def stop(self):
            self.stops += 1

        def close(self):
            self.closes += 1

    return FakeStream, holder


def test_devuelve_clip_y_cierra_stream_en_finally(monkeypatch):
    import audio.mic_capture as mc

    sig = np.concatenate([silence(500), voice(800), silence(1200)]).astype(np.float32)
    Fake, holder = _fake_factory(sig)
    monkeypatch.setattr(mc.sd, "InputStream", lambda **kw: Fake(**kw))
    cap = mc.MicCapture({"audio": {"endpoint": {}}, "stt": {"sample_rate": SR}})
    clip, motivo = cap.record_utterance()
    assert motivo == "ok"
    assert clip is not None and clip.size > 0
    assert holder["stream"].stops == 1 and holder["stream"].closes == 1
    assert not mc._MIC_LOCK.locked()  # el lock queda libre


def test_silencio_devuelve_timeout(monkeypatch):
    import audio.mic_capture as mc

    sig = silence(7000).astype(np.float32)
    Fake, _ = _fake_factory(sig)
    monkeypatch.setattr(mc.sd, "InputStream", lambda **kw: Fake(**kw))
    cap = mc.MicCapture({"audio": {"endpoint": {}}})
    clip, motivo = cap.record_utterance()
    assert clip is None and motivo == "timeout_sin_voz"
    assert not mc._MIC_LOCK.locked()


def test_error_de_apertura_cierra_y_libera_lock(monkeypatch):
    import audio.mic_capture as mc

    def boom(**kw):
        raise RuntimeError("dispositivo ocupado")

    monkeypatch.setattr(mc.sd, "InputStream", boom)
    cap = mc.MicCapture({"audio": {"endpoint": {}}})
    clip, motivo = cap.record_utterance()
    assert clip is None and motivo == "error_microfono"
    assert cap.errored() is True
    assert not mc._MIC_LOCK.locked()
