"""Fase 2 — test unitario del filtro anti-alucinación del STT.

No toca micrófono ni descarga modelos: inyecta un modelo FALSO en el singleton de proceso
y comprueba la lógica de filtrado por ``no_speech_prob`` / ``avg_logprob``.
"""
from __future__ import annotations

import numpy as np

from audio.stt import STTEngine


class _Seg:
    def __init__(self, text, no_speech_prob, avg_logprob):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


class _FakeModel:
    def __init__(self, segs):
        self._segs = segs

    def transcribe(self, audio, **kw):  # firma laxa: ignora language/beam/vad/...
        return list(self._segs), None


def _engine(segs, **stt_over):
    stt = {"sample_rate": 16000}
    stt.update(stt_over)
    eng = STTEngine({"stt": stt})
    STTEngine._model = _FakeModel(segs)  # singleton de proceso
    return eng


def _audio():
    return np.full(16000, 0.01, dtype=np.float32)


def teardown_function(_):
    STTEngine._model = None  # no contaminar el singleton de otros tests/integración


def test_filtro_conserva_voz_y_descarta_alucinacion():
    segs = [
        _Seg(" hola jarvis", 0.05, -0.40),      # voz buena
        _Seg(" ruido inventado", 0.95, -2.50),  # no-voz (no_speech_prob alta + logp bajo)
    ]
    out = _engine(segs).transcribe_array(_audio(), 16000)
    assert "hola jarvis" in out
    assert "ruido" not in out


def test_filtro_no_descarta_palabra_corta_real():
    # 'jarvis' real puede salir con avg_logprob ~ -1.4; con logprob_min=-1.5 NO debe caer.
    out = _engine([_Seg(" jarvis", 0.10, -1.40)]).transcribe_array(_audio(), 16000)
    assert out == "jarvis"


def test_filtro_fail_closed_si_faltan_campos():
    class _SegSinCampos:
        text = " algo"

    # Sin los campos, los getattr usan defaults FAIL-CLOSED -> el segmento se descarta.
    out = _engine([_SegSinCampos()]).transcribe_array(_audio(), 16000)
    assert out == ""


def test_audio_vacio_devuelve_cadena_vacia():
    assert _engine([_Seg(" x", 0.0, 0.0)]).transcribe_array(np.zeros(0, dtype=np.float32), 16000) == ""


# --- camino real listen() (la costura captura -> transcribe -> motivo): lección de Fase 1 ---
class _FakeCapturer:
    def __init__(self, clip, motivo):
        self._r = (clip, motivo)

    def record_utterance(self):
        return self._r


def _engine_con_captura(clip, motivo, segs=None):
    eng = STTEngine({"stt": {"sample_rate": 16000}}, capturer=_FakeCapturer(clip, motivo))
    if segs is not None:
        STTEngine._model = _FakeModel(segs)
    return eng


def test_listen_ok_con_texto():
    eng = _engine_con_captura(_audio(), "ok", [_Seg(" hola jarvis", 0.05, -0.40)])
    assert eng.listen() == ("hola jarvis", "ok")


def test_listen_vacio_si_filtro_descarta_todo():
    eng = _engine_con_captura(_audio(), "ok", [_Seg(" alucinación", 0.95, -3.0)])
    assert eng.listen() == ("", "vacio")


def test_listen_propaga_motivo_si_no_hay_clip():
    eng = _engine_con_captura(None, "timeout_sin_voz")
    assert eng.listen() == ("", "timeout_sin_voz")


def test_listen_sin_modelo_devuelve_sin_modelo():
    class _BoomModel:
        def transcribe(self, audio, **kw):
            raise RuntimeError("modelo no disponible")

    eng = STTEngine({"stt": {"sample_rate": 16000}}, capturer=_FakeCapturer(_audio(), "ok"))
    STTEngine._model = _BoomModel()
    assert eng.listen() == ("", "sin_modelo")
