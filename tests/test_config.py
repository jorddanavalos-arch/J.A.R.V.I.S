"""Guarda-raíl del espejo DEFAULT_SETTINGS <-> config/settings.json (Fase 2).

Si un default se añade al JSON pero no a DEFAULT_SETTINGS (o viceversa) y un día divergen,
estos tests fallan. Cubre el hueco que la revisión señaló (config sin tests).
"""
from __future__ import annotations

from config import DEFAULT_SETTINGS, _deep_merge, load_settings


def test_defaults_presentes_con_json_vacio(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{}", encoding="utf-8")
    s = load_settings(p)
    assert s["stt"]["no_speech_max"] == 0.6
    assert s["stt"]["logprob_min"] == -1.5
    assert isinstance(s["stt"]["vad"], dict)
    assert s["tts"]["voz_path"].endswith("es_ES-davefx-medium.onnx")
    ep = s["audio"]["endpoint"]
    assert ep["silencio_fin_ms"] == 500
    assert ep["max_total_ms"] == 8000
    assert ep["floor_up_rate"] == 0.02


def test_submodulos_vacios_heredan_defaults():
    merged = _deep_merge(DEFAULT_SETTINGS, {"stt": {}, "tts": {}, "audio": {"endpoint": {}}})
    assert merged["stt"]["modelo"] == "small"
    assert merged["audio"]["endpoint"]["hop_ms"] == 20
    assert merged["tts"]["motor"] == "piper"
