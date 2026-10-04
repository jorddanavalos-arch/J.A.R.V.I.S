"""Reconocimiento de voz local (STT) con faster-whisper (Fase 2).

Carga perezosa y cacheada del modelo (singleton de proceso): el coste de carga se paga una
vez, no por activación, y el esqueleto sigue importando sin tocar el stack pesado. ``listen()``
es el único método que toca hardware (vía MicCapture); ``transcribe_array()`` es puro-ish
(modelo) y se usa en el test de integración.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np

from audio.mic_capture import TARGET_SR, MicCapture, _resample_lineal

log = logging.getLogger("jarvis")
ROOT = Path(__file__).resolve().parent.parent


def _as_settings(config: dict) -> dict:
    if isinstance(config, dict) and isinstance(config.get("settings"), dict):
        return config["settings"]
    return config if isinstance(config, dict) else {}


class STTEngine:
    _model = None  # singleton de proceso

    def __init__(self, config: dict, capturer=None) -> None:
        self.config = config
        self.settings = _as_settings(config)
        stt = self.settings.get("stt", {})
        self.modelo = stt.get("modelo", "small")
        # Normaliza el idioma: un "" en settings.json haría que faster-whisper lanzara ValueError
        # (distinto de None). Si queda vacío, volvemos a "es".
        self.idioma = (stt.get("idioma") or "es").strip().lower() or "es"
        self.compute_type = stt.get("compute_type", "int8")
        self.beam_size = int(stt.get("beam_size", 1))
        self.download_root = stt.get("download_root", "models/whisper")
        self.vad = stt.get("vad", {}) if isinstance(stt.get("vad"), dict) else {}
        self.no_speech_max = float(stt.get("no_speech_max", 0.6))
        self.logprob_min = float(stt.get("logprob_min", -1.5))
        self._capturer = capturer  # inyectable para tests

    def _download_root_abs(self) -> str:
        dr = self.download_root
        return dr if Path(dr).is_absolute() else str(ROOT / dr)

    def _ensure_model(self):
        if STTEngine._model is None:
            from faster_whisper import WhisperModel

            log.info("Cargando modelo STT '%s' (%s)...", self.modelo, self.compute_type)
            STTEngine._model = WhisperModel(
                self.modelo, device="cpu", compute_type=self.compute_type,
                download_root=self._download_root_abs(),
            )
            log.info("Modelo STT cargado.")
        return STTEngine._model

    def transcribe_array(self, audio, sr: int) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        if sr != TARGET_SR:
            audio = _resample_lineal(audio, sr, TARGET_SR)
        model = self._ensure_model()
        try:
            from faster_whisper.vad import VadOptions

            vad_params = VadOptions(
                min_silence_duration_ms=int(self.vad.get("min_silence_duration_ms", 500)),
                speech_pad_ms=int(self.vad.get("speech_pad_ms", 200)),
            )
        except (TypeError, ImportError, ValueError):
            log.warning("VadOptions no disponible/firma cambiada; uso defaults de Silero.", exc_info=True)
            vad_params = None
        segments, _info = model.transcribe(
            audio,
            language=self.idioma,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=vad_params,
        )
        # Filtra ALUCINACIONES de Whisper sobre ruido/silencio: descarta segmentos con alta
        # probabilidad de "no-voz" o baja confianza media (avg_logprob). Los defaults del getattr
        # son FAIL-CLOSED (descartan si faltase el campo) para no reabrir las alucinaciones.
        segs = list(segments)
        for s in segs:
            log.debug(
                "STT seg nsp=%.3f logp=%.3f text=%r",
                getattr(s, "no_speech_prob", 1.0),
                getattr(s, "avg_logprob", -9.9),
                getattr(s, "text", ""),
            )
        buenos = [
            s for s in segs
            if getattr(s, "no_speech_prob", 1.0) < self.no_speech_max
            and getattr(s, "avg_logprob", -9.9) > self.logprob_min
        ]
        texto = " ".join(s.text.strip() for s in buenos).strip()
        if not texto and segs:
            log.info("STT: %d segmento(s) descartado(s) por baja confianza/no-voz.", len(segs))
        return re.sub(r"\s+", " ", texto)

    def listen(self, max_total_ms: int | None = None, sin_voz_timeout_ms: int | None = None):
        """Captura una frase del micrófono y la transcribe. Devuelve (texto, motivo).

        Los overrides opcionales acortan el endpointer para escuchas breves (p. ej. la
        confirmación de activación tras una palmada) sin tocar la config global.
        """
        cap = self._capturer
        if cap is None or max_total_ms or sin_voz_timeout_ms:
            s = dict(self.settings)
            audio = dict(s.get("audio", {}))
            ep = dict(audio.get("endpoint", {}))
            if max_total_ms:
                ep["max_total_ms"] = int(max_total_ms)
            if sin_voz_timeout_ms:
                ep["sin_voz_timeout_ms"] = int(sin_voz_timeout_ms)
            audio["endpoint"] = ep
            s["audio"] = audio
            cap = MicCapture(s)
        clip, motivo = cap.record_utterance()
        if clip is None:
            return "", motivo
        try:
            t0 = time.monotonic()
            texto = self.transcribe_array(clip, TARGET_SR)
            log.debug("STT: %.1fs de audio transcritos en %.2fs -> %r",
                      len(clip) / TARGET_SR, time.monotonic() - t0, texto)
        except Exception:
            log.exception("No se pudo cargar/ejecutar el modelo STT.")
            return "", "sin_modelo"
        return texto, ("ok" if texto else "vacio")
