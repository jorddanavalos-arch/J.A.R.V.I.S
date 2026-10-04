"""Captura de una frase por micrófono (Fase 2) — capa IO sobre sounddevice.

Clon de clap_detector: abre UN InputStream, empuja bloques al EndpointCore puro, espera el
evento terminal y cierra el stream SIEMPRE en finally. Comparte ``_MIC_LOCK`` con ClapDetector
para que NUNCA haya dos InputStream a la vez sobre el mismo dispositivo (serialización estricta).

Captura a 16000 Hz (lo que espera Whisper); si el driver lo rechaza al abrir, captura al
samplerate nativo y reamuestrea el clip a 16000 antes de devolverlo.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time

import numpy as np
import sounddevice as sd

from audio.endpoint_core import EndpointCore

log = logging.getLogger("jarvis")

# Un solo micrófono/stream a la vez en todo el proceso (lo comparten ClapDetector y MicCapture).
_MIC_LOCK = threading.Lock()
TARGET_SR = 16000


@contextlib.contextmanager
def mic_owner(timeout: float = 10.0):
    """Adquiere el lock global del micrófono mientras dure el bloque (cede tras timeout)."""
    got = _MIC_LOCK.acquire(timeout=timeout)
    try:
        yield got
    finally:
        if got:
            _MIC_LOCK.release()


def _resample_lineal(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return np.asarray(x, dtype=np.float32)
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xi = np.linspace(0.0, x.size - 1, n_out)
    return np.interp(xi, np.arange(x.size), x).astype(np.float32)


def _as_settings(config: dict) -> dict:
    if isinstance(config, dict) and isinstance(config.get("settings"), dict):
        return config["settings"]
    return config if isinstance(config, dict) else {}


class MicCapture:
    def __init__(self, config: dict) -> None:
        self.settings = _as_settings(config)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._error = threading.Event()
        self._clip: np.ndarray | None = None
        self._motivo = "ok"
        self.core: EndpointCore | None = None
        self.capture_sr = TARGET_SR

    def _device_samplerate(self, default: int = 44100) -> int:
        try:
            return int(round(float(sd.query_devices(kind="input")["default_samplerate"])))
        except Exception:
            return default

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("sounddevice status: %s", status)
        try:
            data = np.asarray(indata, dtype=np.float32)
            block = data.mean(axis=1) if data.ndim == 2 else data.reshape(-1)
            for ev in self.core.push(block):
                if ev.tipo == "VOZ_FIN":
                    self._clip, self._motivo = ev.audio, "ok"
                    self._wake.set()
                elif ev.tipo == "TOPE_MAXIMO":
                    self._clip, self._motivo = ev.audio, "tope_maximo"
                    self._wake.set()
                elif ev.tipo == "TIMEOUT_SIN_VOZ":
                    self._clip, self._motivo = None, "timeout_sin_voz"
                    self._wake.set()
        except Exception:
            log.exception("Error en el endpointer de voz.")
            self._error.set()
            self._wake.set()

    def _open_stream(self):
        # Lazy: el samplerate nativo SOLO se consulta si 16 kHz falla (query_devices puede tardar).
        tried: list[int] = []
        for cand in (TARGET_SR, None):
            sr = cand if cand is not None else self._device_samplerate()
            if sr in tried:
                continue
            tried.append(sr)
            try:
                self.core = EndpointCore.from_config(self.settings, sample_rate=sr)
                stream = sd.InputStream(
                    samplerate=sr, channels=1, dtype="float32",
                    blocksize=self.core.hop, callback=self._callback,
                )
                return sr, stream
            except Exception:
                log.warning("No se pudo abrir el micrófono a %d Hz; pruebo otro.", sr, exc_info=True)
        return None, None

    def record_utterance(self):
        """Captura una frase. Devuelve (clip float32 mono 16k | None, motivo).

        motivo: 'ok' | 'tope_maximo' | 'timeout_sin_voz' | 'cancelado' | 'error_microfono'.
        """
        self._stop.clear()
        self._wake.clear()
        self._error.clear()
        self._clip = None
        self._motivo = "ok"
        with mic_owner() as got:
            if not got:
                log.warning("Micrófono ocupado (lock no adquirido).")
                self._error.set()
                return None, "error_microfono"
            sr, stream = self._open_stream()
            if stream is None:
                self._error.set()
                return None, "error_microfono"
            self.capture_sr = sr
            try:
                stream.start()
                # Deadline de RELOJ (red anti-cuelgue): si el driver deja de entregar bloques
                # (desconexión/suspensión), los topes por-muestras del core nunca disparan.
                deadline = time.monotonic() + (self.core.max_total_n / self.core.sr) + 5.0
                while not self._wake.is_set() and not self._stop.is_set():
                    self._wake.wait(timeout=0.1)
                    if time.monotonic() > deadline:
                        log.warning("Captura de voz superó el deadline de reloj; cierro el stream.")
                        self._error.set()
                        break
            except Exception:
                log.warning("Error en el stream de captura de voz.", exc_info=True)
                self._error.set()
            finally:
                for fn in ("stop", "close"):
                    try:
                        getattr(stream, fn)()
                    except Exception:
                        pass
        if self._stop.is_set():
            return None, "cancelado"
        if self._error.is_set():
            return None, "error_microfono"
        clip = self._clip
        if clip is not None and self.capture_sr != TARGET_SR:
            clip = _resample_lineal(clip, self.capture_sr, TARGET_SR)
        return clip, self._motivo

    def cancelled(self) -> bool:
        return self._stop.is_set()

    def errored(self) -> bool:
        return self._error.is_set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
