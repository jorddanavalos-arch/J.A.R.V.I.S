"""Capa de E/S del detector de doble palmada (Fase 1).

Fina sobre sounddevice; conserva la firma existente ``__init__(config)`` y
``wait_for_double_clap() -> bool``. Toda la lógica DSP vive en ``ClapCore`` (sin
sounddevice), de modo que se prueba sin micrófono.

Decisiones de robustez (de la auditoría en vivo):
- **Samplerate NATIVO del dispositivo**: forzar 16000 hace que muchos micrófonos reamuestreen
  (MME/WASAPI) y eso ABLANDA el transitorio de la palmada, del que dependen crest y hf. Por eso
  capturamos al samplerate por defecto del micro (44100/48000) salvo que se fije uno explícito.
- **Diagnóstico en vivo**: con logs.nivel=DEBUG, se registra el nivel (rms/piso/umbral) y por qué
  se acepta/rechaza cada onset — para calibrar `audio.palmada` con el micro real.
- **Retorno distinguible**: detección vs cancelación (stop) vs error, sin colapsarlos en False.
- La construcción del stream va dentro de try; un fallo de micro no propaga.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import sounddevice as sd

from audio.clap_core import ClapCore
from audio.mic_capture import _resample_lineal, mic_owner
from audio.wake_word import WakeWord

log = logging.getLogger("jarvis")


def _as_settings(config: dict) -> dict:
    """Acepta el dict de settings (audio en la raíz) o el envoltorio {settings, apps}."""
    if isinstance(config, dict) and isinstance(config.get("settings"), dict):
        return config["settings"]
    return config if isinstance(config, dict) else {}


class ClapDetector:
    def __init__(self, config: dict) -> None:
        self.settings = _as_settings(config)
        self.config = config
        audio = self.settings.get("audio", {})
        configured = audio.get("sample_rate", 0)
        if isinstance(configured, (int, float)) and configured and configured > 0:
            self.sample_rate = int(configured)  # forzado (tests o usuario)
        else:
            self.sample_rate = self._device_samplerate(default=44100)  # nativo del dispositivo
        core_settings = {**self.settings, "audio": {**audio, "sample_rate": self.sample_rate}}
        self.core = ClapCore.from_config(core_settings)
        bs = audio.get("blocksize", 0)
        self.blocksize = int(bs) if isinstance(bs, (int, float)) and bs and bs > 0 else self.core.hop
        self.verbose = str(self.settings.get("logs", {}).get("nivel", "INFO")).upper() == "DEBUG"
        if self.verbose:
            self.core._debug = True
        self._detected = threading.Event()
        self._stop = threading.Event()
        self._error = threading.Event()
        self._wake = threading.Event()  # se activa por detección, stop o error
        self._eval_seen = 0
        self._cb_count = 0
        self._callbacks = 0  # bloques de audio recibidos (siempre, no solo en modo verbose)
        self._reseed_flag = False  # pide re-calibrar el piso (lo ejecuta el callback, sin carrera)
        # Re-armado robusto del stream entre activaciones (ver wait_for_double_clap).
        self._settle_s = max(0.0, float(audio.get("mic_settle_ms", 150)) / 1000.0)
        self._watchdog_s = max(0.2, float(audio.get("mic_watchdog_ms", 3000)) / 1000.0)
        self._max_reopen = max(1, int(audio.get("mic_max_reopen", 4)))
        # Palabra de activación opcional ("Hey Jarvis"), en paralelo a las palmadas (mismo micro).
        wake = self.settings.get("wake", {})
        wake = wake if isinstance(wake, dict) else {}
        self.wake = WakeWord(wake) if wake.get("wake_word_activo", False) else None
        self._wake_buf: list = []  # audio 16k int16 que el callback acumula para la wake word
        self._wake_lock = threading.Lock()
        self._origen = ""  # qué disparó la última detección: "palmada" | "voz" | "externo"

    @staticmethod
    def _device_samplerate(default: int = 44100) -> int:
        try:
            info = sd.query_devices(kind="input")
            sr = int(round(float(info["default_samplerate"])))
            if sr > 0:
                return sr
        except Exception:
            log.warning("No se pudo leer el samplerate del micrófono; uso %d.", default, exc_info=True)
        return int(default)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("sounddevice status: %s", status)
        try:
            self._callbacks += 1
            if self._reseed_flag:  # re-calibración pedida desde el hilo principal (mismo hilo que push)
                self._reseed_flag = False
                self.core.reset()
            data = np.asarray(indata, dtype=np.float32)
            block = data.mean(axis=1) if data.ndim == 2 else data.reshape(-1)
            ev = self.core.push(block)
            if self.verbose:
                self._log_diagnostico(block)
            if ev is not None:
                self._origen = "palmada"
                self._detected.set()
                self._wake.set()
            if self.wake is not None and self.wake.loaded:
                # SOLO acumular (barato): reamuestrear a 16k y guardar. La inferencia onnx corre en
                # el HILO PRINCIPAL (_escuchar), NUNCA aquí; hacerla en el callback de audio en
                # tiempo real atascaba el stream y se perdían palmadas y voz.
                pcm = _resample_lineal(block, self.sample_rate, 16000)
                pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
                with self._wake_lock:
                    self._wake_buf.append(pcm16)
        except Exception:
            log.exception("Error procesando audio en el detector de palmada.")
            self._error.set()
            self._wake.set()

    def _log_diagnostico(self, block) -> None:
        evals = self.core.debug_eval
        while self._eval_seen < len(evals):
            e = evals[self._eval_seen]
            self._eval_seen += 1
            log.info("onset %s: %s", "ACEPTADO" if e["accept"] else "rechazado", e)
        self._cb_count += 1
        cada = max(1, int(self.sample_rate / max(1, self.blocksize)))  # ~1 s
        if self._cb_count % cada == 0:
            try:
                rms = float(np.sqrt(np.mean(np.asarray(block, dtype=np.float64) ** 2) + 1e-12))
            except Exception:
                rms = 0.0
            thr = self.core.umbral_factor * max(self.core.noise_floor, self.core.piso_absoluto)
            log.info("nivel rms=%.5f piso=%.5f umbral=%.5f", rms, self.core.noise_floor, thr)

    def wait_for_double_clap(self) -> bool:
        """Bloquea hasta detectar una doble palmada. Devuelve True SOLO si se detectó.

        Re-armado ROBUSTO: en Windows/MME es frecuente que, al reabrir un InputStream entre
        activaciones, el stream se abra pero deje de entregar callbacks (se quedaría colgado para
        siempre). Un watchdog detecta la ausencia de audio y REABRE el stream hasta
        ``mic_max_reopen`` veces antes de rendirse. Devuelve False por cancelación (stop()),
        error o micrófono inaccesible; usa cancelled()/errored() para distinguir el motivo.
        """
        for ev in (self._detected, self._stop, self._error, self._wake):
            ev.clear()
        self._eval_seen = 0
        self._cb_count = 0
        self._callbacks = 0
        self._reseed_flag = False
        self._origen = ""
        self.core.reset()
        if self.wake is not None:
            self.wake.reset()
            with self._wake_lock:
                self._wake_buf = []
        with mic_owner() as got:
            if not got:
                log.warning("Micrófono ocupado; no se pudo escuchar la palmada.")
                self._error.set()
                return False
            intentos = 0
            while not self._wake.is_set() and not self._stop.is_set():
                intentos += 1
                if intentos > self._max_reopen:
                    log.error("El micrófono no entrega audio tras %d intentos; abandono.", self._max_reopen)
                    self._error.set()
                    break
                if self._settle_s:
                    time.sleep(self._settle_s)  # respiro para que MME libere el dispositivo
                try:
                    stream = sd.InputStream(
                        samplerate=self.sample_rate, channels=1, dtype="float32",
                        blocksize=self.blocksize, callback=self._callback,
                    )
                except Exception:
                    log.warning("No se pudo abrir el micrófono (sr=%d); reintento.", self.sample_rate, exc_info=True)
                    continue
                if not self._escuchar(stream):
                    break  # detección, stop o error: no reabrir
                log.warning("El micrófono no entregó audio; reabro (intento %d/%d).", intentos, self._max_reopen)
        return self._detected.is_set()

    def _escuchar(self, stream) -> bool:
        """Corre ``stream`` hasta detección/stop/error o hasta que el watchdog pida REABRIR.

        Devuelve True si hay que REABRIR (el stream dejó de entregar audio); False si terminamos
        por detección, parada o error. Cierra el stream SIEMPRE en finally.
        """
        reabrir = False
        try:
            stream.start()
            prev_cb = self._callbacks
            last_progress = time.monotonic()
            last_beat = last_progress
            last_reseed = last_progress
            while not self._wake.is_set() and not self._stop.is_set():
                self._wake.wait(timeout=0.1)
                now = time.monotonic()
                if self._callbacks != prev_cb:  # llega audio: hay progreso
                    prev_cb = self._callbacks
                    last_progress = now
                if self.wake is not None and self.wake.loaded:  # inferencia wake word (hilo principal)
                    with self._wake_lock:
                        pendiente, self._wake_buf = self._wake_buf, []
                    if pendiente:
                        try:
                            if self.wake.process(np.concatenate(pendiente)):
                                self._origen = "voz"
                                self._detected.set()
                                self._wake.set()
                        except Exception:
                            log.debug("Fallo en la inferencia de la wake word.", exc_info=True)
                if now - last_beat >= 10.0:  # latido SOLO en DEBUG (no spamea la consola en INFO)
                    last_beat = now
                    thr = self.core.umbral_factor * max(self.core.noise_floor, self.core.piso_absoluto)
                    log.debug("Escuchando palmadas... (bloques=%d, piso=%.5f, umbral=%.5f)",
                              self._callbacks, self.core.noise_floor, thr)
                if now - last_reseed >= 60.0:  # red de seguridad esporádica (el piso ya se adapta
                    last_reseed = now             # solo por EMA en ~1s; recalibrar a menudo creaba
                    self._reseed_flag = True      # ventanas de ~400ms sordas que estorbaban las palmadas
                if now - last_progress > self._watchdog_s:
                    reabrir = True  # stream vivo pero sin audio -> reabrir
                    break
        except Exception:
            log.warning("Error en el stream del detector de palmada.", exc_info=True)
            self._error.set()
        finally:
            for fn in ("stop", "close"):
                try:
                    getattr(stream, fn)()
                except Exception:
                    pass
        return reabrir

    def prewarm(self) -> None:
        """Carga la wake word en el HILO PRINCIPAL, antes de abrir el micro (evita el cuelgue
        de inicializar onnxruntime a la vez que PortAudio)."""
        if self.wake is not None:
            self.wake.load()

    def trigger_externo(self) -> None:
        """Activación desde FUERA del audio (atajo de teclado global): equivale a la doble palmada.

        Desbloquea wait_for_double_clap() -> True aunque el micrófono esté ensordecido por
        música. Fuera de la espera (p.ej. en plena conversación) no tiene efecto: los eventos
        se limpian al volver a esperar.
        """
        self._origen = "externo"
        self._detected.set()
        self._wake.set()

    def origen(self) -> str:
        """Qué disparó la última detección: 'palmada' (comité DSP; puede ser un falso tipo
        click de mouse), 'voz' (wake word) o 'externo' (atajo de teclado). Los tres caminos
        reales lo setean siempre; si por lo que sea está vacío, se trata como 'externo'
        (abrir directo molesta menos que exigir confirmación por un origen desconocido)."""
        return self._origen or "externo"

    def cancelled(self) -> bool:
        return self._stop.is_set()

    def errored(self) -> bool:
        return self._error.is_set()

    def stop(self) -> None:
        """Señala parada y desbloquea wait_for_double_clap (devolverá False = cancelación)."""
        self._stop.set()
        self._wake.set()
