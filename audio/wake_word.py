"""Palabra de activación por voz, 100% local. Dos motores:

  - "porcupine"   -> palabra "Jarvis" (pvporcupine; necesita una access key GRATIS de Picovoice,
                     guardada en config/secrets.json como "picovoice_access_key" o en la variable
                     de entorno PICOVOICE_ACCESS_KEY). Inferencia local, sin coste por uso.
  - "openwakeword"-> frase "Hey Jarvis" (modelo preentrenado ONNX, sin clave).

Si se pide Porcupine pero no hay clave, cae automáticamente a openWakeWord ("hey jarvis"), para
no quedarse nunca sin activación por voz.

Corre JUNTO al detector de palmadas con el mismo micrófono: el callback acumula audio (16 kHz
int16) y la INFERENCIA se ejecuta en el hilo principal (nunca en el callback de audio). La carga
del modelo se hace en el hilo principal antes de abrir el micro (igual que Piper/whisper).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis")

_SECRETS = Path(__file__).resolve().parent.parent / "config" / "secrets.json"


class WakeWord:
    def __init__(self, wake_cfg: dict | None = None) -> None:
        cfg = wake_cfg if isinstance(wake_cfg, dict) else {}
        self.motor = str(cfg.get("motor", "openwakeword")).lower()
        self.palabra = cfg.get("palabra", "jarvis")
        self.umbral = float(cfg.get("umbral", 0.5))
        self.vad = float(cfg.get("vad", 0.3))  # VAD Silero (openWakeWord): baja = más permisivo
        self.modelo_oww = cfg.get("modelo_oww", "hey_jarvis")
        self._engine: str | None = None  # "porcupine" | "openwakeword"
        self._pp = None       # instancia de pvporcupine
        self._oww = None      # modelo de openWakeWord
        self._frame = 1280    # tamaño de bloque del motor activo
        self._buf = np.zeros(0, dtype=np.int16)
        self.loaded = False
        self.descripcion = ""  # cómo activarlo, para mensajes ("Jarvis" / "Hey Jarvis")

    # ------------------------------------------------------------------ carga
    def _picovoice_key(self) -> str | None:
        k = (os.environ.get("PICOVOICE_ACCESS_KEY") or "").strip()
        if k:
            return k
        try:
            data = json.loads(_SECRETS.read_text(encoding="utf-8"))
            return (data.get("picovoice_access_key") or "").strip() or None
        except Exception:
            return None

    def load(self) -> bool:
        """Carga el motor en el HILO PRINCIPAL antes de abrir el micro."""
        if self.loaded:
            return True
        if self.motor == "porcupine" and self._load_porcupine():
            return True
        if self.motor == "porcupine":
            log.warning("Porcupine no disponible; uso openWakeWord 'hey jarvis' como respaldo.")
        return self._load_oww()

    def _load_porcupine(self) -> bool:
        key = self._picovoice_key()
        if not key:
            log.info("Sin clave de Picovoice: no puedo usar la palabra '%s' (Porcupine).", self.palabra)
            return False
        try:
            import pvporcupine

            self._pp = pvporcupine.create(
                access_key=key, keywords=[self.palabra], sensitivities=[self.umbral]
            )
            self._frame = self._pp.frame_length
            self._engine = "porcupine"
            self.loaded = True
            self.descripcion = self.palabra.capitalize()
            log.info("Wake word Porcupine '%s' cargada.", self.palabra)
            return True
        except Exception:
            log.warning("No se pudo cargar Porcupine; intento openWakeWord.", exc_info=True)
            self._pp = None
            return False

    def _load_oww(self) -> bool:
        try:
            import openwakeword
            from openwakeword.model import Model

            try:
                openwakeword.utils.download_models([self.modelo_oww])
            except Exception:
                log.debug("No se pudieron descargar modelos de openWakeWord (¿ya están?).", exc_info=True)
            # vad_threshold (Silero) descarta predicciones sin voz: deja bajar el umbral del wake
            # word sin disparos falsos por ruido.
            self._oww = Model(
                wakeword_models=[self.modelo_oww], inference_framework="onnx", vad_threshold=self.vad
            )
            self._frame = 1280
            self._engine = "openwakeword"
            self.loaded = True
            self.descripcion = "Hey Jarvis"
            log.info("Wake word openWakeWord '%s' cargada.", self.modelo_oww)
            return True
        except Exception:
            log.warning("No se pudo cargar ninguna wake word; quedará desactivada.", exc_info=True)
            self.loaded = False
            return False

    # ------------------------------------------------------------------ uso
    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        if self._oww is not None:
            try:
                self._oww.reset()
            except Exception:
                pass

    def _disparo_en_frame(self, frame: np.ndarray) -> bool:
        if self._engine == "porcupine":
            return self._pp.process(frame) >= 0
        scores = self._oww.predict(frame)
        return bool(scores) and max(scores.values()) >= self.umbral

    def process(self, pcm16: np.ndarray) -> bool:
        """Alimenta audio PCM int16 mono a 16 kHz; True si detecta la palabra de activación."""
        if not self.loaded or pcm16 is None or getattr(pcm16, "size", 0) == 0:
            return False
        self._buf = np.concatenate([self._buf, np.asarray(pcm16, dtype=np.int16)])
        disparo = False
        while self._buf.size >= self._frame:
            frame = self._buf[:self._frame]
            self._buf = self._buf[self._frame:]
            try:
                if self._disparo_en_frame(frame):
                    disparo = True
            except Exception:
                log.debug("Fallo en la inferencia de la wake word.", exc_info=True)
                return False
        if disparo:
            self._buf = np.zeros(0, dtype=np.int16)  # evita re-disparos con el mismo audio
        return disparo
