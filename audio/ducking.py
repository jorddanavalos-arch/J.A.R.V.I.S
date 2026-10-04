"""Ducking de música durante la conversación (estilo Alexa/Google Home).

Sin cancelación de eco (AEC) en Python 3.14, la música ensordece al micrófono y Jarvis no
oye al señor. Lo que SÍ se puede: bajar el volumen de la app de música (por SESIÓN de audio
de Windows, vía pycaw) mientras dura la conversación y restaurarlo al terminar.

Fail-open absoluto: sin pycaw, sin sesión de Spotify o con cualquier error, Jarvis sigue
como si nada (solo pierde el ducking). Nunca toca el volumen maestro ni otras apps.
"""
from __future__ import annotations

import logging

log = logging.getLogger("jarvis")


class MusicDucker:
    """bajar() al abrir la conversación, restaurar() al cerrarla (idempotentes)."""

    def __init__(self, cfg: dict | None) -> None:
        cfg = cfg if isinstance(cfg, dict) else {}
        self.activo: bool = bool(cfg.get("activo", True))
        self.volumen_bajo: float = float(cfg.get("volumen", 0.15))
        self.procesos = tuple(str(p).lower() for p in cfg.get("procesos", ["spotify.exe"]))
        self._previos: dict = {}  # sesión -> volumen original

    def _sesiones_musica(self):
        from pycaw.pycaw import AudioUtilities  # import tardío: sin pycaw -> fail-open
        for s in AudioUtilities.GetAllSessions():
            try:
                if s.Process and s.Process.name().lower() in self.procesos:
                    yield s
            except Exception:
                continue

    def bajar(self) -> None:
        if not self.activo or self._previos:
            return
        try:
            for s in self._sesiones_musica():
                vol = s.SimpleAudioVolume
                actual = float(vol.GetMasterVolume())
                if actual <= self.volumen_bajo:
                    continue  # ya está baja: no la toques (ni la "restaures" luego)
                self._previos[id(s)] = (vol, actual)
                vol.SetMasterVolume(self.volumen_bajo, None)
            if self._previos:
                log.info("Música atenuada para escuchar (%d sesión/es).", len(self._previos))
        except Exception:
            log.debug("Ducking no disponible; sigo sin atenuar.", exc_info=True)
            self._previos = {}

    def restaurar(self) -> None:
        if not self._previos:
            return
        try:
            for vol, original in self._previos.values():
                try:
                    vol.SetMasterVolume(original, None)
                except Exception:
                    continue
            log.info("Volumen de la música restaurado.")
        except Exception:
            log.debug("No se pudo restaurar el volumen.", exc_info=True)
        finally:
            self._previos = {}
