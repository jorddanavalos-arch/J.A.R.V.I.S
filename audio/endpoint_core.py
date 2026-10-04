"""Núcleo DSP puro del endpointer de voz (Fase 2).

Decide, SIN micrófono y de forma determinista, cuándo empieza y termina una frase a partir
de bloques de audio mono float32. Mismo patrón que ClapCore: energía/RMS sobre un umbral
ADAPTATIVO (piso de ruido por EMA + mini-calibración por percentil), invariante al troceado
en bloques, timestamps por muestras procesadas. Acumula el audio para devolver el clip de la
frase con PRE-ROLL (no comerse la primera sílaba).

FSM: CALIB -> ESPERANDO_VOZ -> EN_VOZ -> DONE
push(block) devuelve la lista de eventos ocurridos en el bloque:
  - VOZ_INICIO          : arranque de voz sostenido
  - VOZ_FIN(audio)      : silencio sostenido tras voz -> clip recortado (incluye pre_roll)
  - TIMEOUT_SIN_VOZ     : nadie habló dentro de sin_voz_timeout
  - TOPE_MAXIMO(audio)  : tope absoluto (red anti-cuelgue) -> clip acumulado
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class EndpointEvent:
    tipo: str
    audio: np.ndarray | None
    ms: float


class _EP(Enum):
    CALIB = auto()
    ESPERANDO_VOZ = auto()
    EN_VOZ = auto()
    DONE = auto()


class EndpointCore:
    def __init__(
        self,
        sample_rate: int,
        *,
        hop_ms: int = 20,
        calibracion_ms: float = 300,
        umbral_factor: float = 3.0,
        arranque_min_ms: float = 150,
        silencio_fin_ms: float = 800,
        pre_roll_ms: float = 300,
        sin_voz_timeout_ms: float = 6000,
        max_total_ms: float = 12000,
        ema_alpha: float = 0.05,
        floor_up_rate: float = 0.01,
        piso_absoluto: float = 1e-4,
    ) -> None:
        self.sr = int(sample_rate)
        self.umbral_factor = float(umbral_factor)
        self.ema_alpha = float(ema_alpha)
        self.floor_up_rate = float(floor_up_rate)
        self.piso_absoluto = float(piso_absoluto)
        self.hop = max(1, round(self.sr * hop_ms / 1000.0))
        self.hop_ms_actual = self.hop / self.sr * 1000.0
        self.calib_n = round(self.sr * calibracion_ms / 1000.0)
        self.arranque_frames = max(1, round(arranque_min_ms / self.hop_ms_actual))
        self.silencio_fin_n = round(self.sr * silencio_fin_ms / 1000.0)
        self.pre_roll_n = round(self.sr * pre_roll_ms / 1000.0)
        self.sin_voz_n = round(self.sr * sin_voz_timeout_ms / 1000.0)
        self.max_total_n = round(self.sr * max_total_ms / 1000.0)
        self.reset()

    @classmethod
    def from_config(cls, settings: dict, sample_rate: int = 16000) -> "EndpointCore":
        ep = (settings.get("audio", {}) if isinstance(settings, dict) else {}).get("endpoint", {})
        ep = ep if isinstance(ep, dict) else {}
        return cls(
            sample_rate,
            hop_ms=ep.get("hop_ms", 20),
            calibracion_ms=ep.get("calibracion_ms", 300),
            umbral_factor=ep.get("umbral_factor", 3.0),
            arranque_min_ms=ep.get("arranque_min_ms", 150),
            silencio_fin_ms=ep.get("silencio_fin_ms", 800),
            pre_roll_ms=ep.get("pre_roll_ms", 300),
            sin_voz_timeout_ms=ep.get("sin_voz_timeout_ms", 6000),
            max_total_ms=ep.get("max_total_ms", 12000),
            ema_alpha=ep.get("ema_alpha", 0.05),
            floor_up_rate=ep.get("floor_up_rate", 0.01),
            piso_absoluto=ep.get("piso_absoluto", 1e-4),
        )

    def reset(self) -> None:
        self.n = 0
        self._carry = np.zeros(0, dtype=np.float32)
        self._frames: list[np.ndarray] = []
        self.noise_floor = self.piso_absoluto
        self._calib_rms: list[float] = []
        self.state = _EP.CALIB if self.calib_n > 0 else _EP.ESPERANDO_VOZ
        self.voice_start_n = 0
        self.last_voice_n = 0
        self._run_start_n = 0
        self._run_frames = 0

    def push(self, block) -> list[EndpointEvent]:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size:
            block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(block, -1.0, 1.0, out=block)
        if self._carry.size:
            block = np.concatenate([self._carry, block])
        nframes = block.size // self.hop
        events: list[EndpointEvent] = []
        for i in range(nframes):
            ev = self._process_frame(block[i * self.hop:(i + 1) * self.hop])
            if ev is not None:
                events.append(ev)
        self._carry = block[nframes * self.hop:].copy()
        return events

    # ------------------------------------------------------------------ internals
    def _clip_from(self, start_n: int, end_n: int) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._frames)
        a = max(0, int(start_n))
        b = min(len(audio), int(end_n))
        return audio[a:b].copy() if b > a else np.zeros(0, dtype=np.float32)

    def _process_frame(self, frame) -> EndpointEvent | None:
        frame_start = self.n
        frame_end = self.n + self.hop
        if self.state == _EP.DONE:
            self.n = frame_end
            return None  # ya terminó: no acumular más frames ni trabajo

        self._frames.append(np.asarray(frame, dtype=np.float32).copy())
        frame64 = frame.astype(np.float64)
        rms = float(np.sqrt(np.mean(frame64 * frame64) + _EPS)) if frame64.size else 0.0

        # CALIBRACIÓN: siembra del piso por percentil (robusto a un pop inicial)
        if self.state == _EP.CALIB:
            self._calib_rms.append(rms)
            if frame_end >= self.calib_n:
                arr = np.asarray(self._calib_rms, dtype=np.float64)
                seed = float(np.percentile(arr, 25)) if arr.size else self.piso_absoluto
                self.noise_floor = max(seed, self.piso_absoluto)
                self._calib_rms = []
                self.state = _EP.ESPERANDO_VOZ
            self.n = frame_end
            return None

        # Piso de ruido ADAPTATIVO (min-follower): baja rápido a los valles y sube lento hacia el
        # nivel sostenido. Sigue el RUIDO (estacionario) y no la VOZ (fluctúa), y se adapta también
        # DENTRO de EN_VOZ: el ruido de fondo continuo deja de capturarse hasta el tope (~15s).
        if rms < self.noise_floor:
            self.noise_floor = max(rms, self.piso_absoluto)
        else:
            self.noise_floor = min(
                self.noise_floor * (1.0 + self.floor_up_rate), max(rms, self.piso_absoluto)
            )

        thr = self.umbral_factor * max(self.noise_floor, self.piso_absoluto)
        is_voice = rms >= thr
        ev: EndpointEvent | None = None

        if self.state == _EP.ESPERANDO_VOZ:
            if is_voice:
                if self._run_frames == 0:
                    self._run_start_n = frame_start
                self._run_frames += 1
                if self._run_frames >= self.arranque_frames:
                    self.voice_start_n = self._run_start_n
                    self.last_voice_n = frame_end
                    self.state = _EP.EN_VOZ
                    ev = EndpointEvent("VOZ_INICIO", None, self.voice_start_n / self.sr * 1000.0)
            else:
                self._run_frames = 0
                if frame_end >= self.calib_n + self.sin_voz_n:
                    ev = EndpointEvent("TIMEOUT_SIN_VOZ", None, frame_end / self.sr * 1000.0)
                    self.state = _EP.DONE
        elif self.state == _EP.EN_VOZ:
            if is_voice:
                self.last_voice_n = frame_end
            if (frame_end - self.last_voice_n) >= self.silencio_fin_n:
                clip = self._clip_from(self.voice_start_n - self.pre_roll_n, frame_end)
                ev = EndpointEvent("VOZ_FIN", clip, frame_end / self.sr * 1000.0)
                self.state = _EP.DONE
            elif (frame_end - self.voice_start_n) >= self.max_total_n:
                clip = self._clip_from(self.voice_start_n - self.pre_roll_n, frame_end)
                ev = EndpointEvent("TOPE_MAXIMO", clip, frame_end / self.sr * 1000.0)
                self.state = _EP.DONE

        self.n = frame_end
        return ev
