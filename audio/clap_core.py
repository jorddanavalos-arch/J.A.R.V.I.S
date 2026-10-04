"""Núcleo DSP puro del detector de doble palmada (Fase 1).

NO importa sounddevice ni usa reloj de pared: es DETERMINISTA e invariante al troceado
en bloques, por lo que se prueba con arrays numpy sin micrófono. Recibe bloques de audio
mono float32 en [-1,1] y emite un ``ClapEvent`` al detectar dos palmadas válidas separadas
dentro de la ventana temporal.

Comité multi-criterio (AND) por cada onset, para minimizar falsos positivos:
  1) energía/RMS sobre umbral ADAPTATIVO (gatillo barato),
  2) forma del transitorio: crest factor alto + ataque abrupto + duración corta + decay rápido,
  3) confirmación espectral (banda ancha + energía de alta frecuencia) SOLO en el pico.
Una FSM exige dos onsets dentro de ventana_ms; refractario por-onset evita doble conteo y un
cooldown global bloquea re-disparos.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class ClapEvent:
    tipo: str
    t1_ms: float
    t2_ms: float
    dt_ms: float


def spectral_features(seg, sr: int, hf_cutoff_hz: float = 2000.0) -> tuple[float, float]:
    """Devuelve (planitud_espectral, ratio_energia_alta_frecuencia) de un segmento.

    - planitud (flatness) = media_geométrica(|X|) / media_aritmética(|X|): ~1 banda ancha,
      ~0 tonal.
    - hf_ratio = amplitud espectral en f>=hf_cutoff_hz / amplitud total (suma de |X|,
      no de |X|^2): alto = brillante.
    """
    seg = np.asarray(seg, dtype=np.float64)
    n = seg.size
    if n == 0:
        return 0.0, 0.0
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(seg * w))
    total = float(mag.sum()) + _EPS
    flatness = float(np.exp(np.mean(np.log(mag + _EPS))) / (mag.mean() + _EPS))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    hf = float(mag[freqs >= hf_cutoff_hz].sum()) / total
    return flatness, hf


class _FSM(Enum):
    CALIB = auto()
    IDLE = auto()
    WAIT_SECOND = auto()
    COOLDOWN = auto()


class ClapCore:
    SPEC_WIN = 256  # ventana FFT alrededor del pico

    def __init__(
        self,
        sample_rate: int,
        umbral_factor: float,
        ventana_ms,
        cooldown_ms: float,
        *,
        hop_ms: int = 10,
        crest_min: float = 2.5,
        attack_min: float = 3.0,
        attack_max: float = 0.0,
        onset_max_ms: float = 120,
        decay_ratio_max: float = 0.5,
        refractario_ms: float = 90,
        ema_alpha: float = 0.03,
        piso_absoluto: float = 1e-4,
        hf_cutoff_hz: float = 2000,
        flatness_min: float = 0.3,
        hf_ratio_min: float = 0.4,
        clip_thresh: float = 0.98,
        clip_frac_min: float = 0.02,
        calibracion_ms: float = 400,
        exigir_espectro: bool = True,
        debug: bool = False,
    ) -> None:
        self.sr = int(sample_rate)
        self.umbral_factor = float(umbral_factor)
        try:
            win = list(ventana_ms)
        except TypeError as exc:
            raise ValueError("ventana_ms debe ser [win_lo_ms, win_hi_ms]") from exc
        if len(win) != 2:
            raise ValueError("ventana_ms debe tener exactamente 2 valores [lo, hi]")
        self.win_lo_ms = float(win[0])
        self.win_hi_ms = float(win[1])
        if not self.win_lo_ms < self.win_hi_ms:
            raise ValueError("ventana_ms requiere win_lo < win_hi")
        self.cooldown_ms = float(cooldown_ms)
        if self.cooldown_ms < 0:
            raise ValueError("cooldown_ms no puede ser negativo")
        self.hop = max(1, round(self.sr * hop_ms / 1000.0))
        self.hop_ms_actual = self.hop / self.sr * 1000.0
        self.crest_min = float(crest_min)
        self.attack_min = float(attack_min)
        # Tope de ataque OPCIONAL (0 = sin tope). Dato del 11-ago: los GOLPES duros al
        # escritorio (mouse contra la mesa) dan ataque ~96-138 y hasta saturan el micro,
        # mientras las palmadas reales rondan 4-16 — un tope en ~45 los separa con margen.
        self.attack_max = float(attack_max)
        self.onset_max_ms = float(onset_max_ms)
        self.onset_max_n = round(self.sr * self.onset_max_ms / 1000.0)
        self.seg_cap_ms = max(self.onset_max_ms * 4.0, 400.0)  # tope de segmentación (incluye cola)
        self.decay_ratio_max = float(decay_ratio_max)
        self.refractario_n = round(self.sr * refractario_ms / 1000.0)
        self.cooldown_n = round(self.sr * self.cooldown_ms / 1000.0)
        self.calib_n = round(self.sr * calibracion_ms / 1000.0)
        self.ema_alpha = float(ema_alpha)
        self.piso_absoluto = float(piso_absoluto)
        self.hf_cutoff_hz = float(hf_cutoff_hz)
        self.flatness_min = float(flatness_min)
        self.hf_ratio_min = float(hf_ratio_min)
        self.clip_thresh = float(clip_thresh)
        self.clip_frac_min = float(clip_frac_min)
        self.exigir_espectro = bool(exigir_espectro)
        self._debug = bool(debug)
        self.reset()

    @classmethod
    def from_config(cls, settings: dict) -> "ClapCore":
        audio = settings.get("audio") if isinstance(settings, dict) else None
        audio = audio if isinstance(audio, dict) else {}
        p = audio.get("palmada")
        p = p if isinstance(p, dict) else {}
        sr = audio.get("sample_rate", 16000)
        sr = int(sr) if isinstance(sr, (int, float)) and sr and sr > 0 else 16000
        return cls(
            sample_rate=sr,
            umbral_factor=p.get("umbral_factor", 4.0),
            ventana_ms=p.get("ventana_ms", [120, 700]),
            cooldown_ms=p.get("cooldown_ms", 1500),
            hop_ms=p.get("hop_ms", 10),
            crest_min=p.get("crest_min", 2.5),
            attack_min=p.get("attack_min", 3.0),
            attack_max=p.get("attack_max", 0.0),
            onset_max_ms=p.get("onset_max_ms", 120),
            decay_ratio_max=p.get("decay_ratio_max", 0.5),
            refractario_ms=p.get("refractario_ms", 90),
            ema_alpha=p.get("ema_alpha", 0.03),
            piso_absoluto=p.get("piso_absoluto", 1e-4),
            hf_cutoff_hz=p.get("hf_cutoff_hz", 2000),
            flatness_min=p.get("flatness_min", 0.3),
            hf_ratio_min=p.get("hf_ratio_min", 0.4),
            clip_thresh=p.get("clip_thresh", 0.98),
            clip_frac_min=p.get("clip_frac_min", 0.02),
            calibracion_ms=p.get("calibracion_ms", 400),
            exigir_espectro=p.get("exigir_espectro", True),
        )

    def reset(self) -> None:
        self.n = 0  # muestras procesadas (timestamps deterministas)
        self._carry = np.zeros(0, dtype=np.float32)
        self._recent = np.zeros(0, dtype=np.float32)
        self._recent_max = max(8192, self.hop * 16, self.SPEC_WIN * 8)
        self.noise_floor = self.piso_absoluto
        self._seeded = False
        self._calib_rms: list[float] = []
        self.prev_rms = 0.0
        self.state = _FSM.CALIB if self.calib_n > 0 else _FSM.IDLE
        self.t1_n = 0
        self.refractory_until = 0
        self.cooldown_until = 0
        self._in_onset = False
        self._onset_loud_frames = 0
        self._onset_peak_rms = 0.0
        self._onset_peak_crest = 0.0
        self._onset_peak_clipfrac = 0.0
        self._onset_peak_n = 0
        self._onset_start_n = 0
        self._onset_rms_before = 0.0
        self.debug_onsets: list[float] = []  # ms de onsets confirmados (para tests/tuning)
        self.debug_eval: list[dict] = []  # evaluación por onset cuando debug=True (tuning)

    def push(self, block) -> ClapEvent | None:
        """Alimenta un bloque mono float32; devuelve el ClapEvent de doble palmada o None.

        Procesa TODOS los frames del bloque (determinismo) y devuelve el primer evento.
        """
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size:
            # Sanea la entrada: NaN/inf de drivers o glitches no deben envenenar el estado
            # (un solo NaN dejaría el noise_floor en NaN y el detector sordo para siempre).
            block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(block, -1.0, 1.0, out=block)
        if self._carry.size:
            block = np.concatenate([self._carry, block])
        nframes = block.size // self.hop
        event = None
        for i in range(nframes):
            frame = block[i * self.hop:(i + 1) * self.hop]
            ev = self._process_frame(frame)
            if ev is not None and event is None:
                event = ev
        self._carry = block[nframes * self.hop:].copy()
        return event

    # ------------------------------------------------------------------ internals
    def _append_recent(self, frame) -> None:
        self._recent = np.concatenate([self._recent, frame])
        if self._recent.size > self._recent_max:
            self._recent = self._recent[-self._recent_max:]

    def _extract(self, center_n: int, win: int):
        """Extrae ``win`` muestras centradas en ``center_n`` del buffer reciente (con padding)."""
        half = win // 2
        start = center_n - half
        recent_end = self.n + self.hop  # el frame actual ya está en _recent
        rstart = recent_end - self._recent.size
        out = np.zeros(win, dtype=np.float32)
        i0 = start - rstart
        src0 = max(i0, 0)
        src1 = min(i0 + win, self._recent.size)
        if src1 > src0:
            dst0 = src0 - i0
            out[dst0:dst0 + (src1 - src0)] = self._recent[src0:src1]
        return out

    def _spectral_ok(self, peak_n: int) -> bool:
        flat, hf = spectral_features(self._extract(peak_n, self.SPEC_WIN), self.sr, self.hf_cutoff_hz)
        return flat >= self.flatness_min and hf >= self.hf_ratio_min

    def _process_frame(self, frame) -> ClapEvent | None:
        self._append_recent(frame)
        frame64 = frame.astype(np.float64)
        rms = float(np.sqrt(np.mean(frame64 * frame64) + _EPS))
        peak = float(np.max(np.abs(frame64))) if frame64.size else 0.0
        crest = peak / (rms + _EPS)
        clip_frac = float(np.mean(np.abs(frame64) >= self.clip_thresh)) if frame64.size else 0.0
        frame_start = self.n

        # Siembra rápida solo cuando NO hay ventana de calibración (p.ej. tests). En vivo, el
        # piso se siembra de forma robusta al cerrar la CALIBRACIÓN (percentil).
        if self.calib_n == 0 and not self._seeded:
            self.noise_floor = rms
            self._seeded = True

        thr = self.umbral_factor * max(self.noise_floor, self.piso_absoluto)
        is_loud = rms >= thr

        # CALIBRACIÓN: acumula RMS y suprime disparos. Al terminar, siembra el piso con un
        # PERCENTIL BAJO (robusto): un 'pop' de apertura del driver, un carraspeo o la 1ª
        # palmada quedan como outliers y NO inflan el piso (antes lo envenenaban -> sordo).
        if self.state == _FSM.CALIB:
            self._calib_rms.append(rms)
            if self.n + self.hop >= self.calib_n:
                arr = np.asarray(self._calib_rms, dtype=np.float64)
                seed = float(np.percentile(arr, 25)) if arr.size else self.piso_absoluto
                self.noise_floor = max(seed, self.piso_absoluto)
                self._calib_rms = []
                self.state = _FSM.IDLE
            self.prev_rms = rms
            self.n += self.hop
            return None

        # COOLDOWN: ignorar onsets y NO actualizar el piso
        if self.state == _FSM.COOLDOWN:
            if self.n >= self.cooldown_until:
                self.state = _FSM.IDLE
            else:
                self.prev_rms = rms
                self.n += self.hop
                return None

        event = None
        # SEGMENTACIÓN DE ONSET
        if not self._in_onset:
            can_start = (
                is_loud
                and self.prev_rms < thr
                and self.n >= self.refractory_until
                and self.state in (_FSM.IDLE, _FSM.WAIT_SECOND)
            )
            if can_start:
                self._in_onset = True
                self._onset_loud_frames = 1
                self._onset_start_n = frame_start
                self._onset_rms_before = self.prev_rms
                self._onset_peak_rms = rms
                self._onset_peak_crest = crest
                self._onset_peak_clipfrac = clip_frac
                self._onset_peak_n = frame_start + int(np.argmax(np.abs(frame64)))
        else:
            if is_loud:
                self._onset_loud_frames += 1
                if rms > self._onset_peak_rms:
                    self._onset_peak_rms = rms
                    self._onset_peak_crest = crest
                    self._onset_peak_clipfrac = clip_frac
                    self._onset_peak_n = frame_start + int(np.argmax(np.abs(frame64)))
            # Tope de SEGMENTACIÓN (generoso, incluye la cola reverberante). La calidad de la
            # duración se evalúa por la prontitud del pico, NO por el total (ver _finalize).
            seg_over = self._onset_loud_frames * self.hop_ms_actual > self.seg_cap_ms
            if (not is_loud) or seg_over:
                event = self._finalize_onset(rms)

        # ACTUALIZACIÓN DEL PISO: fuera de onset, o dentro de un onset que ya superó la ventana
        # del transitorio (= sonido SOSTENIDO/ruido, no una palmada). Así el ruido continuo eleva
        # el piso aunque haya disparado un onset (evita quedarse "sordo" con un piso mal sembrado);
        # una palmada (transitorio corto) sigue sin inflar su propio umbral.
        en_transitorio = self._in_onset and (
            self._onset_loud_frames * self.hop_ms_actual <= self.onset_max_ms
        )
        if not en_transitorio:
            self.noise_floor = (1 - self.ema_alpha) * self.noise_floor + self.ema_alpha * rms

        self.prev_rms = rms
        self.n += self.hop
        return event

    def _finalize_onset(self, end_rms: float) -> ClapEvent | None:
        self._in_onset = False
        duration_ms = self._onset_loud_frames * self.hop_ms_actual
        peak_rms = self._onset_peak_rms
        piso = max(self.noise_floor, self.piso_absoluto)

        # Una palmada fuerte cerca del micro satura (clipping): el pico se aplana en +-1 y el
        # crest colapsa. Si el frame de pico está recortado, omitimos crest y confiamos en
        # ataque + espectro + forma (que sí se mantienen). Evita un falso negativo silencioso.
        clipped = self._onset_peak_clipfrac >= self.clip_frac_min
        crest_ok = clipped or (self._onset_peak_crest >= self.crest_min)
        attack_ratio = (peak_rms - self._onset_rms_before) / (piso + _EPS)
        attack_ok = attack_ratio >= self.attack_min and (
            self.attack_max <= 0 or attack_ratio <= self.attack_max
        )
        # Prontitud del pico: una palmada alcanza su pico enseguida (ataque seco), aunque su
        # cola/reverberación dure más. Medimos el transitorio (onset->pico), NO la duración
        # total: así una palmada reverberante se acepta y un sonido que crece lento se rechaza.
        peak_prompt_n = self._onset_peak_n - self._onset_start_n
        dur_ok = 0 <= peak_prompt_n <= self.onset_max_n
        decay_ok = end_rms <= self.decay_ratio_max * peak_rms + _EPS
        # Confirmación espectral OPCIONAL: si exigir_espectro=False, no se exige (oye palmadas
        # flojas/sordas que no tienen tanta firma de banda ancha/agudos).
        spec_ok = self._spectral_ok(self._onset_peak_n) if self.exigir_espectro else True
        accepted = crest_ok and attack_ok and dur_ok and decay_ok and spec_ok
        if self._debug:
            flat, hf = spectral_features(
                self._extract(self._onset_peak_n, self.SPEC_WIN), self.sr, self.hf_cutoff_hz
            )
            self.debug_eval.append({
                "crest": round(self._onset_peak_crest, 2),
                "clip": round(self._onset_peak_clipfrac, 2),
                "attack": round(attack_ratio, 1),
                "pico_ms": round(peak_prompt_n / self.sr * 1000.0, 1),
                "dur_ms": round(duration_ms, 1),
                "decay": round(end_rms / (peak_rms + _EPS), 2),
                "flat": round(flat, 2),
                "hf": round(hf, 2),
                "accept": bool(accepted),
            })
            if len(self.debug_eval) > 100:
                self.debug_eval = self.debug_eval[-100:]
        if not accepted:
            return None

        peak_n = self._onset_peak_n
        self.debug_onsets.append(peak_n / self.sr * 1000.0)
        self.refractory_until = self.n + self.hop + self.refractario_n

        if self.state == _FSM.IDLE:
            self.t1_n = peak_n
            self.state = _FSM.WAIT_SECOND
            return None

        if self.state == _FSM.WAIT_SECOND:
            dt_ms = (peak_n - self.t1_n) / self.sr * 1000.0
            if dt_ms < self.win_lo_ms:
                return None  # rebote/eco: ignorar
            if dt_ms > self.win_hi_ms:
                self.t1_n = peak_n  # demasiado separadas: este onset pasa a ser el 1º
                return None
            ev = ClapEvent(
                "doble_palmada",
                self.t1_n / self.sr * 1000.0,
                peak_n / self.sr * 1000.0,
                dt_ms,
            )
            self.state = _FSM.COOLDOWN
            self.cooldown_until = self.n + self.hop + self.cooldown_n
            return ev
        return None
