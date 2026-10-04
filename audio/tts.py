"""Síntesis de voz local (TTS) con Piper (Fase 2) — voz española tipo Jarvis.

Carga perezosa y cacheada de la voz (singleton). speak() sintetiza y reproduce en streaming
por sd.OutputStream. Si la voz no está disponible, NO rompe el bucle: registra y muestra el
texto por consola. synth_to_array() devuelve el audio sin reproducir (para el test de integración).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger("jarvis")
ROOT = Path(__file__).resolve().parent.parent


def _as_settings(config: dict) -> dict:
    if isinstance(config, dict) and isinstance(config.get("settings"), dict):
        return config["settings"]
    return config if isinstance(config, dict) else {}


class TTSEngine:
    _voice = None  # singleton de proceso
    _voice_unavailable = False  # evita repetir el traceback en cada activación si falta la voz

    def __init__(self, config: dict) -> None:
        self.config = config
        self.settings = _as_settings(config)
        tts = self.settings.get("tts", {})
        self.motor = tts.get("motor", "piper")
        self.voz_path = tts.get(
            "voz_path", "models/piper/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
        )
        self.length_scale = float(tts.get("length_scale", 1.08))
        self.noise_scale = float(tts.get("noise_scale", 0.6))
        self.noise_w_scale = float(tts.get("noise_w_scale", 0.8))
        self.volume = float(tts.get("volume", 1.0))

    def _voz_abspath(self) -> Path:
        p = Path(self.voz_path)
        return p if p.is_absolute() else (ROOT / p)

    def _syn_config(self):
        try:
            from piper import SynthesisConfig

            return SynthesisConfig(
                length_scale=self.length_scale,
                noise_scale=self.noise_scale,
                noise_w_scale=self.noise_w_scale,
                # normalize_audio=False: Piper normaliza CADA frase a fondo de escala por
                # separado (un chunk por frase), lo que reescala la ganancia frase a frase en
                # respuestas multi-frase. Lo desactivamos para mantener un nivel uniforme y que
                # 'volume' module el audio crudo (predecible). Aplica a synth_to_array() y speak().
                normalize_audio=False,
                volume=self.volume,
            )
        except Exception:
            log.warning("SynthesisConfig no acepta los parámetros; uso defaults de Piper.", exc_info=True)
            return None

    def _ensure_voice(self):
        if TTSEngine._voice is None:
            from piper import PiperVoice

            onnx = self._voz_abspath()
            if not onnx.exists():
                raise FileNotFoundError(f"No existe la voz Piper: {onnx}")
            cfg = Path(str(onnx) + ".json")
            log.info("Cargando voz Piper %s...", onnx.name)
            TTSEngine._voice = PiperVoice.load(str(onnx), config_path=str(cfg) if cfg.exists() else None)
        return TTSEngine._voice

    def synth_to_array(self, text: str):
        """Devuelve (audio int16 mono, sample_rate) sin reproducir. Para tests/integración."""
        voice = self._ensure_voice()
        chunks = list(voice.synthesize(text, self._syn_config()))
        if not chunks:
            return np.zeros(0, dtype=np.int16), 22050
        sr = int(chunks[0].sample_rate)
        audio = np.concatenate([np.asarray(c.audio_int16_array, dtype=np.int16) for c in chunks])
        return audio, sr

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        # Respuestas LARGAS: por ORACIONES con síntesis adelantada — la primera suena mientras
        # se sintetiza la siguiente (menos espera hasta oír algo). Cada oración es un clip
        # completo, así que NO reaparece el tartamudeo del streaming chunk-a-chunk. La
        # inferencia corre en un worker; la reproducción (PortAudio) queda en ESTE hilo.
        partes = _split_oraciones(text) if len(text) > 140 else [text]
        if len(partes) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-tts") as pool:
                futuros = [pool.submit(self.synth_to_array, p) for p in partes]
                for parte, futuro in zip(partes, futuros):
                    try:
                        audio, sr = futuro.result()
                    except Exception:
                        self._avisar_sin_voz(parte)
                        continue
                    self._reproducir_clip(audio, sr, parte)
            return
        try:
            audio, sr = self.synth_to_array(text)
        except Exception:
            self._avisar_sin_voz(text)
            return
        self._reproducir_clip(audio, sr, text)

    def _avisar_sin_voz(self, text: str) -> None:
        if not TTSEngine._voice_unavailable:  # registra el traceback solo la primera vez
            log.warning("Voz Piper no disponible; muestro el texto.", exc_info=True)
            TTSEngine._voice_unavailable = True
        print(f"[JARVIS dice] {text}")

    def _reproducir_clip(self, audio, sr, text: str) -> None:
        if getattr(audio, "size", 0) == 0:
            return
        stream = None
        try:
            stream = sd.OutputStream(samplerate=int(sr), channels=1, dtype="int16")
            stream.start()
            stream.write(np.ascontiguousarray(audio, dtype=np.int16))  # write exige C-contiguous
        except Exception:
            log.warning("Error reproduciendo TTS; muestro el texto.", exc_info=True)
            print(f"[JARVIS dice] {text}")
        finally:
            # Cierra SIEMPRE el OutputStream (también si write() falla) -> sin fugas.
            if stream is not None:
                for fn in ("stop", "close"):
                    try:
                        getattr(stream, fn)()
                    except Exception:
                        pass


def _split_oraciones(texto: str) -> list[str]:
    """Parte por fin de oración (. ? ! ;) conservando el signo; agrupa trozos muy cortos
    con el siguiente para no hacer pausas antinaturales."""
    crudos = [p.strip() for p in re.split(r"(?<=[.?!;])\s+", texto) if p.strip()]
    partes: list[str] = []
    for p in crudos:
        if partes and len(partes[-1]) < 25:
            partes[-1] = f"{partes[-1]} {p}"
        else:
            partes.append(p)
    return partes or [texto]
