"""Calibración/diagnóstico de la VOZ en vivo (Fase 2), con micrófono real.

Graba una frase, mide latencias por etapa (captura / whisper / tts) y muestra la
transcripción. Cumple la lección de la Fase 1: el camino real se EJECUTA y se MIDE.

Uso:  .venv\\Scripts\\python scripts\\calibrar_voz.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from audio.mic_capture import MicCapture
from audio.stt import STTEngine
from audio.tts import TTSEngine
from config import load_settings


def main() -> None:
    settings = load_settings()
    stt = STTEngine(settings)
    tts = TTSEngine(settings)
    print("[CALIB VOZ] Pulsa ENTER y habla una frase. Ctrl-C para salir.")
    try:
        while True:
            input("\nENTER para grabar... ")
            t0 = time.time()
            clip, motivo = MicCapture(settings).record_utterance()
            t1 = time.time()
            n = 0 if clip is None else clip.size
            print(f"  captura: {t1 - t0:.2f}s | motivo={motivo} | muestras={n} (~{n/16000:.1f}s)")
            if clip is None:
                tts.speak("No le he oído, señor.")
                continue
            texto = stt.transcribe_array(clip, 16000)
            t2 = time.time()
            print(f"  whisper: {t2 - t1:.2f}s -> {texto!r}")
            tts.speak(f"Entendí: {texto}" if texto else "No he entendido, señor.")
            print(f"  tts: {time.time() - t2:.2f}s")
    except KeyboardInterrupt:
        pass
    print("\n[CALIB VOZ] Fin.")


if __name__ == "__main__":
    main()
