"""Calibración AUTOMÁTICA del detector de doble palmada.

Mide tu SALA EN SILENCIO y tus PALMADAS reales, calcula los umbrales que separan ambos y los
ESCRIBE en config/settings.json (con copia de seguridad .bak). Hazlo UNA VEZ en el sitio donde
uses a Jarvis. Después, reinícialo.

Uso:  .venv\\Scripts\\python scripts\\calibrar_palmada.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd

from audio.clap_core import spectral_features
from config import CONFIG_DIR


def _device_sr() -> int:
    try:
        return int(round(float(sd.query_devices(kind="input")["default_samplerate"])))
    except Exception:
        return 44100


def _grabar(sr: int, hop: int, segundos: float, titulo: str) -> np.ndarray:
    print(f"\n=== {titulo} ===")
    input("    Pulsa ENTER cuando estés listo... ")
    buf: list = []

    def cb(indata, frames, time_info, status):
        b = indata.mean(axis=1) if indata.ndim == 2 else indata.reshape(-1)
        buf.append(np.asarray(b, dtype=np.float32).copy())

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=hop, callback=cb):
        fin = time.time() + segundos
        while time.time() < fin:
            print(f"    grabando... {max(0.0, fin - time.time()):4.1f}s ", end="\r", flush=True)
            time.sleep(0.2)
    print("    grabado.                      ")
    return np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)


def _frames_rms(audio: np.ndarray, hop: int) -> np.ndarray:
    n = audio.size // hop
    if n == 0:
        return np.zeros(0)
    fr = audio[: n * hop].reshape(n, hop).astype(np.float64)
    return np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)


def _eventos(rms: np.ndarray, gate: float, hueco: int = 5) -> list[tuple[int, int]]:
    """Agrupa frames consecutivos por encima de `gate` en eventos (a, b)."""
    fuertes = np.where(rms >= gate)[0]
    if fuertes.size == 0:
        return []
    evs, ini, prev = [], int(fuertes[0]), int(fuertes[0])
    for idx in fuertes[1:]:
        if idx - prev > hueco:
            evs.append((ini, prev)); ini = int(idx)
        prev = int(idx)
    evs.append((ini, prev))
    return evs


def main() -> None:
    sr = _device_sr()
    hop = max(1, int(sr * 0.01))  # 10 ms
    print("=" * 60)
    print("  CALIBRACIÓN DE PALMADAS de Jarvis")
    print(f"  Micrófono a {sr} Hz. Sigue las instrucciones.")
    print("=" * 60)

    # ---- 1) SILENCIO ----
    sil = _grabar(sr, hop, 6.0, "PASO 1/2: SILENCIO total durante 6 s (no hagas NADA de ruido)")
    sil_rms = _frames_rms(sil, hop)
    if sil_rms.size < 10:
        print("\nNo se capturó audio. ¿El micrófono está conectado y con permiso?"); return
    ambiente = float(np.percentile(sil_rms, 90))
    ambiente_max = float(np.max(sil_rms))

    # ---- 2) PALMADAS ----
    clap = _grabar(sr, hop, 14.0, "PASO 2/2: APLAUDE 8 veces con tu fuerza NORMAL (una cada ~1,5 s)")
    clap_rms = _frames_rms(clap, hop)
    gate = max(ambiente_max * 2.5, ambiente * 5.0, 0.01)
    picos_rms, crests, flats, hfs = [], [], [], []
    for a, b in _eventos(clap_rms, gate):
        pico = a + int(np.argmax(clap_rms[a:b + 1]))
        picos_rms.append(float(clap_rms[pico]))
        centro = pico * hop + hop // 2
        win = clap[max(0, centro - 128): max(0, centro - 128) + 256]
        if win.size >= 64:
            fr = win.astype(np.float64)
            rms = float(np.sqrt(np.mean(fr * fr) + 1e-12))
            crests.append(float(np.max(np.abs(fr))) / (rms + 1e-12))
            flat, hf = spectral_features(win, sr, 2000.0)
            flats.append(float(flat)); hfs.append(float(hf))

    n = len(picos_rms)
    print(f"\nResultados: ambiente(p90)={ambiente:.4f}  ambiente_max={ambiente_max:.4f}  palmadas={n}")
    if n < 3:
        print("Detecté muy pocas palmadas. Acércate un poco al micro y vuelve a ejecutar la calibración.")
        return

    clap_min = float(np.percentile(picos_rms, 20))  # palmada más floja (robusto a un outlier)
    # Umbral efectivo: entre el ruido y la palmada más floja, sin pasarse (anti-falsos pero que oiga).
    t_efectivo = float(np.sqrt(max(ambiente_max, 1e-4) * clap_min))
    t_efectivo = min(max(t_efectivo, ambiente_max * 1.3), clap_min * 0.6)
    piso = round(max(ambiente, 1e-4), 5)
    reco = {
        "umbral_factor": round(float(np.clip(t_efectivo / max(piso, 1e-9), 2.0, 12.0)), 2),
        "piso_absoluto": piso,
        "crest_min": round(max(1.6, np.percentile(crests, 20) * 0.8), 2) if crests else 2.0,
        "flatness_min": round(max(0.1, np.percentile(flats, 20) * 0.8), 2) if flats else 0.2,
        "hf_ratio_min": round(max(0.1, np.percentile(hfs, 20) * 0.8), 2) if hfs else 0.2,
        "exigir_espectro": True,
    }
    if clap_min <= ambiente_max * 1.3:
        print("AVISO: tus palmadas no destacan mucho sobre el ruido de la sala; quizá haya algún")
        print("       falso disparo. Prueba en un sitio más silencioso o aplaude algo más fuerte.")

    print("\nVALORES CALCULADOS (audio.palmada):")
    for k, v in reco.items():
        print(f"   {k}: {v}")

    path = CONFIG_DIR / "settings.json"
    original = path.read_text(encoding="utf-8")
    path.with_suffix(".json.bak").write_text(original, encoding="utf-8")
    data = json.loads(original)
    data.setdefault("audio", {}).setdefault("palmada", {}).update(reco)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en {path}")
    print(f"(copia de seguridad en {path.name}.bak)")
    print("Reinicia Jarvis para aplicar la calibración.\n")


if __name__ == "__main__":
    main()
