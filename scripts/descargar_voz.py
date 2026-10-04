"""Descarga idempotente de la voz Piper (es_ES-davefx-medium) y del modelo whisper.

Reproduce las descargas en otra máquina (models/ está en .gitignore).
Uso:  .venv\\Scripts\\python scripts\\descargar_voz.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import load_settings


def main() -> None:
    settings = load_settings()
    root = pathlib.Path(__file__).resolve().parent.parent
    base = root / "models" / "piper"
    base.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    rel = "es/es_ES/davefx/medium/es_ES-davefx-medium"
    for ext in (".onnx", ".onnx.json"):
        p = hf_hub_download("rhasspy/piper-voices", rel + ext, local_dir=str(base))
        print(f"[voz] {p}  ({pathlib.Path(p).stat().st_size} bytes)")

    modelo = settings.get("stt", {}).get("modelo", "small")
    download_root = settings.get("stt", {}).get("download_root", "models/whisper")
    compute = settings.get("stt", {}).get("compute_type", "int8")
    print(f"[whisper] descargando/verificando '{modelo}' ({compute})...")
    from faster_whisper import WhisperModel

    WhisperModel(modelo, device="cpu", compute_type=compute, download_root=download_root)
    print("[whisper] OK")
    print("Listo.")


if __name__ == "__main__":
    main()
