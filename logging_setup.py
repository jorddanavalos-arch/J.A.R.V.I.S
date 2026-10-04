"""Configuración de logging de Jarvis: consola + archivos rotativos en ``logs/``.

Hay dos canales:
  - ``jarvis``           -> log general (consola + logs/jarvis.log)
  - ``jarvis.acciones``  -> auditoría de TODA acción ejecutada (logs/acciones.log)
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"


def setup_logging(level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("jarvis")
    if logger.handlers:  # idempotente: no duplicar handlers
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = RotatingFileHandler(
        LOG_DIR / "jarvis.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def action_logger() -> logging.Logger:
    """Logger de auditoría: registra cada acción (tool, args, resultado)."""
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("jarvis.acciones")
    if logger.handlers:
        return logger
    logger.setLevel("INFO")
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    file_handler = RotatingFileHandler(
        LOG_DIR / "acciones.log", maxBytes=1_000_000, backupCount=10, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
