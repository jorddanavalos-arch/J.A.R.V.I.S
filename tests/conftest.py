"""Fixtures globales de la suite."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ducking_inofensivo(monkeypatch):
    """Los tests del orquestador construyen Orchestrator REAL: sin esto, _on_wake bajaría
    el volumen del Spotify de la máquina en cada corrida de pytest. Los tests dedicados
    del ducking usan audio.ducking directamente y no pasan por aquí."""
    import main

    class _NoDuck:
        def __init__(self, *_a, **_k):
            pass

        def bajar(self):
            pass

        def restaurar(self):
            pass

    monkeypatch.setattr(main, "MusicDucker", _NoDuck)
