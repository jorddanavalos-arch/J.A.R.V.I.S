"""Estados del bucle de Jarvis."""
from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    """Estados de la máquina del orquestador.

    Flujo: IDLE -> WAKE -> LISTEN -> THINK -> SPEAK -> IDLE
    """

    IDLE = auto()    # en reposo, esperando la doble palmada
    WAKE = auto()    # activado, va a empezar a escuchar
    LISTEN = auto()  # capturando y transcribiendo voz (STT)
    THINK = auto()   # el cerebro decide y ejecuta acciones (Agent SDK)
    SPEAK = auto()   # responde con voz (TTS)

    def __str__(self) -> str:  # representación legible en logs y consola
        return self.name
