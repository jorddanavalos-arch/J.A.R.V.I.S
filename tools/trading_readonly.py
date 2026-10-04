"""Resúmenes de trading en modo SOLO LECTURA (Fase 5).

Fuentes (sin tocar el repo del bot):
  - MCP ``tradingview`` (precio, indicadores, estado del gráfico).
  - Archivos de estado / exports de NinjaTrader 8 declarados en config.rutas.

Jarvis NUNCA envía órdenes ni mueve dinero: solo lee y resume.
"""
from __future__ import annotations


def resumen_mercado(config: dict) -> str:
    """Resumen del mercado/indicadores vía MCP tradingview. Implementación: Fase 5."""
    raise NotImplementedError("resumen_mercado: pendiente de la Fase 5.")


def resumen_sesion(config: dict) -> str:
    """Resumen de la sesión leyendo exports NT8 (solo lectura). Implementación: Fase 5."""
    raise NotImplementedError("resumen_sesion: pendiente de la Fase 5.")
