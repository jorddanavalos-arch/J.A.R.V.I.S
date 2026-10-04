"""Utilidad de consola en Windows.

Desactiva QuickEdit en la consola ACTUAL para que un clic accidental no pause
(congele) el bucle de Jarvis. Es el mismo problema conocido del bot de trading,
resuelto aquí a nivel de proceso con la API de consola de Windows.
"""
from __future__ import annotations

import sys


def disable_quickedit() -> bool:
    """Desactiva QuickEdit en la consola actual. Devuelve True si lo logró.

    No-op (devuelve False) fuera de Windows o si la consola no lo permite.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        STD_INPUT_HANDLE = -10
        ENABLE_EXTENDED_FLAGS = 0x0080
        ENABLE_QUICK_EDIT_MODE = 0x0040

        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        new_mode = (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception:
        return False
