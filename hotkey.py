"""Activación por ATAJO DE TECLADO global (press-to-wake).

Tercer canal de activación, junto a la doble palmada y la wake word: cuando suena música el
micrófono se ensordece (sin cancelación de eco no hay arreglo por software en Python 3.14),
pero el teclado siempre responde. Un toque al atajo equivale a la doble palmada.

Win32 puro (RegisterHotKey vía ctypes): sin dependencias nuevas y sin hooks de teclado de bajo
nivel (el paquete `keyboard` instala un hook global frágil y a veces pide admin). Reglas Win32
que condicionan el diseño:
  - RegisterHotKey debe llamarse EN EL MISMO HILO que ejecuta GetMessage (el WM_HOTKEY llega a
    la cola de mensajes del hilo que lo registró) -> hilo dedicado con su propio message loop.
  - Para parar el hilo se envía WM_QUIT con PostThreadMessage (GetMessage devuelve 0 y sale).

El atajo solo surte efecto cuando Jarvis está a la ESPERA de activación (los toques durante una
conversación se descartan, igual que una palmada extra).
"""
from __future__ import annotations

import ctypes
import logging
import threading

log = logging.getLogger("jarvis")

# Constantes Win32 (winuser.h)
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000  # mantener pulsado NO repite el disparo
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

_MODS = {
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT, "mayus": MOD_SHIFT, "mayús": MOD_SHIFT,
    "win": MOD_WIN, "windows": MOD_WIN,
}
_TECLAS = {"espacio": 0x20, "space": 0x20}


def _parse_combo(combo: str) -> tuple[int, int]:
    """Convierte "ctrl+alt+j" en (modificadores, virtual-key). Lanza ValueError si no se entiende.

    Acepta ctrl/control, alt, shift/mayús, win/windows como modificadores y como tecla final una
    letra, un dígito, F1-F24 o espacio. Exige al menos un modificador (una letra sola secuestraría
    la tecla en TODO Windows).
    """
    partes = [p.strip().lower() for p in (combo or "").split("+") if p.strip()]
    if len(partes) < 2:
        raise ValueError(f"Atajo '{combo}': se requiere al menos un modificador (p.ej. ctrl+alt+j).")
    mods = 0
    for p in partes[:-1]:
        if p not in _MODS:
            raise ValueError(f"Atajo '{combo}': modificador desconocido '{p}'.")
        mods |= _MODS[p]
    tecla = partes[-1]
    if tecla in _TECLAS:
        return mods, _TECLAS[tecla]
    if len(tecla) == 1 and (tecla.isalpha() or tecla.isdigit()):
        return mods, ord(tecla.upper())
    if tecla.startswith("f") and tecla[1:].isdigit():
        n = int(tecla[1:])
        if 1 <= n <= 24:
            return mods, 0x70 + n - 1  # VK_F1..VK_F24
    raise ValueError(f"Atajo '{combo}': tecla final desconocida '{tecla}'.")


class GlobalHotkey:
    """Atajo de teclado global que dispara ``on_fire()`` (thread-safe) al pulsarse.

    ``start()`` devuelve True solo si el atajo quedó registrado en Windows; si otra app ya lo
    usa (o el combo es inválido), queda inactivo con un warning y Jarvis sigue con palmadas y
    wake word (fail-open al resto de canales).
    """

    _next_id = 1
    _id_lock = threading.Lock()

    def __init__(self, cfg: dict, on_fire) -> None:
        cfg = cfg if isinstance(cfg, dict) else {}
        self.combo: str = str(cfg.get("combinacion", "ctrl+alt+j"))
        self._on_fire = on_fire
        self._thread: threading.Thread | None = None
        self._tid: int | None = None  # id NATIVO del hilo del message loop (para WM_QUIT)
        self._ready = threading.Event()
        self._registered = False
        self._parse_error: str | None = None
        with GlobalHotkey._id_lock:
            self._id = GlobalHotkey._next_id
            GlobalHotkey._next_id += 1
        try:
            self._mods, self._vk = _parse_combo(self.combo)
        except ValueError as e:
            self._parse_error = str(e)
            self._mods = self._vk = 0

    @property
    def active(self) -> bool:
        return self._registered

    def start(self) -> bool:
        """Arranca el hilo del atajo y espera a saber si el registro tuvo éxito."""
        if self._parse_error:
            log.warning("Atajo global desactivado: %s", self._parse_error)
            return False
        if self._thread is not None:
            return self._registered
        self._thread = threading.Thread(target=self._run, name="jarvis-hotkey", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)
        return self._registered

    def stop(self) -> None:
        """Para el message loop (WM_QUIT) y espera al hilo. Inofensivo si no arrancó."""
        if self._tid is not None:
            try:
                ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
            except Exception:
                log.debug("No se pudo enviar WM_QUIT al hilo del atajo.", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        """Hilo dedicado: registra el atajo y bombea mensajes hasta WM_QUIT."""
        try:
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
        except Exception:  # plataforma sin Win32: queda inactivo
            log.warning("Atajo global no disponible en esta plataforma.")
            self._ready.set()
            return
        self._tid = int(kernel32.GetCurrentThreadId())
        try:
            if not user32.RegisterHotKey(None, self._id, self._mods | MOD_NOREPEAT, self._vk):
                log.warning("No se pudo registrar el atajo global '%s' (¿lo usa otra aplicación?).",
                            self.combo)
                self._ready.set()
                return
            self._registered = True
            self._ready.set()
            log.info("Atajo global activo: %s", self.combo)
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam == self._id:
                    log.info("Atajo global pulsado (%s).", self.combo)
                    try:
                        self._on_fire()
                    except Exception:
                        log.exception("Fallo en el callback del atajo global.")
        except Exception:
            log.exception("Error en el hilo del atajo global.")
            self._ready.set()
        finally:
            if self._registered:
                try:
                    user32.UnregisterHotKey(None, self._id)
                except Exception:
                    pass
            self._registered = False
