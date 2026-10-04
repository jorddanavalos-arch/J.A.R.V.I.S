"""Control de Windows con lista blanca y guardarraíles (Fase 4).

Reglas de seguridad:
  - Solo se abren apps declaradas en config/apps.json (lista blanca). Nada fuera de ella.
  - Cada app trae su método de lanzamiento: 'startfile' (ruta .lnk/.exe/.bat), 'aumid' (app de
    la Store por AppUserModelID) o 'comando' (ejecutable en PATH).
  - Devuelve (ok, mensaje) en español para que el cerebro lo confirme por voz.
"""
from __future__ import annotations

import logging
import os
import subprocess
import unicodedata
import webbrowser

log = logging.getLogger("jarvis")


def _norm(s: str) -> str:
    """Minúsculas sin acentos, para casar nombres dichos por voz."""
    s = unicodedata.normalize("NFKD", (s or "").lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def _buscar(nombre: str, apps: dict):
    n = _norm(nombre)
    if not n:
        return None, None
    for key, info in apps.items():
        for alias in info.get("nombres", []):
            a = _norm(alias)
            if a and (a == n or a in n or n in a):
                return key, info
    return None, None


def abrir_app(nombre: str, apps_config: dict) -> tuple[bool, str]:
    """Abre una app de la lista blanca por nombre. Devuelve (ok, mensaje en español)."""
    apps = (apps_config or {}).get("apps", {}) if isinstance(apps_config, dict) else {}
    key, info = _buscar(nombre, apps)
    if not info:
        return False, f"No tengo '{nombre}' en mi lista de aplicaciones, señor."
    lanzar = info.get("lanzar", {}) if isinstance(info, dict) else {}
    tipo, valor = lanzar.get("tipo"), lanzar.get("valor")
    titulo = info.get("titulo", key)
    try:
        if tipo == "aumid":
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{valor}"])
        elif tipo == "comando":
            subprocess.Popen([valor])
        elif tipo == "startfile":
            os.startfile(valor)  # type: ignore[attr-defined]  # noqa: S606 (solo rutas de la lista blanca)
        else:
            return False, f"No sé cómo abrir {titulo}, señor."
        log.info("Abriendo app '%s' (%s).", key, tipo)
        return True, f"Enseguida, señor. Abriendo {titulo}."
    except Exception:
        log.exception("Fallo abriendo la app '%s'.", key)
        return False, f"No he podido abrir {titulo}, señor."


DJ_URI_DEFAULT = "spotify:playlist:37i9dQZF1EYkqdzj48dyYq"  # el DJ de Spotify


def poner_musica(apps_config: dict, musica_cfg: dict | None = None,
                 contexto: str = "") -> tuple[bool, str]:
    """Pone música. motor 'spotify' (navega a un contexto por URI ``spotify:`` y manda
    reproducir) o 'youtube' (abre una lista en el navegador). ``contexto``: '' = la lista
    de siempre (spotify_uri, Tus me gusta); 'dj' = el DJ de Spotify (spotify_uri_dj)."""
    cfg = musica_cfg if isinstance(musica_cfg, dict) else {}
    if str(cfg.get("motor", "spotify")).lower() == "youtube":
        url = cfg.get("youtube_url") or "https://music.youtube.com/"
        try:
            webbrowser.open(url)
            log.info("Poniendo música (YouTube).")
            return True, "Poniendo música, señor."
        except Exception:
            log.exception("No se pudo abrir YouTube Music.")
            return False, "No he podido poner música, señor."
    # Spotify: primero navegar al contexto configurado (así el 'play' cae sobre él y no sobre
    # cualquier resto de cola). Solo se aceptan URIs 'spotify:...' (los maneja la app instalada).
    if "dj" in _norm(contexto):
        uri = str(cfg.get("spotify_uri_dj") or DJ_URI_DEFAULT).strip()
    else:
        uri = str(cfg.get("spotify_uri") or "").strip()
    abierto = False
    if uri.startswith("spotify:"):
        try:
            os.startfile(uri)  # type: ignore[attr-defined]  # noqa: S606 (protocolo spotify:)
            abierto = True
            log.info("Abriendo Spotify en el contexto %s.", uri)
        except Exception:
            log.debug("El protocolo spotify: no respondió; abro la app.", exc_info=True)
    if not abierto:
        ok, _msg = abrir_app("spotify", apps_config)
        if not ok:
            return False, "No he podido poner música, señor; no encuentro Spotify."
    try:
        _reproducir()
    except Exception:
        log.debug("No se pudo enviar la orden de reproducción.", exc_info=True)
    return True, "Poniendo música, señor."


def _reproducir(espera_s: float = 4.0) -> None:
    """Manda 'reproducir' a Spotify una vez que ha tenido tiempo de arrancar.

    Usa APPCOMMAND_MEDIA_PLAY (reproducir, NO alterna play/pausa) sobre la ventana de Spotify si
    la encuentra; si no, recurre a la tecla multimedia global Play/Pausa.
    """
    import ctypes
    import time

    time.sleep(espera_s)  # Spotify (Store) puede tardar varios segundos en estar listo
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    WM_APPCOMMAND, APPCOMMAND_MEDIA_PLAY = 0x0319, 46
    hwnd = user32.FindWindowW(None, "Spotify")  # título "Spotify" cuando está parado/inactivo
    if hwnd:
        user32.SendMessageW(hwnd, WM_APPCOMMAND, hwnd, APPCOMMAND_MEDIA_PLAY << 16)
        return
    # Respaldo: tecla multimedia global Play/Pausa (Spotify recién abierto suele estar en pausa).
    vk_play_pause, key_up = 0xB3, 0x0002
    user32.keybd_event(vk_play_pause, 0, 0, 0)
    user32.keybd_event(vk_play_pause, 0, key_up, 0)
