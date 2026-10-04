"""Fase 4 — control de Windows con lista blanca (sin abrir nada real)."""
from __future__ import annotations

import tools.windows_control as wc

APPS = {
    "apps": {
        "ninjatrader": {
            "titulo": "NinjaTrader",
            "nombres": ["ninjatrader", "ninja"],
            "lanzar": {"tipo": "startfile", "valor": "X:\\nt.lnk"},
        },
        "tradingview": {
            "titulo": "TradingView",
            "nombres": ["tradingview", "trading view"],
            "lanzar": {"tipo": "aumid", "valor": "TV.Desktop!App"},
        },
        "explorador_archivos": {
            "titulo": "el explorador",
            "nombres": ["explorador", "archivos"],
            "lanzar": {"tipo": "comando", "valor": "explorer.exe"},
        },
    }
}


def test_abrir_startfile(monkeypatch):
    visto = {}
    monkeypatch.setattr(wc.os, "startfile", lambda p: visto.setdefault("p", p), raising=False)
    ok, msg = wc.abrir_app("ninja", APPS)
    assert ok and "NinjaTrader" in msg
    assert visto["p"] == "X:\\nt.lnk"


def test_abrir_aumid_via_explorer(monkeypatch):
    visto = {}
    monkeypatch.setattr(wc.subprocess, "Popen", lambda args, **k: visto.setdefault("args", args))
    ok, _ = wc.abrir_app("trading view", APPS)
    assert ok
    assert visto["args"][0] == "explorer.exe"
    assert "shell:AppsFolder\\TV.Desktop!App" in visto["args"][1]


def test_abrir_comando(monkeypatch):
    visto = {}
    monkeypatch.setattr(wc.subprocess, "Popen", lambda args, **k: visto.setdefault("args", args))
    ok, _ = wc.abrir_app("archivos", APPS)
    assert ok and visto["args"] == ["explorer.exe"]


def test_match_insensible_a_acentos_y_mayusculas(monkeypatch):
    monkeypatch.setattr(wc.os, "startfile", lambda p: None, raising=False)
    ok, _ = wc.abrir_app("NINJA", APPS)
    assert ok


def test_app_fuera_de_la_lista_blanca():
    ok, msg = wc.abrir_app("photoshop", APPS)
    assert not ok
    assert "lista de aplicaciones" in msg


def test_poner_musica_youtube(monkeypatch):
    visto = {}
    monkeypatch.setattr(wc.webbrowser, "open", lambda u: visto.setdefault("url", u))
    ok, msg = wc.poner_musica(APPS, {"motor": "youtube", "youtube_url": "https://x.test/lista"})
    assert ok and "música" in msg.lower()
    assert visto["url"] == "https://x.test/lista"


def test_poner_musica_spotify_sin_spotify_en_lista():
    ok, msg = wc.poner_musica(APPS, {"motor": "spotify"})  # APPS no incluye spotify
    assert not ok and "música" in msg.lower()


def test_poner_musica_spotify_navega_al_contexto(monkeypatch):
    """Con spotify_uri, navega al contexto (Tus me gusta / DJ) y manda reproducir; NO
    necesita que 'spotify' esté en la lista blanca (el protocolo lo resuelve Windows)."""
    visto = {}
    monkeypatch.setattr(wc.os, "startfile", lambda u: visto.setdefault("uri", u), raising=False)
    monkeypatch.setattr(wc, "_reproducir", lambda: visto.setdefault("play", True))
    ok, msg = wc.poner_musica(APPS, {"motor": "spotify", "spotify_uri": "spotify:collection:tracks"})
    assert ok and "música" in msg.lower()
    assert visto["uri"] == "spotify:collection:tracks"
    assert visto.get("play") is True


def test_poner_musica_spotify_uri_no_spotify_se_rechaza(monkeypatch):
    """Solo URIs 'spotify:...': una URL http de la config NO se manda a startfile."""
    def no_debio(_u):
        raise AssertionError("startfile no debió llamarse con una URI no spotify:")
    monkeypatch.setattr(wc.os, "startfile", no_debio, raising=False)
    ok, _ = wc.poner_musica(APPS, {"motor": "spotify", "spotify_uri": "https://open.spotify.com/x"})
    assert not ok  # cae a abrir la app y APPS no incluye spotify


def test_poner_musica_spotify_protocolo_roto_cae_a_la_app(monkeypatch):
    def boom(_u):
        raise OSError("protocolo spotify: no registrado")
    monkeypatch.setattr(wc.os, "startfile", boom, raising=False)
    ok, msg = wc.poner_musica(APPS, {"motor": "spotify", "spotify_uri": "spotify:collection:tracks"})
    assert not ok and "Spotify" in msg  # sin app en la lista: falla LIMPIO, sin excepción


def test_poner_musica_dj(monkeypatch):
    """'pon el DJ' -> navega al contexto del DJ (spotify_uri_dj o el default conocido)."""
    visto = {}
    monkeypatch.setattr(wc.os, "startfile", lambda u: visto.setdefault("uri", u), raising=False)
    monkeypatch.setattr(wc, "_reproducir", lambda: None)
    ok, _ = wc.poner_musica(APPS, {"motor": "spotify", "spotify_uri": "spotify:collection:tracks"},
                            contexto="el dj")
    assert ok and visto["uri"] == wc.DJ_URI_DEFAULT
    visto.clear()
    ok, _ = wc.poner_musica(APPS, {"motor": "spotify", "spotify_uri": "spotify:collection:tracks",
                                   "spotify_uri_dj": "spotify:playlist:XYZ"}, contexto="dj")
    assert ok and visto["uri"] == "spotify:playlist:XYZ"
    visto.clear()
    ok, _ = wc.poner_musica(APPS, {"motor": "spotify", "spotify_uri": "spotify:collection:tracks"})
    assert ok and visto["uri"] == "spotify:collection:tracks"  # sin contexto: la de siempre
