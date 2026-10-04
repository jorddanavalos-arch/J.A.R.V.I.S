"""HUD — servidor local de la esfera de archivos (whitelist de rutas, estado en vivo)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from hud.server import HUDServer


@pytest.fixture()
def hud(tmp_path):
    (tmp_path / "nota_alfa.md").write_text("x", encoding="utf-8")
    (tmp_path / "nota_beta.md").write_text("x", encoding="utf-8")
    (tmp_path / "secreto.json").write_text("{}", encoding="utf-8")  # NO es .md: no aparece
    h = HUDServer({"puerto": 0, "abrir_navegador": False, "nodos": "memoria"},
                  {"notas": [str(tmp_path)]})
    assert h.start() is True
    yield h
    h.stop()


def _get(h, ruta):
    return urllib.request.urlopen(f"{h.url}{ruta}", timeout=5)


def test_sirve_html_y_estado_en_vivo(hud):
    body = _get(hud, "/").read().decode("utf-8")
    assert "J.A.R.V.I.S." in body and "/estado" in body
    hud.set_estado("THINK", activaciones=3)
    hud.set_evento("usted", "qué hizo el bot")
    e = json.loads(_get(hud, "/estado").read())
    assert e == {"estado": "THINK", "activaciones": 3, "quien": "usted", "texto": "qué hizo el bot"}


def test_nodos_son_nombres_de_md_sin_contenido(hud):
    nodos = json.loads(_get(hud, "/nodos").read())
    assert nodos == ["Nota alfa", "Nota beta"]  # solo .md, embellecidos, sin extensión


def test_modo_sistema_por_defecto_da_lexico_tech():
    from hud.server import _NODOS_SISTEMA
    h = HUDServer({"puerto": 0}, {"notas": ["C:\\no\\importa"]})
    try:
        assert h.modo_nodos == "sistema"  # default pedido por Kevin (11-ago)
        nodos = h.nodos()
        assert nodos == _NODOS_SISTEMA[:140]
        assert "CLAUDE CORE" in nodos and "VWAP ENGINE" in nodos
    finally:
        h.stop()


def test_titulo_nodo_limpia_prefijos_y_fechas():
    from hud.server import _titulo_nodo
    assert _titulo_nodo("session_2026-07-08_calibracion_zonas") == "Calibracion zonas · 8 jul"
    assert _titulo_nodo("project_mnq_bot") == "Mnq bot"
    assert _titulo_nodo("feedback_workflow") == "Workflow"
    assert _titulo_nodo("MEMORY") == "MEMORY"  # sin prefijo ni fecha: solo capitaliza
    assert _titulo_nodo("reference_nt8_socket_marketposition") == "Nt8 socket marketposition"


def test_rutas_fuera_de_la_whitelist_dan_404(hud):
    for ruta in ("/otra", "/hud.html", "/../config/secrets.json", "/estado/x"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(hud, ruta)
        assert exc.value.code == 404


def test_texto_de_evento_se_acota(hud):
    hud.set_evento("jarvis", "x" * 1000)
    e = json.loads(_get(hud, "/estado").read())
    assert len(e["texto"]) == 600  # respuesta completa visible (600 desde el 11-ago)


def test_stop_apaga_y_reinicio_de_instancia_es_seguro(tmp_path):
    h = HUDServer({"puerto": 0}, {})
    assert h.start() and h.start()  # idempotente
    url = h.url
    h.stop()
    with pytest.raises(Exception):
        urllib.request.urlopen(f"{url}/estado", timeout=1)


def test_puerto_ocupado_no_revienta():
    a = HUDServer({"puerto": 0}, {})
    assert a.start()
    b = HUDServer({"puerto": a.puerto}, {})
    try:
        assert b.start() is False  # ocupado: warning y False, sin excepción
    finally:
        a.stop()
        b.stop()
