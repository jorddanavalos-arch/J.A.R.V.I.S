"""Fase 5 — herramientas de SOLO LECTURA (estado del bot + búsqueda en notas)."""
from __future__ import annotations

import datetime
import json

import brain.agent as ag
import tools.notas as notas
from brain.agent import Brain

METRICAS = {
    "fecha": "2026-08-07",
    "modo": "LIVE-SIM",
    "trades_ejecutados": 1,
    "ganadores": 1,
    "perdedores": 0,
    "break_even": 0,
    "win_rate_pct": 100.0,
    "pnl_usd": 425.5,
    "perdidas_consecutivas": 0,
    "guards_bloqueadores": {"filtro_0_matriz": 349, "estado_dia": 10},
    "reconciliado": True,
}


def _rutas_bot(tmp_path, nombre="metricas_2026-08-07.json", data=None):
    (tmp_path / nombre).write_text(
        json.dumps(data if data is not None else METRICAS), encoding="utf-8"
    )
    return {"estado_bot": str(tmp_path)}


# ---------------------------------------------------------------- estado_bot

def test_estado_bot_fecha_explicita(tmp_path):
    ok, msg = notas.estado_bot(_rutas_bot(tmp_path), "2026-08-07")
    assert ok
    assert "2026-08-07" in msg and "1 trade:" in msg and "1 ganador y" in msg
    assert "100 por ciento" in msg
    assert "+425.50" in msg and "filtro 0 matriz" in msg and "reconciliadas" in msg.lower()


def test_estado_bot_hoy_cae_al_mas_reciente(tmp_path):
    """En finde o con el bot apagado no hay métricas de hoy: da las últimas y lo AVISA."""
    ok, msg = notas.estado_bot(_rutas_bot(tmp_path), "hoy")
    hoy = datetime.date.today().isoformat()
    assert ok
    assert f"No hay métricas del {hoy}" in msg and "2026-08-07" in msg


def test_estado_bot_sin_trades(tmp_path):
    data = {"fecha": "2026-08-03", "trades_ejecutados": 0, "guards_bloqueadores": {"estado_dia": 4}}
    ok, msg = notas.estado_bot(_rutas_bot(tmp_path, "metricas_2026-08-03.json", data), "2026-08-03")
    assert ok and "no ejecutó trades" in msg


def test_estado_bot_errores(tmp_path):
    ok, msg = notas.estado_bot({}, "hoy")
    assert not ok and "No tengo configurada" in msg
    ok, msg = notas.estado_bot({"estado_bot": str(tmp_path / "no_existe")}, "hoy")
    assert not ok and "no existe" in msg
    ok, msg = notas.estado_bot(_rutas_bot(tmp_path), "el martes pasado")
    assert not ok and "No entiendo la fecha" in msg
    (tmp_path / "metricas_2026-08-08.json").write_text("{roto", encoding="utf-8")
    ok, msg = notas.estado_bot({"estado_bot": str(tmp_path)}, "2026-08-08")
    assert not ok and "dañado" in msg


def test_estado_bot_carpeta_vacia(tmp_path):
    ok, msg = notas.estado_bot({"estado_bot": str(tmp_path)}, "hoy")
    assert not ok and "No encuentro métricas" in msg


# ---------------------------------------------------------------- buscar_notas

def test_buscar_notas_sin_acentos_y_recientes_primero(tmp_path):
    (tmp_path / "vieja.md").write_text("La calibración de palmadas quedó pendiente.", encoding="utf-8")
    (tmp_path / "nueva.md").write_text("Pendiente: calibracion del microfono.", encoding="utf-8")
    import os
    os.utime(tmp_path / "vieja.md", (1000, 1000))  # mucho más vieja
    ok, msg = notas.buscar_notas({"notas": [str(tmp_path)]}, "calibración")
    assert ok
    assert "nueva.md" in msg and "vieja.md" in msg
    assert msg.index("nueva.md") < msg.index("vieja.md")  # lo reciente primero


def test_buscar_notas_ignora_html(tmp_path):
    (tmp_path / "cerebro.html").write_text("<p>palmada palmada</p>", encoding="utf-8")
    (tmp_path / "nota.md").write_text("una palmada real", encoding="utf-8")
    ok, msg = notas.buscar_notas({"notas": [str(tmp_path)]}, "palmada")
    assert ok
    assert "cerebro.html" not in msg and "nota.md" in msg


def test_buscar_notas_sin_resultado_y_validaciones(tmp_path):
    (tmp_path / "nota.md").write_text("nada que ver", encoding="utf-8")
    ok, msg = notas.buscar_notas({"notas": [str(tmp_path)]}, "inexistente")
    assert ok and "No encontré nada" in msg
    ok, msg = notas.buscar_notas({}, "algo")
    assert not ok and "No tengo configuradas" in msg
    ok, msg = notas.buscar_notas({"notas": [str(tmp_path)]}, "x")
    assert not ok and "dos letras" in msg


def test_buscar_notas_acepta_raiz_como_string_y_carpeta_rota(tmp_path):
    (tmp_path / "nota.txt").write_text("el bot MNQ opera", encoding="utf-8")
    rutas = {"notas": str(tmp_path)}  # string suelto, no lista
    ok, msg = notas.buscar_notas(rutas, "mnq")
    assert ok and "nota.txt" in msg
    ok, msg = notas.buscar_notas({"notas": [str(tmp_path / "fantasma")]}, "mnq")
    assert ok and "No encontré nada" in msg  # carpeta inexistente no peta


def test_buscar_notas_tope_de_salida(tmp_path):
    for i in range(30):
        (tmp_path / f"n{i:02d}.md").write_text("clave " + "x" * 300, encoding="utf-8")
    ok, msg = notas.buscar_notas({"notas": [str(tmp_path)]}, "clave")
    assert ok and len(msg) < 2600  # acotado aunque haya 30 archivos con match


# ---------------------------------------------------------------- cableado en el cerebro

def _brain(rutas):
    return Brain({"settings": {"rutas": rutas}, "apps": {}})


def test_brain_dispatch_estado_bot(monkeypatch):
    visto = {}
    monkeypatch.setattr(ag.notas, "estado_bot",
                        lambda rutas, fecha: visto.update(r=rutas, f=fecha) or (True, "RESUMEN"))
    out = _brain({"estado_bot": "X"})._ejecutar("estado_bot", {"fecha": "ayer"})
    assert out == "RESUMEN" and visto["f"] == "ayer" and visto["r"] == {"estado_bot": "X"}


def test_brain_dispatch_buscar_notas(monkeypatch):
    monkeypatch.setattr(ag.notas, "buscar_notas", lambda rutas, termino: (True, f"HITS:{termino}"))
    out = _brain({"notas": ["Y"]})._ejecutar("buscar_notas", {"termino": "pendientes"})
    assert out == "HITS:pendientes"


def test_herramientas_incluyen_las_nuevas():
    nombres = {t["name"] for t in _brain({})._herramientas() if isinstance(t.get("name"), str)}
    assert {"estado_bot", "buscar_notas"} <= nombres


# ---------------------------------------------------------------- calendario económico

def _rutas_noticias(tmp_path):
    cal = {"noticias": [
        {"fecha": "2026-08-12", "hora_ny": "12:30", "nombre": "Core CPI m/m", "impacto": 3},
        {"fecha": "2026-08-12", "hora_ny": "08:30", "nombre": "Discurso Fed", "impacto": 1},
        {"fecha": "2026-08-13", "hora_ny": "12:30", "nombre": "PPI", "impacto": 3},
    ]}
    p = tmp_path / "noticias_semana.json"
    p.write_text(json.dumps(cal), encoding="utf-8")
    return {"noticias": str(p)}


def test_noticias_de_una_fecha_ordenadas_por_hora(tmp_path):
    ok, msg = notas.noticias_economicas(_rutas_noticias(tmp_path), "2026-08-12")
    assert ok
    assert msg.index("08:30") < msg.index("12:30")  # orden por hora
    assert "Core CPI" in msg and "impacto ALTO" in msg and "impacto bajo" in msg
    assert "PPI" not in msg.replace("Core CPI", "")  # el evento del 13 no se cuela


def test_noticias_manana_y_sin_eventos(tmp_path, monkeypatch):
    import datetime as dt

    class _Hoy(dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 8, 11)

    monkeypatch.setattr(notas.datetime, "date", _Hoy)
    ok, msg = notas.noticias_economicas(_rutas_noticias(tmp_path), "mañana")
    assert ok and "2026-08-12" in msg and "Core CPI" in msg
    ok, msg = notas.noticias_economicas(_rutas_noticias(tmp_path), "hoy")
    assert ok and "No hay noticias" in msg


def test_noticias_errores(tmp_path):
    ok, msg = notas.noticias_economicas({}, "hoy")
    assert not ok and "No tengo configurado" in msg
    ok, msg = notas.noticias_economicas({"noticias": str(tmp_path / "nope.json")}, "hoy")
    assert not ok and "No encuentro" in msg


def test_brain_dispatch_noticias(monkeypatch):
    monkeypatch.setattr(ag.notas, "noticias_economicas",
                        lambda rutas, fecha: (True, f"CAL:{fecha}"))
    out = _brain({"noticias": "X"})._ejecutar("noticias_economicas", {"fecha": "mañana"})
    assert out == "CAL:mañana"


def test_noticias_autorefresco_cuando_falta_la_fecha(tmp_path, monkeypatch):
    """JSON viejo sin la fecha pedida -> corre el fetch del bot y relee."""
    import os
    p = tmp_path / "noticias_semana.json"
    p.write_text(json.dumps({"noticias": []}), encoding="utf-8")
    os.utime(p, (1000, 1000))  # viejísimo: el throttle de 6h no aplica
    # Script fake: escribe el evento pedido (simula fetch_calendario.py del bot).
    script = tmp_path / "fetch.py"
    script.write_text(
        "import json\n"
        f"json.dump({{'noticias': [{{'fecha': '2026-08-12', 'hora_ny': '12:30', "
        f"'nombre': 'CPI', 'impacto': 3}}]}}, open(r'{p}', 'w'))\n",
        encoding="utf-8",
    )
    rutas = {"noticias": str(p), "fetch_noticias": str(script)}
    ok, msg = notas.noticias_economicas(rutas, "2026-08-12")
    assert ok and "CPI" in msg and "Estados Unidos" in msg


def test_noticias_throttle_json_fresco_no_refresca(tmp_path):
    """JSON recién escrito sin la fecha -> NO corre el fetch (si no está, no hay)."""
    p = tmp_path / "noticias_semana.json"
    p.write_text(json.dumps({"noticias": []}), encoding="utf-8")  # mtime = ahora
    script = tmp_path / "fetch.py"
    script.write_text("raise SystemExit(1)\n", encoding="utf-8")  # si corriera, fallaría
    rutas = {"noticias": str(p), "fetch_noticias": str(script)}
    ok, msg = notas.noticias_economicas(rutas, "2026-08-20")
    assert ok and "No hay noticias" in msg and "Estados Unidos" in msg


def test_noticias_sin_script_configurado_sigue_normal(tmp_path):
    ok, msg = notas.noticias_economicas(_rutas_noticias(tmp_path), "2026-08-12")
    assert ok and "Core CPI" in msg
