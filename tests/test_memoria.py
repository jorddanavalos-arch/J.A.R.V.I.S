"""Memoria persistente v1: tool ``recordar`` (append-only con guardarraíles), carga
acotada e inyección del system prompt CON memoria en ambos backends."""
from __future__ import annotations

import asyncio
import datetime
from types import SimpleNamespace

import brain.agent as agent
from brain.agent import _LOCAL_TOOLS, SYSTEM_PROMPT, Brain
from brain.suscripcion import CerebroSuscripcion
from tools import memoria

_STUB = (
    "# Memoria de Jarvis\n\n"
    "## Perfil\n- Usuario: Kevin. Trader MNQ.\n\n"
    "## Preferencias\n- (pendiente)\n"
)


def _archivo(tmp_path, contenido=_STUB):
    p = tmp_path / "jarvis_memory.md"
    p.write_text(contenido, encoding="utf-8")
    return p


# ---------------------------------------------------------------- recordar

def test_recordar_appende_con_fecha_y_crea_seccion(tmp_path):
    p = _archivo(tmp_path)  # el stub NO trae la sección Recuerdos
    ok, msg = memoria.recordar("opero de 8:30 a 11", p)
    assert ok and "opero de 8:30 a 11" in msg
    texto = p.read_text(encoding="utf-8")
    hoy = datetime.date.today().isoformat()
    assert texto.endswith(f"## Recuerdos\n- [{hoy}] opero de 8:30 a 11\n")
    assert texto.startswith(_STUB)  # append-only: lo existente quedó intacto
    ok, _ = memoria.recordar("segundo hecho", p)  # con la sección ya creada, solo la línea
    assert ok
    assert p.read_text(encoding="utf-8").count("## Recuerdos") == 1


def test_recordar_sanea_html_espacios_y_tope(tmp_path):
    p = _archivo(tmp_path)
    ok, _ = memoria.recordar("  <b>me gusta</b>\n\nel   café  " + "x" * 300, p)
    assert ok
    linea = p.read_text(encoding="utf-8").splitlines()[-1]
    assert "<" not in linea and ">" not in linea and "\n" not in linea
    assert "me gusta el café" in linea
    assert len(linea) <= len("- [2026-01-01] ") + 200  # hecho acotado a 200 chars


def test_recordar_vacio_no_escribe(tmp_path):
    p = _archivo(tmp_path)
    ok, msg = memoria.recordar("   ", p)
    assert not ok and "recordar" in msg
    assert p.read_text(encoding="utf-8") == _STUB


def test_recordar_rechaza_secretos(tmp_path):
    p = _archivo(tmp_path)
    for hecho in (
        "mi api key es sk-ant-abc123",
        "la contraseña del broker es 1234",
        "guarda este token: xyz",
    ):
        ok, msg = memoria.recordar(hecho, p)
        assert not ok and "seguridad" in msg
    assert p.read_text(encoding="utf-8") == _STUB  # nada llegó al archivo
    ok, _ = memoria.recordar("la clave está en la paciencia", p)  # "clave" benigno pasa
    assert ok


def test_recordar_tope_32kb_lo_dice(tmp_path):
    p = _archivo(tmp_path, _STUB + "## Recuerdos\n" + ("- [2026-01-01] relleno\n" * 2000))
    assert p.stat().st_size >= 32 * 1024
    antes = p.read_text(encoding="utf-8")
    ok, msg = memoria.recordar("uno más", p)
    assert not ok and "llena" in msg
    assert p.read_text(encoding="utf-8") == antes


# ---------------------------------------------------------------- cargar

def test_cargar_sin_archivo_devuelve_vacio(tmp_path):
    assert memoria.cargar(tmp_path / "no_existe.md") == ""


def test_cargar_trunca_los_recuerdos_mas_antiguos(tmp_path):
    viejas = "\n".join(f"- [2026-01-{d:02d}] recuerdo viejo {d}" for d in range(1, 10))
    nuevas = "\n".join(f"- [2026-08-{d:02d}] recuerdo nuevo {d} " + "x" * 120
                       for d in range(1, 28))
    p = _archivo(tmp_path, _STUB + "## Recuerdos\n" + viejas + "\n" + nuevas + "\n")
    texto = memoria.cargar(p)
    assert len(texto) <= 4000
    assert "## Perfil" in texto and "Trader MNQ" in texto      # la cabeza se conserva
    assert "recuerdo nuevo 27" in texto                        # lo más nuevo entra
    assert "recuerdo viejo 1" not in texto                     # lo más viejo se omite


# ---------------------------------------------------------------- inyección al prompt

def test_system_prompt_con_memoria(monkeypatch):
    monkeypatch.setattr(memoria, "cargar", lambda path=None: "## Perfil\n- dato X")
    sp = agent.system_prompt_con_memoria()
    assert sp.startswith(SYSTEM_PROMPT) and "- dato X" in sp
    monkeypatch.setattr(memoria, "cargar", lambda path=None: "")
    assert agent.system_prompt_con_memoria() == SYSTEM_PROMPT  # sin memoria, prompt base


def test_recordar_esta_en_las_tools_y_el_dispatch(monkeypatch, tmp_path):
    assert "recordar" in {t["name"] for t in _LOCAL_TOOLS}
    llamado = {}

    def _falso_recordar(hecho, path=None):
        llamado["hecho"] = hecho
        return True, f"Anotado: {hecho}"

    monkeypatch.setattr(memoria, "recordar", _falso_recordar)
    b = Brain({"settings": {"cerebro": {"backend": "api"}}, "apps": {}})
    out = b._ejecutar("recordar", {"hecho": "odia los lunes"})
    assert llamado["hecho"] == "odia los lunes" and "Anotado" in out


def test_backend_api_recompone_el_prompt_por_conversacion(monkeypatch):
    memorias = iter(["MEMORIA UNO", "MEMORIA DOS"])
    monkeypatch.setattr(memoria, "cargar", lambda path=None: next(memorias))
    respuestas = [
        SimpleNamespace(content=[SimpleNamespace(type="text", text="ok1")], stop_reason="end_turn"),
        SimpleNamespace(content=[SimpleNamespace(type="text", text="ok2")], stop_reason="end_turn"),
        SimpleNamespace(content=[SimpleNamespace(type="text", text="ok3")], stop_reason="end_turn"),
    ]
    llamadas = []

    class _FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                llamadas.append(kw)
                return respuestas.pop(0)

    Brain._client = _FakeClient()
    try:
        b = Brain({"settings": {"cerebro": {"backend": "api"}}, "apps": {}})
        b.think("hola")
        b.think("sigo")   # misma conversación: NO se relee la memoria
        assert "MEMORIA UNO" in llamadas[0]["system"]
        assert llamadas[1]["system"] == llamadas[0]["system"]
        b.reset()
        b.think("otra")   # conversación nueva: memoria fresca
        assert "MEMORIA DOS" in llamadas[2]["system"]
    finally:
        Brain._client = None


def test_suscripcion_compone_el_prompt_al_conectar(monkeypatch):
    import claude_agent_sdk

    capturadas = []

    class _FakeCliente:
        def __init__(self, options):
            capturadas.append(options)

        async def connect(self):
            pass

        async def disconnect(self):
            pass

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeCliente)
    versiones = iter(["PROMPT v1", "PROMPT v2"])
    c = CerebroSuscripcion({}, lambda: next(versiones), _LOCAL_TOOLS, lambda n, e: "")
    asyncio.run(c._connect())
    assert capturadas[0].system_prompt == "PROMPT v1"
    asyncio.run(c._drop_client())
    asyncio.run(c._connect())   # conversación nueva -> prompt recompuesto (memoria fresca)
    assert capturadas[1].system_prompt == "PROMPT v2"


def test_suscripcion_sigue_aceptando_prompt_fijo(monkeypatch):
    import claude_agent_sdk

    capturadas = []

    class _FakeCliente:
        def __init__(self, options):
            capturadas.append(options)

        async def connect(self):
            pass

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeCliente)
    c = CerebroSuscripcion({}, "system fijo", _LOCAL_TOOLS, lambda n, e: "")
    asyncio.run(c._connect())
    assert capturadas[0].system_prompt == "system fijo"
