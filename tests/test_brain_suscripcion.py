"""Cerebro por suscripción: selección de backend, capado de herramientas y delegación.

No lanza el CLI real (eso lo cubre el humo manual); aquí se prueba la construcción de las
opciones fail-closed y el cableado Brain <-> CerebroSuscripcion.
"""
from __future__ import annotations

import asyncio

import pytest

from brain.agent import _LOCAL_TOOLS, Brain
from brain.suscripcion import _DISALLOWED, CerebroSuscripcion


def _cerebro(**cfg):
    ejecutado = {}

    def ejecutor(nombre, entrada):
        ejecutado["llamada"] = (nombre, entrada)
        return f"OK {nombre}"

    c = CerebroSuscripcion({"web_search": True, **cfg}, "system", _LOCAL_TOOLS, ejecutor)
    return c, ejecutado


def test_brain_default_usa_suscripcion():
    b = Brain({"settings": {"cerebro": {}}, "apps": {}})
    assert b.backend == "suscripcion"
    assert b._suscripcion is not None


def test_brain_backend_api_no_construye_suscripcion():
    b = Brain({"settings": {"cerebro": {"backend": "api"}}, "apps": {}})
    assert b.backend == "api"
    assert b._suscripcion is None


def test_think_delega_en_suscripcion(monkeypatch):
    b = Brain({"settings": {"cerebro": {}}, "apps": {}})
    monkeypatch.setattr(b._suscripcion, "think", lambda t: f"SUB:{t}")
    assert b.think("hola") == "SUB:hola"


def test_reset_suelta_el_cliente_persistente(monkeypatch):
    b = Brain({"settings": {"cerebro": {}}, "apps": {}})
    sub = b._suscripcion
    b.reset()  # sin loop ni cliente: no-op seguro
    assert sub._client is None
    sub._loop = object()  # simula loop vivo con cliente conectado
    sub._client = object()
    llamado = {}
    monkeypatch.setattr(sub, "_run",
                        lambda coro, t: (llamado.setdefault("drop", True), coro.close())[0])
    b.reset()
    assert llamado.get("drop") and sub._client is None


def test_opciones_fail_closed():
    c, _ = _cerebro()
    opts = c._options_base
    assert opts["setting_sources"] == []          # sin CLAUDE.md/skills/hooks del usuario
    assert opts["strict_mcp_config"] is True      # solo NUESTRO MCP server
    assert set(opts["disallowed_tools"]) == set(_DISALLOWED)
    esperadas = {f"mcp__jarvis__{t['name']}" for t in _LOCAL_TOOLS} | {"WebSearch", "WebFetch"}
    assert set(opts["allowed_tools"]) == esperadas
    assert "WebFetch" not in opts["disallowed_tools"]  # sin ella no puede abrir la página pedida
    # NUNCA bypassPermissions: silencia can_use_tool (CanUseToolShadowedWarning) = fail-open.
    assert opts["permission_mode"] == "default"


def test_env_neutraliza_las_credenciales_de_api(monkeypatch):
    """El SDK MERGEA options.env sobre os.environ ({**os.environ, **options.env}): OMITIR
    una clave del dict NO la borra del proceso hijo. Con ANTHROPIC_API_KEY heredada el CLI
    cobra a la API de consola en vez de a la suscripción -> 14-ago: "Credit balance is too
    low" en cada turno con la suscripción al 0% de uso. Se neutraliza poniéndola a "".
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-de-prueba")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-de-prueba")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
    c, _ = _cerebro()
    env = c._options_base["env"]
    assert env["ANTHROPIC_API_KEY"] == ""     # presente y VACÍA, no ausente
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert env["CLAUDECODE"] == ""
    assert env["CLAUDE_CODE_SESSION_ID"] == ""


def test_sin_web_search_no_permite_websearch():
    c, _ = _cerebro(web_search=False)
    assert "WebSearch" not in c._options_base["allowed_tools"]
    assert "WebFetch" not in c._options_base["allowed_tools"]


def test_can_use_tool_deniega_lo_desconocido():
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny
    c, _ = _cerebro()
    puede = c._options_base["can_use_tool"]
    assert isinstance(asyncio.run(puede("mcp__jarvis__obtener_hora", {}, None)), PermissionResultAllow)
    assert isinstance(asyncio.run(puede("Bash", {"command": "rm -rf"}, None)), PermissionResultDeny)
    assert isinstance(asyncio.run(puede("mcp__otro__cosa", {}, None)), PermissionResultDeny)


def test_handler_de_tool_llama_al_ejecutor_compartido():
    """Los handlers MCP delegan en Brain._ejecutar: una sola implementación de herramientas."""
    c, ejecutado = _cerebro()
    assert set(c._sdk_tools) == {t["name"] for t in _LOCAL_TOOLS}
    salida = asyncio.run(c._sdk_tools["estado_bot"].handler({"fecha": "hoy"}))
    assert ejecutado["llamada"] == ("estado_bot", {"fecha": "hoy"})
    assert "OK estado_bot" in str(salida)


def test_think_vacio_no_lanza_cli():
    c, _ = _cerebro()
    assert c.think("") == ""
    assert c.think("   ") == ""
