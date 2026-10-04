"""Fase 3+4 — tests del cerebro (API de Anthropic mockeada; sin red ni clave real)."""
from __future__ import annotations

from types import SimpleNamespace

import anthropic

import brain.agent as agent
from brain.agent import Brain


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(tid, name, inp):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=inp)


def _server(tid):
    return SimpleNamespace(type="server_tool_use", id=tid, name="web_search", input={})


def _resp(content, stop):
    return SimpleNamespace(content=content, stop_reason=stop)


class _FakeMessages:
    def __init__(self, cola):
        self.cola = list(cola)
        self.llamadas = []

    def create(self, **kw):
        # Congela 'messages' (la lista real sigue creciendo tras la llamada).
        self.llamadas.append({**kw, "messages": list(kw.get("messages", []))})
        return self.cola.pop(0)


class _FakeClient:
    def __init__(self, cola):
        self.messages = _FakeMessages(cola)


def _brain(cola, apps=None):
    cliente = _FakeClient(cola)
    Brain._client = cliente
    # backend "api" explícito: estos tests prueban el camino de la API mockeada.
    b = Brain({"settings": {"cerebro": {"backend": "api"}}, "apps": apps or {"apps": {}}})
    return b, cliente


def teardown_function(_):
    Brain._client = None


def test_sin_clave_avisa(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(agent, "_leer_secreto", lambda: None)
    Brain._client = None
    out = Brain({"settings": {"cerebro": {"backend": "api"}}, "apps": {}}).think("hola")
    assert "clave de la API" in out


def test_respuesta_simple_y_memoria():
    b, _ = _brain([_resp([_text("Buenas, señor.")], "end_turn")])
    out = b.think("hola")
    assert out == "Buenas, señor."
    assert b.messages[0]["role"] == "user"
    assert b.messages[-1]["role"] == "assistant"  # memoria nativa: ambos turnos guardados


def test_limpia_markdown_y_urls():
    b, _ = _brain([_resp([_text("**Hola** señor, mira https://x.com/abc ahí.")], "end_turn")])
    out = b.think("hola")
    assert out == "Hola señor, mira ahí."  # sin markdown ni URLs (van al TTS)


def test_tool_local_obtener_hora():
    cola = [
        _resp([_tool("t1", "obtener_hora", {})], "tool_use"),
        _resp([_text("Son las diez, señor.")], "end_turn"),
    ]
    b, c = _brain(cola)
    out = b.think("¿qué hora es?")
    assert out == "Son las diez, señor."
    assert len(c.messages.llamadas) == 2  # pidió herramienta y luego respondió
    tool_result = c.messages.llamadas[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"


def test_tool_abrir_app_desconocida_devuelve_mensaje():
    cola = [
        _resp([_tool("t1", "abrir_aplicacion", {"app": "photoshop"})], "tool_use"),
        _resp([_text("Eso no lo tengo, señor.")], "end_turn"),
    ]
    b, c = _brain(cola, apps={"apps": {}})
    b.think("abre photoshop")
    enviado = c.messages.llamadas[1]["messages"][-1]["content"][0]["content"]
    assert "lista de aplicaciones" in enviado  # el ejecutor devolvió el rechazo de la lista blanca


def test_web_search_se_incluye_y_pause_turn_reanuda():
    cola = [
        _resp([_server("s1")], "pause_turn"),  # herramienta de servidor en curso
        _resp([_text("Hoy las noticias son tranquilas, señor.")], "end_turn"),
    ]
    b, c = _brain(cola)
    out = b.think("dame las noticias de hoy")
    assert "noticias" in out.lower()
    assert len(c.messages.llamadas) == 2
    # en pause_turn NO se añade un tool_result de usuario: el último mensaje del 2º create es assistant
    assert c.messages.llamadas[1]["messages"][-1]["role"] == "assistant"
    assert any(t.get("type") == "web_search_20250305" for t in c.messages.llamadas[0]["tools"])


def test_reset_olvida_la_conversacion():
    b, _ = _brain([_resp([_text("a")], "end_turn")])
    b.think("uno")
    assert b.messages
    b.reset()
    assert b.messages == []


class _ErrorCreditos(anthropic.APIError):
    """APIError mínima con el texto real del 400 de saldo (sin construir response)."""
    def __init__(self, msg="Error code: 400 - Your credit balance is too low to access the Anthropic API."):
        Exception.__init__(self, msg)


def test_sin_creditos_lo_dice_claro(monkeypatch):
    """Quedarse sin saldo NO debe sonar a 'problema de conexión' (13:33 del 10-ago: parecía
    que Spotify fallaba y era la API sin créditos)."""
    b, _ = _brain([])
    monkeypatch.setattr(Brain, "_loop", lambda self, c: (_ for _ in ()).throw(_ErrorCreditos()))
    out = b.think("pon música")
    assert "créditos" in out
    assert "conexión" not in out


def test_apierror_generico_sigue_hablando_de_conexion(monkeypatch):
    b, _ = _brain([])
    monkeypatch.setattr(
        Brain, "_loop",
        lambda self, c: (_ for _ in ()).throw(_ErrorCreditos("Error code: 529 - overloaded")),
    )
    out = b.think("hola")
    assert "conexión" in out
