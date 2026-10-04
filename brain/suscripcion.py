"""Cerebro por SUSCRIPCIÓN de Claude (Agent SDK sobre el CLI `claude` ya logueado).

Sin costo por token de API: usa la sesión del CLI (la suscripción de Kevin). Las herramientas
de Jarvis se exponen como un MCP server IN-PROCESS (create_sdk_mcp_server): el modelo las
invoca y las ejecuta ESTE proceso con el MISMO dispatch de siempre (Brain._ejecutar), así los
dos backends comparten una sola implementación de herramientas.

Seguridad — lección del 17-jun con el CLI crudo (--dangerously-skip-permissions re-habilitaba
TODAS las herramientas): aquí es al revés, FAIL-CLOSED en 3 capas:
  1. disallowed_tools: las built-in de sistema (Bash/Read/Write/Edit/...) NO existen.
  2. allowed_tools: SOLO mcp__jarvis__* (+ WebSearch para noticias) quedan pre-aprobadas.
  3. can_use_tool: callback que DENIEGA por código cualquier otra herramienta.
Además: setting_sources=[] (no carga CLAUDE.md/skills/hooks del usuario), strict_mcp_config
(solo nuestro MCP server) y env del proceso hijo SIN ANTHROPIC_API_KEY (que no cobre a la
API por accidente) ni variables de sesión de Claude Code.

Velocidad: el cliente (y su proceso CLI) queda CONECTADO y VIVO entre turnos en un event
loop propio (hilo daemon "jarvis-brain") — cada pregunta ya no paga el arranque del CLI
(~2-4s menos por turno). prewarm() conecta al arrancar Jarvis para que la primera pregunta
del día tampoco lo pague. La memoria de conversación la lleva el propio cliente conectado;
reset() lo desconecta (la siguiente conversación conecta uno fresco).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading

log = logging.getLogger("jarvis")

# Built-ins del CLI que NUNCA deben existir para el Jarvis de voz (capa 1). Nombres de más
# no rompen nada: disallowed acepta herramientas inexistentes.
_DISALLOWED = [
    "Bash", "BashOutput", "KillBash", "KillShell", "Read", "Write", "Edit", "MultiEdit",
    "NotebookEdit", "Glob", "Grep", "Task", "Agent", "TodoWrite", "TaskCreate",
    "TaskUpdate", "SlashCommand", "Skill", "ExitPlanMode", "EnterPlanMode", "ListMcpResources",
    "ReadMcpResource",
]
# WebFetch (leer una página) SÍ se permite junto a WebSearch: sin ella el cerebro encuentra
# resultados pero no puede ABRIR la página pedida ("busca en investing.com" fallaba con
# "visite usted la página"). Es solo lectura de web: no toca disco ni shell.
_SERVIDOR = "jarvis"
_URL = re.compile(r"https?://\S+")


def _limpiar(texto: str) -> str:
    """Import tardío para evitar el ciclo agent<->suscripcion."""
    from brain.agent import _limpiar as f
    return f(texto)


class CerebroSuscripcion:
    """think()/reset() con la suscripción de Claude. ``ejecutor`` es Brain._ejecutar:
    (nombre, entrada) -> str, el mismo dispatch de herramientas del backend API."""

    def __init__(self, config_cerebro: dict, system_prompt, tools_spec: list[dict],
                 ejecutor) -> None:
        # system_prompt: str fijo o CALLABLE () -> str. El callable se evalúa en _connect(),
        # o sea al abrir cada conversación: así la memoria persistente entra fresca (un
        # "recuerda que..." de hoy existe en la conversación siguiente sin reiniciar Jarvis).
        import warnings

        from claude_agent_sdk import (  # import aquí: si falta el paquete, Brain degrada
            CanUseToolShadowedWarning,
            ClaudeAgentOptions,
            PermissionResultAllow,
            PermissionResultDeny,
            create_sdk_mcp_server,
            tool,
        )
        # Warning ESPERADO por diseño: nuestras allowed_tools van pre-aprobadas (no consultan
        # el callback) y can_use_tool existe para denegar TODO lo demás. Sin silenciarlo,
        # saldría por stderr en cada turno de conversación.
        warnings.filterwarnings("ignore", category=CanUseToolShadowedWarning)
        c = config_cerebro if isinstance(config_cerebro, dict) else {}
        self.modelo: str = c.get("modelo_suscripcion", "haiku")
        self.timeout_s: float = float(c.get("timeout_s", 60)) + 30  # + margen de arranque CLI
        self.web_search: bool = bool(c.get("web_search", True))
        self._ejecutor = ejecutor
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None  # ClaudeSDKClient CONECTADO (una conversación); vive entre turnos

        def _hacer_handler(nombre: str):
            async def handler(args: dict) -> dict:
                try:
                    salida = self._ejecutor(nombre, args or {})
                except Exception:
                    log.exception("Fallo en la herramienta '%s' (suscripción).", nombre)
                    salida = f"La herramienta {nombre} falló, señor."
                return {"content": [{"type": "text", "text": str(salida)}]}
            return handler

        sdk_tools = [
            tool(spec["name"], spec["description"], spec.get("input_schema", {"type": "object"}))
            (_hacer_handler(spec["name"]))
            for spec in tools_spec
        ]
        self._sdk_tools = {t.name: t for t in sdk_tools}  # accesible para tests/diagnóstico
        permitidas = [f"mcp__{_SERVIDOR}__{spec['name']}" for spec in tools_spec]
        if self.web_search:
            permitidas.extend(["WebSearch", "WebFetch"])
        self._permitidas = set(permitidas)

        async def _puede_usar(tool_name: str, _input: dict, _ctx) -> object:
            if tool_name in self._permitidas:
                return PermissionResultAllow()
            log.warning("Herramienta DENEGADA al cerebro por suscripción: %s", tool_name)
            return PermissionResultDeny(message="Herramienta no disponible para Jarvis.")

        # env del proceso hijo: sin credenciales de API (usar la SUSCRIPCIÓN, no saldo) y sin
        # las variables de una sesión Claude Code que pudiera envolvernos.
        #
        # OJO (14-ago): el SDK MERGEA -> {**os.environ, **options.env}. OMITIR una clave del
        # dict NO la borra del hijo: el filtro anterior era un no-op y el CLI heredaba la
        # ANTHROPIC_API_KEY de usuario, así que cobraba a la API de consola ("Credit balance
        # is too low" en cada turno, con la suscripción marcando 0% de uso). La única forma
        # de anularlas es ponerlas VACÍAS: el CLI trata "" como ausente y cae a la sesión
        # OAuth (verificado con ClaudeSDKClient real).
        # CLAUDE_CODE_ENTRYPOINT lo fija el propio SDK ("sdk-py"): no lo pisamos.
        env = {k: "" for k in os.environ
               if k.startswith(("CLAUDE", "ANTHROPIC_")) and k != "CLAUDE_CODE_ENTRYPOINT"}
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_AUTH_TOKEN"] = ""

        self._system_prompt = system_prompt
        self._options_base = dict(
            mcp_servers={_SERVIDOR: create_sdk_mcp_server(name=_SERVIDOR, tools=sdk_tools)},
            strict_mcp_config=True,
            setting_sources=[],
            allowed_tools=permitidas,
            disallowed_tools=list(_DISALLOWED),
            # OJO: NO usar "bypassPermissions": auto-aprueba todo ANTES del callback (el SDK
            # lo avisa con CanUseToolShadowedWarning) → una built-in nueva no listada pasaría
            # sola = fail-open. En "default", allowed pasa directo y TODO lo demás consulta
            # can_use_tool, que deniega por código: fail-closed de verdad.
            permission_mode="default",
            can_use_tool=_puede_usar,
            model=self.modelo,
            max_turns=12,  # buscar + abrir página + responder puede tomar varias vueltas
            env=env,
        )
        self._ClaudeAgentOptions = ClaudeAgentOptions

    # ---- event loop propio (hilo daemon): el cliente async vive entre turnos ----

    def _ensure_loop(self) -> None:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._loop.run_forever, name="jarvis-brain",
                             daemon=True).start()

    def _run(self, coro, timeout: float):
        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self):
        from claude_agent_sdk import ClaudeSDKClient

        if self._client is None:
            sp = self._system_prompt
            opts = dict(self._options_base,
                        system_prompt=sp() if callable(sp) else sp)
            cliente = ClaudeSDKClient(options=self._ClaudeAgentOptions(**opts))
            await cliente.connect()
            self._client = cliente
        return self._client

    async def _drop_client(self) -> None:
        cliente, self._client = self._client, None
        if cliente is not None:
            try:
                await cliente.disconnect()
            except Exception:
                log.debug("Fallo desconectando el cliente de suscripción.", exc_info=True)

    def prewarm(self) -> None:
        """Conecta el cliente al ARRANCAR Jarvis: la primera pregunta no paga el spawn del CLI."""
        try:
            self._run(self._connect(), 30)
            log.info("Cerebro por suscripción pre-conectado.")
        except Exception:
            log.warning("No se pudo pre-conectar el cerebro (se conectará al primer uso).",
                        exc_info=True)

    def reset(self) -> None:
        """Conversación nueva: suelta el cliente (el siguiente turno conecta uno fresco)."""
        if self._loop is not None and self._client is not None:
            try:
                self._run(self._drop_client(), 10)
            except Exception:
                log.debug("Fallo en reset del cerebro por suscripción.", exc_info=True)
        self._client = None

    def think(self, transcript: str) -> str:
        transcript = (transcript or "").strip()
        if not transcript:
            return ""
        try:
            texto = self._run(self._turno(transcript), self.timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            log.warning("El cerebro por suscripción excedió %.0fs.", self.timeout_s)
            self._descartar_cliente_roto()
            return "Disculpe, señor, me he tardado demasiado pensando; intentémoslo de nuevo."
        except Exception as e:
            log.exception("Fallo del cerebro por suscripción.")
            self._descartar_cliente_roto()  # el próximo turno reconecta limpio
            if "CLINotFound" in type(e).__name__:
                return ("No encuentro el programa de Claude en este equipo, señor; "
                        "revise la instalación del CLI.")
            return "Disculpe, señor, algo ha fallado al pensar la respuesta."
        return _limpiar(texto) or "No estoy seguro de qué responder, señor."

    def _descartar_cliente_roto(self) -> None:
        if self._loop is not None and self._client is not None:
            try:
                self._run(self._drop_client(), 5)
            except Exception:
                self._client = None

    async def _turno(self, transcript: str) -> str:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        cliente = await self._connect()
        partes: list[str] = []
        await cliente.query(transcript)
        async for msg in cliente.receive_response():
            if isinstance(msg, AssistantMessage):
                for bloque in msg.content:
                    if isinstance(bloque, TextBlock):
                        partes.append(bloque.text)
            elif isinstance(msg, ResultMessage):
                if getattr(msg, "is_error", False):
                    log.warning("ResultMessage con error: %s", getattr(msg, "result", ""))
        return " ".join(p.strip() for p in partes if p and p.strip())
