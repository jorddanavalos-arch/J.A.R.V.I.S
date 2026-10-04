"""Cerebro de Jarvis (Fase 3+4): API de Anthropic con HERRAMIENTAS reales.

Usa el SDK ``anthropic`` (Messages API) con tool use, de modo que Jarvis puede DE VERDAD:
  - decir la hora (herramienta local ``obtener_hora``),
  - abrir aplicaciones de la lista blanca (``abrir_aplicacion`` -> tools/windows_control),
  - poner música (``poner_musica`` -> Spotify),
  - dar noticias / información actual (herramienta de servidor ``web_search`` de Anthropic).

Memoria de conversación NATIVA: la lista ``messages`` se mantiene entre turnos (incluye los
bloques de tool_use/tool_result) y se limpia con ``reset()`` (una palmada = una conversación).

La clave de la API se lee de ``ANTHROPIC_API_KEY`` o de ``config/secrets.json``
({"anthropic_api_key": "..."}, gitignored). Sin clave, el cerebro degrada con un aviso hablado.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
from pathlib import Path

import anthropic

from tools import memoria, notas, windows_control

log = logging.getLogger("jarvis")

ROOT = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT / "config" / "secrets.json"

_MD = re.compile(r"[*_`#>]+")
_URL = re.compile(r"https?://\S+")
# Emojis y símbolos que NO deben leerse en voz ni ensuciar la consola.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoticonos y pictogramas
    "\U00002600-\U000027bf"  # símbolos misceláneos + dingbats
    "\U00002190-\U000021ff"  # flechas
    "\U00002b00-\U00002bff"  # flechas/símbolos misceláneos
    "\U0000fe00-\U0000fe0f"  # selectores de variación
    "\U00002022\U00002023\U000025aa\U000025cf\U00002026"  # viñetas y puntos suspensivos (…)
    "\U00002122\U00002139\U000024c2"  # ™ ℹ Ⓜ
    "]",
    flags=re.UNICODE,
)

SYSTEM_PROMPT = (
    "Eres JARVIS, el asistente personal de voz de Kevin (un trader), inspirado en el de Iron "
    "Man: sereno, eficiente, leal y con un punto de ingenio seco. Te diriges a él como 'señor'.\n"
    "Hablas SIEMPRE en español y tu respuesta se LEE EN VOZ ALTA: UNA frase, dos como máximo "
    "(cada frase extra son segundos de espera del señor); lenguaje natural, "
    "SIN markdown, sin listas, sin viñetas, sin asteriscos, sin emojis y sin leer URLs ni enlaces.\n"
    "Tienes HERRAMIENTAS REALES; úsalas cuando corresponda en lugar de decir que no puedes:\n"
    "- obtener_hora: fecha u hora actuales.\n"
    "- abrir_aplicacion: abrir apps del ordenador (NinjaTrader, TradingView, su bot/robot de "
    "trading, Spotify SOLO para abrirlo, el navegador, el explorador de archivos).\n"
    "- poner_musica: poner/reproducir/escuchar música (empieza a sonar). Úsala SIEMPRE que pidan "
    "poner, reproducir o escuchar música; NO digas que solo puedes abrir la app.\n"
    "- web_search: noticias del día, clima, precios u otra información ACTUAL de internet; si el "
    "señor pide una página concreta (p. ej. investing.com), ÁBRELA y saca de ahí los datos; resume "
    "lo esencial en pocas frases. PROHIBIDO citar fuentes: nada de URLs, enlaces, nombres de "
    "páginas web ni 'según tal sitio' — el señor quiere el DATO, no la referencia (solo "
    "menciona la fuente si te la pide explícitamente).\n"
    "- estado_bot: resultados de la sesión del bot de trading MNQ del señor (hoy, ayer o una "
    "fecha): trades, win rate, P&L. Úsala si pregunta qué hizo o cómo le fue al bot.\n"
    "- noticias_economicas: para 'noticias de hoy/mañana/X día' usa SIEMPRE esta herramienta "
    "(calendario de EE.UU., impacto medio/alto — el recorte que le interesa al señor). NUNCA "
    "respondas noticias económicas con búsqueda web salvo que pida detalle de un evento "
    "concreto o noticias de otro tema/país.\n"
    "- noticias_economicas: calendario económico local (CPI, FOMC...) con hora de Nueva York e "
    "impacto; PRIMERA opción para noticias/eventos programados de hoy o mañana.\n"
    "- buscar_notas: busca en las notas y la memoria personal del señor (pendientes, proyectos, "
    "sesiones) y devuelve extractos; resume lo relevante, no los leas literalmente.\n"
    "- recordar: si el señor dice 'recuerda que...', 'apunta que...' o te pide guardar un dato "
    "o preferencia, guárdalo con esta herramienta (persiste entre conversaciones y días). "
    "NUNCA guardes claves, contraseñas ni credenciales.\n"
    "Cuando ejecutes una acción, confírmala en una frase breve. Si una herramienta falla o no está "
    "disponible, dilo con franqueza en una frase; NUNCA afirmes que hiciste algo que la herramienta "
    "no confirmó. Cuando una herramienta devuelva un dato (hora, día, cifras, nombres), repítelo "
    "FIELMENTE tal cual, sin cambiarlo ni recalcularlo. Para conversación normal, responde "
    "directamente con tu conocimiento."
)

_MEMORIA_INSTRUCCION = (
    "\n\nMEMORIA PERSISTENTE del señor (hechos y preferencias guardados en conversaciones "
    "anteriores). Úsala para personalizar tus respuestas; no la recites ni la menciones "
    "salvo que el señor pregunte por ella:\n"
)


def system_prompt_con_memoria() -> str:
    """SYSTEM_PROMPT + la memoria persistente. Se compone al ABRIR cada conversación
    (no al arrancar Jarvis), para que un 'recuerda que...' exista en la siguiente."""
    texto = memoria.cargar()
    if not texto:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + _MEMORIA_INSTRUCCION + texto

# Herramientas LOCALES (las ejecuta este proceso). web_search es de servidor (la ejecuta Anthropic).
_LOCAL_TOOLS = [
    {
        "name": "obtener_hora",
        "description": "Devuelve la fecha y la hora actuales del ordenador del señor.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "abrir_aplicacion",
        "description": (
            "Abre una aplicación del ordenador del señor. Apps disponibles: NinjaTrader, "
            "TradingView, su bot/robot de trading, Spotify, el navegador, el explorador de archivos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "app": {
                    "type": "string",
                    "description": "Nombre de la app: ninjatrader, tradingview, bot, spotify, navegador, explorador.",
                }
            },
            "required": ["app"],
        },
    },
    {
        "name": "poner_musica",
        "description": (
            "Pone/reproduce música y EMPIEZA a sonar (YouTube o Spotify según la configuración). "
            "Úsala siempre que pidan poner, reproducir o escuchar música. Si piden 'el DJ' de "
            "Spotify, pasa contexto='dj'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "contexto": {
                    "type": "string",
                    "description": "'dj' = el DJ de Spotify; vacío = la lista de siempre.",
                }
            },
        },
    },
    {
        "name": "estado_bot",
        "description": (
            "Resultados de la sesión del bot de trading MNQ del señor (SOLO LECTURA): trades, "
            "ganadores/perdedores, win rate, P&L y filtros. Úsala cuando pregunte qué hizo el "
            "bot, cómo le fue, o si ganó/perdió."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fecha": {
                    "type": "string",
                    "description": "'hoy' (por defecto), 'ayer' o una fecha YYYY-MM-DD.",
                }
            },
        },
    },
    {
        "name": "noticias_economicas",
        "description": (
            "Calendario económico LOCAL del bot (CPI, PPI, FOMC...): eventos programados con "
            "hora de Nueva York e impacto. ÚSALA PRIMERO para 'noticias de mañana/hoy', "
            "'qué eventos hay', 'calendario económico' — es instantánea y fiable; la web solo "
            "como complemento."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fecha": {
                    "type": "string",
                    "description": "'hoy' (por defecto), 'mañana', 'ayer' o YYYY-MM-DD.",
                }
            },
        },
    },
    {
        "name": "recordar",
        "description": (
            "Guarda un hecho o preferencia del señor en la memoria persistente de Jarvis "
            "(sobrevive entre conversaciones y días). Úsala cuando diga 'recuerda que...', "
            "'apunta que...' o pida que no se olvide algo. PROHIBIDO guardar claves, "
            "contraseñas o credenciales."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hecho": {
                    "type": "string",
                    "description": "El hecho a recordar, en una frase corta y autocontenida.",
                }
            },
            "required": ["hecho"],
        },
    },
    {
        "name": "buscar_notas",
        "description": (
            "Busca un término en las notas y la memoria personal del señor (pendientes, "
            "proyectos, sesiones de trabajo) y devuelve extractos (SOLO LECTURA)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "termino": {"type": "string", "description": "Palabra o frase corta a buscar."}
            },
            "required": ["termino"],
        },
    },
]
_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 4}

_DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def _as_settings(config: dict) -> dict:
    if isinstance(config, dict) and isinstance(config.get("settings"), dict):
        return config["settings"]
    return config if isinstance(config, dict) else {}


def _leer_secreto() -> str | None:
    try:
        data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
        val = (data.get("anthropic_api_key") or "").strip()
        return val or None
    except Exception:
        return None


def _hora_es() -> str:
    ahora = datetime.datetime.now()
    return (
        f"Son las {ahora.hour:02d}:{ahora.minute:02d} del {_DIAS[ahora.weekday()]} "
        f"{ahora.day} de {_MESES[ahora.month - 1]} de {ahora.year}, señor."
    )


def _texto(content) -> str:
    return " ".join(b.text for b in (content or []) if getattr(b, "type", None) == "text").strip()


def _limpiar(texto: str) -> str:
    """Quita URLs, emojis y restos de markdown; normaliza espacios: el texto va directo al TTS."""
    t = _URL.sub("", texto or "")
    t = _EMOJI.sub("", t)
    t = _MD.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


class Brain:
    _client = None  # singleton de proceso (anthropic.Anthropic)

    def __init__(self, config: dict, tools: list | None = None) -> None:
        self.config = config
        self.settings = _as_settings(config)
        self.apps = config.get("apps", {}) if isinstance(config, dict) else {}
        self.musica = self.settings.get("musica", {}) if isinstance(self.settings.get("musica"), dict) else {}
        self.rutas = self.settings.get("rutas", {}) if isinstance(self.settings.get("rutas"), dict) else {}
        c = self.settings.get("cerebro", {})
        c = c if isinstance(c, dict) else {}
        self.modelo: str = c.get("modelo", "claude-haiku-4-5-20251001")
        self.max_tokens: int = int(c.get("max_tokens", 2048))
        self.timeout_s: float = float(c.get("timeout_s", 60))
        self.web_search: bool = bool(c.get("web_search", True))
        self.tools = tools or []
        self.messages: list = []  # memoria de la conversación en curso
        self._system = SYSTEM_PROMPT  # se recompone CON memoria al abrir cada conversación
        # Backend: "suscripcion" (CLI claude ya logueado, sin gastar saldo de API; default)
        # o "api" (SDK anthropic + créditos). SIN fallback silencioso de uno a otro: cambiar
        # de backend cambia quién paga, y eso lo decide Kevin en config, no un except.
        self.backend: str = str(c.get("backend", "suscripcion")).lower()
        self._suscripcion = None
        if self.backend == "suscripcion":
            try:
                from brain.suscripcion import CerebroSuscripcion
                # Se pasa el CALLABLE (no el string): el prompt se compone en _connect(),
                # es decir, al abrir cada conversación, con la memoria persistente fresca.
                self._suscripcion = CerebroSuscripcion(c, system_prompt_con_memoria,
                                                       _LOCAL_TOOLS, self._ejecutar)
            except Exception:
                log.exception("No se pudo preparar el cerebro por suscripción "
                              "(¿falta claude-agent-sdk?).")

    def reset(self) -> None:
        """Empieza una conversación nueva (una palmada = una conversación)."""
        self.messages = []
        if self._suscripcion is not None:
            self._suscripcion.reset()

    def _api_key(self) -> str | None:
        return (os.environ.get("ANTHROPIC_API_KEY") or "").strip() or _leer_secreto()

    def _get_client(self):
        if Brain._client is None:
            key = self._api_key()
            if not key:
                return None
            Brain._client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s)
        return Brain._client

    def _herramientas(self) -> list:
        tools = list(_LOCAL_TOOLS)
        if self.web_search:
            tools.append(_WEB_SEARCH_TOOL)
        return tools

    def _ejecutar(self, nombre: str, entrada: dict) -> str:
        entrada = entrada or {}
        if nombre == "obtener_hora":
            return _hora_es()
        if nombre == "abrir_aplicacion":
            _ok, msg = windows_control.abrir_app(entrada.get("app", ""), self.apps)
            return msg
        if nombre == "poner_musica":
            _ok, msg = windows_control.poner_musica(self.apps, self.musica,
                                                    entrada.get("contexto", ""))
            return msg
        if nombre == "estado_bot":
            _ok, msg = notas.estado_bot(self.rutas, entrada.get("fecha", ""))
            return msg
        if nombre == "noticias_economicas":
            _ok, msg = notas.noticias_economicas(self.rutas, entrada.get("fecha", ""))
            return msg
        if nombre == "buscar_notas":
            _ok, msg = notas.buscar_notas(self.rutas, entrada.get("termino", ""))
            return msg
        if nombre == "recordar":
            _ok, msg = memoria.recordar(entrada.get("hecho", ""))
            return msg
        return f"Herramienta desconocida: {nombre}"

    def think(self, transcript: str) -> str:
        """Procesa lo dicho por el señor y devuelve la respuesta hablada (texto plano)."""
        transcript = (transcript or "").strip()
        if not transcript:
            return ""
        if self.backend == "suscripcion":
            if self._suscripcion is None:
                return ("Mi cerebro por suscripción no está disponible, señor; revise que "
                        "claude-agent-sdk esté instalado, o cambie el backend a api.")
            return self._suscripcion.think(transcript)
        client = self._get_client()
        if client is None:
            return ("No tengo configurada la clave de la API de Anthropic, señor; "
                    "añádala y podré ayudarle de verdad.")
        if not self.messages:  # conversación nueva: releer la memoria persistente
            self._system = system_prompt_con_memoria()
        self.messages.append({"role": "user", "content": transcript})
        try:
            return self._loop(client)
        except anthropic.APIError as e:
            log.exception("Error de la API de Anthropic en el cerebro.")
            # Sin créditos NO es un problema de red: decirlo claro (si no, parece que
            # "no se pudo conectar" y se pierde tiempo depurando Spotify/la red).
            if "credit balance" in str(e).lower():
                return ("Señor, se agotaron los créditos de mi cerebro en la consola de "
                        "Anthropic; recárguelos y volveré a estar operativo.")
            return "Disculpe, señor, ha habido un problema con la conexión."
        except Exception:
            log.exception("Fallo inesperado en el cerebro.")
            return "Disculpe, señor, algo ha fallado al pensar la respuesta."

    def _loop(self, client) -> str:
        """Bucle agéntico: ejecuta herramientas locales y reanuda las de servidor (web_search)."""
        respuesta = ""
        for _ in range(6):  # tope de iteraciones (anti-bucle)
            resp = client.messages.create(
                model=self.modelo,
                max_tokens=self.max_tokens,
                system=self._system,
                messages=self.messages,
                tools=self._herramientas(),
            )
            self.messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "tool_use":
                resultados = [
                    {"type": "tool_result", "tool_use_id": b.id, "content": self._ejecutar(b.name, b.input)}
                    for b in resp.content if getattr(b, "type", None) == "tool_use"
                ]
                if resultados:
                    self.messages.append({"role": "user", "content": resultados})
                    continue
            if resp.stop_reason == "pause_turn":
                continue  # herramienta de servidor (web_search): reanudar sin mensaje nuevo
            respuesta = _texto(resp.content)
            break
        return _limpiar(respuesta) or "No estoy seguro de qué responder, señor."
