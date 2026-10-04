"""HUD de Jarvis: servidor HTTP LOCAL (stdlib puro) que sirve la esfera de archivos.

La página (hud/hud.html) dibuja la memoria de Jarvis como una esfera-grafo en movimiento
(los nodos son los archivos reales de las notas) y reacciona en vivo al estado del
orquestador (IDLE/WAKE/LISTEN/THINK/SPEAK). El orquestador publica con set_estado() y
set_evento(); la página lo pollea.

Seguridad:
  - Solo 127.0.0.1 (nunca 0.0.0.0): el HUD es de esta máquina.
  - Rutas WHITELIST exactas (/, /estado, /nodos). NO es un file server: nada del disco se
    sirve salvo el propio hud.html; de las notas solo viajan NOMBRES de archivo, no contenido.
  - Solo lectura/observación: desde el HUD no se puede ordenar nada a Jarvis.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("jarvis")

_HTML_PATH = Path(__file__).resolve().parent / "hud.html"
_MAX_NODOS = 140

# Nodos en modo "sistema" (hud.nodos): léxico tech/IA + el stack REAL de esta casa (bot MNQ,
# NT8/TradingView, las piezas de Jarvis). Pedido de Kevin 11-ago: "textos más relacionados
# con tecnología e inteligencia artificial, que se vea más llamativo".
_NODOS_SISTEMA = [
    "NÚCLEO NEURAL", "ARC REACTOR", "CLAUDE CORE", "AGENT SDK", "INFERENCE ENGINE",
    "DEEP LEARNING", "TRANSFORMER", "ATTENTION HEADS", "LATENT SPACE", "EMBEDDINGS",
    "TENSOR CORE", "GRADIENT FLOW", "NEURAL MESH", "SYNAPSE GRID", "COGNITIVE STACK",
    "WHISPER STT", "PIPER TTS", "VOICE PIPELINE", "WAKE ENGINE", "CLAP DSP",
    "ENDPOINTER", "AUDIO DUCKING", "SIGNAL GATE", "NOISE FLOOR", "SPECTRUM SCAN",
    "HOTKEY GLOBAL", "MUTEX LOCK", "WATCHDOG", "HEARTBEAT", "TELEMETRY",
    "UPLINK MNQ", "NT8 SOCKET", "OIF EXECUTOR", "CDP BRIDGE 9222", "RETICULA SYNC",
    "TRADINGVIEW LINK", "MARKET FEED", "ORDERFLOW", "CVD DELTA", "FOOTPRINT MAP",
    "VWAP ENGINE", "WYCKOFF SCAN", "IMBALANCE RADAR", "LIQUIDITY SWEEP", "ORB WINDOW",
    "APEX GUARD", "SIM SHIELD", "RISK MATRIX", "KILL SWITCH", "FAIL-CLOSED",
    "SENTINEL MODE", "FIREWALL", "CRYPTO LAYER", "AUDIT TRAIL", "WHITELIST",
    "MCP SERVER", "TOOL ROUTER", "WEB SEARCH", "CONTEXT WINDOW", "TOKEN STREAM",
    "MEMORY VAULT", "KNOWLEDGE GRAPH", "VECTOR INDEX", "RECALL ENGINE", "SESSION STORE",
    "TELEGRAM LINK", "NOTIF RELAY", "SPOTIFY BUS", "MEDIA CONTROL", "HUD RENDER",
    "SPHERE MATRIX", "FIBONACCI GRID", "PARTICLE FIELD", "PHOTON TRACE", "PLASMA CORE",
    "QUANTUM NOISE", "ENTROPY POOL", "CHRONO SYNC", "EDGE COMPUTE", "LOW LATENCY",
    "PREWARM JIT", "CACHE LAYER", "PIPELINE ASYNC", "EVENT LOOP", "THREAD POOL",
    "DAEMON CORE", "BOOT SEQUENCE", "SELF-TEST", "DIAGNOSTICS", "CALIBRATION",
    "AUTONOMY LVL 9", "PROTOCOL 7", "OVERRIDE MANUAL", "SAFE MODE", "DEEP SCAN",
    "PATTERN LOCK", "ANOMALY DETECT", "FORECAST MODEL", "MONTE CARLO", "BACKTEST GRID",
    "WALK FORWARD", "SHARPE CALC", "DELTA HEDGE", "GAMMA RAY", "ION THRUSTER",
    "STARK PROTOCOL", "MARK 85", "VISION UPLINK", "FRIDAY BACKUP", "LEGACY JARVIS",
]
_PREFIJOS = ("session_", "project_", "reference_", "feedback_", "research_", "jarvis_",
             "historial_", "user_")
_MESES = "ene feb mar abr may jun jul ago sep oct nov dic".split()


def _titulo_nodo(stem: str) -> str:
    """Convierte un slug de nota en un título legible para la esfera:
    'session_2026-07-08_calibracion_zonas' -> 'Calibracion zonas · 8 jul'."""
    s = stem
    for p in _PREFIJOS:
        if s.startswith(p):
            s = s[len(p):]
            break
    fecha = ""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_?(.*)", s)
    if m:
        try:
            fecha = f" · {int(m.group(3))} {_MESES[int(m.group(2)) - 1]}"
        except (ValueError, IndexError):
            fecha = ""
        s = m.group(4) or s
    s = s.replace("_", " ").replace("-", " ").strip()
    if not s:
        return stem
    return s[:1].upper() + s[1:] + fecha


class _Handler(BaseHTTPRequestHandler):
    hud: "HUDServer"  # inyectado por subclase dinámica en HUDServer.start()

    def log_message(self, *_args) -> None:  # silencio: no ensuciar la consola de Jarvis
        pass

    def do_GET(self) -> None:  # noqa: N802 (nombre fijado por BaseHTTPRequestHandler)
        try:
            ruta = self.path.split("?", 1)[0]
            if ruta in ("/", "/index.html"):
                cuerpo, tipo = _HTML_PATH.read_bytes(), "text/html; charset=utf-8"
            elif ruta == "/estado":
                cuerpo, tipo = json.dumps(self.hud.snapshot()).encode("utf-8"), "application/json"
            elif ruta == "/nodos":
                cuerpo, tipo = json.dumps(self.hud.nodos()).encode("utf-8"), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)
        except Exception:
            log.debug("Fallo sirviendo %s al HUD.", self.path, exc_info=True)
            try:
                self.send_error(500)
            except Exception:
                pass


class _Servidor(ThreadingHTTPServer):
    # HTTPServer pone allow_reuse_address=True; en Windows eso permite un SEGUNDO bind al
    # mismo puerto SIN error (dos HUD pisándose en silencio). Con False, puerto ocupado =
    # excepción clara -> start() devuelve False y Jarvis sigue sin HUD.
    allow_reuse_address = False


class HUDServer:
    """Sirve el HUD en http://127.0.0.1:<puerto> y publica el estado del orquestador."""

    def __init__(self, cfg: dict, rutas_cfg: dict | None = None) -> None:
        cfg = cfg if isinstance(cfg, dict) else {}
        self.puerto = int(cfg.get("puerto", 36911))  # 0 = puerto efímero (tests)
        self.abrir_navegador = bool(cfg.get("abrir_navegador", True))
        self.modo_nodos = str(cfg.get("nodos", "sistema"))  # "sistema" (tech/IA) | "memoria"
        self._rutas = rutas_cfg if isinstance(rutas_cfg, dict) else {}
        self._lock = threading.Lock()
        self._estado = {"estado": "IDLE", "activaciones": 0, "quien": "", "texto": ""}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- publicación desde el orquestador (thread-safe) ----

    def set_estado(self, estado: str, activaciones: int | None = None) -> None:
        with self._lock:
            self._estado["estado"] = str(estado)
            if activaciones is not None:
                self._estado["activaciones"] = int(activaciones)

    def set_evento(self, quien: str, texto: str) -> None:
        """Última interacción visible en el HUD ('usted' o 'jarvis' + frase, acotada)."""
        with self._lock:
            self._estado["quien"] = str(quien)
            self._estado["texto"] = (texto or "")[:600]  # respuesta COMPLETA bajo la esfera

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._estado)

    def nodos(self) -> list[str]:
        """Etiquetas de la esfera. Modo "sistema": léxico tech/IA + stack real (default).
        Modo "memoria": nombres de las notas configuradas. Falla suave a lista vacía."""
        if self.modo_nodos != "memoria":
            return list(_NODOS_SISTEMA[:_MAX_NODOS])
        nombres: list[str] = []
        raices = self._rutas.get("notas") or []
        if isinstance(raices, str):
            raices = [raices]
        for raiz in raices:
            try:
                base = Path(raiz)
                if not base.is_dir():
                    continue
                for p in sorted(base.glob("*.md")):
                    nombres.append(_titulo_nodo(p.stem))
                    if len(nombres) >= _MAX_NODOS:
                        return nombres
            except Exception:
                log.debug("No se pudieron listar nodos de %s.", raiz, exc_info=True)
        return nombres

    # ---- ciclo de vida ----

    def start(self) -> bool:
        """Arranca el servidor en un hilo daemon. False (con warning) si el puerto está
        ocupado o algo falla: Jarvis sigue sin HUD, nunca se cae por esto."""
        if self._httpd is not None:
            return True
        try:
            handler = type("HUDHandler", (_Handler,), {"hud": self})
            self._httpd = _Servidor(("127.0.0.1", self.puerto), handler)
            self._httpd.daemon_threads = True
            self.puerto = int(self._httpd.server_address[1])  # resuelve puerto 0 (efímero)
        except Exception:
            log.warning("No se pudo abrir el HUD en el puerto %d (¿ocupado?).", self.puerto)
            self._httpd = None
            return False
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="jarvis-hud", daemon=True)
        self._thread.start()
        log.info("HUD disponible en %s", self.url)
        return True

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.puerto}"
