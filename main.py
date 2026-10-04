"""JARVIS — orquestador principal (máquina de estados).

Bucle objetivo:  IDLE -> WAKE -> LISTEN -> THINK -> SPEAK -> IDLE

Fase 0 (este esqueleto): arranca, carga configuración, prepara logging, desactiva
QuickEdit e imprime el estado IDLE. Las transiciones reales se activan fase a fase:
  - Fase 1: activación por doble palmada (IDLE -> WAKE)
  - Fase 2: voz local (LISTEN / SPEAK)
  - Fase 3: cerebro con Claude Agent SDK (THINK)
  - Fase 4: control de Windows
  - Fase 5: resúmenes de trading (solo lectura)
"""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
import webbrowser

from audio.clap_detector import ClapDetector
from audio.ducking import MusicDucker
from audio.stt import STTEngine
from audio.tts import TTSEngine
from brain.agent import Brain
from config import load_config
from console_win import disable_quickedit
from hotkey import GlobalHotkey
from hud.server import HUDServer
from logging_setup import setup_logging
from states import State


class Orchestrator:
    """Coordina el ciclo de vida de Jarvis mediante una máquina de estados."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.settings = config.get("settings", {})
        self.log = setup_logging(self.settings.get("logs", {}).get("nivel", "INFO"))
        self.state = State.IDLE
        self.activaciones = 0
        self.detector = ClapDetector(self.settings)
        self.stt = STTEngine(self.settings)  # modelos en carga perezosa (no se cargan aquí)
        self.tts = TTSEngine(self.settings)
        self.brain = Brain(self.config)  # cerebro (API Anthropic + herramientas); config incluye apps
        self.ducker = MusicDucker(self.settings.get("ducking"))
        hk = self.settings.get("hotkey", {})
        hk = hk if isinstance(hk, dict) else {}
        # Atajo global press-to-wake (activa aunque suene música); no toca Win32 hasta start().
        self.hotkey = GlobalHotkey(hk, on_fire=self.detector.trigger_externo) if hk.get("activo", True) else None
        hud_cfg = self.settings.get("hud", {})
        hud_cfg = hud_cfg if isinstance(hud_cfg, dict) else {}
        # HUD visual (esfera de archivos); no abre el puerto hasta start() en run().
        self.hud = HUDServer(hud_cfg, self.settings.get("rutas")) if hud_cfg.get("activo", True) else None

    def transition(self, new_state: State) -> None:
        """Cambia de estado dejando traza en el log (y publicándolo al HUD)."""
        self.log.info("Transición: %s -> %s", self.state, new_state)
        self.state = new_state
        if self.hud is not None:
            self.hud.set_estado(new_state.name, self.activaciones)

    # Frases para CERRAR la conversación por voz (sin tener que callar y esperar el timeout).
    _DESPEDIDAS = (
        "adios", "adiós", "hasta luego", "eso es todo", "es todo", "nada mas", "nada más",
        "ya es todo", "ya está bien", "ya esta bien", "gracias jarvis", "es todo por ahora",
        "para de escuchar", "ya puedes parar", "gracias, eso es todo", "terminamos",
        "ya terminamos", "hemos terminado", "termina la conversacion", "termina la conversación",
    )
    # CORTES: cierran la conversación EN SILENCIO, al instante (pedir silencio y recibir
    # una despedida hablada sería no hacer caso).
    _CORTES = (
        "cancelar", "cancela", "cancelado", "silencio", "cállate", "callate", "detente",
        "para ya", "basta", "ya basta", "olvídalo", "olvidalo", "déjalo", "dejalo",
    )

    @staticmethod
    def _ultima_palabra(t: str) -> str:
        palabras = re.findall(r"[a-záéíóúüñ]+", t)
        return palabras[-1] if palabras else ""

    def _es_despedida(self, texto: str) -> bool:
        t = texto.strip().lower()
        # La ÚLTIMA palabra manda: "ok, creo que terminamos" cierra aunque la frase sea larga.
        if self._ultima_palabra(t) in ("terminamos", "terminado", "adios", "adiós", "luego",
                                       "gracias"):
            return True
        if len(t.split()) > 6:  # solo frases cortas: evita falsos cierres en medio de una idea
            return False
        return any(f in t for f in self._DESPEDIDAS)

    def _es_corte(self, texto: str) -> bool:
        t = texto.strip().lower()
        # Ídem: "tiene una idea, falta mucho. Cancelar." DEBE cortar (caso real 11-ago 18:04
        # que se fue al cerebro por el tope de 4 palabras).
        if self._ultima_palabra(t) in ("cancelar", "cancela", "silencio", "basta", "detente",
                                       "cállate", "callate"):
            return True
        if len(t.split()) > 4:
            return False
        return any(f in t for f in self._CORTES)

    def _on_wake(self) -> None:
        """Una palmada abre UNA conversación de varios turnos (sin volver a aplaudir).

        WAKE -> [LISTEN -> THINK -> SPEAK]* -> IDLE. La conversación termina cuando el señor
        guarda silencio (timeout), dice una despedida, o hay un fallo de micro. El cerebro
        recuerda el contexto durante toda la conversación y se olvida al terminar.
        """
        origen = getattr(self.detector, "origen", lambda: "externo")()
        self.transition(State.WAKE)  # el HUD flashea: feedback de que la señal llegó
        # Ducking estilo Alexa: la música baja mientras dura la conversación (sin AEC es la
        # única forma real de que el micro oiga al señor con Spotify sonando).
        try:
            self.ducker.bajar()
        except Exception:
            self.log.debug("Ducking al abrir falló; sigo.", exc_info=True)
        act = self.settings.get("activacion", {})
        act = act if isinstance(act, dict) else {}
        pregunta_inicial = ""
        if origen == "palmada" and act.get("confirmar_con_palabra", True):
            # Anti-falsos: los golpes (mouse contra la mesa) son gemelos acústicos de la
            # palmada, así que la palmada solo ARMA. Abre conversación la palabra clave
            # (sola -> "¿Sí, señor?", o de corrido -> responde directo) O una PETICIÓN
            # directa ("dime las noticias de hoy" — en el uso real Kevin nunca dice
            # 'jarvis' tras aplaudir, visto en el log del 11-ago). La charla ambiental
            # capturada por un armado fantasma ("está muy bien") se ignora SIN hablar.
            palabra = act.get("palabra", "jarvis")
            texto_conf = self._escuchar_confirmacion()
            if self._contiene_palabra_clave(texto_conf, palabra):
                pregunta_inicial = self._quitar_palabra_clave(texto_conf, palabra)
            elif act.get("aceptar_peticion_directa", True) and self._es_peticion_directa(texto_conf):
                pregunta_inicial = texto_conf.strip()
            else:
                self.log.info("Palmada sin palabra clave ni petición (oí: %r); ignorada.", texto_conf)
                self._restaurar_musica()
                self.transition(State.IDLE)
                return
        self.activaciones += 1
        self.transition(State.WAKE)
        self.log.info("Activado (#%d, origen=%s).", self.activaciones, origen)
        self.brain.reset()  # conversación nueva: memoria limpia
        if not pregunta_inicial:
            print("[JARVIS] ¿Sí, señor? Le escucho.")
            self.tts.speak("¿Sí, señor?")  # única confirmación hablada por conversación
        turno = 0
        sin_entender = 0
        while True:
            if pregunta_inicial:  # venía DE CORRIDO con la palabra clave: directo al cerebro
                texto, motivo = pregunta_inicial, "ok"
                pregunta_inicial = ""
            else:
                try:
                    self.transition(State.LISTEN)
                    texto, motivo = self.stt.listen()
                except Exception:
                    self.log.exception("Fallo al escuchar.")
                    texto, motivo = "", "error_microfono"

            if motivo == "sin_modelo":
                self.transition(State.SPEAK)
                self.tts.speak("No tengo instalado el modelo de voz, señor.")
                break
            if motivo in ("timeout_sin_voz", "error_microfono", "cancelado"):
                # Silencio/fin: cierra la conversación (no exige decir "adiós").
                if turno == 0:
                    self.transition(State.SPEAK)
                    self.tts.speak("No le he oído, señor.")
                break
            if not texto:  # hubo audio pero no se entendió (a menudo ruido de fondo)
                sin_entender += 1
                if sin_entender >= 2:  # dos seguidas sin entender -> cierra (corta el bucle con ruido)
                    self.transition(State.SPEAK)
                    self.tts.speak("Quedo a su disposición, señor.")
                    break
                self.transition(State.SPEAK)
                self.tts.speak("¿Perdone, señor?")
                continue
            sin_entender = 0  # entendió algo: reinicia el contador
            if self._es_corte(texto):
                self.log.info("Corte por voz (%r): cierro EN SILENCIO.", texto)
                break
            if self._es_despedida(texto):
                self.transition(State.SPEAK)
                self.tts.speak("A su disposición, señor.")
                break

            self.log.info("Usted: %s", texto)
            print(f"[JARVIS] Usted: {texto}")
            if self.hud is not None:
                self.hud.set_evento("usted", texto)
            self.transition(State.THINK)
            print("[JARVIS] Pensando...")
            try:
                respuesta = self.brain.think(texto) or "No estoy seguro de qué responder, señor."
            except Exception:
                self.log.exception("Fallo al pensar.")
                respuesta = "Disculpe, señor, algo ha fallado al pensar la respuesta."
            print(f"[JARVIS] {respuesta}")
            if self.hud is not None:
                self.hud.set_evento("jarvis", respuesta)
            self.transition(State.SPEAK)
            self.tts.speak(respuesta)
            turno += 1
            time.sleep(0.25)  # breve respiro para que la sala se asiente antes de reabrir el micro

        self._restaurar_musica()
        self.transition(State.IDLE)

    def _restaurar_musica(self) -> None:
        try:
            self.ducker.restaurar()
        except Exception:
            self.log.debug("Restaurar volumen falló.", exc_info=True)

    def _escuchar_confirmacion(self) -> str:
        """Escucha tras una palmada: espera la palabra clave (sola o con la petición de
        corrido, por eso el tope es de frase completa; si en ~3s no hay voz, se rinde)."""
        self.transition(State.LISTEN)
        try:
            try:
                texto, _ = self.stt.listen(max_total_ms=8000, sin_voz_timeout_ms=3000)
            except TypeError:  # stt sin soporte de override (fakes en tests): escucha normal
                texto, _ = self.stt.listen()
        except Exception:
            self.log.exception("Fallo escuchando la confirmación.")
            return ""
        return texto or ""

    # Arranques típicos de una orden/pregunta hablada. Una frase capturada tras la palmada
    # que empiece así ES una petición a Jarvis; la charla ambiental casi nunca empieza así.
    _ARRANQUES_PETICION = (
        "dime", "dame", "pon", "abre", "busca", "cierra", "muestra", "muestrame", "muéstrame",
        "lee", "leeme", "léeme", "reproduce", "cuanto", "cuánto", "cuanta", "cuánta", "cuantos",
        "cuántos", "cuantas", "cuántas", "cual", "cuál", "cuales", "cuáles", "que", "qué",
        "como", "cómo", "cuando", "cuándo", "quien", "quién", "donde", "dónde", "necesito",
        "quiero", "haz", "hazme", "revisa", "checa", "consulta", "apaga", "enciende", "ayuda",
        "buenos", "buenas", "hola", "oye",
    )

    @classmethod
    def _es_peticion_directa(cls, texto: str) -> bool:
        palabras = (texto or "").strip().lower().lstrip("¡¿").split()
        return len(palabras) >= 2 and palabras[0].strip("¡¿!?,.") in cls._ARRANQUES_PETICION

    @classmethod
    def _quitar_palabra_clave(cls, texto: str, palabra: str) -> str:
        """Devuelve lo dicho DESPUÉS de la palabra clave ('Jarvis, dime X' -> 'dime X').
        Si solo se dijo la palabra (o algo antes de ella), devuelve ''."""
        hallado, resto = cls._buscar_palabra_clave(texto, palabra)
        return resto if hallado and len(resto) >= 3 else ""

    # Whisper escribe "Jarvis" como le suena: Yorvis, Yervis, Jarbis, Harvis... (visto en el
    # log del 11-ago: Kevin decía Jarvis y el STT ponía '¡Yorvis!' -> ignorado). Patrón
    # tolerante en vez de lista de variantes.
    _RE_JARVIS = re.compile(r"[jyhi][aeo]?r[bv]+[iey]?s")

    @classmethod
    def _buscar_palabra_clave(cls, texto: str, palabra: str):
        """Match de la palabra clave, tolerante al STT y SOLO al inicio de lo dicho (entre
        las primeras 5 palabras): mencionar a Jarvis a mitad de una frase no activa."""
        t = unicodedata.normalize("NFKD", (texto or "").lower())
        t = "".join(c for c in t if not unicodedata.combining(c))
        p = unicodedata.normalize("NFKD", (palabra or "jarvis").lower())
        p = "".join(c for c in p if not unicodedata.combining(c))
        m = cls._RE_JARVIS.search(t) if p == "jarvis" else None
        if m is None:
            i = t.find(p)
            if i < 0:
                return None, ""
            inicio, fin = i, i + len(p)
        else:
            inicio, fin = m.start(), m.end()
        if len(t[:inicio].split()) >= 5:  # la palabra llegó tarde: era conversación, no orden
            return None, ""
        return True, t[fin:].strip(" ,.;:!?¡¿")

    @classmethod
    def _contiene_palabra_clave(cls, texto: str, palabra: str) -> bool:
        hallado, _ = cls._buscar_palabra_clave(texto, palabra)
        return bool(hallado)

    def _prewarm(self) -> None:
        """Carga STT y voz TTS en el HILO PRINCIPAL, ANTES de abrir el micrófono, para que la
        primera activación no pague el coste de carga.

        Importante: NO hacerlo en un hilo en segundo plano. Cargar onnxruntime/ctranslate2 a la
        vez que PortAudio (sounddevice) abre el micro provoca un CUELGUE (las librerías nativas
        chocan al inicializarse en paralelo). En el hilo principal antes del micro tarda ~5s.
        """
        try:
            self.stt._ensure_model()
            import numpy as _np
            self.stt.transcribe_array(_np.zeros(6400, dtype=_np.float32), 16000)  # warmup JIT:
            # la PRIMERA transcripción real pagaba la compilación (~1-3s extra); así ya no.
        except Exception:
            self.log.debug("Pre-carga STT falló (se cargará al usarse).", exc_info=True)
        try:
            self.tts._ensure_voice()
        except Exception:
            self.log.debug("Pre-carga TTS falló (se cargará al usarse).", exc_info=True)
        try:
            self.detector.prewarm()  # carga la wake word (si está activa) en el hilo principal
        except Exception:
            self.log.debug("Pre-carga de la wake word falló (quedará desactivada).", exc_info=True)
        try:
            sub = getattr(self.brain, "_suscripcion", None)
            if sub is not None:
                sub.prewarm()  # conecta el CLI de Claude ya: la 1ª pregunta no paga el spawn
        except Exception:
            self.log.debug("Pre-conexión del cerebro falló (se conectará al usarse).", exc_info=True)

    def run(self) -> None:
        """Bucle de escucha (Fase 1): IDLE -> (doble palmada) -> WAKE -> IDLE.

        Bloquea esperando la doble palmada; al detectarla anuncia la activación y vuelve a
        IDLE. La escucha de voz (LISTEN/THINK/SPEAK) se conecta en fases posteriores.
        Salir con Ctrl-C.
        """
        modo = "SEGURO (simulación)" if self.settings.get("modo_seguro", True) else "ACTIVO"
        sr = getattr(self.detector, "sample_rate", "?")
        self.log.info("JARVIS iniciado. Estado: %s | modo_seguro: %s | micro=%s Hz", self.state, modo, sr)
        print(f"[JARVIS] Estado: {self.state}  |  Modo: {modo}  |  micro: {sr} Hz")
        print("[JARVIS] Preparando la voz y el oído, un momento, señor...")
        self._prewarm()  # SÍNCRONO y antes del micro (en segundo plano se colgaba con PortAudio)
        canales = ["da DOS palmadas"]
        if getattr(self.detector, "wake", None) is not None:
            canales.append("di 'Hey Jarvis'")
        if self.hotkey is not None and self.hotkey.start():
            canales.append(f"pulsa {self.hotkey.combo.upper()}")
        print(f"[JARVIS] Para activar: {' o '.join(canales)}.  (Ctrl-C para salir)")
        if self.hud is not None and self.hud.start():
            print(f"[JARVIS] HUD: {self.hud.url}")
            # JARVIS_SIN_NAVEGADOR=1 (arranque matutino): no abrir pestaña — robaría el foco
            # a la de TradingView recién preparada. El HUD queda servido por si se quiere ver.
            if self.hud.abrir_navegador and not os.environ.get("JARVIS_SIN_NAVEGADOR"):
                try:
                    webbrowser.open(self.hud.url)
                except Exception:
                    self.log.debug("No se pudo abrir el navegador para el HUD.", exc_info=True)
        fallos = 0
        try:
            while True:
                if self.detector.wait_for_double_clap():
                    self._on_wake()
                    fallos = 0
                elif self.detector.cancelled():
                    break  # parada deliberada (stop / Ctrl-C)
                else:
                    # Fallo transitorio del micro (típico tras abrir otra app de audio como Spotify).
                    # NO nos rendimos nunca: reintentamos con backoff hasta que el micro vuelva.
                    fallos += 1
                    espera = min(3.0, 0.5 * fallos)
                    if fallos <= 3 or fallos % 10 == 0:  # avisar al principio y luego cada 10 (sin spamear)
                        self.log.warning("El micrófono no respondió (intento %d); reintentando.", fallos)
                        print("[JARVIS] Recuperando el micrófono, un momento, señor...")
                    time.sleep(espera)
        except KeyboardInterrupt:
            pass
        finally:
            self._restaurar_musica()  # nunca dejar la música baja al salir
            if self.hotkey is not None:
                self.hotkey.stop()
            if self.hud is not None:
                self.hud.stop()
            self.detector.stop()
            print("\n[JARVIS] Hasta luego, señor.")


def build() -> Orchestrator:
    """Construye el orquestador a partir de la configuración en disco."""
    return Orchestrator(load_config())


def _instancia_unica(nombre: str = "Local\\JarvisVozInstancia") -> bool:
    """Candado de instancia única (mutex Win32 nombrado, vive lo que el proceso).

    Dos Jarvis a la vez se pelean el micrófono, el puerto del HUD y el atajo — pasó el
    11-ago con un doble clic al .lnk y un reinicio casi simultáneos (el kill-then-launch
    del .bat no cubre lanzamientos concurrentes)."""
    try:
        import ctypes
        ctypes.windll.kernel32.CreateMutexW(None, False, nombre)
        return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS
    except Exception:
        return True  # sin Win32 a mano: no bloquear el arranque


def _force_utf8_console() -> None:
    """Evita mojibake con acentos en la consola de Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            # errors="replace": nunca peta por un carácter raro; encoding utf-8 sin tocar el
            # code page de la consola (así NO cambia la tipografía de PowerShell/conhost).
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def main() -> int:
    _force_utf8_console()
    if not _instancia_unica():
        print("[JARVIS] Ya hay un JARVIS corriendo, señor; use REINICIAR_JARVIS.bat "
              "(o el acceso directo JARVIS) para reiniciarlo limpio.")
        return 0
    disable_quickedit()
    build().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
