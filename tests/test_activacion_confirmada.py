"""Activación confirmada (palmada + palabra clave), cortes mudos y utilidades nuevas."""
from __future__ import annotations

import numpy as np

import main
from audio.tts import _split_oraciones
from states import State


class _FakeTTS:
    def __init__(self):
        self.dicho = []

    def speak(self, texto):
        self.dicho.append(texto)


class _FakeSTT:
    """Secuencia de (texto, motivo); registra si le pidieron escucha corta (kwargs)."""

    def __init__(self, secuencia):
        self.secuencia = list(secuencia)
        self.i = 0
        self.kwargs_vistos = []

    def listen(self, **kw):
        self.kwargs_vistos.append(kw)
        r = self.secuencia[self.i] if self.i < len(self.secuencia) else ("", "timeout_sin_voz")
        self.i += 1
        return r


class _FakeBrain:
    def __init__(self):
        self.preguntas = []

    def reset(self):
        pass

    def think(self, texto):
        self.preguntas.append(texto)
        return "ok"


def _orq(stt, origen="palmada"):
    orch = main.build()
    orch.tts = _FakeTTS()
    orch.brain = _FakeBrain()
    orch.stt = stt
    orch.hud = None
    orch.detector = type("D", (), {"origen": lambda self: origen})()
    return orch


def test_palmada_sin_palabra_clave_se_ignora_en_silencio():
    # Se oyó algo, pero ni palabra clave NI petición (charla ambiental).
    stt = _FakeSTT([("eso estuvo bueno", "ok")])
    orch = _orq(stt)
    orch._on_wake()
    assert orch.state is State.IDLE
    assert orch.activaciones == 0          # ni cuenta como activación
    assert orch.tts.dicho == []            # y NO habla (un click de mouse no molesta)
    assert orch.brain.preguntas == []
    assert stt.kwargs_vistos[0] == {"max_total_ms": 8000, "sin_voz_timeout_ms": 3000}


def test_palmada_mas_jarvis_abre_conversacion():
    stt = _FakeSTT([("Jarvis", "ok"), ("qué hora es", "ok"), ("", "timeout_sin_voz")])
    orch = _orq(stt)
    orch._on_wake()
    assert orch.activaciones == 1
    assert orch.tts.dicho[0] == "¿Sí, señor?"
    assert orch.brain.preguntas == ["qué hora es"]


def test_palmada_con_pregunta_de_corrido_responde_directo():
    """'Jarvis, ¿qué hora es?' tras la palmada: sin '¿Sí, señor?', directo al cerebro."""
    stt = _FakeSTT([("Jarvis, ¿qué hora es?", "ok"), ("", "timeout_sin_voz")])
    orch = _orq(stt)
    orch._on_wake()
    assert orch.activaciones == 1
    assert orch.brain.preguntas == ["que hora es"]  # normalizado, sin la palabra clave
    assert orch.tts.dicho == ["ok"]  # la respuesta, SIN saludo previo


def test_quitar_palabra_clave():
    q = main.Orchestrator._quitar_palabra_clave
    assert q("Jarvis, dime las noticias", "jarvis") == "dime las noticias"
    assert q("oye yarvis pon música", "jarvis") == "pon musica"
    assert q("jarvis", "jarvis") == ""          # palabra sola -> saludo normal
    assert q("dime la hora jarvis", "jarvis") == ""  # nada útil tras la palabra
    assert q("hola buenas", "jarvis") == ""


def test_attack_max_llega_al_core():
    from audio.clap_core import ClapCore
    core = ClapCore.from_config({"audio": {"sample_rate": 16000,
                                           "palmada": {"attack_max": 45.0}}})
    assert core.attack_max == 45.0
    assert ClapCore.from_config({"audio": {"sample_rate": 16000}}).attack_max == 0.0


def test_variantes_stt_de_jarvis_cuentan():
    # Casos REALES del log del 11-ago: whisper escribió '¡Yorvis!' y 'Yervis'.
    for oido in ("yarvis", "oye jarbis", "HARVIS ven", "¡Yorvis!", "Yervis", "iorvis dime"):
        assert main.Orchestrator._contiene_palabra_clave(oido, "jarvis"), oido
    assert not main.Orchestrator._contiene_palabra_clave("hola buenas", "jarvis")


def test_palabra_clave_tarde_en_la_frase_no_activa():
    """Mencionar a Jarvis a mitad de conversación (dictado capturado por un armado
    fantasma) NO debe abrir nada — caso real: 'Hice una prueba rápida con Yervis'."""
    assert not main.Orchestrator._contiene_palabra_clave(
        "Hice una prueba rápida con Yervis", "jarvis")
    assert main.Orchestrator._quitar_palabra_clave(
        "Yorvis, dime las noticias de mañana", "jarvis") == "dime las noticias de manana"


def test_hotkey_y_wake_word_no_piden_confirmacion():
    for origen in ("externo", "voz"):
        stt = _FakeSTT([("qué hora es", "ok"), ("", "timeout_sin_voz")])
        orch = _orq(stt, origen=origen)
        orch._on_wake()
        assert orch.activaciones == 1
        assert orch.brain.preguntas == ["qué hora es"]


def test_corte_cancelar_cierra_mudo():
    stt = _FakeSTT([("jarvis", "ok"), ("qué hora es", "ok"), ("cancelar", "ok")])
    orch = _orq(stt)
    orch._on_wake()
    assert orch.state is State.IDLE
    # habló el saludo y la respuesta, pero NINGUNA despedida tras el corte
    assert orch.tts.dicho == ["¿Sí, señor?", "ok"]


def test_silencio_tambien_corta_mudo():
    stt = _FakeSTT([("jarvis", "ok"), ("silencio", "ok")])
    orch = _orq(stt)
    orch._on_wake()
    assert orch.tts.dicho == ["¿Sí, señor?"]  # nada después: obedeció callado


def test_split_oraciones_agrupa_cortas():
    partes = _split_oraciones("Sí, señor. La sesión cerró con ganancias de 425 dólares. "
                              "El filtro cero bloqueó 349 señales; nada más que reportar.")
    assert len(partes) == 3
    assert partes[0].startswith("Sí, señor. La sesión")  # el trozo corto se pegó al siguiente


def test_origen_del_detector():
    from audio.clap_detector import ClapDetector
    d = ClapDetector({"audio": {"sample_rate": 16000}, "logs": {"nivel": "INFO"}})
    d.trigger_externo()
    assert d.origen() == "externo"
    d._origen = ""
    ev = object()
    d.core.push = lambda b: ev  # stub: el comité "acepta" el bloque
    d._callback(np.zeros((160, 1), dtype=np.float32), 160, None, None)
    assert d.origen() == "palmada"
    assert d._detected.is_set()


def test_instancia_unica_mismo_proceso():
    """La 1a adquisicion pasa; una 2a (simula otro Jarvis) ve el mutex tomado. Nombre
    propio del test: el mutex REAL suele estar tomado por el Jarvis vivo de la maquina."""
    import uuid
    nombre = f"Local\\JarvisTest{uuid.uuid4().hex[:8]}"
    assert main._instancia_unica(nombre) is True
    assert main._instancia_unica(nombre) is False


def test_corte_y_despedida_como_ultima_palabra():
    """Caso real 18:04: 'Tiene una idea, falta mucho. Cancelar.' debe CORTAR aunque la
    frase sea larga; ídem 'ok creo que ya terminamos' como despedida."""
    o = _orq(_FakeSTT([]))
    assert o._es_corte("Tiene una idea, falta mucho. Cancelar.")
    assert o._es_corte("mejor déjalo, silencio")
    assert not o._es_corte("no canceles nada de lo que te dije antes porque es importante")
    assert o._es_despedida("ok creo que ya terminamos")
    assert o._es_despedida("con eso quedamos bien, terminamos")
    assert not o._es_despedida("cuando terminamos la sesión de ayer el bot seguía corriendo")


def test_palmada_con_peticion_directa_abre_sin_palabra_clave():
    """Caso real 18:16: palmada -> 'Dime las noticias de hoy.' (sin decir jarvis) DEBE abrir."""
    stt = _FakeSTT([("Dime las noticias de hoy.", "ok"), ("", "timeout_sin_voz")])
    orch = _orq(stt)
    orch._on_wake()
    assert orch.activaciones == 1
    assert orch.brain.preguntas == ["Dime las noticias de hoy."]
    assert orch.tts.dicho == ["ok"]  # directo, sin saludo


def test_charla_ambiental_sigue_ignorada():
    """Casos reales: 'Está muy bien.' / 'Ya lo vi.' capturados por armados fantasma."""
    for frase in ("Está muy bien.", "Ya lo vi.", "no sé si sea posible"):
        orch = _orq(_FakeSTT([(frase, "ok")]))
        orch._on_wake()
        assert orch.activaciones == 0, frase
        assert orch.tts.dicho == [], frase


def test_gracias_cierra_con_despedida_hablada():
    stt = _FakeSTT([("jarvis", "ok"), ("qué hora es", "ok"), ("muchas gracias", "ok")])
    orch = _orq(stt)
    orch._on_wake()
    assert orch.state is State.IDLE
    assert orch.tts.dicho[-1] == "A su disposición, señor."  # a un gracias se responde
