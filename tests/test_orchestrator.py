"""Pruebas de la Fase 0: el esqueleto arranca y queda en IDLE."""
from __future__ import annotations

import main
from states import State


def test_orchestrator_starts_idle():
    orch = main.build()
    assert orch.state is State.IDLE


def test_config_loads_with_safe_defaults():
    cfg = main.load_config()
    assert "settings" in cfg and "apps" in cfg
    # modo_seguro debe venir activo por defecto (requisito de seguridad)
    assert cfg["settings"]["modo_seguro"] is True


def test_transition_changes_state():
    orch = main.build()
    orch.transition(State.WAKE)
    assert orch.state is State.WAKE


def test_stub_modules_import_cleanly():
    # Las capas de cada fase deben importar sin requerir el stack pesado.
    import audio.clap_detector  # noqa: F401
    import audio.stt  # noqa: F401
    import audio.tts  # noqa: F401
    import brain.agent  # noqa: F401
    import tools.windows_control  # noqa: F401
    import tools.trading_readonly  # noqa: F401


def test_orchestrator_activa_en_doble_palmada():
    """Fase 1: el orquestador transiciona a WAKE cuando el detector reporta una palmada."""
    orch = main.build()

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def wait_for_double_clap(self):
            self.calls += 1
            return self.calls == 1  # detecta una vez, luego False

        def cancelled(self):
            return True  # el segundo False se trata como parada -> sale del bucle

        def stop(self):
            pass

    orch.detector = FakeDetector()
    # Fakes de voz para NO cargar modelos ni abrir el micro en este test de bucle.
    orch.stt = type("FakeSTT", (), {"listen": lambda self: ("", "timeout_sin_voz")})()
    orch.tts = type("FakeTTS", (), {"speak": lambda self, texto: None})()
    orch.hud = None  # sin HUD: run() no debe abrir puertos ni el navegador en tests
    orch.run()
    assert orch.activaciones == 1
    assert orch.state is State.IDLE  # vuelve a reposo tras activar


class _FakeTTS:
    def __init__(self):
        self.dicho = []

    def speak(self, texto):
        self.dicho.append(texto)


class _FakeSTT:
    """Devuelve una SECUENCIA de (texto, motivo); al agotarse, silencio (cierra la conversación)."""

    def __init__(self, secuencia):
        self.secuencia = list(secuencia)
        self.i = 0

    def listen(self):
        r = self.secuencia[self.i] if self.i < len(self.secuencia) else ("", "timeout_sin_voz")
        self.i += 1
        return r


class _FakeBrain:
    def __init__(self):
        self.reseteos = 0
        self.preguntas = []

    def reset(self):
        self.reseteos += 1

    def think(self, texto):
        self.preguntas.append(texto)
        return f"respuesta del cerebro a: {texto}"


def test_on_wake_responde_y_termina_en_silencio():
    """Fase 3: una palmada -> oye (STT) -> piensa (cerebro) -> responde (TTS); el silencio cierra."""
    orch = main.build()
    orch.tts = _FakeTTS()
    orch.brain = _FakeBrain()
    orch.stt = _FakeSTT([("hola jarvis", "ok"), ("", "timeout_sin_voz")])
    orch._on_wake()
    assert orch.state is State.IDLE
    assert orch.brain.reseteos == 1  # memoria limpia al empezar
    assert orch.tts.dicho[0] == "¿Sí, señor?"  # confirmación inicial única
    assert "respuesta del cerebro a: hola jarvis" in orch.tts.dicho  # respondió el cerebro


def test_on_wake_silencio_inicial_da_cortesia():
    orch = main.build()
    orch.tts = _FakeTTS()
    orch.brain = _FakeBrain()
    orch.stt = _FakeSTT([("", "timeout_sin_voz")])
    orch._on_wake()
    assert orch.brain.preguntas == []  # no se invocó al cerebro
    assert any("No le he" in d for d in orch.tts.dicho)


def test_on_wake_corta_bucle_si_no_entiende_repetido():
    """Ruido de fondo (audio sin texto) NO debe dejar el bucle escuchando para siempre."""
    orch = main.build()
    orch.tts = _FakeTTS()
    orch.brain = _FakeBrain()
    orch.stt = _FakeSTT([("", "vacio"), ("", "vacio"), ("", "vacio")])
    orch._on_wake()
    assert orch.state is State.IDLE
    assert orch.brain.preguntas == []  # nunca llamó al cerebro
    assert orch.tts.dicho.count("¿Perdone, señor?") == 1  # solo una vez antes de cerrar
    assert orch.tts.dicho[-1] == "Quedo a su disposición, señor."


def test_on_wake_conversacion_multiturno_sin_repetir_palmada():
    """Una sola palmada sostiene varios turnos; una despedida cierra sin esperar el silencio."""
    orch = main.build()
    orch.tts = _FakeTTS()
    orch.brain = _FakeBrain()
    orch.stt = _FakeSTT([
        ("dame las noticias", "ok"),
        ("sí, confírmalo", "ok"),
        ("eso es todo", "ok"),  # despedida -> cierra
    ])
    orch._on_wake()
    assert orch.state is State.IDLE
    assert orch.brain.preguntas == ["dame las noticias", "sí, confírmalo"]  # 2 turnos, la despedida no
    assert orch.tts.dicho[-1] == "A su disposición, señor."
