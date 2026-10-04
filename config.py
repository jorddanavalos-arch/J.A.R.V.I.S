"""Carga y validación de la configuración de Jarvis.

Los valores por defecto viven aquí y se fusionan (deep-merge) con lo que haya en
``config/settings.json``, de modo que un archivo incompleto nunca rompe el arranque.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"

# Espejo de config/settings.json. Si una clave falta en el JSON, se usa esta.
DEFAULT_SETTINGS: dict[str, Any] = {
    "idioma": "es",
    "modo_seguro": True,
    # Activación: por DOBLE PALMADA y/o por VOZ. motor "openwakeword" = frase "Hey Jarvis" (sin
    # cuenta; umbral más bajo = más sensible, p.ej. para que "Jarvis" a secas tenga opción). Para la
    # palabra exacta "Jarvis" se puede usar motor "porcupine" con "picovoice_access_key" en
    # config/secrets.json (clave gratuita de console.picovoice.ai; acepta correo personal).
    "wake": {
        "doble_palmada": True,
        "wake_word_activo": True,
        "motor": "openwakeword",
        "palabra": "jarvis",
        "umbral": 0.35,  # confianza del wake word (baja = más sensible)
        "vad": 0.3,      # gate de voz Silero (baja = más permisivo; evita falsos disparos por ruido)
        "modelo_oww": "hey_jarvis",
    },
    # Confirmación anti-falsos: la doble palmada solo ARMA; la conversación abre si en ~3s
    # se oye la palabra clave (los clicks del mouse son gemelos acústicos de la palmada y
    # disparaban a Jarvis). Atajo de teclado y wake word abren directo (son intencionales).
    "activacion": {
        "confirmar_con_palabra": True,
        "palabra": "jarvis",
        # Tras la palmada también abre una PETICIÓN directa ("dime/pon/abre/qué..."), sin
        # exigir la palabra: en el uso real nadie dice "jarvis" después de aplaudir.
        "aceptar_peticion_directa": True,
    },
    "audio": {
        "sample_rate": 0,  # 0 = samplerate NATIVO del micrófono (recomendado); >0 lo fuerza
        "blocksize": 0,  # 0 = usa hop del detector; >0 fuerza el tamaño de bloque del stream
        # Re-armado robusto del micrófono entre activaciones (Windows/MME a veces deja de
        # entregar callbacks al reabrir un stream): respiro al reabrir, watchdog que reabre si no
        # llega audio, y tope de reaperturas antes de rendirse.
        "mic_settle_ms": 300,
        "mic_watchdog_ms": 3000,
        "mic_max_reopen": 6,
        "palmada": {
            "umbral_factor": 3.0,  # la palmada debe superar 3x el piso (sube si hay falsos, baja si no oye)
            "ventana_ms": [120, 1300],  # margen entre las 2 palmadas: hasta 1.3s (antes 0.7, muy justo)
            "cooldown_ms": 1500,
            # --- parámetros avanzados (afinables con el micrófono real) ---
            "hop_ms": 10,
            "crest_min": 2.0,
            "attack_min": 2.0,
            "onset_max_ms": 120,
            "decay_ratio_max": 0.5,
            "refractario_ms": 90,
            "ema_alpha": 0.03,
            "piso_absoluto": 0.005,   # piso MÍNIMO realista del micro: evita que en silencio el umbral
                                       # caiga a casi cero y el ruido del micro dispare falsas palmadas
            "hf_cutoff_hz": 2000,
            "flatness_min": 0.2,
            "hf_ratio_min": 0.2,
            "exigir_espectro": True,  # firma espectral ON: distingue palmada de ruido/click (clave anti-falsos)
            "clip_thresh": 0.98,
            "clip_frac_min": 0.02,
            "calibracion_ms": 400,
        },
        "endpoint": {
            "hop_ms": 20,
            "calibracion_ms": 300,
            "umbral_factor": 4.0,  # más exigente: el ruido de fondo NO cuenta como voz (antes 3.0)
            "arranque_min_ms": 150,
            "silencio_fin_ms": 500,  # corta antes el fin de voz tras dejar de hablar
            "pre_roll_ms": 300,
            "sin_voz_timeout_ms": 3000,  # tras callarte, cierra la conversación antes y vuelve a las palmadas
            "max_total_ms": 8000,  # tope duro de captura más corto (antes 12000)
            "ema_alpha": 0.05,
            "floor_up_rate": 0.02,  # el piso sigue al ruido sostenido más rápido -> termina antes
            "piso_absoluto": 0.0001,
        },
    },
    "stt": {
        "modelo": "small",
        "idioma": "es",
        "compute_type": "int8",
        "sample_rate": 16000,
        "beam_size": 1,
        "download_root": "models/whisper",
        "vad": {"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        "no_speech_max": 0.6,
        "logprob_min": -1.5,
    },
    "tts": {
        "motor": "piper",
        "voz_path": "models/piper/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
        "length_scale": 0.92,  # < 1.0 = habla más rápida (1.0 normal); afínalo a gusto
        "noise_scale": 0.6,
        "noise_w_scale": 0.8,
        "volume": 1.0,
    },
    # Cerebro. backend "suscripcion" (default): CLI `claude` ya logueado vía Agent SDK — SIN
    # gastar saldo de API; modelo_suscripcion admite alias (haiku/sonnet/opus). backend "api":
    # SDK anthropic con créditos (clave en ANTHROPIC_API_KEY o config/secrets.json). No hay
    # fallback automático entre backends: cambiarlo cambia quién paga y lo decide el usuario.
    # web_search=true permite noticias/datos en vivo en ambos backends.
    "cerebro": {
        "backend": "suscripcion",
        "modelo_suscripcion": "haiku",
        "modelo": "claude-haiku-4-5-20251001",
        "max_tokens": 2048,
        "timeout_s": 60,
        "web_search": True,
        "auth": "auto",
    },
    # Música: "spotify" (navega al contexto de spotify_uri y manda reproducir) o "youtube"
    # (abre youtube_url en el navegador). spotify_uri útiles:
    #   "spotify:collection:tracks"                    -> Tus me gusta (canciones que te gustan)
    #   "spotify:playlist:37i9dQZF1EYkqdzj48dyYq"      -> DJ de Spotify (variado; requiere Premium)
    "musica": {
        "motor": "spotify",
        "spotify_uri": "spotify:collection:tracks",
        "youtube_url": "https://music.youtube.com/",
    },
    # HUD visual (esfera de archivos estilo Iron Man): página local servida por Jarvis en
    # http://127.0.0.1:<puerto> que muestra la memoria (nombres de las notas) como una
    # esfera-grafo en movimiento y reacciona al estado (reposo/escucha/pensando/hablando).
    # hud.nodos: "sistema" = léxico tech/IA + stack real (look Iron Man); "memoria" = los
    # archivos de las notas (rutas.notas) como nodos.
    "hud": {"activo": True, "puerto": 36911, "abrir_navegador": True, "nodos": "sistema"},
    # Ducking: al abrir una conversación se ATENÚA la app de música (por sesión de audio de
    # Windows, vía pycaw) y se restaura al cerrar — sin AEC es la única forma real de que el
    # micro oiga al señor con Spotify sonando. Fail-open si pycaw/las sesiones no están.
    "ducking": {"activo": True, "volumen": 0.15, "procesos": ["Spotify.exe"]},
    # Atajo de teclado GLOBAL (press-to-wake): tercer canal de activación, junto a palmadas y
    # wake word. Útil cuando suena música (el micro se ensordece y sin AEC no hay arreglo por
    # software). Un toque = una activación. Si otra app ya usa el combo, Jarvis sigue sin él.
    "hotkey": {"activo": True, "combinacion": "ctrl+alt+j"},
    # Rutas de SOLO LECTURA para las herramientas de consulta del cerebro:
    #   estado_bot = carpeta reportes/ del bot MNQ (metricas_YYYY-MM-DD.json)
    #   notas      = lista de carpetas con notas .md/.txt donde buscar_notas puede leer
    "rutas": {"nt8_export": None, "estado_bot": None, "noticias": None, "notas": []},
    "logs": {"nivel": "INFO"},
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"El archivo de configuración {path} debe ser un objeto JSON.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusiona ``override`` sobre ``base`` de forma recursiva (sin mutar los originales)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_DIR / "settings.json"
    user = _load_json(path) if path.exists() else {}
    return _deep_merge(DEFAULT_SETTINGS, user)


def load_apps(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_DIR / "apps.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Es la lista blanca de apps que Jarvis puede abrir."
        )
    return _load_json(path)


def load_config() -> dict[str, Any]:
    """Devuelve la configuración completa: ``{'settings': ..., 'apps': ...}``."""
    return {"settings": load_settings(), "apps": load_apps()}
