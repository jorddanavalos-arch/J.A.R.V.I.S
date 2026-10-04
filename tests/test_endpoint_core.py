"""Fase 2 — tests del endpointer de voz (núcleo puro, sin micrófono)."""
from __future__ import annotations

import numpy as np

from audio.endpoint_core import EndpointCore, EndpointEvent

SR = 16000


def make_core(**over) -> EndpointCore:
    params = dict(sample_rate=SR)
    params.update(over)
    return EndpointCore(**params)


def _noise(ms: float, amp: float, seed: int) -> np.ndarray:
    n = int(SR * ms / 1000.0)
    return (np.random.default_rng(seed).standard_normal(n) * amp).astype(np.float32)


def silence(ms: float, seed: int = 1) -> np.ndarray:
    return _noise(ms, 0.001, seed)


def voice(ms: float, seed: int = 2) -> np.ndarray:
    return _noise(ms, 0.05, seed)  # bien por encima del umbral (3 * piso ~0.003)


def feed(core: EndpointCore, sig: np.ndarray, block: int = 320) -> list[EndpointEvent]:
    evs = []
    sig = np.asarray(sig, dtype=np.float32)
    for i in range(0, sig.size, block):
        evs.extend(core.push(sig[i:i + block]))
    return evs


def tipos(evs):
    return [e.tipo for e in evs]


def test_voz_seguida_de_silencio_dispara_voz_fin():
    sig = np.concatenate([silence(500), voice(800), silence(1200)])
    evs = feed(make_core(), sig)
    assert "VOZ_INICIO" in tipos(evs)
    fin = [e for e in evs if e.tipo == "VOZ_FIN"]
    assert len(fin) == 1
    clip = fin[0].audio
    assert clip is not None and clip.size > 0
    # el clip incluye pre_roll: empieza antes del inicio de voz (~500ms) y cubre la voz
    assert clip.size >= int(SR * 0.8)  # al menos la duración de la voz


def test_silencio_total_dispara_timeout():
    sig = silence(7000)
    evs = feed(make_core(), sig)
    assert "VOZ_INICIO" not in tipos(evs)
    assert "TIMEOUT_SIN_VOZ" in tipos(evs)


def _voz_fluctuante(total_ms: float, burst_ms: float = 300, dip_ms: float = 150) -> np.ndarray:
    """Voz REAL continua: ráfagas fuertes separadas por micro-pausas (< silencio_fin).

    Las micro-pausas resetean el piso adaptativo (no se 'autocura' como el ruido
    estacionario), así que nunca llega un VOZ_FIN y debe saltar la red TOPE_MAXIMO.
    """
    trozos: list[np.ndarray] = []
    t, k = 0.0, 10
    while t < total_ms:
        trozos.append(voice(burst_ms, seed=k))
        trozos.append(_noise(dip_ms, 0.002, seed=k + 500))
        t += burst_ms + dip_ms
        k += 1
    return np.concatenate(trozos)


def test_habla_fluctuante_larga_dispara_tope_maximo():
    sig = np.concatenate([silence(300), _voz_fluctuante(14000)])
    evs = feed(make_core(), sig)
    assert "VOZ_INICIO" in tipos(evs)
    assert "TOPE_MAXIMO" in tipos(evs)
    assert "VOZ_FIN" not in tipos(evs)


def test_ruido_estacionario_tras_inicio_no_se_cuelga():
    """Regresión del bug del piso CONGELADO en EN_VOZ.

    Antes: con ruido de fondo continuo tras el inicio de voz, el piso quedaba congelado y
    el endpointer capturaba hasta el tope (~15s) -> Whisper alucinaba. Ahora el piso se
    adapta DENTRO de EN_VOZ (min-follower), el ruido estacionario deja de contar como voz
    y dispara VOZ_FIN bastante antes de la red anti-cuelgue.
    """
    sig = np.concatenate([silence(300), voice(300, seed=7), _noise(11000, 0.02, seed=8)])
    evs = feed(make_core(), sig)
    assert "VOZ_INICIO" in tipos(evs)
    fin = [e for e in evs if e.tipo == "VOZ_FIN"]
    assert len(fin) == 1, tipos(evs)
    assert "TOPE_MAXIMO" not in tipos(evs)
    assert fin[0].ms < 12000  # terminó antes del max_total (la red anti-cuelgue)


def test_pausa_corta_no_corta():
    sig = np.concatenate([silence(500), voice(500, seed=3), silence(400), voice(500, seed=4), silence(1000)])
    evs = feed(make_core(), sig)
    assert tipos(evs).count("VOZ_INICIO") == 1
    assert tipos(evs).count("VOZ_FIN") == 1  # solo corta en el silencio largo final


def test_invariante_al_blocksize():
    sig = np.concatenate([silence(500), voice(700), silence(1100)])
    base = None
    for block in (320, 640, 137, 1024):
        evs = feed(make_core(), sig, block=block)
        firma = [(e.tipo, round(e.ms)) for e in evs]
        if base is None:
            base = firma
        assert firma == base, (block, firma, base)


def test_reset_limpio():
    core = make_core()
    sig = np.concatenate([silence(500), voice(700), silence(1100)])
    evs1 = feed(core, sig)
    assert "VOZ_FIN" in tipos(evs1)
    core.reset()
    evs2 = feed(core, sig)
    assert "VOZ_FIN" in tipos(evs2)
