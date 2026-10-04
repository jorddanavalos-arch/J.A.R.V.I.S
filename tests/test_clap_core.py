"""Fase 1 — tests del detector de doble palmada con señales sintéticas (sin micrófono)."""
from __future__ import annotations

import numpy as np
import pytest

from audio.clap_core import ClapCore, ClapEvent, spectral_features

SR = 16000


# --------------------------------------------------------------------------- helpers
def clap_kernel(n: int = 160, tau: float = 6.0, seed: int = 7) -> np.ndarray:
    """Palmada sintética: ráfaga de ruido blanco (banda ancha) con decaimiento rápido.

    Determinista (seed fijo) y con argmax estable, de modo que dos palmadas con el MISMO
    kernel cancelan el offset del pico al calcular dt.
    """
    rng = np.random.default_rng(seed)
    env = np.exp(-np.arange(n) / tau)
    k = rng.standard_normal(n) * env
    k /= np.max(np.abs(k)) + 1e-9
    return k.astype(np.float32)


CLAP = clap_kernel()


def ambient(nsamp: int, amp: float = 0.002, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(nsamp) * amp).astype(np.float32)


def place(sig: np.ndarray, pos: int, kernel: np.ndarray, amp: float = 0.6) -> None:
    end = min(pos + kernel.size, sig.size)
    sig[pos:end] += amp * kernel[: end - pos]


def signal_with_claps(nsamp, positions, amp=0.6, amb_amp=0.002, kernel=CLAP, seed=1):
    sig = ambient(nsamp, amb_amp, seed)
    for p in positions:
        place(sig, p, kernel, amp)
    return sig


def ms_to_n(ms: float) -> int:
    return round(ms / 1000.0 * SR)


def make_core(**over) -> ClapCore:
    params = dict(
        sample_rate=SR,
        umbral_factor=4.0,
        ventana_ms=[120, 700],
        cooldown_ms=1500,
        calibracion_ms=0,  # desactivada salvo en el test de calibración
    )
    params.update(over)
    return ClapCore(**params)


def feed(core: ClapCore, sig: np.ndarray, block: int = 160) -> list[ClapEvent]:
    evs = []
    sig = np.asarray(sig, dtype=np.float32)
    for i in range(0, sig.size, block):
        ev = core.push(sig[i:i + block])
        if ev is not None:
            evs.append(ev)
    return evs


# --------------------------------------------------------------------------- tests
def test_doble_palmada_dentro_de_ventana_dispara():
    p1 = ms_to_n(200)
    sig = signal_with_claps(int(1.0 * SR), [p1, p1 + ms_to_n(300)])
    evs = feed(make_core(), sig)
    assert len(evs) == 1
    assert evs[0].tipo == "doble_palmada"
    assert 120 <= evs[0].dt_ms <= 700
    assert abs(evs[0].dt_ms - 300) <= 1.0


def test_palmada_unica_no_dispara():
    sig = signal_with_claps(int(0.8 * SR), [ms_to_n(300)])
    core = make_core()
    evs = feed(core, sig)
    assert evs == []
    assert len(core.debug_onsets) <= 1


def test_limites_exactos_de_ventana():
    for dt, debe in [(119, False), (120, True), (121, True), (699, True), (700, True), (701, False)]:
        p1 = ms_to_n(200)
        sig = signal_with_claps(int(1.5 * SR), [p1, p1 + ms_to_n(dt)])
        evs = feed(make_core(), sig)
        assert (len(evs) == 1) is debe, f"dt={dt} esperado dispara={debe}, eventos={len(evs)}"


def test_rebote_y_demasiado_separadas_no_disparan():
    p1 = ms_to_n(200)
    sig_rebote = signal_with_claps(int(1.0 * SR), [p1, p1 + ms_to_n(80)])
    sig_lentas = signal_with_claps(int(2.0 * SR), [p1, p1 + ms_to_n(900)])
    assert feed(make_core(), sig_rebote) == []
    assert feed(make_core(), sig_lentas) == []


def test_umbral_adaptativo_ruido_creciente():
    # Misma palmada sobre ruido bajo (debe verse) y luego sobre ruido alto (no debe verse).
    n = int(3.0 * SR)
    sig = ambient(n, amp=0.002)
    place(sig, ms_to_n(300), CLAP, amp=0.3)  # palmada en zona de ruido bajo
    sig[ms_to_n(1200):] += ambient(n - ms_to_n(1200), amp=0.08, seed=9)  # ruido de fondo alto
    place(sig, ms_to_n(2500), CLAP, amp=0.3)  # misma palmada, ahora bajo ruido alto
    core = make_core()
    feed(core, sig)
    onsets = core.debug_onsets
    assert len(onsets) == 1
    assert abs(onsets[0] - 300) < 30  # solo se detectó la primera (ruido bajo)


def test_habla_no_dispara():
    n = int(2.0 * SR)
    rng = np.random.default_rng(3)
    base = np.convolve(rng.standard_normal(n), np.ones(40) / 40, mode="same")  # paso-bajo (~<400 Hz)
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # modulación 4 Hz tipo habla
    speech = (base * env * 0.4).astype(np.float32)
    assert feed(make_core(), speech) == []


def test_musica_tonal_no_dispara():
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.0, 330.0)) / 3.0
    music = (chord * 0.5 + ambient(n, 0.01)).astype(np.float32)
    assert feed(make_core(), music) == []


def test_tecleo_no_dispara():
    n = int(2.0 * SR)
    sig = ambient(n, amp=0.002)
    rng = np.random.default_rng(5)
    for _ in range(12):
        pos = int(rng.integers(ms_to_n(100), n - 400))
        place(sig, pos, CLAP, amp=0.015)  # clics pequeños, por debajo del umbral adaptativo
    assert feed(make_core(), sig) == []


def test_golpe_grave_no_dispara():
    # (a) la puerta espectral discrimina banda ancha de baja frecuencia
    low = np.sin(2 * np.pi * 150 * np.arange(256) / SR)
    bb = np.random.default_rng(0).standard_normal(256)
    _, hf_low = spectral_features(low, SR)
    fl_bb, hf_bb = spectral_features(bb, SR)
    assert hf_low < 0.4
    assert hf_bb >= 0.4 and fl_bb >= 0.3
    # (b) dos golpes graves impulsivos no disparan
    dur = int(0.012 * SR)
    tt = np.arange(dur) / SR
    thud = (np.sin(2 * np.pi * 120 * tt) * np.exp(-tt / 0.003)).astype(np.float32)
    thud /= np.max(np.abs(thud)) + 1e-9
    n = int(1.2 * SR)
    sig = ambient(n, 0.002)
    place(sig, ms_to_n(200), thud, amp=0.6)
    place(sig, ms_to_n(500), thud, amp=0.6)
    assert feed(make_core(), sig) == []


def test_cooldown_ignora_palmadas_extra():
    p1 = ms_to_n(200)
    positions = [p1, p1 + ms_to_n(300), p1 + ms_to_n(800), p1 + ms_to_n(1100)]
    sig = signal_with_claps(int(2.5 * SR), positions)
    evs = feed(make_core(), sig)
    assert len(evs) == 1  # las palmadas extra dentro del cooldown se ignoran


def test_refractario_palmada_con_eco():
    p1 = ms_to_n(300)
    sig = ambient(int(1.0 * SR), 0.002)
    place(sig, p1, CLAP, amp=0.6)
    place(sig, p1 + ms_to_n(50), CLAP, amp=0.25)  # eco atenuado a 50 ms (dentro del refractario)
    core = make_core()
    evs = feed(core, sig)
    assert evs == []
    assert len(core.debug_onsets) == 1


def test_determinismo_invariante_al_blocksize():
    p1 = ms_to_n(200)
    sig = signal_with_claps(int(1.5 * SR), [p1, p1 + ms_to_n(300)])
    resultados = []
    for block in (128, 160, 512, 1024):
        core = make_core()
        evs = feed(core, sig, block=block)
        resultados.append((len(evs), [round(e.dt_ms) for e in evs], [round(o) for o in core.debug_onsets]))
    assert all(r == resultados[0] for r in resultados), resultados


def test_downmix_estereo_equivale_a_mono():
    p1 = ms_to_n(200)
    mono = signal_with_claps(int(1.0 * SR), [p1, p1 + ms_to_n(300)])
    stereo = np.stack([mono, mono], axis=1)  # (N, 2)
    core_m = make_core()
    evs_m = feed(core_m, mono)
    core_s = make_core()
    # downmix como en la capa de E/S
    evs_s = feed(core_s, stereo.mean(axis=1))
    assert len(evs_m) == len(evs_s) == 1
    assert [round(o) for o in core_m.debug_onsets] == [round(o) for o in core_s.debug_onsets]


def test_from_config_usa_defaults_y_overrides():
    minimo = {"audio": {"sample_rate": SR, "palmada": {"umbral_factor": 4.0, "ventana_ms": [120, 700], "cooldown_ms": 1500}}}
    core = ClapCore.from_config(minimo)
    assert core.crest_min == 2.5 and core.refractario_n == round(SR * 90 / 1000)
    con_overrides = {"audio": {"sample_rate": SR, "palmada": {"crest_min": 9.0, "ventana_ms": [100, 600], "cooldown_ms": 1000}}}
    core2 = ClapCore.from_config(con_overrides)
    assert core2.crest_min == 9.0 and core2.win_hi_ms == 600


def test_modulo_core_importa_sin_sounddevice(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    import audio.clap_core as cc

    importlib.reload(cc)
    core = cc.ClapCore(sample_rate=SR, umbral_factor=4.0, ventana_ms=[120, 700], cooldown_ms=1500, calibracion_ms=0)
    assert core is not None


def test_io_layer_cierra_stream_en_finally(monkeypatch):
    import audio.clap_detector as cd

    p1 = ms_to_n(200)
    sig = signal_with_claps(int(1.2 * SR), [p1, p1 + ms_to_n(300)])
    calls = {"stop": 0, "close": 0}

    class FakeStream:
        def __init__(self, callback=None, blocksize=160, **kw):
            self.callback = callback
            self.blocksize = blocksize

        def start(self):
            b = self.blocksize
            for i in range(0, sig.size, b):
                self.callback(sig[i:i + b].reshape(-1, 1), b, None, None)
            raise RuntimeError("fallo simulado del stream")  # excepción tras alimentar

        def stop(self):
            calls["stop"] += 1

        def close(self):
            calls["close"] += 1

    monkeypatch.setattr(cd.sd, "InputStream", lambda **kw: FakeStream(**kw))
    det = cd.ClapDetector({"audio": {"sample_rate": SR, "palmada": {"calibracion_ms": 0}}})
    result = det.wait_for_double_clap()
    assert result is True
    assert calls["stop"] == 1 and calls["close"] == 1


def test_calibracion_inicial_suprime_disparo():
    p1 = ms_to_n(150)  # par válido DENTRO de la calibración (400 ms)
    sig_calib = signal_with_claps(int(1.0 * SR), [p1, p1 + ms_to_n(200)])
    assert feed(make_core(calibracion_ms=400), sig_calib) == []
    # el mismo par tras la calibración SÍ dispara
    p2 = ms_to_n(600)
    sig_post = signal_with_claps(int(1.5 * SR), [p2, p2 + ms_to_n(200)])
    assert len(feed(make_core(calibracion_ms=400), sig_post)) == 1


# --------------------------------------------------------------- robustez (revisión)
def test_clap_saturado_dispara():
    # Palmada fuerte cerca del micro -> satura a +-1 (clipping) -> crest colapsa.
    # El bypass por clipping debe permitir la detección igualmente.
    p1 = ms_to_n(200)
    sig = signal_with_claps(int(1.0 * SR), [p1, p1 + ms_to_n(300)], amp=4.0)
    sig = np.clip(sig, -1.0, 1.0).astype(np.float32)
    assert len(feed(make_core(), sig)) == 1


def test_nan_inf_no_rompen_la_deteccion():
    core = make_core()
    bad = np.array([np.nan] * 80 + [np.inf] * 40 + [-np.inf] * 40, dtype=np.float32)
    core.push(bad)
    assert np.isfinite(core.noise_floor)  # el piso no se envenena
    sig = signal_with_claps(int(1.5 * SR), [ms_to_n(500), ms_to_n(800)])
    assert len(feed(core, sig)) == 1  # sigue detectando tras el bloque corrupto


def test_config_invalida_lanza_valueerror():
    with pytest.raises(ValueError):
        ClapCore(sample_rate=SR, umbral_factor=4.0, ventana_ms=700, cooldown_ms=1500)
    with pytest.raises(ValueError):
        ClapCore(sample_rate=SR, umbral_factor=4.0, ventana_ms=[700, 120], cooldown_ms=1500)


def test_from_config_tipos_erroneos_caen_a_defaults():
    c1 = ClapCore.from_config({"audio": 16000})  # 'audio' no es dict
    assert c1.sr == 16000 and c1.umbral_factor == 4.0
    c2 = ClapCore.from_config({"audio": {"sample_rate": SR, "palmada": [1, 2, 3]}})  # 'palmada' no es dict
    assert c2.umbral_factor == 4.0


def test_wiring_acepta_wrapper_settings():
    import audio.clap_detector as cd

    wrapper = {
        "settings": {"audio": {"sample_rate": SR, "palmada": {"umbral_factor": 7.0, "ventana_ms": [120, 700], "cooldown_ms": 1500}}},
        "apps": {},
    }
    det = cd.ClapDetector(wrapper)
    assert det.sample_rate == SR
    assert det.core.umbral_factor == 7.0  # leyó settings reales, no defaults


def test_stop_devuelve_false(monkeypatch):
    import audio.clap_detector as cd

    holder = {}

    class CancelStream:
        def __init__(self, **kw):
            pass

        def start(self):
            holder["det"].stop()  # cancelación mientras "escucha"

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(cd.sd, "InputStream", lambda **kw: CancelStream(**kw))
    det = cd.ClapDetector({"audio": {"sample_rate": SR, "palmada": {"calibracion_ms": 0}}})
    holder["det"] = det
    assert det.wait_for_double_clap() is False  # cancelación != detección


def test_error_en_callback_devuelve_false(monkeypatch):
    import audio.clap_detector as cd

    det = cd.ClapDetector({"audio": {"sample_rate": SR, "palmada": {"calibracion_ms": 0}}})

    def boom(_block):
        raise RuntimeError("fallo de DSP simulado")

    monkeypatch.setattr(det.core, "push", boom)
    sig = signal_with_claps(int(0.4 * SR), [ms_to_n(100)])

    class S:
        def __init__(self, callback=None, blocksize=160, **kw):
            self.cb = callback
            self.bs = blocksize

        def start(self):
            for i in range(0, sig.size, self.bs):
                self.cb(sig[i:i + self.bs].reshape(-1, 1), self.bs, None, None)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(cd.sd, "InputStream", lambda **kw: S(**kw))
    assert det.wait_for_double_clap() is False  # error de DSP no se reporta como detección


def test_fallo_al_abrir_microfono_devuelve_false(monkeypatch):
    import audio.clap_detector as cd

    def boom(**kw):
        raise RuntimeError("dispositivo ocupado")

    monkeypatch.setattr(cd.sd, "InputStream", boom)
    det = cd.ClapDetector({"audio": {"sample_rate": SR, "palmada": {"calibracion_ms": 0}}})
    assert det.wait_for_double_clap() is False  # fallo de apertura no propaga
    assert det.errored() is True  # se distingue de una cancelación


def _reverb_clap(n: int = 4000, tau: float = 900.0, seed: int = 3) -> np.ndarray:
    """Palmada con COLA larga (reverberante): pico seco al inicio + decaimiento lento."""
    rng = np.random.default_rng(seed)
    env = np.exp(-np.arange(n) / tau)
    k = rng.standard_normal(n) * env
    k /= np.max(np.abs(k)) + 1e-9
    return k.astype(np.float32)


def test_palmada_reverberante_dispara():
    # Energía por encima del umbral >onset_max_ms (cola), pero pico temprano: debe ACEPTARSE.
    # (Regresión del bug too_long/dur_ok que rechazaba estas palmadas incondicionalmente.)
    rev = _reverb_clap()
    p1 = ms_to_n(200)
    sig = ambient(int(1.5 * SR), 0.002)
    place(sig, p1, rev, amp=0.6)
    place(sig, p1 + ms_to_n(350), rev, amp=0.6)
    assert len(feed(make_core(), sig)) == 1


def test_calibracion_con_pico_no_envenena_el_piso():
    # Un 'pop' fuerte DURANTE la calibración no debe inflar el piso (siembra por percentil):
    # una doble palmada válida tras la calibración SÍ se detecta.
    sig = ambient(int(2.0 * SR), 0.002)
    place(sig, ms_to_n(100), CLAP, amp=0.9)  # pop dentro de los 400 ms de calibración
    p = ms_to_n(700)
    place(sig, p, CLAP, amp=0.6)
    place(sig, p + ms_to_n(300), CLAP, amp=0.6)
    assert len(feed(make_core(calibracion_ms=400), sig)) == 1
