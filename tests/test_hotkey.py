"""Atajo de teclado global (press-to-wake): parser, registro Win32 real y trigger externo."""
from __future__ import annotations

import ctypes
import threading

import pytest

import hotkey as hk
from audio.clap_detector import ClapDetector

# Config que NO toca sounddevice: sample_rate forzado evita query_devices y sin wake word.
_SETTINGS = {"audio": {"sample_rate": 16000}, "logs": {"nivel": "INFO"}}


# ---------- parser ----------

def test_parse_ctrl_alt_letra():
    mods, vk = hk._parse_combo("ctrl+alt+j")
    assert mods == hk.MOD_CONTROL | hk.MOD_ALT
    assert vk == ord("J")


def test_parse_funcion_y_espanol():
    mods, vk = hk._parse_combo("mayús+f5")
    assert mods == hk.MOD_SHIFT
    assert vk == 0x70 + 4  # VK_F5


def test_parse_espacio_y_digito():
    assert hk._parse_combo("win+espacio")[1] == 0x20
    assert hk._parse_combo("ctrl+2")[1] == ord("2")


@pytest.mark.parametrize("combo", ["j", "", "foo+j", "ctrl+esc-rara", "ctrl+f99"])
def test_parse_invalidos(combo):
    with pytest.raises(ValueError):
        hk._parse_combo(combo)


# ---------- GlobalHotkey (Win32 real, sin teclas físicas) ----------

def test_hotkey_registra_dispara_y_para():
    """Registra un combo improbable, le inyecta WM_HOTKEY por PostThreadMessage y verifica
    que on_fire corre; después stop() debe dejar el hilo muerto y desregistrado."""
    fired = threading.Event()
    g = hk.GlobalHotkey({"combinacion": "ctrl+alt+shift+f13"}, on_fire=fired.set)
    assert g.start() is True
    assert g.active
    ctypes.windll.user32.PostThreadMessageW(g._tid, hk.WM_HOTKEY, g._id, 0)
    assert fired.wait(timeout=2.0), "on_fire no corrió tras WM_HOTKEY"
    g.stop()
    assert not g._thread.is_alive()
    assert not g.active


def test_hotkey_combo_ocupado_queda_inactivo():
    """Si el combo ya está registrado (por otra instancia/app), start() falla LIMPIO."""
    a = hk.GlobalHotkey({"combinacion": "ctrl+alt+shift+f14"}, on_fire=lambda: None)
    b = hk.GlobalHotkey({"combinacion": "ctrl+alt+shift+f14"}, on_fire=lambda: None)
    try:
        assert a.start() is True
        assert b.start() is False
        assert not b.active
    finally:
        a.stop()
        b.stop()


def test_hotkey_combo_invalido_no_arranca():
    g = hk.GlobalHotkey({"combinacion": "j"}, on_fire=lambda: None)
    assert g.start() is False
    assert g._thread is None  # ni siquiera crea el hilo


def test_callback_con_error_no_mata_el_loop():
    """Un on_fire que peta no debe tumbar el hilo del atajo (sigue escuchando)."""
    llamadas = []

    def malo():
        llamadas.append(1)
        raise RuntimeError("boom")

    g = hk.GlobalHotkey({"combinacion": "ctrl+alt+shift+f15"}, on_fire=malo)
    try:
        assert g.start() is True
        ctypes.windll.user32.PostThreadMessageW(g._tid, hk.WM_HOTKEY, g._id, 0)
        ctypes.windll.user32.PostThreadMessageW(g._tid, hk.WM_HOTKEY, g._id, 0)
        deadline = threading.Event()
        for _ in range(40):  # espera activa corta a que lleguen ambas
            if len(llamadas) >= 2:
                break
            deadline.wait(0.05)
        assert len(llamadas) >= 2
        assert g._thread.is_alive()
    finally:
        g.stop()


# ---------- trigger externo en el detector ----------

def test_trigger_externo_desbloquea_deteccion():
    """El atajo equivale a una doble palmada: setea detección y despierta la espera."""
    d = ClapDetector(_SETTINGS)
    assert not d._detected.is_set()
    d.trigger_externo()
    assert d._detected.is_set()
    assert d._wake.is_set()
    assert not d.cancelled() and not d.errored()
