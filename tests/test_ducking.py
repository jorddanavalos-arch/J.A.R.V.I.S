"""Ducking de música: atenúa por sesión, restaura, y nunca revienta sin pycaw."""
from __future__ import annotations

from audio.ducking import MusicDucker


class _FakeVol:
    def __init__(self, v):
        self.v = v

    def GetMasterVolume(self):
        return self.v

    def SetMasterVolume(self, v, _ctx):
        self.v = v


class _FakeSesion:
    def __init__(self, vol):
        self.SimpleAudioVolume = vol


def _ducker(vols, **cfg):
    d = MusicDucker({"activo": True, "volumen": 0.15, **cfg})
    d._sesiones_musica = lambda: iter([_FakeSesion(v) for v in vols])
    return d


def test_baja_y_restaura():
    vol = _FakeVol(0.8)
    d = _ducker([vol])
    d.bajar()
    assert vol.v == 0.15
    d.restaurar()
    assert vol.v == 0.8
    assert d._previos == {}


def test_musica_ya_baja_no_se_toca():
    vol = _FakeVol(0.10)  # ya está por debajo del objetivo
    d = _ducker([vol])
    d.bajar()
    assert vol.v == 0.10 and d._previos == {}
    d.restaurar()  # no-op
    assert vol.v == 0.10


def test_bajar_dos_veces_no_pisa_el_volumen_original():
    vol = _FakeVol(0.9)
    d = _ducker([vol])
    d.bajar()
    d.bajar()  # segunda llamada (otra conversación encadenada): no re-captura 0.15 como "original"
    d.restaurar()
    assert vol.v == 0.9


def test_desactivado_y_sin_pycaw_no_revientan():
    d = MusicDucker({"activo": False})
    d.bajar()
    d.restaurar()
    d2 = MusicDucker({})  # _sesiones_musica real: si pycaw faltara, bajar() cae al except
    d2._sesiones_musica = lambda: (_ for _ in ()).throw(ImportError("sin pycaw"))
    d2.bajar()
    assert d2._previos == {}
