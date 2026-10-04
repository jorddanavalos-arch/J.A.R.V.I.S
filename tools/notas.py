"""Consultas de SOLO LECTURA para el cerebro (Fase 5): estado del bot MNQ y notas personales.

Guardarraíles:
  - Solo lectura absoluta: aquí no se escribe, borra ni ejecuta nada.
  - Solo dentro de las carpetas configuradas (config/settings.json -> rutas.estado_bot y
    rutas.notas); nada fuera de ellas.
  - Solo texto plano (.md/.txt/.json/.log). NUNCA HTML (regla de la casa: Jarvis no lee HTML).
  - Salidas acotadas: el resultado va al modelo y de ahí a VOZ (frases, no vertidos de archivo).

Devuelve (ok, mensaje en español), como tools/windows_control.
"""
from __future__ import annotations

import datetime
import json
import logging
import time
import unicodedata
from pathlib import Path

log = logging.getLogger("jarvis")

_EXT_OK = {".md", ".txt", ".json", ".log"}  # texto plano; .html queda fuera A PROPÓSITO
_MAX_FILE_BYTES = 512 * 1024   # una nota mayor no es una nota: se salta
_MAX_ARCHIVOS = 400            # tope de archivos escaneados por búsqueda
_MAX_HITS = 8                  # archivos con resultado que se reportan
_MAX_EXTRACTOS = 3             # extractos por archivo
_MAX_SALIDA = 1800             # tope de caracteres del resultado (va a un modelo, no a un humano)


def _norm(s: str) -> str:
    """Minúsculas sin acentos, para buscar como se habla."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# ---------------------------------------------------------------- estado del bot

def _fecha_objetivo(fecha: str) -> datetime.date | None:
    f = _norm((fecha or "").strip())
    hoy = datetime.date.today()
    if f in ("", "hoy"):
        return hoy
    if f == "ayer":
        return hoy - datetime.timedelta(days=1)
    if f in ("manana", "mañana"):
        return hoy + datetime.timedelta(days=1)
    try:
        return datetime.date.fromisoformat(f)
    except ValueError:
        return None


def _n(n, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _resumen_metricas(data: dict, fecha_txt: str) -> str:
    """Convierte metricas_YYYY-MM-DD.json del bot en frases (el modelo las reformula al hablar)."""
    modo = data.get("modo", "")
    trades = data.get("trades_ejecutados")
    partes = [f"Sesión del bot del {fecha_txt}" + (f" (modo {modo})" if modo else "") + "."]
    if trades == 0:
        partes.append("El bot no ejecutó trades ese día.")
    elif trades is not None:
        g, p = data.get("ganadores", 0), data.get("perdedores", 0)
        be = data.get("break_even", 0)
        det = (f"{_n(trades, 'trade', 'trades')}: {_n(g, 'ganador', 'ganadores')} "
               f"y {_n(p, 'perdedor', 'perdedores')}")
        det += f", {be} break-even." if be else "."
        partes.append(det)
        wr = data.get("win_rate_pct")
        if wr is not None:
            partes.append(f"Win rate {wr:g} por ciento.")
    pnl = data.get("pnl_usd")
    if pnl is not None:
        partes.append(f"P&L del día: {pnl:+,.2f} dólares.")
    consec = data.get("perdidas_consecutivas")
    if consec:
        partes.append(f"Pérdidas consecutivas: {consec}.")
    guards = data.get("guards_bloqueadores")
    if isinstance(guards, dict) and guards:
        nombre, n = max(guards.items(), key=lambda kv: kv[1])
        partes.append(f"El filtro que más señales bloqueó fue {nombre.replace('_', ' ')} ({n}).")
    rec = data.get("reconciliado")
    if rec is True:
        partes.append("Cifras reconciliadas con NinjaTrader.")
    elif rec is False:
        partes.append("Ojo: cifras aún sin reconciliar con NinjaTrader.")
    return " ".join(partes)


def estado_bot(rutas_cfg: dict, fecha: str = "") -> tuple[bool, str]:
    """Lee metricas_<fecha>.json de la carpeta reportes/ del bot y lo resume en español.

    Sin datos del día pedido (finde, bot apagado) cae a las métricas MÁS RECIENTES avisándolo.
    """
    raiz = (rutas_cfg or {}).get("estado_bot") if isinstance(rutas_cfg, dict) else None
    if not raiz:
        return False, "No tengo configurada la carpeta de reportes del bot, señor."
    raiz = Path(raiz)
    if not raiz.is_dir():
        return False, "La carpeta de reportes del bot no existe o no es accesible, señor."
    objetivo = _fecha_objetivo(fecha)
    if objetivo is None:
        return False, f"No entiendo la fecha '{fecha}', señor; dígame hoy, ayer o una fecha concreta."
    path = raiz / f"metricas_{objetivo.isoformat()}.json"
    aviso = ""
    if not path.is_file():
        candidatos = sorted(raiz.glob("metricas_????-??-??.json"))
        if not candidatos:
            return False, "No encuentro métricas del bot en la carpeta de reportes, señor."
        path = candidatos[-1]
        ultima = path.stem.replace("metricas_", "")
        aviso = f"No hay métricas del {objetivo.isoformat()}; las últimas son del {ultima}. "
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Métricas ilegibles: %s", path, exc_info=True)
        return False, "El archivo de métricas del bot está dañado o ilegible, señor."
    if not isinstance(data, dict):
        return False, "El archivo de métricas del bot no tiene el formato esperado, señor."
    fecha_txt = data.get("fecha") or path.stem.replace("metricas_", "")
    return True, aviso + _resumen_metricas(data, fecha_txt)


# ---------------------------------------------------------------- calendario económico

_IMPACTO = {3: "impacto ALTO", 2: "impacto medio", 1: "impacto bajo"}


def _cargar_eventos_dia(path: Path, objetivo: datetime.date) -> list[dict] | None:
    """Eventos del calendario para una fecha; None = archivo ilegible."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        eventos = data.get("noticias", []) if isinstance(data, dict) else []
    except Exception:
        log.warning("Calendario económico ilegible: %s", path, exc_info=True)
        return None
    return [e for e in eventos
            if isinstance(e, dict) and e.get("fecha") == objetivo.isoformat()]


def _refrescar_calendario(rutas_cfg: dict, path_json: Path) -> bool:
    """Corre el fetch del BOT (feed de calendario → eventos de ESTADOS UNIDOS con impacto
    medio/alto, el mismo filtro que Kevin usa en investing.com) cuando el JSON no cubre la
    fecha pedida. Throttle: si el archivo tiene <6h, se asume fresco (si el evento no está,
    es que NO HAY). Fail-soft: cualquier problema deja el calendario como estaba."""
    script = (rutas_cfg or {}).get("fetch_noticias") if isinstance(rutas_cfg, dict) else None
    if not script or not Path(script).is_file():
        return False
    try:
        if path_json.is_file():
            edad_h = (time.time() - path_json.stat().st_mtime) / 3600.0
            if edad_h < 6:
                return False
        import subprocess
        import sys
        log.info("Refrescando calendario económico (%s)...", Path(script).name)
        r = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        log.debug("Refresco del calendario falló; uso lo que haya.", exc_info=True)
        return False


def noticias_economicas(rutas_cfg: dict, fecha: str = "") -> tuple[bool, str]:
    """Eventos del calendario económico que el BOT mantiene (config/noticias_semana.json):
    SOLO Estados Unidos, SOLO impacto medio/alto — el mismo recorte que la página de
    investing.com de Kevin con sus filtros. Si el JSON no cubre la fecha pedida, se
    auto-refresca corriendo el fetch del bot. Acepta hoy/mañana/ayer/YYYY-MM-DD.

    (Se usa la fuente local y no un scrape de investing: su calendario carga por
    JavaScript —un fetch trae el cascarón vacío— y los EVENTOS son los mismos.)
    """
    ruta = (rutas_cfg or {}).get("noticias") if isinstance(rutas_cfg, dict) else None
    if not ruta:
        return False, "No tengo configurado el calendario económico, señor."
    path = Path(ruta)
    objetivo = _fecha_objetivo(fecha or "hoy")
    if objetivo is None:
        return False, f"No entiendo la fecha '{fecha}', señor."
    if not path.is_file():
        if not _refrescar_calendario(rutas_cfg, path) or not path.is_file():
            return False, "No encuentro el calendario económico del bot, señor."
    dia = _cargar_eventos_dia(path, objetivo)
    if dia is None:
        return False, "El calendario económico está dañado o ilegible, señor."
    if not dia and _refrescar_calendario(rutas_cfg, path):
        dia = _cargar_eventos_dia(path, objetivo) or []
    if not dia:
        return True, (f"No hay noticias económicas de Estados Unidos con impacto medio o "
                      f"alto programadas para el {objetivo.isoformat()}, señor.")
    dia.sort(key=lambda e: str(e.get("hora_ny", "")))
    partes = [f"Noticias económicas de Estados Unidos del {objetivo.isoformat()} "
              f"(impacto medio/alto, hora de Nueva York):"]
    for e in dia[:12]:
        imp = _IMPACTO.get(e.get("impacto"), "")
        partes.append(f"{e.get('hora_ny', '?')}: {e.get('nombre', '?')}"
                      + (f" ({imp})" if imp else "") + ".")
    return True, " ".join(partes)


# ---------------------------------------------------------------- búsqueda en notas

def buscar_notas(rutas_cfg: dict, termino: str) -> tuple[bool, str]:
    """Busca ``termino`` (sin distinguir acentos/mayúsculas) en las carpetas de notas
    configuradas y devuelve extractos por archivo, los más recientes primero."""
    raices = (rutas_cfg or {}).get("notas") if isinstance(rutas_cfg, dict) else None
    if isinstance(raices, str):
        raices = [raices]
    raices = [r for r in (raices or []) if r]
    if not raices:
        return False, "No tengo configuradas carpetas de notas, señor."
    t = _norm((termino or "").strip())
    if len(t) < 2:
        return False, "Necesito un término de búsqueda de al menos dos letras, señor."
    hits: list[tuple[float, str, list[str]]] = []
    escaneados = 0
    for raiz in raices:
        base = Path(raiz)
        if not base.is_dir():
            log.debug("Carpeta de notas inexistente: %s", base)
            continue
        for p in sorted(base.rglob("*")):
            if escaneados >= _MAX_ARCHIVOS:
                break
            if not p.is_file() or p.suffix.lower() not in _EXT_OK:
                continue
            escaneados += 1
            try:
                if p.stat().st_size > _MAX_FILE_BYTES:
                    continue
                texto = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue  # ilegible: se salta, la búsqueda sigue
            extractos = [ln.strip()[:200] for ln in texto.splitlines() if t in _norm(ln)]
            if extractos:
                hits.append((p.stat().st_mtime, p.name, extractos[:_MAX_EXTRACTOS]))
    if not hits:
        return True, f"No encontré nada sobre '{termino}' en las notas, señor."
    hits.sort(reverse=True)  # lo más reciente primero
    partes: list[str] = []
    total = 0
    for _, nombre, extractos in hits[:_MAX_HITS]:
        bloque = f"[{nombre}] " + " | ".join(extractos)
        if total + len(bloque) > _MAX_SALIDA:
            break
        total += len(bloque)
        partes.append(bloque)
    encontrados = f"Encontrado sobre '{termino}' en {len(hits)} notas. "
    return True, encontrados + " || ".join(partes)
