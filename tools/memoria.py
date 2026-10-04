"""Memoria PERSISTENTE de Jarvis (v1): hechos y preferencias que sobreviven entre
conversaciones y días, en un único archivo local (memory/jarvis_memory.md).

Guardarraíles (espejo de tools/notas.py):
  - UN solo archivo, append-only: aquí nunca se reescribe ni borra lo existente
    (el olvido es manual: editar el .md).
  - Tope de 32 KB; lleno, se dice por voz en lugar de escribir.
  - Se rechazan textos con pinta de secreto (key/token/password): la memoria viaja
    en cada system prompt y no es sitio para credenciales.
  - Solo texto plano: se quita el HTML y se acota cada hecho a 200 caracteres.
  - La carga al system prompt se acota a 4.000 caracteres; si excede, se omiten los
    recuerdos MÁS ANTIGUOS (Perfil y Preferencias se conservan) avisando en el log.

Devuelve (ok, mensaje en español), como tools/windows_control y tools/notas.
"""
from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path

log = logging.getLogger("jarvis")

ROOT = Path(__file__).resolve().parent.parent
_PATH = ROOT / "memory" / "jarvis_memory.md"

_SECCION = "## Recuerdos"
_MAX_HECHO = 200           # un hecho es una frase, no un párrafo
_MAX_ARCHIVO = 32 * 1024   # lleno -> Jarvis lo dice, no escribe
_MAX_PROMPT = 4000         # tope de lo inyectado al system prompt

_HTML = re.compile(r"<[^>]*>")
# "Pinta de secreto": prefijos de clave reales + palabras de credencial. No se incluye
# "clave" a secas (demasiados usos benignos: "la clave está en la paciencia").
_SECRETO = re.compile(
    r"sk-ant-|\bapi[ _-]?key\b|\btoken\b|\bpassword\b|\bpasswd\b|\bsecreto?\b|"
    r"\bcontrase[ñn]a\b|\bcredencial\w*\b|\bclave\s+(de\s+)?(api|acceso|secreta)\b",
    re.IGNORECASE,
)


def _sanear(hecho: str) -> str:
    """Texto plano en una línea: sin HTML, espacios normalizados, tope de longitud."""
    t = _HTML.sub(" ", str(hecho or ""))
    t = re.sub(r"\s+", " ", t).strip(" -•\t")
    return t[:_MAX_HECHO].strip()


def recordar(hecho: str, path: Path | str | None = None) -> tuple[bool, str]:
    """Añade ``- [YYYY-MM-DD] hecho`` a la sección Recuerdos (creándola si falta)."""
    p = Path(path or _PATH)
    texto = _sanear(hecho)
    if len(texto) < 3:
        return False, "Necesito saber qué debo recordar, señor."
    if _SECRETO.search(texto):
        return False, ("Eso parece una credencial o un secreto, señor; "
                       "no lo guardo en mi memoria por seguridad.")
    try:
        existente = p.read_text(encoding="utf-8") if p.is_file() else ""
        if len(existente.encode("utf-8")) >= _MAX_ARCHIVO:
            return False, ("Mi memoria está llena, señor; conviene depurar a mano "
                           "el archivo jarvis_memory.md.")
        bloque = ""
        if existente and not existente.endswith("\n"):
            bloque += "\n"
        if _SECCION not in existente:
            bloque += f"\n{_SECCION}\n" if existente else f"{_SECCION}\n"
        bloque += f"- [{datetime.date.today().isoformat()}] {texto}\n"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:  # APPEND-ONLY: lo existente no se toca
            f.write(bloque)
    except Exception:
        log.exception("No se pudo escribir la memoria: %s", p)
        return False, "No he podido escribir en mi memoria, señor."
    return True, f"Anotado en mi memoria, señor: {texto}."


def cargar(path: Path | str | None = None) -> str:
    """Contenido de la memoria para el system prompt (acotado a _MAX_PROMPT).

    Fail-soft: sin archivo o ilegible devuelve "" (Jarvis funciona sin memoria).
    """
    p = Path(path or _PATH)
    try:
        texto = p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except Exception:
        log.warning("Memoria ilegible: %s", p, exc_info=True)
        return ""
    if len(texto) <= _MAX_PROMPT:
        return texto
    # Excede el tope: Perfil/Preferencias se conservan; de los recuerdos, los más NUEVOS.
    cabeza, sep, resto = texto.partition(_SECCION)
    if not sep or len(cabeza) >= _MAX_PROMPT:
        log.warning("Memoria de %d chars no truncable por secciones: corte duro al tope.",
                    len(texto))
        return texto[:_MAX_PROMPT]
    presupuesto = _MAX_PROMPT - len(cabeza) - len(sep)
    lineas = resto.strip().splitlines()
    elegidas: list[str] = []
    usado = 0
    for ln in reversed(lineas):  # los recuerdos más nuevos están al final
        if usado + len(ln) + 1 > presupuesto:
            break
        elegidas.append(ln)
        usado += len(ln) + 1
    elegidas.reverse()
    log.warning("Memoria excede %d chars: se omiten %d líneas de recuerdos antiguos.",
                _MAX_PROMPT, len(lineas) - len(elegidas))
    return cabeza + _SECCION + "\n" + "\n".join(elegidas)
