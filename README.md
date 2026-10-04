# JARVIS — asistente personal de voz (Windows)

> **Fases 0 a 5 terminadas**: activación (palmada + wake word + atajo global), voz local
> española (faster-whisper y Piper), cerebro con herramientas reales (API de Anthropic con
> tool use) y consultas de trading de solo lectura. El estado real de cada fase está más
> abajo, sin adornos.
>
> **In English:** a fully local Spanish voice assistant for Windows. Wakes on a **double
> hand-clap** (custom DSP detector), on **"Hey Jarvis"** (openWakeWord) or on a **global
> hotkey** (press-to-wake, works even with music playing); hears with faster-whisper and
> speaks with Piper — **no paid speech SaaS**. The brain is Claude via the Anthropic API
> with tool use; every action goes through an app whitelist, read-only trading queries,
> and an audit log. Phases 0-5 done.

Asistente de voz de escritorio, **separado del bot de trading**. Se activa con **doble
palmada + tu petición directa** ("dime las noticias de mañana" — sin necesidad de decir
"Jarvis"), con la frase **"Hey Jarvis"** o pulsando **Ctrl+Alt+J** (útil cuando suena
música), escucha en español, **abre apps y controla Windows**, consulta **resultados del
bot, calendario económico y notas (solo lectura)** y responde hablando. Se despide con
"gracias", "terminamos" o corta EN SECO con "cancelar"/"silencio".

- **Voz:** 100% local, sin SaaS de pago por uso (faster-whisper para oír; TTS neural
  local — Piper — para hablar, por oraciones con síntesis adelantada). ElevenLabs/Deepgram
  quedan excluidos.
- **Cerebro:** por SUSCRIPCIÓN de Claude (Agent SDK sobre el CLI `claude` logueado —
  cero gasto de API; cliente PERSISTENTE: ~1.5-3.5s por turno) con fallback manual a la
  API de Anthropic (`cerebro.backend`). Herramientas: hora, abrir apps, música, búsqueda
  y lectura web, calendario económico del bot, estado del bot MNQ y notas personales.
- **Convivencia con la música:** ducking estilo Alexa — al abrir una conversación la
  sesión de audio de Spotify se atenúa (pycaw) y se restaura al cerrar.
- **Seguridad:** lista blanca de apps, herramientas del CLI capadas en 3 capas
  fail-closed (las built-in de sistema no existen para el modelo), consultas de
  trading/notas SOLO LECTURA (texto plano, nunca HTML), `modo_seguro` (simulación) por
  defecto, instancia única (mutex) y auditoría en `logs/acciones.log`.

## Estructura
```
jarvis/
  main.py                 # orquestador (máquina de estados IDLE→WAKE→LISTEN→THINK→SPEAK)
  hotkey.py               # atajo de teclado GLOBAL press-to-wake (Win32 RegisterHotKey)
  states.py  config.py  logging_setup.py  console_win.py
  audio/  clap_detector.py  wake_word.py  stt.py  tts.py  mic_capture.py
  brain/  agent.py          # API Anthropic + tool use
  tools/  windows_control.py  notas.py  memoria.py  # notas = SOLO LECTURA; memoria = persistente
  config/ settings.json  apps.json       # apps.json = lista blanca
  memory/ jarvis_memory.example.md       # plantilla; la memoria real (jarvis_memory.md) es personal y NO va a git
  tests/  (111 tests)
  logs/   requirements.txt  pyproject.toml  REINICIAR_JARVIS.bat
```

## Cómo ejecutar
```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py        # o doble clic en REINICIAR_JARVIS.bat
copy memory\jarvis_memory.example.md memory\jarvis_memory.md   # opcional: memoria persistente (se crea sola al primer "recuerda")
```

## Estado por fases
- [x] **Fase 0** — esqueleto: estructura, config, logging, orquestador, tests. Arranca en IDLE.
- [x] **Fase 1** — activación por doble palmada (núcleo DSP `ClapCore` + `ClapDetector`, comité multi-criterio). Validada en vivo; calibrador en `scripts/calibrar_palmada.py`.
- [x] **Fase 2** — voz local: STT faster-whisper + TTS Piper (español), wake word "Hey Jarvis" (openWakeWord; "Jarvis" a secas con Porcupine si se añade una key gratis de Picovoice). Validada en vivo.
- [x] **Fase 3** — cerebro: API de Anthropic (Messages + tool use) con memoria de conversación multiturno (una activación = una conversación). Validado en vivo.
- [x] **Fase 4** — control de Windows: lista blanca (`config/apps.json`), abrir apps por voz, poner música (YouTube/Spotify). Validado en vivo.
- [x] **Fase 5** — consultas de trading y notas SOLO LECTURA: `estado_bot` (métricas de la sesión del bot MNQ: trades, win rate, P&L, filtros) y `buscar_notas` (búsqueda en notas personales). Carpetas en `rutas.*` de settings.json; texto plano únicamente, nunca HTML.
- [x] **Extra** — atajo global press-to-wake (`hotkey.py`): `Ctrl+Alt+J` activa aunque suene música (el micro se ensordece sin AEC y las palmadas no llegan; el teclado siempre responde). Config en `hotkey.activo` / `hotkey.combinacion`.
- [x] **Extra** — memoria PERSISTENTE v1 (`tools/memoria.py` + `memory/jarvis_memory.md`): "Jarvis, recuerda que..." guarda el hecho (append-only, tope 32 KB, rechaza credenciales) y se inyecta al system prompt al abrir cada conversación (ambos backends). Olvidar = borrar la línea del .md.
- [x] **Extra** — HUD visual (`hud/`): esfera-grafo estilo Iron Man en `http://127.0.0.1:36911` (servidor stdlib local, solo 127.0.0.1, rutas whitelist). Los nodos son los archivos reales de las notas; el color y el movimiento reaccionan al estado en vivo (reposo/escuchando/procesando/hablando) y muestra la última frase. Config en `hud.*` (puerto, abrir navegador al arrancar).
- [ ] **Fase 6** — pulido end-to-end.

## Nota de compatibilidad (importante)
El equipo solo tiene **Python 3.14**. Parte del stack de ML (p. ej. PyTorch para Coqui
XTTS-v2) puede no tener wheels para 3.14. Por eso se prefieren motores **ONNX** (Kokoro,
Piper). Si en las Fases 2-3 falta algún wheel, el plan B es instalar un **Python 3.12**
dedicado para el `.venv` de Jarvis.

## Probar la activación por palmada (Fase 1)
- `.venv\Scripts\python main.py` → estado IDLE; **da dos palmadas** → "¡Activado, señor!" (Ctrl-C para salir).
- Calibración (ver métricas de cada palmada para afinar `audio.palmada`):
  `.venv\Scripts\python scripts\calibrar_palmada.py`

## Probar la voz (Fase 2)
- `.venv\Scripts\python main.py` → **dos palmadas**, luego **habla una frase** → Jarvis responde "Entendí: …".
- Reproducir voz/modelos en otra máquina: `.venv\Scripts\python scripts\descargar_voz.py`
- Calibrar la voz (latencias + transcripción + sensibilidad del endpointer): `.venv\Scripts\python scripts\calibrar_voz.py`
- Stack STT/TTS confirmado en Python 3.14 (faster-whisper + Piper). Si el endpointer corta tarde
  con ruido de fondo, sube `audio.endpoint.umbral_factor` en `config/settings.json`.

## Limitaciones conocidas (pendiente de calibración con micrófono real)
Hallazgos de la revisión adversarial que dependen del hardware/acústica y se afinan en
campo (los umbrales viven en `config/settings.json` → `audio.palmada`):
- **Percusión rítmica de banda ancha** (hi-hat/snare a tempo) o **aplausos de TV** pueden
  parecerse acústicamente a una palmada; el cooldown limita la molestia. Mitigación futura:
  veto anti-ritmo (≥3 onsets regulares).
- **Palmadas muy suaves/lejanas** cerca del piso de ruido pueden perderse si el pico cae
  justo en una frontera de frame (efecto marginal; subir el volumen o acercarse lo resuelve).
- **Mejoras de audio de Windows** (Voice Focus/AGC) pueden aplanar el transitorio; si falla
  la detección, desactívalas o ajusta `crest_min`/`attack_min`.
Ya resueltos y cubiertos por tests: clipping (palmada saturada), entrada con NaN/inf,
config inválida, wiring del orquestador, cierre del stream y errores del micrófono.


## Configuración local (no incluida)

`config/secrets.json` está en `.gitignore` y **no viaja en el repo**: cada quien pone su
propia llave.

```json
{ "anthropic_api_key": "sk-ant-..." }
```

`config/apps.json` es la lista blanca de apps que Jarvis puede abrir, con rutas de ESTA
máquina como ejemplo: cámbialas por las tuyas.

## Licencia

MIT — ver [LICENSE](LICENSE). Sin garantía: controla tu escritorio, úsalo con `modo_seguro`
activo hasta que confíes en él.
