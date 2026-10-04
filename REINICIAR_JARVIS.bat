@echo off
REM ============================================================
REM  JARVIS - REINICIO real
REM  1) Mata cualquier instancia previa de Jarvis (si el puerto
REM     del HUD o el micro quedan tomados, la nueva arrancaria
REM     sin esfera ni atajo -- pasaba con el doble clic del .lnk).
REM  2) Desactiva QuickEdit (un clic accidental congela el bucle).
REM  3) Lanza main.py.
REM ============================================================
title JARVIS
cd /d "%~dp0"
REM Sin 'chcp 65001': cambiar el code page a UTF-8 hace que la consola clásica cambie de fuente.
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f >nul 2>&1

powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" -ErrorAction SilentlyContinue | Where-Object { $_.ExecutablePath -like '*\jarvis\*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } ; Start-Sleep -Milliseconds 800" >nul 2>&1

if not exist ".venv\Scripts\python.exe" (
  echo [JARVIS] No existe el entorno virtual .venv.
  echo [JARVIS] Crealo con:  py -3.14 -m venv .venv
  echo [JARVIS] Luego:       .venv\Scripts\python -m pip install -r requirements.txt
  if /I not "%~1"=="--sin-pausa" pause
  exit /b 1
)

".venv\Scripts\python.exe" main.py
echo.
echo [JARVIS] Proceso finalizado.
REM --sin-pausa: lo usa el arranque matutino, que lanza este .bat con la ventana
REM OCULTA (scripts\lanzar_oculto.vbs). Un 'pause' ahi dejaria un cmd invisible
REM colgado para siempre cada vez que Jarvis termina.
if /I not "%~1"=="--sin-pausa" pause
