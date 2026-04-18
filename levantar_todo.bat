@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ==============================================
echo   Launcher Unico Bot + Dashboard (9000)
echo ==============================================

if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
        echo [INFO] No existia .env. Se creo automaticamente desde .env.example
    ) else (
        (
            echo TRADING_MODE=paper
            echo BINANCE_API_KEY=tu_api_key_aqui
            echo BINANCE_API_SECRET=tu_api_secret_aqui
        ) > ".env"
        echo [INFO] No existia .env ni .env.example. Se creo .env por defecto en modo paper.
    )
)

set "TRADING_MODE=paper"
set "BINANCE_API_KEY="
set "BINANCE_API_SECRET="

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    set "K=%%A"
    set "V=%%B"
    if /I "!K!"=="TRADING_MODE" set "TRADING_MODE=!V!"
    if /I "!K!"=="BINANCE_API_KEY" set "BINANCE_API_KEY=!V!"
    if /I "!K!"=="BINANCE_API_SECRET" set "BINANCE_API_SECRET=!V!"
)

if /I "%TRADING_MODE%"=="real" (
    if "%BINANCE_API_KEY%"=="" (
        echo [ERROR] BINANCE_API_KEY vacia en .env
        pause
        exit /b 1
    )
    if "%BINANCE_API_SECRET%"=="" (
        echo [ERROR] BINANCE_API_SECRET vacia en .env
        pause
        exit /b 1
    )
    if /I "%BINANCE_API_KEY%"=="tu_api_key_aqui" (
        echo [ERROR] BINANCE_API_KEY sigue en placeholder.
        pause
        exit /b 1
    )
    if /I "%BINANCE_API_SECRET%"=="tu_api_secret_aqui" (
        echo [ERROR] BINANCE_API_SECRET sigue en placeholder.
        pause
        exit /b 1
    )

    echo.
    echo [ALERTA] MODO REAL detectado.
    set /p CONFIRM=Escribi SI para continuar: 
    if /I not "%CONFIRM%"=="SI" (
        echo Cancelado por usuario.
        exit /b 0
    )
)

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py"
) else (
    set "PY_CMD=python"
)

if not exist "requirements.txt" (
    echo [ERROR] No existe requirements.txt
    pause
    exit /b 1
)

echo.
echo [INFO] Cerrando instancias anteriores si existen...
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":9000 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%P >nul 2>nul
)
for /f "tokens=2" %%P in ('tasklist /FI "WINDOWTITLE eq Scalping Bot" /NH 2^>nul ^| findstr /I "cmd py python"') do (
    taskkill /F /PID %%P >nul 2>nul
)
echo [OK] Limpieza completada.

echo.
echo [INFO] Instalando/verificando dependencias...
%PY_CMD% -m pip install -r requirements.txt
if not %errorlevel%==0 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)

echo.
echo [INFO] Validando sintaxis de archivos principales...
%PY_CMD% -m py_compile scalping_bot.py dashboard.py preflight_real.py train_ai_model.py
if not %errorlevel%==0 (
    echo [ERROR] Hay errores de sintaxis. Revisa los archivos antes de iniciar.
    pause
    exit /b 1
)

echo.
echo [INFO] Ejecutando preflight de operacion...
%PY_CMD% preflight_real.py
if not %errorlevel%==0 (
    echo [ERROR] Preflight fallido. Corregi los errores antes de iniciar.
    pause
    exit /b 1
)

start "Scalping Bot" cmd /k "%PY_CMD% scalping_bot.py"
start "Dashboard 9000" cmd /k "%PY_CMD% dashboard.py"

echo.
echo Lanzado en modo: %TRADING_MODE%
echo Ventanas abiertas:
echo - Scalping Bot
echo - Dashboard 9000
echo URL: http://localhost:9000
endlocal
