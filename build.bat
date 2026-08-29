@echo off
REM Build cs2translate.exe. Run this ON a Windows machine with Python 3.10+.
REM Produces dist\cs2translate\ -- a self-contained folder you can copy
REM anywhere. Models are NOT bundled; they download to
REM %USERPROFILE%\.cache\cs2translate on first run.
REM
REM   build.bat            full build, bundles cuBLAS + cuDNN (~2GB, GPU works
REM                        on any machine with just an NVIDIA driver)
REM   build.bat nocuda     smaller build; the target machine must already have
REM                        CUDA 12 + cuDNN 9 on PATH, or run on CPU

setlocal
set CS2T_CUDA=1
if /I "%~1"=="nocuda" set CS2T_CUDA=0

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found on PATH. Install Python 3.11 from python.org
  echo        and tick "Add python.exe to PATH".
  exit /b 1
)

echo [1/5] Creating build venv...
if not exist .venv-build python -m venv .venv-build || exit /b 1
call .venv-build\Scripts\activate.bat

echo [2/5] Installing dependencies...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || exit /b 1
pip install pyinstaller || exit /b 1
if "%CS2T_CUDA%"=="1" (
  echo       ...plus cuBLAS/cuDNN for GPU inference
  pip install -r requirements-cuda.txt || exit /b 1
)

echo [3/5] Running tests...
pip install pytest >nul
python -m pytest tests -q || exit /b 1

echo [4/5] Building with PyInstaller (several minutes)...
pyinstaller --noconfirm --clean cs2translate.spec || exit /b 1

echo [5/5] Packaging...
copy /Y config.example.toml dist\cs2translate\ >nul
dist\cs2translate\cs2translate.exe --help >nul
if errorlevel 1 (
  echo ERROR: the frozen exe did not start.
  exit /b 1
)

echo.
echo Done. Distributable folder: dist\cs2translate\
echo Copy that whole folder to the target machine and run cs2translate.exe
echo Optional: powershell Compress-Archive -Path dist\cs2translate\* -DestinationPath cs2translate.zip
endlocal
