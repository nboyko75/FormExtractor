@echo off
setlocal EnableExtensions DisableDelayedExpansion

if /i "%~1"=="--help" goto :help
if /i "%~1"=="/?" goto :help
if not "%~1"=="" (
    echo ERROR: Unknown argument "%~1".
    exit /b 2
)

rem Resolve all paths relative to this script, including when called elsewhere.
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "BUILD_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%BUILD_PYTHON%" (
    echo ERROR: Create the project environment first:
    echo   py -3.10 -m venv .venv
    goto :failed
)

"%BUILD_PYTHON%" -c "import sys, struct; sys.exit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
    echo ERROR: Use 64-bit Python 3.10 or 3.11 for this app's Paddle dependencies.
    goto :failed
)

echo Installing application and build dependencies...
rem Prefer compatible local wheels when available; pip can also use its index.
if exist "Whl\" (
    "%BUILD_PYTHON%" -m pip install --find-links "Whl" . "pyinstaller>=6,<7"
) else (
    "%BUILD_PYTHON%" -m pip install . "pyinstaller>=6,<7"
)
if errorlevel 1 goto :failed

echo Building FormExtractor...
rem Keep Paddle's native libraries in a folder beside the executable.
rem Collect dynamically imported OCR modules, resources and package metadata.
"%BUILD_PYTHON%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --windowed ^
    --noupx ^
    --name FormExtractor ^
    --paths "src" ^
    --distpath "dist" ^
    --workpath "build\pyinstaller" ^
    --specpath "build" ^
    --additional-hooks-dir "packaging\hooks" ^
    --collect-all paddle ^
    --collect-all paddleocr ^
    --collect-all paddlex ^
    --recursive-copy-metadata paddleocr ^
    --recursive-copy-metadata paddlex ^
    --recursive-copy-metadata paddlepaddle ^
    --copy-metadata formextractor ^
    --collect-data formextractor ^
    "main.py"
if errorlevel 1 goto :failed

if not exist "dist\FormExtractor\FormExtractor.exe" goto :failed
echo.
echo Build complete: "%CD%\dist\FormExtractor\FormExtractor.exe"
echo Distribute the entire dist\FormExtractor folder.
echo OCR models are downloaded on first recognition unless already cached.
popd
exit /b 0

:failed
echo.
echo ERROR: Build failed. See the error above.
popd
exit /b 1

:help
echo Usage: build.cmd
echo Builds the Windows desktop app using .venv and PyInstaller.
echo Requires 64-bit Python 3.10 or 3.11 in .venv.
echo Installs missing dependencies using pip and optional local Whl files.
echo Replaces the previous build in dist\FormExtractor.
echo Output: dist\FormExtractor\FormExtractor.exe
exit /b 0
