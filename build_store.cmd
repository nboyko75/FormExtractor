@echo off
setlocal EnableExtensions DisableDelayedExpansion
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0store\build_msix.ps1" -ProjectRoot "%~dp0." %*
exit /b %errorlevel%
