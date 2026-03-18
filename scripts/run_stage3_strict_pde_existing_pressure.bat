@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_stage3_strict_pde_existing_pressure.ps1" %*
exit /b %ERRORLEVEL%

