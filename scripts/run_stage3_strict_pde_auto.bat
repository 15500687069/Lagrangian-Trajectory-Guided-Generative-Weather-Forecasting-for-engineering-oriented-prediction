@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_stage3_strict_pde_auto.ps1" %*
exit /b %ERRORLEVEL%

