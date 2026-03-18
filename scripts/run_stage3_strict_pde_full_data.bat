@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_stage3_strict_pde_full_data.ps1" %*
exit /b %ERRORLEVEL%

