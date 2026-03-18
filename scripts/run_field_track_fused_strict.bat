@echo off
setlocal EnableExtensions

echo [INFO] scripts\run_field_track_fused_strict.bat is deprecated.
echo [INFO] Redirecting to scripts\run_field_track_strict.bat ^(field+track only^).

call "%~dp0run_field_track_strict.bat"
exit /b %errorlevel%
