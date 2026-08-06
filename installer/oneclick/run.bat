@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" %*
set "MODERN_IOPAINT_EXIT_CODE=%ERRORLEVEL%"
if not "%MODERN_IOPAINT_EXIT_CODE%"=="0" (
  echo.
  echo Modern-IOPaint setup failed. Review the message above and setup.log.
  pause
)
exit /b %MODERN_IOPAINT_EXIT_CODE%
