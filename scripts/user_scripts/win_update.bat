@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@call pip3 install -U modern-iopaint

PAUSE
