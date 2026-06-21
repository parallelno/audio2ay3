@echo off
setlocal

REM Dual-AY conversion of every long sample with the 6-stem Demucs separator
REM (--separation demucs6) + the default basic-pitch transcriber.
REM Produces, per song, in results\demucs6_dual\ :
REM   <name>.ym  + <name>.ay2.ym   (chip 0 / chip 1)   and   <name>.mp3 (mixed preview)
REM ONE neural pass per song (the Python driver writes the .ym files and renders the mixed
REM .mp3 from the same in-memory dual-chip song -- it does NOT run the front-end twice; a plain
REM `validate <name>.ym` would render only chip 0).

set PY=C:\Users\parallelno\AppData\Local\Programs\Python\Python312\python.exe

"%PY%" scripts\convert_long_dual.py --separation demucs6 --out-dir results\demucs6_dual %*
if errorlevel 1 ( echo. & echo One or more songs failed -- see the log above. & exit /b 1 )

echo.
echo All done -^> results\demucs6_dual\
endlocal
