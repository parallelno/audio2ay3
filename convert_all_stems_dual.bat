@echo off
setlocal

REM Dual-AY conversion using pre-separated stems (samples\stems\).
REM Demucs is skipped entirely: Synth / Bass / Drums stems are loaded directly, giving
REM the transcriber a perfectly clean signal instead of a Demucs-separated approximation.
REM FX stems (when present) are mixed into the instrumental for additional colour.
REM
REM Produces, per song, in results\stems_dual\ :
REM   <name>.ym  + <name>.ay2.ym   (chip 0 / chip 1)   and   <name>.mp3 (mixed preview)

set PY=C:\Users\parallelno\AppData\Local\Programs\Python\Python312\python.exe

"%PY%" scripts\convert_long_dual.py --stems-dir samples\stems --stems-only --noise-volume 0.3 --format vtx --out-dir results\stems_dual %*
if errorlevel 1 ( echo. & echo One or more songs failed -- see the log above. & exit /b 1 )

echo.
echo All done -^> results\stems_dual\
endlocal
