@echo off
setlocal

REM Dual-AY YourMT3 (YMT3+) batch conversion for every long sample.
REM Produces, per song, in results\ymt3plus_dual\ :
REM   <name>.ym  + <name>.ay2.ym   (chip 0 / chip 1)   and   <name>.mp3 (mixed preview)
REM ONE YourMT3 inference per song (the Python driver writes the .ym files and renders the
REM mixed .mp3 from the same in-memory dual-chip song -- it does NOT run inference twice).
REM
REM Requires: pip install -e ".[yourmt3,mp3]"  and  `python -m audio2ay3 setup-yourmt3`
REM (clones the GPL YourMT3 backend into the per-user cache; ships the YMT3+ checkpoint via LFS).

set PY=C:\Users\parallelno\AppData\Local\Programs\Python\Python312\python.exe

"%PY%" scripts\convert_long_dual.py --transcription yourmt3 --model "YMT3+" --separation none --out-dir results\ymt3plus_dual %*
if errorlevel 1 ( echo. & echo One or more songs failed -- see the log above. & exit /b 1 )

echo.
echo All done -> results\ymt3plus_dual\
endlocal
