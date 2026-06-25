@echo off
setlocal

REM Dual-AY conversion of every long sample with the *fine-tuned* Demucs separator
REM (--separation demucs-ft) + the default basic-pitch transcriber.
REM Produces, per song, in results\demucs-ft_dual\ :
REM   <name>.ym  + <name>.ay2.ym   (chip 0 / chip 1)   and   <name>.mp3 (mixed preview)
REM
REM Uses the one-pass driver so the .mp3 mixes BOTH chips. The old convert + `validate`
REM approach was broken for dual-AY: `validate <name>.ym` renders only chip 0, so the .mp3
REM dropped chip 1's voices. The driver renders the mixed .mp3 from the in-memory dual-chip
REM song right after writing the two .ym files -- one neural pass, correct mix.

set PY=%~dp0.venv\Scripts\python.exe

"%PY%" scripts\convert_long_dual.py --separation demucs-ft --vocals lead --vibrato vocals --noise-volume 0.5 --arpeggio --out-dir results\demucs-ft_dual_arpeggio %* --explain 
if errorlevel 1 ( echo. & echo One or more songs failed -- see the log above. & exit /b 1 )

echo.
echo All done -^> results\demucs-ft_dual\
endlocal
