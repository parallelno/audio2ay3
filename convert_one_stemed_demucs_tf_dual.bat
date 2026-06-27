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

SET o0=01 (Bass).mp3
SET o1=01 (Drums).mp3
SET o2=01 (Other).mp3
SET o3=01 (Vocals).mp3

SET in1=samples\stems\ducktails\

SET out=results\

mkdir %out% 2>nul

%PY% -m audio2ay3 preview "%in1%%o0%" -o "%out%%o0%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in1%%o1%" -o "%out%%o1%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in1%%o2%" -o "%out%%o2%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in1%%o3%" -o "%out%%o3%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems

echo.
echo All done -^> %out%
endlocal
