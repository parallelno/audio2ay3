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

SET o0=00.m4a
SET o1=01.m4a
SET o2=02.m4a
SET o3=03.m4a
SET o4=04.m4a
SET o5=05.m4a

SET in1=samples\long_chiptunes\
SET in2=samples\long_real\

SET out1=results\demucs-ft_dual\long_chiptunes\
SET out2=results\demucs-ft_dual\long_real\

mkdir %out1% 2>nul
mkdir %out2% 2>nul

%PY% -m audio2ay3 preview "%in1%%o0%" -o "%out1%%o0%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in1%%o1%" -o "%out1%%o1%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in1%%o2%" -o "%out1%%o2%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in1%%o3%" -o "%out1%%o3%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in1%%o4%" -o "%out1%%o4%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in1%%o5%" -o "%out1%%o5%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems

%PY% -m audio2ay3 preview "%in2%%o0%" -o "%out2%%o0%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in2%%o1%" -o "%out2%%o1%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in2%%o2%" -o "%out2%%o2%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in2%%o3%" -o "%out2%%o3%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in2%%o4%" -o "%out2%%o4%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "%in2%%o5%" -o "%out2%%o5%" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems

echo.
echo All done -^> results\demucs-ft_dual\
endlocal
