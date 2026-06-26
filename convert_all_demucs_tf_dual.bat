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

mkdir results\demucs-ft_dual 2>nul
mkdir results\demucs-ft_dual\long_real 2>nul

%PY% -m audio2ay3 preview "samples\long_chiptunes\00.m4a" -o "results\demucs-ft_dual\long_chiptunes\00.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_chiptunes\01.m4a" -o "results\demucs-ft_dual\long_chiptunes\01.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_chiptunes\02.m4a" -o "results\demucs-ft_dual\long_chiptunes\02.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_chiptunes\03.m4a" -o "results\demucs-ft_dual\long_chiptunes\03.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_chiptunes\04.m4a" -o "results\demucs-ft_dual\long_chiptunes\04.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems

%PY% -m audio2ay3 preview "samples\long_real\00.m4a" -o "results\demucs-ft_dual\long_real\00.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_real\01.m4a" -o "results\demucs-ft_dual\long_real\01.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_real\02.m4a" -o "results\demucs-ft_dual\long_real\02.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_real\03.m4a" -o "results\demucs-ft_dual\long_real\03.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_real\04.m4a" -o "results\demucs-ft_dual\long_real\04.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems
%PY% -m audio2ay3 preview "samples\long_real\05.m4a" -o "results\demucs-ft_dual\long_real\05.mp3" --separation demucs-ft --chips 2 --noise-volume 0.5 --vocals lead --vibrato vocals --explain --save-midi --save-stems

echo.
echo All done -^> results\demucs-ft_dual\
endlocal
