@echo off
setlocal

set PY=C:\Users\parallelno\AppData\Local\Programs\Python\Python312\python.exe
set OUT=results\long

if not exist %OUT% mkdir %OUT%

for %%F in (
    Dungeon_Ore
    Goblins_Lair
    Pixel_Hearthbeat
    The_Dragons_Lair
    The_Forgotten_Sanctum
    The_Last_Pixel
) do (
    echo.
    echo === %%F ===
    "%PY%" -m audio2ay3 convert samples\long\%%F.mp3 -o %OUT%\%%F.ym --separation demucs
    if errorlevel 1 ( echo FAILED: convert %%F & goto :eof )

    "%PY%" -m audio2ay3 validate %OUT%\%%F.ym -o %OUT%\%%F.mp3
    if errorlevel 1 ( echo FAILED: validate %%F & goto :eof )

    echo ok: %%F.ym + %%F.mp3
)

echo.
echo All done.
endlocal
