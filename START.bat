@echo off
chcp 65001 >nul
title AI Image Enhancer
cd /d "%~dp0"

C:\Users\Artur\AppData\Local\Programs\Python\Python310\python.exe main.py
if errorlevel 1 (
    echo.
    echo [BLAD] Program zakonczyl sie z bledem. Szczegoly powyzej.
    pause
)
