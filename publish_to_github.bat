@echo off
REM Uploads this folder to https://github.com/Varshini-arsh/kar-drishti
REM First create an EMPTY public repository named kar-drishti at https://github.com/new
setlocal
cd /d "%~dp0"
set "REPO=https://github.com/Varshini-arsh/kar-drishti.git"

where git >nul 2>nul
if errorlevel 1 (
    echo Git is not installed on this computer.
    echo Install it from https://git-scm.com/download/win - keep the default options - then run this file again.
    pause
    exit /b 1
)

if not exist ".git" git init -b main
git config user.name >nul 2>nul
if errorlevel 1 git config user.name "Varshini-arsh"
git config user.email >nul 2>nul
if errorlevel 1 git config user.email "Varshini-arsh@users.noreply.github.com"

git add -A
git commit -m "Kar Drishti online demo" >nul 2>nul
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin %REPO%

echo.
echo Uploading to %REPO%
echo If a GitHub sign-in window opens, choose "Sign in with your browser" and approve it.
echo.
git push -u origin main
if errorlevel 1 (
    echo.
    echo Upload failed. Check that the repository kar-drishti exists at https://github.com/Varshini-arsh
    echo and that it is EMPTY - no README - then run this file again.
    pause
    exit /b 1
)
echo.
echo Done. Your code is at https://github.com/Varshini-arsh/kar-drishti
pause
