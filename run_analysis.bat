@echo off
echo Starting Full Market Analysis...
echo This may take several minutes.
echo Logs will be saved to logs/scheduler.log

:: Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Run Update Script
python us_market\update_all.py

echo.
echo Update Complete.
pause
