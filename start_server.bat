@echo off
echo Starting US Market Dashboard Server...
echo Logs will be saved to logs/server.log

:: Activate virtual environment if it exists (adjust path as needed)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Run Flask App
python flask_app.py

pause
