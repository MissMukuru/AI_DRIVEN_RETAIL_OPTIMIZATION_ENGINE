@echo off
echo Starting Retail Demand Pulse...
call "%~dp0..\..\..\..\venv\Scripts\activate.bat"
python -m uvicorn retail_demand_pulse.main:app --app-dir "%~dp0.." --host 127.0.0.1 --port 8000 --reload
pause
