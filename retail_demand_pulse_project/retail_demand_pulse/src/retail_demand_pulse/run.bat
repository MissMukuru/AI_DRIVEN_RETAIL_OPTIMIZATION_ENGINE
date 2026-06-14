@echo off
echo Starting Retail Demand Pulse...
call venv\Scripts\activate
python -m uvicorn src.retail_demand_pulse.main:app --reload
pause
