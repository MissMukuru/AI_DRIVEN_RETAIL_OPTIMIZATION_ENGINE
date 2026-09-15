FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "retail_demand_pulse.main:app", "--app-dir", "retail_demand_pulse_project/retail_demand_pulse/src", "--host", "0.0.0.0", "--port", "8000"]
