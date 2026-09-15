"""FastAPI application for model-driven next-day retail replenishment."""

from datetime import date, timedelta
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from retail_demand_pulse.config import (
    DEFAULT_LEAD_TIME_DAYS,
    KENYAN_HOLIDAYS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATASET,
    RAW_DATASET,
    SAFETY_STOCK_FACTOR,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
REPORT_PATH = PROCESSED_DATA_DIR / "replenishment_report.csv"

app = FastAPI(
    title="Retail Demand Pulse",
    description="AI-driven next-day demand forecasting and replenishment",
    version="2.0.0",
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def convert_to_csv(input_path: Path) -> Path:
    if input_path.suffix.lower() not in {".xlsx", ".xls"}:
        return input_path
    csv_path = input_path.with_suffix(".csv")
    pd.read_excel(input_path).to_csv(csv_path, index=False)
    return csv_path


def process_dataset(input_path: Path) -> Path:
    """Run cleaning then feature engineering, preserving the stage contract."""
    from retail_demand_pulse.clean import main as clean_main
    from retail_demand_pulse.features import main as features_main
    from retail_demand_pulse.config import INTERIM_DATASET

    csv_path = convert_to_csv(input_path)
    clean_main(input_path=csv_path, output_path=INTERIM_DATASET)
    features_main(input_path=INTERIM_DATASET, output_path=PROCESSED_DATASET)
    return PROCESSED_DATASET


def _school_event(target: pd.Timestamp) -> Optional[str]:
    windows = {
        (1, range(3, 10)): "school opening",
        (5, range(3, 10)): "school term restart",
        (9, range(3, 10)): "school term restart",
    }
    for (month, days), label in windows.items():
        if target.month == month and target.day in days:
            return label
    return None


def _neighbourhood_activity(target: pd.Timestamp) -> int:
    if _school_event(target):
        return 2
    if target.weekday() in (1, 4) or target.day >= 28:
        return 1
    return 0


def _forecast_next_day(df: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Predict tomorrow per product using the trained demand model and event context."""
    target = target_date or (df["date"].max() + pd.Timedelta(days=1))
    last_rows = (
        df.sort_values("date")
        .groupby("product_id", observed=True)
        .tail(1)
        .copy()
    )
    last_rows["date"] = target
    last_rows["month"] = target.month
    last_rows["week_of_year"] = int(target.isocalendar().week)
    last_rows["weekday_num"] = target.weekday()
    last_rows["is_weekend"] = int(target.weekday() >= 5)
    last_rows["is_holiday"] = int(target.strftime("%Y-%m-%d") in KENYAN_HOLIDAYS)
    last_rows["neighbourhood_activity"] = _neighbourhood_activity(target)
    last_rows["day_sin"] = np.sin(2 * np.pi * target.dayofyear / 365.25)
    last_rows["day_cos"] = np.cos(2 * np.pi * target.dayofyear / 365.25)
    last_rows["month_sin"] = np.sin(2 * np.pi * target.month / 12)
    last_rows["month_cos"] = np.cos(2 * np.pi * target.month / 12)
    last_rows["trend_day"] = (target - df["date"].min()).days

    model_path = MODELS_DIR / "xgboost_demand.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Demand model not found at {model_path}. "
            "Run the pipeline with run_training=true before generating a report."
        )

    with model_path.open("rb") as model_file:
        artefact = pickle.load(model_file)
    features = artefact["features"]
    missing_features = sorted(set(features) - set(last_rows.columns))
    if missing_features:
        raise ValueError(
            "The feature dataset does not contain the trained model features: "
            + ", ".join(missing_features)
        )
    predictions = np.clip(
        artefact["model"].predict(last_rows[features].fillna(0)), 0, None
    )
    source = "xgboost_demand"

    return pd.DataFrame({
        "product_id": last_rows["product_id"].to_numpy(),
        "forecast_date": target.date().isoformat(),
        "forecasted_demand_per_day": np.round(predictions, 2),
        "forecasted_demand_7d": np.ceil(predictions * 7).astype(int),
        "is_forecast_holiday": bool(last_rows["is_holiday"].iloc[0]),
        "holiday_name": KENYAN_HOLIDAYS.get(target.strftime("%Y-%m-%d")),
        "school_event": _school_event(target),
        "forecast_model": source,
    })


def generate_replenishment_report(input_path: Path = PROCESSED_DATASET) -> pd.DataFrame:
    """Create the per-product report from the trained model's next-day forecast."""
    from retail_demand_pulse.replenishment import compute_replenishment

    if not (MODELS_DIR / "xgboost_demand.pkl").exists():
        raise FileNotFoundError(
            "Cannot generate a replenishment report without a trained demand model. "
            "Enable run_training or train the demand model first."
        )

    df = pd.read_csv(input_path, parse_dates=["date"])
    report = compute_replenishment(df)
    forecast = _forecast_next_day(df)
    report = report.drop(columns=["forecasted_demand_per_day", "forecasted_demand_7d"])
    report = report.merge(forecast, on="product_id", how="left")

    report["reorder_needed"] = report["current_stock"] <= report["reorder_point"]
    report["recommended_order_qty"] = np.where(
        report["reorder_needed"],
        np.ceil(report["forecasted_demand_7d"] + report["safety_stock"] - report["current_stock"]).clip(lower=0),
        0,
    ).astype(int)
    report["estimated_order_cost_kes"] = (
        report["recommended_order_qty"] * report["cost_price"]
    ).round(2)
    report.to_csv(REPORT_PATH, index=False)
    return report


def run_pipeline(input_path: Path, run_training: bool, generate_report: bool) -> dict:
    processed_path = process_dataset(input_path)
    result = {"processed_dataset": str(processed_path)}
    if run_training:
        from retail_demand_pulse.train_demand import main as train_demand
        from retail_demand_pulse.train_spoilage import main as train_spoilage

        train_demand(input_path=processed_path)
        train_spoilage(input_path=processed_path)
        result["training"] = "completed"
    if generate_report:
        report = generate_replenishment_report(processed_path)
        result["replenishment"] = "completed"
        result["forecast_date"] = report["forecast_date"].iloc[0]
        result["products"] = int(len(report))
    return result


@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = static_dir / "index.html"
    return FileResponse(index_path) if index_path.exists() else HTMLResponse("<h1>Retail Demand Pulse</h1>")


@app.get("/health")
async def health():
    return {"status": "healthy", "model_ready": (MODELS_DIR / "xgboost_demand.pkl").exists()}


@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    run_training: bool = Form(True),
    generate_replenishment: bool = Form(True),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only CSV, XLSX, or XLS files allowed")
    temp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)
    try:
        with temp_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        return {"status": "success", "filename": file.filename, **run_pipeline(temp_path, run_training, generate_replenishment)}
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
        temp_path.with_suffix(".csv").unlink(missing_ok=True)


@app.get("/replenishment/report")
async def get_replenishment_report():
    if not REPORT_PATH.exists():
        raise HTTPException(404, "Report not found. Generate one first.")
    return FileResponse(REPORT_PATH, filename=REPORT_PATH.name)


@app.get("/replenishment/products/{product_id}")
async def get_product_recommendation(product_id: str):
    if not REPORT_PATH.exists():
        raise HTTPException(404, "Report not found. Generate one first.")
    report = pd.read_csv(REPORT_PATH)
    match = report[report["product_id"] == product_id]
    if match.empty:
        raise HTTPException(404, "Product not found in the latest report.")
    return {key: to_native(value) for key, value in match.iloc[0].to_dict().items()}


@app.post("/predict/replenishment")
async def predict_single_replenishment(
    product_name: str = Form(...),
    category: str = Form(...),
    is_perishable: int = Form(0),
    current_stock: int = Form(...),
    avg_daily: float = Form(...),
    forecasted_demand_per_day: Optional[float] = Form(None),
    product_id: str = Form("P999"),
    shelf_life_days: int = Form(730),
    unit_price: float = Form(110.0),
    cost_price: float = Form(78.0),
    std_daily: float = Form(3.0),
):
    """Return the same replenishment calculation used by the dashboard."""
    if forecasted_demand_per_day is None and REPORT_PATH.exists():
        report = pd.read_csv(REPORT_PATH)
        match = report[report["product_id"] == product_id]
        if not match.empty:
            forecasted_demand_per_day = float(match.iloc[0]["forecasted_demand_per_day"])
    forecast = max(float(forecasted_demand_per_day or avg_daily), 0.0)
    safety_stock = round(SAFETY_STOCK_FACTOR * max(std_daily, 0.5) * np.sqrt(DEFAULT_LEAD_TIME_DAYS), 1)
    reorder_point = round(avg_daily * DEFAULT_LEAD_TIME_DAYS + safety_stock, 1)
    reorder_needed = current_stock <= reorder_point
    order_qty = max(0, int(np.ceil(forecast * 7 + safety_stock - current_stock))) if reorder_needed else 0
    days_left = round(current_stock / avg_daily, 1) if avg_daily > 0 else 0.0
    status = "OUT OF STOCK" if current_stock == 0 else "CRITICALLY LOW" if days_left < 3 else "LOW" if reorder_needed else "OK"
    priority = "URGENT" if status in {"OUT OF STOCK", "CRITICALLY LOW"} else "HIGH" if status == "LOW" and is_perishable else "MEDIUM" if status == "LOW" else "OK"
    return {"product_id": product_id, "product_name": product_name, "category": category,
            "is_perishable": bool(is_perishable), "shelf_life_days": shelf_life_days,
            "unit_price": unit_price, "cost_price": cost_price, "current_stock": current_stock,
            "avg_daily": avg_daily, "std_daily": std_daily, "forecasted_demand_per_day": forecast,
            "forecasted_demand_7d": int(np.ceil(forecast * 7)), "safety_stock": safety_stock,
            "reorder_point": reorder_point, "days_of_stock_remaining": days_left,
            "stock_status": status, "reorder_needed": reorder_needed,
            "recommended_order_qty": order_qty, "estimated_order_cost_kes": round(order_qty * cost_price, 2),
            "replenishment_priority": priority}


if __name__ == "__main__":
    uvicorn.run("retail_demand_pulse.main:app", host="0.0.0.0", port=8000, reload=True)