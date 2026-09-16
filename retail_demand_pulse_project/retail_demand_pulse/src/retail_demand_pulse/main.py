"""FastAPI application for model-driven next-day retail replenishment."""

from datetime import date, datetime, timedelta
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Optional
from zoneinfo import ZoneInfo

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
    CURRENT_PRICE_AS_OF,
    CURRENT_PRICES_KES,
    kenyan_holidays_for_year,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATASET,
    RAW_DATASET,
    SAFETY_STOCK_FACTOR,
    WEATHER_PROFILES,
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
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, np.generic):
        return to_native(value.item())
    if isinstance(value, np.ndarray):
        return to_native(value.tolist())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if pd.isna(value):
        return None
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
    term = _school_term(target)
    if term is None:
        return None
    if target.day <= 14 and target.month in (1, 5, 9):
        return f"{term} opening"
    return term


def _school_term(target: pd.Timestamp) -> Optional[str]:
    """Return the local three-term school calendar used for demand context."""
    if 1 <= target.month <= 4:
        return "school term 1"
    if 5 <= target.month <= 8:
        return "school term 2"
    if 9 <= target.month <= 10:
        return "school term 3"
    return None


def _holiday_name(target: pd.Timestamp) -> Optional[str]:
    holiday_name = kenyan_holidays_for_year(target.year).get(target.strftime("%Y-%m-%d"))
    if holiday_name:
        return holiday_name
    exact_name = KENYAN_HOLIDAYS.get(target.strftime("%Y-%m-%d"))
    if exact_name:
        return exact_name
    month_day = target.strftime("-%m-%d")
    return next(
        (name for holiday_date, name in KENYAN_HOLIDAYS.items()
         if holiday_date.endswith(month_day)),
        None,
    )


def _neighbourhood_activity(target: pd.Timestamp) -> int:
    if _school_event(target) and target.day <= 14:
        return 2
    if _school_term(target):
        return 1
    if target.weekday() in (1, 4) or target.day >= 28:
        return 1
    return 0


def _expected_weather(target: pd.Timestamp) -> dict:
    """Build a deterministic future weather row from the configured profile."""
    temperature, rain_probability, average_rain, weights = WEATHER_PROFILES[target.month]
    conditions = ["Sunny", "Partly Cloudy", "Overcast", "Light Rain", "Heavy Rain"]
    condition = conditions[int(np.argmax(weights))]
    rainfall = average_rain * rain_probability if condition != "Sunny" else 0.0
    return {
        "temperature_avg": float(temperature),
        "rainfall_mm": float(rainfall),
        "weather_condition": condition,
    }


def _current_kenya_date() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(ZoneInfo("Africa/Nairobi")).date())


def _apply_current_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Use the dated catalogue for economics while preserving product identity."""
    result = df.copy()
    for product_id, prices in CURRENT_PRICES_KES.items():
        mask = result["product_id"].eq(product_id)
        for column, value in prices.items():
            if column in result.columns:
                result.loc[mask, column] = value
    result["profit_margin"] = (
        (result["unit_price"] - result["cost_price"]) / result["unit_price"]
    ).round(4)
    return result


def _future_feature_rows(df: pd.DataFrame, target: pd.Timestamp, features: list[str]) -> pd.DataFrame:
    """Create one model input row per product for the requested future date."""
    rows = (
        df.sort_values("date")
        .groupby("product_id", observed=True)
        .tail(1)
        .copy()
    )
    weather = _expected_weather(target)
    holiday_name = _holiday_name(target)
    rows["date"] = target
    rows["month"] = target.month
    rows["week_of_year"] = int(target.isocalendar().week)
    rows["weekday_num"] = target.weekday()
    rows["is_weekend"] = int(target.weekday() >= 5)
    rows["is_holiday"] = int(holiday_name is not None)
    rows["neighbourhood_activity"] = _neighbourhood_activity(target)
    rows["temperature_avg"] = weather["temperature_avg"]
    rows["rainfall_mm"] = weather["rainfall_mm"]
    rows["day_sin"] = np.sin(2 * np.pi * target.dayofyear / 365.25)
    rows["day_cos"] = np.cos(2 * np.pi * target.dayofyear / 365.25)
    rows["month_sin"] = np.sin(2 * np.pi * target.month / 12)
    rows["month_cos"] = np.cos(2 * np.pi * target.month / 12)
    rows["weekday_sin"] = np.sin(2 * np.pi * target.weekday() / 7)
    rows["weekday_cos"] = np.cos(2 * np.pi * target.weekday() / 7)
    rows["trend_day"] = (target - df["date"].min()).days

    encoder_path = MODELS_DIR / "label_encoders.pkl"
    if "weather_condition_enc" in features and encoder_path.exists():
        with encoder_path.open("rb") as encoder_file:
            encoders = pickle.load(encoder_file)
        encoder = encoders.get("weather_condition")
        if encoder is not None:
            rows["weather_condition_enc"] = int(
                encoder.transform([weather["weather_condition"]])[0]
            )

    missing_features = sorted(set(features) - set(rows.columns))
    if missing_features:
        raise ValueError(
            "The feature dataset does not contain the trained model features: "
            + ", ".join(missing_features)
        )
    return rows


def _parse_recent_sales(value: str) -> list[float]:
    if not value.strip():
        return []
    try:
        sales = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("recent_sales must be comma-separated numbers") from exc
    if any(sale < 0 for sale in sales):
        raise ValueError("recent_sales cannot contain negative values")
    return sales[-14:]


def _scenario_forecast(
    product_id: str,
    target: pd.Timestamp,
    recent_sales: list[float],
    weather_condition: Optional[str],
) -> dict:
    """Run the trained model for an owner-entered what-if scenario."""
    df = _apply_current_prices(pd.read_csv(PROCESSED_DATASET, parse_dates=["date"]))
    if product_id not in set(df["product_id"].astype(str)):
        raise ValueError(f"Unknown product_id {product_id}. Choose a product from the catalogue.")

    with (MODELS_DIR / "xgboost_demand.pkl").open("rb") as model_file:
        artefact = pickle.load(model_file)
    features = artefact["features"]
    rows = _future_feature_rows(df[df["product_id"].astype(str) == product_id], target, features)

    if recent_sales:
        recent = np.asarray(recent_sales, dtype=float)
        rows["sales_lag_1"] = recent[-1]
        rows["sales_lag_7"] = recent[-7] if len(recent) >= 7 else recent.mean()
        rows["sales_lag_14"] = recent[-14] if len(recent) >= 14 else recent.mean()
        rows["sales_roll_mean_7"] = recent[-7:].mean()
        rows["sales_roll_std_7"] = recent[-7:].std(ddof=1) if len(recent[-7:]) > 1 else 0.0
        rows["sales_roll_mean_14"] = recent.mean()

    if weather_condition:
        encoder_path = MODELS_DIR / "label_encoders.pkl"
        with encoder_path.open("rb") as encoder_file:
            encoder = pickle.load(encoder_file)["weather_condition"]
        if weather_condition not in set(encoder.classes_):
            raise ValueError(f"Unsupported weather_condition: {weather_condition}")
        rows["weather_condition_enc"] = int(encoder.transform([weather_condition])[0])

    prediction = float(np.clip(artefact["model"].predict(rows[features].fillna(0))[0], 0, None))
    return {
        "forecasted_demand_per_day": round(prediction, 2),
        "forecasted_demand_7d": int(np.ceil(prediction * 7)),
        "forecast_date": target.date().isoformat(),
        "is_forecast_holiday": bool(rows["is_holiday"].iloc[0]),
        "holiday_name": _holiday_name(target),
        "school_term": _school_term(target),
        "school_event": _school_event(target),
        "expected_weather": weather_condition or _expected_weather(target)["weather_condition"],
        "forecast_model": "xgboost_demand",
    }


def _forecast_next_day(df: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Predict tomorrow per product using the trained demand model and event context."""
    target = target_date or (_current_kenya_date() + pd.Timedelta(days=1))
    model_path = MODELS_DIR / "xgboost_demand.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Demand model not found at {model_path}. "
            "Run the pipeline with run_training=true before generating a report."
        )

    with model_path.open("rb") as model_file:
        artefact = pickle.load(model_file)
    features = artefact["features"]
    last_rows = _future_feature_rows(df, target, features)
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
        "holiday_name": _holiday_name(target),
        "school_event": _school_event(target),
        "school_term": _school_term(target),
        "neighbourhood_activity": int(last_rows["neighbourhood_activity"].iloc[0]),
        "expected_weather": _expected_weather(target)["weather_condition"],
        "price_catalogue_as_of": CURRENT_PRICE_AS_OF,
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

    df = _apply_current_prices(pd.read_csv(input_path, parse_dates=["date"]))
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
        result["as_of_date"] = _current_kenya_date().date().isoformat()
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


@app.get("/replenishment/report/json")
async def get_replenishment_report_json():
    if not REPORT_PATH.exists():
        raise HTTPException(404, "Report not found. Generate one first.")
    report = pd.read_csv(REPORT_PATH)
    return to_native(report.to_dict(orient="records"))


@app.get("/replenishment/products/{product_id}")
async def get_product_recommendation(product_id: str):
    if not REPORT_PATH.exists():
        raise HTTPException(404, "Report not found. Generate one first.")
    report = pd.read_csv(REPORT_PATH)
    match = report[report["product_id"] == product_id]
    if match.empty:
        raise HTTPException(404, "Product not found in the latest report.")
    return {key: to_native(value) for key, value in match.iloc[0].to_dict().items()}


@app.post("/predict/scenario")
async def predict_scenario(
    product_id: str = Form(...),
    target_date: Optional[str] = Form(None),
    current_stock: float = Form(...),
    recent_sales: str = Form(""),
    avg_daily: Optional[float] = Form(None),
    std_daily: Optional[float] = Form(None),
    weather_condition: Optional[str] = Form(None),
):
    """Forecast an owner-entered scenario and return the complete restock decision."""
    try:
        if current_stock < 0:
            raise ValueError("current_stock cannot be negative")
        sales = _parse_recent_sales(recent_sales)
        if not sales and (avg_daily is None or avg_daily < 0):
            raise ValueError("Enter recent_sales or a non-negative avg_daily")
        if sales:
            scenario_avg = float(np.mean(sales))
            scenario_std = float(np.std(sales, ddof=1)) if len(sales) > 1 else 0.0
        else:
            scenario_avg = float(avg_daily)
            scenario_std = float(std_daily or 0.0)
        target = pd.Timestamp(target_date) if target_date else _current_kenya_date() + pd.Timedelta(days=1)
        forecast = _scenario_forecast(product_id, target, sales, weather_condition)
        product = pd.read_csv(PROCESSED_DATASET).query("product_id == @product_id").iloc[0]
        safety_stock = round(SAFETY_STOCK_FACTOR * max(scenario_std, 0.5) * np.sqrt(DEFAULT_LEAD_TIME_DAYS), 1)
        reorder_point = round(scenario_avg * DEFAULT_LEAD_TIME_DAYS + safety_stock, 1)
        reorder_needed = current_stock <= reorder_point
        order_qty = max(0, int(np.ceil(forecast["forecasted_demand_7d"] + safety_stock - current_stock))) if reorder_needed else 0
        days_left = round(current_stock / scenario_avg, 1) if scenario_avg > 0 else 0.0
        status = "OUT OF STOCK" if current_stock == 0 else "CRITICALLY LOW" if days_left < 3 else "LOW" if reorder_needed else "OK"
        priority = "URGENT" if status in {"OUT OF STOCK", "CRITICALLY LOW"} else "HIGH" if status == "LOW" and bool(product["is_perishable"]) else "MEDIUM" if status == "LOW" else "OK"
        return to_native({
            "product_id": product_id, "product_name": product["product_name"], "category": product["category"],
            "current_stock": current_stock, "avg_daily": round(scenario_avg, 2), "std_daily": round(scenario_std, 2),
            "safety_stock": safety_stock, "reorder_point": reorder_point, "days_of_stock_remaining": days_left,
            "stock_status": status, "reorder_needed": reorder_needed, "recommended_order_qty": order_qty,
            "estimated_order_cost_kes": round(order_qty * float(product["cost_price"]), 2),
            "replenishment_priority": priority, **forecast,
        })
    except (ValueError, FileNotFoundError, IndexError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc


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
    response = {"product_id": product_id, "product_name": product_name, "category": category,
            "is_perishable": bool(is_perishable), "shelf_life_days": shelf_life_days,
            "unit_price": unit_price, "cost_price": cost_price, "current_stock": current_stock,
            "avg_daily": avg_daily, "std_daily": std_daily, "forecasted_demand_per_day": forecast,
            "forecasted_demand_7d": int(np.ceil(forecast * 7)), "safety_stock": safety_stock,
            "reorder_point": reorder_point, "days_of_stock_remaining": days_left,
            "stock_status": status, "reorder_needed": reorder_needed,
            "recommended_order_qty": order_qty, "estimated_order_cost_kes": round(order_qty * cost_price, 2),
            "replenishment_priority": priority}
    return to_native(response)


if __name__ == "__main__":
    uvicorn.run("retail_demand_pulse.main:app", host="0.0.0.0", port=8000, reload=True)