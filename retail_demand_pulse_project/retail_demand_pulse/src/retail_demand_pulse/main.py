"""
Retail Demand Pulse - COMPLETE & FIXED FastAPI Backend
Supports CSV + Excel uploads (Windows-friendly)
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys
import importlib
import tempfile
import shutil
import uvicorn
from loguru import logger
import numpy as np
import inspect
from typing import Optional
import pandas as pd

# ====================== PROJECT SETUP ======================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

static_dir = PROJECT_ROOT / "src" / "retail_demand_pulse" / "static"
static_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Retail Demand Pulse",
    description="AI-Driven Retail Optimization Engine",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ====================== UTILITIES ======================
def to_native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def run_module(module_name: str, input_path: Optional[Path] = None, **kwargs):
    try:
        module = importlib.import_module(f"retail_demand_pulse.{module_name}")
        main_func = getattr(module, "main")
        sig = inspect.signature(main_func)
        if input_path and "input_path" in sig.parameters:
            return main_func(input_path=input_path, **kwargs)
        return main_func(**kwargs)
    except Exception as e:
        error_msg = str(e)
        if "Feature shape mismatch" in error_msg or "expected" in error_msg.lower():
            error_msg = "Feature mismatch: Please use /upload-and-process first."
        logger.error(f"Error running {module_name}: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

def convert_to_csv(input_path: Path) -> Path:
    """Convert Excel to CSV if needed"""
    if input_path.suffix.lower() in ['.xlsx', '.xls']:
        try:
            df = pd.read_excel(input_path)
            csv_path = input_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Converted {input_path.name} to CSV")
            return csv_path
        except Exception as e:
            logger.warning(f"Excel conversion failed: {e}. Trying original file.")
            return input_path
    return input_path

def process_dataset(input_path: Path) -> Path:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_path = processed_dir / "processed_dataset.csv"

    logger.info("🔄 Auto-processing dataset...")
    
    # Convert Excel to CSV if necessary
    input_path = convert_to_csv(input_path)

    try:
        from retail_demand_pulse.clean import main as clean_main
        clean_main(input_path=input_path)
    except Exception as e:
        logger.warning(f"Clean stage skipped: {e}")

    try:
        from retail_demand_pulse.features import main as features_main
        features_main(input_path=input_path, output_path=final_path)
        logger.success("✅ Dataset processed successfully")
        return final_path
    except Exception as e:
        logger.warning(f"Features stage failed: {e}. Using original file.")
        shutil.copy(input_path, final_path)
        return final_path

# ====================== ROUTES ======================

@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Welcome to Retail Demand Pulse</h1>")

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ------------------- Upload + Process -------------------
@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    run_training: bool = Form(True),
    generate_replenishment: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(400, "Only CSV, XLSX, or XLS files allowed")

    # Create temp file with original extension
    suffix = Path(file.filename).suffix.lower()
    temp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        processed_path = process_dataset(temp_path)
        results = {"status": "success", "filename": file.filename}

        if run_training:
            if background_tasks:
                background_tasks.add_task(run_module, "train_demand", input_path=processed_path)
                background_tasks.add_task(run_module, "train_spoilage", input_path=processed_path)
                results["training"] = "started"
            else:
                run_module("train_demand", input_path=processed_path)
                run_module("train_spoilage", input_path=processed_path)
                results["training"] = "completed"

        if generate_replenishment:
            if background_tasks:
                background_tasks.add_task(run_module, "replenishment", input_path=processed_path)
                results["replenishment"] = "started"
            else:
                run_module("replenishment", input_path=processed_path)
                results["replenishment"] = "completed"

        return results
    finally:
        temp_path.unlink(missing_ok=True)
        # Clean up converted CSV if created
        if temp_path.suffix.lower() in ['.xlsx', '.xls']:
            converted = temp_path.with_suffix('.csv')
            converted.unlink(missing_ok=True)

# ------------------- Replenishment Report -------------------
@app.get("/replenishment/report")
async def get_replenishment_report():
    report_path = PROJECT_ROOT / "data" / "processed" / "replenishment_report.csv"
    if report_path.exists():
        return FileResponse(report_path, filename="replenishment_report.csv")
    raise HTTPException(404, "Report not found. Generate one first.")

# ------------------- FIXED Single Product Predictor -------------------
@app.post("/predict/replenishment")
async def predict_single_replenishment(
    product_name: str = Form(...),
    category: str = Form(...),
    is_perishable: int = Form(0),
    current_stock: int = Form(...),
    avg_daily: float = Form(...),
    forecasted_demand_per_day: float = Form(...),
    # Optional fields with defaults
    product_id: str = Form("P999"),
    shelf_life_days: int = Form(730),
    unit_price: float = Form(110.0),
    cost_price: float = Form(78.0),
    std_daily: float = Form(3.0),
):
    try:
        from retail_demand_pulse.config import DEFAULT_LEAD_TIME_DAYS, SAFETY_STOCK_FACTOR

        lead_time = DEFAULT_LEAD_TIME_DAYS
        safety_stock = round(SAFETY_STOCK_FACTOR * max(std_daily, 0.5) * np.sqrt(lead_time), 1)
        demand_during_lead = round(avg_daily * lead_time, 1)
        reorder_point = round(demand_during_lead + safety_stock, 1)

        forecasted_7d = round(forecasted_demand_per_day * 7)
        reorder_needed = bool(current_stock <= reorder_point)

        recommended_qty = max(0, int(np.ceil(forecasted_7d + safety_stock - current_stock))) if reorder_needed else 0
        estimated_cost = round(recommended_qty * cost_price, 2)
        days_left = round(current_stock / avg_daily, 1) if avg_daily > 0 else 0.0

        if current_stock == 0:
            stock_status = "OUT OF STOCK"
        elif days_left < 3:
            stock_status = "CRITICALLY LOW"
        elif current_stock <= reorder_point:
            stock_status = "LOW"
        else:
            stock_status = "OK"

        if stock_status in ["OUT OF STOCK", "CRITICALLY LOW"]:
            priority_label = "🔴 URGENT"
        elif stock_status == "LOW":
            priority_label = "🟠 HIGH" if is_perishable else "🟡 MEDIUM"
        else:
            priority_label = "🟢 OK"

        result = {
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
            "is_perishable": bool(is_perishable),
            "shelf_life_days": int(shelf_life_days),
            "unit_price": float(unit_price),
            "cost_price": float(cost_price),
            "current_stock": int(current_stock),
            "avg_daily": float(round(avg_daily, 4)),
            "std_daily": float(round(std_daily, 4)),
            "forecasted_demand_per_day": float(round(forecasted_demand_per_day, 4)),
            "forecasted_demand_7d": int(forecasted_7d),
            "safety_stock": float(safety_stock),
            "reorder_point": float(reorder_point),
            "days_of_stock_remaining": float(days_left),
            "stock_status": stock_status,
            "reorder_needed": reorder_needed,
            "recommended_order_qty": int(recommended_qty),
            "estimated_order_cost_kes": float(estimated_cost),
            "replenishment_priority": priority_label,
        }

        result = {k: to_native(v) for k, v in result.items()}
        return result

    except Exception as e:
        logger.error(f"Predict error: {e}")
        raise HTTPException(500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("retail_demand_pulse.main:app", host="0.0.0.0", port=8000, reload=True)