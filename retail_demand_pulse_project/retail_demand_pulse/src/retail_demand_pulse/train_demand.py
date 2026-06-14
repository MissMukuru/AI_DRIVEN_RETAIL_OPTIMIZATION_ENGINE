"""
train_demand.py
---------------
6A. Demand Forecasting Models

  1. XGBoost regressor with feature importance
  2. LSTM multivariate time-series model
  3. Evaluation: MAE, RMSE, MAPE
  4. Comparison plot saved to reports/figures/

Run:
    python -m retail_demand_pulse.train_demand
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import typer
import xgboost as xgb

warnings.filterwarnings("ignore")

from retail_demand_pulse.config import (
    FIGURES_DIR,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LOOKBACK,
    LSTM_UNITS,
    MODELS_DIR,
    PROCESSED_DATASET,
    RANDOM_SEED,
    XGBOOST_PARAMS,
)

app = typer.Typer()

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoids divide-by-zero)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def print_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae_  = mean_absolute_error(y_true, y_pred)
    rmse_ = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_ = mape(y_true, y_pred)
    logger.info("  [{label}]  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%",
                label=label, mae=mae_, rmse=rmse_, mape=mape_)
    return {"model": label, "MAE": mae_, "RMSE": rmse_, "MAPE": mape_}


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

XGBOOST_FEATURES = [
    "category_enc", "is_perishable", "shelf_life_days",
    "unit_price", "cost_price", "profit_margin",
    "is_weekend", "is_holiday", "neighbourhood_activity",
    "temperature_avg", "rainfall_mm", "weather_condition_enc",
    "month", "week_of_year", "weekday_num",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "trend_day",
    "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_roll_mean_7", "sales_roll_std_7", "sales_roll_mean_14",
]
TARGET = "quantity_sold"


def _prepare_xgb_data(df: pd.DataFrame):
    """Return X, y arrays with available features (handle missing columns)."""
    available = [c for c in XGBOOST_FEATURES if c in df.columns]
    X = df[available].fillna(0).values
    y = df[TARGET].values.astype(float)
    return X, y, available


def train_xgboost(df: pd.DataFrame) -> dict:
    """Train XGBoost with time-series cross-validation."""
    logger.info("Training XGBoost demand forecasting model …")

    X, y, feature_names = _prepare_xgb_data(df)

    # Time-based train/test split (last 20% as test)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = np.clip(model.predict(X_test), 0, None)
    metrics = print_metrics("XGBoost", y_test, y_pred)

    # Save model
    model_path = MODELS_DIR / "xgboost_demand.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": feature_names}, f)
    logger.info("  XGBoost model saved to {}", model_path)

    # Feature importance plot
    _plot_feature_importance(model, feature_names)

    # Predictions vs actuals sample plot
    _plot_pred_vs_actual(y_test, y_pred, "XGBoost", "xgb_pred_vs_actual")

    return {**metrics, "y_test": y_test, "y_pred": y_pred}


def _plot_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> None:
    importance = model.feature_importances_
    fi = pd.Series(importance, index=feature_names).sort_values(ascending=True)
    fi_top = fi.tail(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    fi_top.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance (Top 20)", fontsize=13)
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    path = FIGURES_DIR / "08_xgb_feature_importance.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Feature importance plot saved → {}", path.name)


def _plot_pred_vs_actual(y_true, y_pred, label, fname) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    x_idx = np.arange(min(300, len(y_true)))
    ax.plot(x_idx, y_true[:300], label="Actual", lw=1.2, color="royalblue")
    ax.plot(x_idx, y_pred[:300], label="Predicted", lw=1.2,
            color="tomato", linestyle="--")
    ax.set_title(f"{label} – Predicted vs Actual (first 300 test samples)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Quantity Sold")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"{fname}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Pred-vs-actual plot saved → {}", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

LSTM_FEATURES = [
    "quantity_sold",
    "temperature_avg", "rainfall_mm",
    "is_weekend", "is_holiday", "neighbourhood_activity",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "sales_roll_mean_7",
]


def _build_sequences(data: np.ndarray, lookback: int):
    """
    Reshape flat 2-D array into 3-D sequences:
    X: (n_samples, lookback, n_features)
    y: (n_samples,)  ← next-step target (index 0 = quantity_sold)
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(data)):
        X_seq.append(data[i - lookback: i, :])
        y_seq.append(data[i, 0])          # quantity_sold is column 0
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def train_lstm(df: pd.DataFrame) -> dict:
    """Train an LSTM model on a single-product aggregated time series."""
    logger.info("Training LSTM demand forecasting model …")

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        logger.error("TensorFlow is not installed – skipping LSTM training.")
        return {"model": "LSTM", "MAE": None, "RMSE": None, "MAPE": None}

    # ── Aggregate to total daily store sales for LSTM (simpler + faster) ──
    daily = (
        df.groupby("date")
          .agg(
              quantity_sold        =("quantity_sold",        "sum"),
              temperature_avg      =("temperature_avg",      "mean"),
              rainfall_mm          =("rainfall_mm",          "mean"),
              is_weekend           =("is_weekend",           "max"),
              is_holiday           =("is_holiday",           "max"),
              neighbourhood_activity=("neighbourhood_activity","max"),
              sales_roll_mean_7    =("sales_roll_mean_7",    "mean"),
          )
          .reset_index()
          .sort_values("date")
    )

    daily["day_sin"]   = np.sin(2 * np.pi * daily["date"].dt.dayofyear / 365.25)
    daily["day_cos"]   = np.cos(2 * np.pi * daily["date"].dt.dayofyear / 365.25)
    daily["month_sin"] = np.sin(2 * np.pi * daily["date"].dt.month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["date"].dt.month / 12)

    feature_cols = ["quantity_sold",
                    "temperature_avg","rainfall_mm",
                    "is_weekend","is_holiday","neighbourhood_activity",
                    "day_sin","day_cos","month_sin","month_cos",
                    "sales_roll_mean_7"]

    data = daily[feature_cols].fillna(0).values.astype(np.float32)

    # Normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    lookback = LSTM_LOOKBACK
    X, y = _build_sequences(data_scaled, lookback)

    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    n_features = X.shape[2]

    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True,
             input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(LSTM_UNITS // 2),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform on quantity_sold column only
    def _inv_qty(arr_scaled):
        dummy = np.zeros((len(arr_scaled), n_features), dtype=np.float32)
        dummy[:, 0] = arr_scaled
        return scaler.inverse_transform(dummy)[:, 0]

    y_test_inv = _inv_qty(y_test)
    y_pred_inv = np.clip(_inv_qty(y_pred_scaled), 0, None)

    metrics = print_metrics("LSTM", y_test_inv, y_pred_inv)

    # Save model + scaler
    model.save(MODELS_DIR / "lstm_demand.keras")
    with open(MODELS_DIR / "lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("  LSTM model saved to {}", MODELS_DIR / "lstm_demand.keras")

    # Plot training loss
    _plot_lstm_loss(history)
    _plot_pred_vs_actual(y_test_inv, y_pred_inv, "LSTM", "lstm_pred_vs_actual")

    return {**metrics, "y_test": y_test_inv, "y_pred": y_pred_inv}


def _plot_lstm_loss(history) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history.history["loss"],     label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title("LSTM Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "09_lstm_training_loss.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  LSTM loss plot saved → {}", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(xgb_metrics: dict, lstm_metrics: dict) -> None:
    """Side-by-side bar chart comparing MAE / RMSE / MAPE."""
    results = []
    for m in [xgb_metrics, lstm_metrics]:
        if m.get("MAE") is not None:
            results.append({
                "Model": m["model"],
                "MAE":   m["MAE"],
                "RMSE":  m["RMSE"],
                "MAPE":  m["MAPE"],
            })

    if len(results) < 2:
        logger.warning("Comparison skipped — not enough models with results.")
        return

    res_df = pd.DataFrame(results).set_index("Model")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
        res_df[metric].plot(kind="bar", ax=axes[i],
                            color=["steelblue", "tomato"])
        axes[i].set_title(metric)
        axes[i].set_xticklabels(res_df.index, rotation=0)
        axes[i].set_xlabel("")
        unit = "%" if metric == "MAPE" else "units"
        axes[i].set_ylabel(unit)

    fig.suptitle("Demand Forecasting Model Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "10_model_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.success("Model comparison chart saved → {}", path.name)

# CLI entry-point

@app.command()
def main(
    input_path: Path = PROCESSED_DATASET,
) -> None:
    """Train XGBoost and LSTM demand-forecasting models and compare them."""

    logger.info("Loading feature dataset from {}", input_path)
    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    logger.info("  {:,} rows × {} columns", *df.shape)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    xgb_results  = train_xgboost(df)
    lstm_results = train_lstm(df)

    plot_model_comparison(xgb_results, lstm_results)
    logger.success("Demand forecasting training complete.")


if __name__ == "__main__":
    app()
