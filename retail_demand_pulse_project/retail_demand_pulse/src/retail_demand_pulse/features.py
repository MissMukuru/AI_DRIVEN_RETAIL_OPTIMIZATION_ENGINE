"""
features.py
-----------
C + D. Feature Engineering, Encoding & Scaling

Adds:
  - Lag features:  sales_lag_1, sales_lag_7, sales_lag_14
  - Rolling stats: sales_roll_mean_7, sales_roll_std_7
  - Holiday encoding
  - Label-encoded & one-hot categorical columns
  - Scaled numerical columns (StandardScaler saved for later reuse)
  - Additional derived features: day_sin/cos, trend, etc.

Run:
    python -m retail_demand_pulse.features
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import typer

from retail_demand_pulse.config import (
    INTERIM_DATASET,
    MODELS_DIR,
    PROCESSED_DATASET,
)

app = typer.Typer()


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-product lag features on quantity_sold.
    Sorted by [product_id, date] before computing to avoid leakage.
    """
    logger.info("  Adding lag features …")
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    for lag in tqdm([1, 7, 14], desc="  Lag features"):
        df[f"sales_lag_{lag}"] = (
            df.groupby("product_id", observed=True)["quantity_sold"]
              .shift(lag)
        )
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std over a 7-day window per product."""
    logger.info("  Adding rolling statistics …")

    grp = df.groupby("product_id", observed=True)["quantity_sold"]

    df["sales_roll_mean_7"] = grp.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["sales_roll_std_7"] = grp.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).std().fillna(0)
    )
    df["sales_roll_mean_14"] = grp.transform(
        lambda s: s.shift(1).rolling(14, min_periods=1).mean()
    )
    return df


def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode day-of-year and week-of-year as sin/cos pairs."""
    logger.info("  Adding cyclical time features …")

    df["day_of_year"]    = df["date"].dt.dayofyear
    df["day_sin"]        = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"]        = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)

    df["weekday_num"]    = df["date"].dt.weekday   # 0=Mon … 6=Sun
    df["weekday_sin"]    = np.sin(2 * np.pi * df["weekday_num"] / 7)
    df["weekday_cos"]    = np.cos(2 * np.pi * df["weekday_num"] / 7)
    return df


def _add_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Integer day-index from start of dataset as a simple trend proxy."""
    start = df["date"].min()
    df["trend_day"] = (df["date"] - start).dt.days
    return df


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Margin and price-tier encoding."""
    logger.info("  Adding price & margin features …")
    # profit_margin may already exist; recompute to be safe
    df["profit_margin"] = (
        (df["unit_price"] - df["cost_price"]) / df["unit_price"]
    ).round(4)

    # Revenue-per-unit (same as unit_price here, but kept explicit)
    df["revenue_per_unit"] = df["unit_price"]
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables:
      - Label encode:  category, weather_condition, day_of_week
      - Binary encode: is_perishable, is_weekend, is_holiday (already bool → int)
      - product_id left as-is (used as groupby key)
    Encoders are saved for inference use.
    """
    logger.info("  Encoding categorical features …")

    encoders = {}

    for col in ["category", "weather_condition", "day_of_week"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("    {} → {}_enc  ({} classes)", col, col, len(le.classes_))

    # Boolean → int
    for col in ["is_perishable", "is_weekend", "is_holiday"]:
        df[col] = df[col].astype(int)

    # Save encoders
    enc_path = MODELS_DIR / "label_encoders.pkl"
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)
    logger.info("  Label encoders saved to {}", enc_path)

    return df


def _scale_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    StandardScaler on selected numeric features.
    Scaler is saved for inference use.
    Scaled columns are suffixed with `_scaled`.
    """
    logger.info("  Scaling numeric features …")

    scale_cols = [
        "temperature_avg", "rainfall_mm",
        "unit_price", "cost_price",
        "profit_margin", "sales_roll_mean_7", "sales_roll_std_7",
        "sales_roll_mean_14", "trend_day",
        "sales_lag_1", "sales_lag_7", "sales_lag_14",
    ]
    # Only scale columns that actually have values (lag cols have NaNs at start)
    df_tmp = df[scale_cols].fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_tmp)

    for i, col in enumerate(scale_cols):
        df[f"{col}_scaled"] = scaled[:, i]

    scaler_path = MODELS_DIR / "standard_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "columns": scale_cols}, f)
    logger.info("  StandardScaler saved to {}", scaler_path)

    return df


def _fill_lag_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN lag/rolling values created at the start of each product's series.
    Strategy: backfill within product, then fill remaining with 0.
    """
    lag_roll_cols = [c for c in df.columns if
                     c.startswith("sales_lag_") or c.startswith("sales_roll_")]
    for col in lag_roll_cols:
        df[col] = (
            df.groupby("product_id", observed=True)[col]
              .transform(lambda s: s.bfill().fillna(0))
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    input_path:  Path = INTERIM_DATASET,
    output_path: Path = PROCESSED_DATASET,
) -> None:
    """Run the full feature-engineering and encoding pipeline."""

    logger.info("Loading cleaned dataset from {}", input_path)
    df = pd.read_csv(input_path, parse_dates=["date"])
    logger.info("  {:,} rows × {} columns", *df.shape)

    # ── Feature engineering ───────────────────────────────────────────────
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_cyclical_features(df)
    df = _add_trend_feature(df)
    df = _add_price_features(df)
    df = _fill_lag_nans(df)

    # ── Encoding & scaling ────────────────────────────────────────────────
    df = _encode_categoricals(df)
    df = _scale_numerics(df)

    # ── Final sort & save ─────────────────────────────────────────────────
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(
        "Feature dataset saved to {} — {:,} rows × {} columns",
        output_path, len(df), len(df.columns),
    )

    # Print final feature list
    logger.info("Final feature columns:\n  {}",
                "\n  ".join(df.columns.tolist()))


if __name__ == "__main__":
    app()
