"""
clean.py
--------
A. Data Cleaning Pipeline
  - Correct data types
  - Handle missing values
  - Detect & handle outliers in `quantity_sold` and `total_amount`

Run:
    python -m retail_demand_pulse.clean
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from retail_demand_pulse.config import (
    INTERIM_DATASET,
    RAW_DATASET,
)

app = typer.Typer()


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all columns to their correct types."""
    logger.info("  Fixing data types …")

    df["date"]          = pd.to_datetime(df["date"])
    df["is_perishable"] = df["is_perishable"].astype(bool)
    df["is_weekend"]    = df["is_weekend"].astype(bool)
    df["is_holiday"]    = df["is_holiday"].astype(bool)

    int_cols = ["quantity_sold", "shelf_life_days", "month", "year",
                "week_of_year", "neighbourhood_activity"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    float_cols = ["unit_price", "cost_price", "total_amount",
                  "temperature_avg", "rainfall_mm",
                  "profit_margin", "spoilage_risk_score"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cat_cols = ["product_id", "product_name", "category",
                "day_of_week", "weather_condition"]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values.
    Strategy:
      - quantity_sold / total_amount: forward-fill per product then fill 0
      - weather features: interpolate
      - Any remaining: drop
    """
    logger.info("  Handling missing values …")
    before = df.isnull().sum().sum()

    # Sort to make forward-fill meaningful
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    for col in ["quantity_sold", "total_amount"]:
        df[col] = (
            df.groupby("product_id", observed=True)[col]
              .transform(lambda s: s.ffill().bfill().fillna(0))
        )

    for col in ["temperature_avg", "rainfall_mm"]:
        df[col] = df[col].interpolate(method="linear").bfill().ffill()

    df["spoilage_risk_score"] = df["spoilage_risk_score"].fillna(0.0)

    after = df.isnull().sum().sum()
    logger.info("  Missing values: {} → {}", before, after)

    # Drop any rows still containing NaNs (should be 0)
    df = df.dropna()
    return df


def _detect_outliers_iqr(series: pd.Series, factor: float = 3.0):
    """Return a boolean mask of outliers using the IQR method."""
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


def _handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme outliers in `quantity_sold` and `total_amount` per product.
    Uses a 3×IQR fence; values beyond the fence are winsorized to the fence.
    Also creates a boolean flag column for downstream modelling.
    """
    logger.info("  Detecting & handling outliers …")
    df["is_quantity_outlier"]     = False
    df["is_total_amount_outlier"] = False

    for pid, grp in tqdm(df.groupby("product_id", observed=True),
                         desc="  Outlier scan"):
        idx = grp.index

        for col, flag_col in [
            ("quantity_sold",  "is_quantity_outlier"),
            ("total_amount",   "is_total_amount_outlier"),
        ]:
            mask     = _detect_outliers_iqr(grp[col])
            outlier_idx = idx[mask]

            if outlier_idx.any():
                df.loc[outlier_idx, flag_col] = True
                q1  = grp[col].quantile(0.25)
                q3  = grp[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                # Winsorize (convert to float to avoid Int64 dtype clash)
                df.loc[idx, col] = (
                    grp[col].astype(float).clip(lower=lower, upper=upper).round(0).astype(int)
                )

    n_qty  = df["is_quantity_outlier"].sum()
    n_amt  = df["is_total_amount_outlier"].sum()
    logger.info("  Outliers capped — qty: {}, total_amount: {}", n_qty, n_amt)
    return df


def _recompute_total_amount(df: pd.DataFrame) -> pd.DataFrame:
    """After outlier capping, ensure total_amount = quantity_sold × unit_price."""
    df["total_amount"] = (
        df["quantity_sold"].astype(float) * df["unit_price"]
    ).round(2)
    return df


def _add_cleaning_report(df: pd.DataFrame) -> None:
    """Print a concise cleaning summary."""
    logger.info("─── Cleaning Report ───────────────────────────────────────")
    logger.info("  Shape      : {} rows × {} columns", *df.shape)
    logger.info("  Date range : {} → {}", df["date"].min().date(),
                df["date"].max().date())
    logger.info("  Products   : {}", df["product_id"].nunique())
    logger.info("  Nulls      : {}", df.isnull().sum().sum())
    logger.info("  Qty outliers flagged   : {}", df["is_quantity_outlier"].sum())
    logger.info("  Amt outliers flagged   : {}", df["is_total_amount_outlier"].sum())
    logger.info("───────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    input_path:  Path = RAW_DATASET,
    output_path: Path = INTERIM_DATASET,
) -> None:
    """Run the full data-cleaning pipeline on the raw retail dataset."""

    logger.info("Loading raw dataset from {}", input_path)
    df = pd.read_csv(input_path)
    logger.info("  Loaded {:,} rows × {} columns", *df.shape)

    df = _fix_dtypes(df)
    df = _handle_missing(df)
    df = _handle_outliers(df)
    df = _recompute_total_amount(df)

    _add_cleaning_report(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success("Cleaned dataset saved to {}", output_path)


if __name__ == "__main__":
    app()
