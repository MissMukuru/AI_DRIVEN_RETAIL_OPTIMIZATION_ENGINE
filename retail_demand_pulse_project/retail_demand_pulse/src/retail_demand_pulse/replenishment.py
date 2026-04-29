"""
replenishment.py
----------------
6D. Automatic Replenishment Engine

Rule-based replenishment logic that combines:
  - Forecasted demand (loaded from XGBoost predictions)
  - Simulated current stock levels
  - Safety stock (= SAFETY_STOCK_FACTOR × σ of recent demand)
  - Supplier lead time
  - Reorder quantity recommendation (Economic Order Quantity-inspired)

Outputs a replenishment report CSV and summary visualisations.

Run:
    python -m retail_demand_pulse.replenishment
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import typer

from retail_demand_pulse.config import (
    DEFAULT_LEAD_TIME_DAYS,
    FIGURES_DIR,
    LOW_STOCK_THRESHOLD,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATASET,
    RANDOM_SEED,
    SAFETY_STOCK_FACTOR,
)

app = typer.Typer()

rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Stock simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a current_stock level for each product based on the last
    30 days of sales data.  Stock starts at 2× peak 30-day demand and
    is depleted by daily quantity_sold.
    """
    logger.info("  Simulating current stock levels …")

    recent_cutoff = df["date"].max() - pd.Timedelta(days=30)
    recent = df[df["date"] >= recent_cutoff]

    stock_records = []
    for pid, grp in recent.groupby("product_id", observed=True):
        peak_30  = grp["quantity_sold"].sum()
        start_stock = int(peak_30 * 2)          # generous opening stock
        net_sold    = grp["quantity_sold"].sum()
        current     = max(0, start_stock - net_sold)

        # Add a bit of noise to make it realistic
        current = int(current * rng.uniform(0.7, 1.1))
        stock_records.append({"product_id": pid, "current_stock": current,
                               "peak_30_demand": peak_30})

    return pd.DataFrame(stock_records)


# ─────────────────────────────────────────────────────────────────────────────
# Demand forecasting (use saved XGBoost model)
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_next_n_days(
    df: pd.DataFrame,
    n_days: int = 7,
) -> pd.DataFrame:
    """
    Use the saved XGBoost model to forecast demand for each product
    over the next *n_days*.  Returns a dataframe with columns:
      product_id, forecasted_demand_per_day, forecasted_demand_total
    """
    model_path = MODELS_DIR / "xgboost_demand.pkl"
    if not model_path.exists():
        logger.warning(
            "XGBoost model not found at {}. "
            "Using rolling mean as fallback forecast.",
            model_path,
        )
        return _fallback_forecast(df, n_days)

    with open(model_path, "rb") as f:
        artefact = pickle.load(f)

    model    = artefact["model"]
    features = artefact["features"]

    # Use the last row per product as a proxy for "tomorrow"
    last_rows = (
        df.sort_values("date")
          .groupby("product_id", observed=True)
          .last()
          .reset_index()
    )

    available = [c for c in features if c in last_rows.columns]
    X = last_rows[available].fillna(0).values
    daily_pred = np.clip(model.predict(X), 0, None)

    return pd.DataFrame({
        "product_id":                last_rows["product_id"].values,
        "forecasted_demand_per_day": daily_pred,
        "forecasted_demand_total":   daily_pred * n_days,
    })


def _fallback_forecast(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Simple rolling-mean fallback when the ML model is missing."""
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=14)]
    summary = (
        recent.groupby("product_id", observed=True)["quantity_sold"]
              .mean()
              .reset_index()
              .rename(columns={"quantity_sold": "forecasted_demand_per_day"})
    )
    summary["forecasted_demand_total"] = summary["forecasted_demand_per_day"] * n_days
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Replenishment logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_replenishment(
    df: pd.DataFrame,
    forecast_days: int = 7,
) -> pd.DataFrame:
    """
    Core replenishment algorithm.

    For each product:

    1.  safety_stock  = SAFETY_STOCK_FACTOR × σ(daily demand, last 30 days)
    2.  demand_during_lead_time = avg_daily_demand × lead_time_days
    3.  reorder_point = demand_during_lead_time + safety_stock
    4.  If current_stock ≤ reorder_point  →  trigger reorder
    5.  reorder_qty  = forecasted_demand_total + safety_stock − current_stock

    Returns a per-product replenishment recommendation dataframe.
    """
    logger.info("Computing replenishment recommendations …")

    stock_df    = _simulate_stock(df)
    forecast_df = _forecast_next_n_days(df, n_days=forecast_days)

    # Per-product statistics from last 30 days
    recent_cutoff = df["date"].max() - pd.Timedelta(days=30)
    recent = df[df["date"] >= recent_cutoff]
    stats = (
        recent.groupby("product_id", observed=True)["quantity_sold"]
              .agg(avg_daily="mean", std_daily="std")
              .fillna(0)
              .reset_index()
    )

    # Merge product metadata
    meta = (
        df[["product_id", "product_name", "category",
            "is_perishable", "shelf_life_days", "unit_price", "cost_price"]]
          .drop_duplicates("product_id")
    )

    reco = (
        meta.merge(stock_df,    on="product_id", how="left")
            .merge(forecast_df, on="product_id", how="left")
            .merge(stats,       on="product_id", how="left")
    )

    # Lead time (days) — could be product-specific in production
    reco["lead_time_days"]         = DEFAULT_LEAD_TIME_DAYS
    reco["safety_stock"]           = (
        SAFETY_STOCK_FACTOR * reco["std_daily"] * np.sqrt(DEFAULT_LEAD_TIME_DAYS)
    ).clip(lower=1).round(1)

    reco["demand_during_lead_time"] = (
        reco["avg_daily"] * DEFAULT_LEAD_TIME_DAYS
    ).round(1)

    reco["reorder_point"] = (
        reco["demand_during_lead_time"] + reco["safety_stock"]
    ).round(1)

    reco["stock_status"] = reco.apply(
        lambda r: _classify_stock(r["current_stock"],
                                   r["reorder_point"],
                                   r["peak_30_demand"]),
        axis=1,
    )

    reco["reorder_needed"] = (
        reco["current_stock"] <= reco["reorder_point"]
    )

    reco["reorder_qty"] = reco.apply(
        lambda r: max(0, int(
            r["forecasted_demand_total"] + r["safety_stock"] - r["current_stock"]
        )) if r["reorder_needed"] else 0,
        axis=1,
    )

    reco["estimated_reorder_cost"] = (
        reco["reorder_qty"] * reco["cost_price"]
    ).round(2)

    reco["days_of_stock_left"] = (
        reco["current_stock"] / reco["avg_daily"].replace(0, np.nan)
    ).fillna(999).round(1)

    # Priority: perishables with low stock get highest priority
    reco["replenishment_priority"] = reco.apply(_priority, axis=1)

    reco = reco.sort_values(
        ["replenishment_priority", "days_of_stock_left"]
    ).reset_index(drop=True)

    return reco


def _classify_stock(current, reorder_point, peak) -> str:
    if current == 0:
        return "OUT OF STOCK"
    ratio = current / max(peak, 1)
    if ratio < LOW_STOCK_THRESHOLD:
        return "CRITICALLY LOW"
    if current <= reorder_point:
        return "LOW"
    return "OK"


def _priority(row) -> int:
    """
    1 = urgent (perishable + out of stock / critically low)
    2 = high
    3 = medium
    4 = low
    """
    if row["stock_status"] == "OUT OF STOCK":
        return 1
    if row["stock_status"] == "CRITICALLY LOW":
        return 1 if row["is_perishable"] else 2
    if row["stock_status"] == "LOW":
        return 2 if row["is_perishable"] else 3
    return 4


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_replenishment_dashboard(reco: pd.DataFrame) -> None:
    """Visual overview of the replenishment state."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Stock status breakdown (pie)
    status_counts = reco["stock_status"].value_counts()
    colors = {
        "OUT OF STOCK":   "#d62728",
        "CRITICALLY LOW": "#ff7f0e",
        "LOW":            "#ffdd57",
        "OK":             "#2ca02c",
    }
    pie_colors = [colors.get(s, "gray") for s in status_counts.index]
    axes[0, 0].pie(status_counts, labels=status_counts.index,
                   autopct="%1.0f%%", colors=pie_colors, startangle=90)
    axes[0, 0].set_title("Current Stock Status Distribution")

    # 2. Days of stock remaining (horizontal bar, top 15 urgent)
    urgent = reco[reco["reorder_needed"]].nsmallest(15, "days_of_stock_left")
    bar_colors = [colors.get(s, "gray") for s in urgent["stock_status"]]
    axes[0, 1].barh(urgent["product_name"], urgent["days_of_stock_left"],
                    color=bar_colors)
    axes[0, 1].axvline(DEFAULT_LEAD_TIME_DAYS, color="red",
                       linestyle="--", label=f"Lead time ({DEFAULT_LEAD_TIME_DAYS}d)")
    axes[0, 1].set_title("Days of Stock Left (Top 15 Urgent)")
    axes[0, 1].set_xlabel("Days")
    axes[0, 1].legend()

    # 3. Reorder quantity by product
    to_order = reco[reco["reorder_qty"] > 0].nlargest(15, "reorder_qty")
    axes[1, 0].barh(to_order["product_name"], to_order["reorder_qty"],
                    color="steelblue")
    axes[1, 0].set_title("Recommended Reorder Quantities")
    axes[1, 0].set_xlabel("Units to Order")

    # 4. Estimated reorder cost by category
    cat_cost = (
        reco.groupby("category", observed=True)["estimated_reorder_cost"]
            .sum()
            .sort_values(ascending=False)
    )
    axes[1, 1].bar(cat_cost.index, cat_cost.values, color="coral")
    axes[1, 1].set_title("Estimated Reorder Cost by Category (KES)")
    axes[1, 1].set_xlabel("Category")
    axes[1, 1].set_ylabel("Cost (KES)")
    axes[1, 1].tick_params(axis="x", rotation=35)

    fig.suptitle("Automatic Replenishment Dashboard – Eldoret Shop",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "19_replenishment_dashboard.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Replenishment dashboard saved → {}", path.name)


def plot_stock_vs_reorder(reco: pd.DataFrame) -> None:
    """Scatter: current_stock vs reorder_point for all products."""
    fig, ax = plt.subplots(figsize=(9, 7))

    c_map = {
        "OUT OF STOCK":   "#d62728",
        "CRITICALLY LOW": "#ff7f0e",
        "LOW":            "#f0c040",
        "OK":             "#2ca02c",
    }

    for status, grp in reco.groupby("stock_status"):
        ax.scatter(grp["reorder_point"], grp["current_stock"],
                   label=status, color=c_map.get(status, "gray"),
                   s=80, alpha=0.85)
        for _, row in grp.iterrows():
            ax.annotate(row["product_name"][:12],
                        (row["reorder_point"], row["current_stock"]),
                        fontsize=6, alpha=0.7)

    lims = [0, max(reco["current_stock"].max(), reco["reorder_point"].max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="Stock = Reorder Point")
    ax.set_xlabel("Reorder Point")
    ax.set_ylabel("Current Stock")
    ax.set_title("Current Stock vs Reorder Point")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = FIGURES_DIR / "20_stock_vs_reorder_point.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Stock-vs-reorder scatter saved → {}", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    input_path:  Path = PROCESSED_DATASET,
    output_path: Path = PROCESSED_DATA_DIR / "replenishment_report.csv",
    forecast_days: int = 7,
) -> None:
    """Generate and save the automatic replenishment report."""

    logger.info("Loading feature dataset from {}", input_path)
    df = pd.read_csv(input_path, parse_dates=["date"])
    logger.info("  {:,} rows × {} columns", *df.shape)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    reco = compute_replenishment(df, forecast_days=forecast_days)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reco.to_csv(output_path, index=False)
    logger.info("Replenishment report saved to {}", output_path)

    plot_replenishment_dashboard(reco)
    plot_stock_vs_reorder(reco)

    # Print urgent items
    urgent = reco[reco["replenishment_priority"] <= 2]
    logger.warning(
        "\n⚠  {} products need urgent replenishment:\n{}",
        len(urgent),
        urgent[["product_name", "category", "current_stock",
                "reorder_point", "reorder_qty", "days_of_stock_left",
                "stock_status"]].to_string(index=False),
    )

    total_cost = reco["estimated_reorder_cost"].sum()
    logger.success(
        "Replenishment engine complete. "
        "Total estimated reorder cost: KES {:,.0f}",
        total_cost,
    )


if __name__ == "__main__":
    app()
