"""
replenishment.py
----------------
6D. Automatic Replenishment Engine

Stock simulation reflects an END-OF-WEEK scenario: the shop has been
selling for several days since its last delivery, so shelf levels are
realistically depleted and many products genuinely need reordering.

Key fix over previous version
------------------------------
Old logic:  current_stock = 2 × peak_30_demand  (far too generous — nothing
            ever triggered a reorder)

New logic:  simulate a 7-day selling week day-by-day, starting from a
            reasonable Monday opening stock, then read the Friday closing
            balance.  This naturally produces a mix of:
              • OK products (slow movers, still have buffer)
              • LOW / CRITICALLY LOW (fast movers, sold down mid-week)
              • OUT OF STOCK (very fast perishables)

Run:
    python -m retail_demand_pulse.replenishment
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

app  = typer.Typer()
rng  = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Realistic end-of-week stock simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_end_of_week_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a 7-day selling week (Mon → Sun) per product and return the
    closing stock on the last day.

    Algorithm
    ---------
    1.  Compute each product's average daily demand and std from the most
        recent 30 days of actual data.
    2.  Monday opening stock  =  avg_daily × delivery_cycle_days × restock_factor
        where delivery_cycle_days = 7 (weekly delivery) and restock_factor
        is drawn from U(0.85, 1.10) to add realistic variability.
        → For a product selling ~15 units/day this gives ~105 units on Monday.
    3.  Each of the 7 simulated days consumes Poisson(avg_daily) units, but
        stock cannot go negative (stockout is floored at 0).
    4.  Friday closing balance is returned as current_stock.

    This naturally creates:
      • Fast-selling products (milk, bread, maize flour) that drop to low
        or zero stock by mid-week.
      • Slow-selling products (soap, matches) that still have buffer left.
    """
    logger.info("Simulating end-of-week stock levels …")

    recent_cutoff = df["date"].max() - pd.Timedelta(days=30)
    recent = df[df["date"] >= recent_cutoff]

    product_meta = (
        df[["product_id","product_name","category",
            "is_perishable","shelf_life_days","unit_price","cost_price"]]
          .drop_duplicates("product_id")
    )

    stats = (
        recent.groupby("product_id", observed=True)["quantity_sold"]
              .agg(avg_daily="mean", std_daily="std")
              .fillna(0)
              .reset_index()
    )

    records = []
    for _, row in stats.iterrows():
        pid        = row["product_id"]
        avg_daily  = max(row["avg_daily"], 0.5)   # floor at 0.5 to avoid 0-demand edge cases
        std_daily  = row["std_daily"]

        # Monday opening stock: enough for one delivery cycle (7 days),
        # scaled by a random restock factor so some products start slightly
        # over- or under-stocked.
        restock_factor   = float(rng.uniform(0.75, 1.45))
        delivery_cycle   = 10  # 10-day supply gives realistic spread
        opening_stock    = int(avg_daily * delivery_cycle * restock_factor)

        # Simulate 7 days of sales (Mon=0 … Sun=6)
        stock = opening_stock
        for day in range(7):
            # Weekend gets a ~30 % demand bump (mirrors dataset pattern)
            day_mult   = 1.30 if day >= 5 else 1.0
            daily_sold = int(rng.poisson(avg_daily * day_mult))
            stock      = max(0, stock - daily_sold)

        records.append({
            "product_id":       pid,
            "opening_stock":    opening_stock,
            "current_stock":    stock,          # end-of-week closing balance
            "avg_daily":        round(avg_daily, 2),
            "std_daily":        round(std_daily, 2),
        })

    stock_df = pd.DataFrame(records)
    result   = product_meta.merge(stock_df, on="product_id", how="left")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  7-day demand forecast from saved XGBoost model
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_next_7_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the saved XGBoost model to predict demand for the next 7 days per
    product, then return total and daily averages.
    Falls back to 7-day rolling mean if the model file is missing.
    """
    model_path = MODELS_DIR / "xgboost_demand.pkl"

    if not model_path.exists():
        logger.warning("XGBoost model not found — using rolling-mean fallback.")
        return _rollmean_forecast(df, n_days=7)

    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model    = artefact["model"]
    features = artefact["features"]

    # Use each product's most recent row as the feature vector for "tomorrow"
    last_rows = (
        df.sort_values("date")
          .groupby("product_id", observed=True)
          .last()
          .reset_index()
    )
    available = [c for c in features if c in last_rows.columns]
    X         = last_rows[available].fillna(0).values
    daily_pred = np.clip(model.predict(X), 0, None)

    return pd.DataFrame({
        "product_id":                last_rows["product_id"].values,
        "forecasted_demand_per_day": daily_pred.round(2),
        "forecasted_demand_7d":      (daily_pred * 7).round(0).astype(int),
    })


def _rollmean_forecast(df: pd.DataFrame, n_days: int = 7) -> pd.DataFrame:
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=14)]
    s = (
        recent.groupby("product_id", observed=True)["quantity_sold"]
              .mean()
              .reset_index()
              .rename(columns={"quantity_sold": "forecasted_demand_per_day"})
    )
    s["forecasted_demand_7d"] = (s["forecasted_demand_per_day"] * n_days).round(0).astype(int)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Core replenishment logic
# ─────────────────────────────────────────────────────────────────────────────

def _classify_stock(current: int, reorder_point: float, opening: int) -> str:
    if current == 0:
        return "OUT OF STOCK"
    ratio = current / max(opening, 1)
    if ratio < LOW_STOCK_THRESHOLD:
        return "CRITICALLY LOW"
    if current <= reorder_point:
        return "LOW"
    return "OK"


def _priority(row) -> int:
    """1 = most urgent → 4 = no action needed."""
    if row["stock_status"] == "OUT OF STOCK":
        return 1
    if row["stock_status"] == "CRITICALLY LOW":
        return 1 if row["is_perishable"] else 2
    if row["stock_status"] == "LOW":
        return 2 if row["is_perishable"] else 3
    return 4


def compute_replenishment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full replenishment recommendation table.

    Columns returned (clean, actionable set):
        product_id, product_name, category, is_perishable, shelf_life_days,
        unit_price, cost_price,
        opening_stock, current_stock, avg_daily_sales, std_daily_sales,
        forecasted_demand_per_day, forecasted_demand_7d,
        safety_stock, reorder_point, days_of_stock_remaining,
        stock_status, reorder_needed, recommended_order_qty,
        estimated_order_cost_kes, replenishment_priority, priority_label
    """
    logger.info("Computing replenishment recommendations …")

    stock_df    = _simulate_end_of_week_stock(df)
    forecast_df = _forecast_next_7_days(df)

    reco = stock_df.merge(forecast_df, on="product_id", how="left")

    # ── Safety stock & reorder point ────────────────────────────────────────
    reco["lead_time_days"] = DEFAULT_LEAD_TIME_DAYS

    reco["safety_stock"] = (
        SAFETY_STOCK_FACTOR
        * reco["std_daily"].clip(lower=0)
        * np.sqrt(DEFAULT_LEAD_TIME_DAYS)
    ).clip(lower=1).round(1)

    reco["demand_during_lead_time"] = (
        reco["avg_daily"] * DEFAULT_LEAD_TIME_DAYS
    ).round(1)

    reco["reorder_point"] = (
        reco["demand_during_lead_time"] + reco["safety_stock"]
    ).round(1)

    # ── Stock status ─────────────────────────────────────────────────────────
    reco["stock_status"] = reco.apply(
        lambda r: _classify_stock(
            int(r["current_stock"]),
            r["reorder_point"],
            int(r["opening_stock"]),
        ),
        axis=1,
    )

    reco["reorder_needed"] = reco["current_stock"] <= reco["reorder_point"]

    # ── Reorder quantity  ────────────────────────────────────────────────────
    # Order enough to cover the next 7-day forecast PLUS safety stock,
    # minus whatever stock is left on the shelf.
    reco["recommended_order_qty"] = reco.apply(
        lambda r: max(
            0,
            int(np.ceil(
                r["forecasted_demand_7d"] + r["safety_stock"] - r["current_stock"]
            ))
        ) if r["reorder_needed"] else 0,
        axis=1,
    )

    reco["estimated_order_cost_kes"] = (
        reco["recommended_order_qty"] * reco["cost_price"]
    ).round(2)

    # ── Days of stock remaining ──────────────────────────────────────────────
    reco["days_of_stock_remaining"] = (
        reco["current_stock"] / reco["avg_daily"].replace(0, np.nan)
    ).fillna(0).round(1)

    # ── Priority ─────────────────────────────────────────────────────────────
    reco["replenishment_priority"] = reco.apply(_priority, axis=1)
    priority_map = {1: "🔴 URGENT", 2: "🟠 HIGH", 3: "🟡 MEDIUM", 4: "🟢 OK"}
    reco["priority_label"] = reco["replenishment_priority"].map(priority_map)

    # ── Clean output columns (no intermediate noise) ─────────────────────────
    output_cols = [
        "product_id", "product_name", "category",
        "is_perishable", "shelf_life_days",
        "unit_price", "cost_price",
        "opening_stock", "current_stock",
        "avg_daily", "std_daily",
        "forecasted_demand_per_day", "forecasted_demand_7d",
        "safety_stock", "reorder_point",
        "days_of_stock_remaining",
        "stock_status", "reorder_needed",
        "recommended_order_qty", "estimated_order_cost_kes",
        "replenishment_priority", "priority_label",
    ]
    reco = (
        reco[output_cols]
          .sort_values(["replenishment_priority", "days_of_stock_remaining"])
          .reset_index(drop=True)
    )

    # ── Summary log ──────────────────────────────────────────────────────────
    n_reorder  = reco["reorder_needed"].sum()
    total_cost = reco["estimated_order_cost_kes"].sum()
    by_status  = reco["stock_status"].value_counts().to_dict()

    logger.info("  Stock status breakdown: {}", by_status)
    logger.info("  Products needing reorder : {} / {}", n_reorder, len(reco))
    logger.success(
        "  Total estimated order cost : KES {:,.0f}", total_cost
    )

    return reco


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_replenishment_dashboard(reco: pd.DataFrame) -> None:
    """4-panel dashboard: stock status pie, days-of-stock bar,
       order quantities, cost by category."""

    status_colors = {
        "OUT OF STOCK":   "#d62728",
        "CRITICALLY LOW": "#ff7f0e",
        "LOW":            "#f0c040",
        "OK":             "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: Stock status pie ────────────────────────────────────────
    counts = reco["stock_status"].value_counts()
    pie_colors = [status_colors.get(s, "grey") for s in counts.index]
    axes[0, 0].pie(
        counts, labels=counts.index, autopct="%1.0f%%",
        colors=pie_colors, startangle=90,
        textprops={"fontsize": 10},
    )
    axes[0, 0].set_title("Current Stock Status (End of Week)", fontsize=12, fontweight="bold")

    # ── Panel 2: Days of stock remaining (products that need reorder) ────
    urgent = reco[reco["reorder_needed"]].nsmallest(15, "days_of_stock_remaining")
    bar_colors = [status_colors.get(s, "grey") for s in urgent["stock_status"]]
    axes[0, 1].barh(urgent["product_name"], urgent["days_of_stock_remaining"],
                    color=bar_colors, edgecolor="white")
    axes[0, 1].axvline(
        DEFAULT_LEAD_TIME_DAYS, color="red",
        linestyle="--", linewidth=1.5,
        label=f"Lead time = {DEFAULT_LEAD_TIME_DAYS}d",
    )
    axes[0, 1].set_title("Days of Stock Left — Products Needing Reorder",
                         fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Days")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].invert_yaxis()

    # ── Panel 3: Recommended order quantities ────────────────────────────
    to_order = reco[reco["recommended_order_qty"] > 0].nlargest(15, "recommended_order_qty")
    bar2_colors = [status_colors.get(s, "steelblue") for s in to_order["stock_status"]]
    axes[1, 0].barh(to_order["product_name"], to_order["recommended_order_qty"],
                    color=bar2_colors, edgecolor="white")
    axes[1, 0].set_title("Recommended Order Quantities (Units)",
                         fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Units to Order")
    axes[1, 0].invert_yaxis()

    # ── Panel 4: Estimated reorder cost by category ───────────────────────
    cat_cost = (
        reco.groupby("category", observed=True)["estimated_order_cost_kes"]
            .sum()
            .sort_values(ascending=False)
    )
    bar_c = axes[1, 1].bar(cat_cost.index, cat_cost.values,
                           color="steelblue", edgecolor="white")
    axes[1, 1].set_title("Estimated Reorder Cost by Category (KES)",
                         fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Cost (KES)")
    axes[1, 1].tick_params(axis="x", rotation=35)
    for bar, val in zip(bar_c, cat_cost.values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 10,
                        f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Automatic Replenishment Dashboard — Eldoret Shop (End-of-Week Snapshot)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "19_replenishment_dashboard.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Dashboard saved → {}", path.name)


def plot_stock_vs_reorder(reco: pd.DataFrame) -> None:
    """Scatter: current stock vs reorder point, labelled by product."""
    status_colors = {
        "OUT OF STOCK":   "#d62728",
        "CRITICALLY LOW": "#ff7f0e",
        "LOW":            "#f0c040",
        "OK":             "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for status, grp in reco.groupby("stock_status"):
        ax.scatter(
            grp["reorder_point"], grp["current_stock"],
            label=status, color=status_colors.get(status, "grey"),
            s=90, alpha=0.9, zorder=3,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                row["product_name"][:14],
                (row["reorder_point"], row["current_stock"]),
                fontsize=7, alpha=0.75,
                xytext=(4, 3), textcoords="offset points",
            )

    lim_max = max(reco["current_stock"].max(), reco["reorder_point"].max()) * 1.15
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=1.2, label="Stock = Reorder Point")
    ax.fill_between([0, lim_max], [0, 0], [0, lim_max],
                    alpha=0.05, color="red", label="Reorder Zone (below line)")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Reorder Point (units)", fontsize=11)
    ax.set_ylabel("Current Stock (units)", fontsize=11)
    ax.set_title("Current Stock vs Reorder Point — End-of-Week Status",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "20_stock_vs_reorder_point.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Scatter saved → {}", path.name)


def plot_opening_vs_closing(reco: pd.DataFrame) -> None:
    """
    Horizontal grouped bar: opening stock (Monday) vs closing stock (end of week).
    Highlights how much each product sold during the week.
    """
    reco_s = reco.sort_values("opening_stock", ascending=True)
    y      = np.arange(len(reco_s))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    bars_open  = ax.barh(y + height/2, reco_s["opening_stock"],  height, label="Opening Stock (Mon)", color="steelblue",  alpha=0.85)
    bars_close = ax.barh(y - height/2, reco_s["current_stock"],  height, label="Closing Stock (End-of-Week)", color="coral", alpha=0.85)

    # Colour closing bars by status
    status_colors = {"OUT OF STOCK":"#d62728","CRITICALLY LOW":"#ff7f0e","LOW":"#f0c040","OK":"#2ca02c"}
    for bar, status in zip(bars_close, reco_s["stock_status"]):
        bar.set_color(status_colors.get(status, "coral"))

    ax.set_yticks(y)
    ax.set_yticklabels(reco_s["product_name"], fontsize=9)
    ax.set_xlabel("Units", fontsize=11)
    ax.set_title("Weekly Stock Consumption: Opening vs Closing Balance",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    fig.tight_layout()
    path = FIGURES_DIR / "21_opening_vs_closing_stock.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Opening vs closing stock saved → {}", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    input_path:  Path = PROCESSED_DATASET,
    output_path: Path = PROCESSED_DATA_DIR / "replenishment_report.csv",
) -> None:
    """Generate the end-of-week replenishment report."""

    logger.info("Loading feature dataset from {}", input_path)
    df = pd.read_csv(input_path, parse_dates=["date"])
    logger.info("  {:,} rows × {} columns", *df.shape)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    reco = compute_replenishment(df)

    # ── Save CSV ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reco.to_csv(output_path, index=False)
    logger.info("Replenishment report saved → {}", output_path)

    # ── Pretty-print the full table ───────────────────────────────────────
    display_cols = [
        "priority_label", "product_name", "category",
        "opening_stock", "current_stock",
        "days_of_stock_remaining", "reorder_point",
        "stock_status", "reorder_needed",
        "recommended_order_qty", "estimated_order_cost_kes",
    ]
    pd.set_option("display.max_rows", 30)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 22)
    logger.info("\n\nFULL REPLENISHMENT REPORT\n{}\n",
                reco[display_cols].to_string(index=False))

    # ── Urgent items summary ──────────────────────────────────────────────
    urgent = reco[reco["replenishment_priority"] <= 2]
    if len(urgent):
        logger.warning(
            "\n⚠  {} products need URGENT / HIGH-priority replenishment:\n{}",
            len(urgent),
            urgent[["product_name","stock_status","current_stock",
                     "recommended_order_qty","estimated_order_cost_kes",
                     "days_of_stock_remaining"]].to_string(index=False),
        )

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_replenishment_dashboard(reco)
    plot_stock_vs_reorder(reco)
    plot_opening_vs_closing(reco)

    total_cost = reco["estimated_order_cost_kes"].sum()
    logger.success(
        "Done.  {} products to reorder | Total order cost: KES {:,.0f}",
        reco["reorder_needed"].sum(), total_cost,
    )


if __name__ == "__main__":
    app()
