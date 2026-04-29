"""
eda.py
------
B. Exploratory Data Analysis
  - Summary statistics
  - Sales trends (time-series)
  - Seasonality: monthly, weekly, holiday effect
  - Weather impact on sales
  - Perishable vs non-perishable behaviour
  - Correlation analysis
  - All figures saved to reports/figures/

Run:
    python -m retail_demand_pulse.eda
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import typer

from retail_demand_pulse.config import (
    FIGURES_DIR,
    INTERIM_DATASET,
)

app = typer.Typer()

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = sns.color_palette("tab10")
FIG_DPI = 120


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save figure
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved figure → {}", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# EDA functions
# ─────────────────────────────────────────────────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> None:
    """Print and save key summary statistics."""
    logger.info("── Summary Statistics ────────────────────────────────────")
    numeric_cols = ["quantity_sold", "total_amount", "temperature_avg",
                    "rainfall_mm", "profit_margin", "spoilage_risk_score"]
    stats = df[numeric_cols].describe().T
    logger.info("\n{}", stats.to_string())

    # Per-category
    cat_stats = (
        df.groupby("category", observed=True)["quantity_sold"]
          .agg(["mean", "median", "std", "sum"])
          .sort_values("sum", ascending=False)
    )
    logger.info("\nSales by category:\n{}", cat_stats.to_string())


def plot_sales_trend(df: pd.DataFrame) -> None:
    """Daily total sales revenue over the full 2-year period."""
    daily = (
        df.groupby("date")["total_amount"]
          .sum()
          .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Raw daily
    axes[0].plot(daily["date"], daily["total_amount"],
                 color=PALETTE[0], linewidth=0.7, alpha=0.8)
    axes[0].set_title("Daily Total Revenue (KES)", fontsize=13)
    axes[0].set_ylabel("Revenue (KES)")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha="right")

    # 30-day rolling average
    daily["rolling_30"] = daily["total_amount"].rolling(30, center=True).mean()
    axes[1].plot(daily["date"], daily["rolling_30"],
                 color=PALETTE[1], linewidth=1.5, label="30-day rolling mean")
    axes[1].fill_between(daily["date"],
                         daily["rolling_30"] * 0.85,
                         daily["rolling_30"] * 1.15,
                         alpha=0.2, color=PALETTE[1])
    axes[1].set_title("30-Day Rolling Average Revenue (KES)", fontsize=13)
    axes[1].set_ylabel("Revenue (KES)")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle("Retail Sales Trend – Eldoret Shop 2024-2025",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "01_sales_trend")


def plot_seasonality(df: pd.DataFrame) -> None:
    """Monthly and weekly seasonality in quantity sold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Monthly
    monthly = (
        df.groupby("month")["quantity_sold"]
          .mean()
          .reset_index()
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])

    sns.barplot(data=monthly, x="month_name", y="quantity_sold",
                palette="Blues_d", ax=axes[0], order=month_names)
    axes[0].set_title("Average Daily Qty Sold by Month")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Avg Qty Sold")
    axes[0].tick_params(axis="x", rotation=30)

    # Weekly
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday",
                 "Saturday","Sunday"]
    weekly = (
        df.groupby("day_of_week", observed=True)["quantity_sold"]
          .mean()
          .reindex(day_order)
          .reset_index()
    )
    sns.barplot(data=weekly, x="day_of_week", y="quantity_sold",
                palette="Oranges_d", ax=axes[1])
    axes[1].set_title("Average Qty Sold by Day of Week")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("Demand Seasonality", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "02_seasonality")


def plot_holiday_effect(df: pd.DataFrame) -> None:
    """Compare sales on public holidays vs regular days."""
    df2 = df.copy()
    df2["day_type"] = np.where(df2["is_holiday"], "Public Holiday",
                     np.where(df2["is_weekend"], "Weekend", "Weekday"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    order = ["Weekday", "Weekend", "Public Holiday"]
    sns.boxplot(data=df2, x="day_type", y="quantity_sold",
                order=order, palette="Set2", ax=axes[0],
                showfliers=False)
    axes[0].set_title("Qty Sold by Day Type")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Quantity Sold")

    # Revenue version
    daily_type = (
        df2.groupby(["date", "day_type"])["total_amount"]
           .sum()
           .reset_index()
    )
    sns.violinplot(data=daily_type, x="day_type", y="total_amount",
                   order=order, palette="Set2", ax=axes[1], cut=0)
    axes[1].set_title("Daily Revenue by Day Type")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Daily Revenue (KES)")

    fig.suptitle("Holiday & Weekend Effect on Sales",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "03_holiday_effect")


def plot_weather_impact(df: pd.DataFrame) -> None:
    """Weather condition and temperature effects on sales."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Qty by weather condition
    cond_order = ["Sunny","Partly Cloudy","Overcast","Light Rain","Heavy Rain"]
    cond_data = df[df["weather_condition"].isin(cond_order)].copy()
    sns.boxplot(data=cond_data, x="weather_condition", y="quantity_sold",
                order=cond_order, palette="coolwarm", ax=axes[0, 0],
                showfliers=False)
    axes[0, 0].set_title("Qty Sold by Weather Condition")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # 2. Temperature vs beverage sales
    bev = df[df["category"] == "Beverages"]
    axes[0, 1].scatter(bev["temperature_avg"], bev["quantity_sold"],
                       alpha=0.2, s=10, color=PALETTE[3])
    m, b = np.polyfit(bev["temperature_avg"], bev["quantity_sold"], 1)
    xs = np.linspace(bev["temperature_avg"].min(), bev["temperature_avg"].max(), 100)
    axes[0, 1].plot(xs, m * xs + b, "r--", lw=2)
    axes[0, 1].set_title("Temperature vs Beverage Sales")
    axes[0, 1].set_xlabel("Avg Temperature (°C)")
    axes[0, 1].set_ylabel("Qty Sold")

    # 3. Rainfall vs perishable sales
    perish = df[df["is_perishable"] == True]
    axes[1, 0].scatter(perish["rainfall_mm"], perish["quantity_sold"],
                       alpha=0.2, s=10, color=PALETTE[2])
    axes[1, 0].set_title("Rainfall vs Perishable Sales")
    axes[1, 0].set_xlabel("Rainfall (mm)")
    axes[1, 0].set_ylabel("Qty Sold")

    # 4. Monthly avg rainfall overlay on sales
    monthly_weather = (
        df.groupby("month")
          .agg(avg_qty=("quantity_sold", "mean"),
               avg_rain=("rainfall_mm",  "mean"))
          .reset_index()
    )
    ax4a = axes[1, 1]
    ax4b = ax4a.twinx()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax4a.bar(range(1, 13), monthly_weather["avg_rain"],
             color="steelblue", alpha=0.4, label="Avg Rainfall (mm)")
    ax4b.plot(range(1, 13), monthly_weather["avg_qty"],
              "ro-", lw=2, label="Avg Qty Sold")
    ax4a.set_xticks(range(1, 13))
    ax4a.set_xticklabels(month_names, rotation=30)
    ax4a.set_ylabel("Avg Rainfall (mm)", color="steelblue")
    ax4b.set_ylabel("Avg Qty Sold", color="red")
    axes[1, 1].set_title("Monthly Rainfall vs Average Sales")

    fig.suptitle("Weather Impact on Retail Sales",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "04_weather_impact")


def plot_perishable_behaviour(df: pd.DataFrame) -> None:
    """Perishable vs non-perishable sales volume and spoilage risk."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Avg daily sales
    ptype = (
        df.groupby(["is_perishable", "product_name"], observed=True)
          ["quantity_sold"]
          .mean()
          .reset_index()
    )
    ptype["type"] = ptype["is_perishable"].map(
        {True: "Perishable", False: "Non-Perishable"}
    )
    sns.boxplot(data=ptype, x="type", y="quantity_sold",
                palette="Pastel1", ax=axes[0], showfliers=False)
    axes[0].set_title("Avg Qty per Product\n(Perishable vs Non)")
    axes[0].set_xlabel("")

    # 2. Spoilage risk by category (perishables only)
    perish = df[df["is_perishable"] == True]
    sns.boxplot(data=perish, x="category", y="spoilage_risk_score",
                palette="Reds", ax=axes[1], showfliers=False)
    axes[1].set_title("Spoilage Risk by Category")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_xlabel("")

    # 3. Spoilage risk over month (seasonal)
    monthly_spol = (
        perish.groupby("month")["spoilage_risk_score"]
              .mean()
              .reset_index()
    )
    sns.lineplot(data=monthly_spol, x="month", y="spoilage_risk_score",
                 marker="o", ax=axes[2], color=PALETTE[1])
    axes[2].set_title("Monthly Avg Spoilage Risk (Perishables)")
    axes[2].set_xticks(range(1, 13))
    axes[2].set_xticklabels(["J","F","M","A","M","J",
                              "J","A","S","O","N","D"])
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Avg Spoilage Risk Score")

    fig.suptitle("Perishable Product Behaviour",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "05_perishable_behaviour")


def plot_correlation(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap of key numeric features."""
    num_cols = [
        "quantity_sold", "total_amount", "unit_price",
        "temperature_avg", "rainfall_mm",
        "is_weekend", "is_holiday",
        "neighbourhood_activity",
        "profit_margin", "spoilage_risk_score",
        "month", "week_of_year",
    ]
    corr_df = df[num_cols].copy()
    corr_df["is_weekend"] = corr_df["is_weekend"].astype(int)
    corr_df["is_holiday"] = corr_df["is_holiday"].astype(int)

    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "06_correlation_heatmap")


def plot_top_products(df: pd.DataFrame) -> None:
    """Revenue and volume rankings."""
    top_rev = (
        df.groupby("product_name", observed=True)["total_amount"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )
    top_vol = (
        df.groupby("product_name", observed=True)["quantity_sold"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=top_rev, x="total_amount", y="product_name",
                palette="Blues_r", ax=axes[0])
    axes[0].set_title("Top 10 Products by Revenue")
    axes[0].set_xlabel("Total Revenue (KES)")
    axes[0].set_ylabel("")

    sns.barplot(data=top_vol, x="quantity_sold", y="product_name",
                palette="Greens_r", ax=axes[1])
    axes[1].set_title("Top 10 Products by Volume Sold")
    axes[1].set_xlabel("Total Units Sold")
    axes[1].set_ylabel("")

    fig.suptitle("Product Performance Rankings",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "07_top_products")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    input_path: Path = INTERIM_DATASET,
) -> None:
    """Run the full EDA pipeline and save all figures."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cleaned dataset from {}", input_path)
    df = pd.read_csv(input_path, parse_dates=["date"])
    logger.info("  {:,} rows × {} columns", *df.shape)

    logger.info("Running EDA …")

    summary_statistics(df)
    plot_sales_trend(df)
    plot_seasonality(df)
    plot_holiday_effect(df)
    plot_weather_impact(df)
    plot_perishable_behaviour(df)
    plot_correlation(df)
    plot_top_products(df)

    logger.success("EDA complete — {} figures saved to {}/",
                   7, FIGURES_DIR)


if __name__ == "__main__":
    app()
