"""
anomaly_detection.py
--------------------
6C. Anomaly Detection in Sales Patterns

Uses Isolation Forest to detect anomalous transactions in:
  - quantity_sold
  - total_amount
  - combined multivariate feature space

Results are saved as a flagged CSV and visualised.

Run:
    python -m retail_demand_pulse.anomaly_detection
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import typer

from retail_demand_pulse.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATASET,
    RANDOM_SEED,
)

app = typer.Typer()

ANOMALY_FEATURES = [
    "quantity_sold",
    "total_amount",
    "temperature_avg",
    "rainfall_mm",
    "is_weekend",
    "is_holiday",
    "neighbourhood_activity",
    "sales_roll_mean_7",
    "sales_lag_1",
]

CONTAMINATION = 0.03      # expected fraction of anomalies (~3 %)


# ─────────────────────────────────────────────────────────────────────────────
# Training & detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit Isolation Forest on the full dataset and add anomaly labels.
    Returns the dataframe with two new columns:
      - anomaly_score  (lower = more anomalous)
      - is_anomaly     (True/False)
    """
    logger.info("Fitting Isolation Forest (contamination={}) …", CONTAMINATION)

    available = [c for c in ANOMALY_FEATURES if c in df.columns]
    X = df[available].fillna(0).values.astype(np.float32)

    # Scale before feeding to IF
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # score_samples: higher = more normal; decision_function same but offset
    scores = iso.score_samples(X_scaled)
    labels = iso.predict(X_scaled)           # 1 = normal, -1 = anomaly

    df = df.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"]    = (labels == -1)

    n_anomalies = df["is_anomaly"].sum()
    logger.info(
        "  Anomalies detected: {:,} / {:,} ({:.1%})",
        n_anomalies, len(df), n_anomalies / len(df),
    )

    # Save model + scaler
    model_path = MODELS_DIR / "isolation_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":    iso,
            "scaler":   scaler,
            "features": available,
        }, f)
    logger.info("  Isolation Forest saved to {}", model_path)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_timeline(df: pd.DataFrame) -> None:
    """Daily revenue with anomalous points highlighted."""
    daily = (
        df.groupby("date")
          .agg(
              total_amount=("total_amount", "sum"),
              is_anomaly  =("is_anomaly",   "any"),
          )
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date"], daily["total_amount"],
            color="steelblue", lw=0.8, label="Daily Revenue")

    anomaly_days = daily[daily["is_anomaly"]]
    ax.scatter(anomaly_days["date"], anomaly_days["total_amount"],
               color="red", s=40, zorder=5, label="Anomaly Day")

    ax.set_title("Daily Revenue with Detected Anomalies", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue (KES)")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "15_anomaly_timeline.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Anomaly timeline saved → {}", path.name)


def plot_anomaly_scatter(df: pd.DataFrame) -> None:
    """2-D scatter: quantity_sold vs total_amount coloured by anomaly label."""
    fig, ax = plt.subplots(figsize=(8, 6))

    normal  = df[~df["is_anomaly"]]
    anomaly = df[df["is_anomaly"]]

    ax.scatter(normal["quantity_sold"], normal["total_amount"],
               alpha=0.15, s=5, color="steelblue", label="Normal")
    ax.scatter(anomaly["quantity_sold"], anomaly["total_amount"],
               alpha=0.7,  s=20, color="red",       label="Anomaly")

    ax.set_xlabel("Quantity Sold")
    ax.set_ylabel("Total Amount (KES)")
    ax.set_title("Anomaly Detection – Qty Sold vs Revenue")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "16_anomaly_scatter.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Anomaly scatter saved → {}", path.name)


def plot_anomaly_score_dist(df: pd.DataFrame) -> None:
    """Distribution of anomaly scores."""
    fig, ax = plt.subplots(figsize=(9, 4))
    normal_scores  = df.loc[~df["is_anomaly"], "anomaly_score"]
    anomaly_scores = df.loc[df["is_anomaly"],  "anomaly_score"]

    ax.hist(normal_scores,  bins=60, alpha=0.6, color="steelblue",
            label="Normal", density=True)
    ax.hist(anomaly_scores, bins=30, alpha=0.7, color="red",
            label="Anomaly", density=True)
    ax.axvline(df["anomaly_score"].quantile(CONTAMINATION),
               color="orange", linestyle="--", label="Threshold")
    ax.set_title("Distribution of Isolation Forest Anomaly Scores")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "17_anomaly_score_dist.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Score distribution saved → {}", path.name)


def plot_anomalies_by_category(df: pd.DataFrame) -> None:
    """Anomaly rate per product category."""
    cat_stats = (
        df.groupby("category", observed=True)
          .agg(
              total=("is_anomaly", "count"),
              anomalies=("is_anomaly", "sum"),
          )
          .assign(rate=lambda d: d["anomalies"] / d["total"] * 100)
          .sort_values("rate", ascending=True)
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(cat_stats["category"], cat_stats["rate"], color="tomato")
    ax.axvline(CONTAMINATION * 100, color="navy", linestyle="--",
               label=f"Global rate ({CONTAMINATION*100:.0f}%)")
    ax.set_xlabel("Anomaly Rate (%)")
    ax.set_title("Anomaly Rate by Product Category")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "18_anomaly_by_category.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Anomaly-by-category saved → {}", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    input_path:  Path = PROCESSED_DATASET,
    output_path: Path = PROCESSED_DATA_DIR / "retail_eldoret_anomalies.csv",
) -> None:
    """Run Isolation Forest anomaly detection and save flagged dataset."""

    logger.info("Loading feature dataset from {}", input_path)
    df = pd.read_csv(input_path, parse_dates=["date"])
    logger.info("  {:,} rows × {} columns", *df.shape)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = detect_anomalies(df)

    plot_anomaly_timeline(df)
    plot_anomaly_scatter(df)
    plot_anomaly_score_dist(df)
    plot_anomalies_by_category(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success("Flagged dataset saved to {}", output_path)

    # Summary of top anomalous records
    top_anomalies = (
        df[df["is_anomaly"]]
          .sort_values("anomaly_score")
          [["date", "product_name", "category",
            "quantity_sold", "total_amount", "anomaly_score"]]
          .head(10)
    )
    logger.info("Top 10 most anomalous records:\n{}", top_anomalies.to_string())

    logger.success("Anomaly detection complete.")


if __name__ == "__main__":
    app()
