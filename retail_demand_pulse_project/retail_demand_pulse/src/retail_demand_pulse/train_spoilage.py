"""
train_spoilage.py
-----------------
6B. Spoilage Risk Prediction

Trains two models on perishable products only:
  1. Regression  – predict spoilage_risk_score (float 0–1)
  2. Classification – predict HIGH spoilage (score ≥ 0.35)

Metrics: MAE / RMSE for regression; Accuracy / F1 / ROC-AUC for classification.
Feature importance and ROC curve are saved to reports/figures/.

Run:
    python -m retail_demand_pulse.train_spoilage
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import typer

from retail_demand_pulse.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATASET,
    RANDOM_SEED,
)

app = typer.Typer()

# ── Features for spoilage modelling ──────────────────────────────────────────
SPOILAGE_FEATURES = [
    "shelf_life_days",
    "temperature_avg",
    "rainfall_mm",
    "weather_condition_enc",
    "month",
    "is_weekend",
    "is_holiday",
    "neighbourhood_activity",
    "quantity_sold",          # lower demand → more unsold stock → more spoilage
    "sales_roll_mean_7",
    "sales_lag_1",
    "unit_price",
    "day_sin",
    "day_cos",
    "category_enc",
]
SPOILAGE_THRESHOLD = 0.35     # binary classification boundary


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_data(df: pd.DataFrame):
    """Filter to perishables and build X, y arrays."""
    perish = df[df["is_perishable"] == 1].copy()
    if len(perish) == 0:
        raise ValueError("No perishable rows found in dataset!")

    available = [c for c in SPOILAGE_FEATURES if c in perish.columns]
    X = perish[available].fillna(0).values
    y_reg = perish["spoilage_risk_score"].values.astype(float)
    y_cls = (y_reg >= SPOILAGE_THRESHOLD).astype(int)

    return X, y_reg, y_cls, available


# ─────────────────────────────────────────────────────────────────────────────
# Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_spoilage_regression(X, y_reg, feature_names):
    logger.info("Training spoilage REGRESSION model (GradientBoosting) …")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
    )

    reg = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    reg.fit(X_tr, y_tr)
    y_pred = np.clip(reg.predict(X_te), 0, 1)

    mae_  = mean_absolute_error(y_te, y_pred)
    rmse_ = np.sqrt(mean_squared_error(y_te, y_pred))
    logger.info("  [Spoilage Regression]  MAE={:.4f}  RMSE={:.4f}", mae_, rmse_)

    # Save
    model_path = MODELS_DIR / "spoilage_regression.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": reg, "features": feature_names}, f)
    logger.info("  Regression model saved to {}", model_path)

    # Feature importance
    _plot_feature_importance(
        reg.feature_importances_, feature_names,
        "Spoilage Regression Feature Importance",
        "11_spoilage_reg_feature_importance",
    )

    # Scatter: predicted vs actual
    _plot_regression_scatter(y_te, y_pred)

    return reg, y_te, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def train_spoilage_classification(X, y_cls, feature_names):
    logger.info(
        "Training spoilage CLASSIFICATION model (threshold={}) …",
        SPOILAGE_THRESHOLD
    )

    pos_rate = y_cls.mean()
    logger.info("  Positive class (high risk) rate: {:.1%}", pos_rate)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_cls, test_size=0.2, random_state=RANDOM_SEED,
        stratify=y_cls, shuffle=True,
    )

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_tr, y_tr)
    y_pred      = clf.predict(X_te)
    y_pred_prob = clf.predict_proba(X_te)[:, 1]

    logger.info("  Classification report:\n{}",
                classification_report(y_te, y_pred,
                                      target_names=["Low Risk","High Risk"]))
    auc = roc_auc_score(y_te, y_pred_prob)
    logger.info("  ROC-AUC: {:.4f}", auc)

    # Save
    model_path = MODELS_DIR / "spoilage_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "features": feature_names,
                     "threshold": SPOILAGE_THRESHOLD}, f)
    logger.info("  Classifier saved to {}", model_path)

    # Feature importance
    _plot_feature_importance(
        clf.feature_importances_, feature_names,
        "Spoilage Classifier Feature Importance",
        "12_spoilage_cls_feature_importance",
    )

    # ROC curve
    _plot_roc_curve(y_te, y_pred_prob, auc)

    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _plot_feature_importance(importances, feature_names, title, fname):
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = FIGURES_DIR / f"{fname}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  {} saved → {}", title, path.name)


def _plot_regression_scatter(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Spoilage Risk")
    ax.set_ylabel("Predicted Spoilage Risk")
    ax.set_title("Spoilage Regression – Predicted vs Actual")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "13_spoilage_reg_scatter.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Regression scatter saved → {}", path.name)


def _plot_roc_curve(y_true, y_prob, auc):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "navy", linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Spoilage Risk – ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIGURES_DIR / "14_spoilage_roc_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ROC curve saved → {}", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    input_path: Path = PROCESSED_DATASET,
) -> None:
    """Train spoilage risk regression and classification models."""

    logger.info("Loading feature dataset from {}", input_path)
    df = pd.read_csv(input_path, parse_dates=["date"])
    logger.info("  {:,} rows × {} columns", *df.shape)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y_reg, y_cls, feature_names = _prepare_data(df)
    logger.info("  Perishable rows: {:,}", len(y_reg))

    train_spoilage_regression(X, y_reg, feature_names)
    train_spoilage_classification(X, y_cls, feature_names)

    logger.success("Spoilage modelling complete.")


if __name__ == "__main__":
    app()
