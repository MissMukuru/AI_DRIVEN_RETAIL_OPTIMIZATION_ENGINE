"""
pipeline.py
-----------
Master pipeline — runs all stages in the correct order:

  1. dataset            → generate raw data
  2. clean              → data cleaning
  3. eda                → exploratory data analysis
  4. features           → feature engineering + encoding + scaling
  5. train_demand       → XGBoost + LSTM demand forecasting
  6. train_spoilage     → spoilage risk models
  7. anomaly_detection  → Isolation Forest
  8. replenishment      → replenishment engine

Run the full pipeline:
    python -m retail_demand_pulse.pipeline

Run specific stages only:
    python -m retail_demand_pulse.pipeline --stages dataset clean features
"""

from pathlib import Path
from typing import List, Optional

from loguru import logger
import typer

app = typer.Typer(help="Master pipeline for Retail Demand Pulse project.")

STAGE_ORDER = [
    "dataset",
    "clean",
    "eda",
    "features",
    "train_demand",
    "train_spoilage",
    "anomaly_detection",
    "replenishment",
]


def _run_dataset():
    from retail_demand_pulse.dataset import main as _main
    _main()


def _run_clean():
    from retail_demand_pulse.clean import main as _main
    _main()


def _run_eda():
    from retail_demand_pulse.eda import main as _main
    _main()


def _run_features():
    from retail_demand_pulse.features import main as _main
    _main()


def _run_train_demand():
    from retail_demand_pulse.train_demand import main as _main
    _main()


def _run_train_spoilage():
    from retail_demand_pulse.train_spoilage import main as _main
    _main()


def _run_anomaly_detection():
    from retail_demand_pulse.anomaly_detection import main as _main
    _main()


def _run_replenishment():
    from retail_demand_pulse.replenishment import main as _main
    _main()


STAGE_RUNNERS = {
    "dataset":          _run_dataset,
    "clean":            _run_clean,
    "eda":              _run_eda,
    "features":         _run_features,
    "train_demand":     _run_train_demand,
    "train_spoilage":   _run_train_spoilage,
    "anomaly_detection": _run_anomaly_detection,
    "replenishment":    _run_replenishment,
}


@app.command()
def main(
    stages: Optional[List[str]] = typer.Argument(
        default=None,
        help="Stages to run (default: all). "
             f"Choices: {STAGE_ORDER}",
    ),
    skip: Optional[List[str]] = typer.Option(
        default=None,
        help="Stages to skip.",
    ),
) -> None:
    """Run the Retail Demand Pulse pipeline end-to-end."""

    stages_to_run = stages if stages else STAGE_ORDER

    # Validate stage names
    unknown = set(stages_to_run) - set(STAGE_ORDER)
    if unknown:
        logger.error("Unknown stage(s): {}", unknown)
        raise typer.Exit(1)

    # Apply skip list
    skip_set = set(skip) if skip else set()
    stages_to_run = [s for s in stages_to_run if s not in skip_set]

    logger.info("=" * 60)
    logger.info("  Retail Demand Pulse – Master Pipeline")
    logger.info("  Stages to run: {}", stages_to_run)
    logger.info("=" * 60)

    for stage in stages_to_run:
        logger.info("\n{'─'*55}")
        logger.info("▶  Stage: {}", stage.upper())
        logger.info("{'─'*55}")
        try:
            STAGE_RUNNERS[stage]()
            logger.success("✔  {} complete", stage)
        except Exception as exc:
            logger.error("✘  {} failed: {}", stage, exc)
            raise typer.Exit(1) from exc

    logger.success("\n🎉  All pipeline stages completed successfully!")


if __name__ == "__main__":
    app()
