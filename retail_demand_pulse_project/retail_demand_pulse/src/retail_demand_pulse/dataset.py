"""
dataset.py
----------
Generates a synthetic but realistic 2-year daily retail sales dataset
for a small-to-medium shop in Eldoret, Kenya (2024-2025).

Run:
    python -m retail_demand_pulse.dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from retail_demand_pulse.config import (
    KENYAN_HOLIDAYS,
    PRODUCTS,
    RAW_DATASET,
    RANDOM_SEED,
    START_DATE,
    END_DATE,
    WEATHER_PROFILES,
)

app = typer.Typer()

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Helper generators
# ─────────────────────────────────────────────────────────────────────────────

def _generate_weather(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Simulate Eldoret weather for each date based on monthly profiles."""
    conditions = ["Sunny", "Partly Cloudy", "Overcast", "Light Rain", "Heavy Rain"]
    rows = []
    for d in dates:
        avg_temp, rain_prob, avg_rain, weights = WEATHER_PROFILES[d.month]
        temp = avg_temp + rng.normal(0, 1.5)
        condition = rng.choice(conditions, p=weights)
        if condition in ("Light Rain", "Heavy Rain"):
            mm = rng.exponential(avg_rain) if avg_rain > 0 else 0.0
        else:
            mm = 0.0
        rows.append(
            {
                "date": d,
                "temperature_avg": round(float(temp), 1),
                "rainfall_mm": round(float(mm), 1),
                "weather_condition": condition,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def _neighbourhood_activity(d: pd.Timestamp) -> int:
    """
    0 = normal day
    1 = market day  (every Tuesday & Friday in Eldoret)
    2 = special event (school opening weeks, harvest festival, etc.)
    """
    if d.weekday() in (1, 4):  # Tuesday=1, Friday=4
        activity = 1
    else:
        activity = 0
    # School opening weeks: early Jan, early Feb (2nd term), early Sep
    if (d.month == 1 and 3 <= d.day <= 9) or \
       (d.month == 5 and 3 <= d.day <= 9) or \
       (d.month == 9 and 3 <= d.day <= 9):
        activity = 2
    # Month-end effect (last 3 days of month)
    if d.day >= 28:
        activity = max(activity, 1)
    return activity


def _demand_multiplier(
    d: pd.Timestamp,
    weather_row: pd.Series,
    product: tuple,
    is_holiday: bool,
) -> float:
    """
    Compute a multiplicative demand adjustment based on:
      - day-of-week / weekend
      - public holiday
      - month seasonality
      - weather
      - neighbourhood activity
    """
    pid, name, category, perishable, shelf_life, base, unit_p, cost_p = product

    mult = 1.0

    # ── Weekend boost ──────────────────────────────────────────────────────
    if d.weekday() >= 5:        # Saturday/Sunday
        mult *= 1.30

    # ── Public holiday boost ───────────────────────────────────────────────
    if is_holiday:
        mult *= rng.uniform(1.4, 1.9)

    # ── Market day boost ───────────────────────────────────────────────────
    activity = _neighbourhood_activity(d)
    if activity == 1:
        mult *= 1.15
    elif activity == 2:
        mult *= 1.30

    # ── Monthly seasonality ────────────────────────────────────────────────
    seasonal = {
        "Dairy":        [1.0,1.0,1.0,1.0,1.0,1.1,1.1,1.0,1.0,1.0,1.1,1.2],
        "Bakery":       [1.0,1.0,1.1,1.0,1.0,1.0,1.0,1.0,1.0,1.1,1.1,1.3],
        "Vegetables":   [1.0,1.0,0.9,1.0,1.1,1.1,1.0,1.0,1.0,0.9,1.0,1.1],
        "Fruits":       [0.9,0.9,1.0,1.1,1.1,1.0,0.9,0.9,1.0,1.1,1.1,1.2],
        "Staples":      [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.1,1.3],
        "Beverages":    [1.1,1.1,1.0,1.0,0.9,0.8,0.8,0.9,1.0,1.0,1.1,1.2],
        "Condiments":   [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.1],
        "Cooking":      [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.1,1.2],
        "Personal Care":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.1],
        "Household":    [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.1],
    }
    mult *= seasonal.get(category, [1.0] * 12)[d.month - 1]

    # ── Weather effects ────────────────────────────────────────────────────
    temp   = weather_row["temperature_avg"]
    rain   = weather_row["rainfall_mm"]
    wcond  = weather_row["weather_condition"]

    if category == "Beverages":
        # Hot days → more drinks
        if temp > 22:
            mult *= 1.0 + 0.03 * (temp - 22)
        # Heavy rain → fewer people out
        if wcond == "Heavy Rain":
            mult *= 0.80

    if perishable and wcond == "Heavy Rain":
        # Fewer shoppers brave rain to buy perishables
        mult *= 0.85

    if category in ("Staples", "Condiments") and wcond == "Heavy Rain":
        # People stock up before / during rain
        mult *= 1.10

    # ── Price sensitivity (simple elasticity) ─────────────────────────────
    # Higher-priced items have slightly lower demand
    if unit_p > 200:
        mult *= 0.95

    return float(mult)


def _spoilage_risk(
    d: pd.Timestamp,
    product: tuple,
    weather_row: pd.Series,
    qty_sold: int,
    base_demand: int,
) -> float:
    """
    Compute a spoilage risk score in [0, 1] for perishable items.
    Higher score = greater spoilage risk.
    """
    pid, name, category, perishable, shelf_life, base, unit_p, cost_p = product
    if not perishable:
        return 0.0

    # Base risk inversely proportional to shelf life
    base_risk = 1.0 / shelf_life

    # Unsold stock increases risk (demand shortfall)
    demand_ratio = qty_sold / max(base_demand, 1)
    unsold_risk  = max(0.0, 1.0 - demand_ratio) * 0.4

    # Temperature risk
    temp = weather_row["temperature_avg"]
    temp_risk = max(0.0, (temp - 18) / 10.0) * 0.3

    # Rain / humidity
    if weather_row["weather_condition"] in ("Light Rain", "Heavy Rain"):
        humidity_risk = 0.1
    else:
        humidity_risk = 0.0

    score = min(1.0, base_risk + unsold_risk + temp_risk + humidity_risk)
    return round(float(score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main generation logic
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    output_path: Path = RAW_DATASET,
) -> None:
    """Generate the synthetic Kenyan retail dataset and save to *output_path*."""

    logger.info("Starting dataset generation for Eldoret Retail Shop (2024-2025)")

    dates        = pd.date_range(START_DATE, END_DATE, freq="D")
    holiday_set  = set(KENYAN_HOLIDAYS.keys())

    logger.info("Simulating daily weather …")
    weather_df = _generate_weather(dates)

    records = []

    logger.info("Simulating daily sales for {} products × {} days …",
                len(PRODUCTS), len(dates))

    for product in tqdm(PRODUCTS, desc="Products"):
        pid, name, category, perishable, shelf_life, base_demand, unit_p, cost_p = product

        for d in dates:
            date_str   = d.strftime("%Y-%m-%d")
            is_holiday = date_str in holiday_set
            weather_row = weather_df.loc[d]
            activity   = _neighbourhood_activity(d)

            mult = _demand_multiplier(d, weather_row, product, is_holiday)

            # Poisson noise around adjusted demand
            adjusted_base = max(1, base_demand * mult)
            qty_sold = int(rng.poisson(adjusted_base))

            # Occasionally inject a zero (stockout / closed)
            if rng.random() < 0.01:
                qty_sold = 0

            total_amount = round(qty_sold * unit_p, 2)
            spoilage     = _spoilage_risk(d, product, weather_row, qty_sold, base_demand)

            records.append(
                {
                    # Core columns
                    "date":               d,
                    "product_id":         pid,
                    "product_name":       name,
                    "category":           category,
                    "is_perishable":      perishable,
                    "shelf_life_days":    shelf_life,
                    "quantity_sold":      qty_sold,
                    "unit_price":         unit_p,
                    "cost_price":         cost_p,
                    "total_amount":       total_amount,
                    # Time features
                    "day_of_week":        d.day_name(),
                    "is_weekend":         d.weekday() >= 5,
                    "is_holiday":         is_holiday,
                    "month":              d.month,
                    "year":               d.year,
                    "week_of_year":       int(d.isocalendar()[1]),
                    # External contextual
                    "temperature_avg":    weather_row["temperature_avg"],
                    "rainfall_mm":        weather_row["rainfall_mm"],
                    "weather_condition":  weather_row["weather_condition"],
                    "neighbourhood_activity": activity,
                    # Pre-computed features
                    "profit_margin":      round((unit_p - cost_p) / unit_p, 4),
                    "spoilage_risk_score": spoilage,
                }
            )

    df = pd.DataFrame(records)
    df.sort_values(["date", "product_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(
        "Dataset saved to {} — {:,} rows × {} columns",
        output_path, len(df), len(df.columns),
    )


if __name__ == "__main__":
    app()
