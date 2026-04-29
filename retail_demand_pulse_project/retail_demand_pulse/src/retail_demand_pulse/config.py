"""
config.py
---------
Central configuration for the Retail Demand Pulse project.
All paths, constants, and hyper-parameters live here so every
other module can import from a single source of truth.
"""

from pathlib import Path

# ── Project root (two levels up from this file) ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Data directories ──────────────────────────────────────────────────────────
RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR   = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ── Model artefacts ──────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"

# ── Reports / figures ────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist at import time
for _dir in (RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
             MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Dataset file names ────────────────────────────────────────────────────────
RAW_DATASET      = RAW_DATA_DIR      / "retail_eldoret_raw.csv"
INTERIM_DATASET  = INTERIM_DATA_DIR  / "retail_eldoret_cleaned.csv"
PROCESSED_DATASET = PROCESSED_DATA_DIR / "retail_eldoret_features.csv"

# ── Simulation parameters ────────────────────────────────────────────────────
RANDOM_SEED   = 42
START_DATE    = "2024-01-01"
END_DATE      = "2025-12-31"
SHOP_LOCATION = "Eldoret, Kenya"

# ── Kenyan public holidays (YYYY-MM-DD) ──────────────────────────────────────
KENYAN_HOLIDAYS = {
    # 2024
    "2024-01-01": "New Year's Day",
    "2024-04-19": "Good Friday",
    "2024-04-22": "Easter Monday",
    "2024-05-01": "Labour Day",
    "2024-06-01": "Madaraka Day",
    "2024-10-10": "Huduma Day",
    "2024-10-20": "Mashujaa Day",
    "2024-12-12": "Jamhuri Day",
    "2024-12-25": "Christmas Day",
    "2024-12-26": "Boxing Day",
    # 2025
    "2025-01-01": "New Year's Day",
    "2025-04-18": "Good Friday",
    "2025-04-21": "Easter Monday",
    "2025-05-01": "Labour Day",
    "2025-06-01": "Madaraka Day",
    "2025-10-10": "Huduma Day",
    "2025-10-20": "Mashujaa Day",
    "2025-12-12": "Jamhuri Day",
    "2025-12-25": "Christmas Day",
    "2025-12-26": "Boxing Day",
}

# ── Product catalogue ─────────────────────────────────────────────────────────
PRODUCTS = [
    # id, name,                  category,         perishable, shelf_life, base_demand, unit_price, cost_price
    ("P001", "Fresh Milk (1L)",       "Dairy",          True,  3,  22, 65.0,  48.0),
    ("P002", "Eggs (tray 30)",        "Dairy",          True,  14, 8,  350.0, 270.0),
    ("P003", "Sliced Bread",          "Bakery",         True,  5,  18, 55.0,  38.0),
    ("P004", "Tomatoes (kg)",         "Vegetables",     True,  7,  15, 80.0,  55.0),
    ("P005", "Onions (kg)",           "Vegetables",     True,  21, 12, 90.0,  60.0),
    ("P006", "Bananas (bunch)",       "Fruits",         True,  5,  10, 120.0, 80.0),
    ("P007", "Oranges (kg)",          "Fruits",         True,  10, 8,  130.0, 85.0),
    ("P008", "Maize Flour (2kg)",     "Staples",        False, 180,30, 160.0, 120.0),
    ("P009", "Wheat Flour (1kg)",     "Staples",        False, 180,20, 85.0,  62.0),
    ("P010", "Rice (2kg)",            "Staples",        False, 365,18, 220.0, 165.0),
    ("P011", "Cooking Oil (1L)",      "Cooking",        False, 365,15, 250.0, 190.0),
    ("P012", "Sugar (1kg)",           "Condiments",     False, 730,25, 130.0, 98.0),
    ("P013", "Salt (500g)",           "Condiments",     False, 730,20, 35.0,  22.0),
    ("P014", "Soda (500ml)",          "Beverages",      False, 180,20, 60.0,  42.0),
    ("P015", "Water (500ml)",         "Beverages",      False, 365,25, 30.0,  18.0),
    ("P016", "Black Tea Leaves",      "Beverages",      False, 365,22, 70.0,  48.0),
    ("P017", "Bathing Soap",          "Personal Care",  False, 730,10, 85.0,  58.0),
    ("P018", "Washing Powder (500g)", "Household",      False, 730,8,  110.0, 78.0),
    ("P019", "Matches (box)",         "Household",      False, 730,30, 15.0,  8.0),
    ("P020", "Mandazi Mix (500g)",    "Bakery",         True,  7,  12, 90.0,  62.0),
]

# ── Eldoret weather profiles by month ─────────────────────────────────────────
# (avg_temp_c, rain_prob, avg_rain_mm, condition_weights)
# Conditions: "Sunny", "Partly Cloudy", "Overcast", "Light Rain", "Heavy Rain"
WEATHER_PROFILES = {
    1:  (22, 0.15, 3,  [0.55, 0.25, 0.10, 0.07, 0.03]),
    2:  (23, 0.20, 5,  [0.50, 0.25, 0.12, 0.10, 0.03]),
    3:  (22, 0.45, 18, [0.20, 0.25, 0.20, 0.25, 0.10]),
    4:  (20, 0.65, 35, [0.10, 0.15, 0.20, 0.35, 0.20]),
    5:  (19, 0.60, 28, [0.15, 0.20, 0.20, 0.30, 0.15]),
    6:  (18, 0.30, 10, [0.35, 0.25, 0.20, 0.15, 0.05]),
    7:  (17, 0.25, 7,  [0.40, 0.30, 0.15, 0.12, 0.03]),
    8:  (18, 0.20, 5,  [0.45, 0.30, 0.15, 0.08, 0.02]),
    9:  (20, 0.25, 8,  [0.40, 0.28, 0.18, 0.10, 0.04]),
    10: (21, 0.55, 25, [0.20, 0.20, 0.22, 0.28, 0.10]),
    11: (21, 0.55, 22, [0.20, 0.22, 0.22, 0.26, 0.10]),
    12: (21, 0.30, 10, [0.35, 0.28, 0.18, 0.14, 0.05]),
}

# ── Modelling hyper-parameters ────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LSTM_LOOKBACK    = 14   # days of history fed to the LSTM
LSTM_EPOCHS      = 30
LSTM_BATCH_SIZE  = 32
LSTM_UNITS       = 64

# ── Replenishment defaults ────────────────────────────────────────────────────
DEFAULT_LEAD_TIME_DAYS  = 2
SAFETY_STOCK_FACTOR     = 1.5   # safety_stock = factor * std(demand)
LOW_STOCK_THRESHOLD     = 0.25  # reorder when stock < 25 % of max
