"""Settings shared across pipeline steps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

SEED = 42

TARGET = "tickets_sold"
TIER_CODES = {"A": 0, "B": 1, "C": 2}

# Pricing: nudge price in proportion to how far projected sell-through is from target.
TARGET_SELL_THROUGH = 0.85
PRICE_SENSITIVITY = 0.35

# Inventory service
API_MAX_RETRIES = 5
API_TIMEOUT_SECONDS = 2.0
API_TIMEOUT_RATE = 0.15
API_ERROR_RATE = 0.10
