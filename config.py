#/Users/nashe/nested_policy_pipeline/config.py
"""
Central configuration module
============================
• Meta-algorithm & environment constants
• LLM orchestration parameters (NOT part of T_current)
• Synthetic price / demand series generation
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# 1. Meta-algorithm & environment
# ------------------------------------------------------------------
HORIZON    = int(os.getenv("HORIZON", "150"))
META_STEPS = int(os.getenv("META_STEPS", "3"))

BATTERY_CAPACITY_KWH = float(os.getenv("BATTERY_CAPACITY_KWH", "100.0"))
INITIAL_SOC          = float(os.getenv("INITIAL_SOC",
                                        str(BATTERY_CAPACITY_KWH / 2)))

# ------------------------------------------------------------------
# 2. Market price & demand series
# ------------------------------------------------------------------
raw_price  = os.getenv("PRICE_SERIES")
raw_demand = os.getenv("DEMAND_SERIES")

if raw_price in (None, "", "GENERATE"):
    from src.data.series_model import generate_price_series
    PRICE_SERIES = generate_price_series(HORIZON)
else:
    PRICE_SERIES = [float(p) for p in raw_price.split(",")]

if raw_demand in (None, "", "CONSTANT"):
    from src.data.series_model import constant_demand
    DEMAND_SERIES = constant_demand(HORIZON)
else:
    DEMAND_SERIES = [float(d) for d in raw_demand.split(",")]

if len(PRICE_SERIES) < HORIZON + 1:
    raise ValueError("PRICE_SERIES must have at least HORIZON + 1 values.")
if len(DEMAND_SERIES) < HORIZON + 1:
    raise ValueError("DEMAND_SERIES must have at least HORIZON + 1 values.")

# ------------------------------------------------------------------
# 3. LLM orchestration  (outside T_current)
# ------------------------------------------------------------------
DEESEEK_BASE_URL = os.getenv("DEESEEK_BASE_URL",
                             "https://api.deepseek.com/v1")   # ← NEW
DEESEEK_API_KEY  = os.getenv("DEESEEK_API_KEY")
QWEN_API_KEY     = os.getenv("QWEN_API_KEY")

MODEL_DEEPSEEK = os.getenv("MODEL_DEEPSEEK", "deepseek-reasoner")  # ← NEW
MODEL_QWEN     = os.getenv("MODEL_QWEN",    "Qwen2.5-Coder-32B-Instruct")

if not DEESEEK_API_KEY or not QWEN_API_KEY:
    raise ValueError("Both DEESEEK_API_KEY and QWEN_API_KEY must be set in .env")

TASK_TEMPERATURE = float(os.getenv("TASK_TEMPERATURE", "0.30"))
TASK_MAX_TOKENS  = int(os.getenv("TASK_MAX_TOKENS",  "512"))
CODE_TEMPERATURE = float(os.getenv("CODE_TEMPERATURE", "0.20"))
CODE_MAX_TOKENS  = int(os.getenv("CODE_MAX_TOKENS",  "512"))

# ------------------------------------------------------------------
# 4. Physical limits & efficiencies
# ------------------------------------------------------------------
MAX_RATE_KWH  = float(os.getenv("MAX_RATE_KWH", "10.0"))
EFF_CHARGE    = float(os.getenv("EFF_CHARGE",    "0.95"))
EFF_DISCHARGE = float(os.getenv("EFF_DISCHARGE", "0.95"))
