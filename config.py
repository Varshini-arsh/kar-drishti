"""
Kar Drishti - central configuration.

Every threshold used by the analyzers lives here so an officer (or developer)
can tune the system without touching the detection code.
Values can be overridden with environment variables where noted.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEMO_DIR = DATA_DIR / "demo_case"
REPORTS_DIR = BASE_DIR / "reports"

# ---------------------------------------------------------------- local LLM
# provider: "ollama" (default) or "openai" (any OpenAI-compatible local server,
# e.g. LM Studio or llama.cpp server). Nothing is ever sent to the internet.
LLM_PROVIDER = os.getenv("KD_LLM_PROVIDER", "ollama")
LLM_BASE_URL = os.getenv("KD_LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("KD_LLM_MODEL", "qwen2.5:7b")
LLM_TIMEOUT_SEC = int(os.getenv("KD_LLM_TIMEOUT", "240"))
LLM_TEMPERATURE = 0.1

# Public online demo (e.g. Hugging Face Spaces): synthetic data only, uploads disabled
PUBLIC_DEMO = os.getenv("KD_PUBLIC_DEMO", "0") == "1" or (Path(__file__).resolve().parent / "public_demo.flag").exists()
# True when the host also runs a local model (Hugging Face Space); False on Streamlit Community Cloud
LLM_EXPECTED = os.getenv("KD_LLM_EXPECTED", "0") == "1"

# ---------------------------------------------------------------- demo data
DEMO_SEED = 42
FY_START = "2025-04-01"
FY_END = "2026-03-31"
TAX_YEAR = "2025-26"

# ---------------------------------------------------------------- detection thresholds
# Structuring: many cash deposits kept just below the Rs 50,000 limit above which
# PAN must be quoted for a cash deposit.
CASH_PAN_THRESHOLD = 50_000
STRUCTURING_BAND = 0.10          # deposits within 10% below the threshold (45,000 - 49,999)
STRUCTURING_MIN_COUNT = 8        # at least this many such deposits in the year

# Large round-figure cash withdrawals
ROUND_CASH_MIN_AMOUNT = 200_000
ROUND_CASH_MULTIPLE = 50_000
ROUND_CASH_MIN_COUNT = 3

# Pass-through ("mule") accounts: money comes in and goes out almost at once
PASS_THROUGH_WINDOW_DAYS = 2
PASS_THROUGH_MIN_RATIO = 0.85
PASS_THROUGH_MIN_TOTAL = 1_000_000

# Round-tripping: money leaves an entity and comes back through others
ROUND_TRIP_MIN_AMOUNT = 500_000
ROUND_TRIP_MAX_DAYS_PER_HOP = 10
ROUND_TRIP_AMOUNT_TOLERANCE = 0.10   # each hop keeps 90-102% of previous amount
ROUND_TRIP_MAX_LEN = 5

# Income / turnover mismatch
MISMATCH_MIN_RATIO = 3.0
MISMATCH_MIN_CREDITS = 1_000_000

# Link finder
CLUSTER_MIN_SIZE = 3
SHELL_RECENT_REG_DAYS = 730      # firm registered within ~2 years of FY end

# GST
EWAY_BILL_THRESHOLD = 50_000         # e-way bill generally needed for goods consignments above this
GST_CYCLE_MIN_EDGE = 1_000_000       # each hop of an invoice loop must carry at least Rs 10 lakh in the year
GST_CYCLE_MAX_LEN = 6
GST_PASSTHROUGH_MIN_TAX = 500_000    # output tax in the year
GST_PASSTHROUGH_MAX_CASH_RATIO = 0.02
GST_NO_EWAY_MIN_COUNT = 5
GST_NO_EWAY_MIN_SHARE = 0.5
GST_RISKY_ITC_MIN = 500_000

# GST checks are an optional add-on (GST is administered by CBIC, not the Income Tax Department).
# Off by default; switch on in the app (Load data page) or with KD_GST=1.
GST_ENABLED = os.getenv("KD_GST", "0") == "1"

# Risk score: each red flag adds weight x strength (0-1) points; total capped at 100.
# The ML outlier model (Isolation Forest) is shown only as a hint and adds no points.
RISK_WEIGHTS = {
    "structuring": 40,
    "round_trip": 55,
    "pass_through": 40,
    "income_mismatch": 35,
    "non_filer": 40,
    "round_cash": 35,
    "shell_cluster": 30,
    "gst_circular": 45,
    "gst_itc_passthrough": 35,
    "gst_no_eway": 20,
    "gst_risky_itc": 40,
}
RISK_HIGH = 70
RISK_MEDIUM = 35
