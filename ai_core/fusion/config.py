"""Shared constants for the fusion layer. The two modules keep their own configs; this only joins them."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "Pipeline_V1.2"
NLP = ROOT / "NLP_Module"
for p in (PIPELINE, NLP):                      # `src` (Pipeline_V1.2) and `nlp_module` (NLP_Module) import as top-level packages
    if str(p) not in sys.path:
        sys.path.append(str(p))

STOCK_DIR = PIPELINE / "stock"                 # canonical prices: the newer copy, kept current by sync_stock_parquet.py
# 20 tickers: prices (Pipeline_V1.2/stock) and news (NLP nlp.db) for all; app/trial_forecast use the same list
UNIVERSE = "ACB BID CTG FPT GAS HDB HPG MBB MSN MWG NVL SSI STB TCB VCB VHM VIC VNM VPB VRE".split()
HORIZONS = (2, 5)

TEST_END = "2026-10-30"                        # pre-registered: fold C's test window ends here, frozen before any fusion result
COST, KAPPA = 0.0045, 0.5
# FOLDS lives in ai_core.splits: importing nlp_module here would drag NLP's import graph into the HTTP-only finhay cron
