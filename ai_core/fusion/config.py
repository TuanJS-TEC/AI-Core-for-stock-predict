"""Shared constants for the fusion layer. The two modules keep their own configs; this only joins them."""
from ai_core import paths

ROOT = paths.ROOT

STOCK_DIR = paths.PRICES                 # canonical prices: the newer copy, kept current by ai_core/prices/sync.py
# 20 tickers: prices (data/prices) and news (data/news/nlp.db) for all; app/trial_forecast use the same list
UNIVERSE = "ACB BID CTG FPT GAS HDB HPG MBB MSN MWG NVL SSI STB TCB VCB VHM VIC VNM VPB VRE".split()
HORIZONS = (2, 5)

TEST_END = "2026-10-30"                        # pre-registered: fold C's test window ends here, frozen before any fusion result
from ai_core.trading import COST, KAPPA   # noqa: F401  (shared with ai_core.models.mmf)
# FOLDS lives in ai_core.fusion.splits: importing ai_core.news here would drag NLP's import graph into the HTTP-only finhay cron
