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
# tickers with both prices (Pipeline) and news (NLP); BID CTG HDB VHM NVL lack prices, VNM SSI MBB GAS VRE lack news
UNIVERSE = ["ACB", "FPT", "HPG", "MSN", "MWG", "STB", "TCB", "VCB", "VIC", "VPB"]
HORIZONS = (2, 5)

TEST_END = "2026-10-30"                        # pre-registered: fold C's test window ends here, frozen before any fusion result
COST, KAPPA = 0.0045, 0.5
from nlp_module.forecast import walk_forward   # noqa: E402  (after the sys.path setup above)
FOLDS = [(n, a, b, c, TEST_END if n == "C" else e) for n, a, b, c, e in walk_forward.FOLDS]
