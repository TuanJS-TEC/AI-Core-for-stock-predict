"""Constants for the multimodal trend forecast (price + financials + news sentiment, T+2 and T+5)."""
from ai_core.fusion import config as AC
import ai_core.prices.config

TICKERS = ai_core.prices.config.TICKERS; HORIZONS = (2, 5); START = "2022-01-01"
BANKS = ("ACB", "MBB", "STB", "TCB", "VCB", "VPB")
LP = 60; LT = 10                     # window length: price branch, text branch
D = 32; EMB = 4; DROPOUT = 0.2; PRICE_KINDS = ("lstm", "gru", "transformer")
LR = 1e-3; WD = 1e-2; BATCH = 256; MAX_EPOCHS = 60; PATIENCE = 10
SEEDS = (59, 60, 61, 62, 63); SELECT_SEEDS = SEEDS[:2]
FOLD_NLP = {"A": "phobert-foldB", "B": "phobert-foldB", "C": "phobert-foldC", "F": "phobert-phobert-v1"}
FINAL_FOLD = ("F", "2026-05-04", "2026-07-30", "2026-10-30")
HOLDOUT_AFTER = "2026-10-30"; HOLDOUT_MIN = 40; HOLDOUT_MIN_EARLY = 20
HOLDOUT_BLOCK = 5                    # bootstrap block of the final zone (prereg addendum 2026-11-10)
ARMS = {"dl_price": (("price",), "none"), "dl_fund": (("fund",), "none"), "dl_text": (("text",), "none"),
        "fusion_concat": (("price", "fund", "text"), "concat"), "fusion_attention": (("price", "fund", "text"), "attention"),
        "no_price": (("fund", "text"), "concat"), "no_fund": (("price", "text"), "concat"), "no_text": (("price", "fund"), "concat")}
BASELINES = ("prior_fixed", "prior_adaptive", "logit_price", "gbm_price", "logit_text", "gbm_text", "logit_all", "gbm_all")
LEAK_ALARM_F1 = 0.50; LEAK_ALARM_AUC = 0.65
DATA = AC.ROOT / "data/mmf"; REPORTS = AC.ROOT / "docs/reports"
NLP_DB = AC.NLP / "data/nlp.db"; BCTC_FILE = AC.NLP / "data/bctc.parquet"
