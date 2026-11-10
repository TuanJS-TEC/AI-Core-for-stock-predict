"""Where things live. The only module that knows the repo layout; everything else asks here.
Data stays in its historical home dirs (Pipeline_V1.2/, NLP_Module/) until a later data move."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # the directory holding pyproject.toml
PRICES_HOME = ROOT / "Pipeline_V1.2"            # stock/ (parquet + snapshots), data/hose_holidays.csv, outputs/
NEWS_HOME = ROOT / "NLP_Module"                 # data/ (nlp.db, models, tessdata, vncorenlp, companies.csv, bctc.parquet)
DATA = ROOT / "data"                            # finhay/, mmf/
