"""Where things live. The only module that knows the repo layout; everything else asks here.

    config/        small tracked reference files (hose_holidays.csv, ext_universe.csv, companies.csv)
    data/prices/   {symbol}_stock.parquet/.csv, manifest.json, snapshots/   (written by jobs/sync_prices.py)
    data/prices_ext/  the external training universe
    data/news/     nlp.db, models/, raw/, vncorenlp/, tessdata/, bctc.parquet, reports/, annotation csv
    data/finhay/, data/mmf/, data/global/   other datasets
    logs/          cron logs
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # the directory holding pyproject.toml
CONFIG = ROOT / "config"
LOGS = ROOT / "logs"
DATA = ROOT / "data"
PRICES = DATA / "prices"
PRICES_EXT = DATA / "prices_ext"
NEWS = DATA / "news"
