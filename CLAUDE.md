# AI_Core

Vietnamese stock research (HOSE): prices + news + financials → T+2/T+5 forecasts. One package, `ai_core`, installed editable in two venvs (`pip install -e .`).

```
ai_core/prices   price data layer (sync, validation, snapshots, features, labels, splits)   
ai_core/news     news → sentiment → point-in-time text features (SQLite nlp.db)             
ai_core/models   mmf (multimodal trend forecast) and volcast (volatility forecast)
ai_core/fusion   live decision board, shadow report, direction GO/NO-GO, Finhay sync, Streamlit app
ai_core/paths.py the only module that knows where things live
jobs/            cron entry points (thin); tests/ mirrors the packages
```

Dependencies are one-way and enforced by `tests/test_architecture.py`: `prices` and `news` never import each other; `models` import them; `fusion` sits on top. `ai_core/__init__.py`, `prices/__init__.py` and `fusion/__init__.py` stay empty (the Finhay and cross-check crons must not drag in torch).

Layout of what is not code:
```
config/     small tracked reference files: hose_holidays.csv, ext_universe.csv, companies.csv
data/       prices/ (+ snapshots/, the frozen dev data), prices_ext/, news/ (nlp.db, models, raw, tessdata…), finhay/, mmf/, global/, reference/
logs/       cron logs (sync, crosscheck, live)
requirements/  prices.txt, vnstock6.txt, news.txt
.venv       main venv (news, prices, fusion, cron);  .venv_mmf  mmf training venv (reuses .venv's packages via nlp_base.pth)
```
All code asks `ai_core/paths.py` for these locations; run everything from the repo root.

Price sync (`jobs/sync_prices.py`, cron 15:20) runs on `~/.venv_6x` (vnstock generation 6, source tcbs, preview build valid until 2026-12-07) with `PYTHONPATH=<repo root>`; tcbs needs `vnstock auth tcbs` (Smart OTP) before each run, otherwise no new prices are fetched.

Tests (two venvs, run from the repo root; expected 208 and 236 passed):
```
.venv/bin/python -m pytest -q tests/prices tests/news tests/models/volcast tests/fusion/test_board.py tests/test_architecture.py tests/test_paths.py
.venv_mmf/bin/python -m pytest -q tests --ignore=tests/prices --ignore=tests/news --ignore=tests/models/volcast --ignore=tests/fusion/test_board.py --ignore=tests/test_architecture.py --ignore=tests/test_paths.py
```
The mmf holdout (`ai_core.models.mmf.holdout`) is single-use: see `docs/mmf_thu_nghiem.md`. Never edit `docs/reports/mmf_lock.json`.
