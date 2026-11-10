# AI_Core

Vietnamese stock research (HOSE): prices + news + financials → T+2/T+5 forecasts. One package, `ai_core`, installed editable in two venvs (`pip install -e .`).

```
ai_core/prices   price data layer (sync, validation, snapshots, features, labels, splits)   ← was Pipeline_V1.2/src
ai_core/news     news → sentiment → point-in-time text features (SQLite nlp.db)             ← was NLP_Module/nlp_module
ai_core/models   mmf (multimodal trend forecast) and volcast (volatility forecast)
ai_core/fusion   live decision board, shadow report, direction GO/NO-GO, Finhay sync, Streamlit app
ai_core/paths.py the only module that knows where things live
jobs/            cron entry points (thin); tests/ mirrors the packages
```

Dependencies are one-way and enforced by `tests/test_architecture.py`: `prices` and `news` never import each other; `models` import them; `fusion` sits on top. `ai_core/__init__.py`, `prices/__init__.py` and `fusion/__init__.py` stay empty (the Finhay and cross-check crons must not drag in torch).

Data stays in its old homes: `Pipeline_V1.2/stock` (prices), `NLP_Module/data` (nlp.db, models, tessdata), `data/` (finhay, mmf). Run news commands from `NLP_Module/`, everything else from the repo root.

Price sync (`jobs/sync_prices.py`, cron 15:20) runs on `~/.venv_6x` (vnstock generation 6, source tcbs, preview build valid until 2026-12-07) with `PYTHONPATH=<repo root>`; tcbs needs `vnstock auth tcbs` (Smart OTP) before each run, otherwise no new prices are fetched.

Tests (two venvs; expected 201 + 196 passed):
```
NLP_Module/.venv/bin/python -m pytest -q tests/prices tests/news tests/models/volcast tests/fusion/test_board.py
.venv_mmf/bin/python -m pytest -q tests --ignore=tests/prices --ignore=tests/news --ignore=tests/models/volcast --ignore=tests/fusion/test_board.py
```
The mmf holdout (`ai_core.models.mmf.holdout`) is single-use: see `docs/mmf_thu_nghiem.md`. Never edit `docs/reports/mmf_lock.json`.
