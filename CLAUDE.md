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

Price sync (`jobs/sync_prices.py`, cron 15:20) runs on `.venv` (vnstock 4.x, source kbs, no key or OTP needed). `sync.py` also works on `~/.venv_6x` (vnstock 6.0.0, source tcbs, needs `vnstock auth tcbs` and an unlocked keyring); the cross-check cron uses that one. vnstock 6 and 4 cannot share a venv. The vendor disabled all pre-6.0.0 builds on 2026-11-11 (rc4 stopped working then); 4.0.9 still worked that day but may be cut off, in which case move the sync back to `.venv_6x` or put the tcbs key in env/.env.

Tests (two venvs, run from the repo root; expected 258 passed + 22 skipped (280 passed with Postgres), and 280 passed; the 22 Postgres tests of
`tests/fusion/test_stocksense.py` and `test_stocksense_news.py` run when `STOCKSENSE_TEST_DATABASE_URL` points at a database they may create a schema in):
```
.venv/bin/python -m pytest -q tests/prices tests/news tests/models/volcast tests/models/rangecast tests/fusion/test_board.py tests/fusion/test_stocksense.py tests/fusion/test_stocksense_news.py tests/test_architecture.py tests/test_live_schedule.py tests/test_paths.py
.venv_mmf/bin/python -m pytest -q tests --ignore=tests/prices --ignore=tests/news --ignore=tests/models/volcast --ignore=tests/models/rangecast --ignore=tests/fusion/test_board.py --ignore=tests/fusion/test_stocksense.py --ignore=tests/fusion/test_stocksense_news.py --ignore=tests/test_architecture.py --ignore=tests/test_live_schedule.py --ignore=tests/test_paths.py
```
The mmf holdout (`ai_core.models.mmf.holdout`) is single-use: see `docs/mmf_thu_nghiem.md`. Never edit `docs/reports/mmf_lock.json`.
Corporate-event calendar (VCI, via vnstock 4 in `.venv`): `python -m ai_core.prices.events` writes `data/prices/events/<day>.parquet`; meant as a weekday 15:50 cron (not installed yet) so forward checks have point-in-time events. R9 (HAR + events for the mag label): `docs/reports/mmf_har_ev_report.md`, run once on 2026-11-12, do not rerun variants.

rangecast (`ai_core/models/rangecast`, `.venv`, no torch): 80% price band + volatility level for T+2/T+5 from a pooled HAR
regression; `eval` (R11, run once 2026-11-13: `docs/reports/rangecast_report.md`, do not rerun variants), `train` (bundle
`data/rangecast/rangecast.joblib`, fit on `data/prices`), `predict` (SQLite `range_forecast`, rows never rewritten; cron step
in `jobs/live_update.sh` after volcast).

TCBS active buy/sell share (`ai_core/fusion/tcbs_flow.py`, `~/.venv_6x`, needs a live TCBS session): the source keeps about one
month, so `data/tcbs/supply_demand.parquet` grows only by running it every session; meant as a weekday 15:45 cron (not installed yet).
Quarterly statements with real publication dates (VCI raw endpoint, 35 quarters from 2018) sit in `data/reference/bctc_quarter_vci_*.parquet`;
loader `ai_core/models/mmf/fund_quarter.py`. R12 (quarterly instead of annual fund branch, `ai_core/models/mmf/r12.py`, tables
`data/mmf/exp/r12/r12.md`) was run once on 2026-11-14: no proven gain, do not rerun variants.

StockSenseVN (web app, sibling repo `../StockSenseVN`, Node 22 + PostgreSQL) shows our forecasts: `jobs/live_update.sh` runs
`python -m ai_core.fusion.stocksense` after `ai_core.fusion.live`, pushing `range_forecast` (rangecast-v1, with the news snapshot
of `fusion_decision`) into its `market_batches`/`forecast_ranges`/`market_summary` (its migration 025). Since 2026-11-13 the web
shows the volatility level and price range instead of up/hold/down; `fusion_decision` (decision-v1-trend) is no longer pushed
(`stocksense.push` is kept, unused). The connection string is `STOCKSENSE_DATABASE_URL` in the git-ignored `.env`
at the repo root; unset means skip. Changes made on the StockSenseVN side (uncommitted there): migration 022b (023 failed
on a fresh database), per-symbol newest forecast with a stale tag in `apps/api/src/market/repository.js`, and a
"mô hình thử nghiệm" note for state 'experimental'. Its API runs on port 3100 here (3000 is taken).
