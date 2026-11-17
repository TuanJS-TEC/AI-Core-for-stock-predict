# Price Range Forecast (rangecast) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast an 80% price range and a volatility level for T+2 and T+5 per ticker, and show them on StockSenseVN in place of the up/hold/down labels.

**Architecture:** A pooled HAR regression predicts each ticker's h-session volatility; two empirical quantiles of the standardised return turn it into a band. One module evaluates it on the six frozen folds, trains a live bundle and writes a SQLite table; the existing bridge pushes that table to a new Postgres table; StockSenseVN's API and web read it instead of `forecasts`.

**Tech Stack:** Python 3.14 (`.venv`: numpy, pandas, scikit-learn, joblib, psycopg), SQLite (`data/news/nlp.db`), PostgreSQL 18, Node 22 / Fastify / React 19 / vitest (`../StockSenseVN`).

**Spec:** `docs/superpowers/specs/2026-11-13-price-range-forecast-design.md`

## Global Constraints

- No commits in either repo (owner's standing request). Steps end at "tests pass".
- One model configuration, evaluated once. No variants after seeing results (spec §2, §5).
- Never read sessions after 2026-10-30 in any evaluation; never touch `docs/reports/mmf_lock.json`, the volcast bundle, or `ai_core.models.mmf.holdout`.
- `ai_core/models/rangecast` must import without torch (cron runs it in `.venv`); `tests/test_architecture.py` must stay green.
- Prices in AI_Core are thousand VND, vendor-adjusted: only ratios to the close (`lo_pct`, `hi_pct`) cross to the web.
- StockSenseVN: add tables/fields, never change the meaning of an existing column; pushed rows are immutable (`ON CONFLICT DO NOTHING`).
- Band = central 80% (quantiles 0.10 and 0.90). Risk level: σ̂ percentile within the ticker's last 250 σ̂ values, ≥ 70 → `high`, ≤ 30 → `low`, else `normal`.
- Web copy (exact): "Biến động cao" / "Biến động bình thường" / "Biến động thấp"; note "Mô hình thử nghiệm: dự báo mức biến động và dải giá, không dự báo chiều tăng giảm"; band caption "Dải giá: 8/10 lần giá đóng cửa sau h phiên được kỳ vọng nằm trong dải. Không phải giá mục tiêu."
- Run everything from the repo root; tests: the two pytest commands of `CLAUDE.md`, plus `npm test` in `../StockSenseVN`.

## Review Focus

1. Today's session not yet synced when `predict` runs → it forecasts the newest stored session and `base_date` says so; a rerun after the sync adds the new date, never rewrites the old one. (Task 2 test)
2. A ticker with a missing bar or a > 7.2% jump inside the HAR window → no row for it, the other tickers still get theirs. (Task 2 test)
3. σ̂ far outside the training range → band still ordered (`lo_pct < 0 < hi_pct`) and inside the exchange limit `±7%` compounded over h. (Task 1 test)
4. A symbol with a range but `analysis_enabled = false`, or an old batch that has `forecasts` rows but no `forecast_ranges` → API returns `forecasts.tN = null` / old batch ignored, web renders "chưa có dự báo" without crashing. (Tasks 3, 4, 5 tests)
5. Bundle file missing or of another version → `predict` exits non-zero with a message naming the `train` command; `live_update.sh` continues to the next step. (Task 2 test)

---

### Task 1: HAR range model and its one evaluation

**Files:**
- Create: `ai_core/models/rangecast/__init__.py` (one-line docstring), `ai_core/models/rangecast/rangecast.py`
- Test: `tests/models/rangecast/__init__.py`, `tests/models/rangecast/test_rangecast.py` (run in `.venv`; add `tests/models/rangecast` to the first pytest command of `CLAUDE.md` and to the `--ignore` list of the second)
- Modify: `docs/reports/mmf_preregistration.md` (append the addendum, before the scoring run), `ai_core/paths.py` (`RANGECAST = DATA / "rangecast"`)
- Create (output): `docs/reports/rangecast_report.md`, `data/rangecast/eval.md`

**Interfaces (produced):**
- `HAR = ["har_rv_d","har_rv_w","har_rv_m","har_pk_d","har_pk_w","har_pk_m","har_gap_w"]`, `FEATS = HAR + [c + "_mkt" for c in HAR]`, `Q = (0.10, 0.90)`, `LIMIT = 0.07`, `VERSION = "rangecast-v1"`.
- `features(frames: dict[str, DataFrame]) -> DataFrame` indexed (symbol, as_of_session), columns FEATS; raw HAR terms (same formulas as `mmf.har._raw_har`, copied so the module needs no torch) plus each term's cross-sectional mean of the session.
- `targets(frames, h) -> DataFrame` indexed the same: `r` (= `make_mag_labels` realized_return), `status`, `label_end_date`, `logrv` = ln(Σ_{t+1..t+h} r1² + 1e-8).
- `fit(x: DataFrame, logrv: Series) -> LinearRegression`; `sigma(model, x) -> ndarray` = sqrt(exp(pred)).
- `z_quantiles(r, sig) -> tuple[float, float]`; `band(sig, z, h) -> (lo, hi)` log-return bounds clipped to `[h·ln(1−LIMIT), h·ln(1+LIMIT)]`.
- `evaluate(panel) -> dict` and `main(argv)` with `eval --stock-dir DIR`.

- [ ] **Step 1: Tests first** (`test_rangecast.py`), on synthetic OHLC frames:
  - `test_features_equal_mmf_har_raw_terms` (skip if torch missing): the 7 HAR columns equal `mmf.har._raw_har` per symbol.
  - `test_future_prices_do_not_change_past_features`: change bars after session k → rows ≤ k unchanged.
  - `test_band_is_ordered_and_inside_the_exchange_limit`: `band(np.array([1e-4, 0.02, 5.0]), (-1.3, 1.2), 5)` gives `lo < 0 < hi`, `lo >= 5*ln(0.93)`, `hi <= 5*ln(1.07)`.
  - `test_coverage_on_synthetic_gaussian_returns`: r = σ·ε with known σ per row, quantiles from one half → coverage on the other half within 0.80 ± 0.03.
  - `test_verdict_thresholds`: `verdict(cov2, cov5, d2_hi, d5_hi)` → "Dùng được" for (0.80, 0.79, −0.001, −0.002); "Dùng được, có lưu ý" for (0.80, 0.79, +0.001, −0.002); "Không đạt" for (0.74, 0.80, …).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/models/rangecast` → fails (module missing).
- [ ] **Step 3:** Implement the interfaces. Evaluation: panel from `--stock-dir` (the frozen long snapshot `data/prices_2018/snapshots/a0674e0c56bf9013`, tickers `prices.config.TICKERS`); folds = `prices.config.DATE_FOLDS` + `mmf.config.BACK_FOLDS` via `prices.splits.split_samples` on rows with `status == "ok"` and finite FEATS; per (h, fold): fit on train + tune, `z_quantiles` on calib, score eval. Arms: `har`; `naive20` (σ = 20-session std of r1 × √h, same calib quantiles procedure); `hist` (per-symbol train quantiles of r, no σ). Metrics per (h, arm), pooled and per fold: coverage, share below / above, mean width (hi − lo, as % of price), pinball loss at Q (sum of the two), `mmf.evaluate.paired_diff` of pinball har − naive20 and har − hist (20-session blocks, 2000 draws), coverage by symbol and by risk level. `verdict` exactly as spec §5: coverage of both h inside 0.77–0.83 and both `har − naive20` upper bounds < 0 → "Dùng được"; coverage inside 0.75–0.85 otherwise → "Dùng được, có lưu ý"; else "Không đạt". Writes `data/rangecast/eval.md`.
- [ ] **Step 4:** Tests pass; `.venv/bin/python -m pytest -q tests/test_architecture.py` passes.
- [ ] **Step 5:** Append the addendum "Phụ lục 2026-11-13 (R11: dải giá rangecast…)" to `docs/reports/mmf_preregistration.md`: copy spec §4 and §5 (model, data, folds, arms, metrics, the three verdicts, stop rule, "new target, not a variant of R7–R10"), with the timestamp, before running.
- [ ] **Step 6:** Run once: `.venv/bin/python -m ai_core.models.rangecast.rangecast eval --stock-dir data/prices_2018/snapshots/a0674e0c56bf9013`. Write `docs/reports/rangecast_report.md` (Vietnamese, same layout as `mmf_har_ev_report.md`: conclusion first, tables, limits). **Checkpoint: report the verdict to the owner.** "Không đạt" → Tasks 2–6 switch to the spec's fallback (volcast strong/weak chip, no band); stop and confirm before continuing.

### Task 2: Live bundle, daily prediction, cron step

**Files:**
- Modify: `ai_core/models/rangecast/rangecast.py`, `ai_core/news/schema.sql` (table), `jobs/live_update.sh`, `tests/test_live_schedule.py`
- Test: `tests/models/rangecast/test_rangecast.py`

**Interfaces:**
- Consumes: Task 1's `features`, `targets`, `fit`, `sigma`, `z_quantiles`, `band`.
- Produces: SQLite table `range_forecast(symbol TEXT, base_date TEXT, h INTEGER, close REAL, lo_pct REAL, hi_pct REAL, sigma REAL, risk_level TEXT CHECK (risk_level IN ('low','normal','high')), model_version TEXT, created_at TEXT, PRIMARY KEY (symbol, base_date, h, model_version))`; `lo_pct`/`hi_pct` are simple returns (exp(log bound) − 1). `BUNDLE = paths.RANGECAST / "rangecast.joblib"` = `{"version", h: {"model", "z": (z10, z90)}}`. CLI `train`, `predict`.
- `train_bundle(panel) -> dict`: per h, fit on labelled rows except the last 126 sessions, `z` from those 126.
- `predict_rows(panel, bundle) -> list[tuple]` in table column order; `risk_level(sig_hist: ndarray) -> str` on the last ≤ 250 values (fewer than 60 → `normal`).

- [ ] **Step 1: Tests:** `test_train_bundle_holds_out_the_last_126_sessions` (z computed only from them); `test_predict_uses_the_newest_stored_session_and_never_rewrites_an_older_one` (run twice with one more session → two base_dates, first rows unchanged; INSERT OR IGNORE); `test_ticker_with_a_gap_in_the_har_window_gets_no_row_others_do`; `test_risk_level_percentiles` (70th → high, 30th → low, 50th → normal, < 60 values → normal); `test_predict_without_bundle_exits_with_the_train_hint` (SystemExit message contains "rangecast train"; same for a bundle whose version differs).
- [ ] **Step 2:** Run → fail.
- [ ] **Step 3:** Implement. `predict` reads `build_panel(prices.config.TICKERS)` from the live store `data/prices` and `storage.connect()`.
- [ ] **Step 4:** `jobs/live_update.sh`: after the `predict` (volcast) line add `dry rangecast || .venv/bin/python -m ai_core.models.rangecast.rangecast predict || echo "rangecast predict failed, continuing"`; extend `tests/test_live_schedule.py`'s expected step list for the 15:00–16:00 window.
- [ ] **Step 5:** Tests pass. Run `train` then `predict` on the real store: prints 30 rows written (15 tickers × 2 h); show them.

### Task 3: Bridge to StockSenseVN

**Files:**
- Modify: `ai_core/fusion/stocksense.py`, `tests/fusion/test_stocksense.py`, `CLAUDE.md` (the StockSenseVN paragraph)
- Create: `../StockSenseVN/db/migrations/025_forecast_ranges.sql`

**Interfaces:**
- Consumes: `range_forecast` (Task 2); `fusion_decision` rows of `live.VERSION` for the news context (same symbol, base_date, h).
- Produces: Postgres `forecast_ranges(id, batch_id → market_batches ON DELETE CASCADE, symbol → symbols, horizon 't2'|'t5', low_pct numeric(8,6) < 0, high_pct numeric(8,6) > 0, coverage numeric(4,3), risk_level 'low'|'normal'|'high', has_news boolean, sentiment_24h real NULL, risk_flags text[] DEFAULT '{}', created_at, UNIQUE (batch_id, symbol, horizon))`; `market_summary` gains `vol_low`, `vol_normal`, `vol_high integer NOT NULL DEFAULT 0`. Batches use `model_version = rangecast.VERSION`, state `experimental`.
- `push_ranges(sq, pg, version=rangecast.VERSION, since=None) -> int`; `main()` calls it instead of `push`. `push`/`to_forecast` stay (unused by the cron) so the old path can be restored.

- [ ] **Step 1: Tests** (Postgres ones follow the file's existing fixture and skip without `STOCKSENSE_TEST_DATABASE_URL`): rows inserted with the batch; rerun inserts 0 and changes nothing; symbol not `analysis_enabled` skipped; no batch when nothing lands; a range without a `fusion_decision` row is pushed with `has_news = false`, `sentiment_24h NULL`; `market_summary.vol_*` count the t5 levels; nothing is written to `forecasts`.
- [ ] **Step 2:** Run → fail. **Step 3:** Write the migration and `push_ranges`. **Step 4:** Tests pass (ask the owner to export `STOCKSENSE_TEST_DATABASE_URL` if unset; apply the migration to the local database with StockSenseVN's `npm run db:migrate`). **Step 5:** Run `python -m ai_core.fusion.stocksense` once; `SELECT count(*) FROM forecast_ranges` = 30.

### Task 4: StockSenseVN API

**Files (in `../StockSenseVN`):** `apps/api/src/market/repository.js`, its test file beside it, `openapi.yaml`, any `mocks/` fixture the web's mock mode reads for forecasts.

**Interfaces:**
- Produces: every `forecasts.t2` / `forecasts.t5` object becomes `{ riskLevel: 'low'|'normal'|'high', range: { lowPct, highPct, coverage }, news: { hasNews, sentiment24h, riskFlags }, dataDate, stale, label: null, probabilities: null, confidence: null }`, or `null` when the symbol has no range in the last 7 days of the newest batch's model version. Market overview returns `volatilityLevels: { low, normal, high }` beside the old `labels` (zeros).

- [ ] **Step 1: Tests:** symbol with ranges → the shape above with numbers; symbol without → `null`; newest batch of another model version with only `forecasts` rows → `null`, no error; stale tag when the range is from an older batch date.
- [ ] **Step 2:** fail → **Step 3:** switch `forecastJoin`/`FORECAST_COLS`/`forecastOf` to `forecast_ranges`, update `openapi.yaml` (old fields nullable, new fields documented) → **Step 4:** `npm test` passes.

### Task 5: Web display

**Files (in `../StockSenseVN`):** `packages/core/src/labels/forecast.js`, `packages/core/src/labels/disclaimer.js`, `apps/web/src/components/forecast.jsx`, `apps/web/src/components/ForecastBadge.jsx`, `apps/web/src/features/symbol-detail/SymbolDetailPage.jsx`, `apps/web/src/features/market-overview/OverviewPage.jsx`, `apps/web/src/app/AppShell.jsx`, `apps/web/src/app/TabbedHub.jsx`, `apps/web/src/features/chat/ChatPage.jsx`, `apps/api/src/chat/assistant.js`, their tests.

**Interfaces:**
- Consumes: Task 4's forecast object.
- Produces (core): `VOLATILITY_LABELS = { low: { text: 'Biến động thấp', symbol: '◔', tone: 'hold' }, normal: { text: 'Biến động bình thường', symbol: '◑', tone: 'hold' }, high: { text: 'Biến động cao', symbol: '◕', tone: 'warn' } }`; `rangePrices(close, range) -> { low, high }`; `newsContext(news, changePercent) -> string` returning "Không có tin trong 72 giờ" | "Có tin trong 72 giờ, không có tin trong 24 giờ" | "Tin 24h: tích cực|tiêu cực|trung tính (±0.00), cùng chiều|trái chiều với biến động giá hôm nay" (neutral band |s| < 0.1; "cùng chiều/trái chiều" only when both the sentiment and the day's change are non-neutral).

- [ ] **Step 1: Tests** for `rangePrices` (122.5, −0.038/+0.036 → 117.845 / 126.91), `newsContext` (each branch), badge renders the three levels and "chưa có dự báo" for `null`, no "Tăng"/"Giảm"/"Đi ngang" forecast text left in the rendered symbol page and overview.
- [ ] **Step 2:** fail → **Step 3:** implement: badge + band line + news line + risk-flag warning on every forecast card; overview counts the three levels; experimental note and band caption from Global Constraints; chat assistant sentences use level and band → **Step 4:** `npm test` passes; open the symbol page and the overview on port 3100 and look at them.

### Task 6: Trade Plan stop-loss hint

**Files (in `../StockSenseVN`):** `apps/web/src/features/plans/PlansPage.jsx`, its test.

- [ ] **Step 1: Test:** with a symbol whose t5 range exists, the form shows "Dải giá 80% T+5: low – high" under "Cắt lỗ" and a button that sets `stop_price` to `low` for `side: 'long'` and `high` for `side: 'short'`; no range → nothing rendered; the field stays editable.
- [ ] **Step 2–4:** fail, implement with `rangePrices`, `npm test` passes, try it in the browser.

### Task 7 (only if Tasks 1–6 are done): fundamentals block

Push the latest known ratios per symbol (`mmf.fundamentals.add_fundamentals`: E/P, ROE, D/E, EPS and revenue growth, statement year) to a small Postgres table, one API field, one block on the symbol page. Plan the details when reached.

### Task 8: Final verification

- [ ] Both pytest commands of `CLAUDE.md` (update its expected counts and add the rangecast paragraph); `npm test` in StockSenseVN.
- [ ] `LIVE_DRY=1 LIVE_CLOCK="2026-11-13 15:30" jobs/live_update.sh` lists `rangecast` between `predict` and `live`.
- [ ] Browser check on port 3100: symbol page, overview, chat and Trade Plan per spec §9; note for the web team about mobile and PDF (old fields now null).
