# Cầu nối AI_Core → StockSenseVN: kế hoạch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Sau khi duyệt, chép file này vào `docs/superpowers/plans/2026-11-12-stocksense-bridge.md`.

**Goal:** kéo StockSenseVN (https://github.com/trongdaiCN178/StockSenseVN) về máy, chạy được, và để bảng dự báo T+2/T+5 của nó hiện dự báo thật từ model của AI_Core thay cho dữ liệu mẫu.

**Architecture:** StockSenseVN đã định sẵn ranh giới (SDD D2, D3, mục 10.2): "worker mô hình" Python chỉ giao tiếp qua PostgreSQL, ghi `market_batches`, `forecasts`, `market_summary`. API của họ luôn đọc batch có `data_date` mới nhất. Bảng `fusion_decision` (SQLite `nlp.db`, `ai_core/fusion/live.py`, version `decision-v1-trend`) đã cho đúng nhãn tăng/đi ngang/giảm theo (mã, h). Vì vậy chỉ cần **một module** `ai_core/fusion/stocksense.py`, đọc `fusion_decision` và ghi sang Postgres, rồi gọi nó trong `jobs/live_update.sh` ngay sau `ai_core.fusion.live`. Không sửa code Node/React của StockSenseVN.

**Tech Stack:** Python 3 `.venv` + `psycopg[binary]` 3.x (dependency mới duy nhất); StockSenseVN: Node ≥ 22.12, PostgreSQL (dùng server 18 đang chạy sẵn trên máy, máy không có Docker).

**Spec:** không có spec riêng. Nguồn hợp đồng: StockSenseVN `Tài liệu chạy dự án/Software_Design_Document_v0.0.5.md` (D2, D3, D4, mục 6.3, 10.2), `db/migrations/001_initial_schema.sql` (bảng `market_batches`, `market_summary`, `forecasts`), `apps/api/src/market/repository.js` (`latestBatch`, `board`, `symbolSelect`). Phía AI_Core: docstring của `ai_core/fusion/live.py`.

## Quyết định đã chốt (đổi được, nói nếu không đồng ý)

1. **Vị trí:** clone StockSenseVN thành repo anh em `/home/administrator/TLS/tuanhm@linux/StockSenseVN`, không nhét vào AI_Core (hai stack và hai lịch sử git khác nhau). Cầu nối nằm trong AI_Core vì nó import `ai_core`.
2. **Model đưa lên là `decision-v1-trend`** (volcast cho đi ngang/có biến động, model hướng cho tăng/giảm). Model hướng đang **NO-GO** (`docs/reports/direction_report.md`), nên tăng/giảm không có lợi thế đã chứng minh. Vì vậy `confidence` không bao giờ là `high`, `state` của batch là `experimental`. mmf không lên vì nó chưa có đường suy luận live (holdout chỉ dùng một lần).
3. **Không commit** cho tới khi chủ dự án cho phép: `live.py` (version `decision-v1-trend`) mà cầu nối dựa vào vẫn đang chưa commit trên nhánh `multimodal-decision`.

## Global Constraints

- `ai_core/__init__.py`, `prices/__init__.py`, `fusion/__init__.py` để trống. `stocksense.py` không import torch, không import gì từ `models/mmf`.
- Luồng một chiều: `stocksense.py` nằm ở `fusion`, chỉ đọc SQLite và ghi Postgres. `tests/test_architecture.py` phải vẫn xanh.
- Chuỗi kết nối lấy từ env `STOCKSENSE_DATABASE_URL`. Thiếu biến này thì `main()` in một dòng rồi thoát 0, cron live vẫn chạy tiếp. Không ghi mật khẩu vào repo.
- Postgres `forecasts`: `probability_*` kiểu `numeric(8,7)`, `CHECK (abs(up+hold+down-1) < 0.0001)`, `label IN ('up','hold','down')`, `horizon IN ('t2','t5')`, `confidence IN ('low','mid','high')`, `UNIQUE (batch_id, symbol, horizon)`, FK `symbol → symbols`.
- Dự báo đã ghi là bất biến (SDD 6.3): chạy lại dùng `ON CONFLICT DO NOTHING`, không sửa xác suất cũ.
- Không chạy `ai_core.models.mmf.holdout`, không sửa `docs/reports/mmf_lock.json`.

## Review Focus

1. **Chạy trước giờ sync 15:20** (mọi dòng mới là `DATA_INVALID`, hoặc chưa có volcast): không được tạo batch rỗng cho ngày mới. Batch rỗng thành "mới nhất" sẽ xóa trắng dự báo trên cả web. Test: `test_push_skips_invalid_and_incomplete`.
2. **Chạy lại cùng ngày** (cron 15 phút một lần, 15:00–16:00): không nhân đôi dòng, không đổi xác suất đã ghi. Test: `test_push_is_idempotent`.
3. **Mã có trong `UNIVERSE` nhưng StockSenseVN chỉ hiện giá hoặc không có** (BID, CTG, HDB, VHM, NVL): bỏ qua, không lỗi FK. Test: `test_push_only_analysis_enabled`.
4. **Xác suất ở biên** (`p_strong` 0 hoặc 1, `p_up + p_down = 0`): vẫn qua CHECK sau khi làm tròn 7 chữ số. Test: `test_probs_edge_cases`.
5. **Postgres không với tới hoặc thiếu env:** `live_update.sh` in lỗi rồi chạy tiếp, `fusion_decision` không bị ảnh hưởng. Kiểm bằng tay ở Task 3.

---

### Task 1: Clone và chạy StockSenseVN trên máy này

Không có code, chỉ là môi trường. Một số bước cần `sudo`: chủ dự án tự chạy.

- [ ] `git clone https://github.com/trongdaiCN178/StockSenseVN.git /home/administrator/TLS/tuanhm@linux/StockSenseVN`
- [ ] Node: `nvm install 22 && nvm use 22`. Kiểm tra: `node --version` ≥ v22.12.
- [ ] Postgres (dùng server có sẵn, không cần Docker, **chủ dự án chạy**): `sudo -u postgres createuser -P stocksense` (mật khẩu tự đặt), rồi `sudo -u postgres createdb -O stocksense stocksense`.
- [ ] `cd StockSenseVN && npm install && cp .env.example .env && cp apps/web/.env.real.example apps/web/.env.local`. Trong `.env`: sửa `DATABASE_URL` theo mật khẩu vừa đặt, và đặt `AUTH_RATE_LIMIT_SECRET` (≥ 32 ký tự ngẫu nhiên).
- [ ] `npm run db:migrate && npm run db:seed:demo`. Rồi nạp giá thật: `cd apps/worker && node --env-file=../../.env src/cli.js --from 2021-10-27`.
- [ ] `npm run dev:api` và `npm run dev:web`, mở http://localhost:5173, đăng nhập `retail@stocksense.vn` / `retail12345678`. Kỳ vọng: bảng giá có giá SSI, thẻ dự báo hiện "(dữ liệu mẫu, mô hình AI chưa triển khai)", ngày dữ liệu 2026-10-07, model `demo-0.0.1`.
- [ ] `npm test` xanh. Đây là mốc trước khi tích hợp.

### Task 2: `ai_core/fusion/stocksense.py`

**Files:**
- Create: `ai_core/fusion/stocksense.py`
- Create: `tests/fusion/test_stocksense.py`
- Modify: `requirements/news.txt` (thêm `psycopg[binary]>=3.2`), rồi `.venv/bin/pip install 'psycopg[binary]>=3.2'`

**Interfaces:**
- Consumes: bảng SQLite `fusion_decision` (cột: `symbol, base_date, h, decision, p_up, p_down, margin, p_strong, call, has_news, sentiment_mean_24h, risk_flags, reason, version, created_at`), `ai_core.news.storage.connect()`, `ai_core.fusion.live.VERSION`.
- Produces:
  - `to_forecast(row: Mapping) -> dict | None`: trả `{"symbol", "data_date", "horizon", "label", "up", "hold", "down", "confidence"}`, hoặc `None` nếu không đưa lên.
  - `push(sq: sqlite3.Connection, pg: psycopg.Connection, version: str = live.VERSION) -> int`: số dòng `forecasts` mới được chèn.
  - `main() -> None`: đọc `STOCKSENSE_DATABASE_URL`, gọi `push`, in một dòng tóm tắt. Chạy bằng `python -m ai_core.fusion.stocksense`.

**Quy tắc ánh xạ (đây là các quyết định, phần còn lại implementer tự viết):**
- `None` khi `decision == "DATA_INVALID"`, hoặc `p_strong` là NULL, hoặc `p_up` là NULL. Không đẩy dòng thiếu, để tránh một dòng dở dang bị `DO NOTHING` đóng băng.
- `label`: `LONG→up`, `AVOID→down`, `NO_TRADE→hold`. `horizon`: `2→"t2"`, `5→"t5"`. `data_date = base_date`.
- Xác suất: `hold = 1 − p_strong`. Phần `p_strong` chia theo tỷ lệ `p_up : p_down` (nếu tổng bằng 0 thì chia đôi). Làm tròn `up` và `hold` 7 chữ số, `down = 1 − up − hold`, kẹp vào [0, 1]. Nhãn không bắt buộc là argmax của ba số: nhãn do `live.trend` quyết. Thẻ của StockSenseVN tự hiện "Độ tin cậy thấp" khi xác suất của nhãn < 0.45.
- `confidence`: `"mid"` khi `label == "hold"` và `call == "weak"` (volcast vượt ngưỡng tin cậy riêng của nó). Mọi trường hợp khác là `"low"`. Không bao giờ `"high"` (model hướng NO-GO).

**SQL trong `push` (một transaction cho mỗi `data_date` có ít nhất một dòng hợp lệ):**
- `INSERT INTO market_batches (data_date, batch_completed_at, model_version, state, source_status) VALUES (%s, now(), %s, 'experimental', 'ai_core') ON CONFLICT (data_date, model_version) DO UPDATE SET batch_completed_at = EXCLUDED.batch_completed_at RETURNING id`
- `INSERT INTO forecasts (batch_id, symbol, horizon, label, probability_up, probability_hold, probability_down, confidence) SELECT … WHERE EXISTS (SELECT 1 FROM symbols WHERE symbol = %s AND analysis_enabled) ON CONFLICT (batch_id, symbol, horizon) DO NOTHING`
- Sau đó upsert `market_summary` của batch: `labels_up/hold/down` = số dòng `t2` theo nhãn trong `forecasts` của batch đó (đếm như `mocks/market.js:80`; implementer kiểm lại `row[7]` là nhãn T+2), còn `advances/nochanges/declines = 0`. API chỉ dùng ba cột này khi bảng giá trống (`repository.js` dòng ~154).
- Chỉ đọc `fusion_decision WHERE version = %s`. Nếu không có dòng hợp lệ nào thì không tạo batch.

- [ ] **Step 1: Viết test đỏ** trong `tests/fusion/test_stocksense.py`.

Test thuần (không cần Postgres):
```python
def test_to_forecast_maps_labels_and_probs():
    f = to_forecast(row(decision="LONG", h=2, p_up=0.6, p_down=0.2, p_strong=0.7, call=None))
    assert (f["label"], f["horizon"]) == ("up", "t2")
    assert f["hold"] == pytest.approx(0.3) and f["up"] == pytest.approx(0.525) and f["down"] == pytest.approx(0.175)
    assert f["confidence"] == "low"

def test_to_forecast_hold_mid_only_with_weak_call():
    assert to_forecast(row(decision="NO_TRADE", p_strong=0.3, call="weak"))["confidence"] == "mid"
    assert to_forecast(row(decision="NO_TRADE", p_strong=0.3, call=None))["confidence"] == "low"

def test_to_forecast_skips_invalid_and_incomplete():
    assert to_forecast(row(decision="DATA_INVALID")) is None
    assert to_forecast(row(p_strong=None)) is None
    assert to_forecast(row(p_up=None, p_down=None)) is None

@pytest.mark.parametrize("ps,up,down", [(0.0, 0.5, 0.2), (1.0, 0.5, 0.2), (0.6, 0.0, 0.0), (0.123456789, 0.333333, 0.333333)])
def test_probs_edge_cases(ps, up, down):
    f = to_forecast(row(p_strong=ps, p_up=up, p_down=down))
    assert all(0 <= f[k] <= 1 for k in ("up", "hold", "down"))
    assert abs(f["up"] + f["hold"] + f["down"] - 1) < 1e-4
    assert all(round(f[k], 7) == f[k] for k in ("up", "hold", "down"))
```
(`row(**kw)` là một helper trong file test, trả dict đủ cột với giá trị mặc định hợp lệ: `symbol="FPT", base_date="2026-11-12", h=2, decision="LONG", p_up=0.5, p_down=0.3, p_strong=0.6, call=None, version="decision-v1-trend"`.)

Test Postgres: fixture `pg` bỏ qua (`pytest.skip`) khi không có `STOCKSENSE_TEST_DATABASE_URL`. Fixture tạo một schema tạm, chép DDL 4 bảng `symbols` (thêm `analysis_enabled boolean`), `market_batches`, `market_summary`, `forecasts` từ `001_initial_schema.sql` (giữ nguyên CHECK và UNIQUE), seed `FPT` (bật), `BID` (tắt), rồi drop schema sau test. SQLite `sq` là `:memory:` có bảng `fusion_decision` như `storage.py:40`.
```python
def test_push_writes_batch_forecasts_summary(pg, sq):   # 2 dòng FPT h=2,5 hợp lệ
    assert push(sq, pg) == 2
    assert one(pg, "SELECT data_date::text, model_version, state FROM market_batches") == ("2026-11-12", "decision-v1-trend", "experimental")
    assert one(pg, "SELECT labels_up + labels_hold + labels_down FROM market_summary") == (1,)

def test_push_is_idempotent(pg, sq):
    push(sq, pg); before = all_rows(pg, "SELECT * FROM forecasts ORDER BY horizon")
    sq.execute("UPDATE fusion_decision SET p_strong = 0.99")          # rerun with different numbers
    assert push(sq, pg) == 0 and all_rows(pg, "SELECT * FROM forecasts ORDER BY horizon") == before
    assert one(pg, "SELECT count(*) FROM market_batches") == (1,)

def test_push_skips_invalid_and_incomplete(pg, sq):        # only DATA_INVALID / p_strong NULL rows
    assert push(sq, pg) == 0
    assert one(pg, "SELECT count(*) FROM market_batches") == (0,)

def test_push_only_analysis_enabled(pg, sq):               # FPT + BID (off) + NVL (absent)
    assert push(sq, pg) == 2
    assert one(pg, "SELECT array_agg(DISTINCT symbol) FROM forecasts") == (["FPT"],)
```

- [ ] **Step 2: Chạy, thấy đỏ.** `.venv/bin/python -m pytest -q tests/fusion/test_stocksense.py`. Kỳ vọng: lỗi `ModuleNotFoundError: ai_core.fusion.stocksense`.
- [ ] **Step 3: Cài `to_forecast`, `push`, `main`** theo các quy tắc ở trên. Module docstring ghi: nguồn, đích, quy tắc bất biến, và lý do không có `high`.
- [ ] **Step 4: Chạy xanh, cả hai chế độ.** Không có env: 4 test thuần pass, 4 test Postgres skip. Có env (chủ dự án tạo `createdb stocksense_test`): `STOCKSENSE_TEST_DATABASE_URL=postgresql://stocksense:…@127.0.0.1/stocksense_test .venv/bin/python -m pytest -q tests/fusion/test_stocksense.py`, kỳ vọng 8 pass (hoặc 11 nếu tính từng tham số).
- [ ] **Step 5: Bộ test `.venv` đầy đủ** (lệnh trong CLAUDE.md, thêm `tests/fusion/test_stocksense.py`) vẫn xanh, kể cả `tests/test_architecture.py`.
- [ ] **Step 6: Commit** (chỉ khi chủ dự án cho phép): `git add ai_core/fusion/stocksense.py tests/fusion/test_stocksense.py requirements/news.txt && git commit -m "feat(fusion): push decision board to StockSenseVN forecasts"`

### Task 3: Nối vào cron live và kiểm end-to-end

**Files:**
- Modify: `jobs/live_update.sh` (dòng cuối, trong cùng khối 15:00–16:00)
- Modify: `CLAUDE.md` (lệnh test `.venv` thêm `tests/fusion/test_stocksense.py`, số test kỳ vọng mới, một dòng về `STOCKSENSE_DATABASE_URL`)
- Tạo (không track): `env/.env` hoặc dòng export trong crontab cho `STOCKSENSE_DATABASE_URL` (chủ dự án điền mật khẩu)

- [ ] **Step 1:** sau `python -m ai_core.fusion.live || echo …`, thêm `; .venv/bin/python -m ai_core.fusion.stocksense || echo "stocksense push failed, continuing"`.
- [ ] **Step 2: Chạy tay một lần** với dữ liệu hôm nay: `STOCKSENSE_DATABASE_URL=… .venv/bin/python -m ai_core.fusion.stocksense`. Kỳ vọng một dòng kiểu `stocksense: batch 2026-11-12 decision-v1-trend, 30 forecasts inserted` (15 mã × 2 h, nếu volcast đủ 15 mã).
- [ ] **Step 3: Chạy lại ngay**: kỳ vọng `0 forecasts inserted`, `SELECT count(*) FROM market_batches WHERE model_version='decision-v1-trend'` không đổi.
- [ ] **Step 4: Kiểm trên web:** `curl -s http://127.0.0.1:3000/api/v1/market/status` (hoặc đường dẫn status trong `openapi.yaml`) trả `modelVersion: decision-v1-trend`, `state: experimental`. Trên giao diện, thẻ T+2/T+5 của FPT hiện nhãn và xác suất khớp `SELECT * FROM fusion_decision WHERE symbol='FPT' AND version='decision-v1-trend' ORDER BY base_date DESC LIMIT 2`, và không còn dòng "(dữ liệu mẫu…)".
- [ ] **Step 5: Lỗi không lan:** chạy với `STOCKSENSE_DATABASE_URL=postgresql://x@127.0.0.1:1/x`. Kỳ vọng exit ≠ 0 và một dòng lỗi, còn `bash jobs/live_update.sh` vẫn chạy hết. Không có env: in "STOCKSENSE_DATABASE_URL not set, skip" và exit 0.
- [ ] **Step 6: Commit** (chỉ khi được phép): `git add jobs/live_update.sh CLAUDE.md && git commit -m "chore: push the live board to StockSenseVN after each decision write"`

## Ngoài phạm vi (làm sau nếu cần)

- Bảng giải thích, `forecast_input`, replay, hit rate (StockSenseVN M3, M6, M7). Cột `reason` của `fusion_decision` đã có câu tiếng Việt, có thể đưa vào khi họ có bảng `forecast_explanation`.
- Sentiment tin cho `news_items.sentiment` (hai bên dùng nguồn tin khác nhau, phải ghép URL trước).
- Docker hóa worker mô hình, HTTPS, đưa mmf lên (cần holdout xong và cấu hình đã khóa).
