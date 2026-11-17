# Tin tức + NLP trên StockSenseVN: kế hoạch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** trang chi tiết mã của StockSenseVN hiện tin thật của mã từ kho tin AI_Core kèm kết quả PhoBERT (phương án B), lấy theo khoảng ngày (mặc định 7 ngày), tự cập nhật vài giây sau khi NLP chấm xong; vòng tin chạy mỗi 2 phút trong 07:00–22:00.

**Architecture:** AI_Core thêm `ai_core/fusion/stocksense_news.py`. Mỗi vòng live, module này đọc kết quả PhoBERT mới hoặc đổi trong `nlp.db` (theo mốc `processed_at`), upsert vào bảng mới `nlp_news` trong Postgres của StockSenseVN, rồi `pg_notify('nlp_news', '<các mã>')`. API Node mở một kết nối `LISTEN nlp_news` và phát lại sự kiện qua SSE cho các trình duyệt đang xem mã đó; trình duyệt tải lại phần tin. Hai bên không gọi trực tiếp nhau (SDD D3).

**Tech Stack:** Python 3 `.venv` + psycopg 3 + SQLite; Node 22, Fastify 5, `pg`, React 19 + Vite, vitest; PostgreSQL 18.

**Spec:** `docs/superpowers/specs/2026-11-12-stocksense-news-nlp-design.md`

## Global Constraints

- Nguồn tin: kho tin AI_Core (`data/news/nlp.db`), chỉ kết quả `nlp_model_version = 'phobert-phobert-v1'`, chỉ bài đại diện (`document_version.is_representative = 1`), chỉ phiên bản mới nhất của mỗi bài cho mỗi mã.
- Bảng `nlp_news` giữ toàn bộ lịch sử; mỗi vòng chỉ đẩy phần mới hoặc đổi.
- Hai bên chỉ giao tiếp qua PostgreSQL (SDD D3): không HTTP giữa Python và Node.
- Tab Tin tức giữ hai ô "Từ ngày / Đến ngày", mặc định 7 ngày gần nhất (hôm nay − 7 → hôm nay, giờ Việt Nam). Không có dòng tóm tắt đầu tab.
- Mọi phần tin trên trang chi tiết mã (tab Tin tức, thẻ "Tin mới nhất", chấm trên biểu đồ nến, "Tin tức: N bài", chip "Thiếu tin") lấy từ `nlp_news` và cập nhật cùng lúc khi có sự kiện.
- Bài model abstain hiện **"Chưa chắc"**, không bao giờ hiện là "Trung tính".
- Nhịp vòng tin: 2 phút trong 07:00–22:00, 15 phút ngoài khung đó. Các bước khác của `jobs/live_update.sh` giữ cửa sổ giờ cũ và nhịp 15 phút.
- AI_Core: `tests/test_architecture.py` xanh; `fusion/__init__.py` trống; chuỗi kết nối `STOCKSENSE_DATABASE_URL` trong `.env`, đọc bằng `sed`; kết nối Postgres dùng chung timeout của `stocksense.connect`.
- StockSenseVN: migration mới, không sửa file đã áp dụng; SQL tham số hóa; endpoint mới dùng quyền `news.read`; `npm test` và `npm run build:web` xanh.
- Không commit ở repo nào khi chủ dự án chưa cho phép (các bước commit chỉ chạy khi được phép).

## Review Focus

1. **Bài được sửa (hai phiên bản):** chỉ hiện một dòng, là phiên bản mới nhất. Test: `test_latest_version_wins` (Task 1).
2. **Mốc `processed_at` trùng giây:** bài chấm cùng giây với mốc vẫn được đẩy; chạy lại không phát lại sự kiện. Test: `test_watermark_boundary_and_quiet_rerun` (Task 1).
3. **Biên ngày giờ Việt Nam:** bài đăng 23:30 VN ngày `to` phải có; bài 00:30 VN ngày hôm sau thì không. Test: `vnDayRange` trong `apps/api/src/market/nlp-news.test.js` (Task 3).
4. **Mất kết nối `LISTEN`** (Postgres khởi động lại): API tự nghe lại, trình duyệt không cần tải lại trang. Test: `startNewsListener reconnects after error` (Task 4).
5. **Bài "chưa chắc" có `p_neu` lớn nhất:** vẫn hiện "Chưa chắc". Test: `sentimentOf` trong `packages/core/src/labels/news.test.js` (Task 5).

---

### Task 1: Bảng `nlp_news` và cầu nối đẩy tin

**Files:**
- Create: `../StockSenseVN/db/migrations/024_nlp_news.sql`
- Modify: `ai_core/fusion/stocksense.py` (tách `connect`)
- Create: `ai_core/fusion/stocksense_news.py`
- Test: `tests/fusion/test_stocksense_news.py`

**Interfaces:**
- Produces (Postgres, hợp đồng cho Task 3–5): migration `024_nlp_news.sql`:
```sql
CREATE TABLE nlp_news (
  symbol varchar(12) NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
  document_id text NOT NULL,
  document_version_id text NOT NULL,
  source text NOT NULL,
  url text,
  title text NOT NULL,
  published_at timestamptz NOT NULL,
  relation text NOT NULL CHECK (relation IN ('direct', 'sector', 'market', 'ambiguous')),
  p_pos real NOT NULL, p_neu real NOT NULL, p_neg real NOT NULL,
  uncertain boolean NOT NULL,
  risk_flags text[] NOT NULL DEFAULT '{}',
  passage text NOT NULL,
  model_version text NOT NULL,
  processed_at timestamptz NOT NULL,
  PRIMARY KEY (symbol, document_id)
);
CREATE INDEX nlp_news_symbol_published_idx ON nlp_news (symbol, published_at DESC);
CREATE INDEX nlp_news_processed_idx ON nlp_news (processed_at);
```
- Produces: kênh `pg_notify('nlp_news', payload)`, `payload` = các mã đã đổi, nối bằng dấu phẩy, sắp xếp (ví dụ `"FPT,VNM"`).
- Produces (Python):
  - `stocksense.connect(url: str) -> psycopg.Connection`: `autocommit=True, connect_timeout=10, options="-c statement_timeout=60000 -c lock_timeout=30000"`; `stocksense.main` dùng nó (test timeout hiện có vẫn phải xanh).
  - `stocksense_news.MODEL = "phobert-phobert-v1"`
  - `rows_since(sq, since: str | None, model: str = MODEL) -> list[dict]`: kết quả có `processed_at >= since` (chuỗi ISO `YYYY-MM-DDTHH:MM:SSZ`, `None` = tất cả). Chỉ bài đại diện, có tiêu đề, có `published_at`; mỗi `(document_id, symbol)` chỉ lấy phiên bản có `version_first_seen_at` lớn nhất. Khóa dict trùng tên cột `nlp_news`; `risk_flags` = `ai_core.news.risk.risk_flags(passage_text)`.
  - `push_news(sq, pg, model: str = MODEL) -> list[str]`: mốc = `max(processed_at)` trong `nlp_news` (định dạng lại thành chuỗi ISO UTC như trên), lấy `rows_since(sq, mốc)`, bỏ mã không có trong `symbols`, upsert `ON CONFLICT (symbol, document_id) DO UPDATE ... WHERE nlp_news.processed_at < EXCLUDED.processed_at RETURNING symbol`. Nếu có mã đổi thì `pg_notify` trong cùng transaction. Trả các mã đã đổi, sắp xếp, không trùng.
  - `main() -> None`: `python -m ai_core.fusion.stocksense_news`; thiếu env thì in `stocksense news: STOCKSENSE_DATABASE_URL not set, skip` và thoát 0; còn lại in `stocksense news: {n} symbols changed: {...}`.

- [ ] **Step 1: Viết migration 024**, chạy `node db/migrate.js` trên database thật (`../StockSenseVN/.env`) và trên một database mới tạo trong cluster tạm. Kỳ vọng: `Applied: 024_nlp_news.sql` ở cả hai.
- [ ] **Step 2: Viết test đỏ** `tests/fusion/test_stocksense_news.py`. Fixture `pg` giống `tests/fusion/test_stocksense.py` (schema tạm, skip khi thiếu `STOCKSENSE_TEST_DATABASE_URL`), nhưng **giữ `autocommit=True`** như `stocksense.connect`: `NOTIFY` chỉ được phát khi commit, mà `pg.transaction()` trong một transaction đang mở chỉ là savepoint. Dọn bằng `DROP SCHEMA ... CASCADE` sau test, không dùng rollback. với DDL `symbols(symbol varchar(12) PRIMARY KEY)` + DDL `nlp_news` ở trên, seed `FPT`, `VNM`. Helper `add(sq, doc, symbol, version="v1", processed="2026-11-12T08:00:00Z", published="2026-11-12T07:00:00Z", title="T", uncertain=0, rep=1, model=MODEL, seen="2026-11-12T07:01:00Z", passage="FPT bị khởi tố")` ghi `document`, `document_version`, `article_company_result` vào `storage.connect(":memory:")`.
```python
def test_push_inserts_and_notifies(pg, sq):         # 1 bài FPT
    pg.execute("LISTEN nlp_news")
    assert push_news(sq, pg) == ["FPT"]
    assert one(pg, "SELECT title, uncertain, risk_flags FROM nlp_news") == ("T", False, ["prosecution"])
    assert [n.payload for n in pg.notifies(timeout=1, stop_after=1)] == ["FPT"]

def test_only_phobert_representative_titled(pg, sq):  # thêm 1 bài baseline, 1 bài rep=0, 1 bài title=None
    assert push_news(sq, pg) == ["FPT"] and one(pg, "SELECT count(*) FROM nlp_news") == (1,)

def test_latest_version_wins(pg, sq):                # doc "d1" có v1 (seen 07:01) và v2 (seen 07:30, title "T2")
    push_news(sq, pg)
    assert all_rows(pg, "SELECT document_version_id, title FROM nlp_news") == [("v2", "T2")]

def test_watermark_boundary_and_quiet_rerun(pg, sq):
    push_news(sq, pg)
    assert push_news(sq, pg) == []                     # chạy lại: không đổi, không notify
    add(sq, "d9", "VNM", processed="2026-11-12T08:00:00Z")   # cùng giây với mốc
    assert push_news(sq, pg) == ["VNM"]

def test_rescored_row_updates(pg, sq):               # đổi p_pos + processed_at muộn hơn của bài đã đẩy
    assert push_news(sq, pg) == ["FPT"] and one(pg, "SELECT round(p_pos::numeric, 2) FROM nlp_news") == (0.9,)

def test_unknown_symbol_skipped(pg, sq):             # bài của NVL (không có trong symbols)
    assert push_news(sq, pg) == [] and one(pg, "SELECT count(*) FROM nlp_news") == (0,)
```
- [ ] **Step 3: Chạy, thấy đỏ.** `STOCKSENSE_TEST_DATABASE_URL=... .venv/bin/python -m pytest -q tests/fusion/test_stocksense_news.py`. Kỳ vọng: `ModuleNotFoundError: ai_core.fusion.stocksense_news`.
- [ ] **Step 4: Cài** `connect`, `rows_since`, `push_news`, `main` theo Interfaces.
- [ ] **Step 5: Chạy xanh** cả `tests/fusion/test_stocksense_news.py` và `tests/fusion/test_stocksense.py` (có env Postgres), rồi bộ `.venv` trong CLAUDE.md (thêm file mới vào lệnh `.venv`, thêm `--ignore` vào lệnh `.venv_mmf`). Kỳ vọng: tất cả pass, `tests/test_architecture.py` xanh.
- [ ] **Step 6: Nạp lịch sử lần đầu** vào database thật: `STOCKSENSE_DATABASE_URL=... .venv/bin/python -m ai_core.fusion.stocksense_news`. Kỳ vọng: in danh sách mã; `SELECT count(*) FROM nlp_news` cỡ vài chục nghìn; chạy lại in `0 symbols changed`.
- [ ] **Step 7: Commit** (chỉ khi được phép): AI_Core `ai_core/fusion/stocksense*.py tests/fusion/test_stocksense_news.py CLAUDE.md`; StockSenseVN `db/migrations/024_nlp_news.sql`.

### Task 2: Nhịp vòng live 2 phút / 15 phút

**Files:**
- Modify: `jobs/live_update.sh`
- Test: `tests/test_live_schedule.py`
- Modify: crontab của user (dòng `*/15 ... live_update.sh` → `*/2 ...`)

**Interfaces:**
- Consumes: `python -m ai_core.fusion.stocksense_news` (Task 1).
- Produces: hai biến môi trường cho test: `LIVE_CLOCK="YYYY-MM-DD HH:MM"` (thay cho giờ hiện tại, dùng qua `date -d`) và `LIVE_DRY=1` (mỗi bước chỉ in tên thay vì chạy, và bỏ qua `flock`). Tên bước in ra: `collect`, `process`, `news-push`, `quotes`, `predict`, `live`, `forecast-push`.

Quy tắc (giờ VN): `full` = phút % 15 < 2; `day` = 07:00 ≤ HH:MM < 22:00. Không `full` và không `day` thì thoát ngay. Bước tin (`collect`, `process`, `news-push`) chạy khi `full` hoặc `day`. `quotes` chỉ khi `full` và trong cửa sổ cũ (thứ 2–6, 09:00–15:15). `predict`, `live`, `forecast-push` chỉ khi `full` và trong cửa sổ cũ (thứ 2–6, 15:00–16:00). `news-push` chạy ngay sau `process`.

- [ ] **Step 1: Viết test đỏ** `tests/test_live_schedule.py`, chạy script bằng `subprocess` với `LIVE_DRY=1` và `LIVE_CLOCK`:
```python
@pytest.mark.parametrize("clock,steps", [
    ("2026-11-12 03:04", []),                                             # đêm, không phải mốc 15'
    ("2026-11-12 03:00", ["collect", "process", "news-push"]),            # đêm, mốc 15'
    ("2026-11-12 10:02", ["collect", "process", "news-push"]),            # ngày, giữa mốc
    ("2026-11-12 10:00", ["collect", "process", "news-push", "quotes"]),
    ("2026-11-12 15:16", ["collect", "process", "news-push", "predict", "live", "forecast-push"]),
    ("2026-11-12 15:20", ["collect", "process", "news-push"]),
    ("2026-11-14 10:00", ["collect", "process", "news-push"]),            # thứ Bảy: không quotes
    ("2026-11-12 22:00", ["collect", "process", "news-push"]),            # 22:00 vẫn là mốc 15'
    ("2026-11-12 22:02", []),
])
def test_schedule(clock, steps): ...
```
- [ ] **Step 2: Chạy, thấy đỏ.** `.venv/bin/python -m pytest -q tests/test_live_schedule.py`. Kỳ vọng: fail (script chưa biết `LIVE_DRY`, nên nó chạy thật). Chạy test với `timeout=20` để không treo.
- [ ] **Step 3: Sửa `jobs/live_update.sh`** theo quy tắc trên, giữ nguyên dòng đọc `.env` bằng `sed`, `flock` và mọi `|| echo "... failed, continuing"`.
- [ ] **Step 4: Chạy xanh** `tests/test_live_schedule.py`, và `bash -n jobs/live_update.sh`.
- [ ] **Step 5: Đổi crontab:** `crontab -l` → thay `*/15 * * * * ".../jobs/live_update.sh"` bằng `*/2 * * * * ".../jobs/live_update.sh"`, giữ nguyên phần chuyển log. Kiểm: `crontab -l | grep live_update` hiện `*/2`. Sau 2–4 phút, `logs/live.log` có dòng `stocksense news:`.
- [ ] **Step 6: Commit** (chỉ khi được phép): `jobs/live_update.sh tests/test_live_schedule.py`.

### Task 3: API đọc `nlp_news`

**Files:**
- Create: `../StockSenseVN/apps/api/src/market/nlp-news.js`
- Test: `../StockSenseVN/apps/api/src/market/nlp-news.test.js`
- Modify: `../StockSenseVN/apps/api/src/market/repository.js` (Postgres + fixture repository, `readSymbol`)
- Modify: `../StockSenseVN/apps/api/src/server.js`, `../StockSenseVN/openapi.yaml`

**Interfaces:**
- Consumes: bảng `nlp_news` (Task 1).
- Produces (`nlp-news.js`, thuần, không truy cập DB):
  - `vnDayRange(from: string, to: string) -> { start: string, end: string }`: ISO UTC; `start` = 00:00 VN ngày `from`, `end` = 00:00 VN ngày sau `to` (khoảng nửa mở).
  - `defaultRange(now: Date) -> { from: string, to: string }`: `to` = hôm nay theo VN, `from` = `to` − 7 ngày.
  - `parseRange(query, now) -> { from, to } | null`: thiếu thì dùng `defaultRange`; sai định dạng `YYYY-MM-DD` hoặc `from > to` thì `null`.
  - `mapNlpNews(row) -> { id, title, url, source, publishedAt, relation, probabilities: { positive, neutral, negative }, uncertain, riskFlags, passage, modelVersion }` (`id` = `document_id`; `url` = `null` khi `source === 'hose'`, vì link HOSE là link API trả JSON).
- Produces (repository, cả Postgres lẫn fixture):
  - `nlpNews(symbol, { from, to }) -> { from, to, items } | null`: mới nhất trước, `LIMIT 500`, `null` khi mã không bật. Fixture trả `{ from, to, items: [] }`.
  - `nlpNewsTimes(symbol) -> { items: string[] } | null`: mọi `published_at` (ISO) của mã, dùng cho chấm trên biểu đồ.
  - `readSymbol`: `dataFlags.recentNews` = số bài `nlp_news` của mã trong 7 ngày gần nhất; `missingNews = recentNews === 0`.
- Produces (HTTP, `requireCapability('news.read')`):
  - `GET /api/v1/symbols/:symbol/nlp-news?from=YYYY-MM-DD&to=YYYY-MM-DD` → `envelope({ from, to, items })`; khoảng sai → `400 INVALID_RANGE`; mã không có → `404 NOT_FOUND`.
  - `GET /api/v1/symbols/:symbol/nlp-news/times` → `envelope({ items })`.

- [ ] **Step 1: Viết test đỏ** `nlp-news.test.js`:
```js
expect(vnDayRange('2026-11-12', '2026-11-12')).toEqual({ start: '2026-11-11T17:00:00.000Z', end: '2026-11-12T17:00:00.000Z' });
expect(defaultRange(new Date('2026-11-12T18:30:00Z'))).toEqual({ from: '2026-11-06', to: '2026-11-13' });  // 01:30 VN ngày 13
expect(parseRange({ from: '2026-11-13', to: '2026-11-12' }, now)).toBeNull();
expect(parseRange({ from: '12/11/2026' }, now)).toBeNull();
expect(mapNlpNews({ ...row, source: 'hose' }).url).toBeNull();
expect(mapNlpNews(row).probabilities).toEqual({ positive: 0.88, neutral: 0.08, negative: 0.04 });
```
- [ ] **Step 2: Chạy, thấy đỏ.** `npx vitest run apps/api/src/market/nlp-news.test.js`. Kỳ vọng: không import được `./nlp-news.js`.
- [ ] **Step 3: Cài** `nlp-news.js`, các hàm repository, hai route, và mục tương ứng trong `openapi.yaml`. Câu SQL dùng `published_at >= $2 AND published_at < $3` với `vnDayRange`.
- [ ] **Step 4: Chạy xanh** `npx vitest run`. Kỳ vọng: toàn bộ pass.
- [ ] **Step 5: Kiểm trên database thật** (API cổng 3100, đăng nhập `retail@stocksense.vn`): `curl .../symbols/FPT/nlp-news` trả `from`/`to` 7 ngày và các bài mới nhất trước, có `probabilities`; `...?from=2026-11-13&to=2026-11-12` trả 400; `.../nlp-news/times` trả danh sách có ngày từ 2021; `.../symbols/FPT` có `dataFlags.recentNews > 0`, `missingNews: false`.
- [ ] **Step 6: Commit** (chỉ khi được phép).

### Task 4: Đẩy thời gian thực (LISTEN → SSE)

**Files:**
- Create: `../StockSenseVN/apps/api/src/market/news-hub.js`
- Test: `../StockSenseVN/apps/api/src/market/news-hub.test.js`
- Modify: `../StockSenseVN/apps/api/src/server.js`

**Interfaces:**
- Consumes: kênh `nlp_news` (Task 1).
- Produces:
  - `createNewsHub() -> { subscribe(symbol: string, fn: () => void): () => void, publish(payload: string): void }`: `publish("FPT,VNM")` gọi mọi `fn` đã đăng ký cho FPT và VNM; hàm trả về từ `subscribe` gỡ đăng ký.
  - `startNewsListener(pool, hub, { retryMs = 5000 } = {}) -> { stop(): Promise<void> }`: lấy client riêng từ `pool`, `LISTEN nlp_news`, chuyển mỗi `notification.payload` cho `hub.publish`; khi client `error` hoặc `end` thì nhả client và thử lại sau `retryMs`. Chỉ khởi động khi `databaseEnabled`.
  - `GET /api/v1/symbols/:symbol/nlp-news/stream` (`requireCapability('news.read')`): `text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`; gửi `event: news\ndata: {"symbol":"FPT"}\n\n` khi hub báo mã đó; comment `: ping` mỗi 25 giây; gỡ đăng ký khi kết nối đóng.

- [ ] **Step 1: Viết test đỏ** `news-hub.test.js`:
```js
it('routes payload symbols to their subscribers only', ...)   // publish('FPT,VNM') → fpt 1 lần, vnm 1 lần, acb 0
it('unsubscribe stops delivery', ...)
it('startNewsListener reconnects after error', async () => {   // pool giả: connect() trả client EventEmitter giả có query()
  // client đầu emit('error'); sau retryMs=10 phải có lần connect() thứ hai và query('LISTEN nlp_news') lần hai
  // notification trên client mới đến được hub.publish
});
```
- [ ] **Step 2: Chạy, thấy đỏ.** `npx vitest run apps/api/src/market/news-hub.test.js`. Kỳ vọng: không import được `./news-hub.js`.
- [ ] **Step 3: Cài** `news-hub.js` và route SSE (`reply.hijack()`, ghi vào `reply.raw`), khởi động `startNewsListener(pool, hub)` khi `databaseEnabled`, dừng nó khi app đóng.
- [ ] **Step 4: Chạy xanh** `npx vitest run`.
- [ ] **Step 5: Kiểm thật qua proxy web:** giữ `curl -N -b <cookie> http://127.0.0.1:5173/api/v1/symbols/FPT/nlp-news/stream` mở; ở terminal khác chạy `psql "$URL" -c "SELECT pg_notify('nlp_news','FPT')"`. Kỳ vọng: trong 1 giây, curl in `event: news` với `{"symbol":"FPT"}`; `pg_notify(...,'VNM')` không in gì. Khởi động lại Postgres không được: thay vào đó `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query = 'LISTEN nlp_news'`, chờ 6 giây, notify lại FPT, curl vẫn nhận.
- [ ] **Step 6: Commit** (chỉ khi được phép).

### Task 5: Giao diện tab Tin tức (phương án B) và các phần tin khác trên trang

**Files:**
- Create: `../StockSenseVN/packages/core/src/labels/news.js` (+ export `"./news"` trong `packages/core/package.json`)
- Test: `../StockSenseVN/packages/core/src/labels/news.test.js`
- Create: `../StockSenseVN/apps/web/src/features/symbol-detail/NlpNewsItem.jsx`
- Modify: `../StockSenseVN/apps/web/src/features/symbol-detail/SymbolDetailPage.jsx`, `apps/web/src/api/realApi.js`, `apps/web/src/api/mockApi.js`, `apps/web/src/styles/app.css` (nếu cần cho thanh xác suất)

**Interfaces:**
- Consumes: các endpoint của Task 3 và 4; `dataFlags.recentNews` (Task 3).
- Produces (`@stocksense/core/news`):
  - `sentimentOf(item) -> { tone: 'up'|'hold'|'down'|'unsure', text: string }`: `uncertain` → `{ tone: 'unsure', text: 'Chưa chắc' }`; còn lại là nhãn có xác suất lớn nhất, text `'Tích cực 88%'` / `'Trung tính 80%'` / `'Tiêu cực 52%'` (làm tròn %).
  - `relationText(relation, symbol) -> string`: `direct` → `Nói trực tiếp về ${symbol}`, `sector` → `Tin ngành`, `market` → `Tin thị trường chung`, `ambiguous` → `Liên quan chưa rõ`.
  - `RISK_TEXT = { guarantee: 'Bảo lãnh nợ', seizure: 'Kê biên / phong tỏa', prosecution: 'Khởi tố / điều tra', sanction: 'Xử phạt / hạn chế GD', bond_default: 'Chậm trả trái phiếu' }`.
  - `UNSURE_NOTE = 'Model không đủ tự tin để gán nhãn; bài này không được tính vào điểm cảm xúc của mã.'`
- Produces (web API client, cả `realApi` và `mockApi`): `api.symbols.nlpNews(symbol, { from, to })`, `api.symbols.nlpNewsTimes(symbol)`, `api.symbols.newsStream(symbol, onUpdate) -> close()` (`realApi`: `EventSource` tới `/api/v1/symbols/:symbol/nlp-news/stream`, nghe `news`; `mockApi`: trả mảng rỗng và `close` không làm gì).

Hành vi trang (`SymbolDetailPage.jsx`):
- Hai ô ngày mặc định `defaultRange` (hôm nay − 7 → hôm nay theo VN); đổi ngày thì gọi lại `nlpNews`; nút "Bỏ lọc" đưa về mặc định.
- Danh sách tab dùng `NlpNewsItem`: dòng gọn (tiêu đề, nguồn, giờ, chip cảm xúc, chip liên quan, chip rủi ro nếu có). Bấm vào bài thì mở/đóng panel: thanh 3 màu + ba con số, `UNSURE_NOTE` khi `uncertain`, đoạn văn `passage`, cờ rủi ro hoặc "không có", `modelVersion`, link "Mở bài gốc ↗" (`target="_blank" rel="noopener noreferrer"`, ẩn khi `url` là `null`). Trạng thái mở theo `id`, không mất khi danh sách tải lại.
- Thẻ "Tin mới nhất" (tab Dự báo): 3 bài đầu của danh sách 7 ngày mặc định, không theo khoảng ngày đang chọn.
- Chấm biểu đồ: `newsDays` tính từ `nlpNewsTimes` qua `newsSession`.
- "Tin tức: N bài" và chip "Thiếu tin": từ `detail.dataFlags.recentNews` / `missingNews`.
- Mở `newsStream` khi trang mã được mở, đóng khi rời trang. Mỗi sự kiện tải lại song song: danh sách theo khoảng đang chọn, danh sách 7 ngày, `nlpNewsTimes`, `symbols.detail` (cho cờ). Bỏ câu "Nhãn sắc thái sẽ chính xác khi mô hình sentiment sẵn sàng."; thay bằng "Mới nhất trước. Cảm xúc chấm bằng PhoBERT (AI_Core)."

- [ ] **Step 1: Viết test đỏ** `packages/core/src/labels/news.test.js`:
```js
expect(sentimentOf({ uncertain: true, probabilities: { positive: 0.04, neutral: 0.76, negative: 0.2 } })).toEqual({ tone: 'unsure', text: 'Chưa chắc' });
expect(sentimentOf({ uncertain: false, probabilities: { positive: 0.88, neutral: 0.08, negative: 0.04 } })).toEqual({ tone: 'up', text: 'Tích cực 88%' });
expect(sentimentOf({ uncertain: false, probabilities: { positive: 0.05, neutral: 0.8, negative: 0.15 } }).text).toBe('Trung tính 80%');
expect(relationText('direct', 'FPT')).toBe('Nói trực tiếp về FPT');
expect(Object.keys(RISK_TEXT).sort()).toEqual(['bond_default', 'guarantee', 'prosecution', 'sanction', 'seizure']);
```
- [ ] **Step 2: Chạy, thấy đỏ.** `npx vitest run packages/core/src/labels/news.test.js`. Kỳ vọng: không import được `./news.js`.
- [ ] **Step 3: Cài** `news.js`, client API, `NlpNewsItem.jsx` và các thay đổi trang như trên.
- [ ] **Step 4: Chạy xanh** `npx vitest run` và `npm run build:web`; build mock (`VITE_API_MODE=mock npx vite build` trong `apps/web`) cũng phải thành công.
- [ ] **Step 5: Kiểm end-to-end trên trình duyệt** (http://localhost:5173, tài khoản retail, mã FPT; kiểm qua giao diện, không chỉ curl):
  - tab Tin tức hiện bài thật của 7 ngày với chip đúng; bài "Cơn bĩ cực của cổ đông FPT" hiện "Chưa chắc";
  - bấm một bài thì mở panel, bấm lại thì đóng;
  - đổi "Từ ngày" về 2026-10-01 thì có thêm bài cũ;
  - chấm tin xuất hiện trên biểu đồ nến; "Tin tức: N bài" khớp số bài 7 ngày; header không còn "Thiếu tin";
  - giữ trang mở, rồi chèn một bài FPT giả vào `nlp.db` (hoặc hạ `processed_at` của một dòng trong `nlp_news` rồi chạy `python -m ai_core.fusion.stocksense_news`): bài hiện trên trang trong vài giây mà không tải lại, panel đang mở vẫn mở. Xóa bài giả sau khi kiểm.
- [ ] **Step 6: Commit** (chỉ khi được phép).
