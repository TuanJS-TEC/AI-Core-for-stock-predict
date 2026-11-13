# Repo Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gộp `NLP_Module` (giữ lịch sử commit) và `Pipeline_V1.2` vào repo gốc thành một package `ai_core` cài được, không đổi hành vi.

**Architecture:** Toàn bộ việc làm trong một git worktree riêng (`.claude/worktrees/restructure`, nhánh `restructure`) nên cây đang chạy cron không bị động tới cho đến Task 8. Lịch sử NLP được dời chỗ trong một bản clone tạm rồi merge vào nhánh; code chỉ đổi đường dẫn import và đường dẫn dữ liệu; kết quả được đối chiếu với mốc "trước" đo bằng chính code cũ.

**Tech Stack:** git (không có `git filter-repo`), Python 3.14, pytest, setuptools (chỉ lúc build editable), `.venv_mmf` và `NLP_Module/.venv` hiện có.

**Spec:** `docs/superpowers/specs/2026-11-10-repo-restructure-design.md`

## Global Constraints

- Không sửa logic, tên hàm, tham số hay hằng số; chỉ đổi đường dẫn import và đường dẫn tệp. Ngoại lệ duy nhất đã duyệt: `FUSION_VERSION` trong `board.py` thành import từ `live.VERSION` (cùng giá trị `"decision-v0-nogo"`).
- Không đổi phiên bản torch/numpy/pandas, không thêm dependency vào bất kỳ venv nào (spec: "không thêm dependency mới").
- `docs/reports/mmf_lock.json`, `mmf_dev_report.md`, `mmf_preregistration.md` không được sửa. `mmf.holdout` không được chạy.
- Dữ liệu ở nguyên chỗ: `Pipeline_V1.2/stock`, `Pipeline_V1.2/data`, `NLP_Module/data` (nlp.db, models, tessdata, vncorenlp), `data/`. Không `rm` thứ gì ngoài bản build; thứ cần dọn thì dời sang `~/restructure-backup/`.
- Đạt khi: 391 test pass (88 + 113 + 190) và mọi băm trong mốc "trước" khớp từng bit.
- `ai_core/__init__.py`, `ai_core/prices/__init__.py`, `ai_core/fusion/__init__.py` để trống: cron Finhay và cron cross-check chỉ được kéo vào phần import tối thiểu.
- Mọi commit kết thúc bằng dòng `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

Các tình huống spec ngầm đòi hỏi nhưng dễ lọt, theo thứ tự dễ xảy ra:
1. Mã cũ còn bị import nhầm sau cutover (thư mục `src/`, `nlp_module/` cũ còn trong cwd của cron) → Task 8 kiểm `import src`, `import nlp_module`, `import mmf` phải lỗi từ cả hai cwd.
2. Cron cross-check chạy bằng `~/.venv_6x` không cài `ai_core` → dòng cron phải thêm `PYTHONPATH`; Task 7 chạy đúng dòng lệnh đó.
3. Cron Finhay (chỉ HTTP) lỡ kéo `ai_core.news`/torch → Task 6 test kiến trúc chặn.
4. Mất file được track của NLP khi merge → Task 3 so băm blob từng file với bản đồ đích.
5. Cutover khi `live_update.sh` đang chạy → Task 8 tắt cron và chờ `flock` trước.

## Bản đồ chuyển file (dùng xuyên suốt)

| Từ | Đến |
|---|---|
| `Pipeline_V1.2/src/*` | `ai_core/prices/*` (đổi `src.` thành `ai_core.prices.`) |
| `Pipeline_V1.2/sync_stock_parquet.py` | `ai_core/prices/sync.py`; `jobs/sync_prices.py` là wrapper gọi `main()` |
| `Pipeline_V1.2/crosscheck_tcbs.py` | `ai_core/prices/crosscheck.py`; `jobs/crosscheck_tcbs.py` là wrapper |
| `Pipeline_V1.2/tests/*` | `tests/prices/*` |
| `Pipeline_V1.2/{data,docs,pytest.ini,requirements.txt}` | `data/` và `requirements.txt` giữ nguyên chỗ (là nhà dữ liệu); `docs/` → `docs/prices/`; `pytest.ini` xoá |
| NLP `nlp_module/forecast/*` | `ai_core/models/volcast/*` |
| NLP `nlp_module/board.py` | `ai_core/fusion/board.py` |
| NLP `nlp_module/*` (còn lại) | `ai_core/news/*` |
| NLP `tests/forecast/*` | `tests/models/volcast/*` |
| NLP `tests/test_board.py` | `tests/fusion/test_board.py` |
| NLP `tests/*` (còn lại) | `tests/news/*` |
| NLP `docs/*` | `docs/news/*` (kể cả `docs/superpowers` → `docs/news/superpowers`) |
| NLP `README.md` | `docs/news/README.md` |
| NLP `app.py` | `ai_core/fusion/app.py` |
| NLP `scripts/live_update.sh` | `jobs/live_update.sh` |
| NLP mọi thứ còn lại (`.gitignore`, `requirements.txt`, `data/*`, `.claude/*`) | `NLP_Module/<cùng đường dẫn>` |
| `mmf/*` | `ai_core/models/mmf/*` |
| `tests/test_mmf_*.py` | `tests/models/mmf/` |
| `ai_core/{config,direction,finhay,live,prices,shadow,splits}.py` | `ai_core/fusion/` |
| `tests/test_{alignment,direction,finhay,live,shadow,splits}.py` | `tests/fusion/`; `tests/test_env_numpy.py` ở nguyên chỗ |

Thư mục test nào cũng có `__init__.py` (tên `test_snapshot.py` và `test_splits.py` trùng nhau giữa prices/news/fusion).

Quy tắc đổi tên import (áp dụng cho dòng `import`, `from … import` và đích của `python -m`, theo thứ tự này): `from ai_core import` và `ai_core.<live|shadow|direction|finhay|splits|prices|config>` → `ai_core.fusion…`; `nlp_module.forecast` → `ai_core.models.volcast`; `nlp_module.board` → `ai_core.fusion.board`; `nlp_module` → `ai_core.news`; `src` → `ai_core.prices`; `mmf` → `ai_core.models.mmf`. Import tương đối `..x` trong `volcast/*` (trỏ ra `news`) phải thành `ai_core.news.x` tuyệt đối. Docs lịch sử (`docs/superpowers/plans|specs`, `docs/reports`) không sửa; chỉ sửa code, test, jobs, shell và ba tài liệu đang dùng (`docs/mmf_thu_nghiem.md`, `docs/prices/data.md`, `docs/news/README.md`).

---

### Task 1: Mốc "trước" và worktree

**Files:**
- Create (ngoài repo): `~/restructure-baseline/golden.py`, `~/restructure-baseline/before.json`, `~/restructure-baseline/crontab.txt`, `~/restructure-baseline/untracked/`
- Create: worktree `.claude/worktrees/restructure` trên nhánh `restructure` từ `multimodal-decision` (`deb3c29`)

**Interfaces:**
- Produces: `golden.py --layout {old,new} --root PATH --out FILE.json` ghi `{tên_mục: sha256}`. Mục: `dataset_h2` (băm `to_csv` của `build_dataset(panel, 2)` trên snapshot `Pipeline_V1.2/stock/snapshots/24251acc1a5695bd`), `frame_h2_foldC` (`build_frame` h=2 với `FOLD_NLP["C"]`, cột `created_at` không có), `live_rows` (`live.rows(hist, results, vol, now)` với `now = 2026-11-10T08:30Z`, bỏ phần tử cuối mỗi tuple vì là timestamp), `smoke_preds` (băm CSV các parquet dự đoán của `mmf.experiment --smoke --device cpu` với `C.DATA` trỏ vào thư mục tạm), `volcast_out` (một đầu ra volcast tất định, người thực hiện chọn cái rẻ nhất chạm `STOCK_DIR` và mô hình đã lưu, ghi lựa chọn vào `~/restructure-baseline/README.txt`). Bảng ánh xạ tên module cũ/mới nằm trong chính `golden.py`.

- [ ] **Step 1:** `crontab -l > ~/restructure-baseline/crontab.txt`; sao chép các tệp không được track của NLP_Module vào `~/restructure-baseline/untracked/` giữ nguyên cây thư mục: `git -C NLP_Module status --short --ignored` liệt kê; bắt buộc có `CLAUDE.md`, `AGENTS.md`, `.claude/`, `data/*.csv`, `data/reports/`, `.claude/settings.json` (bản đã sửa). Kiểm: so `diff -r` từng tệp đã sao chép.
- [ ] **Step 2:** Chạy ba bộ test cũ, ghi số pass vào `before.json` dưới `tests`: `(cd Pipeline_V1.2 && ../NLP_Module/.venv/bin/python -m pytest -q)` → `88 passed`; `(cd NLP_Module && .venv/bin/python -m pytest -q)` → `113 passed`; `.venv_mmf/bin/python -m pytest -q tests` → `190 passed`.
- [ ] **Step 3:** Viết `golden.py` (chạy được ở cả hai layout) và chạy `--layout old --root <cây hiện tại> --out before.json`. Chạy lần hai và `diff`: hai lần phải giống hệt (kiểm tính tất định; nếu không thì dừng và báo, không tiếp tục).
- [ ] **Step 4:** `git worktree add .claude/worktrees/restructure -b restructure multimodal-decision`; trong worktree tạo symlink tới dữ liệu trong cây chính cho các đường dẫn gitignored cần đọc: `Pipeline_V1.2/stock`, `Pipeline_V1.2/outputs`, `NLP_Module/data` (tạo `NLP_Module/` rồi liên kết từng mục của `data/` trừ các tệp được track), `data/finhay`, `data/mmf`, `NLP_Module/.venv`.
- [ ] **Step 5:** Không commit (không có gì trong repo thay đổi).

### Task 2: Dời chỗ lịch sử NLP trong bản clone tạm

**Files:**
- Create (ngoài repo): `~/restructure-work/nlp` (clone), nhánh `nlp-moved`

**Interfaces:**
- Produces: nhánh `nlp-moved` chứa toàn bộ lịch sử `volcast-v2-bctc` (25 commit) cộng đúng một commit cuối `refactor: move files to ai_core layout` trong đó mọi tệp được track nằm ở đường dẫn đích của bảng bản đồ (phần NLP), nội dung không đổi (100% rename).

- [ ] **Step 1:** `git clone -b volcast-v2-bctc /home/administrator/TLS/tuanhm@linux/AI_Core/NLP_Module ~/restructure-work/nlp` (clone cục bộ, không đụng cây đang chạy); kiểm `git -C ~/restructure-work/nlp rev-list --count HEAD` = 25.
- [ ] **Step 2:** Viết bản đồ NLP thành một danh sách `(mẫu cũ, đường dẫn mới)` trong `~/restructure-work/move_nlp.sh` dùng `git mv`; chạy, rồi commit duy nhất. Kiểm: `git diff --find-renames=100% --name-status HEAD~1 HEAD` không có `A`/`D` (chỉ `R100`), và `git ls-files` không còn tệp nào ngoài các đường dẫn đích.
- [ ] **Step 3:** Kiểm lịch sử còn nguyên: `git log --follow --oneline -- ai_core/news/storage.py | wc -l` ≥ 2 và bằng `git log --oneline -- nlp_module/storage.py` trên `HEAD~1`.

### Task 3: Merge lịch sử NLP vào nhánh `restructure`

**Files:**
- Modify: `.gitignore` (xoá dòng `NLP_Module/`, thay bằng ignore các tệp dữ liệu cụ thể bên dưới), `NLP_Module/.gitignore`
- Test: `~/restructure-work/verify_merge.py` (không vào repo)

**Interfaces:**
- Consumes: nhánh `nlp-moved` (Task 2).
- Produces: nhánh `restructure` có lịch sử của cả hai repo.

- [ ] **Step 1:** Trong worktree: `git fetch ~/restructure-work/nlp nlp-moved` rồi `git merge --allow-unrelated-histories FETCH_HEAD -m "chore: merge NLP_Module history"`. Nếu xung đột `.gitignore`/`requirements.txt`: không xảy ra vì NLP đã nằm dưới `NLP_Module/`; nếu xảy ra thì dừng và báo.
- [ ] **Step 2:** `verify_merge.py`: với mọi tệp trong `git ls-tree -r volcast-v2-bctc` của repo NLP (qua `git -C NLP_Module`), tính đường dẫn đích bằng bản đồ và khẳng định blob băm giống hệt (`git rev-parse HEAD:<đích>`); in số tệp. Kỳ vọng: toàn bộ khớp, 0 thiếu. Khẳng định `git rev-list --count HEAD` = 70 + 25 + 1.
- [ ] **Step 3:** Sửa `.gitignore` gốc: bỏ `NLP_Module/`; thêm ignore cho `NLP_Module/.venv/`, `NLP_Module/data/*.csv` chưa track, `NLP_Module/data/{reports,stock,stock.bak-*,tessdata,live.lock,live.log,prices.csv}`. Kiểm: trong cây chính (không phải worktree), `git status --short` sau cutover sẽ không đầy tệp rác; ở worktree kiểm `git status --short` rỗng. Commit `chore: ignore NLP data files now that the repo is merged`.

### Task 4: Gói hóa, `paths.py`, đổi import (cơ học)

**Files:**
- Create: `pyproject.toml` (ghi đè), `ai_core/__init__.py` (rỗng), `ai_core/paths.py`, `tests/test_paths.py`, `~/restructure-work/rewrite_imports.py`
- Modify/Move: mọi tệp theo bản đồ (Pipeline, mmf, ai_core cũ, tests)
- Modify: `ai_core/prices/config.py`, `ai_core/fusion/config.py`, `ai_core/news/{storage,entities}.py`, `ai_core/news/sentiment/segment.py`, `ai_core/news/connectors/hose.py`, `ai_core/models/volcast/volcast.py`, `ai_core/fusion/app.py`

**Interfaces:**
- Produces: `ai_core.paths` với `ROOT: Path` (thư mục chứa `pyproject.toml`), `PRICES_HOME: Path = ROOT/"Pipeline_V1.2"`, `NEWS_HOME: Path = ROOT/"NLP_Module"`, `DATA: Path = ROOT/"data"`. Mọi chỗ trước đây tính `parents[n]` dùng các hằng này.
- `ai_core.fusion.config` giữ tên `ROOT`, `PIPELINE` (= `PRICES_HOME`), `NLP` (= `NEWS_HOME`), `STOCK_DIR` để `mmf` không phải đổi gì ngoài import.

- [ ] **Step 1: Test.** `tests/test_paths.py::test_paths_resolve`: `paths.ROOT/"pyproject.toml"` tồn tại; `paths.PRICES_HOME/"data"/"hose_holidays.csv"` tồn tại; `ai_core.prices.config.STOCK_DIR == paths.PRICES_HOME/"stock"`; `ai_core.news.storage.DB_PATH == paths.NEWS_HOME/"data"/"nlp.db"`; `ai_core.models.volcast.volcast.STOCK_DIR == paths.PRICES_HOME/"stock"`; `ai_core.fusion.config.STOCK_DIR == paths.PRICES_HOME/"stock"`. Chạy: lỗi vì chưa có `ai_core.paths`.
- [ ] **Step 2:** Thực hiện bản đồ chuyển file cho Pipeline/mmf/ai_core cũ/tests bằng `git mv`; xoá `Pipeline_V1.2/pytest.ini`; thêm `__init__.py` cho các thư mục test và package còn thiếu. Commit riêng `refactor: move files to ai_core layout` (chỉ rename, import chưa sửa) để `git log --follow` còn 100% rename.
- [ ] **Step 3:** `rewrite_imports.py PATH...` áp quy tắc đổi tên import (đã nêu trên) lên `.py`/`.sh` và ba tài liệu đang dùng; in diff theo tệp. Chạy, rồi gõ tay: `board.py` đổi `FUSION_VERSION = "decision-v0-nogo"` thành `from .live import VERSION as FUSION_VERSION`; `volcast/*` đổi `from ..x`/`from .. import x` thành `ai_core.news` tuyệt đối; `app.py` bỏ chèn `sys.path` nếu có. Cổng: `grep -rnE "^\s*(from|import) (src|nlp_module|mmf)\b|from ai_core import" ai_core tests jobs` không ra dòng nào; `grep -rn "sys.path" ai_core tests` không ra gì.
- [ ] **Step 4:** Tạo `ai_core/paths.py`; thay `parents[n]`/`__file__` trong 7 chỗ ở danh sách Files bằng `paths.*`, gỡ khối `sys.path` trong `fusion/config.py`.
- [ ] **Step 5:** `pyproject.toml`: `[project] name="ai-core" version="0.1" requires-python=">=3.14" dependencies=[]`; `[tool.setuptools.packages.find] include=["ai_core*"]`; package-data `ai_core/news/schema.sql`; `[tool.pytest.ini_options] testpaths=["tests"] pythonpath=["."]`.
- [ ] **Step 6:** Chạy `tests/test_paths.py`: PASS. Commit `refactor: package ai_core, replace sys.path hacks with ai_core.paths`.

### Task 5: Bộ test pass trong layout mới

**Files:** sửa các tệp test và import còn lỗi sau Task 4, không sửa logic.

- [ ] **Step 1:** Chạy với `PYTHONPATH=<worktree>` (chưa cài editable): `NLP_Module/.venv/bin/python -m pytest -q tests/prices tests/news tests/models/volcast tests/fusion/test_board.py` → `88 + 113 = 201 passed`; `.venv_mmf/bin/python -m pytest -q tests --ignore=tests/prices --ignore=tests/news --ignore=tests/models/volcast --ignore=tests/fusion/test_board.py` → `190 passed`. (Tổng 391.) Mọi lỗi collection là lỗi import/đường dẫn cần sửa, không phải test phải bỏ.
- [ ] **Step 2:** Với `--layout new`, chạy `golden.py --layout new --root <worktree> --out after.json`; `diff before.json after.json` phải rỗng. Lệch ở mục nào thì tìm gốc ở mục đó, không sửa mốc.
- [ ] **Step 3:** Commit `test: fix imports and paths after the move`.

### Task 6: Test kiến trúc

**Files:**
- Create: `tests/test_architecture.py`

**Interfaces:**
- Consumes: cây `ai_core/` đã chuyển. Dùng `ast` thuần, không thêm dependency.

- [ ] **Step 1: Test.** Hàm `imports_of(path) -> set[str]` (tên module tuyệt đối đầy đủ, kể cả import trong hàm) và các test:
  - `test_prices_and_news_do_not_import_each_other_or_upward`: các module `ai_core.prices.*` và `ai_core.news.*` không import `ai_core.news`/`ai_core.prices` lẫn nhau, cũng không import `ai_core.models` hay `ai_core.fusion`.
  - `test_models_do_not_import_fusion`.
  - `test_finhay_is_http_only`: `ai_core/fusion/finhay.py` không import `ai_core.news`, `ai_core.models`, `torch`, `transformers`.
  - `test_package_inits_stay_empty`: `ai_core/__init__.py`, `ai_core/prices/__init__.py`, `ai_core/fusion/__init__.py` không chứa câu lệnh nào.
  - `test_no_sys_path_mutation`: không tệp nào dưới `ai_core/` chứa `sys.path`.
- [ ] **Step 2:** Chạy: kỳ vọng PASS ngay; nếu một quy tắc fail thì đó là phụ thuộc ngược thật — báo cáo cụ thể và dừng, không nới quy tắc. Để chắc test thật sự bắt lỗi, tạm thêm một dòng `import ai_core.news` vào một module `prices`, thấy test fail, rồi bỏ dòng.
- [ ] **Step 3:** Commit `test: enforce one-way dependencies between prices, news, models and fusion`.

### Task 7: Jobs, cài editable, dòng cron mới (chưa bật)

**Files:**
- Create: `jobs/sync_prices.py`, `jobs/crosscheck_tcbs.py`; sửa `jobs/live_update.sh`
- Create (ngoài repo): `~/restructure-work/crontab.new`

**Interfaces:**
- Dòng cron mới (thay đường dẫn tuyệt đối cũ; giữ nguyên lịch và biến môi trường):
  - sync giá: `cd <ROOT> && VNSTOCK_*… NLP_Module/.venv/bin/python jobs/sync_prices.py >> Pipeline_V1.2/outputs/sync.log 2>&1`
  - cross-check: `… PYTHONPATH=<ROOT> ~/.venv_6x/bin/python jobs/crosscheck_tcbs.py >> Pipeline_V1.2/outputs/crosscheck.log 2>&1`
  - Finhay: `cd <ROOT> && NLP_Module/.venv/bin/python -m ai_core.fusion.finhay >> data/finhay/sync.log 2>&1`
  - live: `<ROOT>/jobs/live_update.sh >> NLP_Module/data/live.log 2>&1`
- `jobs/live_update.sh` bắt đầu bằng `cd "$(dirname "$0")/../NLP_Module"` (các đường dẫn `data/…` tương đối trong `cli.py` vẫn đúng); các lệnh `-m nlp_module.cli` và `-m nlp_module.forecast.volcast` thành `-m ai_core.news.cli` và `-m ai_core.models.volcast.volcast`; dòng `(cd .. && … -m ai_core.live)` thành `-m ai_core.fusion.live`; `.venv/bin/python` giữ.

- [ ] **Step 1:** Viết wrapper (3–5 dòng, gọi hàm `main` có sẵn của `ai_core.prices.sync` / `ai_core.prices.crosscheck`); `bash -n jobs/live_update.sh`: không lỗi.
- [ ] **Step 2:** Với worktree làm cwd, chạy từng dòng lệnh cron mới ở chế độ không phá dữ liệu: `jobs/sync_prices.py --help`, `jobs/crosscheck_tcbs.py --help` (bằng `~/.venv_6x` + `PYTHONPATH`), `-m ai_core.fusion.finhay --help` nếu có `--help`, nếu không thì `python -c "import ai_core.fusion.finhay"` và khẳng định `"torch" not in sys.modules`; `python -m ai_core.news.cli --help`. Kỳ vọng: thoát 0.
- [ ] **Step 3:** Soạn `crontab.new` từ `crontab.txt` và các dòng trên (chưa cài).
- [ ] **Step 4:** Commit `refactor: cron entry points under jobs/`.

### Task 8: Cutover (cần sự đồng ý của chủ dự án ngay trước bước 1)

**Files:** cây chính `/home/administrator/TLS/tuanhm@linux/AI_Core`.

- [ ] **Step 1:** Hỏi chủ dự án xác nhận bắt đầu. Sau đó `crontab -l > …/crontab.txt` (làm mới), cài crontab đã bỏ 4 dòng dự án, chờ `flock -n 9` trên `NLP_Module/data/live.lock` rảnh (nghĩa là không còn `live_update.sh` nào đang chạy).
- [ ] **Step 2:** Trong cây chính: các thay đổi chưa commit (`.gitignore`, `data/finhay/*.parquet`) — hỏi chủ dự án commit hay stash; sau đó `git merge restructure` (fast-forward nếu `multimodal-decision` chưa tiến thêm; nếu không thì merge thường). Merge xoá `Pipeline_V1.2/src`, `mmf/`, `ai_core/*.py` cũ do git quản lý.
- [ ] **Step 3:** Dời (không xoá) mã NLP cũ không còn được track: `NLP_Module/nlp_module`, `NLP_Module/tests`, `NLP_Module/docs`, `NLP_Module/app.py`, `NLP_Module/scripts` → `~/restructure-backup/`; dời `NLP_Module/.git` → `~/restructure-backup/NLP_Module.git` (repo GitHub không bị đụng); xoá mọi `__pycache__` cũ trong `Pipeline_V1.2/` và `NLP_Module/`; khôi phục `NLP_Module/.claude/settings.json` bản đã sửa từ `~/restructure-baseline/untracked/` nếu merge đã ghi đè.
- [ ] **Step 4:** `pip install -e .` bằng `NLP_Module/.venv/bin/python` và `.venv_mmf/bin/python` (cần mạng để tải setuptools lúc build; nếu không có mạng thì thêm tệp `.pth` trỏ vào `<ROOT>` ở site-packages mỗi venv và ghi lại việc đó). Không được xuất hiện thay đổi phiên bản gói nào: so `pip freeze` trước/sau, chỉ khác `ai-core`.
- [ ] **Step 5:** Kiểm tra mã cũ không còn: từ cwd `<ROOT>` và từ cwd `<ROOT>/NLP_Module`, chạy `python -c "import src"`, `import nlp_module`, `import mmf` bằng cả hai venv: tất cả phải `ModuleNotFoundError`.
- [ ] **Step 6:** Chạy lại bộ ba test (391 pass) và `golden.py --layout new --root <ROOT>`; `diff before.json after.json` rỗng. `git log --follow` trên `ai_core/news/storage.py` còn thấy commit cũ.
- [ ] **Step 7:** Cài `crontab.new`. Chờ phiên 15:20 kế tiếp (hoặc chạy tay dòng sync khi ngoài giờ giao dịch) và xác nhận `Pipeline_V1.2/outputs/sync.log`, `NLP_Module/data/live.log` có dòng chạy mới không lỗi; nếu hỏng thì cài lại `crontab.txt` cũ và báo.
- [ ] **Step 8:** Cập nhật hướng dẫn: `docs/mmf_thu_nghiem.md` (lệnh `python -m ai_core.models.mmf.…`), `docs/prices/data.md`, `docs/news/README.md`; thêm `CLAUDE.md` gốc ngắn mô tả cây thư mục và lệnh chạy test theo venv. Commit `docs: update commands for the ai_core layout`.

---

## Self-review (đã chạy)

- **Phủ spec:** gộp repo giữ lịch sử (Task 2–3); cây đích (bản đồ + Task 4); `paths.py` (Task 4); kiểm chứng bằng mốc/băm/test (Task 1, 5, 8); test kiến trúc (Task 6); cron và `jobs/` (Task 7–8); dữ liệu ở nguyên chỗ (Global Constraints); file lock không sửa (Global Constraints); worktree `price-v2`/`vnstock-gen6` không đụng (không có task nào chạm). Bước "dời dữ liệu" tùy chọn của spec cố ý không có trong kế hoạch này.
- **Hai điều tôi quyết thay chủ dự án** (cần xem khi đọc): `board.py` và `app.py` vào `ai_core.fusion` (vì chúng đọc/hiển thị `fusion_decision` và nối giá với tin); `sync_stock_parquet.py` và `crosscheck_tcbs.py` thành module `ai_core/prices/{sync,crosscheck}.py` với wrapper trong `jobs/`, vì `tests/prices/test_sync.py` và `test_crosscheck.py` import chúng như module.
