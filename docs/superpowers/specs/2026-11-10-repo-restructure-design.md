# Tái cấu trúc repo AI_Core: một repo, một package

Ngày: 2026-11-10. Trạng thái: bản nháp chờ duyệt.

## Mục đích

Hiện nay phần xử lý giá (`Pipeline_V1.2`) và phần xử lý tin tức (`NLP_Module`) là hai khối tách rời, nối với nhau bằng `sys.path` và một lớp keo (`ai_core`). Việc này gây ba loại đau:

1. Hai package top-level (`src`, `nlp_module`) không cài được; mọi script phải chạy từ đúng thư mục và `ai_core/config.py` phải chèn `sys.path`.
2. Ranh giới đặt sai chỗ: `nlp_module/forecast/` (volcast, walk_forward, fundamentals…) là mô hình hóa nhưng nằm trong NLP; `mmf` và `ai_core` import ngược vào đó, kể cả hàm private `walk_forward._split`. Phía NLP không import được `ai_core`, nên `board.py` chép tay `FUSION_VERSION`.
3. `NLP_Module` là một git repo riêng nằm lồng trong repo gốc (bị `.gitignore`), nên không có một commit nào mô tả trọn trạng thái hệ thống.

Mục tiêu là làm cho việc bảo trì, nâng cấp và phát triển hai phần giá/tin dễ hơn, bằng cách gom về một repo, một package cài được, ranh giới phụ thuộc rõ và có kiểm tra tự động.

## Mục tiêu

- Một repo gốc chứa toàn bộ code, **giữ nguyên lịch sử commit của NLP_Module** (quyết định của chủ dự án).
- Một package `ai_core` cài editable (`pip install -e .`), bỏ mọi `sys.path` hack và `# noqa` do import.
- Quy tắc phụ thuộc một chiều, có test chặn: `prices` và `news` không import nhau; `models` import `prices` và `news`; `fusion` import cả ba.
- **Không đổi hành vi**: sau khi chuyển, mọi test và số liệu dev giống hệt trước.

## Không phải mục tiêu

- Không sửa logic, không đổi tên hàm hay tham số (chỉ đổi đường dẫn import).
- Không gộp định nghĩa fold, không gộp virtualenv, không thống nhất danh sách mã (15/20/43). Các việc đó là bước sau, và việc gộp fold phải chờ sau holdout.
- Không tách thành nhiều package cài riêng.

## Điều làm thất bại dù "chạy được"

- Bất kỳ thay đổi nào làm đổi số liệu mmf/volcast so với trước (đặc biệt `dataset_id`, thứ tự cột, đơn vị giá), vì `mmf.holdout` chỉ được chạy một lần, khoảng cuối tháng 12/2026, với cấu hình đã khóa.
- Cron (đồng bộ giá 15:20, cross-check 15:35, Finhay 15:40, live mỗi 15 phút) ngừng chạy lặng lẽ.
- Mất lịch sử commit của NLP_Module hoặc mất file dữ liệu chưa được track.

## Cây thư mục đích

```
AI_Core/
├─ pyproject.toml              # package ai_core, pytest config, không còn pythonpath hack
├─ ai_core/
│   ├─ paths.py                # nơi duy nhất biết thư mục gốc và thư mục dữ liệu
│   ├─ prices/                 # ← Pipeline_V1.2/src (gói tên `src` đổi thành ai_core.prices)
│   ├─ news/                   # ← NLP_Module/nlp_module, trừ forecast/
│   ├─ models/
│   │   ├─ mmf/                # ← mmf/
│   │   └─ volcast/            # ← nlp_module/forecast/ (volcast, walk_forward, fundamentals, trial_forecast, text_return)
│   └─ fusion/                 # ← ai_core hiện tại (live, shadow, direction, finhay, splits, prices, config)
├─ jobs/                       # điểm vào cho cron, mỏng, không chứa logic
│   ├─ sync_prices.py          # ← Pipeline_V1.2/sync_stock_parquet.py
│   ├─ crosscheck_tcbs.py      # ← Pipeline_V1.2/crosscheck_tcbs.py
│   └─ live_update.sh          # ← NLP_Module/scripts/live_update.sh
├─ tests/                      # phản chiếu package: prices/, news/, models/, fusion/ (+ test kiến trúc)
├─ docs/                       # gộp docs của ba nơi; docs/reports/ giữ nguyên nội dung
└─ data/                       # xem "Dữ liệu"
```

`ai_core` hiện là package của lớp hợp nhất; trong cây mới nó là package gốc và lớp đó thành `ai_core.fusion`. Lệnh trong docs như `python -m mmf.experiment` đổi thành `python -m ai_core.models.mmf.experiment`.

## Dữ liệu

Dữ liệu không được git track, và một phần đang được cron ghi liên tục (giá, nlp.db 647 MB, live.lock). Vì vậy:

- Giai đoạn code: **dữ liệu ở nguyên chỗ cũ**. `ai_core/paths.py` trỏ về các vị trí hiện tại (`Pipeline_V1.2/stock`, `NLP_Module/data`, `data/`), thay cho các đường dẫn tính bằng `parents[n]` và đường dẫn tương đối rải rác (ví dụ `storage.py`, `segment.py`, `hose.py`, `entities.py`, `volcast.py`, `cli.py`).
- Giai đoạn dữ liệu (tùy chọn, làm cuối, trong một lần tạm dừng cron): dời dữ liệu về `data/prices`, `data/news`; chỉ đổi `paths.py` và crontab. Snapshot `24251acc1a5695bd` và các tham chiếu `Pipeline_V1.2/stock/snapshots/...` trong docs được cập nhật cùng lúc.
- Các file không được track trong NLP_Module (`data/*.csv`, `data/reports/`, `data/stock*`, `data/tessdata`, `CLAUDE.md`, `AGENTS.md`, `.claude/skills/`, thay đổi chưa commit trong `.claude/settings.json`) phải được sao lưu trước và đặt lại đúng chỗ mới; không được mất.

## Giữ lịch sử NLP_Module

`git filter-repo` không có trên máy. Cách làm là dùng git thuần: trong repo NLP_Module, dựng một nhánh mới chỉ `git mv` file vào đúng vị trí đích dưới thư mục con (và commit), rồi trong repo gốc `git fetch` nhánh đó và `git merge --allow-unrelated-histories`. Sau đó `git log --follow` trên một file của NLP phải còn thấy các commit cũ. NLP_Module hiện đứng ở nhánh `volcast-v2-bctc` (còn nhánh `main`); cả hai nhánh cần được xử lý (xem câu hỏi mở).

## Cách kiểm chứng "không đổi hành vi"

1. **Mốc trước khi chuyển** (lưu vào thư mục ngoài repo): kết quả `pytest` của ba bộ test (Pipeline_V1.2/tests, NLP_Module/tests, tests/); băm của khung dữ liệu mmf (`build_frame`) trên snapshot `24251acc1a5695bd`; kết quả `volcast predict` và `ai_core.live` trên dữ liệu hiện có; một lần chạy `mmf.experiment --smoke` với seed cố định.
2. **Sau khi chuyển**: cùng bộ lệnh, cùng số test pass (số test có thể đổi vị trí nhưng không đổi số lượng), băm và đầu ra giống từng bit.
3. `mmf_lock.json` chỉ chứa `dataset_id` (băm nội dung, không phụ thuộc đường dẫn) và `git_commit` (commit của repo gốc, vẫn tồn tại). Hai trường đó không bị ảnh hưởng, nhưng file lock không được sửa.
4. Test kiến trúc mới: quét import, báo lỗi khi `prices` import `news` (hoặc ngược lại), hoặc khi tầng dưới import tầng trên.

## Rủi ro đã biết

- **Hai worktree đang mở**: `price-v2` (chưa merge, chưa chứng minh có lợi) và `vnstock-gen6` (vnstock-migration/wt). Cả hai nằm trên cấu trúc cũ và sẽ lệch khi cây mới vào. Không đụng vào chúng trong đợt này; nếu sau này muốn giữ thì rebase.
- **Hai môi trường**: `NLP_Module/.venv` chạy cron, `.venv_mmf` dùng huấn luyện. Cả hai phải `pip install -e .` thì import mới chạy. Torch/numpy không được đổi phiên bản (bài học numpy 2.2.6 trên Python 3.14).
- **Cron dùng đường dẫn tuyệt đối** tới `Pipeline_V1.2/` và `NLP_Module/`; crontab chỉ đổi sau khi mọi kiểm chứng ở trên đạt, và có bản sao lưu crontab cũ.

## Thứ tự thực hiện (chi tiết để bước lập kế hoạch quyết định)

1. Sao lưu: crontab, file không được track của NLP_Module, mốc "trước".
2. Dựng nhánh NLP với lịch sử đã dời chỗ và hợp nhất vào repo gốc.
3. Chuyển code của Pipeline_V1.2 và các module còn lại; thêm `pyproject.toml` và `paths.py`; sửa import; bỏ `sys.path` hack.
4. Chạy kiểm chứng; đạt mới sang bước sau.
5. Đổi crontab và hướng dẫn; cập nhật docs.
6. (Tùy chọn, sau) Dời dữ liệu.

## Việc để người xây quyết

- Cách đặt tên file trong `jobs/` và cách chia `docs/` khi gộp ba nơi.
- Cách đặt test kiến trúc (bằng `ast` thuần hay công cụ có sẵn), miễn là không thêm dependency mới.
- Có xoá `*_backup_*` và `*.bak-*` không (không xoá trong đợt này).

## Quyết định của người viết (chủ dự án kiểm tra)

- **Nhánh nào được đưa vào**: `volcast-v2-bctc`. Đã kiểm tra: nó hơn `main` 24 commit và `main` không có commit nào mà nó thiếu, nên đưa nhánh này vào là mang theo trọn lịch sử của cả hai. Trong đó 23 commit chưa có trên `origin/volcast-v2-bctc`; việc gộp không đẩy gì lên remote. Repo GitHub `NLP_Module` giữ nguyên, không bị xoá hay sửa.
- **Thư mục `NLP_Module/` cũ** không bị xoá trong đợt này; sau khi kiểm chứng đạt và bạn đồng ý, nó chỉ được dời đi (không `rm`).
