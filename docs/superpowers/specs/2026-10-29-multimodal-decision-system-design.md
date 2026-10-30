# Hệ thống hỗ trợ quyết định đa phương thức (AI_Core) — thiết kế

Ngày: 2026-10-29. Thay thế bản plan "Late Fusion 2 tầng / 6 tuần" sau khi đối chiếu với mã nguồn.
Phạm vi: `ai_core/` (tầng tích hợp), `Pipeline_V1.2/` (nhánh giá DL, giữ nguyên làm đối chứng), `NLP_Module/`
(tin tức, volcast, evidence). Tầm dự báo T+2 và T+5.

Ký hiệu: **[người dùng]** = quyết định của chủ dự án; **[Claude]** = lựa chọn của người viết thiết kế, chủ dự án
có thể bác; **[builder]** = để người triển khai tự quyết.

---

## 1. Mục đích và lý do

Mục tiêu là trả lời câu hỏi: kết hợp giá, biến động và tin tức có cho ra một quyết định giao dịch tốt hơn
từng nhánh riêng lẻ không. Nếu không, hệ thống phải đóng băng thành một công cụ có ích mà không cần alpha:
cảnh báo biến động kèm bằng chứng từ tin tức (bản pivot trong plan gốc). **[người dùng]**

Bằng chứng đã có trong `NLP_Module/docs/model_report.md` quyết định thứ tự làm việc:

- **Hướng giá khó dự báo.** Dùng đặc trưng giá với logit, purged walk-forward 3 fold, AUC test chỉ đạt 0.49–0.585
  với target `direction` và 0.474–0.535 với target `excess`.
- **Tin tức không thêm giá trị cho dự báo hướng.** Kết luận này đứng vững qua 4 lần kiểm tra nhiễu: drift, lựa chọn
  model NLP, target excess, target absmove. Thử cả hồi quy text→return (Ridge) cũng không đạt quy tắc đặt trước.
- **Độ lớn biến động thì dự báo được.** `volcast-v2.1` có cổng Wilson. Accuracy test 0.620–0.720 trên các lệnh
  được phát, độ phủ 7.7–39.7%. Fold C đạt 0.626 (h=2) và 0.620 (h=5), vẫn dưới mốc 0.65.
- **Evidence layer đã hữu ích.** AUC pos-vs-neg 0.97, nhưng chỉ dựa trên 4 mẫu negative. Accuracy trên các nhãn được
  hiển thị (sau abstain) là 0.852.

Hệ quả: thí nghiệm hướng giá phải là **cổng go/no-go rẻ, làm sớm**. Không đầu tư vào fusion hay meta-model trước
khi cổng này trả lời.

## 2. Mục tiêu, không-mục-tiêu, phản-mục-tiêu

**Mục tiêu**
1. Dữ liệu giá đúng một chuẩn, không có vết nối do sự kiện quyền, chung cho mọi nhánh.
2. Một thí nghiệm hướng giá có chi phí, giao dịch được, đánh giá một lần trên test và có quyết định GO/NO-GO đặt trước.
3. Sửa các lỗi đã biết của NLP (địa danh, tài khoản ngân hàng), thêm cờ rủi ro sự kiện và một bộ challenge set
   cho lớp negative.
4. Một màn hình quyết định có bằng chứng, chạy shadow live hằng ngày sau 15:00.

**Không-mục-tiêu**
- Không viết lại `Pipeline_V1.2/src`. Nó giữ nguyên làm đối chứng E1 và tiếp tục chạy độc lập.
- Không thêm mạng sâu mới (LSTM/TCN/Transformer).
- Không làm meta-model (E8–E10 của plan gốc) trừ khi E7 vượt cổng ở bước 4.
- Không đổi model sentiment đang live (`baseline.joblib`, `phobert-v1`).
- Không thêm vào volcast các đặc trưng xếp hạng chéo (`cs_vol10`, `cs_range`, ...). Đã thử ở v3 và bị loại theo quy
  tắc đặt trước.
- Không đặt trần độ phủ cho cổng volcast dựa trên số liệu test đã đọc.

**Phản-mục-tiêu** (code chạy được nhưng vẫn coi là thất bại)
- Bất kỳ ngưỡng, siêu tham số hay lựa chọn model nào được rút ra từ test.
- Một kết quả fusion "tốt hơn" chỉ vì đã bỏ bớt giao dịch, không hơn một lệnh phủ quyết ngẫu nhiên cùng tỷ lệ.
- Phát tín hiệu trên dữ liệu có vết nối giá hoặc dữ liệu cũ (stale) thay vì trả `DATA_INVALID`.
- Lợi nhuận tính trên giao dịch không thực hiện được trên HOSE (bán khống, mua tại giá đóng cửa sau giờ chốt).

## 3. Ràng buộc

- **Mốc chốt dữ liệu:** 15:00 giờ Việt Nam (08:00 UTC) của ngày t. Đã có sẵn trong `trial_forecast.CLOSE_UTC`,
  `contract.available_at` và `BACKTEST_DELAY`; dùng lại, không định nghĩa lại.
- **Phiên khớp lệnh:** ATC trên HOSE kết thúc 14:45, trước mốc chốt. Quyết định của ngày t vì vậy **chỉ khớp được
  sớm nhất ở giá mở cửa t+1**.
- **Thanh toán T+2:** cổ phiếu mua ngày D về tài khoản chiều D+2 và bán được từ đó. Bán khống cổ phiếu cơ sở coi như
  không khả dụng. Hai điểm này cần chủ dự án xác nhận theo quy định hiện hành (xem mục 9).
- **Môi trường chạy:** CPU, xác định (deterministic). Seed lấy từ config của từng module. Venv `NLP_Module/.venv`.
  Test của `ai_core` nằm ở `AI_Core/tests` (pythonpath khai báo trong `pyproject.toml`).
- **Quy tắc đặt trước:** mọi quyết định ship hoặc go phải có tiêu chí ghi ra trước khi chạy và được ghi lại trong
  báo cáo, theo đúng nếp đang dùng trong `model_report.md`.
- **GitNexus:** `NLP_Module/CLAUDE.md` yêu cầu chạy `gitnexus_impact` trước khi sửa symbol và `gitnexus_detect_changes`
  trước khi commit. Nếu công cụ không có trong phiên thì ghi rõ trong báo cáo bước.
- Git: `NLP_Module/` đã có `.git` riêng. `ai_core/`, `tests/`, `docs/` và `Pipeline_V1.2/` chưa thuộc repo nào. Cách
  quản lý phiên bản: **[builder]**. Không bọc `NLP_Module` vào một repo lồng nhau.
- **Cửa sổ test đóng băng [Claude]:** mọi thí nghiệm trong thiết kế này (volcast ở bước 0–1, hướng giá ở bước 2,
  fusion ở bước 4) dùng fold C với `test_end = 2026-10-30`, tức là gồm cả phiên 29/10. Dữ liệu sau mốc này chỉ dùng
  cho shadow ở bước 5. `walk_forward.FOLDS` không đổi; các thí nghiệm truyền mốc này vào.
- **Model sentiment [Claude]:**
  - Backtest dùng snapshot chấm bằng model theo từng fold (`phobert-fold{A,B,C}`, như `walk_forward`), vì model live
    được train trên nhãn tới 2026-05 nên chồng lên test của fold A và B.
  - Live dùng model production `baseline-tfidf-lr-v1`.
  - Snapshot của 5 mã mới cũng phải dựng theo cách này.
- **Chi phí `c = 0.0045` [người dùng]:** chi phí khứ hồi, gồm phí môi giới hai chiều, thuế bán 0.1% và trượt giá.
  Cùng một hằng số dùng cho cả sàn của τ lẫn phần trừ mỗi lệnh (`r_trade - c`). Khai báo một lần trong
  `ai_core/config.py`.

## 4. Kiến trúc

```
Pipeline_V1.2/stock/*.parquet  (nguồn giá DUY NHẤT, 20 mã, một chuẩn điều chỉnh)
        │                                   NLP_Module/data/nlp.db (tin, market_quote, vol_forecast)
        ▼                                                │
 ai_core.prices.load_prices ──► volcast.add_features ◄───┘ (live bars qua ref)
        │                              │
        ├──► ai_core/direction.py (E0/E3/E4)   ──► side, margin        [Bước 2: GO/NO-GO]
        ├──► volcast (giữ nguyên, universe 20)  ──► p_strong, call
        └──► contract.build_snapshot            ──► has_news, sentiment_*, difficult_ratio, risk_flags
                                       ▼
                     ai_core/fusion.py (RuleGate, chỉ khi GO)           [Bước 4]
                                       ▼
        Quyết định: LONG | AVOID | NO_TRADE | DATA_INVALID  ──►  bảng fusion_decision + màn hình
```

Đổi so với plan gốc: `TRADE_UP` đổi thành `LONG`, `TRADE_DOWN` đổi thành `AVOID` (tránh mua hoặc giảm tỷ trọng,
không bán khống). **[Claude]**, lý do ở mục 3.

---

## 5. Các bước

Mỗi bước có tiêu chí xong. Bước sau chỉ bắt đầu khi bước trước đã xong, trừ khi bước đó ghi rõ có thể chạy song song.

### Bước −1 — Khôi phục live ở vị trí mới (khẩn)

Live chạy từ `AI_Core/NLP_Module` **[người dùng]**.

**Hiện trạng** (kiểm tra ngày 2026-10-29)
- Crontab gọi `/home/administrator/TLS/tuanhm@linux/NLP_Module/scripts/live_update.sh` và ghi log vào
  `.../NLP_Module/data/live.log`. Thư mục này không còn tồn tại, nên live đã ngừng từ khoảng 17:30.
- `.venv/bin/python` vẫn chạy được, vì nó là symlink tới `/usr/bin/python3` và `live_update.sh` gọi qua
  `python -m`. Riêng các console script (`pip`, `streamlit`, ...) có shebang trỏ về đường dẫn cũ nên bị hỏng.
- **vnstock chưa được cài ở môi trường nào trên máy** (`NLP_Module/.venv`, `~/.venv`, python hệ thống), và cũng không
  có trong `Pipeline_V1.2/requirements.txt`. Vì vậy `sync_stock_parquet.py` hiện không tải được dữ liệu.
- Chưa có lịch nào cho `sync_stock_parquet.py`.

**Việc cần làm**
1. Sửa crontab: đổi cả đường dẫn script lẫn đường dẫn log sang `AI_Core/NLP_Module`. Đường dẫn có `@` nên giữ
   nguyên dấu nháy.
2. Venv:
   - Dùng `python -m pip` và `python -m streamlit` thay cho console script, hoặc tạo lại venv từ `requirements.txt`:
     **[builder]**.
   - Cài vnstock vào venv mà sync sẽ dùng và ghi nó vào `Pipeline_V1.2/requirements.txt`.
3. Lên lịch sync: thêm một dòng cron cho các ngày thứ Hai–thứ Sáu, chạy `sync_stock_parquet.py` lúc 15:20 (sau
   `SESSION_DATA_READY` 15:00). Các lần `volcast predict` sau đó trong khung 15:00–16:00 sẽ đọc được bar đã sync.
   Sau bước 0, đây cũng là nơi chạy kiểm tra vết nối. Log ghi riêng, ví dụ `Pipeline_V1.2/outputs/sync.log`.
4. Chạy tay `scripts/live_update.sh` một lần để bù phần đã lỡ. Lệnh `collect` lấy cửa sổ hôm qua → hôm nay nên bù
   được tin. Quote sau giờ đóng cửa của hôm nay có thể đã mất; lúc đó volcast dùng bar lịch sử sau khi sync.

**Kết quả thực hiện (2026-10-29 19:10)**
- Mục 1 và 4: xong. Crontab đã trỏ sang `AI_Core/NLP_Module`, bản cũ lưu ở `~/crontab.bak-20261029`. Chạy tay một lần
  thành công: collect, process, và `vol_forecast` có `base_date` 2026-10-29.
- Mục 2 và 3: xong lúc 19:25.
  - Trên PyPI, `vnstock` và `vnai` đều ở trạng thái `quarantined`. Theo quyết định của chủ dự án, đã cài `vnai`
    2.6.4 (kho gói `vnstocks.com/api/simple`) và `vnstock` 4.0.9 (GitHub `thinh-vu/vnstock`, commit `71b853f`)
    vào `NLP_Module/.venv`, sau khi đọc mã.
  - Mã `vnai` không bị làm rối và không có exec/eval/subprocess. Nó ghi vào `~/.vnstock` và đăng ký mã thiết bị với
    `vnstocks.com`. Thống kê sử dụng mặc định tắt. Tính năng ghi vào `~/.claude/CLAUDE.md`, `AGENTS.md`, ... chỉ chạy
    khi được bật.
  - Cron sync chạy 15:20 các ngày thứ Hai–thứ Sáu, với `VNSTOCK_TELEMETRY=0 VNSTOCK_DISABLE_AGENT_SETUP=1
    VNSTOCK_AGENT_TARGETS=none`, log ở `Pipeline_V1.2/outputs/sync.log`. Đã ghi phiên bản vào
    `Pipeline_V1.2/requirements.txt`.
  - Chạy thử 15 mã: thành công, không cảnh báo lệch lịch, dữ liệu FPT giữ nguyên. Không có file agent nào được tạo.
  - Cron live chạy được lúc 19:15.
- Còn lại cho bước 0: sync chỉ cập nhật `Pipeline_V1.2/stock`; `NLP_Module/data/stock` vẫn là bản sao tay.

**Xong khi**
- `live.log` mới có dòng chạy sau 15 phút.
- Sync chạy được một lần thành công bằng vnstock (`check_alignment` trả True).
- `vol_forecast` có `base_date` của phiên giao dịch gần nhất.

### Bước 0 — Sửa vết nối giá (P0, đang ảnh hưởng live)

**Lỗi:** `TCB` đóng cửa 33.55 ngày 2026-10-27, rồi 22.35 ngày 2026-10-28 (-33.4%, ngày giao dịch không hưởng quyền).
Lỗi có ở cả `Pipeline_V1.2/stock` và `NLP_Module/data/stock`.

**Nguyên nhân:** `sync_stock_parquet.process_stock_file` chỉ tải các phiên từ ngày cuối trở đi rồi nối vào lịch sử
cũ. Lịch sử trước ex-date không bao giờ được điều chỉnh lại. Bản sửa của volcast v2.1 (`ret1 = close/ref - 1`) chỉ
chạy cho dòng live. Từ khi lịch sử phủ tới ngày đó, dòng lịch sử được ưu tiên trong `volcast.bars()`, `ref` thành
NaN, nên bản sửa không còn hiệu lực. Hệ quả: vol10/vol20 của TCB bị thổi phồng khoảng 20 phiên, nhãn absmove của
các ngày 21–27/10 sai, và Pipeline_V1.2 cũng dự báo trên chuỗi giá gãy.

**Việc cần làm**
1. Kiểm tra tay trước: tải lại toàn bộ lịch sử TCB từ vnstock, xem nguồn có trả chuỗi đã điều chỉnh qua ngày
   28/10 hay không. Kết quả quyết định nhánh a hoặc b ở mục 2. **Đã kiểm tra ngày 2026-10-29:** nguồn có điều chỉnh
   (TCB 27/10 tải mới là 22.38, bản lưu là 33.55; FPT không đổi), nên đi theo **nhánh a**.
2. Sửa trong sync:
   - a. Nếu nguồn có điều chỉnh: sync so sánh phiên chồng lấn. Lần tải mới vốn đã bắt đầu từ `last_record_date`
     nên luôn có một phiên chồng. Nếu giá đóng cửa của phiên đó lệch quá mức làm tròn, hoặc lịch sử sau khi ghép
     có bước nhảy vượt `RET_BAND` (0.072), thì **tải lại toàn bộ lịch sử của mã đó và ghi đè**.
   - b. Nếu nguồn không điều chỉnh: phát hiện như trên, ghi `(symbol, date)` vào danh sách đen. Mọi cửa sổ đặc
     trưng hoặc nhãn vắt qua ngày đó cho ra `DATA_INVALID` (fail-closed). Hệ số điều chỉnh lấy từ
     `market_quote.ref_price` nếu có.
   - Chọn cách cài đặt cụ thể: **[builder]**.
3. Đảm bảo một nguồn giá duy nhất. Hiện `NLP_Module/data/stock` là bản sao tay của `Pipeline_V1.2/stock`. Cho volcast
   đọc thẳng từ thư mục của Pipeline, hoặc tạo một bước copy có kiểm tra: **[builder]**. Test
   `test_pipeline_prices_cover_nlp_copy` phải còn đúng.
4. Chạy lại eval volcast trên dữ liệu đã sửa, 15 mã, cửa sổ test đóng băng. Con số
   `mean(test_acc - rule_acc)` thu được là **mốc so sánh mới** cho bước 1, thay cho +0.0549 vốn tính trên chuỗi TCB
   gãy. Sau đó train lại và chạy lại `predict`.
5. Pipeline_V1.2 **không train lại** trong thiết kế này **[Claude]**. E1 dùng kết quả đã có, kèm chú thích là được
   train trên dữ liệu trước khi sửa. Nếu sửa dữ liệu làm thay đổi giá FPT (nhánh a), test
   `test_pipeline_prices_cover_nlp_copy` được viết lại theo nguồn giá duy nhất, không giữ so sánh với bản sao cũ.

**Xong khi**
- `check_prices()` trên 15 mã hiện có trả `n_big_moves == 0`.
- Test mới `test_no_corporate_action_seam` pass.
- Test tổng hợp cho logic phát hiện và tải lại của sync pass.
- `vol_forecast` của TCB đã được cập nhật.

### Bước 1 — Universe 20 mã

Universe chung = hợp của hai danh sách **[Claude]**:
- Pipeline 15: FPT VNM STB ACB SSI VIC HPG VCB MSN MBB GAS MWG VPB VRE TCB.
- NLP 15: ACB BID CTG FPT HDB HPG MSN MWG NVL STB TCB VCB VHM VIC VPB.

Cả 20 mã đều có tin trong `nlp.db`. VNM, SSI, MBB, VRE và GAS có ít tin trực tiếp (9–136 bài), nên phần lớn dòng sẽ
có `has_news = 0`. Đây là dữ liệu hợp lệ, không phải lỗi.

**Việc cần làm**
1. Tải lịch sử OHLCV từ 2021-01-04 cho BID, CTG, HDB, VHM, NVL vào `Pipeline_V1.2/stock/`. Hiện sync cần một CSV có
   sẵn để bắt đầu (nó glob `*_stock.csv`), nên cần tạo file khởi đầu hoặc thêm chế độ khởi tạo: **[builder]**.
2. `Pipeline_V1.2/src/config.TICKERS` **giữ nguyên 15 mã**. Pipeline nạp theo `C.TICKERS` nên không bị ảnh hưởng.
3. Thêm cấu hình universe vào `ai_core/config.py` (`UNIVERSE` = 20 mã) và sửa chú thích cũ. Đặt
   `trial_forecast.UNIVERSE` bằng 20 mã.
4. Tạo snapshot NLP cho 5 mã mới thêm vào backtest (VNM, SSI, MBB, GAS, VRE).
5. Volcast dùng rổ 20 mã. Rổ thay đổi thì median của absmove và `mret*` cũng đổi, nên phải chạy lại eval
   walk-forward như một lần thử có quy tắc đặt trước. Quy tắc ship giống lần thử Wilson: (a) trung bình
   `test_acc - rule_acc` trên 6 ô không thấp hơn mốc 15 mã đã sửa ở bước 0; (b) fold C `test_n ≥ 100` ở cả hai h.
   - Nếu không đạt: volcast live giữ rổ 15 mã cũ. Fusion vẫn dùng 20 mã, và 5 mã không có volcast nhận `call = None`.
   - Ngưỡng (a) và (b) do **[Claude]** đặt.

**Xong khi**
- `tests/test_alignment.py` pass với 20 mã: cùng lịch, không trùng ngày, không NaN, không có vết nối.
- Có báo cáo eval volcast 20 mã cùng quyết định ship.

### Bước 2 — Thí nghiệm hướng giá (cổng GO/NO-GO)

File mới `ai_core/direction.py` và `tests/test_direction.py`. Không sửa `Pipeline_V1.2/src`.

**Dữ liệu và đặc trưng**
- Đặc trưng lấy từ `volcast.add_features`, dùng lại nguyên các cột giá: ret1, ret5, ret20, vol10, vol20, ma20,
  range, vz, mret1, mret5.
- **Không dùng BCTC** cho bài toán hướng, vì giá trị BCTC gần như là hằng số theo mã **[Claude]**.
- Bổ sung đặc trưng "lợi suất vượt rổ trong quá khứ": `exret_k = ret_k - mean_k` theo ngày, với k ∈ {1, 5, 20}.

**Nhãn** (chọn **[Claude]**, đổi so với plan gốc để giao dịch được, xem mục 3)
- Lợi suất giao dịch được: `r_trade(i,t,h) = log(close[t+1+h] / open[t+1])`. Mua ở giá mở cửa t+1, giữ h phiên.
  Với h ≥ 2 thì hợp lệ theo T+2.
- Lợi suất vượt rổ: `r_ex = r_trade - mean_rổ(r_trade)` trong cùng ngày t.
- Ngưỡng: `τ(i,t,h) = max(c, κ · vol20(i,t) · √h)`, với `c = 0.0045` và `κ = 0.5`. Cố định trước, không tối ưu, vì
  đổi κ là đổi nhãn.
- `y` = UP nếu `r_ex > τ`; DOWN nếu `r_ex < -τ`; NONE nếu ở giữa.
- Dòng không đủ dữ liệu tương lai, hoặc vắt qua ngày `DATA_INVALID`, nhận nhãn NaN và bị loại.

**Mô hình**
- E0: prior/majority, cộng random-same-count (chọn ngẫu nhiên đúng số lệnh như mô hình, 1000 lần).
- E3: `StandardScaler + LogisticRegression` đa lớp, `C ∈ {0.01, 0.1, 1.0}`.
- E4: `HistGradientBoostingClassifier` với cấu hình như `volcast.MODELS["gbm"]`.
- E1: kết quả hiện có của Pipeline_V1.2 (split 70/15/15 riêng), chỉ ghi để tham khảo, kèm chú thích không cùng fold.
  **Không có E2**: NLinear trên log(close) vốn đã học trên log-return tương đối.

**Chia dữ liệu**
- `ai_core.splits.folds(df, h)` (fold A/B/C, purge h phiên ở mỗi ranh giới).
- Trên train: fit model, chọn C hoặc model, và hiệu chỉnh sigmoid. Trên val: chọn margin `m` và loại model.
- Refit trên train+val, chấm test **một lần**.
- Đặc trưng có thể vắt qua nhiều phiên nên purge `1 + h` phiên thay vì `h`, vì nhãn đọc tới t+1+h **[Claude]**.

**Quy tắc giao dịch**
- `LONG` nếu `P(UP) - P(DOWN) ≥ m`. `AVOID` nếu `P(DOWN) - P(UP) ≥ m` (chỉ ghi nhận, không tính PnL).
  Trường hợp còn lại là `NO_TRADE`.
- Chọn `m` trên val để tối đa lợi suất ròng trung bình của mỗi lệnh LONG, với tối thiểu 50 lệnh.

**Chỉ số trên test**
- Cho LONG: số lệnh, hit rate, lợi suất ròng trung bình của mỗi lệnh (`r_trade - c`) và lợi suất vượt rổ ròng
  (`r_ex - c`), kèm CI 95% bằng block bootstrap theo ngày (khối 20 phiên, như `Pipeline_V1.2` config).
- Turnover, và max drawdown của danh mục đều trọng số các lệnh LONG trong ngày.

**Quy tắc GO** (đặt trước, xét riêng từng h). GO khi model được chọn trên val thỏa cả ba:
1. Lợi suất ròng trung bình mỗi lệnh `> 0` ở ít nhất 2/3 fold.
2. Hơn trung vị của random-same-count ở ít nhất 2/3 fold.
3. Gộp 3 fold, cận dưới CI 95% của lợi suất ròng vượt rổ `> 0`. Điều kiện này **[Claude]** thêm vào để GO không
   đến từ nhiễu.

Nếu cả hai h đều NO-GO thì dừng nhánh hướng (quy tắc dừng 1 của plan gốc), bỏ qua bước 4 và đi tiếp bước 5.

**Xong khi**
- Có `data/reports/direction_report.txt` (hoặc tương đương trong `AI_Core`) gồm bảng E0/E1/E3/E4 × fold × h và
  quyết định GO/NO-GO.
- Các test ở mục 6 liên quan tới bước này đều pass.

### Bước 3 — NLP: lỗi đã biết, cờ rủi ro, challenge set

Có thể chạy song song với bước 2.

1. **Địa danh và tài khoản trong `entities.py`.**
   - Bỏ link khi alias đứng sau ngữ cảnh địa điểm, ví dụ "tại|ở|biệt thự tại|dự án" + tên khu đô thị (Vinhomes
     Riverside, Ocean Park, Smart City, ...).
   - Bỏ link khi alias đứng sau ngữ cảnh tài khoản, ví dụ "tài khoản|chuyển khoản|số tài khoản (mở) tại".
   - Không thêm taxonomy 6 trạng thái: `contract` chỉ phân biệt `market` với phần còn lại, nên chỉ cần loại link.
     Nếu cần audit thì đánh dấu `relation='mention'`; `build_snapshot` loại `mention` giống như loại `market`
     **[Claude]**.
   - Đổi linker làm thay đổi kết quả cũ. Phải chạy `link_changes` (đã có test) để báo cáo số link bị bỏ, và chạy lại
     `process` cho các phiên bản model đang live.
2. **Cờ rủi ro sự kiện.** Hàm thuần `risk_flags(passage_text) -> list[str]`, dùng regex, gồm các nhóm:
   - `guarantee` (bảo lãnh vô điều kiện, bảo lãnh khoản vay)
   - `seizure` (thu giữ tài sản, phát mại)
   - `prosecution` (khởi tố, bắt tạm giam)
   - `sanction` (thuế chống bán phá giá, phạt)
   - `bond_default` (chậm thanh toán lãi/gốc trái phiếu)

   Gọi lúc dựng snapshot. Thêm đặc trưng `risk_flag_72h` (số bài có cờ trong cửa sổ), tăng `FEATURE_VERSION` lên
   `v4`, và hiển thị cờ trên màn hình. Không đổi schema DB **[Claude]**.
3. **Challenge set cho lớp negative.** Lấy mẫu bằng `annotate.py` các passage mà một trong hai model chấm negative,
   hoặc khớp `risk_flags`, từ ít nhất 30 bài độc lập. **Chủ dự án tự gán nhãn** theo guideline v1-strict. Bộ này đánh
   giá riêng, không gộp vào eval 240 mẫu.
4. **Kiểm định BCTC có phải "học vẹt mã" không**, trong volcast eval:
   - (a) Hoán vị: xáo trộn ánh xạ mã → giá trị BCTC giữa các mã, 20 lần.
   - (b) Leave-one-ticker-out: bỏ từng mã khỏi train, chấm trên mã đó.
   - Quy tắc đặt trước: nếu trung bình test_acc của (a) không thấp hơn BCTC thật quá 0.01, thì kết luận BCTC chỉ mã
     hóa danh tính mã và ghi vào báo cáo. Volcast live không đổi trừ khi có quyết định riêng.
5. **Hoãn:** parser số tuyệt đối của STB trong `fundamentals.py` (đặc trưng FUND không được dùng ở đâu) và
   `source_coverage_status`. Các mốc thời gian của bài báo đã có trong schema (`published_at`, `source_updated_at`,
   `first_seen_at`, `version_first_seen_at`, `processed_at`).

**Xong khi**
- Các test 5 và 6 ở mục 6 pass.
- Có báo cáo `link_changes`.
- Có báo cáo challenge set: số mẫu, F1 negative và CI.
- Có báo cáo kiểm định BCTC.

### Bước 4 — Fusion RuleGate (chỉ khi bước 2 GO)

File `ai_core/fusion.py`. Áp dụng cho h nào GO. Thứ tự luật:

1. Dữ liệu không hợp lệ (vết nối, thiếu phiên, quote live cũ, lỗi parser): trả `DATA_INVALID`.
2. `|P(UP) - P(DOWN)| < m` (lấy từ bước 2): trả `NO_TRADE`.
3. Có tin (`has_news = 1` và `difficult_ratio < 1`):
   - Có `risk_flag_72h > 0`, hoặc `sentiment_mean_24h` ngược dấu với side và vượt ngưỡng `s`: trả `NO_TRADE` (phủ quyết).
   - Nếu không: giữ side.
4. Không có tin: giữ side chỉ khi margin `≥ m_strict` và volcast `call != "weak"`.

`s` và `m_strict` chọn trên val. Hiệu chỉnh xác suất (sigmoid) của từng nhánh fit trên val.

**Đối chứng bắt buộc**
- E4 (hoặc E3) đứng một mình.
- Phủ quyết ngẫu nhiên đúng bằng tỷ lệ mà RuleGate phủ quyết trên từng fold, 1000 lần.
- Báo cáo riêng 3 chiến lược: `price-only`, `news-gated`, `all-gated`.

**Quy tắc vượt cổng:** E7 vượt khi lợi suất ròng trung bình mỗi lệnh lớn hơn E4 **và** lớn hơn phân vị 95 của phủ
quyết ngẫu nhiên, ở ít nhất 2/3 fold. Không vượt thì giữ price-only (nếu GO) và ghi kết luận "tin tức không cải
thiện quyết định".

**Meta-model (E8):** chỉ mở thành thiết kế riêng nếu E7 vượt cổng.

### Bước 5 — Màn hình quyết định và shadow live

1. **Bảng `fusion_decision`** trong `nlp.db`, khai báo trong `schema.sql`, gồm: symbol, base_date, h, decision,
   p_up, p_down, margin, p_strong, call, has_news, sentiment_mean_24h, risk_flags, reason, version, created_at.
   - Khi NO-GO: `decision` chỉ nhận `NO_TRADE` hoặc `DATA_INVALID`; màn hình là terminal biến động và rủi ro.
2. **Bước `fusion predict`** trong `scripts/live_update.sh`, chạy cùng khung với `volcast predict` (15:00–16:00).
   Circuit breaker: không có quote sau giờ đóng cửa của hôm nay vào ngày giao dịch, hoặc sync báo vết nối, thì ghi
   `DATA_INVALID` cho mã bị ảnh hưởng.
3. **Màn hình:** thêm tab vào `NLP_Module/app.py` (Streamlit đã có). Mỗi mã một dòng gồm quyết định, volcast call
   và p_strong, các bài tin trong 72 giờ (passage, nhãn, cờ rủi ro) và lý do của quyết định.
4. **Shadow:** chạy tối thiểu 40 phiên (khoảng 2 tháng) **[Claude]**. Sau đó chấm các quyết định đã ghi trên dữ liệu
   thật bằng chính các chỉ số của bước 2, và so sánh với kết quả backtest.

**Xong khi**
- Bảng có dữ liệu mỗi ngày giao dịch.
- Màn hình chạy được.
- Có báo cáo shadow sau 40 phiên.

---

## 6. Bộ test bắt buộc (regression gate)

Mọi test phải pass trước khi ghi model vào `data/models/` hoặc `outputs/models/`.

| # | Test | Nội dung |
|---|---|---|
| 1 | `test_no_corporate_action_seam` | Mọi mã trong universe: không có `|ret1| > RET_BAND` trong lịch sử, trừ ngày đã vào danh sách đen. Thêm một ca tổng hợp: phiên chồng lấn lệch giá → sync tải lại toàn bộ (mock nguồn) |
| 2 | `test_session_calendar` | Lịch phiên = các ngày có trong lịch sử, giống nhau ở mọi mã. Ngày nghỉ quá khứ (Tết, 30/4, 2/9) không bị coi là thiếu phiên. Ca tổng hợp: đoạn live thiếu một ngày thường không thuộc `MARKET_HOLIDAYS` → các cửa sổ vắt qua cho ra `DATA_INVALID` |
| 3 | `test_purged_split_no_leakage` | Đổi giá của tập test → đặc trưng, scaler, calibrator, `m` và model chọn trên train/val không đổi |
| 4 | `test_gate_wilson_bound` | Với ngưỡng mà `pick_threshold` trả về, cận dưới Wilson trên val ≥ `min_acc` và số lệnh ≥ `min_n`. Không có ngưỡng nào đạt thì trả None |
| 5 | `test_entity_landmark_and_account` | "BIDV thu giữ biệt thự tại Vinhomes Riverside" không link VHM. "chuyển tiền vào tài khoản mở tại Vietcombank" không link VCB. Bài thật về Vinhomes vẫn link VHM |
| 6 | `test_risk_flags` | "MWG bảo lãnh vô điều kiện cho Bách Hóa Xanh vay 2.000 tỷ" trả `["guarantee"]`. Bài không có từ khóa trả `[]` |
| 7 | `test_tradable_label` | `r_trade` dùng open[t+1] → close[t+1+h]. Dòng cuối dữ liệu không có nhãn. Nhãn ba lớp tôn trọng τ |
| 8 | `test_fusion_rules` | Mỗi luật ở bước 4 có một ca. Đầu vào `DATA_INVALID` không bao giờ ra `LONG` (chỉ khi bước 4 được làm) |

## 7. Phần để lại cho builder

- Cách cài đặt phát hiện và tải lại trong sync, và cách tạo CSV khởi đầu cho mã mới.
- Volcast đọc giá từ đâu (đọc thẳng Pipeline hay qua bước copy có kiểm tra).
- Cấu trúc nội bộ của `ai_core/direction.py` và `ai_core/fusion.py`, định dạng file báo cáo.
- Danh sách regex cụ thể cho `risk_flags` và cho ngữ cảnh địa danh/tài khoản (bắt đầu từ các ví dụ ở trên và từ
  các ca lỗi trong `model_report.md`).
- Bố cục tab Streamlit.

## 8. Các lựa chọn của Claude (chủ dự án nên rà lại)

1. Universe 20 mã (hợp hai danh sách) thay vì chọn 15.
2. `TRADE_DOWN` đổi thành `AVOID`. PnL chỉ tính cho chiều long.
3. Nhãn và PnL dùng lợi suất giao dịch được: open t+1 → close t+1+h, purge `1 + h` phiên.
4. `c = 0.0045` và `κ = 0.5` cố định trước, không tối ưu.
5. Bỏ BCTC khỏi bài toán hướng. Bỏ E2.
6. Quy tắc GO có thêm điều kiện cận dưới CI gộp 3 fold > 0.
7. Ngưỡng ship cho volcast 20 mã: không thấp hơn mốc 15 mã đã sửa ở bước 0, fold C `test_n ≥ 100`.
11. Đóng băng cửa sổ test fold C tại `2026-10-30`.
12. Backtest dùng snapshot sentiment theo từng fold; live dùng `baseline-tfidf-lr-v1`.
13. Không train lại Pipeline_V1.2; E1 lấy kết quả đã có.
8. Không đổi schema DB cho entity hay cờ rủi ro. Thêm `relation='mention'` (chỉ nếu cần audit) và `FEATURE_VERSION v4`.
9. Kiểm định BCTC: hoán vị 20 lần, ngưỡng 0.01.
10. Shadow tối thiểu 40 phiên.

## 9. Câu hỏi mở cho chủ dự án

- Xác nhận quy định hiện hành: thanh toán T+2 (bán được từ chiều D+2) và không bán khống cổ phiếu cơ sở. Nếu
  khác, bước 2 phải đổi định nghĩa `r_trade` và có thể khôi phục chiều short.
