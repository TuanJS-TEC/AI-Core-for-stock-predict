# Tab Tin tức của StockSenseVN hiện kết quả NLP của AI_Core: thiết kế

Ngày: 2026-11-12. Người yêu cầu: chủ dự án AI_Core.
Repo liên quan: AI_Core (repo này) và StockSenseVN (`../StockSenseVN`, web Node 22 + PostgreSQL của nhóm khác; chủ dự án cho phép sửa code bên đó).
Bối cảnh đã có: cầu nối `ai_core/fusion/stocksense.py` đẩy dự báo sang Postgres của StockSenseVN (`docs/superpowers/plans/2026-11-12-stocksense-bridge.md`).

## Mục đích và lý do

Tab **Tin tức** trên trang chi tiết mã của StockSenseVN hiện chỉ có tin mẫu ("StockSense Demo") với nhãn cảm xúc giả. Module NLP của AI_Core (`ai_core/news`) đã thu thập và chấm khoảng 17.500 bài (CafeF, Vietstock, HOSE). Với mỗi cặp (bài, mã), nó có: xác suất tích cực / trung tính / tiêu cực, cờ "không chắc" (abstain), mức liên quan, cờ rủi ro sự kiện, cụm bài trùng, đoạn văn đã chấm. Chủ dự án muốn người dùng mở tab Tin tức của một mã là thấy tin thật của mã đó kèm kết quả NLP thật, và tin mới phải lên web gần như ngay khi có.

## Những gì người dùng thấy (chủ dự án đã chốt)

- Tab Tin tức của một mã mặc định hiện **các tin trong 7 ngày gần nhất** của mã đó, mới nhất trước. **Không hiện gì thêm**: không có dòng tóm tắt đầu tab.
- **Giữ hai ô "Từ ngày / Đến ngày"** như hiện nay. Mặc định là 7 ngày gần nhất (từ hôm nay − 7 đến hôm nay); người dùng đổi được để xem tin cũ hơn, vì bảng mới giữ toàn bộ lịch sử.
- **Trang đang mở tự cập nhật theo thời gian thực**: khi NLP chấm xong và bài được đẩy sang, bài mới hiện trên trang trong vài giây mà người dùng không phải tải lại. Bài mới chỉ hiện nếu nằm trong khoảng ngày đang chọn. Bài người dùng đang mở phân tích vẫn giữ trạng thái mở.
- Mỗi bài là một dòng gọn (phương án **B** trong bản phác thảo đã duyệt):
  - tiêu đề, nguồn, giờ đăng (giờ Việt Nam);
  - chip **cảm xúc** kèm phần trăm của nhãn (ví dụ "Tích cực 88%"), hoặc chip **"Chưa chắc"** khi model abstain;
  - chip **mức liên quan**: "Nói trực tiếp về {MÃ}" (direct), "Tin ngành" (sector), "Tin thị trường chung" (market), "Liên quan chưa rõ" (ambiguous);
  - chip **cờ rủi ro**, chỉ khi có (tên tiếng Việt cho các khóa `guarantee`, `seizure`, `prosecution`, `sanction`, `bond_default` của `ai_core.news.risk.FLAGS`).
- **Bấm vào một bài** thì mở phần phân tích ngay dưới bài; bấm lần nữa thì đóng. Phần này gồm:
  - thanh xác suất ba màu và ba con số tích cực / trung tính / tiêu cực;
  - nếu "Chưa chắc": một câu giải thích rằng model không đủ tự tin để gán nhãn và bài không được tính vào điểm cảm xúc của mã;
  - đoạn văn model đã đọc (`passage_text`);
  - cờ rủi ro (hoặc "không có");
  - tên model đã chấm;
  - link mở bài gốc ở tab mới.
- Tin mới lên web khoảng **1–3 phút** sau khi xuất hiện trong danh sách bài của trang nguồn, **từ 07:00 đến 22:00** giờ Việt Nam. Ngoài khung giờ đó giữ nhịp 15 phút như hiện nay. Chủ dự án đã chấp nhận cả hai.
- **Mọi phần tin trên trang chi tiết mã dùng chung nguồn tin mới**, để một trang không bao giờ hiện hai bộ tin mâu thuẫn:
  - thẻ "Tin mới nhất" ở tab Dự báo (3 bài mới nhất);
  - chấm đánh dấu ngày có tin trên biểu đồ nến, vẫn trải suốt lịch sử giá như hiện nay;
  - dòng "Tin tức: N bài" ở khung Trạng thái dữ liệu;
  - chip "Thiếu tin" / "Thiếu tin 7 ngày gần nhất": bật khi mã không có bài nào trong 7 ngày gần nhất theo nguồn mới.

## Mục tiêu, ngoài phạm vi, điều không được xảy ra

**Mục tiêu**
- Tab Tin tức và mọi phần tin khác trên trang chi tiết mã hiển thị đúng như trên, với dữ liệu thật từ kho tin của AI_Core.
- Chu kỳ thu thập, chấm và đẩy tin rút từ 15 phút xuống khoảng 1–2 phút trong khung 07:00–22:00.

**Ngoài phạm vi**
- Các tính năng ngoài trang chi tiết mã đang dùng bảng `news_items` của StockSenseVN giữ nguyên: chat, PDF, app mobile, bộ đồng bộ tin CafeF/VnExpress của họ.
- Không chấm NLP cho tin do StockSenseVN tự thu thập.
- Không thêm nguồn tin mới, không huấn luyện lại model.

**Không được xảy ra**
- Hiện nhãn cảm xúc cho bài mà model đã abstain (bài "chưa chắc" phải hiện là "Chưa chắc", không phải "Trung tính").
- Cùng một tin hiện nhiều lần (bài trùng, hoặc cùng bài với hai phiên bản).
- Thu thập dày làm hỏng vòng live hiện có: dự báo volcast, bảng quyết định, quote SSI và việc đẩy dự báo phải chạy đúng như trước.
- Bị trang nguồn chặn vì gửi quá nhiều request.

## Cách làm (do trợ lý chọn; chủ dự án đã nghe và không phản đối)

1. **Nguồn tin là kho tin của AI_Core** (`data/news/nlp.db`), không phải bảng `news_items` của StockSenseVN. Lý do: mọi bài trong kho đã có kết quả NLP.
2. **Điểm hiển thị là của PhoBERT** (`nlp_model_version = 'phobert-phobert-v1'`). Mỗi bài cũng có điểm của model baseline (`baseline-tfidf-lr-v1`), nhưng không hiện.
3. **Chỉ bài đại diện** (`is_representative`) của mỗi cụm bài trùng, và chỉ phiên bản mới nhất của mỗi bài.
4. **Giữ đúng quy ước giao tiếp của StockSenseVN** (SDD D3: các tiến trình chỉ nói chuyện qua PostgreSQL). Cầu nối AI_Core ghi bài và kết quả NLP vào **một bảng mới** trong Postgres của StockSenseVN, tạo bằng một migration mới của họ. API của họ đọc bảng đó. API Node không gọi Python, Python không gọi HTTP sang Node.
5. **Bảng mới giữ toàn bộ lịch sử tin của các mã** (để biểu đồ nến vẫn có chấm tin suốt lịch sử), không chỉ 7 ngày. Lần đẩy đầu tiên nạp toàn bộ lịch sử; các vòng sau chỉ đẩy phần mới hoặc đổi, để một vòng 1–2 phút vẫn rẻ. Ghi được cập nhật lại (khác với dự báo, vốn bất biến): bài được chấm lại thì ghi đè dòng cũ của cùng (bài, mã).
6. **"Chấm ngay khi có tin"** nghĩa là: vòng thu thập → chấm → đẩy chạy khoảng mỗi 1–2 phút trong khung 07:00–22:00 (hiện mất khoảng 75 giây mỗi vòng), và đẩy ngay sau khi chấm xong. Ngoài khung giờ đó vẫn 15 phút một vòng. Bấm vào bài chỉ hiện kết quả đã chấm sẵn, không chạy model theo từng lần bấm. Bài đã thu thập nhưng chưa có kết quả PhoBERT thì chưa hiện, và sẽ hiện ở vòng kế tiếp.
7. **Phía StockSenseVN:**
   - API: endpoint mới trả tin của mã từ bảng mới kèm các trường NLP, cùng quyền `news.read` như endpoint tin hiện có. Tab Tin tức và thẻ "Tin mới nhất" lấy 7 ngày; biểu đồ nến cần đủ ngày có tin suốt lịch sử giá.
   - `missingNews` trong `readSymbol` (`apps/api/src/market/repository.js`) tính từ bảng mới: không có bài nào của mã trong 7 ngày gần nhất.
   - Web: tab Tin tức chuyển sang giao diện B, giữ hai ô "Từ ngày / Đến ngày" với mặc định 7 ngày. Thẻ "Tin mới nhất", chấm tin trên biểu đồ nến và dòng "Tin tức: N bài" chuyển sang nguồn mới.
   - Endpoint tin cũ giữ nguyên cho các tính năng ngoài phạm vi.
8. **Cập nhật thời gian thực** vẫn phải giữ quy ước D3 (không gọi trực tiếp giữa Python và Node). Cách hợp với D3: cầu nối phát tín hiệu qua PostgreSQL (`NOTIFY`) sau mỗi lần đẩy có bài mới hoặc đổi; API Node nghe (`LISTEN`) rồi báo cho trình duyệt đang mở trang của mã đó (SSE hoặc WebSocket); trình duyệt tải lại phần tin. Người xây có thể chọn cách khác nếu đạt cùng yêu cầu "vài giây, không tải lại trang" và không phá D3. Mọi phần tin trên trang (danh sách, thẻ "Tin mới nhất", chấm trên biểu đồ, "N bài", chip "Thiếu tin") cập nhật cùng lúc.

## Ràng buộc

- AI_Core: luồng phụ thuộc một chiều (`tests/test_architecture.py`); `ai_core/__init__.py`, `prices/__init__.py`, `fusion/__init__.py` để trống; chạy trên `.venv`. Chuỗi kết nối Postgres là `STOCKSENSE_DATABASE_URL` trong `.env` ở gốc repo, đọc bằng `sed`, không bao giờ `source`. Kết nối có timeout như cầu nối dự báo (kết nối 10 giây, câu lệnh 60 giây, chờ khóa 30 giây).
- Vòng live dùng chung khóa `data/news/live.lock`. Một vòng chạy quá lâu thì vòng sau bỏ qua, không chạy chồng.
- Các bước khác trong `jobs/live_update.sh` (quote SSI 9:00–15:15, `volcast predict` + `fusion.live` + đẩy dự báo 15:00–16:00) giữ đúng cửa sổ giờ và nhịp hiện tại. Chỉ phần tin chạy dày hơn, và chỉ trong 07:00–22:00.
- Thu thập phải giữ phép lịch sự hiện có của `ai_core.news.http` (robots.txt, giới hạn tốc độ). Số lần tải danh sách bài của mỗi nguồn trong ngày tăng khoảng 8–15 lần so với hiện nay (chủ dự án chấp nhận, đổi lại ban đêm giữ 15 phút); không tăng thêm nữa.
- StockSenseVN: migration mới thêm file mới, không sửa file đã áp dụng (`db/migrate.js` kiểm checksum). SQL tham số hóa, không ORM. Test bằng vitest; `npm test` và `npm run build:web` phải xanh.
- Cả hai repo chưa được commit; chủ dự án quyết định khi nào commit.

## Để người xây quyết định

- Tên và cột của bảng mới, tên endpoint (một hay hai endpoint), hình dạng JSON trả về.
- Cơ chế đẩy thời gian thực tới trình duyệt (trong giới hạn ở mục 8), cách tự kết nối lại khi mất kết nối.
- Cách cho phần tin chạy mỗi 1–2 phút trong 07:00–22:00 mà không kéo các bước khác theo: tách script, thêm cờ, hay đổi lịch cron.
- Cách nhận biết "phần mới hoặc đổi" để đẩy tăng dần (ví dụ theo `processed_at`).
- Cách lấy title, URL, giờ đăng, đoạn văn, cờ rủi ro và cờ đại diện từ `nlp.db`. `ai_core.news.pipeline.load_results` đã trả `ArticleCompanyResult` có `risk_flags` và `is_representative`; title, URL và đoạn văn nằm ở `document_version`, `document`, `article_company_result`. Theo kiểm tra của reviewer: không có cặp (phiên bản bài, mã) nào có hai kết quả PhoBERT, và bài đại diện không đổi sau khi đã gán.
- Câu chữ tiếng Việt của các chip và câu giải thích, theo đúng nghĩa ở trên.

## Quyết định đã chốt với chủ dự án ngày 2026-11-12

- Giữ hai ô "Từ ngày / Đến ngày", mặc định 7 ngày.
- Trang đang mở cập nhật thời gian thực sau khi NLP chấm xong.
- Chỉ hiện điểm PhoBERT.
- Bảng mới giữ toàn bộ lịch sử tin, mỗi vòng chỉ đẩy phần mới hoặc đổi.

## Các quyết định trợ lý tự đưa ra (chủ dự án kiểm tra khi đọc)

- Bài chưa có kết quả PhoBERT thì chưa hiện, thay vì hiện kèm "đang chờ phân tích".
- Mọi phần tin trên trang cập nhật cùng lúc khi có bài mới, không chỉ danh sách trong tab.
- Bài mới chỉ chen vào danh sách nếu nằm trong khoảng ngày đang chọn.
