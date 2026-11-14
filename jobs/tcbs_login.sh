#!/usr/bin/env bash
# Hỏi OTP TCBS khi phiên thiếu hoặc còn dưới MARGIN_H giờ; còn đủ hạn thì im lặng.
#   tcbs_login.sh           hỏi OTP nếu cần (chỉ khi có terminal; tối đa 3 lần nhập)
#   tcbs_login.sh --check   không hỏi gì: in số giờ còn lại, thoát 1 nếu thiếu/dưới ngưỡng (dùng cho cron)
# Gắn vào ~/.bashrc:  [[ $- == *i* ]] && /home/administrator/TLS/tuanhm@linux/AI_Core/jobs/tcbs_login.sh
MARGIN_H=${MARGIN_H:-6}
export VNSTOCK_OAPI_SESSION_BACKEND=file    # keyring bị khoá thì vnstock treo; phiên lưu file 600
f="${VNSTOCK_OAPI_STATE_DIR:-$HOME/.local/state/vnstock-oapi}/sessions/tcbs-default.json"

hours_left() {
  python3 -c "import json,sys,time; print((json.load(open(sys.argv[1]))['expires_at']-time.time())/3600)" "$f" 2>/dev/null || echo 0
}
enough() { python3 -c "import sys; sys.exit(not float(sys.argv[1]) >= float(sys.argv[2]))" "$1" "$MARGIN_H"; }

left=$(hours_left)
if [[ $1 == --check ]]; then
  printf 'tcbs: còn %.1f giờ (ngưỡng %s)\n' "$left" "$MARGIN_H"
  enough "$left"; exit
fi
enough "$left" && exit 0
[[ -t 0 && -t 1 ]] || exit 0

printf 'Phiên TCBS %s. Nhập OTP (Ctrl+C để bỏ qua).\n' "$([[ $left == 0 ]] && echo 'chưa có hoặc đã hết hạn' || printf 'còn %.1f giờ' "$left")"
cd "$(dirname "$0")/.." || exit 1   # vnstock chỉ đọc .env ở thư mục hiện tại; không thấy TCBS_API_KEY thì quay sang keyring (khoá → treo)
for _ in 1 2 3; do
  "${VENV_6X:-$HOME/.venv_6x}/bin/vnstock" auth tcbs && exit 0
done
echo 'Không mở được phiên TCBS; chạy lại jobs/tcbs_login.sh khi cần.' >&2
