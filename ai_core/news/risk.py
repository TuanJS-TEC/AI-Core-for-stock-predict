"""Event-risk keyword flags over a passage. Stdlib only; order of FLAGS is the order of the output.

ponytail: plain regexes over Vietnamese legal/financial phrases, not a classifier. A flag means "this passage
mentions the event type", never that the company is the subject; widen/tune only against labelled data.
"""
import re
import unicodedata

_F = re.I


def _p(s):
    return re.compile(r"\b(?:" + s + r")\b", _F)


_CUE = r"(?:vay|nợ|nghĩa\s+vụ|công\s+ty\s+con|liên\s+kết|trái\s+phiếu)"
_G = r"(?<!thư\s)(?<!nhận\s)\bbảo\s+lãnh\b(?!\s+(?:phát\s+hành|thanh\s+toán|thực\s+hiện|dự\s+thầu))"

FLAGS = {
    # guaranteeing someone's debt: needs a borrower/debt cue in the same sentence; routine bank products
    # (thư bảo lãnh, bảo lãnh phát hành/thanh toán/dự thầu) carry none or are excluded
    "guarantee": re.compile(
        rf"{_G}(?=[^.;]{{0,60}}\b{_CUE}\b)|\b{_CUE}\b[^.;]{{0,40}}{_G}", _F),
    "seizure": _p(r"kê\s+biên|phong\s+(?:tỏa|toả)|phát\s+mại|siết\s+nợ"),
    "prosecution": _p(r"khởi\s+tố|truy\s+tố|(?:bắt\s+)?tạm\s+giam|bị\s+bắt(?!\s+buộc)|điều\s+tra\s+hình\s+sự"),
    "sanction": re.compile(r"\b(?:xử\s+phạt|bị\s+phạt|đình\s+chỉ\s+giao\s+dịch|hạn\s+chế\s+giao\s+dịch)\b|\bphạt\s+\d", _F),
    "bond_default": re.compile(
        r"\b(?:chậm|trễ|không)\s+(?:thanh\s+toán|trả)\b[^.;]{0,30}trái\s+phiếu"
        r"|\btrái\s+phiếu\b[^.;]{0,30}(?:quá\s+hạn|vỡ\s+nợ|(?:chậm|trễ)\s+(?:thanh\s+toán|trả))", _F),
}


def risk_flags(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text or "")
    return [k for k, p in FLAGS.items() if p.search(text)]
