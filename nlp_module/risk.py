"""Event-risk keyword flags over a passage. Stdlib only; order of FLAGS is the order of the output.

ponytail: plain regexes over Vietnamese legal/financial phrases, not a classifier. A flag means "this passage
mentions the event type", never that the company is the subject; widen/tune only against labelled data.
"""
import re
import unicodedata

_F = re.I


def _p(s):
    return re.compile(r"\b(?:" + s + r")\b", _F)


FLAGS = {
    # guaranteeing someone's debt; underwriting ("bảo lãnh phát hành") is routine and excluded
    "guarantee": re.compile(r"\b(?<!nhận\s)bảo\s+lãnh\b(?!\s+phát\s+hành)", _F),
    "seizure": _p(r"kê\s+biên|phong\s+(?:tỏa|toả)|phát\s+mại|siết\s+nợ"),
    "prosecution": _p(r"khởi\s+tố|truy\s+tố|(?:bắt\s+)?tạm\s+giam|bị\s+bắt|điều\s+tra\s+hình\s+sự"),
    "sanction": re.compile(r"\b(?:xử\s+phạt|bị\s+phạt|đình\s+chỉ\s+giao\s+dịch|hạn\s+chế\s+giao\s+dịch)\b|\bphạt\s+\d", _F),
    "bond_default": re.compile(
        r"\b(?:chậm|trễ|không)\s+(?:thanh\s+toán|trả)\b[^.;]{0,30}trái\s+phiếu"
        r"|\btrái\s+phiếu\b[^.;]{0,30}(?:quá\s+hạn|vỡ\s+nợ|(?:chậm|trễ)\s+(?:thanh\s+toán|trả))", _F),
}


def risk_flags(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text or "")
    return [k for k, p in FLAGS.items() if p.search(text)]
