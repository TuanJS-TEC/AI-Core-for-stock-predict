"""Bootstrap theo khối trên trục ngày (dùng bởi ai_core.direction)."""
import numpy as np


def block_bootstrap_dates(dates: np.ndarray, block: int, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Moving block bootstrap trên trục ngày (vòng tròn). Mỗi lần lấy ceil(n/block) khối liền nhau dài `block`
    rồi cắt còn n ngày. Trả về danh sách mảng chỉ số vào `dates` (không sắp xếp, có thể trùng).
    Giữ được tương quan theo thời gian trong khối và tương quan giữa các mã cùng ngày (mọi mã dùng chung ngày)."""
    n = len(dates)
    k = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=k)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        out.append(idx[:n])
    return out
