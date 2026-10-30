"""Metric hồi quy (thang chuẩn hóa), metric phân loại 3 nhãn và hiệu chỉnh xác suất.

Hiệu chỉnh và quy tắc quyết định nhãn: mọi tham số chỉ fit trên val, không đọc test.
  T       nhiệt độ: p = softmax(logits / T), cực tiểu NLL trên val
  alpha   hiệu chỉnh prior khi chọn nhãn: argmax(p_c / prior_c^alpha), alpha chọn theo Macro-F1 val
  lambda  co về prior: p' = lambda * p + (1 - lambda) * prior, cực tiểu Brier trên val (nghiệm đóng)
prior là tần suất lớp ở train. Nhãn dùng p (chưa co), Brier dùng p' (đã co).
"""
import numpy as np

from . import config as C

N_CLASS = C.N_CLASSES


def regression_scaled(pred: np.ndarray, y: np.ndarray) -> dict:
    """MSE và MAE trên thang chuẩn hóa (log giá sau StandardScaler)."""
    err = pred - y
    return {"mse_scaled": float(np.mean(err ** 2)), "mae_scaled": float(np.mean(np.abs(err)))}


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Hàng = nhãn thật, cột = nhãn dự đoán. Thứ tự: Giảm, Đi ngang, Tăng."""
    cm = np.zeros((N_CLASS, N_CLASS), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    prec_den = cm.sum(axis=0)
    rec_den = cm.sum(axis=1)
    prec = np.divide(tp, prec_den, out=np.zeros_like(tp), where=prec_den > 0)
    rec = np.divide(tp, rec_den, out=np.zeros_like(tp), where=rec_den > 0)
    s = prec + rec
    return np.divide(2 * prec * rec, s, out=np.zeros_like(tp), where=s > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_per_class(confusion(y_true, y_pred)).mean())


def classification(cm: np.ndarray) -> dict:
    """Accuracy, macro-F1, weighted-F1 và F1 từng lớp từ confusion matrix."""
    total = cm.sum()
    f1 = f1_per_class(cm)
    support = cm.sum(axis=1)
    return {
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "f1_macro": float(f1.mean()),
        "f1_weighted": float((f1 * support).sum() / total) if total else 0.0,
        "f1_down": float(f1[C.CLASS_DOWN]),
        "f1_flat": float(f1[C.CLASS_FLAT]),
        "f1_up": float(f1[C.CLASS_UP]),
    }


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Brier đa lớp: trung bình theo mẫu của Σ_c (p_c − 1[y=c])². Khoảng 0 đến 2, thấp hơn là tốt hơn."""
    onehot = np.eye(N_CLASS)[np.asarray(y, dtype=np.int64)]
    return float(np.mean(np.sum((np.asarray(probs, dtype=float) - onehot) ** 2, axis=1)))


def hit_rate(cm: np.ndarray) -> float:
    """Trong các mẫu bị dự đoán Giảm hoặc Tăng, tỷ lệ mẫu có nhãn thật trùng nhãn dự đoán.

    cm có hàng = nhãn thật, cột = nhãn dự đoán. NaN nếu không bao giờ dự đoán Giảm hoặc Tăng.
    """
    cm = np.asarray(cm)
    d, u = C.CLASS_DOWN, C.CLASS_UP
    n_pred = cm[:, d].sum() + cm[:, u].sum()
    return float((cm[d, d] + cm[u, u]) / n_pred) if n_pred > 0 else float("nan")


def cohen_kappa(cm: np.ndarray) -> float:
    """Cohen's kappa: mức khớp vượt quá mức ngẫu nhiên khi giữ nguyên phân bố nhãn thật và nhãn dự đoán.

    0 là ngang với dự đoán độc lập với nhãn thật, 1 là khớp hoàn toàn. 0 nếu cm rỗng hoặc chỉ có một lớp.
    """
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=1) * cm.sum(axis=0)).sum() / n ** 2)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def permutation_f1(y_true: np.ndarray, y_pred: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Macro-F1 kỳ vọng khi hoán vị ngẫu nhiên nhãn dự đoán: giữ nguyên tỷ lệ từng nhãn dự đoán nhưng cắt
    mọi liên hệ với nhãn thật. Mức F1 mà một bộ dự đoán không có tín hiệu với cùng tỷ lệ nhãn đạt được."""
    y_pred = np.asarray(y_pred)
    return float(np.mean([macro_f1(y_true, rng.permutation(y_pred)) for _ in range(n_perm)]))


def block_bootstrap_dates(dates: np.ndarray, block: int, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Moving block bootstrap trên trục ngày (vòng tròn). Mỗi lần lấy ceil(n/block) khối liền nhau dài `block`
    rồi cắt còn n ngày. Trả về danh sách mảng chỉ số vào `dates` (đã sắp xếp, không trùng).
    Giữ được tương quan theo thời gian trong khối và tương quan giữa các mã cùng ngày (mọi mã dùng chung ngày)."""
    n = len(dates)
    k = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=k)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        out.append(idx[:n])
    return out


# ---------------------------------------------------------------------------------------------------------------
# Hiệu chỉnh xác suất
# ---------------------------------------------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, beta: float = 1.0) -> float:
    """Negative log-likelihood trung bình của softmax(beta · logits), beta = 1/T."""
    z = beta * logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-np.mean(log_p[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lo: float = C.TEMP_BETA_RANGE[0],
                    hi: float = C.TEMP_BETA_RANGE[1]) -> float:
    """Nhiệt độ T > 0 cực tiểu NLL của softmax(logits / T). Chỉ dùng logits và nhãn của val.

    NLL lồi theo beta = 1/T nên dùng tìm kiếm chia ba trên beta trong [lo, hi] (xác định, không cần scipy).
    """
    for _ in range(C.TEMP_SEARCH_ITERS):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(logits, labels, m1) < nll(logits, labels, m2):
            hi = m2
        else:
            lo = m1
    return float(1.0 / ((lo + hi) / 2))


def class_prior(labels: np.ndarray) -> np.ndarray:
    """Tần suất các lớp của một vector nhãn 1 chiều (dùng nhãn train)."""
    return np.bincount(labels, minlength=C.N_CLASSES) / len(labels)


def fit_shrink(p: np.ndarray, labels: np.ndarray, prior: np.ndarray) -> float:
    """lambda trong [0, 1] cực tiểu Brier val của lambda · p + (1 - lambda) · prior.

    Brier là hàm bậc hai của lambda nên có nghiệm đóng: lambda = <onehot - prior, p - prior> / |p - prior|^2.
    """
    d = p - prior
    den = float((d ** 2).sum())
    if den == 0.0:
        return 1.0
    return float(min(1.0, max(0.0, ((np.eye(C.N_CLASSES)[labels] - prior) * d).sum() / den)))


def shrink_probs(p: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    return lam * p + (1.0 - lam) * prior


def adjusted_scores(p: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    """p_c / prior_c^alpha, chuẩn hóa về tổng 1. alpha = 0 giữ nguyên p."""
    s = p / np.maximum(prior, C.PRIOR_EPS) ** alpha
    return s / s.sum(axis=1, keepdims=True)


def fit_alpha(p: np.ndarray, labels: np.ndarray, prior: np.ndarray, grid) -> float:
    """alpha trong grid cho Macro-F1 val cao nhất. Hòa thì lấy alpha nhỏ nhất."""
    f1 = [macro_f1(labels, adjusted_scores(p, prior, a).argmax(axis=1)) for a in grid]
    return float(grid[int(np.argmax(f1))])


def apply_calibration(logits: np.ndarray, temperature: float, prior: np.ndarray, alpha: float,
                      lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xác suất đã co về prior, xác suất sau chia nhiệt độ, điểm quyết định). Nhãn = argmax điểm quyết định."""
    p = softmax(logits / temperature)
    return shrink_probs(p, prior, lam), p, adjusted_scores(p, prior, alpha)


def ticker_metrics(z_val: np.ndarray, y_val: np.ndarray, z_test: np.ndarray, y_test: np.ndarray,
                   prior: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Hiệu chỉnh (T, alpha, lambda) trên val của một mã rồi tính chỉ số val và test (test chỉ để ghi nhận).

    Trả về (dòng chỉ số kèm tham số hiệu chỉnh, xác suất test đã co, điểm quyết định test).
    """
    temp = fit_temperature(z_val, y_val)
    p = softmax(z_val / temp)
    alpha = fit_alpha(p, y_val, prior, C.ALPHA_GRID)
    lam = fit_shrink(p, y_val, prior)
    pv, _, sv = apply_calibration(z_val, temp, prior, alpha, lam)
    pt, _, st = apply_calibration(z_test, temp, prior, alpha, lam)
    lv, lt = sv.argmax(axis=1), st.argmax(axis=1)
    cm_v, cm_t = confusion(y_val, lv), confusion(y_test, lt)
    brier_v = brier_score(pv, y_val)
    brier_v0 = brier_score(np.tile(prior, (len(y_val), 1)), y_val)
    row = {"temperature": temp, "alpha": alpha, "shrink": lam,
           "val_nll_raw": nll(z_val, y_val),
           "val_f1_macro": float(f1_per_class(cm_v).mean()),
           "val_f1_macro_raw": macro_f1(y_val, z_val.argmax(axis=1)),
           "val_f1_flat": float(f1_per_class(cm_v)[C.CLASS_FLAT]),
           "val_kappa": cohen_kappa(cm_v), "val_accuracy": float(np.mean(lv == y_val)),
           "val_brier": brier_v, "val_brier_skill": 1.0 - brier_v / brier_v0}
    row.update(classification(cm_t))
    row.update({"kappa": cohen_kappa(cm_t), "brier": brier_score(pt, y_test)})
    row.update({f"cm_{i}{j}": int(cm_t[i, j]) for i in range(N_CLASS) for j in range(N_CLASS)})
    return row, pt, st
