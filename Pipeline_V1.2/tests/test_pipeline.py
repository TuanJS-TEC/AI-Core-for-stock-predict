"""Kiểm thử pipeline gộp 15 mã với embedding mã: chia dữ liệu, chống rò rỉ, mô hình, metric, huấn luyện, báo cáo."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import config as C, metrics as M, pipeline as P, predict as PR, report
from src.data import Panel, assign_labels, build_panel, build_pooled, load_raw, make_loader, split_sizes
from src.models import DLinear, EmbedClsHead, MovingAvg, NLinear, PooledMultiTask

SRC = Path(C.ROOT) / "src"
N = 1447
TICKERS = ["AAA", "BBB", "CCC"]


def offset_panel() -> Panel:
    """Mã k có log close = k·10 + bước ngẫu nhiên riêng, nên mỗi mã nằm trong một dải giá trị tách biệt."""
    rng = np.random.default_rng(C.SEED)
    dates = pd.bdate_range("2021-01-04", periods=N)
    return Panel(dates, dates.to_numpy(),
                 {tk: k * 10 + np.log(50) + np.cumsum(rng.normal(0, 0.002, N)) for k, tk in enumerate(TICKERS)})


@pytest.fixture(scope="module")
def panel():
    return build_panel(C.TICKERS, load_raw())


# ---------------------------------------------------------------------------------------------------------------
# Dữ liệu: chia theo thời gian, cửa sổ, scaler, nhãn
# ---------------------------------------------------------------------------------------------------------------
def test_split_sizes_1447():
    # 1447 phiên: 70:15:15 theo thời gian cho 1012 / 217 / 218
    assert split_sizes(1447) == (1012, 217, 218)


@pytest.mark.parametrize("L", [30, 90])
@pytest.mark.parametrize("T", [2, 5])
def test_windows_one_ticker_each_and_no_leakage(L, T):
    panel = offset_panel()
    pdata = build_pooled(L, T, TICKERS, panel)
    n_train, n_val, n_test = pdata.sizes
    assert pdata.sizes == split_sizes(N)
    for k, tk in enumerate(TICKERS):
        series, s = panel.log_close[tk], pdata.scalers[tk]
        tr, va, te = pdata.train.of(k), pdata.val.of(k), pdata.test.of(k)
        assert (tr.sum(), va.sum(), te.sum()) == (n_train - L - T + 1, n_val - T + 1, n_test - T + 1)
        # mục tiêu train nằm trong train, mục tiêu val/test nằm sau train/val
        assert (pdata.train.end_idx[tr] + T).max() < n_train
        assert (pdata.val.end_idx[va] + 1).min() == n_train
        assert (pdata.test.end_idx[te] + 1).min() == n_train + n_val
        # cửa sổ là L dòng liền nhau của chính mã đó, kết thúc tại end_idx; đầu vào val là L điểm cuối của train
        inv = lambda a: a.astype(float) * s.scale_[0] + s.mean_[0]
        np.testing.assert_allclose(inv(pdata.val.x[va][0, 0]), series[n_train - L:n_train], atol=1e-4)
        np.testing.assert_allclose(inv(pdata.test.y[te][0]), series[n_train + n_val:n_train + n_val + T], atol=1e-4)
        np.testing.assert_allclose(inv(pdata.train.y[tr][-1]), series[n_train - T:n_train], atol=1e-4)
    assert pdata.train.date.max() < pdata.val.date.min()


def test_scaler_per_ticker_fit_on_its_train_rows_only():
    panel = offset_panel()
    pdata = build_pooled(30, 2, TICKERS, panel)
    n_train = pdata.sizes[0]
    for tk in TICKERS:
        s = panel.log_close[tk]
        assert pdata.scalers[tk].mean_[0] == pytest.approx(s[:n_train].mean())
        assert pdata.scalers[tk].mean_[0] != pytest.approx(s.mean())


def test_theta_and_prior_from_train_only():
    pdata = build_pooled(30, 2, TICKERS, offset_panel())
    for k, tk in enumerate(TICKERS):
        m = pdata.train.of(k)
        assert pdata.theta[tk] == pytest.approx(C.K_THETA * pdata.train.r[m].std())
        np.testing.assert_allclose(pdata.prior[tk], M.class_prior(pdata.train.labels[m]))
        np.testing.assert_array_equal(pdata.test.labels[pdata.test.of(k)],
                                      assign_labels(pdata.test.r[pdata.test.of(k)], pdata.theta[tk]))


def test_labels_threshold():
    r = np.array([-0.02, -0.005, 0.0, 0.005, 0.02])
    assert assign_labels(r, 0.01).tolist() == [0, 1, 1, 1, 2]


def test_split_follows_current_data_and_test_dates(panel):
    raw_rows = len(pd.read_parquet(C.STOCK_DIR / C.STOCK_FILE.format(ticker="VNM"), columns=[C.TIME_COL]))
    assert len(panel.dates) == raw_rows                  # dữ liệu sync thường xuyên nên số dòng thật thay đổi
    for T in C.HORIZONS:
        pdata = build_pooled(30, T, C.TICKERS, panel)
        n_train, n_val, n_test = pdata.sizes
        assert pdata.sizes == split_sizes(raw_rows)
        for k in range(len(C.TICKERS)):
            m = pdata.test.of(k)
            np.testing.assert_array_equal(pdata.test.date[m], panel.stamps[n_train + n_val + T - 1:])
            assert m.sum() == n_test - T + 1


def test_only_train_loader_shuffles():
    pdata = build_pooled(30, 2, TICKERS, offset_panel())
    assert isinstance(make_loader(pdata.train, shuffle=True, seed=C.SEED).sampler, torch.utils.data.RandomSampler)
    va = [xb for xb, *_ in make_loader(pdata.val, shuffle=False, seed=C.SEED)]
    vb = [xb for xb, *_ in make_loader(pdata.val, shuffle=False, seed=C.SEED + 1)]
    assert all(torch.equal(i, j) for i, j in zip(va, vb))
    assert isinstance(make_loader(pdata.val, shuffle=False, seed=C.SEED).sampler, torch.utils.data.SequentialSampler)


# ---------------------------------------------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------------------------------------------
def test_nlinear_adds_back_last_value():
    m = NLinear(30, 2)
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(4, 30)
    assert torch.allclose(m(x), x[:, -1:].repeat(1, 2))


def test_dlinear_moving_average_keeps_length_and_kernel_25():
    assert C.MA_KERNEL == 25
    x = torch.randn(3, 90)
    ma = MovingAvg(25)
    assert ma(x).shape == x.shape
    assert DLinear(30, 5)(torch.randn(3, 30)).shape == (3, 5)
    c = torch.full((1, 40), 3.0)                         # chuỗi hằng giữ nguyên nhờ padding biên
    assert torch.allclose(ma(c), c)


@pytest.mark.parametrize("name", C.MODELS)
def test_heads_share_no_params_and_shapes(name):
    m = PooledMultiTask(name, 30, 2, 1, 15)
    assert {id(p) for p in m.reg.parameters()}.isdisjoint({id(p) for p in m.cls.parameters()})
    r, z = m(torch.randn(8, 1, 30), torch.arange(8))
    assert r.shape == (8, 2) and z.shape == (8, C.N_CLASSES)


def test_nlinear_reg_adds_back_last_value_in_pooled_model():
    m = PooledMultiTask("NLinear", 30, 2, 1, 15)
    x = torch.randn(4, 1, 30)
    tid = torch.zeros(4, dtype=torch.long)
    torch.testing.assert_close(m(x + 5.0, tid)[0] - m(x, tid)[0], torch.full((4, 2), 5.0))


@pytest.mark.parametrize("name", C.MODELS)
def test_ticker_embedding_changes_logits(name):
    torch.manual_seed(C.SEED)
    h = EmbedClsHead(name, 30, 1, 15).eval()
    x = torch.randn(1, 1, 30).repeat(2, 1, 1)
    z = h(x, torch.tensor([0, 1]))
    assert not torch.allclose(z[0], z[1])                # cùng cửa sổ, khác mã -> khác logits


def test_nlinear_cls_head_keeps_level():
    """NLinear trừ giá trị cuối nhưng vẫn đưa giá trị cuối vào tầng cuối, nên mức hiện tại không bị mất."""
    torch.manual_seed(C.SEED)
    h = EmbedClsHead("NLinear", 30, 1, 15).eval()
    x = torch.randn(4, 1, 30)
    tid = torch.zeros(4, dtype=torch.long)
    assert not torch.allclose(h(x, tid), h(x + 1.0, tid))


# ---------------------------------------------------------------------------------------------------------------
# Metric và hiệu chỉnh
# ---------------------------------------------------------------------------------------------------------------
def test_metrics_confusion_and_f1():
    y = np.array([0, 1, 2, 2, 1, 0])
    p = np.array([0, 1, 2, 1, 1, 2])
    cm = M.confusion(y, p)
    assert cm.tolist() == [[1, 0, 1], [0, 2, 0], [0, 1, 1]]
    out = M.classification(cm)
    assert out["accuracy"] == pytest.approx(4 / 6)
    assert out["f1_down"] == pytest.approx(2 * 1 * 0.5 / 1.5)


def test_brier_perfect_uniform_and_confident_wrong():
    y = np.array([0, 1, 2])
    assert M.brier_score(np.eye(3), y) == pytest.approx(0.0)
    assert M.brier_score(np.full((3, 3), 1 / 3), y) == pytest.approx(2 / 3)
    assert M.brier_score(np.eye(3)[[1, 2, 0]], y) == pytest.approx(2.0)


def test_hit_rate_on_hand_built_matrix_and_nan():
    # Hàng = thật, cột = dự đoán. Dự đoán Giảm: 4 + 1 + 5 = 10 (đúng 4). Dự đoán Tăng: 2 + 3 + 6 = 11 (đúng 6).
    cm = np.array([[4, 7, 2], [1, 9, 3], [5, 8, 6]])
    assert M.hit_rate(cm) == pytest.approx((4 + 6) / (10 + 11))
    assert np.isnan(M.hit_rate(np.array([[0, 5, 0], [0, 7, 0], [0, 3, 0]])))


def test_kappa_perfect_and_independent():
    assert M.cohen_kappa(np.diag([5, 5, 5])) == pytest.approx(1.0)
    assert M.cohen_kappa(np.full((3, 3), 4)) == pytest.approx(0.0)


def test_fit_temperature_minimizes_val_nll_and_keeps_argmax():
    rng = np.random.default_rng(C.SEED)
    y = rng.integers(0, 3, 600)
    logits = 5.0 * (np.eye(3)[y] * 0.3 + rng.standard_normal((600, 3)))   # quá tự tin so với tín hiệu
    t = M.fit_temperature(logits, y)
    assert t > 1.0
    assert M.nll(logits, y, 1.0 / t) <= min(M.nll(logits, y, 1.0), M.nll(logits, y, 1.0 / (t * 1.2)))
    assert (M.softmax(logits / t).argmax(axis=1) == logits.argmax(axis=1)).all()


# ---------------------------------------------------------------------------------------------------------------
# Huấn luyện, chạy tiếp và báo cáo
# ---------------------------------------------------------------------------------------------------------------
def test_run_resumes_and_writes_report_with_repeat_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MAX_EPOCHS", 2)
    panel = offset_panel()
    kw = dict(out_dir=tmp_path, horizons=(2,), lookbacks=(30,), models=("NLinear",), wds=(0.0,),
              seeds=C.SEEDS[:2], tickers=TICKERS, panel=panel, log=lambda *_: None)
    runs, preds, _ = P.run_experiment(**kw)
    assert len(runs) == 2 * len(TICKERS) and len(preds) == 2 * len(TICKERS) * 217
    np.testing.assert_allclose(preds[[*C.PROB_COLS]].sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_array_equal(preds[[*C.SCORE_COLS]].to_numpy().argmax(axis=1), preds["y_pred"])
    logs = []
    runs2, _, _ = P.run_experiment(**{**kw, "log": logs.append})
    assert "còn 0/2 lượt" in logs[0]
    pd.testing.assert_frame_equal(runs, runs2)

    table = P.finalize(tmp_path, tmp_path / "fig", runs, preds, TICKERS, panel)
    assert set(table["method"]) == {"model", "ensemble", "repeat", "class_freq"}
    assert (tmp_path / C.FINAL_TABLE_FILE).exists() and (tmp_path / "fig" / C.CONFUSION_FIG_FILE.format(T=2)).exists()
    cfg = pd.read_csv(tmp_path / C.FINAL_CONFIG_FILE)
    assert cfg["val_f1_macro_mean_over_tickers"].iloc[0] == pytest.approx(
        runs.groupby("ticker")["val_f1_macro"].mean().mean())


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MAX_EPOCHS", 2)
    panel = offset_panel()
    run_dir = tmp_path / "run"
    runs, _, _ = P.run_experiment(run_dir, horizons=(2,), lookbacks=(30,), models=("NLinear", "Linear"), wds=(0.0,),
                                  seeds=C.SEEDS[:2], tickers=TICKERS, panel=panel, log=lambda *_: None)
    return run_dir, tmp_path / "models", runs, panel


def test_models_saved_per_ticker_are_best_on_val_and_rescore_exactly(small_run):
    run_dir, models_dir, runs, panel = small_run
    hist = PR.update_models(run_dir, panel, TICKERS, models_dir, log=lambda *_: None)
    assert list(hist["action"]) == ["new"] * len(TICKERS)
    assert len(list(models_dir.glob("*.pt"))) == len(TICKERS)
    pdata = build_pooled(30, 2, TICKERS, panel)
    for t in TICKERS:
        a = PR.load_artifact(PR.model_path(t, 2, 30, models_dir))
        r = runs[runs["ticker"] == t]
        assert a["metrics"]["val_f1_macro"] == pytest.approx(r["val_f1_macro"].max())         # tốt nhất trên val của chính mã
        best = r[(r["model"] == a["model"]) & (r["seed"] == a["seed"])].iloc[0]
        row = PR.rescore(a, pdata, panel)                                        # nạp lại từ file cho đúng chỉ số
        for k in ("val_f1_macro", "val_kappa", "f1_macro", "kappa", "temperature", "alpha"):
            assert row[k] == pytest.approx(best[k]), k
    reg = pd.read_csv(models_dir / C.MODEL_REGISTRY_FILE)
    assert len(reg) == len(TICKERS)


def test_update_keeps_better_saved_model_and_replaces_worse(small_run, monkeypatch):
    run_dir, models_dir, runs, panel = small_run
    quiet = dict(tickers=TICKERS, models_dir=models_dir, log=lambda *_: None)
    PR.update_models(run_dir, panel, **quiet)
    assert set(PR.update_models(run_dir, panel, **quiet)["action"]) == {"kept"}     # cùng dữ liệu: hòa -> giữ
    real = PR.rescore
    monkeypatch.setattr(PR, "rescore", lambda *a: {**real(*a), C.MODEL_SELECT_METRIC: 2.0})
    assert set(PR.update_models(run_dir, panel, **quiet)["action"]) == {"kept"}
    monkeypatch.setattr(PR, "rescore", lambda *a: {**real(*a), C.MODEL_SELECT_METRIC: -1.0})
    assert set(PR.update_models(run_dir, panel, **quiet)["action"]) == {"replaced"}
    monkeypatch.setattr(C, "K_THETA", 0.25)
    assert set(PR.update_models(run_dir, panel, **quiet)["action"]) == {"replaced (khác K_THETA/kiến trúc)"}
    hist = pd.read_csv(models_dir / C.MODEL_HISTORY_FILE)
    assert len(hist) == 5 * len(TICKERS)


def test_predict_gives_probabilities_and_label(small_run, capsys):
    run_dir, models_dir, _, panel = small_run
    PR.update_models(run_dir, panel, TICKERS, models_dir, log=lambda *_: None)
    last = panel.dates[-1]                                          # thứ Hai (bdate_range), dữ liệu không có ngày nghỉ
    now = last + pd.Timedelta(hours=18)                             # 18:00 cùng ngày: dữ liệu đã mới nhất
    table = PR.predict(TICKERS, panel, models_dir, now=now)
    assert len(table) == len(TICKERS) and set(table["label"]) <= set(C.LABEL_NAMES)
    np.testing.assert_allclose(table[[*C.PROB_COLS]].sum(axis=1), 1.0, atol=1e-3)
    assert (table["up_above_pct"] > 0).all() and (table["down_below_pct"] < 0).all()
    assert (table["last_date"] == str(last.date())).all()
    assert (table["target_date"] == str((last + pd.offsets.BDay(2)).date())).all()   # T = 2
    assert (table["predicted_at"] == now.isoformat(timespec="seconds")).all() and (table["target_note"] == "").all()
    later = PR.predict(TICKERS, panel, models_dir, now=last + pd.Timedelta(days=7))
    assert (later["target_note"] == "đã qua").all()


def test_store_treats_new_dataset_id_as_stale(tmp_path):
    spec = {"n_rows": 10, "last_date": "2026-01-02", "dataset_id": "aaa"}
    first = P.Store(tmp_path, spec, force=False, log=lambda *_: None)
    first.runs.append({"T": 2, "L": 30, "model": "Linear", "wd": 0.0, "seed": 1})
    first.preds.append(pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "y_true": [1]}))
    first.save()
    same = P.Store(tmp_path, spec, force=False, log=lambda *_: None)
    assert len(same.runs) == 1                                      # cùng dataset_id: chạy tiếp
    logs = []
    again = P.Store(tmp_path, {**spec, "dataset_id": "bbb"}, force=False, log=logs.append)
    assert any("chạy lại từ đầu" in m and "dataset_id" in m for m in logs) and again.runs == []


def test_run_experiment_spec_has_dataset_id(small_run):
    run_dir, _, _, panel = small_run
    spec = json.loads((run_dir / C.EXPERIMENT_FILE).read_text(encoding="utf-8"))
    assert spec["dataset_id"] == panel.dataset_id


def test_predict_data_invalid_when_session_missing(small_run):
    run_dir, models_dir, _, panel = small_run
    PR.update_models(run_dir, panel, TICKERS, models_dir, log=lambda *_: None)
    now = panel.dates[-1] + pd.offsets.BDay(3) + pd.Timedelta(hours=18)        # lỡ 3 phiên
    table = PR.predict(TICKERS, panel, models_dir, now=now)
    assert len(table) == len(TICKERS)
    assert (table["status"] == "DATA_INVALID").all() and (table["reason"] == "MISSING_SESSION").all()
    assert table[[*C.PROB_COLS]].isna().all().all() and (table["label"] == "DATA_INVALID").all()
    assert (table["target_note"] == "đã qua").all()
    ok = PR.predict(TICKERS, panel, models_dir, now=panel.dates[-1] + pd.Timedelta(hours=18))
    assert (ok["status"] == "OK").all() and (ok["reason"] == "").all()


def test_predict_data_invalid_when_gap_inside_window(small_run):
    run_dir, models_dir, _, panel = small_run
    PR.update_models(run_dir, panel, TICKERS, models_dir, log=lambda *_: None)
    now = panel.dates[-1] + pd.Timedelta(hours=18)
    hol = PR.market_holidays(panel)
    assert PR.invalid_reason(panel, 30, now, hol) is None           # gaps None: bỏ qua
    panel.gaps = pd.DatetimeIndex([panel.dates[-10]])               # trong cửa sổ L = 30
    assert PR.invalid_reason(panel, 30, now, hol) == "MISSING_SESSION"
    table = PR.predict(TICKERS, panel, models_dir, now=now)
    assert (table["L"] == 30).all() and (table["status"] == "DATA_INVALID").all()
    assert (table["reason"] == "MISSING_SESSION").all() and table[[*C.PROB_COLS]].isna().all().all()
    panel.gaps = pd.DatetimeIndex([panel.dates[-31]])               # ngoài cửa sổ: vẫn hợp lệ
    assert PR.invalid_reason(panel, 30, now, hol) is None
    assert (PR.predict(TICKERS, panel, models_dir, now=now)["status"] == "OK").all()


def test_target_date_skips_weekends_and_holidays(panel):
    hol = PR.market_holidays(panel)
    d = panel.dates
    for T in C.HORIZONS:                                            # trùng phiên thật của dữ liệu, kể cả qua Tết
        got = np.busday_offset(d[:-T].values.astype("datetime64[D]"), T, weekmask=PR.WEEKMASK, holidays=hol)
        np.testing.assert_array_equal(got, d[T:].values.astype("datetime64[D]"))
    assert PR.session_after("2026-04-29", 2, hol) == pd.Timestamp("2026-05-05")    # nghỉ 30/4, 1/5
    assert PR.session_after("2026-10-23", 2, hol) == pd.Timestamp("2026-10-27")    # qua cuối tuần
    assert PR.session_after("2026-12-31", 2, hol) == pd.Timestamp("2027-01-05")    # nghỉ 1/1 khai báo trong config
    fri = "2026-10-23"
    assert PR.missing_sessions(fri, pd.Timestamp("2026-10-25 10:00"), hol) == []           # Chủ Nhật
    assert PR.missing_sessions(fri, pd.Timestamp("2026-10-26 10:00"), hol) == []           # thứ Hai, phiên chưa xong
    assert PR.missing_sessions(fri, pd.Timestamp("2026-10-26 16:00"), hol) == [pd.Timestamp("2026-10-26")]
    assert PR.missing_sessions("2026-04-29", pd.Timestamp("2026-05-02 09:00"), hol) == []  # 30/4, 1/5 là ngày nghỉ


def test_plot_models_val_test_only_and_mse_matches_run(small_run, tmp_path):
    run_dir, models_dir, runs, panel = small_run
    PR.update_models(run_dir, panel, TICKERS, models_dir, log=lambda *_: None)
    out = tmp_path / "reg.csv"
    table = report.plot_models(panel, models_dir, tmp_path / "figs", out, log=lambda *_: None)
    assert out.exists() and len(table) == len(TICKERS) * 2 * 2          # (val, test) x (model, repeat)
    pdata = build_pooled(30, 2, TICKERS, panel)
    n_train = pdata.sizes[0]
    assert (pd.to_datetime(table["date_from"]) > panel.dates[n_train - 1]).all()     # không vẽ train
    for k, t in enumerate(TICKERS):
        assert (tmp_path / "figs" / C.MODEL_FIG_FILE.format(ticker=t, T=2, L=30)).exists()
        a = PR.load_artifact(PR.model_path(t, 2, 30, models_dir))
        best = runs[(runs["ticker"] == t) & (runs["model"] == a["model"]) & (runs["wd"] == a["wd"])
                    & (runs["seed"] == a["seed"])].iloc[0]
        r = table[(table["ticker"] == t) & (table["split"] == "test")].set_index("method")
        assert r.loc["model", "mse_scaled"] == pytest.approx(best["mse_scaled"], rel=1e-4)
        assert r.loc["model", "mae_scaled"] == pytest.approx(best["mae_scaled"], rel=1e-4)
        m = pdata.test.of(k)
        rep = np.mean((pdata.test.x[m, 0, -1:] - pdata.test.y[m]) ** 2)       # repeat: lặp giá trị cuối
        assert r.loc["repeat", "mse_scaled"] == pytest.approx(rep, rel=1e-4)
        assert r.loc["model", "n"] == m.sum()


def test_rerun_same_seed_is_deterministic(monkeypatch):
    monkeypatch.setattr(C, "MAX_EPOCHS", 2)
    pdata = build_pooled(30, 2, TICKERS, offset_panel())
    a = P.fit(pdata, "DLinear", C.SEED, 0.0)
    b = P.fit(pdata, "DLinear", C.SEED, 0.0)
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k])


def test_seed_declared_once_and_derived():
    assert C.SEED == 59
    assert C.SEEDS == [59, 60, 61, 62, 63]
    for f in SRC.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"\b42\b", text), f
        if f.name != "config.py":
            assert not re.search(r"^\s*SEED\s*=", text, re.M), f
