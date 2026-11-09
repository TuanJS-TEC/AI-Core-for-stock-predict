import json
from types import SimpleNamespace

import numpy as np, pandas as pd, pytest
from mmf import compare as CP, config as C, evaluate as E, experiment as X, frame as FR, holdout as H
from mmf.windows import BINARY

SYMS = ["AAA", "BBB", "CCC"]
LOCK = {"5": {"baseline": "prior_fixed", "unimodal": "dl_price", "fusion": "fusion_attention", "price_kind": "gru", "dataset_id": "devdataset"}}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """C.DATA under tmp_path, so the untracked guard of a real holdout run never reaches these tests."""
    monkeypatch.setattr(C, "DATA", tmp_path / "data")
    return C.DATA


def frame_with(n_days):
    """15 symbols: 10 sessions before HOLDOUT_AFTER, n_days matured after it, then 5 not matured with no label end."""
    days = pd.bdate_range(end=C.HOLDOUT_AFTER, periods=10).append(pd.bdate_range("2026-11-02", periods=n_days + 5))
    end = pd.Series(days).shift(-5).to_numpy()
    d = pd.DataFrame({"as_of_session": days, "label_end_date": end, "status": np.where(pd.notna(end), "ok", "not_matured")})
    f = pd.concat([d.assign(symbol=s) for s in C.TICKERS], ignore_index=True)
    return f.assign(usable=f["status"] == "ok")


def test_refuses_below_minimum_sessions(tmp_path):
    with pytest.raises(ValueError, match="phiên"):
        H.check_ready(H.holdout_rows(frame_with(39)), 5, tmp_path)
    H.check_ready(H.holdout_rows(frame_with(40)), 5, tmp_path)


def test_early_flag_lowers_minimum_only_to_twenty(tmp_path):
    H.check_ready(H.holdout_rows(frame_with(20)), 5, tmp_path, early=True)
    with pytest.raises(ValueError):
        H.check_ready(H.holdout_rows(frame_with(19)), 5, tmp_path, early=True)


def test_second_run_is_refused(tmp_path):
    (tmp_path / "mmf_holdout_h5.lock").write_text("{}")
    with pytest.raises(ValueError, match="đã chấm"):
        H.check_ready(H.holdout_rows(frame_with(50)), 5, tmp_path)


def test_rows_exclude_dev_and_unmatured():
    f = frame_with(50)
    r = H.holdout_rows(f)
    assert r.as_of_session.min() > pd.Timestamp("2026-10-30")
    assert r.label_end_date.notna().all() and r.status.eq("ok").all()
    assert r.as_of_session.nunique() == f[f.status == "ok"].as_of_session.nunique() - f[f.as_of_session <= "2026-10-30"].as_of_session.nunique()


def test_refuses_without_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "LOCK_FILE", tmp_path / "nope.json")
    with pytest.raises(ValueError, match="khóa"):
        H.load_lock()


def r_table(text):
    """Rows of the R1-R3 table, cells stripped."""
    return [[c.strip() for c in line.split("|")[1:-1]] for line in text.splitlines() if line.startswith("| R")]


def full_frame():
    """3 symbols of random features over all four blocks of fold F, 5-session labels, the last 5 sessions not matured."""
    days = pd.bdate_range("2025-06-02", "2027-01-15")
    r = np.random.default_rng(0)
    cols = [c for g in FR.GROUPS.values() for c in g]
    end = pd.Series(days).shift(-5)
    parts = []
    for s in SYMS:
        d = pd.DataFrame(r.normal(size=(len(days), len(cols))), columns=cols)
        d[BINARY] = r.integers(0, 2, (len(days), len(BINARY))).astype(float)
        parts.append(d.assign(symbol=s, as_of_session=days, label_end_date=end, open=r.uniform(9, 11, len(days)), close=r.uniform(9, 11, len(days)),
                              label=r.integers(0, 3, len(days)).astype(float), status=np.where(end.notna(), "ok", "not_matured")))
    f = pd.concat(parts, ignore_index=True)
    f["label"] = f["label"].where(f["status"] == "ok")
    return f.assign(usable=f["status"] == "ok")


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """Panel, lock file, data dir and a trainer that writes random probabilities; `trained` logs (arm, seed, guard present)."""
    frame, trained = full_frame(), []
    guard = tmp_path / "mmf_holdout_h5.lock"
    def fake_arm(b, arm, seed, kind, path, device):
        trained.append((arm, seed, guard.exists()))
        r = np.random.default_rng([seed, len(arm)])
        X._write_preds(path, b, arm, seed, kind, {k: r.dirichlet(np.ones(3), len(b.samples[k])) for k in X.OUT_BLOCKS}, 0.0)
    monkeypatch.setattr(X, "build_frame", lambda panel, h, version: frame.copy())
    monkeypatch.setattr(X, "run_arm", fake_arm)
    (tmp_path / "lock.json").write_text(json.dumps(LOCK))
    monkeypatch.setattr(H, "LOCK_FILE", tmp_path / "lock.json")
    return SimpleNamespace(frame=frame, trained=trained, guard=guard, panel=SimpleNamespace(dataset_id="d" * 64, ohlcv=dict.fromkeys(SYMS)))


def test_run_guards_before_scoring_and_runs_once(tmp_path, stubbed, monkeypatch):
    seen, real = [], CP.load_h
    monkeypatch.setattr(CP, "load_h", lambda *a: (seen.append(stubbed.guard.exists()), real(*a))[1])
    out = H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu")
    rows = H.holdout_rows(stubbed.frame)
    assert seen == [True] and not any(g for *_, g in stubbed.trained)                  # trained before the guard, scored after it
    assert {(a, s) for a, s, _ in stubbed.trained} == {(a, s) for a in {"fusion_attention", "fusion_concat", "dl_price", "dl_fund", "dl_text", *CP.ABLATIONS} for s in C.SEEDS}
    body = json.loads(stubbed.guard.read_text())
    assert (body["n_sessions"], body["early"], body["dataset_id"]) == (rows.as_of_session.nunique(), False, "d" * 64) and body["git_commit"] and body["created_at"]
    d = pd.read_parquet(C.DATA / "holdout" / ("d" * 16) / "h5/F/prior_fixed/seed0.parquet")
    ev = d[d["block"].astype(str) == "eval"]
    assert pd.to_datetime(ev["as_of_session"]).min() > pd.Timestamp(C.HOLDOUT_AFTER) and len(ev) <= len(rows)
    assert ev["as_of_session"].nunique() == rows.as_of_session.nunique()               # the fold's end keeps every matured session
    text = out.read_text()
    assert out == tmp_path / "mmf_holdout_h5.md" and all(k in text for k in ("R1", "R2", "R3", "fusion_attention", "prior_fixed", "coverage", "T+2", "no_text"))
    table = r_table(text)
    assert [r[:3] for r in table] == [["R1", "fusion_attention", "prior_fixed"], ["R2", "fusion_attention", "dl_price"], *(["R3", "fusion_concat", a] for a in CP.ABLATIONS)]
    assert all(r[7] == "1" and r[10] in (E.verdict(-1, -1, 1, 1), E.verdict(-1, 1, 1, 1), E.verdict(1, 1, 0, 1)) for r in table)
    before = len(stubbed.trained)
    with pytest.raises(ValueError, match="đã chấm"):
        H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu")
    assert len(stubbed.trained) == before and seen == [True]


def test_run_that_fails_in_scoring_cannot_run_again(tmp_path, stubbed, monkeypatch):
    def boom(*a):
        raise RuntimeError("scoring failed")
    monkeypatch.setattr(CP, "load_h", boom)
    with pytest.raises(RuntimeError):
        H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu")
    assert stubbed.guard.exists() and not (tmp_path / "mmf_holdout_h5.md").exists()
    with pytest.raises(ValueError, match="đã chấm"):
        H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu")


def test_locked_stack_is_scored_without_training_it(tmp_path, stubbed):
    H.LOCK_FILE.write_text(json.dumps({"5": {**LOCK["5"], "fusion": "stack"}}))
    text = H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu").read_text()
    assert "stack" not in {a for a, *_ in stubbed.trained} and r_table(text)[0][:2] == ["R1", "stack"]


def test_early_run_never_claims_evidence(tmp_path, stubbed, monkeypatch):
    monkeypatch.setattr(CP.E, "verdict", lambda *a: "Có bằng chứng")
    blocks, boot = [], CP.E.block_bootstrap_dates
    for mod in (CP.E, H.R):
        monkeypatch.setattr(mod, "block_bootstrap_dates", lambda d, block, *a: (blocks.append(block), boot(d, block, *a))[1])
    monkeypatch.setattr(CP, "_row", lambda *a: {})              # tables() R1-R3 use dev-fold rules and are not printed
    text = H.run(5, stubbed.panel, guard_dir=tmp_path, early=True, device="cpu").read_text()
    table = r_table(text)
    assert len(table) == 5 and all(r[10] == H.EARLY_VERDICT and r[3] != "" for r in table)
    assert set(blocks) == {C.HOLDOUT_BLOCK} and len(blocks) >= 5 + 1      # R1, R2, three R3, then the simulations
    assert "Có bằng chứng" not in text and json.loads(stubbed.guard.read_text())["early"] is True


def test_untracked_guard_alone_refuses(tmp_path, data_dir):
    (data_dir / "holdout").mkdir(parents=True)
    (data_dir / "holdout" / "mmf_holdout_h5.lock").write_text("{}")
    with pytest.raises(ValueError, match="đã chấm"):
        H.check_ready(H.holdout_rows(frame_with(50)), 5, tmp_path)


def test_run_writes_both_guards(tmp_path, stubbed, data_dir):
    H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu")
    assert stubbed.guard.read_text() == (data_dir / "holdout" / "mmf_holdout_h5.lock").read_text()
    stubbed.guard.unlink()                                                           # a checkout or clean of docs/reports
    with pytest.raises(ValueError, match="đã chấm"):
        H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu")


def test_refuses_eval_with_too_little_news_before_any_guard(tmp_path, stubbed, data_dir):
    f = stubbed.frame
    f.loc[f["as_of_session"] > pd.Timestamp(C.HOLDOUT_AFTER), "has_news"] = 0.0
    with pytest.raises(ValueError, match="có tin"):
        H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu")
    assert not stubbed.trained and not stubbed.guard.exists() and not (data_dir / "holdout" / "mmf_holdout_h5.lock").exists()


def test_failed_report_is_recovered_from_the_saved_run(tmp_path, stubbed, monkeypatch):
    real = CP.load_h
    monkeypatch.setattr(CP, "load_h", lambda *a: (_ for _ in ()).throw(RuntimeError("scoring failed")))
    with pytest.raises(RuntimeError):
        H.run(5, stubbed.panel, guard_dir=tmp_path, device="cpu")
    monkeypatch.setattr(CP, "load_h", real)
    n = len(stubbed.trained)
    with pytest.raises(ValueError, match="dữ liệu"):
        H.write_report(5, SimpleNamespace(dataset_id="e" * 64, ohlcv=stubbed.panel.ohlcv), guard_dir=tmp_path)
    out = H.write_report(5, stubbed.panel, guard_dir=tmp_path)
    assert out == tmp_path / "mmf_holdout_h5.md" and len(r_table(out.read_text())) == 5 and len(stubbed.trained) == n and stubbed.guard.exists()


def test_report_needs_a_guard(tmp_path, stubbed):
    with pytest.raises(ValueError, match="chưa"):
        H.write_report(5, stubbed.panel, guard_dir=tmp_path)
