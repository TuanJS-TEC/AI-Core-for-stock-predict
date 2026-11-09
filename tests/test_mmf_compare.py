import json
import zlib

import numpy as np, pandas as pd, pytest
from scipy.special import softmax
from mmf import compare as C, config as MC, evaluate as E

ALL_ARMS = list(MC.ARMS) + list(MC.BASELINES)
SYMS = ["AAA", "BBB", "CCC"]
PRIOR = [0.4, 0.3, 0.3]


def fake(arm, fold, offset, n=300, seed=0, h=5):
    """Fake preds: fixed labels shared by all arms, logits = noise + offset on the true class, calib and eval blocks."""
    per = n // 2 // len(SYMS)
    cal = pd.bdate_range("2023-01-02", periods=per); ev = pd.bdate_range("2023-06-01", periods=per)
    y = np.random.default_rng(1).integers(0, 3, n)
    noise = np.random.default_rng(zlib.crc32(f"{arm}{fold}{seed}".encode())).normal(size=(n, 3))
    z = noise + offset * np.eye(3)[y]
    d = pd.DataFrame({"symbol": np.tile(SYMS, 2 * per), "as_of_session": np.repeat(list(cal) + list(ev), len(SYMS)), "y_true": y,
                      "block": ["calib"] * (n // 2) + ["eval"] * (n // 2)})
    d["label_end_date"] = d["as_of_session"] + pd.offsets.BDay(5)
    d[E.PROB_COLS] = softmax(z, axis=1)
    d[["prior_down", "prior_flat", "prior_up"]] = PRIOR
    return d.assign(alpha=0.0, arm=arm, h=h, fold=fold, seed=seed, price_kind="lstm" if arm == "dl_price" else "none",
                    nlp_version="n", dataset_id="ds", git_commit="abc")


def _run_dir(tmp_path, folds, hs=(5,), seeds=(59, 60), arms=ALL_ARMS, **kw):
    for h in hs:
        for f in folds:
            for a in arms:
                for s in ((0,) if a in MC.BASELINES else seeds):
                    p = tmp_path / f"h{h}/{f}/{a}/seed{s}.parquet"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    fake(a, f, kw.get(a, 0.5), seed=s, h=h).to_parquet(p, index=False)
    return tmp_path


def test_pick_uses_calib_only():
    p = pd.concat([fake("a", "A", 2.0).query("block == 'calib'"), fake("a", "A", 0.0).query("block == 'eval'"),
                   fake("b", "A", 0.0).query("block == 'calib'"), fake("b", "A", 2.0).query("block == 'eval'")])
    assert C.pick(p, "A", ["a", "b"]) == "a"


def test_clear_winner_gets_evidence_verdict():
    p = pd.concat([fake("good", f, 2.0) for f in "ABC"] + [fake("bad", f, 0.0) for f in "ABC"])
    r = C.compare(p, dict.fromkeys("ABC", "good"), dict.fromkeys("ABC", "bad"))
    assert r["verdict"] == "Có bằng chứng" and r["fold_wins"] == 3 and r["hi"] < 0


def test_stack_trains_on_calib_never_eval():
    p = pd.concat([fake(a, "A", 1.0) for a in ("dl_price", "dl_fund", "dl_text")])
    a = C.stack(p, "A")
    q = p.copy(); q.loc[q.block == "eval", "y_true"] = 0
    b = C.stack(q, "A")
    np.testing.assert_array_equal(a[E.PROB_COLS].to_numpy(), b[E.PROB_COLS].to_numpy())


def test_stack_calib_probabilities_are_out_of_fold():
    p = pd.concat([fake(a, "A", 1.0) for a in ("dl_price", "dl_fund", "dl_text")])
    a = C.stack(p, "A")
    cal = a[a.block == "calib"]
    first = cal.iloc[: len(cal) // 5][["symbol", "as_of_session"]]      # first contiguous block of the cross-fit
    hit = p.set_index(["symbol", "as_of_session"]).index.isin(first.set_index(["symbol", "as_of_session"]).index) & (p.block == "calib").to_numpy()
    q = p.copy(); q.loc[hit, "y_true"] = (q.loc[hit, "y_true"] + 1) % 3
    b = C.stack(q, "A")
    n = len(first)
    np.testing.assert_array_equal(a[a.block == "calib"][E.PROB_COLS].to_numpy()[:n], b[b.block == "calib"][E.PROB_COLS].to_numpy()[:n])
    assert not np.allclose(a[a.block == "calib"][E.PROB_COLS].to_numpy()[n:], b[b.block == "calib"][E.PROB_COLS].to_numpy()[n:])


def test_leak_alarm_fires_on_suspicious_score():
    t = C.tables(pd.concat([fake(a, f, 8.0) for a in ALL_ARMS for f in "ABC"]))
    assert len(t["leak_alarms"]) > 0


def test_load_h_averages_seeds(tmp_path):
    d = _run_dir(tmp_path, "A")
    out = C.load_h(d, 5)
    one = out[(out.arm == "dl_text") & (out.block == "eval")].sort_values(["as_of_session", "symbol"])
    want = (fake("dl_text", "A", 0.5, seed=59)[E.PROB_COLS].to_numpy() + fake("dl_text", "A", 0.5, seed=60)[E.PROB_COLS].to_numpy()) / 2
    want = fake("dl_text", "A", 0.5, seed=59).assign(**dict(zip(E.PROB_COLS, want.T))).query("block == 'eval'").sort_values(["as_of_session", "symbol"])
    np.testing.assert_allclose(one[E.PROB_COLS].to_numpy(), want[E.PROB_COLS].to_numpy())


def test_partial_run_reports_but_lock_refuses(tmp_path):
    d = _run_dir(tmp_path / "run", "A")
    out = C.write_report(d, tmp_path / "r.md").read_text()
    assert "chưa phải kết quả cuối" in out and "## R1" in out and "stack" in out and C.NOT_ENOUGH in out
    with pytest.raises(ValueError):
        C.write_lock(d, tmp_path / "lock.json")


def test_mid_fold_partial_run_skips_missing_sides(tmp_path):
    d = _run_dir(tmp_path / "run", "A")
    _run_dir(d, "B", arms=MC.BASELINES)                          # fold B: baselines only
    _run_dir(d, "C", arms=list(MC.BASELINES) + list(C.UNIMODAL) + ["fusion_concat"])   # fold C: no ablations yet
    out = C.write_report(d, tmp_path / "r.md").read_text()
    t = C.tables(C._all_preds(d))
    assert t["r1"]["skipped"].iloc[0] == "B" and t["r1"]["n_folds"].iloc[0] == 2
    assert set(t["r3"]["skipped"]) == {"B C"} and (t["r3"]["verdict"] == C.NOT_ENOUGH).all()
    assert C.NOT_ENOUGH in out


def test_early_interrupted_runs_still_report(tmp_path):
    for name, arms in (("base", MC.BASELINES), ("nofusion", list(MC.BASELINES) + list(C.UNIMODAL))):
        d = _run_dir(tmp_path / name, "A", arms=arms)
        out = C.write_report(d, tmp_path / f"{name}.md").read_text()
        assert "## R1" in out and C.NOT_ENOUGH in out


def test_full_folds_keep_real_verdict():
    t = C.tables(pd.concat([fake(a, f, 2.0 if a == "fusion_concat" else 0.0) for a in ALL_ARMS for f in "ABC"]))
    assert t["r3"]["verdict"].iloc[0] != C.NOT_ENOUGH and (t["r3"]["skipped"] == "").all()


def _full(tmp_path, **kw):
    return _run_dir(tmp_path / "run", "ABC", hs=MC.HORIZONS, seeds=MC.SEEDS, **kw)


def test_lock_writes_picks_for_complete_run(tmp_path):
    d = _full(tmp_path, fusion_attention=3.0)
    lock = json.loads(C.write_lock(d, tmp_path / "lock.json").read_text())
    assert set(lock) == {str(h) for h in MC.HORIZONS}
    lock = lock["5"]
    assert lock["fusion"] == "fusion_attention" and set(lock["picks"]) == set("ABC") and lock["price_kind"] == "lstm"
    assert lock["dataset_id"] == "ds" and lock["baseline"] in MC.BASELINES and lock["unimodal"] in C.UNIMODAL


def test_lock_refuses_missing_horizon_seed_or_existing_file(tmp_path):
    with pytest.raises(ValueError, match="horizons"):
        C.write_lock(_run_dir(tmp_path / "one_h", "ABC", seeds=MC.SEEDS), tmp_path / "a.json")
    d = _full(tmp_path)
    (d / "h2/B/fusion_concat/seed63.parquet").unlink()
    with pytest.raises(ValueError, match="missing"):
        C.write_lock(d, tmp_path / "b.json")
    fake("fusion_concat", "B", 0.5, seed=63, h=2).to_parquet(d / "h2/B/fusion_concat/seed63.parquet", index=False)
    lock = C.write_lock(d, tmp_path / "c.json")
    with pytest.raises(ValueError, match="exists"):
        C.write_lock(d, lock)


def test_r3_says_when_removing_a_branch_made_fusion_better():
    r3 = pd.DataFrame([{"h": 5, "b": "A=no_text B=no_text C=no_text", "mean": 0.0082, "lo": 0.0037, "hi": 0.0121, "fold_wins": 0, "n_folds": 3},
                       {"h": 2, "b": "A=no_text B=no_text C=no_text", "mean": 0.0017, "lo": -0.0005, "hi": 0.0037, "fold_wins": 0, "n_folds": 3},
                       {"h": 2, "b": "A=no_price B=no_price C=no_price", "mean": -0.0112, "lo": -0.0242, "hi": 0.0006, "fold_wins": 3, "n_folds": 3},
                       {"h": 5, "b": "", "n_folds": 0}])
    notes = C._branch_notes(r3)
    assert len(notes) == 2
    assert "T+5" in notes[0] and "bỏ nhánh text làm fusion tốt hơn" in notes[0] and "+0.0082" in notes[0] and "bằng chứng chống lại nhánh text" in notes[0]
    assert "T+2" in notes[1] and "chưa chứng minh" in notes[1] and "+0.0017" in notes[1] and "0/3" in notes[1]
