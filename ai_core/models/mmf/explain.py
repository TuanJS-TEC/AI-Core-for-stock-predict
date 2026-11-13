"""Group permutation, SHAP and LIME for the locked fusion, and the CLI writing the explanation and recommendation report."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer

from . import config as C
from . import baselines as B, compare as CP, evaluate as E, experiment as EX, recommend as R, train as TR
from .frame import GROUPS
from .windows import BINARY
from ai_core.prices import config as PC
from ai_core.prices.data import build_panel
from ai_core.prices.splits import folds_from_config

BRANCHES = ("price", "fund", "text")
ATTENTION = "fusion_attention"
SHAP_SEED = 59
N_PERM = 5            # permutations averaged per group
MAX_BG = 200; MAX_SHAP = 500; SHAP_DRAWS = 200
PROB_TOL = 1e-3       # reloaded weights vs the saved seed parquet
VN_CLASS = {"down": "Giảm", "flat": "Đi ngang", "up": "Tăng"}
COLORS = {"down": "#2a78d6", "flat": "#eb6834", "up": "#1baf7a"}
MARKERS = {"down": "o", "flat": "s", "up": "^"}


def _shuffle_group(samples, group, rng):
    """Copy of samples with the group's vectors permuted between the symbols of each session."""
    x = getattr(samples, f"x_{group}").copy()
    for d in np.unique(samples.as_of):
        k = np.flatnonzero(samples.as_of == d)
        x[k] = x[rng.permutation(k)]
    return replace(samples, **{f"x_{group}": x})


def group_permutation(predict_fn, samples, groups=BRANCHES, seed=0):
    """Log-loss rise when a group is shuffled inside each session (mean of N_PERM draws), CI by block bootstrap of sessions."""
    rng = np.random.default_rng(seed)
    base = E.row_logloss(samples.y, predict_fn(samples))
    rows = []
    for g in groups:
        ll = np.mean([E.row_logloss(samples.y, predict_fn(_shuffle_group(samples, g, rng))) for _ in range(N_PERM)], axis=0)
        rows.append(dict(zip(("delta_logloss", "lo", "hi"), E.paired_diff(ll, base, samples.as_of, seed=seed)), group=g))
    return pd.DataFrame(rows)[["group", "delta_logloss", "lo", "hi"]]


def shap_by_group(net, background, samples, device="cpu"):
    """|SHAP| of the logits summed over time and features, averaged over rows; share = part of the three groups, per class.

    The ticker index cannot take a gradient, so its embedding is fed as a fourth input and its attribution left out."""
    rng = np.random.default_rng(SHAP_SEED)
    emb, net.ticker = net.ticker, torch.nn.Identity()

    def inputs(s, n):
        i = rng.choice(len(s), min(n, len(s)), replace=False)
        return [torch.as_tensor(a[i]).to(device) for a in (s.x_price, s.x_text, s.x_fund)] + [emb(torch.as_tensor(s.ticker[i]).to(device)).detach()]

    try:
        ex = shap.GradientExplainer(net.to(device), inputs(background, MAX_BG), batch_size=SHAP_DRAWS)
        phi = ex.shap_values(inputs(samples, MAX_SHAP), nsamples=SHAP_DRAWS, rseed=SHAP_SEED)
    finally:
        net.ticker = emb
    per = {g: np.abs(p).sum(axis=tuple(range(1, p.ndim - 1))).mean(0) for g, p in zip(("price", "text", "fund"), phi)}
    total = sum(per.values())
    return pd.DataFrame([{"group": g, "class": n, "mean_abs_shap": per[g][k], "share": per[g][k] / total[k]}
                         for g in BRANCHES for k, n in enumerate(E.NAMES)])


def lime_cases(gbm, X, cols, rows, num_features=5):
    """LIME on the predicted class of each row of `rows`; X (train rows only) is the sampling distribution."""
    fn = lambda a: B._proba(gbm, a)
    ex = LimeTabularExplainer(np.asarray(X), feature_names=list(cols), class_names=E.NAMES, random_state=SHAP_SEED,
                              categorical_features=[i for i, c in enumerate(cols) if c in BINARY])
    out = []
    for x in np.asarray(rows):
        p = fn(x[None])[0]
        k = int(p.argmax())
        out.append({"class": E.NAMES[k], "proba": p, "weights": ex.explain_instance(x, fn, labels=(k,), num_features=num_features).as_list(label=k)})
    return out


def explained_arm(fusion):
    """The arm whose weights get explained: stack has none, so a locked stack goes through fusion_attention."""
    return ATTENTION if fusion == "stack" else fusion


def load_checked_net(b, run_dir, arm):
    """(net, temperature) of seed 59 on CPU; raises unless it reproduces that seed's saved eval probabilities on this bundle."""
    base = Path(run_dir) / f"h{b.h}" / b.fold.name
    kind = json.loads((base / "select.json").read_text())["price_kind"]
    EX._reusable(base / arm / f"seed{SHAP_SEED}.parquet", b, kind)
    net = EX._net_maker(b, arm, kind)()
    net.load_state_dict(torch.load(base / arm / f"seed{SHAP_SEED}.pt", map_location="cpu", weights_only=True))
    net.eval()
    t = EX.calibrate(b, TR.predict_logits(net, b.samples["calib"], "cpu"))[0]
    saved = pd.read_parquet(base / arm / f"seed{SHAP_SEED}.parquet")
    saved = saved.loc[saved["block"].astype(str) == "eval", E.PROB_COLS].to_numpy()
    gap = np.abs(E.apply_temperature(TR.predict_logits(net, b.samples["eval"], "cpu"), t) - saved).max()
    if gap > PROB_TOL:
        raise ValueError(f"h{b.h} {b.fold.name} {arm}: weights do not reproduce the saved probabilities on this data, max gap {gap:.4g}")
    return net, t


def _attention_weights(net, s):
    with torch.no_grad():
        net(*TR._tensors(s, "cpu"))
    return dict(zip(net.branches, net.last_weights.mean(0).numpy()))


def _gbm_all(b):
    """gbm_all as the baseline fits it: train rows only."""
    cols = [c for g in B.SUFFIX["all"] for c in GROUPS[g]]
    x = lambda k: b.scaled.loc[b.blocks[k].index, cols].to_numpy(float)
    gbm = B._model("gbm").fit(x("train"), b.blocks["train"]["label"].to_numpy(dtype=int))
    return gbm, x, cols


def _lime_rows(b, gbm, x, cols):
    """Train rows are LIME's distribution; the three eval rows are picked by the model alone, never by the outcome."""
    ev = x("eval")
    p = B._proba(gbm, ev)
    gap = p[:, 2] - p[:, 0]
    picks = {"p_up - p_down lớn nhất": gap.argmax(), "p_down - p_up lớn nhất": gap.argmin(), "gần nhau nhất": np.abs(gap).argmin()}
    s = b.samples["eval"]
    rows = []
    for (why, i), c in zip(picks.items(), lime_cases(gbm, x("train"), cols, ev[list(picks.values())])):
        rows.append({"h": b.h, "fold": b.fold.name, "ca": why, "symbol": s.symbol[i], "phiên": str(s.as_of[i])[:10], "dự báo": VN_CLASS[c["class"]],
                     "p": c["proba"].max(), "thực tế": VN_CLASS[E.NAMES[s.y[i]]], "đặc trưng": "; ".join(f"{n} ({w:+.3f})" for n, w in c["weights"])})
    return rows


def _label_shares(ev):
    """Share of the two labels that are not trades among the rows with a return (coverage's base), each at its fold's margin."""
    ev = ev[ev["net_ret"].notna()]
    k = pd.Series(R.label(ev["p_up"], ev["p_down"], ev["m"].to_numpy())).value_counts(normalize=True)
    return {"share_tranh": k.get(R.RULES[1], 0.0), "share_khong_du": k.get(R.RULES[2], 0.0)}


def _simulations(preds, b, arm, block=PC.BOOTSTRAP_BLOCK):
    """(row of the fold, eval rows with the fold's margin): margin from calib ensemble probabilities, scored on eval."""
    d = preds[(preds["fold"] == b.fold.name) & (preds["arm"] == arm)]
    cal = R.with_net_ret(d[d["block"] == "calib"], b.frame, b.h, b.fold.calib_end)
    ev = R.with_net_ret(d[d["block"] == "eval"], b.frame, b.h, b.fold.eval_end)
    m = R.pick_margin(cal["p_up"], cal["p_down"], cal["net_ret"])
    mm = np.inf if m is None else m       # no margin on calib: no trade
    ev = ev.assign(m=mm)
    c = R.simulate(cal, mm, b.h, block=block)
    return {**R.simulate(ev, mm, b.h, block=block), **_label_shares(ev), "fold": b.fold.name, "m": m,
            "calib_mean_net": c["mean_net"], "calib_coverage": c["coverage"], "calib_n": c["n"]}, ev


def _reliability(ev, h, path):
    """One-vs-rest reliability on pooled eval, ten equal-count bins per class."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, n in enumerate(E.NAMES):
        p, hit = ev[E.PROB_COLS[k]], (ev["y_true"] == k).astype(float)
        g = pd.DataFrame({"p": p, "hit": hit}).groupby(pd.qcut(p, 10, duplicates="drop"), observed=True).mean()
        ax.plot(g["p"], g["hit"], marker=MARKERS[n], color=COLORS[n], linewidth=2, markersize=8, label=VN_CLASS[n])
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color="#8a8984", linewidth=1, linestyle="--", label="Hiệu chỉnh hoàn hảo")
    ax.set(xlim=lim, ylim=lim, xlabel="Xác suất dự báo (trung bình mỗi nhóm)", ylabel="Tần suất thực tế",
           title=f"T+{h}: độ tin cậy xác suất, eval gộp A, B, C ({len(ev)} mẫu)")
    ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def _confusion(ev, h, path):
    """Counts and row share of the decided class (fold's alpha and prior) on pooled eval."""
    pred = np.concatenate([E.decide(g[E.PROB_COLS].to_numpy(), g["alpha"].iloc[0], g[EX.PRIOR_COLS].to_numpy()) for _, g in ev.groupby("fold", sort=False)])
    y = np.concatenate([g["y_true"].to_numpy() for _, g in ev.groupby("fold", sort=False)])
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (y, pred), 1)
    share = cm / cm.sum(1, keepdims=True).clip(1)
    names = [VN_CLASS[n] for n in E.NAMES]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", color="white" if share[i, j] > 0.55 else "#0b0b0b")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names, xlabel="Dự báo", ylabel="Thực tế",
           title=f"T+{h}: ma trận nhầm lẫn, eval gộp A, B, C ({len(y)} mẫu)")
    fig.colorbar(im, ax=ax, label="Tỷ lệ trong hàng thực tế", shrink=0.8); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def _p_vs_real(d):
    """Mean p_up and p_down against how often Up and Down happened."""
    return (f"p_up {d['p_up'].mean():.3f}, tần suất Tăng {(d['y_true'] == 2).mean():.3f}; "
            f"p_down {d['p_down'].mean():.3f}, tần suất Giảm {(d['y_true'] == 0).mean():.3f}")


def _calibration(ev, h):
    """(p_up overstated on eval?, line with predicted against observed on all eval and on the rows the rule buys)."""
    b = ev[(ev["p_up"] - ev["p_down"] >= ev["m"]) & ev["net_ret"].notna()]
    return ev["p_up"].mean() > (ev["y_true"] == 2).mean(), f"T+{h}, toàn eval: {_p_vs_real(ev)}. {len(b)} lệnh của luật: {_p_vs_real(b) if len(b) else 'không có'}."


def _negative_calib(sim):
    """Folds whose chosen margin still had a negative mean net on calib; the rule trades on them anyway."""
    d = sim[(sim["pham_vi"] == "theo luật") & (sim["calib_mean_net"] < 0)]
    return ", ".join(f"h{r.h} {r.fold} ({r.calib_mean_net:+.4f}, độ phủ calib {r.calib_coverage:.1%}, {int(r.calib_n)} lệnh)" for r in d.itertuples()) or "không có ô nào"


SIM_COLS = ["h", "fold", "pham_vi", "m", "coverage", "n", "n_days", "mean_net", "ci_lo", "ci_hi", "rand_median", "share_tranh", "share_khong_du", "calib_mean_net", "calib_coverage", "calib_n"]


def _cells(rows, extra=lambda r: ""):
    return ", ".join(f"{r['fold']} ({r['mean_net']:+.4f}, độ phủ {r['coverage']:.1%}, {int(r['n'])} lệnh{extra(r)})" for _, r in rows.iterrows()) or "không có"


def _verdict(pool):
    """A loss is said as a loss; only a CI above zero and above random counts as support."""
    no_use = "Không dùng nhãn \"Ưu tiên mua\" ra quyết định."
    less_bad = " Cao hơn ngẫu nhiên chỉ có nghĩa là lỗ ít hơn." if pool["mean_net"] > pool["rand_median"] else ""
    if pool["n"] == 0:
        return "Không có lệnh nào."
    if pool["ci_hi"] < 0:
        return f"**Luật lỗ sau phí:** trung bình âm và cả CI nằm dưới 0, bằng chứng nghiêng về lỗ chứ không chỉ là thiếu bằng chứng.{less_bad} {no_use}"
    if pool["mean_net"] < 0:
        return f"**Luật không có lãi:** trung bình âm, CI chứa 0.{less_bad} {no_use}"
    if pool["ci_lo"] > 0 and pool["mean_net"] > pool["rand_median"]:
        return "CI nằm trên 0 và trung bình vượt ngẫu nhiên, nhưng đây là vùng phát triển nên chưa phải bằng chứng cuối."
    return f"Trung bình dương nhưng chưa đủ bằng chứng. {no_use}"


def _h_section(h, fusion, sim, perm, shares):
    """Plain conclusion of one horizon, read off the tables."""
    rule = sim[(sim["h"] == h) & (sim["pham_vi"] == "theo luật")]
    pool, cells = rule[rule["fold"] == "gộp"].iloc[0], rule[rule["fold"] != "gộp"]
    every = sim[(sim["h"] == h) & (sim["fold"] == "gộp") & (sim["pham_vi"] == "toàn mẫu")].iloc[0]
    p = perm[perm["h"] == h]
    help_ = [f"{g} ({', '.join(f'{r.fold} {r.delta_logloss:+.4f}' for r in gg.itertuples())})" for g, gg in p.groupby("group") if (gg["lo"] > 0).sum() * 3 >= 2 * len(gg)]
    hurt = [f"{g} ({', '.join(f'{r.fold} {r.delta_logloss:+.4f}, hi {r.hi:+.4f}' for r in gg.itertuples())})" for g, gg in p[p["hi"] < 0].groupby("group")]
    s = shares[shares["h"] == h].groupby("group")["share"].mean().sort_values(ascending=False)
    return [f"**T+{h}** (fusion khóa {fusion}). Theo luật: độ phủ {pool['coverage']:.1%}, {int(pool['n'])} lệnh, {int(pool['n_days'])} phiên, "
            f"lợi suất ròng trung bình {pool['mean_net']:+.4f} (CI 95% {pool['ci_lo']:+.4f} đến {pool['ci_hi']:+.4f}), ngẫu nhiên cùng số lệnh {pool['rand_median']:+.4f}. "
            f"Toàn mẫu: độ phủ {every['coverage']:.0%}, {int(every['n'])} lệnh, {every['mean_net']:+.4f}. "
            + _verdict(pool),
            "",
            f"Fold có CI nằm hẳn dưới 0: {_cells(cells[cells['ci_hi'] < 0])}. "
            f"Fold kém hơn ngẫu nhiên: {_cells(cells[cells['mean_net'] < cells['rand_median']], lambda r: f', ngẫu nhiên {r.rand_median:+.4f}')}.",
            "",
            f"Nhóm làm log-loss tăng khi hoán vị (lo > 0 ở ít nhất 2/3 fold; một seed, CI chưa hiệu chỉnh đa so sánh): {', '.join(help_) or 'không có'}. "
            f"Nhóm làm log-loss giảm khi hoán vị (hi < 0; một seed, CI chưa hiệu chỉnh đa so sánh): {', '.join(hurt) or 'không có'}. "
            f"Tỷ trọng SHAP trung bình: {', '.join(f'{g} {v:.0%}' for g, v in s.items())}.", ""]


def write_report(run_dir, lock_path, build, out=C.REPORTS / "mmf_explain.md", fig_dir=C.REPORTS / "mmf_figures"):
    """Per (h, fold) of the lock: permutation, SHAP, attention weights, LIME, margin and simulation; figures per h. build(h, fold name) -> Bundle."""
    lock = json.loads(Path(lock_path).read_text())
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    perm, shares, attn, cases, sims, figs, used, calibration = [], [], [], [], [], [], [], []
    for hs, entry in lock.items():
        h, fusion = int(hs), entry["fusion"]
        arm = explained_arm(fusion)
        used.append({"h": h, "fusion khóa": fusion, "arm giải thích": arm})
        preds = CP.add_stack(CP.load_h(run_dir, h))
        evs = []
        for name in entry["picks"]:
            b = build(h, name)
            net, t = load_checked_net(b, run_dir, arm)
            s_ev = b.samples["eval"]
            tag = {"h": h, "fold": name}
            perm.append(group_permutation(lambda s: E.apply_temperature(TR.predict_logits(net, s, "cpu"), t), s_ev).assign(**tag))
            shares.append(shap_by_group(net, b.samples["train"], s_ev).assign(**tag))
            att_net = net if arm == ATTENTION else load_checked_net(b, run_dir, ATTENTION)[0]
            attn.append({**tag, **_attention_weights(att_net, s_ev)})
            cases += _lime_rows(b, *_gbm_all(b))
            row, ev = _simulations(preds, b, fusion)
            sims.append({**row, "pham_vi": "theo luật"})
            sims.append({**R.simulate(ev, -1.0, h), "fold": name, "m": np.nan, "pham_vi": "toàn mẫu"})
            evs.append(ev)
            print(f"h{h} {name} done", flush=True)
        ev = pd.concat(evs, ignore_index=True)
        # ponytail: pooled rows shifted by their fold's margin, so one call to simulate serves folds with different m
        pool = ev.assign(p_up=ev["p_up"] - ev["m"])
        sims.append({**R.simulate(pool, 0.0, h), **_label_shares(ev), "fold": "gộp", "m": np.nan, "pham_vi": "theo luật"})
        sims.append({**R.simulate(ev, -1.0, h), "fold": "gộp", "m": np.nan, "pham_vi": "toàn mẫu"})
        calibration.append(_calibration(ev, h))
        _reliability(ev, h, fig_dir / f"reliability_h{h}.png")
        _confusion(ev, h, fig_dir / f"confusion_h{h}.png")
        figs += [f"![Reliability T+{h}]({fig_dir.name}/reliability_h{h}.png)", "", f"![Ma trận nhầm lẫn T+{h}]({fig_dir.name}/confusion_h{h}.png)", ""]
    perm, shares, sim = pd.concat(perm, ignore_index=True), pd.concat(shares, ignore_index=True), pd.DataFrame(sims)
    table = sim.assign(calib_n=[("" if pd.isna(v) else str(int(v))) for v in sim["calib_n"]])
    summary = [line for hs, entry in lock.items() for line in _h_section(int(hs), entry["fusion"], sim, perm, shares)]
    ids = sorted({e.get("dataset_id", "?") for e in lock.values()})
    lines = [
        "# Giải thích và khuyến nghị: model đa phương thức T+2, T+5", "",
        "**Đây là vùng phát triển, chưa phải kết quả cuối.** Khối eval của fold A, B, C đã bị nhìn trong các thí nghiệm trước. "
        "Mọi đo lường ở đây dùng hàng eval; không có gì được fit trên eval: nhiệt độ và ngưỡng `m` từ calib, nền SHAP, phân phối LIME và `gbm_all` từ train.", "",
        f"**Giả định chưa xác nhận:** lợi suất tính theo thanh toán T+2 và không bán khống, chủ dự án chưa xác nhận giả định này. "
        f"Lệnh chỉ LONG, mua ở giá mở cửa t+1, bán ở giá đóng cửa t+h+1, trừ phí {C.COST:.4f}. Không tính trượt giá, kẹt trần sàn, hay giới hạn vốn; các lệnh của phiên liền nhau chồng lấn thời gian.", "",
        f"Dataset: {', '.join(ids)}. Arm giải thích theo file khóa, dùng chung cho mọi fold của h: khóa là `stack` (không có trọng số) thì dùng `fusion_attention`.", "",
        pd.DataFrame(used).to_markdown(index=False), "",
        "## Kết luận theo h", "", *summary,
        "## Luật khuyến nghị", "",
        f"- **{R.RULES[0]}**: `p_up - p_down >= m`.", f"- **{R.RULES[1]}**: `p_down - p_up >= m`.", f"- **{R.RULES[2]}**: còn lại.", "",
        "`m` chọn trên calib của từng fold từ xác suất ensemble (trung bình 5 seed), lấy trung bình lợi suất ròng cao nhất với ít nhất 50 lệnh; không có `m` đạt thì không có lệnh. "
        "Lệnh có phiên bán t+h+1 vượt cuối khối bị bỏ khỏi cả việc chọn `m` lẫn mô phỏng (các hàng ở phiên có t+h trùng cuối khối), nên calib không đọc giá của eval và eval không đọc giá sau khối.", "",
        f"Luật không đòi calib dương: `m` tốt nhất có trung bình calib âm thì luật vẫn giao dịch và kết quả vẫn được tính. Các ô có trung bình calib âm: {_negative_calib(sim)}.", "",
        "## Kết quả mô phỏng kèm độ phủ", "",
        "`coverage` (độ phủ) là tỷ lệ hàng eval có lợi suất được gắn \"Ưu tiên mua\". `toàn mẫu` là mua mọi hàng, để so sánh. `gộp` nối eval của ba fold, mỗi fold giữ `m` của nó. "
        "`rand_median` là trung vị của 1000 lần rút ngẫu nhiên cùng số lệnh. CI 95% theo block bootstrap phiên. "
        "`calib_mean_net`, `calib_coverage`, `calib_n` là lợi suất ròng trung bình, độ phủ và số lệnh trên calib tại `m` đã chọn: đó là số dùng để chọn `m`, không phải kết quả ngoài mẫu."
        " Độ phủ thấp nghĩa là kết quả chỉ nói về phần nhỏ đó; `n` dưới khoảng 30 thì CI gần như không đọc được.", "",
        CP._md(table, SIM_COLS), "",
        "## Trọng số attention theo nhánh", "",
        "Trọng số attention của `fusion_attention` seed 59 (`net.last_weights`), trung bình trên hàng eval. Trọng số cho biết model chú ý nhánh nào, không chứng minh nhánh đó giúp dự báo.", "",
        CP._md(pd.DataFrame(attn), ["h", "fold", *BRANCHES]), "",
        "## Permutation theo nhóm", "",
        f"Log-loss seed 59 đã hiệu chỉnh nhiệt độ, sau hoán vị trừ trước hoán vị, trung bình {N_PERM} lần hoán vị; dương là nhóm có ích. Vector của nhóm được hoán vị giữa các mã trong cùng phiên, "
        "nên phần thông tin chung cả thị trường trong phiên không bị phá: bảng chỉ đo khả năng phân biệt giữa các mã. Các CI chưa hiệu chỉnh đa so sánh.", "",
        CP._md(perm, ["h", "fold", "group", "delta_logloss", "lo", "hi"]), "",
        "## Tỷ trọng SHAP theo nhóm và lớp", "",
        f"`GradientExplainer` trên logit của seed 59, CPU, {MAX_BG} mẫu nền từ train, {MAX_SHAP} mẫu từ eval. |SHAP| cộng qua thời gian và đặc trưng rồi chia tổng ba nhóm; phần của embedding mã không tính. "
        "Nhóm giá có nhiều ô hơn hẳn (cửa sổ 60 phiên), nên tỷ trọng không chuẩn hóa theo số đặc trưng và không phải mức đóng góp vào độ chính xác.", "",
        CP._md(shares, ["h", "fold", "class", "group", "mean_abs_shap", "share"]), "",
        "## Ba ca LIME mỗi (h, fold)", "",
        "`gbm_all` refit trên train. Ba hàng eval chọn theo xác suất của chính model, không theo kết quả. Trọng số giải thích lớp được dự báo; dương là đẩy về lớp đó. Ngưỡng trong tên đặc trưng là giá trị đã chuẩn hóa theo train.", "",
        pd.DataFrame(cases).to_markdown(index=False, floatfmt=".3f"), "",
        "## Độ tin cậy xác suất và ma trận nhầm lẫn", "",
        "Xác suất ensemble của fusion khóa trên eval gộp. Ma trận dùng lớp quyết định theo alpha và prior của từng fold.", "",
        ("Hiệu chỉnh chưa đúng trên eval: p_up bị đánh giá cao (xem hình reliability), và điều này làm `p_up - p_down` rộng ra đúng về phía luật mua, nên `m` không đọc được như một mức tin cậy thật."
         if any(over for over, _ in calibration) else "Trung bình p_up trên eval không vượt tần suất Tăng thực tế."), "",
        *[f"- {line}" for _, line in calibration], "", *figs,
    ]
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stock-dir")
    ap.add_argument("--lock", default=C.REPORTS / "mmf_lock.json")
    ap.add_argument("--out", default=C.REPORTS / "mmf_explain.md")
    ap.add_argument("--fig-dir", default=C.REPORTS / "mmf_figures")
    a = ap.parse_args(argv)
    panel = build_panel(C.TICKERS, stock_dir=a.stock_dir)
    folds = {f.name: f for f in folds_from_config()}
    print(write_report(a.run_dir, a.lock, lambda h, name: EX.fold_bundle(panel, h, folds[name]), a.out, a.fig_dir))


if __name__ == "__main__":
    main()
