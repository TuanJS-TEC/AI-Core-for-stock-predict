"""Recommendation rule on p_up - p_down, its margin picked on calib, and a LONG-only simulation (T+2, no short sale)."""
import numpy as np
import pandas as pd

from . import config as C
from ai_core.prices import config as PC
from ai_core.prices.stats import block_bootstrap_dates

RULES = ("Ưu tiên mua", "Tránh", "Không đủ tin cậy")


def trade_return(frame, h, cost=C.COST):
    """log(close[t+h+1] / open[t+1]) - cost per symbol, on the frame's index; NaN when the bars do not exist yet."""
    f = frame.sort_values("as_of_session")
    g = f.groupby(f["symbol"].astype(str))
    return (np.log(g["close"].shift(-(h + 1)) / g["open"].shift(-1)) - cost).reindex(frame.index)


def with_net_ret(rows, frame, h, block_end, cost=C.COST):
    """`rows` of one block plus net_ret, from prices up to block_end only.

    The exit t+h+1 is one session later than the label's t+h, so the block's last rows would read the next block.
    Their net_ret is NaN here, and pick_margin and simulate leave them out."""
    f = frame[frame["as_of_session"] <= block_end]
    ret = pd.DataFrame({"symbol": f["symbol"].astype(str), "as_of_session": f["as_of_session"], "net_ret": trade_return(f, h, cost)})
    return rows.merge(ret, on=["symbol", "as_of_session"], how="left", validate="one_to_one")


def pick_margin(p_up, p_down, net_ret, grid=np.arange(0, 0.51, 0.02), min_n=50):
    """Margin with the best mean net return over at least min_n LONG rows (ties keep the smaller); None when no margin has them."""
    gap, net = np.asarray(p_up) - np.asarray(p_down), np.asarray(net_ret, dtype=float)
    ok = np.isfinite(net)
    means = {float(m): net[k].mean().round(12) for m in grid if (k := ok & (gap >= m)).sum() >= min_n}
    return max(means, key=means.get) if means else None


def label(p_up, p_down, m):
    """One of RULES per row: p_up - p_down >= m buys, p_down - p_up >= m avoids, the rest are not confident enough."""
    gap = np.asarray(p_up) - np.asarray(p_down)
    return np.select([gap >= m, -gap >= m], RULES[:2], RULES[2])


def simulate(rows, m, h, seed=0, block=PC.BOOTSTRAP_BLOCK):
    """LONG rows of `rows` at margin m. Rows without a return are left out, CI by block bootstrap over the sessions.

    rand_median: median mean return of 1000 random draws of the same number of trades from the same rows."""
    d = rows[np.isfinite(rows["net_ret"])]
    net, pick = d["net_ret"].to_numpy(), (d["p_up"] - d["p_down"]).to_numpy() >= m
    n, rng = int(pick.sum()), np.random.default_rng(seed)
    out = {"h": h, "m": m, "coverage": n / len(d) if len(d) else np.nan, "n": n,
           "n_days": d.loc[pick, "as_of_session"].nunique(), "mean_net": np.nan, "rand_median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    if not n:
        return out
    day = pd.DataFrame({"s": net * pick, "n": pick.astype(int)}).groupby(d["as_of_session"].to_numpy()).sum().to_numpy()
    idx = block_bootstrap_dates(day, block, 2000, rng)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.array([day[i, 0].sum() / day[i, 1].sum() for i in idx])    # a resample without trades is NaN
    return {**out, "mean_net": float(net[pick].mean()), "rand_median": float(np.median([rng.choice(net, n, replace=False).mean() for _ in range(1000)])),
            "ci_lo": float(np.nanpercentile(boot, 2.5)), "ci_hi": float(np.nanpercentile(boot, 97.5))}
