import numpy as np
import pandas as pd
import pytest
import torch

from ai_core.models.mmf import config as C, v3
from ai_core.models.mmf.frame import GROUPS
from ai_core.prices import config as PC


def _rows(z, h=2, sigma=0.02):
    """Train rows whose |return| is z times their own threshold, labelled the way make_labels does."""
    theta = PC.K_ADAPT * sigma * np.sqrt(h)
    r = np.asarray(z) * theta
    return pd.DataFrame({"std_60": sigma, "realized_return": r, "label": np.where(r > theta, 2, np.where(r < -theta, 0, 1))})


def test_margin_weights_drop_only_the_band_around_the_threshold():
    w = v3.margin_weights(_rows([0.0, 0.5, 0.84, 0.9, 1.1, -0.9, -1.1, 1.16, -3.0]), 2)
    assert w.tolist() == [1, 1, 1, 0, 0, 0, 0, 1, 1]


def test_margin_weights_refuse_rows_whose_labels_do_not_match_the_threshold():
    rows = _rows([0.5, 2.0])
    rows.loc[0, "label"] = 2
    with pytest.raises(ValueError):
        v3.margin_weights(rows, 2)


def test_short_price_net_reads_only_the_last_sessions_of_the_step_features():
    torch.manual_seed(0)
    f = len(GROUPS["price"])
    idx = [GROUPS["price"].index(c) for c in v3.STEP]
    net = v3.ShortPriceNet(3, f, idx).eval()
    x = torch.randn(4, C.LP, f)
    args = (torch.zeros(4, C.LT, len(GROUPS["text"])), torch.zeros(4, len(GROUPS["fund"])), torch.tensor([0, 1, 2, 0]))
    base = net(x, *args)
    assert base.shape == (4, 3)
    early = x.clone(); early[:, :C.LP - v3.SHORT] += 1.0          # sessions before the short window
    assert torch.equal(net(early, *args), base)
    late = x.clone(); late[:, -2, idx[0]] += 1.0                   # a step feature inside the window
    assert not torch.equal(net(late, *args), base)


def test_picks_take_the_best_macro_f1_on_tune_and_calib_not_eval():
    y = np.array([0, 1, 2] * 4)
    good, bad = np.eye(3)[y] * 0.8 + 0.1 / 3 * 2, np.eye(3)[(y + 1) % 3] * 0.8 + 0.1 / 3 * 2
    def arm(name, fit_p, eval_p):
        d = pd.DataFrame({"y_true": np.tile(y, 3), "block": np.repeat(["tune", "calib", "eval"], len(y)), "arm": name, "fold": "A", "alpha": 0.0})
        d[v3.E.PROB_COLS] = np.vstack([fit_p, fit_p, eval_p]); d[v3.X.PRIOR_COLS] = 1 / 3
        return d
    d = pd.concat([arm("fusion_concat", bad, good), arm("fusion_attention", good, bad)], ignore_index=True)
    assert v3.picks(d, v3.FUSION) == {"A": "fusion_attention"}
