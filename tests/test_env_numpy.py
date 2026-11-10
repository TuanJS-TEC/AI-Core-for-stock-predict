"""Canary for the numpy 2.2.6 / Python 3.14 bug: temporary elision overwrote a live array once it reached 256 KiB,
which corrupted pandas unstack/pivot_table labels (bctc_features came back with 2 of 693 symbols). Fixed by numpy >= 2.3."""
import numpy as np
import pandas as pd


def _scale(n):
    a = np.arange(n, dtype=np.int64)
    _ = np.zeros(n, dtype=np.int64) + 5 * a + 1
    return int(a[1])


def test_large_array_expression_leaves_operand_intact():
    assert _scale(100_000) == 1


def test_unstack_keeps_row_labels_above_32768_cells():
    idx = pd.MultiIndex.from_product([[f"T{i}" for i in range(10)], [2000], [f"c{i}" for i in range(3300)]],
                                     names=["ticker", "year", "item"])
    wide = pd.Series(1.0, index=idx).unstack("item")
    assert wide.index.get_level_values("ticker").nunique() == 10
