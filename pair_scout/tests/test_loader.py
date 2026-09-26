"""Loader alignment tests: sparse union-index edges must not drop the panel.

Regression for the 2026-09-24 incident: two pairs fetched seconds apart ended
on different hourly bars, so the union index gained one extra leading bar and
649/651 pairs were dropped as "leading NaNs" (panel: 2 pairs).
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from pair_scout.data.loader import DataError, load_panel


def _write_pair(directory, pair, idx, px=100.0):
    n = len(idx)
    close = px * (1.0 + 0.001 * np.arange(n))
    df = pd.DataFrame({
        "Open Time": idx,
        "Close": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Volume": np.full(n, 1000.0),
    })
    with open(directory / f"{pair}_test.pkl", "wb") as fh:
        pickle.dump({"dataframe": df, "metadata": {"pair": pair}}, fh)


def _panel_dir(tmp_path):
    d = tmp_path / "binance" / "1h"
    d.mkdir(parents=True)
    return d


def test_leading_off_by_one_pair_keeps_panel(tmp_path):
    d = _panel_dir(tmp_path)
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    _write_pair(d, "AAAUSDT", idx)
    _write_pair(d, "BBBUSDT", idx[1:])  # starts one bar late (fast fetch)
    panel = load_panel(data_dir=tmp_path, top_n_volume=2, min_history_bars=50)
    assert set(panel.pairs) == {"AAAUSDT", "BBBUSDT"}


def test_trailing_off_by_one_pair_keeps_panel(tmp_path):
    d = _panel_dir(tmp_path)
    _write_pair(d, "AAAUSDT", pd.date_range("2025-01-01", periods=100, freq="h"))
    # BBB extends one bar past AAA (e.g. open candle seen by one fetch only)
    _write_pair(d, "BBBUSDT", pd.date_range("2025-01-01", periods=101, freq="h"))
    panel = load_panel(data_dir=tmp_path, top_n_volume=2, min_history_bars=50)
    assert set(panel.pairs) == {"AAAUSDT", "BBBUSDT"}
    assert panel.index[-1] == pd.Timestamp("2025-01-05 03:00:00")


def test_interior_gaps_still_drop_pair(tmp_path):
    d = _panel_dir(tmp_path)
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    _write_pair(d, "AAAUSDT", idx)
    _write_pair(d, "BBBUSDT", idx)
    gappy = idx.delete(np.arange(30, 60))  # 30% interior gap
    _write_pair(d, "GAPUSDT", gappy)
    panel = load_panel(data_dir=tmp_path, top_n_volume=3, min_history_bars=50)
    assert set(panel.pairs) == {"AAAUSDT", "BBBUSDT"}


def test_unrecoverable_union_raises(tmp_path):
    d = _panel_dir(tmp_path)
    a = pd.date_range("2025-01-01", periods=60, freq="h")
    b = pd.date_range("2025-03-01", periods=60, freq="h")  # disjoint from a
    _write_pair(d, "AAAUSDT", a)
    _write_pair(d, "BBBUSDT", b)
    with pytest.raises(DataError):
        load_panel(data_dir=tmp_path, top_n_volume=2, min_history_bars=50)
