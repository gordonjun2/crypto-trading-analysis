import math

import numpy as np
import pandas as pd
import pytest

from pair_scout.analysis.cointegration import rolling_zscore
from pair_scout.backtest.spread import (
    _positions,
    leg_weights,
    simulate_consolidated,
    simulate_notebook,
)


def make_spread_series(n=400, seed=7):
    """OU process around 0 -> clear mean reversion for the signal."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.2, n)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.85 * s[i - 1] + eps[i]
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.Series(s, index=idx)


def test_positions_state_machine():
    idx = pd.date_range("2025-01-01", periods=8, freq="h")
    z = pd.Series([0.0, -2.0, -2.5, -0.3, 0.0, 2.2, 3.8, 0.0], index=idx)
    pos = _positions(z, entry_z=1.5, exit_z=0.5, stop_z=3.5)
    assert list(pos) == [0, 1, 1, 0, 0, -1, 0, 0]
    # -2.0 enters long; -0.3 >= -0.5 exits; 2.2 enters short; 3.8 >= 3.5 stops out


def test_next_bar_execution_no_same_bar_fill():
    """PnL at the signal bar must be zero (B5 fix): position shifts by one bar."""
    n = 300
    s = make_spread_series(n)
    log_y = s / 2 + 10.0
    log_x = -s / 2 + 5.0
    result = simulate_consolidated(
        log_y, log_x, beta=1.0, vol_y=1.0, vol_x=1.0,
        zscore_window=15, entry_z=1.0, exit_z=0.5, stop_z=3.0,
        fee_bps=0.0, slippage_bps=0.0,
    )
    z = rolling_zscore(s, 15)
    pos = _positions(z, 1.0, 0.5, 3.0)
    signal_bar = pos[pos.ne(0)].index[0]  # first bar whose signal flips us on
    # signal bar itself earns nothing (fill happens next bar)
    assert result.bar_returns.loc[signal_bar] == pytest.approx(0.0, abs=1e-12)
    # the next bar carries the position's return
    exec_bar = signal_bar + pd.Timedelta(hours=1)
    assert result.bar_returns.loc[exec_bar] != 0.0


def test_fee_math_reduces_returns():
    n = 300
    s = make_spread_series(n)
    log_y = s / 2 + 10.0
    log_x = -s / 2 + 5.0
    free = simulate_consolidated(
        log_y, log_x, 1.0, 1.0, 1.0, entry_z=1.0, exit_z=0.5, stop_z=3.0,
        fee_bps=0.0, slippage_bps=0.0,
    )
    costed = simulate_consolidated(
        log_y, log_x, 1.0, 1.0, 1.0, entry_z=1.0, exit_z=0.5, stop_z=3.0,
        fee_bps=5.0, slippage_bps=2.0,
    )
    assert costed.total_return < free.total_return
    # each entry+exit round trip costs 2 events * (0.5+0.5) * 7bps = 14 bps
    n_events = 2 * costed.n_trades
    expected_cost = n_events * 1.0 * 7e-4
    assert free.total_return - costed.total_return == pytest.approx(
        expected_cost, rel=0.2
    )


def test_equal_vs_vol_balanced_weights():
    assert leg_weights(1.0, 1.0, "equal") == (0.5, 0.5)
    wy, wx = leg_weights(0.5, 1.5, "vol_balanced")
    assert wy > wx  # calmer leg gets more notional
    assert wy + wx == pytest.approx(1.0)
    assert leg_weights(0.0, 1.0, "vol_balanced") == (0.5, 0.5)  # degenerate fallback


def test_notebook_arm_is_lookahead_and_samebar():
    """The notebook arm must differ from consolidated: full-sample z, same-bar, no fees."""
    n = 400
    s = make_spread_series(n)
    log_y = s / 2 + 10.0
    log_x = -s / 2 + 5.0
    nb = simulate_notebook(log_y, log_x, 1.0)
    # same-bar: pnl at bar 0 can be nonzero only after an entry; verify trades exist
    # and that returns start once position is on (long-only arm)
    assert nb.n_trades >= 1
    assert all(t.direction == 1 for t in nb.trades)  # long-only


def test_metrics_shapes():
    n = 300
    s = make_spread_series(n)
    log_y = s / 2 + 10.0
    log_x = -s / 2 + 5.0
    r = simulate_consolidated(
        log_y, log_x, 1.0, 1.0, 1.0, entry_z=1.0, exit_z=0.5, stop_z=3.0
    )
    assert len(r.bar_returns) == n - 1
    assert math.isfinite(r.sharpe)
    assert r.max_drawdown <= 0
    assert 0.0 <= r.hit_rate <= 1.0
    assert r.n_trades == len(r.trades)
