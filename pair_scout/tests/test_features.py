import math

import pytest

from pair_scout.features import METRIC_REFERENCE, PairCandidate, build_jev_state
from pair_scout.filters import apply_filters
from pair_scout.config import ScreenConfig

from .test_ranker import make_candidate


def test_metric_reference_present_in_state():
    state = build_jev_state(make_candidate())
    assert set(state["pair"]) == {"asset_long", "asset_short", "direction_basis"}
    assert len(METRIC_REFERENCE) >= 8
    assert "metric_reference" in state
    assert "cointegration_pvalue" in METRIC_REFERENCE
    # a few key metrics are covered by a reference entry
    joined = " ".join(METRIC_REFERENCE.values()).lower()
    for token in ("p-value", "half", "mean-reverting", "std devs", "skew"):
        assert token in joined


def test_state_values_rounded():
    c = make_candidate(pvalue=0.00356789, hedge_ratio=1.23456789)
    state = build_jev_state(c)
    assert state["metrics"]["pvalue"] if "pvalue" in state["metrics"] else True
    assert state["metrics"]["cointegration_pvalue"] == 0.0036
    assert state["metrics"]["hedge_ratio"] == 1.2346


def test_nan_metrics_do_not_crash_filters():
    c = make_candidate(
        pvalue=float("nan"),
        half_life_bars=float("nan"),
        half_life_days=float("nan"),
        hurst=float("nan"),
        spread_zscore=float("nan"),
        vol_ratio=float("nan"),
    )
    outcome = apply_filters(c, ScreenConfig())
    assert not outcome.passed
    assert len(outcome.reasons) >= 4


def test_inf_vol_ratio_flagged():
    c = make_candidate(vol_ratio=float("inf"))
    outcome = apply_filters(c, ScreenConfig())
    assert any("vol_ratio" in r for r in outcome.reasons)


def test_candidate_key_and_volume():
    c = make_candidate()
    assert c.key == "AAAUSDT/BBBUSDT"
    assert c.avg_daily_volume_usd == 40_000_000.0
