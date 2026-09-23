import numpy as np
import pytest

from pair_scout.config import JevConfig, ScreenConfig
from pair_scout.features import PairCandidate
from pair_scout.filters import apply_filters
from pair_scout.jev.client import JevAnswerSet
from pair_scout.jev.ranker import (
    Assessment,
    jev_assessment,
    normalize,
    rank,
    rule_based_assessment,
    rule_based_score,
)
from typesafe_sdk import ScoreAnswer, NoulAnswer


def make_candidate(**overrides) -> PairCandidate:
    base = dict(
        asset_long="AAAUSDT",
        asset_short="BBBUSDT",
        direction_basis="z-score sign",
        strategy="cointegration",
        pvalue=0.004,
        adf_stat=-4.0,
        hedge_ratio=1.5,
        spread_zscore=2.2,
        half_life_bars=120.0,
        half_life_days=5.0,
        hurst=0.35,
        return_correlation=0.6,
        momentum_long=0.05,
        momentum_short=-0.03,
        avg_daily_volume_long=50_000_000.0,
        avg_daily_volume_short=40_000_000.0,
        annualized_vol_long=0.6,
        annualized_vol_short=0.9,
        atr_pct_long=2.0,
        atr_pct_short=3.0,
        vol_ratio=1.5,
        skewness_long=-0.3,
        skewness_short=-0.5,
        long_notional_share=0.6,
        history_bars=1500,
        history_days=62,
        train_end="2025-08-01",
    )
    base.update(overrides)
    return PairCandidate(**base)


def make_answers(rev=2.8, entry=2.2, exec_=1.7, conf=0.8, direction=0.9, red=0.1):
    def score(v, top):
        return ScoreAnswer(
            type="score", score=v, confidence=conf, legend={}, probabilities={}
        )

    return JevAnswerSet(
        reversion=score(rev, 3),
        entry=score(entry, 3),
        executability=score(exec_, 2),
        direction_noul=direction,
        red_flag_noul=red,
        model="jev-test",
    )


class TestFilters:
    def test_all_pass(self):
        c = make_candidate()
        outcome = apply_filters(c, ScreenConfig())
        assert outcome.passed
        assert outcome.reasons == []
        assert outcome.codes == []

    def test_codes_recorded_alongside_reasons(self):
        c = make_candidate(pvalue=0.2, vol_ratio=9.0)
        outcome = apply_filters(c, ScreenConfig())
        assert outcome.codes == ["pvalue", "vol_ratio"]

    def test_watch_tier_detection(self):
        from pair_scout.filters import is_watch_tier

        cfg = ScreenConfig()
        near = make_candidate(spread_zscore=1.2)  # only entry_z fails
        near.filter_codes = ["entry_z"]
        assert is_watch_tier(near, cfg)
        two_fail = make_candidate(spread_zscore=1.2, vol_ratio=9.0)
        two_fail.filter_codes = ["entry_z", "vol_ratio"]
        assert not is_watch_tier(two_fail, cfg)
        too_low = make_candidate(spread_zscore=0.6)
        too_low.filter_codes = ["entry_z"]
        assert not is_watch_tier(too_low, cfg)  # below watch_z

    def test_watch_tier_never_enters(self):
        from pair_scout.jev.ranker import jev_assessment as ja

        cfg = JevConfig()
        c = make_candidate(spread_zscore=1.2, watch_tier=True)
        a = ja(c, make_answers(), cfg)
        assert a.action == "WATCH"
        assert "near-miss" in a.reason

    @pytest.mark.parametrize(
        "override, needle",
        [
            ({"pvalue": 0.2}, "p="),
            ({"half_life_bars": 2.0, "half_life_days": 0.08}, "half-life"),
            ({"half_life_days": 45.0, "half_life_bars": 1080.0}, "half-life"),
            ({"spread_zscore": 0.8}, "|z|"),
            ({"avg_daily_volume_short": 100.0}, "volume"),
            ({"atr_pct_short": 30.0}, "ATR"),
            ({"skewness_short": -9.0}, "skew"),
            ({"vol_ratio": 9.0}, "vol_ratio"),
            ({"hedge_ratio": -1.0}, "hedge"),
        ],
    )
    def test_each_rule_blocks_with_reason(self, override, needle):
        c = make_candidate(**override)
        outcome = apply_filters(c, ScreenConfig())
        assert not outcome.passed
        assert any(needle in r for r in outcome.reasons)

    def test_nan_pvalue_blocks(self):
        c = make_candidate(pvalue=float("nan"))
        assert not apply_filters(c, ScreenConfig()).passed


class TestNormalize:
    def test_basic(self):
        assert normalize(3.0, 3) == 1.0
        assert normalize(0.0, 3) == 0.0
        assert normalize(1.5, 3) == 0.5

    def test_clamps(self):
        assert normalize(5.0, 3) == 1.0
        assert normalize(-1.0, 3) == 0.0

    def test_invalid_top_level(self):
        with pytest.raises(ValueError):
            normalize(1.0, 0)


class TestRanker:
    def test_composite_weighting(self):
        cfg = JevConfig()
        answers = make_answers(rev=3.0, entry=0.0, exec_=0.0)
        # 0.45*1 + 0.35*0 + 0.20*0
        from pair_scout.jev.ranker import composite_score

        assert composite_score(answers, cfg) == pytest.approx(0.45)

    def test_enter_when_all_gates_pass(self):
        cfg = JevConfig()
        a = jev_assessment(make_candidate(), make_answers(), cfg)
        assert a.action == "ENTER"

    @pytest.mark.parametrize(
        "kwargs, needle",
        [
            (dict(conf=0.1), "confidence"),
            (dict(direction=0.3), "direction"),
            (dict(red=0.9), "red flag"),
            (dict(rev=0.5, entry=0.5, exec_=0.5), "composite"),
        ],
    )
    def test_each_gate_fails_to_watch(self, kwargs, needle):
        cfg = JevConfig()
        a = jev_assessment(make_candidate(), make_answers(**kwargs), cfg)
        assert a.action == "WATCH"
        assert needle in a.reason

    def test_rank_orders_and_caps_top_k(self):
        cfg = JevConfig()
        cands = [make_candidate(asset_long=f"C{i}USDT") for i in range(6)]
        answers = [make_answers(rev=3 - 0.3 * i) for i in range(6)]
        assessments = [jev_assessment(c, a, cfg) for c, a in zip(cands, answers)]
        ranked = rank(assessments, top_k=2)
        enters = [a for a in ranked if a.action == "ENTER"]
        assert len(enters) == 2
        comps = [a.composite for a in ranked]
        assert comps == sorted(comps, reverse=True)
        demoted = [a for a in ranked if "top-K" in a.reason]
        assert len(demoted) == 4

    def test_rule_based_score_orders_reasonably(self):
        good = make_candidate(pvalue=0.001, spread_zscore=2.8, vol_ratio=1.1)
        bad = make_candidate(pvalue=0.045, spread_zscore=1.6, vol_ratio=3.5)
        assert rule_based_score(good) > rule_based_score(bad)


class TestDegraded:
    def test_rule_based_assessment_never_enters(self):
        a = rule_based_assessment(make_candidate())
        assert a.action == "WATCH"
        assert a.jev_used is False
