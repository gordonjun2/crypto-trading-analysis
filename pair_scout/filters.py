"""Hard rule filters (plan §3 step 5) — retained regardless of JEV.

Every failure records a stable code (for aggregate reporting) and a
human-readable reason with the actual values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pair_scout.config import ScreenConfig
from pair_scout.features import PairCandidate

CODE_LABELS = {
    "history": "history too short",
    "pvalue": "cointegration p-value above cap",
    "half_life": "half-life outside tradeable range",
    "entry_z": "|z| below entry threshold",
    "mom_spread": "momentum divergence below minimum",
    "flat": "a leg is too flat (low volatility) to pay fees",
    "corr": "leg returns too correlated for a divergence trade",
    "beta": "leg BTC beta too weak for beta balancing",
    "combo_beta": "residual market beta outside cap",
    "volume": "dollar volume below minimum",
    "atr": "ATR% too high on a leg",
    "skew": "|skew| extreme on a leg",
    "vol_ratio": "one leg dominates risk (vol_ratio)",
    "hedge": "degenerate hedge ratio",
}

ENTRY_CODES = {"entry_z", "mom_spread"}  # strategy-specific "actionable setup" codes


@dataclass(frozen=True)
class FilterOutcome:
    passed: bool
    reasons: list[str]
    codes: list[str]


def _bad(value: float) -> bool:
    return not math.isfinite(value)


def apply_filters(c: PairCandidate, cfg: ScreenConfig) -> FilterOutcome:
    reasons: list[str] = []
    codes: list[str] = []

    def fail(code: str, reason: str) -> None:
        codes.append(code)
        reasons.append(reason)

    if c.history_bars < 100:
        fail("history", f"history too short ({c.history_bars} bars)")

    if c.strategy == "divergence":
        if (
            _bad(c.annualized_vol_long)
            or _bad(c.annualized_vol_short)
            or min(c.annualized_vol_long, c.annualized_vol_short) < cfg.min_leg_ann_vol
        ):
            fail(
                "flat",
                f"leg too flat ({c.annualized_vol_long:.0%}/{c.annualized_vol_short:.0%} "
                f"ann vol < {cfg.min_leg_ann_vol:.0%})",
            )
        if _bad(c.momentum_spread) or c.momentum_spread < cfg.min_momentum_spread:
            fail(
                "mom_spread",
                f"momentum spread {c.momentum_spread:+.1%} < {cfg.min_momentum_spread:.0%}",
            )
        if _bad(c.return_correlation) or c.return_correlation > cfg.max_return_corr:
            fail(
                "corr",
                f"return corr {c.return_correlation:.2f} > {cfg.max_return_corr}",
            )
        if _bad(c.beta_long) or _bad(c.beta_short) or min(
            c.beta_long, c.beta_short
        ) < cfg.beta_min:
            fail(
                "beta",
                f"leg BTC beta too weak ({c.beta_long:.2f}/{c.beta_short:.2f} "
                f"< {cfg.beta_min})",
            )
        if _bad(c.combo_beta) or abs(c.combo_beta) > cfg.combo_beta_max:
            fail(
                "combo_beta",
                f"combo beta {c.combo_beta:+.2f} outside ±{cfg.combo_beta_max}",
            )
    else:
        if _bad(c.pvalue) or c.pvalue > cfg.pvalue_max:
            fail("pvalue", f"cointegration p={c.pvalue:.3f} > {cfg.pvalue_max}")
        if _bad(c.half_life_bars) or c.half_life_bars < cfg.half_life_min_bars:
            fail("half_life", f"half-life {c.half_life_bars:.1f} bars < {cfg.half_life_min_bars}")
        elif c.half_life_days > cfg.half_life_max_days:
            fail("half_life", f"half-life {c.half_life_days:.1f}d > {cfg.half_life_max_days}d")
        if _bad(c.spread_zscore) or abs(c.spread_zscore) < cfg.entry_z:
            fail("entry_z", f"|z|={abs(c.spread_zscore):.2f} < entry {cfg.entry_z}")
        if _bad(c.hedge_ratio) or c.hedge_ratio <= 0:
            fail("hedge", f"degenerate hedge ratio ({c.hedge_ratio})")

    if c.avg_daily_volume_usd < cfg.min_dollar_volume:
        fail(
            "volume",
            f"volume ${c.avg_daily_volume_usd:,.0f}/d < ${cfg.min_dollar_volume:,.0f}",
        )
    if (
        _bad(c.atr_pct_long)
        or _bad(c.atr_pct_short)
        or max(c.atr_pct_long, c.atr_pct_short) > cfg.atr_pct_max
    ):
        fail(
            "atr",
            f"ATR% too high ({c.atr_pct_long:.1f}/{c.atr_pct_short:.1f} > {cfg.atr_pct_max})",
        )
    if (
        _bad(c.skewness_long)
        or _bad(c.skewness_short)
        or max(abs(c.skewness_long), abs(c.skewness_short)) > cfg.skew_abs_max
    ):
        fail(
            "skew",
            f"|skew| extreme ({c.skewness_long:.2f}/{c.skewness_short:.2f} > {cfg.skew_abs_max})",
        )
    if _bad(c.vol_ratio) or c.vol_ratio > cfg.vol_ratio_max:
        fail("vol_ratio", f"vol_ratio {c.vol_ratio:.1f} > {cfg.vol_ratio_max}")
    return FilterOutcome(passed=not reasons, reasons=reasons, codes=codes)


def is_watch_tier(c: PairCandidate, cfg: ScreenConfig) -> bool:
    """True when the ONLY failing rule is the actionable-setup rule and it near-missed."""
    if c.watch_tier:
        return True
    if len(c.filter_codes) != 1 or not ENTRY_CODES.intersection(c.filter_codes):
        return False
    if "entry_z" in c.filter_codes:
        return math.isfinite(c.spread_zscore) and abs(c.spread_zscore) >= cfg.watch_z
    # mom_spread near-miss: 75% of the required divergence
    return math.isfinite(c.momentum_spread) and (
        c.momentum_spread >= 0.75 * cfg.min_momentum_spread
    )
