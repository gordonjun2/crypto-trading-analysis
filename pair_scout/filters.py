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
    "volume": "dollar volume below minimum",
    "atr": "ATR% too high on a leg",
    "skew": "|skew| extreme on a leg",
    "vol_ratio": "one leg dominates risk (vol_ratio)",
    "hedge": "degenerate hedge ratio",
}


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
    if _bad(c.pvalue) or c.pvalue > cfg.pvalue_max:
        fail("pvalue", f"cointegration p={c.pvalue:.3f} > {cfg.pvalue_max}")
    if _bad(c.half_life_bars) or c.half_life_bars < cfg.half_life_min_bars:
        fail("half_life", f"half-life {c.half_life_bars:.1f} bars < {cfg.half_life_min_bars}")
    elif c.half_life_days > cfg.half_life_max_days:
        fail("half_life", f"half-life {c.half_life_days:.1f}d > {cfg.half_life_max_days}d")
    if _bad(c.spread_zscore) or abs(c.spread_zscore) < cfg.entry_z:
        fail("entry_z", f"|z|={abs(c.spread_zscore):.2f} < entry {cfg.entry_z}")
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
    if _bad(c.hedge_ratio) or c.hedge_ratio <= 0:
        fail("hedge", f"degenerate hedge ratio ({c.hedge_ratio})")
    return FilterOutcome(passed=not reasons, reasons=reasons, codes=codes)


def is_watch_tier(c: PairCandidate, cfg: ScreenConfig) -> bool:
    """True when the ONLY failing rule is entry_z and |z| still clears watch_z."""
    if c.watch_tier:
        return True
    if set(c.filter_codes) != {"entry_z"}:
        return False
    return math.isfinite(c.spread_zscore) and abs(c.spread_zscore) >= cfg.watch_z
