"""PairCandidate dataclass — one row of everything we know about a candidate pair."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PairCandidate:
    """All metrics are computed on the screening (train) window only (M1 fix)."""

    asset_long: str
    asset_short: str
    direction_basis: str
    strategy: str = "cointegration"  # "cointegration" | "divergence"
    # cointegration (NaN for divergence candidates)
    pvalue: float = float("nan")
    adf_stat: float = float("nan")
    hedge_ratio: float = float("nan")
    spread_zscore: float = float("nan")
    half_life_bars: float = float("nan")
    half_life_days: float = float("nan")
    hurst: float = float("nan")
    # correlation / divergence
    return_correlation: float = float("nan")
    momentum_long: float = float("nan")
    momentum_short: float = float("nan")
    # liquidity / volatility / risk
    avg_daily_volume_long: float = float("nan")
    avg_daily_volume_short: float = float("nan")
    annualized_vol_long: float = float("nan")
    annualized_vol_short: float = float("nan")
    atr_pct_long: float = float("nan")
    atr_pct_short: float = float("nan")
    vol_ratio: float = float("nan")
    skewness_long: float = float("nan")
    skewness_short: float = float("nan")
    long_notional_share: float = float("nan")
    choppiness_long: float = float("nan")
    choppiness_short: float = float("nan")
    beta_long: float = float("nan")
    beta_short: float = float("nan")
    momentum_spread: float = float("nan")  # rank-window momentum, long minus short
    signal_momentum: float = float("nan")  # signal-window momentum of the pair spread
    weight_long: float = float("nan")  # sizing weight of the long leg (sums to 1)
    combo_beta: float = float("nan")  # BTC beta of the balanced long-short book
    history_bars: int = 0
    history_days: int = 0
    # outcome fields
    filter_reasons: list[str] = field(default_factory=list)
    filter_codes: list[str] = field(default_factory=list)
    watch_tier: bool = False  # near-miss: only entry_z failed, |z| >= watch_z
    train_end: str = ""
    test_pvalue_cache: float = float("nan")  # filled during evaluation
    # EG orientation actually used: spread = log(y) - hedge_ratio * log(x)
    spread_y_asset: str = ""
    spread_x_asset: str = ""

    @property
    def key(self) -> str:
        return f"{self.asset_long}/{self.asset_short}"

    @property
    def avg_daily_volume_usd(self) -> float:
        return min(self.avg_daily_volume_long, self.avg_daily_volume_short)


METRIC_REFERENCE: dict[str, str] = {
    "cointegration_pvalue": (
        "Engle-Granger p-value; lower = stronger evidence the spread is stationary "
        "and mean-reverts. <0.01 strong, <0.05 acceptable, >0.10 weak."
    ),
    "hedge_ratio": (
        "OLS slope relating log prices of the two legs; the spread is log(short-leg) "
        "minus hedge_ratio times log(long-leg)."
    ),
    "spread_half_life_days": (
        "Days for a spread shock to decay halfway. 0.2-15 days is tradeable; >30 too "
        "slow; <0.2 mostly noise/fees."
    ),
    "hurst_exponent": (
        "<0.5 mean-reverting spread, >0.5 trending. Lower is better for this strategy."
    ),
    "current_spread_zscore": (
        "How many std devs the spread is from its rolling mean. |z|>=1.5 is the "
        "typical entry zone; entry near 0 has no edge."
    ),
    "return_correlation_90d": (
        "Correlation of daily-style log returns between the legs; low or negative "
        "correlation with cointegration means the link is a stable relationship, not "
        "co-movement with the market."
    ),
    "momentum": (
        "Trailing return of each leg. For divergence setups we long the stronger "
        "token and short the weaker one; momentum_spread is that gap over the "
        "ranking window and signal_momentum is the pair spread's recent momentum."
    ),
    "beta_vs_btc / combo_beta": (
        "Sensitivity of each leg to bitcoin. The leg weights are chosen so the "
        "long-short book's combined beta (combo_beta) is ~0: the trade is a bet on "
        "relative strength, not on market direction. |combo_beta| near 0 is best."
    ),
    "avg_daily_volume_usd": (
        "Average quote-volume per day over the trailing week; lower bound for "
        "realistic execution."
    ),
    "annualized_vol / atr_pct / vol_ratio": (
        "Per-leg volatility and the ratio between legs. A vol_ratio far above 1 means "
        "one leg dominates risk; very high ATR% means wider stops and higher "
        "execution cost."
    ),
    "skewness": (
        "Negative skew = fatter left tail (crash risk) on that leg; large negative "
        "skew on the short leg is a warning sign."
    ),
    "history_days": (
        "Length of clean overlapping history; short history means weaker statistical "
        "evidence."
    ),
}


def build_jev_state(c: PairCandidate) -> dict:
    """JSON state for one JEV request (plan §4.2)."""
    return {
        "strategy": (
            "momentum divergence: long the stronger token, short the weaker one, "
            "leg weights beta-balanced so the book has ~zero bitcoin exposure"
            if c.strategy == "divergence"
            else "statistical mean reversion: short the rich leg, long the cheap leg "
            "when the cointegrated spread is stretched"
        ),
        "pair": {
            "asset_long": c.asset_long,
            "asset_short": c.asset_short,
            "direction_basis": c.direction_basis,
        },
        "metrics": _round_floats(
            {
                "cointegration_pvalue": c.pvalue,
                "adf_statistic": c.adf_stat,
                "hedge_ratio": c.hedge_ratio,
                "spread_half_life_days": c.half_life_days,
                "hurst_exponent": c.hurst,
                "current_spread_zscore": c.spread_zscore,
                "return_correlation_90d": c.return_correlation,
                "momentum_30d_long": c.momentum_long,
                "momentum_30d_short": c.momentum_short,
                "momentum_spread_rank_window": c.momentum_spread,
                "signal_momentum": c.signal_momentum,
                "avg_daily_volume_long": c.avg_daily_volume_long,
                "avg_daily_volume_short": c.avg_daily_volume_short,
                "annualized_vol_long": c.annualized_vol_long,
                "annualized_vol_short": c.annualized_vol_short,
                "atr_pct_long": c.atr_pct_long,
                "atr_pct_short": c.atr_pct_short,
                "vol_ratio": c.vol_ratio,
                "skewness_long": c.skewness_long,
                "skewness_short": c.skewness_short,
                "beta_vs_btc_long": c.beta_long,
                "beta_vs_btc_short": c.beta_short,
                "weight_long": c.weight_long,
                "combo_beta": c.combo_beta,
                "history_days": c.history_days,
            }
        ),
        "metric_reference": METRIC_REFERENCE,
    }


def _round_floats(obj: dict, ndigits: int = 4) -> dict:
    out = {}
    for k, v in obj.items():
        if isinstance(v, float):
            # JSON has no NaN; non-finite metrics are simply omitted from state
            out[k] = round(v, ndigits) if math.isfinite(v) else None
        else:
            out[k] = v
    return out
