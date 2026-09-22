"""PairCandidate dataclass — one row of everything we know about a candidate pair."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PairCandidate:
    """All metrics are computed on the screening (train) window only (M1 fix)."""

    asset_long: str
    asset_short: str
    direction_basis: str
    # cointegration
    pvalue: float
    adf_stat: float
    hedge_ratio: float
    spread_zscore: float
    half_life_bars: float
    half_life_days: float
    hurst: float
    # correlation / divergence
    return_correlation: float
    momentum_long: float
    momentum_short: float
    # liquidity / volatility / risk
    avg_daily_volume_long: float
    avg_daily_volume_short: float
    annualized_vol_long: float
    annualized_vol_short: float
    atr_pct_long: float
    atr_pct_short: float
    vol_ratio: float
    skewness_long: float
    skewness_short: float
    long_notional_share: float
    choppiness_long: float = float("nan")
    choppiness_short: float = float("nan")
    beta_long: float = float("nan")
    beta_short: float = float("nan")
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
        "Trailing return of each leg; for divergence setups we long the stronger and "
        "short the weaker asset."
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
                "history_days": c.history_days,
            }
        ),
        "metric_reference": METRIC_REFERENCE,
    }


def _round_floats(obj: dict, ndigits: int = 4) -> dict:
    out = {}
    for k, v in obj.items():
        if isinstance(v, float):
            out[k] = round(v, ndigits)
        else:
            out[k] = v
    return out
