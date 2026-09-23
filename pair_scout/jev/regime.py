"""JEV regime gate: P(cross-sectional momentum works now), as a daily Noul answer.

State = market-context metrics a panel of experts would weigh for a few-day-horizon
long-strong/short-weak book: bitcoin trend, volatility percentile, breadth,
dispersion of token returns, and average pairwise correlation. Answers are cached
per UTC date so backtests cost one request per day (once).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from typesafe_sdk import Noul

from pair_scout.config import AppConfig
from pair_scout.data.loader import Panel
from pair_scout.jev.client import JevClient, JevError

logger = logging.getLogger(__name__)

REGIME_QUESTION = Noul(
    criteria={
        "true": "The market state favors cross-sectional RELATIVE-VALUE trades: "
        "token returns are dispersed (clear leaders and laggards rather than "
        "everything moving together), correlations are moderate, and volatility is "
        "high enough to pay for the trade but not in crash territory. Recent "
        "strength/weakness tends to persist for a few more days.",
        "false": "The market state is hostile to relative-value momentum: everything "
        "is correlated and grinding in one direction (beta dominates), correlations "
        "are extreme, volatility is in a violent crash regime, or choppy "
        "mean-reversion is whipping recent leaders and laggards around.",
    }
)

DEFAULT_CACHE = Path("pair_scout/output/regime_cache.json")


def market_state(panel: Panel, asof: pd.Timestamp) -> dict:
    """Market-context metrics using only data up to `asof` (decision-time safe)."""
    closes = panel.closes.loc[:asof]
    if len(closes) < panel.bars_per_day * 10:
        raise ValueError("not enough history for a regime state")
    btc = closes["BTCUSDT"] if "BTCUSDT" in closes.columns else closes.iloc[:, 0]
    day = panel.bars_per_day

    trend_7d = float(btc.iloc[-1] / btc.iloc[-7 * day] - 1.0)
    rets = np.log(btc).diff().dropna()
    vol7 = float(rets.tail(7 * day).std() * np.sqrt(panel.bars_per_day * 365))
    vol_history = rets.rolling(7 * day).std().dropna() * np.sqrt(panel.bars_per_day * 365)
    vol_pctile = float((vol_history < vol7).mean()) if len(vol_history) > 20 else 0.5

    moms = {
        p: float(closes[p].iloc[-1] / closes[p].iloc[-7 * day] - 1.0)
        for p in closes.columns
        if len(closes[p].dropna()) > 7 * day
    }
    mom_values = np.array(list(moms.values()))
    breadth = float((mom_values > 0).mean())
    dispersion = float(mom_values.std())

    sample = closes.tail(3 * day)
    corr = sample.pct_change().dropna().corr()
    n = len(corr)
    avg_corr = float((corr.to_numpy().sum() - n) / max(n * (n - 1), 1))

    return {
        "as_of": str(pd.Timestamp(asof).date()),
        "btc_trend_7d": round(trend_7d, 4),
        "btc_annualized_vol_7d": round(vol7, 2),
        "btc_vol_percentile_90d": round(vol_pctile, 2),
        "universe_breadth_7d": round(breadth, 2),
        "momentum_dispersion_7d": round(dispersion, 3),
        "avg_pairwise_corr_3d": round(avg_corr, 2),
        "universe_size": len(moms),
    }


STATE_REFERENCE = {
    "btc_trend_7d": "Bitcoin's trailing 7-day return; everything-up or "
    "everything-down markets make relative-value trades beta bets.",
    "btc_annualized_vol_7d / percentile": "Realized bitcoin volatility and where it "
    "sits vs the last 90 days; extremes are hostile.",
    "universe_breadth_7d": "Fraction of the universe with positive 7d returns; "
    "near 0.5 means a dispersed market with distinct leaders/laggards (good), "
    "near 0 or 1 means a single macro move (bad for relative value).",
    "momentum_dispersion_7d": "Cross-sectional std of 7d returns; higher = more "
    "raw material for long-strong/short-weak books.",
    "avg_pairwise_corr_3d": "Mean pairwise return correlation; >0.9 means the "
    "universe moves as one asset.",
}


def _prob_from_response(response) -> float:
    a = response.answers.get("regime_gate")
    if a is None:
        raise JevError("regime_gate answer missing")
    return float(a.noul)


def daily_regime_gates(
    cfg: AppConfig,
    panel: Panel,
    client: JevClient,
    cache_path: Path = DEFAULT_CACHE,
    step_days: int = 1,
) -> dict[str, float]:
    """P(momentum regime) per UTC date, computed once and cached on disk."""
    cache: dict[str, float] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    day = panel.bars_per_day
    start = cfg.eval.train_days * day
    stamps = list(panel.index[start :: step_days * day])
    todo = [t for t in stamps if str(t.date()) not in cache]
    logger.info(
        "regime gate: %d days to score (%d cached)", len(todo), len(stamps) - len(todo)
    )

    def score(t: pd.Timestamp) -> tuple[str, float]:
        state = market_state(panel, t)
        state["metric_reference"] = STATE_REFERENCE
        response = client._client.system_one(
            state=state, questions={"regime_gate": REGIME_QUESTION}
        )
        return str(t.date()), _prob_from_response(response)

    if todo:
        with ThreadPoolExecutor(max_workers=cfg.jev.max_workers) as pool:
            for date, prob in pool.map(score, todo):
                cache[date] = prob
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=0))
    return cache


def entry_allowed_series(
    panel: Panel, gates: dict[str, float], min_prob: float, shift_days: int = 1
) -> pd.Series:
    """Boolean per-bar entry permission from the PREVIOUS day's gate probability."""
    dates = pd.Series(panel.index.floor("D"), index=panel.index) - pd.Timedelta(
        days=shift_days
    )
    probs = dates.map(lambda d: gates.get(str(d.date())))
    return probs.map(lambda p: bool(p is not None and p >= min_prob)).fillna(False)


def live_regime_probability(cfg: AppConfig, panel: Panel, client: JevClient) -> float:
    """One-shot gate for the live screener (state at the last closed bar)."""
    state = market_state(panel, panel.index[-1])
    state["metric_reference"] = STATE_REFERENCE
    response = client._client.system_one(
        state=state, questions={"regime_gate": REGIME_QUESTION}
    )
    return _prob_from_response(response)
