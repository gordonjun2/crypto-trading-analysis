"""Walk-forward evaluation comparing three arms on identical test windows (plan §5).

Arms:
1. notebook     — in-sample selection + full-sample z look-ahead + same-bar execution,
                  no fees (faithful-but-runnable; quantifies the bias).
2. consolidated — train-only stats, rule-based ranking, next-bar execution, fees,
                  vol-balanced vs equal sizing compared.
3. consolidated+JEV — same engine, ranked by JEV composite.

Metrics: OOS Sharpe / maxDD / hit-rate (net of fees) per arm, Spearman between the
arm's train score and test Sharpe, precision@3, cointegration persistence.

Stated limitation (verbatim in output): ~62 days, one regime, one exchange, funding
rates not modeled, small candidate count per fold. Directional evidence about ranking
usefulness, not proof of profitability.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pair_scout.analysis.cointegration import engle_granger
from pair_scout.backtest.spread import (
    SimResult,
    simulate_consolidated,
    simulate_divergence,
    simulate_notebook,
)
from pair_scout.config import AppConfig
from pair_scout.data.loader import Panel, slice_days
from pair_scout.jev.client import JevClient, JevError
from pair_scout.jev.regime import entry_allowed_series
from pair_scout.jev.ranker import (
    Assessment,
    jev_assessment,
    rank,
    rule_based_assessment,
)
from pair_scout.screen import DEFAULT_BENCHMARK, screen_candidates

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairOutcome:
    key: str
    arm: str
    score: float
    test_sharpe: float
    test_return: float
    max_drawdown: float
    hit_rate: float
    n_trades: int
    train_pvalue: float
    test_pvalue: float
    mean_exposure: float = 1.0  # mean notional multiplier (1.0 = unscaled)


@dataclass
class FoldResult:
    fold: int
    arm: str
    mode: str = "cointegration"
    sizing: str = "vol_balanced"
    n_tested: int = 0
    top: list[PairOutcome] = field(default_factory=list)

    @property
    def mean_top_sharpe(self) -> float:
        if not self.top:
            return float("nan")
        return float(np.mean([o.test_sharpe for o in self.top]))

    @property
    def precision_at_k(self) -> float:
        if not self.top:
            return float("nan")
        return float(np.mean([1.0 if o.test_return > 0 else 0.0 for o in self.top]))


@dataclass
class EvaluationReport:
    generated_at: datetime
    folds: list[FoldResult] = field(default_factory=list)
    spearman: dict[str, float] = field(default_factory=dict)  # arm -> rho (pooled)
    persistence: float = float("nan")
    n_persistence: int = 0
    jev_error: str | None = None

    def to_markdown(self, cfg: AppConfig) -> str:
        lines = [
            "# PairScout walk-forward evaluation",
            "",
            f"Generated: {self.generated_at:%Y-%m-%d %H:%M} UTC",
            "",
            "## Per-fold results",
            "",
            "| Fold | Mode | Arm | Sizing | Tested | Mean Sharpe (top-K) | Precision@K | Best pick |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in self.folds:
            best = (
                f"{f.top[0].key} (SR {f.top[0].test_sharpe:.2f})" if f.top else "—"
            )
            lines.append(
                f"| {f.fold} | {f.mode} | {f.arm} | {f.sizing} | {f.n_tested} | "
                f"{f.mean_top_sharpe:.2f} | {f.precision_at_k:.2f} | {best} |"
            )
        lines += ["", "## Ranking quality (pooled across folds)", ""]
        for arm, rho in self.spearman.items():
            lines.append(f"- Spearman(score, test Sharpe) — {arm}: {rho:.3f}")
        if self.n_persistence:
            lines.append(
                f"- Cointegration persistence (train p<0.05 → test p<0.05): "
                f"{self.persistence:.0%} of {self.n_persistence}"
            )
        if self.jev_error:
            lines.append(f"- JEV arm skipped/unavailable: {self.jev_error}")
        lines += [
            "",
            "## Limitation",
            "",
            "~62 days, one regime, one exchange, funding rates of the short perp leg "
            "not modeled, small candidate count per fold. Results are directional "
            "evidence about ranking usefulness, **not** proof of profitability.",
            "",
        ]
        return "\n".join(lines)


def _simulate_divergence_result(
    panel_test: Panel, c, sizing: str, cfg: AppConfig,
    warmup_closes: dict[str, pd.Series] | None = None,
    entry_allowed: pd.Series | None = None,
    size_scale: pd.Series | None = None,
    funding: dict[str, pd.Series] | None = None,
) -> SimResult:
    """Raw divergence simulation for one candidate (shared by eval + research)."""
    bars_per_year = panel_test.bars_per_day * 365

    def leg_series(asset: str) -> pd.Series:
        test_close = panel_test.frames[asset]["Close"]
        if warmup_closes and asset in warmup_closes:
            return pd.concat([warmup_closes[asset], test_close])
        return test_close

    log_long = np.log(leg_series(c.asset_long))
    log_short = np.log(leg_series(c.asset_short))
    trade_start = (
        panel_test.index[0] if warmup_closes and c.asset_long in warmup_closes
        else None
    )
    window = max(int(cfg.backtest.signal_window_days * panel_test.bars_per_day), 2)
    # "vol_balanced" has no meaning without the spread convention here; the
    # risk-balanced sizing for a divergence book is beta_balanced.
    sizing_eff = "beta_balanced" if sizing == "vol_balanced" else sizing
    max_hold = (
        int(cfg.backtest.max_hold_days * panel_test.bars_per_day)
        if cfg.backtest.max_hold_days > 0 else None
    )
    f_long = f_short = None
    if cfg.backtest.include_funding and funding:
        from pair_scout.data.funding import funding_rate_series

        bars = log_long.index
        f_long = funding_rate_series(c.asset_long, funding, bars)
        f_short = funding_rate_series(c.asset_short, funding, bars)
    return simulate_divergence(
        log_long,
        log_short,
        c.beta_long,
        c.beta_short,
        sizing=sizing_eff,
        signal_window_bars=window,
        entry_mom_per_bar=cfg.backtest.divergence_entry_mom / window,
        exit_mom_per_bar=cfg.backtest.divergence_exit_mom / window,
        fee_bps=cfg.backtest.fee_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        bars_per_year=bars_per_year,
        max_hold_bars=max_hold,
        trade_start=trade_start,
        entry_mode=cfg.backtest.entry_mode,
        z_entry=cfg.backtest.z_entry,
        z_lookback_bars=max(
            int(cfg.backtest.z_lookback_days * panel_test.bars_per_day), 5
        ),
        momentum_shift_bars=int(
            cfg.backtest.momentum_shift_hours * panel_test.bars_per_day / 24
        ),
        spread_stop_pct=cfg.backtest.spread_stop_pct,
        entry_allowed=entry_allowed,
        cooldown_bars=int(
            cfg.backtest.reentry_cooldown_hours * panel_test.bars_per_day / 24
        ),
        size_scale=size_scale,
        funding_long=f_long,
        funding_short=f_short,
        funding_stress=cfg.backtest.funding_stress,
        spread_trail_pct=cfg.backtest.spread_trail_pct,
    )


def _simulate_candidate(
    panel_test: Panel, c, sizing: str, cfg: AppConfig, arm: str, score: float,
    warmup_closes: dict[str, pd.Series] | None = None,
    entry_allowed: pd.Series | None = None,
    size_scale: pd.Series | None = None,
    gates: dict[str, float] | None = None,
    funding: dict[str, pd.Series] | None = None,
) -> PairOutcome:
    """Simulate a screened candidate on the test panel, per its strategy."""
    bars_per_year = panel_test.bars_per_day * 365
    if c.strategy == "divergence":
        result = _simulate_divergence_result(
            panel_test, c, sizing, cfg, warmup_closes, entry_allowed,
            size_scale=size_scale
            if cfg.backtest.size_scaling in ("market", "jev_soft") else None,
            funding=funding,
        )
        if cfg.backtest.size_scaling in (
            "vol_target", "efficiency", "conviction", "combo"
        ):
            from pair_scout.sizing import apply_scale_to_result, scale_for_config

            scale = scale_for_config(
                panel_test, cfg, result.bar_returns.index,
                candidate_returns=result.bar_returns,
                candidate_signal=result.signal,
                gates=gates,
            )
            result = apply_scale_to_result(result, scale, bars_per_year)
        return PairOutcome(
            key=f"LONG {c.asset_long} / SHORT {c.asset_short}",
            arm=arm,
            score=score,
            test_sharpe=result.sharpe,
            test_return=result.total_return,
            max_drawdown=result.max_drawdown,
            hit_rate=result.hit_rate,
            n_trades=result.n_trades,
            train_pvalue=c.pvalue,
            test_pvalue=float("nan"),  # EG persistence is not a divergence metric
            mean_exposure=result.mean_scale,
        )

    y_a, x_a = c.spread_y_asset or c.asset_short, c.spread_x_asset or c.asset_long
    vol_of = {c.asset_long: c.annualized_vol_long, c.asset_short: c.annualized_vol_short}
    outcome = _simulate_on_test(
        panel_test, y_a, x_a, c.pvalue, c.hedge_ratio,
        vol_of.get(y_a, float("nan")), vol_of.get(x_a, float("nan")),
        arm, score, sizing, cfg,
    )
    return PairOutcome(
        key=f"LONG {c.asset_long} / SHORT {c.asset_short}",
        arm=outcome.arm,
        score=outcome.score,
        test_sharpe=outcome.test_sharpe,
        test_return=outcome.test_return,
        max_drawdown=outcome.max_drawdown,
        hit_rate=outcome.hit_rate,
        n_trades=outcome.n_trades,
        train_pvalue=outcome.train_pvalue,
        test_pvalue=outcome.test_pvalue,
    )


def _simulate_on_test(
    panel_test: Panel, y_asset: str, x_asset: str, pvalue_train: float,
    beta: float, vol_y: float, vol_x: float, arm: str, score: float,
    sizing: str, cfg: AppConfig,
) -> PairOutcome:
    """y_asset/x_asset follow the EG orientation: spread = log(Y) - beta*log(X)."""
    log_y = np.log(panel_test.frames[y_asset]["Close"])
    log_x = np.log(panel_test.frames[x_asset]["Close"])
    bars_per_year = panel_test.bars_per_day * 365
    if arm == "notebook":
        result = simulate_notebook(log_y, log_x, beta, bars_per_year=bars_per_year)
    else:
        result = simulate_consolidated(
            log_y, log_x, beta, vol_y, vol_x, sizing=sizing,
            zscore_window=cfg.backtest.zscore_window_bars,
            entry_z=cfg.backtest.entry_z, exit_z=cfg.backtest.exit_z,
            stop_z=cfg.backtest.stop_z, fee_bps=cfg.backtest.fee_bps,
            slippage_bps=cfg.backtest.slippage_bps,
            bars_per_year=bars_per_year,
        )
    test_pvalue = _test_pvalue(panel_test, y_asset, x_asset)
    return PairOutcome(
        key=f"LONG {x_asset} / SHORT {y_asset}",
        arm=arm,
        score=score,
        test_sharpe=result.sharpe,
        test_return=result.total_return,
        max_drawdown=result.max_drawdown,
        hit_rate=result.hit_rate,
        n_trades=result.n_trades,
        train_pvalue=pvalue_train,
        test_pvalue=test_pvalue,
    )


def _test_pvalue(panel: Panel, a: str, b: str) -> float:
    try:
        eg = engle_granger(a, b, panel.closes[a], panel.closes[b])
    except (ValueError, KeyError):
        return float("nan")
    return eg.pvalue


def run_evaluation(
    cfg: AppConfig,
    panel: Panel,
    use_jev: bool = True,
    jev_client: JevClient | None = None,
    regime_gates: dict[str, float] | None = None,
) -> EvaluationReport:
    report = EvaluationReport(generated_at=datetime.now(timezone.utc))
    total_days = panel.days
    # rolling walk-forward: fold starts advance by roll_days until the data ends
    folds: list[tuple[int, int, int]] = []
    t0 = cfg.eval.train_days
    while t0 + cfg.eval.test_days <= total_days:
        folds.append((1 + len(folds), 0, t0, t0, t0 + cfg.eval.test_days))
        t0 += cfg.eval.roll_days
    if not folds:
        logger.warning("panel too short (%d days) for walk-forward folds", total_days)
        return report

    jev_failed: str | None = None
    if use_jev and cfg.jev.enabled:
        try:
            jev_client = jev_client or JevClient(cfg)
        except JevError as exc:
            jev_failed, jev_client = str(exc), None

    spearman_data: dict[str, tuple[list[float], list[float]]] = {}
    persist_pass = persist_total = 0

    funding: dict[str, pd.Series] | None = None
    if cfg.backtest.include_funding:
        from pair_scout.data.funding import load_funding
        from pathlib import Path as _P

        fpath = _P(cfg.data.data_dir) / "funding_rates.json"
        funding = load_funding(fpath)
        if funding:
            logger.info(
                "funding modeled: %d symbols from %s (stress x%.1f)",
                len(funding), fpath, cfg.backtest.funding_stress,
            )
        else:
            logger.warning("funding cache missing at %s — funding NOT modeled", fpath)

    # dispersion-scaled sizing: per-bar multiplier from the FULL panel (causal:
    # each day's value uses only strictly-past data; see sizing.py)
    scale_full: pd.Series | None = None
    gates: dict[str, float] | None = None
    if cfg.backtest.size_scaling == "market":
        from pair_scout.sizing import market_daily_metrics, scale_for_config

        metrics = market_daily_metrics(panel)
        scale_full = scale_for_config(panel, cfg, panel.index, metrics=metrics)
        logger.info(
            "size scaling: %s/%s, mean exposure %.2f",
            cfg.backtest.size_scaling, cfg.backtest.scale_metric,
            float(scale_full.mean()),
        )
    elif cfg.backtest.size_scaling in ("jev_soft", "combo"):
        import json as _json
        from pathlib import Path as _Path

        from pair_scout.jev.regime import DEFAULT_CACHE
        from pair_scout.sizing import ScaleSpec, expand_to_bars, gates_scale_series

        if DEFAULT_CACHE.exists():
            gates = _json.loads(DEFAULT_CACHE.read_text())
        if gates:
            spec = ScaleSpec(
                mode="jev_soft",
                scale_min=cfg.backtest.scale_min,
                scale_max=cfg.backtest.scale_max,
            )
            scale_full = expand_to_bars(gates_scale_series(gates, panel.index, spec),
                                        panel.index)
            logger.info(
                "size scaling: %s, mean gate size %.2f (%d cached days)",
                cfg.backtest.size_scaling, float(scale_full.mean()), len(gates),
            )
        else:
            logger.warning(
                "size scaling %s requested but %s is missing — running unscaled",
                cfg.backtest.size_scaling, DEFAULT_CACHE,
            )

    for fold_no, s0, s1, t0, t1 in folds:
        train = slice_days(panel, s0, s1)
        test = slice_days(panel, t0, t1)
        logger.info(
            "fold %d: train %s→%s (%d pairs), test %s→%s (%d pairs)",
            fold_no, train.index[0].date(), train.index[-1].date(), len(train.pairs),
            test.index[0].date(), test.index[-1].date(), len(test.pairs),
        )
        screen = screen_candidates(train, cfg)
        if not screen.passed:
            logger.warning("fold %d: no candidates passed filters", fold_no)
            continue
        warmup_bars = max(
            int(cfg.backtest.z_lookback_days * panel.bars_per_day),
            int(cfg.backtest.signal_window_days * panel.bars_per_day)
            + int(cfg.backtest.momentum_shift_hours * panel.bars_per_day / 24)
            + 5,
        )
        warmup_closes = {
            p: train.frames[p]["Close"].tail(warmup_bars) for p in train.pairs
        }
        allowed = (
            entry_allowed_series(test, regime_gates, cfg.jev.regime_gate_min)
            if regime_gates
            else None
        )

        # Arm 1: notebook logic — in-sample selection p<0.01 on TEST window stats
        arm1_pairs = _notebook_arm_selection(test, cfg)
        outcomes1 = [
            _simulate_on_test(
                test, y_asset, x_asset, float("nan"), beta, float("nan"), float("nan"),
                "notebook", -p, "equal", cfg,
            )
            for (y_asset, x_asset, p, beta) in arm1_pairs
        ]
        outcomes1.sort(key=lambda o: o.score, reverse=True)
        report.folds.append(
            FoldResult(fold=fold_no, arm="1-notebook", mode=cfg.screen.mode,
                       sizing="equal",
                       n_tested=len(outcomes1), top=outcomes1[: cfg.eval.top_k])
        )
        _pool_spearman(spearman_data, f"1-notebook [{cfg.screen.mode}]", outcomes1)

        # Arm 2: consolidated, rule-based ranking, both sizings
        for sizing in ("vol_balanced", "equal"):
            outcomes2 = []
            for i, c in enumerate(screen.passed):
                outcome = _simulate_candidate(
                    test, c, sizing, cfg, "consolidated",
                    rule_based_assessment(c).composite,
                    warmup_closes=warmup_closes,
                    size_scale=scale_full,
                    gates=gates,
                    funding=funding,
                )
                c.test_pvalue_cache = outcome.test_pvalue
                outcomes2.append(outcome)
            outcomes2.sort(key=lambda o: o.score, reverse=True)
            report.folds.append(
                FoldResult(fold=fold_no, arm="2-consolidated", mode=cfg.screen.mode,
                           sizing=sizing,
                           n_tested=len(outcomes2), top=outcomes2[: cfg.eval.top_k])
            )
            if sizing == "vol_balanced":
                _pool_spearman(spearman_data, f"2-consolidated [{cfg.screen.mode}]", outcomes2)

        # persistence over scanned candidates (test p-values now cached)
        for c in screen.candidates:
            if c.pvalue < 0.05:
                persist_total += 1
                if c.test_pvalue_cache < 0.05:
                    persist_pass += 1

        # Arm 3: consolidated + JEV ranking
        if jev_client is not None:
            subset = screen.passed[: cfg.jev.max_candidates]
            try:
                assessments = _jev_rank_subset(jev_client, cfg, subset)
            except JevError as exc:
                logger.warning("JEV failed in fold %d: %s", fold_no, exc)
                jev_failed = str(exc)
                assessments = [rule_based_assessment(c) for c in subset]
            outcomes3 = []
            for a in assessments:
                c = a.candidate
                outcome = _simulate_candidate(test, c, "vol_balanced", cfg,
                                              "consolidated+jev", a.composite,
                                              warmup_closes=warmup_closes,
                                              entry_allowed=allowed,
                                              size_scale=scale_full,
                                              gates=gates,
                                              funding=funding)
                c.test_pvalue_cache = outcome.test_pvalue
                outcomes3.append(outcome)
            outcomes3.sort(key=lambda o: o.score, reverse=True)
            report.folds.append(
                FoldResult(fold=fold_no, arm="3-consolidated+jev", mode=cfg.screen.mode,
                           sizing="vol_balanced",
                           n_tested=len(outcomes3), top=outcomes3[: cfg.eval.top_k])
            )
            _pool_spearman(spearman_data, f"3-consolidated+jev [{cfg.screen.mode}]", outcomes3)

    for arm, (scores, sharpes) in spearman_data.items():
        if len(scores) >= 3:
            rho, _ = spearmanr(scores, sharpes)
            report.spearman[arm] = float(rho) if np.isfinite(rho) else float("nan")
        else:
            report.spearman[arm] = float("nan")
    report.persistence = (
        persist_pass / persist_total if persist_total else float("nan")
    )
    report.n_persistence = persist_total
    report.jev_error = jev_failed
    return report


def _pool_spearman(store: dict, arm: str, outcomes: list[PairOutcome]) -> None:
    scores, sharpes = store.get(arm, ([], []))
    for o in outcomes:
        if np.isfinite(o.test_sharpe):
            scores.append(o.score)
            sharpes.append(o.test_sharpe)
    store[arm] = (scores, sharpes)


def _notebook_arm_selection(panel: Panel, cfg: AppConfig) -> list[tuple[str, str, float, float]]:
    """In-sample selection on the traded window itself (the bias we're quantifying)."""
    from itertools import combinations

    closes = panel.closes
    picks = []
    for a, b in combinations(panel.pairs, 2):
        try:
            eg = engle_granger(a, b, closes[a], closes[b])
        except ValueError:
            continue
        if eg.pvalue < 0.01:
            picks.append((eg.asset_y, eg.asset_x, eg.pvalue, eg.hedge_ratio))
    picks.sort(key=lambda t: t[2])
    return picks[: cfg.eval.top_k * 2]


def _jev_rank_subset(
    client: JevClient, cfg: AppConfig, candidates: list
) -> list[Assessment]:
    def score_one(c):
        return jev_assessment(c, client.score_candidate(c), cfg.jev)

    with ThreadPoolExecutor(max_workers=cfg.jev.max_workers) as pool:
        assessments = list(pool.map(score_one, candidates))
    return rank(assessments, top_k=cfg.jev.top_k)
