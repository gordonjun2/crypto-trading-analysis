"""TA family lab: scan alternative price-action signal families under the
pass-10 protocol (point-in-time universe, continuous sim, 3 slots x 1/3 book,
<=7d hold, 5% stop, 7 bps/side fees, real funding, next-bar fills).

Each family maps 4h OHLCV to per-pair (long_active, long_exit, short_active,
short_exit) boolean states on the 1h index; entries fire on rising edges of
the active states, exits on the exit states / time stop / adverse stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pair_scout.config import load_config
from pair_scout.data.funding import funding_rate_series, load_funding
from pair_scout.data.loader import load_panel
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

logger = logging.getLogger(__name__)

FEE = 7e-4
BTC = "BTCUSDT"
SLOT = 1.0 / 3.0
MAX_HOLD_BARS = 7 * 24
STOP = 0.05
BPD4 = 6  # 4h bars per day


@dataclass
class Trade:
    sig_sym: str
    sig_dir: int
    entry_i: int = 0
    exit_i: int = -1
    entry_px: float = float("nan")
    exit_px: float = float("nan")
    reason: str = ""
    exit_ts: pd.Timestamp = None
    entry_ts: pd.Timestamp = None


Signals = dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]]


def _resample_ohlcv(panel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h = pd.DataFrame({s: panel.frames[s]["High"] for s in panel.pairs}).resample("4h").max()
    l = pd.DataFrame({s: panel.frames[s]["Low"] for s in panel.pairs}).resample("4h").min()
    c = pd.DataFrame({s: panel.frames[s]["Close"] for s in panel.pairs}).resample("4h").last()
    v = pd.DataFrame({s: panel.frames[s]["Volume"] for s in panel.pairs}).resample("4h").sum()
    return h, l, c, v


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _atr(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, n: int) -> pd.DataFrame:
    pc = c.shift(1)
    tr = pd.DataFrame(
        np.maximum.reduce([(h - l).abs(), (h - pc).abs(), (l - pc).abs()]),
        index=h.index, columns=h.columns,
    )
    return _wilder_colwise(tr, n)


def _wilder_colwise(df: pd.DataFrame, n: int) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        s = df[col]
        if s.notna().sum() < n * 3:
            out[col] = pd.Series(np.nan, index=s.index)
            continue
        out[col] = _wilder(s.dropna(), n).reindex(s.index)
    return pd.DataFrame(out)


def _rsi(c: pd.DataFrame, n: int) -> pd.DataFrame:
    out = {}
    for col in c.columns:
        s = c[col].dropna()
        if len(s) < n * 3:
            out[col] = pd.Series(np.nan, index=c.index)
            continue
        d = s.diff()
        up = _wilder(d.clip(lower=0), n)
        dn = _wilder((-d).clip(lower=0), n)
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        out[col] = rsi.reindex(c.index)
    return pd.DataFrame(out)


def _adx(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = h.diff()
    dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = _atr(h, l, c, n)
    pdi = 100 * _wilder_colwise(plus_dm, n) / atr
    mdi = 100 * _wilder_colwise(minus_dm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wilder_colwise(dx, n)


def _psar_states(h: pd.Series, l: pd.Series, c: pd.Series,
                 af0=0.02, afmax=0.2, step=0.02) -> tuple[pd.Series, pd.Series]:
    hv, lv = h.values, l.values
    n = len(hv)
    bull = np.zeros(n, dtype=bool)
    trend = True
    af = af0
    ep = lv[0]
    sar = hv[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if trend:
            if lv[i] < sar:
                trend, sar, af, ep = False, ep, af0, hv[i]
            elif hv[i] > ep:
                ep, af = hv[i], min(af + step, afmax)
        else:
            if hv[i] > sar:
                trend, sar, af, ep = True, ep, af0, lv[i]
            elif lv[i] < ep:
                ep, af = lv[i], min(af + step, afmax)
        bull[i] = trend
    idx = h.index
    return pd.Series(bull, index=idx), pd.Series(~bull, index=idx)


def _supertrend_states(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame,
                       n: int = 14, m: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    atr = _atr(h, l, c, n)
    hl2 = (h + l) / 2
    bu = hl2 + m * atr
    bl = hl2 - m * atr
    long_state = {}
    short_state = {}
    for col in c.columns:
        cv = c[col].values
        buv, blv = bu[col].values, bl[col].values
        fbu = buv.copy()
        fbl = blv.copy()
        trend = np.zeros(len(cv), dtype=bool)
        cur: bool | None = None
        for i in range(1, len(cv)):
            if np.isnan(fbu[i]) or np.isnan(cv[i]):
                trend[i] = bool(cur) if cur is not None else False
                continue
            if np.isnan(fbu[i - 1]):
                fbu[i] = buv[i]
                fbl[i] = blv[i]
            else:
                fbu[i] = buv[i] if (buv[i] < fbu[i - 1] or cv[i - 1] > fbu[i - 1]) else fbu[i - 1]
                fbl[i] = blv[i] if (blv[i] > fbl[i - 1] or cv[i - 1] < fbl[i - 1]) else fbl[i - 1]
            if cur is None:
                cur = bool(cv[i] >= (fbl[i] + fbu[i]) / 2)
            elif cur and cv[i] < fbl[i]:
                cur = False
            elif not cur and cv[i] > fbu[i]:
                cur = True
            trend[i] = cur
        long_state[col] = pd.Series(trend, index=c.index)
        short_state[col] = pd.Series(~trend, index=c.index)
    return pd.DataFrame(long_state), pd.DataFrame(short_state)


def _states_to_signals(states_long: pd.DataFrame, exit_long: pd.DataFrame,
                       states_short: pd.DataFrame, exit_short: pd.DataFrame,
                       idx_1h: pd.DatetimeIndex, c4: pd.DataFrame | None = None,
                       min_bars4: int = 400) -> Signals:
    out = {}
    for sym in states_long.columns:
        if c4 is not None and int(c4[sym].notna().sum()) < min_bars4:
            continue
        la = states_long[sym].dropna()
        if len(la) < min_bars4:
            continue
        le = exit_long[sym].reindex(la.index).fillna(True)
        sa = states_short[sym].reindex(la.index).fillna(False)
        se = exit_short[sym].reindex(la.index).fillna(True)
        out[sym] = tuple(
            s.reindex(idx_1h, method="ffill").fillna(False).astype(bool)
            for s in (la, le, sa, se)
        )
    return out


def _channel_states(c4: pd.DataFrame, entry_days: int, exit_days: int):
    hi = c4.rolling(entry_days * BPD4).max()
    lo_e = c4.rolling(entry_days * BPD4).min()
    lo_x = c4.rolling(exit_days * BPD4).min()
    hi_x = c4.rolling(exit_days * BPD4).max()
    long_act = c4 > hi.shift(1)
    long_exit = c4 < lo_x.shift(1)
    short_act = c4 < lo_e.shift(1)
    short_exit = c4 > hi_x.shift(1)
    return long_act, long_exit, short_act, short_exit


def psar_signals_panel(panel, af0: float = 0.02, afmax: float = 0.2) -> Signals:
    """PSAR flip states (long while price above SAR) for the whole panel."""
    h4, l4, c4, _ = _resample_ohlcv(panel)
    longs, shorts = {}, {}
    for sym in c4.columns:
        cc = c4[sym].dropna()
        if len(cc) < 400:
            continue
        la_s, sa_s = _psar_states(h4[sym].reindex(cc.index).ffill(),
                                  l4[sym].reindex(cc.index).ffill(), cc,
                                  af0=af0, afmax=afmax)
        longs[sym] = la_s
        shorts[sym] = sa_s
    la, sa = pd.DataFrame(longs), pd.DataFrame(shorts)
    return _states_to_signals(la, ~la, sa, ~sa, panel.index, c4=c4)


def build_signals(family: str, panel) -> Signals:
    h4, l4, c4, v4 = _resample_ohlcv(panel)
    idx = panel.index

    if family.startswith("donchian"):
        parts = family.split("_")
        e_d, x_d = (int(parts[1]), int(parts[2])) if len(parts) == 3 else (20, 10)
        la, le, sa, se = _channel_states(c4, e_d, x_d)
        return _states_to_signals(la, le, sa, se, idx, c4=c4)

    if family.startswith("ema_cross"):
        f_d, s_d = (int(x) for x in family.split("_")[2:4])
        f = _ema(c4, f_d * BPD4)
        s = _ema(c4, s_d * BPD4)
        la, le = f > s, f < s
        return _states_to_signals(la, le, le, la, idx, c4=c4)

    if family == "macd":
        f = _ema(c4, 12 * BPD4)
        s = _ema(c4, 26 * BPD4)
        macd = f - s
        sig = _ema(macd, 9 * BPD4)
        la, le = macd > sig, macd < sig
        return _states_to_signals(la, le, le, la, idx, c4=c4)

    if family == "bollinger":
        mid = c4.rolling(20 * BPD4).mean()
        sd = c4.rolling(20 * BPD4).std(ddof=0)
        up, dn = mid + 2 * sd, mid - 2 * sd
        la, le = c4 > up, c4 < mid
        sa, se = c4 < dn, c4 > mid
        return _states_to_signals(la, le, sa, se, idx, c4=c4)

    if family == "keltner":
        mid = _ema(c4, 20 * BPD4)
        atr = _atr(h4, l4, c4, 14 * BPD4)
        up, dn = mid + 2 * atr, mid - 2 * atr
        la, le = c4 > up, c4 < mid
        sa, se = c4 < dn, c4 > mid
        return _states_to_signals(la, le, sa, se, idx, c4=c4)

    if family == "supertrend":
        la, sa = _supertrend_states(h4, l4, c4)
        le, se = ~la, ~sa
        return _states_to_signals(la, le, sa, se, idx, c4=c4)

    if family == "psar":
        longs, shorts = {}, {}
        for sym in c4.columns:
            cc = c4[sym].dropna()
            if len(cc) < 400:
                continue
            la_s, sa_s = _psar_states(h4[sym].reindex(cc.index).ffill(),
                                      l4[sym].reindex(cc.index).ffill(), cc)
            longs[sym] = la_s
            shorts[sym] = sa_s
        la = pd.DataFrame(longs)
        sa = pd.DataFrame(shorts)
        return _states_to_signals(la, ~la, sa, ~sa, idx, c4=c4)

    if family == "rsi_regime":
        r = _rsi(c4, 14 * BPD4)
        la, le = r > 60, r < 50
        sa, se = r < 40, r > 50
        return _states_to_signals(la, le, sa, se, idx, c4=c4)

    if family == "adx_donchian":
        la0, le, sa0, se = _channel_states(c4, 20, 10)
        adx = _adx(h4, l4, c4)
        gate = adx > 25
        return _states_to_signals(la0 & gate, le, sa0 & gate, se, idx, c4=c4)

    if family == "donchian_vol":
        la0, le, sa0, se = _channel_states(c4, 20, 10)
        vsma = v4.rolling(20 * BPD4).mean()
        gate = v4 > 1.5 * vsma
        return _states_to_signals(la0 & gate, le, sa0 & gate, se, idx, c4=c4)

    if family == "donchian_trend":
        la0, le, sa0, se = _channel_states(c4, 20, 10)
        ema = _ema(c4, 50 * BPD4)
        return _states_to_signals(la0 & (c4 > ema), le, sa0 & (c4 < ema), se, idx, c4=c4)

    if family == "range_expansion":
        atr = _atr(h4, l4, c4, 14 * BPD4)
        rng = (h4 - l4) > 1.5 * atr
        la0, le, sa0, se = _channel_states(c4, 20, 10)
        return _states_to_signals(la0 & rng, le, sa0 & rng, se, idx, c4=c4)

    if family == "donchian_10_5":
        la, le, sa, se = _channel_states(c4, 10, 5)
        return _states_to_signals(la, le, sa, se, idx, c4=c4)

    if family == "donchian_40_20":
        la, le, sa, se = _channel_states(c4, 40, 20)
        return _states_to_signals(la, le, sa, se, idx, c4=c4)

    raise ValueError(f"unknown family {family}")


def run_family_sim(panel, fund_cost: pd.DataFrame, in_universe: pd.DataFrame,
                   signals: Signals, direction: str = "both",
                   max_hold_bars: int = MAX_HOLD_BARS, stop: float = STOP,
                   extra_bps: float = 0.0, max_slots: int = 3,
                   entry_score: pd.DataFrame | None = None,
                   bar_scale: pd.Series | None = None,
                   slot_frac: float = SLOT):
    idx = panel.index
    closes = panel.closes
    rets = closes.pct_change()
    start_i = 30 * panel.bars_per_day
    open_trades: list[Trade] = []
    trades: list[Trade] = []
    net = pd.Series(0.0, index=idx[start_i:])
    pos_of = {t: k for k, t in enumerate(net.index)}
    for i in range(start_i, len(idx)):
        t = idx[i]
        still = []
        for tr in open_trades:
            c_s = signals.get(tr.sig_sym)
            if c_s is None:
                still.append(tr)
                continue
            la, le, sa, se = c_s
            px = closes[tr.sig_sym].iloc[i]
            if tr.sig_dir > 0:
                reason = ("signal" if le.iloc[i]
                          else "time" if i - tr.entry_i >= max_hold_bars
                          else "stop" if px / tr.entry_px - 1 <= -stop else "")
            else:
                reason = ("signal" if se.iloc[i]
                          else "time" if i - tr.entry_i >= max_hold_bars
                          else "stop" if px / tr.entry_px - 1 >= stop else "")
            if reason:
                tr.exit_i, tr.exit_px, tr.reason = i, px, reason
                tr.exit_ts = t
                trades.append(tr)
            else:
                still.append(tr)
        open_trades = still
        if len(open_trades) < max_slots:
            held = {tr.sig_sym for tr in open_trades}
            cands = []
            for sym in signals:
                if sym in held or sym == BTC:
                    continue
                if sym not in in_universe.columns or not bool(in_universe.at[t, sym]):
                    continue
                la, _, sa, _ = signals[sym]
                want_dir = 0
                if direction in ("long", "both") and (
                    la.iloc[i] and not la.iloc[i - 1]
                ):
                    want_dir = +1
                elif direction in ("short", "both") and (
                    sa.iloc[i] and not sa.iloc[i - 1]
                ):
                    want_dir = -1
                if want_dir == 0:
                    continue
                px = float(closes[sym].iloc[i])
                if not np.isfinite(px):
                    continue
                score = (float(entry_score[sym].iloc[i - 1])
                         if entry_score is not None else 0.0)
                cands.append((-score * want_dir, sym, want_dir, px))
            cands.sort()
            for _, sym, want_dir, px in cands:
                if len(open_trades) >= max_slots:
                    break
                tr = Trade(sig_sym=sym, sig_dir=want_dir, entry_i=i, entry_px=px,
                           entry_ts=t)
                open_trades.append(tr)
                held.add(sym)
        bar = 0.0
        for tr in open_trades:
            w = slot_frac * tr.sig_dir
            bar += w * rets[tr.sig_sym].iloc[i]
            bar -= fund_cost[tr.sig_sym].iloc[i] * w
        if bar_scale is not None:
            bar *= float(bar_scale.iloc[i])
        net.iloc[pos_of[t]] = bar
    last_i = len(idx) - 1
    for tr in open_trades:
        tr.exit_i = last_i
        tr.exit_px = float(closes[tr.sig_sym].iloc[-1])
        tr.exit_ts = idx[-1]
        trades.append(tr)
    fee_rt = slot_frac * 2 * (FEE + extra_bps * 1e-4)
    fees = fee_rt * len(trades)
    net_net = net - fees / max(len(net), 1)
    expct = [tr.sig_dir * (tr.exit_px / tr.entry_px - 1.0) for tr in trades
             if np.isfinite(tr.entry_px) and np.isfinite(tr.exit_px)]
    wins = [e for e in expct if e > 0]
    lo, hi_, pn = bootstrap_sharpe_ci(net_net)
    return {
        "sr": sharpe_of(net_net), "ci": (lo, hi_, pn),
        "ret": float(net_net.sum()), "dd": max_dd_of(net_net),
        "trades": len(trades),
        "expectancy": float(np.mean(expct)) if expct else float("nan"),
        "win_rate": len(wins) / len(expct) if expct else float("nan"),
        "fees": fees,
        "net": net_net, "trades_list": trades,
    }


def universe_mask(panel, top_n: int) -> pd.DataFrame:
    dollar_vol = pd.DataFrame(
        {s: panel.frames[s]["Volume"] * panel.frames[s]["Close"] for s in panel.pairs}
    )
    rank = dollar_vol.rolling(30 * 24).mean().rank(axis=1, ascending=False)
    return (rank <= top_n) & rank.notna()


FAMILIES_ROUND1 = [
    "donchian_20_10", "donchian_10_5", "donchian_40_20",
    "ema_cross_4_16", "ema_cross_8_32", "macd", "bollinger", "keltner",
    "supertrend", "psar", "rsi_regime", "adx_donchian", "donchian_vol",
    "donchian_trend", "range_expansion",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair_scout.research_ta_families")
    parser.add_argument("--config", default=None)
    parser.add_argument("--families", default="all")
    parser.add_argument("--universe-top", type=int, default=200)
    parser.add_argument("--direction", default="both",
                        choices=("both", "long", "short"))
    parser.add_argument("--extra-bps", type=float, default=0.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="pair_scout/output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval, data_dir=cfg.data.data_dir,
        top_n_volume=658,
        min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    funding = load_funding(Path(cfg.data.data_dir) / "funding_rates.json")
    fund_cost = pd.DataFrame(
        {s: funding_rate_series(s, funding, panel.index) for s in panel.pairs}
    ).fillna(0.0)
    in_univ = universe_mask(panel, args.universe_top)
    logger.info("panel %d pairs %d bars; universe top-%d direction %s",
                len(panel.pairs), len(panel.index), args.universe_top,
                args.direction)

    families = (FAMILIES_ROUND1 if args.families == "all"
                else [f.strip() for f in args.families.split(",")])
    results = []
    for fam in families:
        try:
            signals = build_signals(fam, panel)
        except Exception as e:
            logger.error("family %s failed: %s", fam, e)
            continue
        logger.info("%s: %d pairs with signals", fam, len(signals))
        if not signals:
            continue
        r = run_family_sim(panel, fund_cost, in_univ, signals,
                           direction=args.direction, extra_bps=args.extra_bps)
        r["label"] = fam
        results.append(r)
        logger.info("%s: SR %.2f P0 %.0f%% ret %+.1f%% dd %.1f%% trades %d "
                    "expct %.2f%% win %.0f%%",
                    fam, r["sr"], r["ci"][2] * 100, r["ret"] * 100,
                    r["dd"] * 100, r["trades"], r["expectancy"] * 100,
                    r["win_rate"] * 100)

    results.sort(key=lambda r: -(r["sr"] if np.isfinite(r["sr"]) else -9))
    lines = [
        "# TA family scan — round 1",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"Universe top-{args.universe_top} point-in-time, direction={args.direction},"
        f" extra slippage {args.extra_bps} bps/side."
        f" {('Tag: ' + args.tag) if args.tag else ''}",
        "",
        "3 slots x 1/3, <=7d hold, 5% stop, 7bps/side + funding, continuous 12m.",
        "",
        "| Family | Port SR [90% CI] | P(SR<=0) | Ret | maxDD | Trades | Expectancy | Win | Fees |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lo, hi_, pn = r["ci"]
        lines.append(
            f"| {r['label']} | {r['sr']:.2f} [{lo:.2f},{hi_:.2f}] | {pn:.0%} | "
            f"{r['ret']:+.1%} | {r['dd']:.1%} | {r['trades']} | "
            f"{r['expectancy']:+.2%} | {r['win_rate']:.0%} | {r['fees']:.1%} |"
        )
    report = "\n".join(lines)
    print(report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    suffix = f"_{args.tag}" if args.tag else ""
    (out_dir / f"ta_families_{stamp}{suffix}.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
