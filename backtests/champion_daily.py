"""Champion book, realistic cadence: 8 slots x 1/8, decisions ONLY at 10:00 SGT.

The published champion headline (daily-block SR [2.90, 3.53]) assumes hourly
decisions. This sim is what the Telegram user can actually trade:
- decision points: one per day, at the 02:00 UTC (10:00 SGT) bar close
- entries: fresh PSAR flips (state changed within last 24h), in-universe top-200
  (point-in-time), Score > gate, sym != BTC, not already held; priority by
  Score desc; max 8 concurrent slots, 1/8 each (unhedged, per iteration-10)
- exits: evaluated at decision bars only -> SAR flip / 5% adverse stop / 7d cap
- costs: 7 bps/side taker + 2 bps slippage + real funding on held legs
- accrual: hourly close-to-close AFTER entry (no entry-bar credit)

Run from repo root:  ./venv/bin/python backtests/champion_daily.py
Data: saved_data_12m (12-month hourly panel + funding_rates.json)
"""
import sys
import logging

sys.path.insert(0, "/root/crypto-trading-analysis")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from pathlib import Path

from pair_scout.config import load_config
from pair_scout.data.funding import funding_rate_series, load_funding
from pair_scout.data.loader import load_panel
from pair_scout.research_ta_families import psar_signals_panel, universe_mask
from pair_scout.research_sizing import (
    bootstrap_sharpe_ci, max_dd_of, sharpe_of,
)

FEE = 7e-4
SLIP_BPS = 2.0
BTC = "BTCUSDT"
SLOTS = 8
SLOT_FRAC = 1.0 / SLOTS
STOP = 0.05
CAP_BARS = 7 * 24
UNIV = 200
DECISION_HOUR_UTC = 2

cfg = load_config(None)
panel = load_panel(
    cex=cfg.data.cex, interval=cfg.data.interval, data_dir="saved_data_12m",
    top_n_volume=658, min_history_bars=cfg.data.min_history_bars,
    max_nan_fraction=cfg.data.max_nan_fraction,
    trailing_volume_days=cfg.data.trailing_volume_days,
)
idx = panel.index
closes = panel.closes
bpd = panel.bars_per_day
funding = load_funding(Path("saved_data_12m") / "funding_rates.json")
fund_cost = pd.DataFrame(
    {s: funding_rate_series(s, funding, idx) for s in panel.pairs}
).fillna(0.0)
in_u_all = universe_mask(panel, UNIV)
sig = psar_signals_panel(panel)

dollar = pd.DataFrame(
    {s: panel.frames[s]["Volume"] * panel.frames[s]["Close"] for s in panel.pairs})
rank_df = dollar.rolling(30 * bpd).mean().rank(axis=1, ascending=False)
mom7_df = closes / closes.shift(7 * bpd) - 1.0
ret1d_df = closes / closes.shift(bpd) - 1.0
atr_arr = {}
for s in panel.pairs:
    f = panel.frames[s]
    p = f["Close"].shift(1)
    tr = pd.concat([f["High"] - f["Low"], (f["High"] - p).abs(),
                    (f["Low"] - p).abs()], axis=1).max(axis=1)
    atr_arr[s] = (tr.rolling(bpd).mean() / f["Close"]).values


def score_at(sym, i, direction):
    px = closes[sym].iloc[i]
    if not np.isfinite(px) or px <= 0 or i < 7 * bpd:
        return None
    m = mom7_df[sym].iloc[i]
    if not np.isfinite(m):
        return None
    aligned = (direction == "LONG") == (m > 0)
    trend = 40.0 * min(abs(m) / 0.30, 1.0) if aligned else 0.0
    a = float(atr_arr[sym][i])
    if not np.isfinite(a) or a <= 0:
        return None
    r1 = ret1d_df[sym].iloc[i]
    if not np.isfinite(r1):
        return None
    thrust = 30.0 * min(abs(r1) / (2.0 * a), 1.0)
    rk = rank_df[sym].iloc[i]
    liq = 30.0 * max(0.0, 1.0 - (float(rk) - 1.0) / 300.0) if np.isfinite(rk) else 0.0
    return round(trend + thrust + liq)


sig_arr = {s: (la.values.astype(bool), sa.values.astype(bool))
           for s, (la, _le, sa, _se) in sig.items()}


def sim(gate):
    dec = [i for i in range(30 * bpd, len(idx)) if idx[i].hour == DECISION_HOUR_UTC]
    open_tr, trades = [], []
    for k, i in enumerate(dec):
        # exits (daily only)
        still = []
        for tr in open_tr:
            d = tr["dir"]
            la_v, sa_v = sig_arr[tr["sym"]]
            flipped = (not la_v[i]) if d == "LONG" else (not sa_v[i])
            c_i = float(closes[tr["sym"]].iloc[i])
            if d == "LONG":
                adverse = c_i / tr["px0"] - 1.0 <= -STOP
            else:
                adverse = c_i / tr["px0"] - 1.0 >= STOP
            expired = (i - tr["i0"]) >= CAP_BARS
            if flipped or adverse or expired:
                reason = "flip" if flipped else ("stop" if adverse else "time")
                tr.update(j0=i, exit_px=c_i, reason=reason)
                trades.append(tr)
            else:
                still.append(tr)
        open_tr = still
        # entries (daily only, Score gate, priority by Score desc)
        held = {t["sym"] for t in open_tr}
        cands = []
        lookback = dec[k - 1] if k else i - 24
        for symv, (la_v, sa_v) in sig_arr.items():
            if symv == BTC or symv in held or symv not in closes.columns:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            if la_v[i] != la_v[lookback]:
                direction = "LONG" if la_v[i] else "SHORT"
            elif sa_v[i] != sa_v[lookback]:
                direction = "SHORT" if sa_v[i] else "LONG"
            else:
                continue
            sc = score_at(symv, i, direction)
            if sc is None or (gate is not None and sc <= gate):
                continue
            cands.append((-sc, symv, direction, sc))
        cands.sort()
        for _neg, symv, direction, sc in cands:
            if len(open_tr) >= SLOTS:
                break
            la_v, sa_v = sig_arr[symv]
            open_tr.append({"sym": symv, "dir": direction, "i0": i, "j0": None,
                            "px0": float(closes[symv].iloc[i]), "score": sc,
                            "exit_px": None, "reason": "open",
                            "la_v": la_v, "sa_v": sa_v})
        # accrual deferred: hourly returns accumulated from trade records below
    for t in open_tr:
        t["j0"], t["exit_px"], t["reason"] = len(idx) - 1, float(closes[t["sym"]].iloc[-1]), "open"
        trades.append(t)

    net = pd.Series(0.0, index=idx)
    pc = closes.pct_change()
    for t in trades:
        i0, j0 = t["i0"] + 1, t["j0"]
        if j0 <= t["i0"]:
            continue
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        net.iloc[i0:j0 + 1] += (SLOT_FRAC * sgn
                                * pc[t["sym"]].iloc[i0:j0 + 1].fillna(0).values)
        net.iloc[i0:j0 + 1] -= (SLOT_FRAC * sgn
                                * fund_cost[t["sym"]].iloc[i0:j0 + 1].values)
    fee_rt = SLOT_FRAC * 2 * (FEE + SLIP_BPS * 1e-4)
    net_net = net - fee_rt * len(trades) / max(len(net), 1)
    expct = [leg_pnl(t) for t in trades]
    wins = sum(1 for e in expct if e > 0)
    holds = [(t["j0"] - t["i0"]) / bpd for t in trades]
    return {"net": net_net, "trades": trades, "n": len(trades),
            "expct": float(np.mean(expct)) if expct else 0.0,
            "win": wins / len(trades) if trades else 0.0,
            "avg_hold": float(np.mean(holds)) if holds else 0.0}


def leg_pnl(t):
    r = (t["exit_px"] / t["px0"] - 1.0 if t["dir"] == "LONG"
         else t["px0"] / max(t["exit_px"], 1e-12) - 1.0)
    return r


def report(label, r):
    lo, hi, pn = bootstrap_sharpe_ci(r["net"])
    days = len(r["net"]) / 24
    daily = r["net"].resample("1D").sum()
    sr_d = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    print(f"| {label} | {r['n']} | {sharpe_of(r['net']):.2f} | "
          f"[{lo:.2f},{hi:.2f}] P0 {pn:.0%} | {sr_d:.2f} | "
          f"{r['net'].sum():+.1%} (~{r['net'].sum() * 365 / days:+.0%}/yr) | "
          f"{max_dd_of(r['net']):.1%} | {r['win']:.0%} | "
          f"{r['expct']:+.2%} | {r['avg_hold']:.1f}d |")


print("| variant | trades | SR(bar) | daily-block CI | daily SR | ret | maxDD | win | expct | hold |")
print("|---|---|---|---|---|---|---|---|---|---|")
for label, gate in (("8-slot, 10:00 SGT, Score>80", 80),
                    ("8-slot, 10:00 SGT, no gate (FIFO sym)", None)):
    r = sim(gate)
    report(label, r)
    if gate == 80:
        net = r["net"]
        half = len(net) // 2
        for name, seg in (("H1", net.iloc[:half]), ("H2", net.iloc[half:])):
            lo, hi, pn = bootstrap_sharpe_ci(seg)
            print(f"| {name} | | {sharpe_of(seg):.2f} | [{lo:.2f},{hi:.2f}] "
                  f"P0 {pn:.0%} | | {seg.sum():+.1%} | {max_dd_of(seg):.1%} | | | |")
        mix = {}
        for t in r["trades"]:
            mix[t["reason"]] = mix.get(t["reason"], 0) + 1
        print("exit mix:", " ".join(f"{k}:{v}" for k, v in sorted(mix.items())),
              flush=True)
