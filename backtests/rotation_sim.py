"""Score-rotation variant: continuously re-score held trades and swap out a
holder when a fresh candidate outscores it by a margin. Daily 10:00 SGT
decisions, 8 slots x 1/8, Score>80 entries, hard exits kept (flip/stop/7d).

Variants: margin None (no rotation = current book), 0, 10, 20 points.
Run from repo root:  ./venv/bin/python backtests/rotation_sim.py
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
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

FEE = 7e-4
SLIP_BPS = 2.0
BTC = "BTCUSDT"
SLOTS = 8
SLOT_FRAC = 1.0 / SLOTS
STOP = 0.05
CAP_BARS = 7 * 24
UNIV = 200
GATE = 80
HOUR_UTC = 2

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
dec = [i for i in range(30 * bpd, len(idx)) if idx[i].hour == HOUR_UTC]


def sim(margin):
    open_tr, trades = [], []
    rotations = 0
    for k, i in enumerate(dec):
        # 1) hard exits: flip / stop / 7d cap
        still = []
        for t in open_tr:
            la_v, sa_v = sig_arr[t["sym"]]
            flipped = (not la_v[i]) if t["dir"] == "LONG" else (not sa_v[i])
            c_i = float(closes[t["sym"]].iloc[i])
            adverse = ((c_i / t["px0"] - 1.0 <= -STOP) if t["dir"] == "LONG"
                       else (c_i / t["px0"] - 1.0 >= STOP))
            expired = (i - t["i0"]) >= CAP_BARS
            if flipped or adverse or expired:
                reason = "flip" if flipped else ("stop" if adverse else "time")
                t.update(j0=i, exit_px=c_i, reason=reason)
                trades.append(t)
            else:
                still.append(t)
        open_tr = still
        # 2) candidates: fresh flips in last 24h, gated
        cands = []
        lookback = dec[k - 1] if k else i - 24
        held = {t["sym"] for t in open_tr}
        for sym, (la_v, sa_v) in sig_arr.items():
            if sym == BTC or sym in held or sym not in closes.columns:
                continue
            if not bool(in_u_all[sym].iloc[i]):
                continue
            if la_v[i] != la_v[lookback]:
                d = "LONG" if la_v[i] else "SHORT"
            elif sa_v[i] != sa_v[lookback]:
                d = "SHORT" if sa_v[i] else "LONG"
            else:
                continue
            sc = score_at(sym, i, d)
            if sc is not None and sc > GATE:
                cands.append([sc, sym, d])
        cands.sort(key=lambda x: -x[0])
        # 3) fill free slots
        while cands and len(open_tr) < SLOTS:
            sc, sym, d = cands.pop(0)
            open_tr.append({"sym": sym, "dir": d, "i0": i, "j0": None,
                            "px0": float(closes[sym].iloc[i]), "score": sc,
                            "exit_px": None, "reason": "open"})
        # 4) rotation: swap worst holder if best candidate beats it + margin
        if margin is not None and cands and open_tr:
            while cands:
                worst_i = min(range(len(open_tr)),
                              key=lambda j: score_at(open_tr[j]["sym"], i,
                                                     open_tr[j]["dir"]) or -1)
                w_sc = score_at(open_tr[worst_i]["sym"], i,
                                open_tr[worst_i]["dir"])
                w_sc = w_sc if w_sc is not None else -1
                best = cands[0]
                if best[0] > w_sc + margin:
                    out_t = open_tr.pop(worst_i)
                    out_t.update(j0=i, exit_px=float(closes[out_t["sym"]].iloc[i]),
                                 reason="rotate")
                    trades.append(out_t)
                    rotations += 1
                    sc, sym, d = cands.pop(0)
                    open_tr.append({"sym": sym, "dir": d, "i0": i, "j0": None,
                                    "px0": float(closes[sym].iloc[i]),
                                    "score": sc, "exit_px": None,
                                    "reason": "open"})
                else:
                    break
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
    expct = [(t["exit_px"] / t["px0"] - 1.0 if t["dir"] == "LONG"
              else t["px0"] / max(t["exit_px"], 1e-12) - 1.0) for t in trades]
    wins = sum(1 for e in expct if e > 0)
    holds = [(t["j0"] - t["i0"]) / bpd for t in trades]
    return {"net": net_net, "n": len(trades), "rot": rotations,
            "win": wins / len(trades) if trades else 0,
            "expct": float(np.mean(expct)) if expct else 0,
            "avg_hold": float(np.mean(holds)) if holds else 0}


print("| margin | trades | rotations | SR(bar) | daily-block CI | daily SR | ret | maxDD | win | expct | hold |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for margin in (None, 20, 10, 0):
    r = sim(margin)
    lo, hi, pn = bootstrap_sharpe_ci(r["net"])
    daily = r["net"].resample("1D").sum()
    sr_d = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    days = len(r["net"]) / 24
    print(f"| {margin if margin is not None else 'none'} | {r['n']} | "
          f"{r['rot']} | {sharpe_of(r['net']):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {sr_d:.2f} | {r['net'].sum():+.1%} "
          f"(~{r['net'].sum() * 365 / days:+.0%}/yr) | {max_dd_of(r['net']):.1%} | "
          f"{r['win']:.0%} | {r['expct']:+.2%} | {r['avg_hold']:.1f}d |",
          flush=True)
