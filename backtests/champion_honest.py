"""Champion 8-slot book, truthful accrual, both cadences. KEEP & REUSE.

PRODUCTION (2026-09-26, round 10): this file covers the DAILY half of the
production pair (PSAR champion, gated, 10:00 SGT). The HOURLY half is FADE v2
(spike-only flash-fade, revert-to-origin exit — see backtests/best4.py and
backtests/iterate10.py). 50/50 capital across the two books:
COMBO SR 3.28, CI [0.36,1.00] P0 0%, +71%/yr, DD -15.0%, halves 4.67/2.41.

IMPORTANT (2026-09-26, causal fix): 4h signal states were previously mapped to
hourly bars with their label-bar START, making entries fill up to 4h BEFORE
the 4h close was knowable. Fixed in research_ta_families._states_to_signals
(state shifted one 4h label = actionable only after bar close). Consequences:
- HOURLY cadence has NO edge for any of 18 signal families — the old hourly
  results (incl. documented SR ~2-3) were a lookahead artifact.
- DAILY 10:00 SGT cadence survives: decisions act on confirmed history anyway.
  This is the cadence the Telegram probe runs.

Mechanics: hourly = enter at flip-bar close, exits every bar; daily = decisions
only at the 02:00 UTC bar. BOTH: entry at bar-close px, accrual from next bar
(no entry-bar credit), 7 bps/side taker + 2 bps slippage + real funding,
exits SAR flip / 5% stop / 7d cap.

Techniques (2026-09-26): blowoff (skip |24h move| > 25%) + ivol (weight =
clip(median-coin-vol / coin 7d vol, 0.5, 2)/8) + voltarget (gross scaled by
min(1, expanding-median / BTC 30d vol)) + riskoff (halve new entries while
book DD > 20%). Default --tech blowoff,ivol,voltarget,riskoff; --tech none
= bare.

Baselines AFTER the causal fix (2026-09-26, saved_data_12m, 11 months):
blowoff,ivol only:
| variant                     | trades | daily-block CI      | ret      | maxDD  |
|-----------------------------|--------|---------------------|----------|--------|
| hourly, Score>80            | 1282   | [-0.36,0.29] P0 55% | -5%/yr   | -44.3% |
| hourly, no gate (FIFO)      | 2594   | [-0.47,0.18] P0 78% | -31%/yr  | -64.0% |
| daily 10:00, Score>80       | 676    | [-0.03,0.59] P0 8%  | +53%/yr  | -28.6% |
| daily 10:00, no gate (FIFO) | 1191   | [-0.03,0.59] P0 7%  | +65%/yr  | -33.9% |
blowoff,ivol,voltarget,riskoff (current default — DD-controlled):
| variant                     | daily-block CI      | ret      | maxDD  |
|-----------------------------|---------------------|----------|--------|
| hourly, Score>80            | [-0.39,0.25] P0 64% | -6%/yr   | -21.7% |
| hourly, no gate (FIFO)      | [-0.48,0.16] P0 81% | -17%/yr  | -30.7% |
| daily 10:00, Score>80       | [-0.06,0.58] P0 9%  | +33%/yr  | -19.1% |
| daily 10:00, no gate (FIFO) | [0.00,0.60] P0 5%   | +54%/yr  | -25.5% |
Riskoff threshold plateau verified at 15/20/25 (maxDD -19 to -23% daily
gated, no cliff). Entry-delay rescue of hourly REJECTED (dly4 spike not on
a plateau). Hourly has NO edge structurally — risk overlays only cut its DD.

Usage:
  ./venv/bin/python backtests/champion_honest.py                 # all 4
  ./venv/bin/python backtests/champion_honest.py --cadence daily
  ./venv/bin/python backtests/champion_honest.py --gate nogate
  ./venv/bin/python backtests/champion_honest.py --tech none     # bare
Data: saved_data_12m (12-month hourly panel + funding_rates.json)
"""
import argparse
import sys
import logging
from collections import defaultdict

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
BLOWOFF_MAX = 0.25   # skip entries with |24h move| above this
VOL_WIN = 7 * 24     # realized-vol window (7d of hourly bars)

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
atr_np, rank_np, mom_np, r1_np = {}, {}, {}, {}
for s in panel.pairs:
    f = panel.frames[s]
    p = f["Close"].shift(1)
    tr = pd.concat([f["High"] - f["Low"], (f["High"] - p).abs(),
                    (f["Low"] - p).abs()], axis=1).max(axis=1)
    atr_np[s] = (tr.rolling(bpd).mean() / f["Close"]).values
    rank_np[s] = rank_df[s].values
    mom_np[s] = mom7_df[s].values
    r1_np[s] = ret1d_df[s].values
vol_mat = np.column_stack([
    (closes[s] / closes[s].shift(1) - 1.0).rolling(VOL_WIN).std().values
    for s in panel.pairs])
med_vol_bar = np.nanmedian(vol_mat, axis=1)
PAIR_IX = {s: j for j, s in enumerate(panel.pairs)}
pc_np = {s: (closes[s] / closes[s].shift(1) - 1.0).fillna(0.0).values
         for s in panel.pairs}
fund_np = {s: fund_cost[s].values for s in panel.pairs}
btc_r = closes[BTC] / closes[BTC].shift(1) - 1.0
btc_vol30 = btc_r.rolling(30 * bpd).std().values
btc_med = btc_r.rolling(30 * bpd).std().expanding(min_periods=30 * bpd).median().values

sig_arr = {s: (la.values.astype(bool), sa.values.astype(bool))
           for s, (la, _le, sa, _se) in sig.items()}

# flip events per bar (for the hourly cadence: enter at flip-bar close)
flips_at = defaultdict(list)
for symv, (la_v, sa_v) in sig_arr.items():
    if symv == BTC:
        continue
    for i in np.flatnonzero(la_v[1:] != la_v[:-1]) + 1:
        flips_at[int(i)].append((symv, "LONG" if la_v[i] else "SHORT"))
    for i in np.flatnonzero(sa_v[1:] != sa_v[:-1]) + 1:
        if any(sy == symv for sy, _d in flips_at[int(i)]):
            continue
        flips_at[int(i)].append((symv, "SHORT" if sa_v[i] else "LONG"))


def score_at(sym, i, direction):
    if not bool(np.isfinite(rank_np[sym][i])) or i < 7 * bpd:
        return None
    m = mom_np[sym][i]
    if not np.isfinite(m):
        return None
    aligned = (direction == "LONG") == (m > 0)
    trend = 40.0 * min(abs(m) / 0.30, 1.0) if aligned else 0.0
    a = atr_np[sym][i]
    if not np.isfinite(a) or a <= 0:
        return None
    r1 = r1_np[sym][i]
    if not np.isfinite(r1):
        return None
    thrust = 30.0 * min(abs(r1) / (2.0 * a), 1.0)
    rk = rank_np[sym][i]
    liq = 30.0 * max(0.0, 1.0 - (float(rk) - 1.0) / 300.0)
    return round(trend + thrust + liq)


def sim(hourly: bool, gate, tech=frozenset(("blowoff", "ivol", "voltarget",
                                            "riskoff"))):
    tech = tech or set()
    dec = ([i for i in range(30 * bpd, len(idx))]
           if hourly else
           [i for i in range(30 * bpd, len(idx)) if idx[i].hour == DECISION_HOUR_UTC])
    open_tr, trades = [], []
    net_arr = np.zeros(len(idx))
    fee_cost = FEE + SLIP_BPS * 1e-4
    last_acc = 30 * bpd - 1
    net_run, eq_max, risk_scale = 0.0, 0.0, 1.0

    def accrue_to(i):
        nonlocal last_acc, net_run
        if i > last_acc:
            for t in open_tr:
                s = t["sym"]
                sgn = 1.0 if t["dir"] == "LONG" else -1.0
                w = t.get("w", SLOT_FRAC)
                net_arr[last_acc + 1:i + 1] += (
                    w * sgn * (pc_np[s][last_acc + 1:i + 1]
                               - fund_np[s][last_acc + 1:i + 1]))
            net_run += float(net_arr[last_acc + 1:i + 1].sum())
            last_acc = i

    for k, i in enumerate(dec):
        accrue_to(i)
        if "riskoff" in tech:
            eq_max = max(eq_max, net_run)
            dd = 1.0 - net_run / eq_max if eq_max > 1e-12 else 0.0
            risk_scale = 0.5 if dd > 0.20 else 1.0
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
        held = {t["sym"] for t in open_tr}
        cands = []
        if hourly:
            for symv, direction in flips_at.get(i, ()):
                if symv in held or not bool(in_u_all[symv].iloc[i]):
                    continue
                if "blowoff" in tech:
                    r1 = r1_np[symv][i]
                    if np.isfinite(r1) and abs(r1) > BLOWOFF_MAX:
                        continue
                sc = score_at(symv, i, direction)
                if sc is None or (gate is not None and sc <= gate):
                    continue
                cands.append((-sc, symv, direction, sc))
        else:
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
                if "blowoff" in tech:
                    r1 = r1_np[symv][i]
                    if np.isfinite(r1) and abs(r1) > BLOWOFF_MAX:
                        continue
                sc = score_at(symv, i, direction)
                if sc is None or (gate is not None and sc <= gate):
                    continue
                cands.append((-sc, symv, direction, sc))
        cands.sort()
        for _neg, symv, direction, sc in cands:
            if len(open_tr) >= SLOTS:
                break
            if "ivol" in tech:
                j = PAIR_IX[symv]
                v, mv = vol_mat[i, j], med_vol_bar[i]
                w = (float(np.clip((mv / v) * SLOT_FRAC,
                                   0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
                     if (np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0)
                     else SLOT_FRAC)
            else:
                w = SLOT_FRAC
            if "voltarget" in tech:
                bv, bm_ = btc_vol30[i], btc_med[i]
                if np.isfinite(bv) and np.isfinite(bm_) and bv > 0:
                    w *= float(min(1.0, bm_ / bv))
            if "riskoff" in tech:
                w *= risk_scale
            net_arr[i] -= w * 2 * fee_cost
            open_tr.append({"sym": symv, "dir": direction, "i0": i, "j0": None,
                            "px0": float(closes[symv].iloc[i]), "score": sc,
                            "exit_px": None, "reason": "open", "w": w})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        t["j0"], t["exit_px"], t["reason"] = len(idx) - 1, float(closes[t["sym"]].iloc[-1]), "open"
        trades.append(t)

    net_net = pd.Series(net_arr, index=idx)
    expct = [(t["exit_px"] / t["px0"] - 1.0 if t["dir"] == "LONG"
              else t["px0"] / max(t["exit_px"], 1e-12) - 1.0) for t in trades]
    wins = sum(1 for e in expct if e > 0)
    holds = [(t["j0"] - t["i0"]) / bpd for t in trades]
    return {"net": net_net, "trades": trades, "n": len(trades),
            "expct": float(np.mean(expct)) if expct else 0.0,
            "win": wins / len(trades) if trades else 0.0,
            "avg_hold": float(np.mean(holds)) if holds else 0.0}


def report(label, r):
    lo, hi, pn = bootstrap_sharpe_ci(r["net"])
    days = len(r["net"]) / 24
    daily = r["net"].resample("1D").sum()
    sr_d = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    print(f"| {label} | {r['n']} | {sharpe_of(r['net']):.2f} | "
          f"[{lo:.2f},{hi:.2f}] P0 {pn:.0%} | {sr_d:.2f} | "
          f"{r['net'].sum():+.1%} (~{r['net'].sum() * 365 / days:+.0%}/yr) | "
          f"{max_dd_of(r['net']):.1%} | {r['win']:.0%} | "
          f"{r['expct']:+.2%} | {r['avg_hold']:.1f}d |", flush=True)
    return r


ap = argparse.ArgumentParser(description="Champion 8-slot, truthful accrual")
ap.add_argument("--cadence", choices=["hourly", "daily", "both"], default="both")
ap.add_argument("--gate", choices=["gated", "nogate", "both"], default="both")
ap.add_argument("--tech", default="blowoff,ivol,voltarget,riskoff",
                help="comma list: blowoff,ivol,voltarget,riskoff | 'none' for bare")
args = ap.parse_args()
TECH = frozenset(args.tech.split(",")) if args.tech != "none" else frozenset()

ALL = (("hourly", "gated", "8-slot hourly, Score>80"),
       ("hourly", "nogate", "8-slot hourly, no gate (FIFO sym)"),
       ("daily", "gated", "8-slot daily 10:00, Score>80"),
       ("daily", "nogate", "8-slot daily 10:00, no gate (FIFO sym)"))
VARIANTS = [(h, g, lab) for h, g, lab in ALL
            if args.cadence in (h, "both") and args.gate in (g, "both")]

print("| variant | trades | SR(bar) | daily-block CI | daily SR | ret | maxDD | win | expct | hold |")
print("|---|---|---|---|---|---|---|---|---|---|")
for cadence, gate_mode, label in VARIANTS:
    r = sim(cadence == "hourly", 80 if gate_mode == "gated" else None, TECH)
    report(label, r)
    if gate_mode == "gated":
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
