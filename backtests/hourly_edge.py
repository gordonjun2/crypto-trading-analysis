"""Can ANY strategy make hourly beat the daily champion? — YES (2026-09-26).

VERDICT: the FADE book — 1h bar closes with |ret| > 5x its 24h ATR in a
top-200 alt -> FADE it (sell spike / buy dump), hold 12h, 8 slots, ivol —
is hourly-native, fee-light (~3 events/day) and standalone-strong
(SR 1.58, CI [0.02,0.66] P0 4%, +54%/yr). Correlation with the daily
champion is -0.08; the 50/50 COMBO: SR 2.23, CI [0.16,0.78] P0 1%,
+47%/yr, DD -15.9%, halves 2.55/2.04 — the best result of the project.
Mechanism: liquidation-cascade overshoots revert (monotone in extremity:
2xATR reversion fails, 3x weak, 5x strong, 8x positive). Caveats: fade
alone H2-tilted; hold 3h weak / 6h ok / 12h best.

Parts below were the search path (all kept for reference):
Part 1  DRIFT CURVE — gated 4h-flip candidates have NEGATIVE mean drift at
every horizon (+0.06% at 1h -> -0.51% at 24h): explains why the flip alpha
cannot be traded hourly (daily book wins via selection, not speed).
Part 2  FLASH CONTINUATION (fade's mirror) — negative at all k.
Part 3  US-HOURS SEASONALITY — all windows negative; no time-of-day alpha
in this alt sample.
Part 4  COMBO of champion + fade book (the breakthrough).

Run:  ./venv/bin/python backtests/hourly_edge.py
Data: saved_data_12m (loads via technique_research module, ~1 min)
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR

idx, closes, bpd = TR.idx, TR.closes, TR.bpd
sig_arr, FLIPS_AT_BAR = TR.sig_arr, TR.FLIPS_AT_BAR
in_u_all, r1_np = TR.in_u_all, TR.r1_np
pc_np, fund_np = TR.pc_np, TR.fund_np
SLOTS, SLOT_FRAC = TR.SLOTS, TR.BASE_W
FEE, SLIP_BPS = TR.FEE, TR.SLIP_BPS
BTC = TR.BTC
fee_rt = FEE + SLIP_BPS * 1e-4

from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def report(label, net):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    print(f"| {label} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


# ---------- Part 1: drift curve ----------
HORIZONS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36)
events = []
last_bar = len(idx) - 1 - max(HORIZONS)
for i in range(30 * bpd, last_bar):
    for symv, direction in FLIPS_AT_BAR.get(i, ()):
        if symv == BTC or not bool(in_u_all[symv].iloc[i]):
            continue
        sc = TR.score_at(symv, i, direction)
        if sc is None or sc <= 80:
            continue
        r1 = r1_np[symv][i]
        if np.isfinite(r1) and abs(r1) > TR.BLOWOFF_MAX:
            continue
        events.append((symv, direction, i))
print(f"drift-curve events: {len(events)} gated flips", flush=True)
rows = {}
for h in HORIZONS:
    vals = []
    for symv, d, i in events:
        sgn = 1.0 if d == "LONG" else -1.0
        c0 = float(closes[symv].iloc[i])
        c1 = float(closes[symv].iloc[i + h])
        vals.append(sgn * (c1 / c0 - 1.0))
    rows[h] = float(np.mean(vals))
print("| hours after confirmation | mean cumulative return |")
print("|---|---|")
for h in HORIZONS:
    print(f"| {h}h | {rows[h]:+.2%} |", flush=True)

# ---------- Part 2: flash continuation ----------
atr1_np = {}
for s in TR.panel.pairs:
    f = TR.panel.frames[s]
    c = f["Close"]
    p = c.shift(1)
    tr1 = pd.concat([f["High"] - f["Low"], (f["High"] - p).abs(),
                     (f["Low"] - p).abs()], axis=1).max(axis=1)
    atr1_np[s] = (tr1.rolling(24).mean() / c).values


def flash_sim(k, hold=6, fade=False, ivol=False, tech=frozenset()):
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr = []
    n_trades = 0
    last_acc = 30 * bpd - 1

    def accrue_to(i):
        nonlocal last_acc
        if i > last_acc:
            for t in open_tr:
                s = t["sym"]
                sgn = 1.0 if t["dir"] == "LONG" else -1.0
                w = t.get("w", SLOT_FRAC)
                lo = max(last_acc + 1, t["i0"] + 1)
                if lo <= i:
                    net_arr[lo:i + 1] += (w * sgn * (pc_np[s][lo:i + 1]
                                                     - fund_np[s][lo:i + 1]))
            last_acc = i

    def weight(symv, i):
        if not ivol:
            return SLOT_FRAC
        v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            return float(np.clip((mv / v) * SLOT_FRAC,
                                 0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
        return SLOT_FRAC

    for i in range(30 * bpd, n):
        accrue_to(i)
        # exits: fixed hold
        open_tr = [t for t in open_tr if i - t["i0"] < hold]
        held = {t["sym"] for t in open_tr}
        # entries: flash bars
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv not in pc_np:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a:
                continue
            direction = ("SHORT" if r > 0 else "LONG") if fade else \
                        ("LONG" if r > 0 else "SHORT")
            if len(open_tr) >= SLOTS:
                break
            w = weight(symv, i)
            n_trades += 1
            net_arr[i] -= w * 2 * fee_rt
            open_tr.append({"sym": symv, "dir": direction, "i0": i, "w": w})
    accrue_to(len(idx) - 1)
    return pd.Series(net_arr, index=idx), n_trades


print()
print("| flash variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
print("|---|---|---|---|---|---|")
for k in (3.0, 5.0, 8.0):
    net, nt = flash_sim(k)
    report(f"cont {k}xATR hold6h [{nt}]", net)
for k in (3.0, 5.0, 8.0):
    net, nt = flash_sim(k, fade=True)
    report(f"fade {k}xATR hold6h [{nt}]", net)

# robustness battery for the fade family
for hold in (3, 12):
    net, nt = flash_sim(5.0, hold=hold, fade=True)
    report(f"fade 5xATR hold{hold}h [{nt}]", net)
for k in (5.0, 8.0):
    net, nt = flash_sim(k, fade=True, ivol=True)
    report(f"fade {k}xATR hold6h ivol [{nt}]", net)
net, nt = flash_sim(5.0, fade=True, ivol=True)
half = len(net) // 2
for name, seg in (("H1", net.iloc[:half]), ("H2", net.iloc[half:])):
    lo, hi, pn = bootstrap_sharpe_ci(seg)
    days = len(seg) / 24
    print(f"| fade5 ivol {name} | | {sharpe_of(seg):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {seg.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(seg):.1%} |", flush=True)
m = net.resample("MS").sum()
print(f"  months: {int((m > 0).sum())}/{len(m)} positive | worst {m.min():+.1%} | "
      f"median {m.median():+.1%}", flush=True)

# ---------- Part 4: combine fade book with the daily champion ----------
fade_net, fade_n = flash_sim(5.0, hold=12, fade=True, ivol=True)
champ = TR.sim(False, 80, frozenset(("blowoff", "ivol", "voltarget", "riskoff")))
champ_net = champ["net"]
corr = float(np.corrcoef(champ_net.values, fade_net.values)[0, 1])
combo = 0.5 * champ_net + 0.5 * fade_net
print(f"\ncorr(champion daily, fade-hourly) = {corr:+.2f}")
print("| book | SR(bar) | daily-block CI | ret | maxDD |")
print("|---|---|---|---|---|")
report("champion alone (dy/gated)", champ_net)
report(f"fade5x12h ivol alone [{fade_n}]", fade_net)
report("50/50 combo", combo)
half = len(combo) // 2
for name, seg in (("H1", combo.iloc[:half]), ("H2", combo.iloc[half:])):
    lo, hi, pn = bootstrap_sharpe_ci(seg)
    days = len(seg) / 24
    print(f"| combo {name} | | {sharpe_of(seg):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {seg.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(seg):.1%} |", flush=True)
m = combo.resample("MS").sum()
print(f"  combo months: {int((m > 0).sum())}/{len(m)} positive | worst "
      f"{m.min():+.1%} | median {m.median():+.1%}", flush=True)

# ---------- Part 3: seasonality ----------
mom7_df = closes / closes.shift(7 * bpd) - 1.0


def season_sim(h_in, h_out, top=8, tech=frozenset()):
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr = []  # list of (sym, dir, i0, w)
    fee_paid = False
    last_acc = 30 * bpd - 1

    def accrue_to(i):
        nonlocal last_acc
        if i > last_acc:
            for t in open_tr:
                s = t[0]
                w = t[3]
                net_arr[last_acc + 1:i + 1] += (
                    w * (pc_np[s][last_acc + 1:i + 1]
                         - fund_np[s][last_acc + 1:i + 1]))
            last_acc = i

    in_win_prev = False
    for i in range(30 * bpd, n):
        accrue_to(i)
        hr = idx[i].hour
        in_win = (h_in <= hr < h_out)
        if in_win and not in_win_prev:
            # session open: pick top momentum alts
            scores = mom7_df.iloc[i]
            cand = [(s, scores[s]) for s in TR.panel.pairs
                    if s != BTC and bool(in_u_all[s].iloc[i])
                    and np.isfinite(scores[s])]
            cand.sort(key=lambda x: -x[1])
            open_tr = [(s, i, SLOT_FRAC * (1.0 if v > 0 else -1.0), SLOT_FRAC)
                       for s, v in cand[:top]]
            for t in open_tr:
                net_arr[i] -= t[3] * 2 * fee_rt
        elif (not in_win) and in_win_prev:
            # session close: flat
            open_tr = []
        in_win_prev = in_win
    accrue_to(len(idx) - 1)
    return pd.Series(net_arr, index=idx)


print()
print("| season window UTC | SR(bar) | daily-block CI | ret | maxDD |")
print("|---|---|---|---|---|")
for h_in, h_out in ((13, 21), (8, 16), (0, 8)):
    net = season_sim(h_in, h_out)
    report(f"hold {h_in:02d}-{h_out:02d} UTC", net)
