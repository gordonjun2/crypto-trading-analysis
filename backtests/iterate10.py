"""Round 10: iterate on the production pair (FADE hourly + PSAR daily).

Fade improvements tested:
- exit design : fixed 12h hold | revert-to-origin (price returns to the
                pre-flash close, cap 24h) | ATR-target (+/-2xATR back, cap 24h)
- entry       : market at flash close | limit 1xATR deeper, 6h window
- sizing      : equal 1/8 | ivol | extremity (bigger fade for bigger overshoot)
- filters     : none | BTC-quiet (|BTC 1h ret| < 1x BTC ATR -> idiosyncratic)
- diagnostics : direction split, BTC-state split, hour distribution

Also: daily decision-hour robustness (informational) and combo split plateau.

Run:  ./venv/bin/python backtests/iterate10.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR

idx, closes, bpd = TR.idx, TR.closes, TR.bpd
in_u_all, pc_np, fund_np = TR.in_u_all, TR.pc_np, TR.fund_np
SLOTS, SLOT_FRAC = TR.SLOTS, TR.BASE_W
BTC = TR.BTC
fee_rt = TR.FEE + TR.SLIP_BPS * 1e-4
CHAMP = frozenset(("blowoff", "ivol", "voltarget", "riskoff"))

atr1_np = {}
for s in TR.panel.pairs:
    f = TR.panel.frames[s]
    c = f["Close"]
    p = c.shift(1)
    tr1 = pd.concat([f["High"] - f["Low"], (f["High"] - p).abs(),
                     (f["Low"] - p).abs()], axis=1).max(axis=1)
    atr1_np[s] = (tr1.rolling(24).mean() / c).values
btc_atr1 = atr1_np[BTC]
last_fund = TR.last_fund
vol_raw = {s: TR.panel.frames[s]["Volume"].values for s in TR.panel.pairs}
vol_ma = {s: pd.Series(v).rolling(24).mean().values for s, v in vol_raw.items()}

from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def report(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


def fade_sim(k=5.0, hold=12, ivol=True, exit_mode="hold", entry_mode="market",
             btc_quiet=False, ext_size=False, direction_filter=None,
             gate=None, hours=None, hour_max=None, vol_min=None,
             fund_min=None, ext_cap=2.0, cap_h=24, slots=None):
    """exit_mode: hold (fixed) | origin (revert to pre-flash close, cap 24h) |
    atr (+-2xATR back toward origin, cap 24h).
    entry_mode: market (flash close) | limit (1xATR deeper, 6h window).
    direction_filter: None | 'dump' (fade dumps = LONG only) | 'spike'.
    gate: Score-gate entries at the flash bar (kills counter-momentum fades).
    hours: entry-hour filter set (None = hourly).
    hour_max: only enter events before this UTC hour.
    vol_min: flash-bar volume must exceed vol_min x its 24h mean.
    fund_min: last funding must be >= fund_min (crowded-long confirmation).
    ext_cap: sizing cap multiple for extremity sizing.
    cap_h: max-hold cap for origin/atr exits.
    slots: concurrent-position cap (default TR.SLOTS)."""
    slots = slots or SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades = [], []
    pend_lim = []
    last_acc = 30 * bpd - 1
    recs = []

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

    def weight(symv, i, r, a):
        if ext_size:
            excess = abs(r) / a - k
            return float(np.clip(1.0 + excess / 3.0, 0.5, ext_cap) * SLOT_FRAC)
        if ivol:
            v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
            if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
                return float(np.clip((mv / v) * SLOT_FRAC,
                                     0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
        return SLOT_FRAC

    low_np = {s: TR.panel.frames[s]["Low"].values for s in TR.panel.pairs}
    high_np = {s: TR.panel.frames[s]["High"].values for s in TR.panel.pairs}

    def close_trade(t, i, reason):
        c_i = float(closes[t["sym"]].iloc[i])
        t.update(j0=i, exit_px=c_i, reason=reason)
        trades.append(t)
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        recs.append({"sym": t["sym"], "dir": t["dir"], "i0": t["i0"], "j0": i,
                     "ret": sgn * (c_i / t["px0"] - 1.0),
                     "btc": t.get("btc_abs", np.nan)})

    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            expired = (i - t["i0"]) >= (hold if exit_mode == "hold" else cap_h)
            hit = False
            if exit_mode == "origin":
                o = t["origin"]
                hit = (c_i >= o) if t["dir"] == "LONG" else (c_i <= o)
            elif exit_mode == "atr":
                tgt = (t["px0"] * (1.0 + 2.0 * t["atr0"]) if t["dir"] == "LONG"
                       else t["px0"] * (1.0 - 2.0 * t["atr0"]))
                hit = (c_i >= tgt) if t["dir"] == "LONG" else (c_i <= tgt)
            if expired:
                close_trade(t, i, "time")
            elif hit:
                close_trade(t, i, exit_mode)
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        pend_syms = {p[1] for p in pend_lim}
        # limit fills first
        if pend_lim:
            fill_cands, keep = [], []
            for exp_i, symv, direction, lim, meta in pend_lim:
                if i > exp_i or symv in held:
                    continue
                hit = (low_np[symv][i] <= lim if direction == "LONG"
                       else high_np[symv][i] >= lim)
                if not hit:
                    keep.append((exp_i, symv, direction, lim, meta))
                    continue
                fill_cands.append((symv, direction, lim, meta))
            pend_lim = keep
            for symv, direction, fill, meta in fill_cands:
                if len(open_tr) >= slots:
                    continue
                w = weight(symv, i, meta["r"], meta["a"])
                net_arr[i] -= w * 2 * fee_rt
                open_tr.append({"sym": symv, "dir": direction, "i0": i,
                                "px0": fill, "origin": meta["origin"],
                                "atr0": meta["a"], "w": w,
                                "btc_abs": meta["btc_abs"]})
                held.add(symv)
        # flash events
        if hours is not None and idx[i].hour not in hours:
            continue
        if hour_max is not None and idx[i].hour >= hour_max:
            continue
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv in pend_syms or symv not in pc_np:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a:
                continue
            if vol_min is not None:
                v0 = vol_raw[symv][i]
                vma = vol_ma[symv][i]
                if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                        and v0 >= vol_min * vma):
                    continue
            if fund_min is not None:
                lf = last_fund[symv].iloc[i] if symv in last_fund.columns else 0.0
                if not (np.isfinite(lf) and lf >= fund_min):
                    continue
            raw_dir = "LONG" if r > 0 else "SHORT"
            direction = "SHORT" if raw_dir == "LONG" else "LONG"  # fade
            if gate is not None:
                sc = TR.score_at(symv, i, direction)
                if sc is None or sc <= gate:
                    continue
            if direction_filter == "dump" and direction != "LONG":
                continue
            if direction_filter == "spike" and direction != "SHORT":
                continue
            br = pc_np[BTC][i]
            ba = btc_atr1[i]
            if btc_quiet and not (np.isfinite(br) and np.isfinite(ba)
                                  and abs(br) < ba):
                continue
            if len(open_tr) >= slots:
                continue
            w = weight(symv, i, r, a)
            origin = float(closes[symv].iloc[i - 1])
            meta = {"r": r, "a": a, "origin": origin,
                    "btc_abs": abs(br) if np.isfinite(br) else np.nan}
            if entry_mode == "limit":
                lim = (float(closes[symv].iloc[i]) * (1.0 - a)
                       if direction == "LONG"
                       else float(closes[symv].iloc[i]) * (1.0 + a))
                pend_lim.append((i + 6, symv, direction, lim, meta))
            else:
                px = float(closes[symv].iloc[i])
                net_arr[i] -= w * 2 * fee_rt
                open_tr.append({"sym": symv, "dir": direction, "i0": i,
                                "px0": px, "origin": origin, "atr0": a,
                                "w": w, "btc_abs": meta["btc_abs"]})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, recs


if __name__ == "__main__":
    # ---------- diagnostics on the base config ----------
    net0, tr0, recs = fade_sim()
    dump = [r for r in recs if r["dir"] == "LONG"]
    spk = [r for r in recs if r["dir"] == "SHORT"]
    quiet = [r for r in recs if np.isfinite(r["btc"]) and r["btc"] < btc_atr1.mean()]
    print(f"base fade: {len(recs)} trades | dump-fades {len(dump)} "
          f"avg {np.mean([r['ret'] for r in dump]):+.2%} | "
          f"spike-fades {len(spk)} avg {np.mean([r['ret'] for r in spk]):+.2%} | "
          f"BTC-quiet {len(quiet)} avg {np.mean([r['ret'] for r in quiet]):+.2%}",
          flush=True)

    print()
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    report("base (hold12, market, ivol)", net0, len(recs))
    net, tr, rc = fade_sim(exit_mode="origin")
    report("exit=revert-to-origin cap24h", net, len(rc))
    net, tr, rc = fade_sim(exit_mode="atr")
    report("exit=2xATR-target cap24h", net, len(rc))
    net, tr, rc = fade_sim(entry_mode="limit")
    report("entry=limit 1xATR deeper", net, len(rc))
    net, tr, rc = fade_sim(ext_size=True)
    report("sizing=extremity", net, len(rc))
    net, tr, rc = fade_sim(btc_quiet=True)
    report("filter=BTC-quiet only", net, len(rc))
    net, tr, rc = fade_sim(direction_filter="dump")
    report("only dump-fades (LONG)", net, len(rc))
    net, tr, rc = fade_sim(direction_filter="spike")
    report("only spike-fades (SHORT)", net, len(rc))

    # ---------- daily decision-hour robustness (informational) ----------
    print()
    print("| daily decision hour UTC | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    for h in (0, 2, 6, 14, 20):
        TR.DECISION_HOUR_UTC = h
        r = TR.sim(False, 80, CHAMP)
        report(f"{h:02d}:00", r["net"])
    TR.DECISION_HOUR_UTC = 2

    # ---------- combo split plateau ----------
    print()
    print("| combo split (fade/daily) | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    champ = TR.sim(False, 80, CHAMP)["net"]
    for wf in (0.4, 0.5, 0.6):
        combo = wf * net0 + (1.0 - wf) * champ
        report(f"{wf:.0%} fade / {1-wf:.0%} daily", combo)
