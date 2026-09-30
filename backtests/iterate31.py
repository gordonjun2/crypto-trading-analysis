"""Round 31: 5x leverage — hard stops, liquidation impossible.

v8 books run at 1x notional (per-slot w = ivol*SLOT_FRAC <= 0.25 NAV,
gross rarely > ~1.5x). Directive: 5x leverage, stop-loss mandatory,
no liquidation. At 5x isolated margin liquidation sits ~20% adverse
(1/5, pre-maintenance) so stops must be well inside; fills modeled
conservatively (gap-through -> open, stop checked before limit exits).

Sizing at 5x: per-slot w = clip(ivol, 0.5, 1.0) * SLOT_FRAC * LEV in
[0.3125, 0.625] NAV (the 2x ivol up-stretch is removed when levered),
portfolio gross capped at 5.0 NAV at entry. Funding/fees scale with
notional automatically (all accounting is linear in w).

31a  FADE v7 + hard stop sweep (ATR-k and fixed-%) at 5x
31b  CLIM v1 + hard floor-stop sweep at 5x
31c  SQUEEZE close-stop sweep at 5x (native FS.sim STOP)
31d  5x trio assembly: splits, months, gross, worst-trade, no-liq check

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate31.py [a|b|c|d|all]
      PS_DATA_DIR=saved_data_12m ... d   # cross-check on 12m
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
import family_scan_honest as FS
from iterate10 import (idx, closes, bpd, in_u_all, pc_np, fund_np, SLOTS,
                       SLOT_FRAC, BTC, fee_rt, atr1_np, vol_raw, vol_ma,
                       report)
from iterate14 import low_np, high_np, squeeze2, SQ_OVER
from iterate25 import load_flow
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

LEV = 5.0
GROSS_CAP = 5.0


# ------------------------------------------------------------------ FADE 5x
def fade_sim8(k=5.0, cap_h=42, tf=None, tf_max=0.60, gate=None,
              lev=LEV, stop_pct=None, stop_atr=None, gross_cap=GROSS_CAP):
    """FADE v7 + per-position hard stop. Stop checked before limit exits
    (same-bar worst case). Fill = max(open, stop_px) for shorts."""
    flow = tf
    slots = SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades, recs = [], [], []
    last_close_i = {}

    def accrue_to(i):
        nonlocal last_acc
        if i > last_acc:
            for t in open_tr:
                s = t["sym"]
                sgn = 1.0 if t["dir"] == "LONG" else -1.0
                lo = max(last_acc + 1, t["i0"] + 1)
                if lo <= i:
                    net_arr[lo:i + 1] += (t["w"] * sgn *
                                          (pc_np[s][lo:i + 1]
                                           - fund_np[s][lo:i + 1]))
            last_acc = i

    last_acc = 30 * bpd - 1

    def weight(symv, i):
        v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            return float(np.clip(mv / v, 0.5, 1.0)) * SLOT_FRAC * lev
        return SLOT_FRAC * lev

    def realize(t, i, frac, px):
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        net_arr[i] += frac * t["w0"] * sgn * (px / t["px0"] - 1.0)
        net_arr[i] -= frac * t["w0"] * fee_rt
        t["w"] -= frac * t["w0"]

    def close_trade(t, i, reason, px=None):
        c_i = float(closes[t["sym"]].iloc[i]) if px is None else px
        t.update(j0=i, exit_px=c_i, reason=reason)
        trades.append(t)
        last_close_i[t["sym"]] = i
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        recs.append({"sym": t["sym"], "dir": t["dir"], "i0": t["i0"], "j0": i,
                     "ret": sgn * (c_i / t["px0"] - 1.0)})

    def stop_px_of(t):
        if stop_pct is not None:
            return t["px0"] * (1.0 + stop_pct)
        if stop_atr is not None:
            return t["px0"] + stop_atr * t["atr0"]
        return None

    V5_TR = [(0.5, 0.5), (0.75, 0.25)]
    n_stop = 0
    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            expired = (i - t["i0"]) >= cap_h
            o, px0 = t["origin"], t["px0"]
            span = px0 - o if t["dir"] == "SHORT" else o - px0
            lo_i, hi_i = low_np[s][i], high_np[s][i]
            spx = stop_px_of(t)
            if spx is not None and t["w"] > 1e-12 and hi_i >= spx:
                # gap-through modeled: fill at max(prev close ~ open, stop)
                fill = max(float(closes[s].iloc[i - 1]), spx)
                realize(t, i, t["w"], fill)
                close_trade(t, i, "stop", px=fill)
                n_stop += 1
                continue
            if t["w"] > 1e-12:
                for ti, (rf, sf) in enumerate(V5_TR):
                    if ti in t["done"]:
                        continue
                    lvl = px0 - rf * span
                    if lo_i <= lvl:
                        realize(t, i, sf, lvl)
                        t["done"].add(ti)
            o_hit = lo_i <= o
            if expired:
                close_trade(t, i, "time")
            elif o_hit:
                close_trade(t, i, "origin", px=o)
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        gross_now = sum(t["w"] for t in open_tr)
        cands = []
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv not in pc_np:
                continue
            if symv in last_close_i and i - last_close_i[symv] < 12:
                continue
            if symv not in flow:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a or r < 0:
                continue
            v0, vma = vol_raw[symv][i], vol_ma[symv][i]
            if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                    and v0 >= 3.0 * vma):
                continue
            tfe = flow[symv][i]
            if tf_max is not None and not (np.isfinite(tfe)
                                           and tfe <= tf_max):
                continue
            if gate is not None:
                sc = TR.score_at(symv, i, "SHORT")
                if sc is None or sc <= gate:
                    continue
            cands.append((symv, a))
        if cands:
            room = slots - len(open_tr)
            for symv, a in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                w = weight(symv, i)
                w = min(w, gross_cap - gross_now)
                if w < 0.5 * SLOT_FRAC * lev:
                    continue
                gross_now += w
                net_arr[i] -= w * 2 * fee_rt
                open_tr.append({"sym": symv, "dir": "SHORT", "i0": i,
                                "px0": float(closes[symv].iloc[i]),
                                "origin": origin, "atr0": a, "w": w,
                                "w0": w, "done": set()})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return (pd.Series(net_arr, index=idx), trades, recs, n_stop)


# ------------------------------------------------------------------ CLIM 5x
def cont_sim8(tf=None, tf_min=0.60, trail_atr=1.0, cap_h=24, k=5.0,
              lev=LEV, stop_pct=0.08, gross_cap=GROSS_CAP):
    """CLIM v1 + hard floor stop (fill = min(open, stop_px), stop checked
    before the close-based trail)."""
    slots = SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades = [], []
    low_npv = {s: TR.panel.frames[s]["Low"].values for s in TR.panel.pairs}

    def accrue_to(i):
        nonlocal last_acc
        if i > last_acc:
            for t in open_tr:
                s = t["sym"]
                lo = max(last_acc + 1, t["i0"] + 1)
                if lo <= i:
                    net_arr[lo:i + 1] += (t["w"] * (pc_np[s][lo:i + 1]
                                                    - fund_np[s][lo:i + 1]))
            last_acc = i

    last_acc = 30 * bpd - 1

    def weight(symv, i):
        v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            return float(np.clip(mv / v, 0.5, 1.0)) * SLOT_FRAC * lev
        return SLOT_FRAC * lev

    def close_trade(t, i, reason, px=None):
        c_i = float(closes[t["sym"]].iloc[i]) if px is None else px
        t.update(j0=i, exit_px=c_i, reason=reason)
        trades.append(t)

    n_stop = 0
    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            t["hi"] = max(t["hi"], float(TR.panel.frames[s]["High"].iloc[i]))
            stop_px = t["px0"] * (1.0 - stop_pct)
            if low_npv[s][i] <= stop_px:
                # gap-through modeled: fill at min(prev close ~ open, stop)
                fill = min(float(closes[s].iloc[i - 1]), stop_px)
                t["w"] = 0.0
                close_trade(t, i, "stop", px=fill)
                n_stop += 1
                continue
            exit_now = c_i < t["hi"] - trail_atr * t["atr0"]
            if exit_now or (i - t["i0"]) >= cap_h:
                close_trade(t, i, "trail" if exit_now else "time")
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        gross_now = sum(t["w"] for t in open_tr)
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv not in pc_np:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a or r < 0:
                continue
            v0, vma = vol_raw[symv][i], vol_ma[symv][i]
            if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                    and v0 >= 3.0 * vma):
                continue
            if tf is not None:
                tfe = tf[symv][i]
                if not (np.isfinite(tfe) and tfe >= tf_min):
                    continue
            if len(open_tr) >= slots:
                break
            w = weight(symv, i)
            w = min(w, gross_cap - gross_now)
            if w < 0.5 * SLOT_FRAC * lev:
                continue
            gross_now += w
            net_arr[i] -= w * 2 * fee_rt
            open_tr.append({"sym": symv, "i0": i,
                            "px0": float(closes[symv].iloc[i]),
                            "atr0": a, "w": w, "w0": w,
                            "hi": float(TR.panel.frames[symv]["High"].iloc[i])})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, n_stop


def gross_series(trades):
    g = pd.Series(0.0, index=idx)
    for t in trades:
        g.iloc[t["i0"] + 1:t["j0"] + 1] += t.get("w0", t.get("w", 0.0))
    return g


def line(label, net, n=None, extra=""):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |{extra}", flush=True)


def trade_stats(trades, recs):
    """(stop_count, worst single trade in NAV frac, max adverse %)."""
    worst = 0.0
    max_adv = 0.0
    for t in trades:
        sgn = 1.0 if t.get("dir", "LONG") == "LONG" else -1.0
        pnl = t["w0"] * sgn * (t["exit_px"] / t["px0"] - 1.0)
        worst = min(worst, pnl)
        adv = (t["exit_px"] / t["px0"] - 1.0) if sgn < 0 \
            else (1.0 - t["exit_px"] / t["px0"])
        max_adv = max(max_adv, adv)
    nstop = sum(1 for t in trades if t.get("reason") == "stop")
    return nstop, worst, max_adv


# ---------------------------------------------------------------- battery a
def battery_a(tf):
    print("## Round 31a — FADE v7 @5x + hard stop sweep (24m)")
    print("stop checked intrabar, fill = max(open, stop); liq at 5x ~ 20% adverse")
    print("| variant | SR(bar) | daily-block CI | ret | maxDD | stops | worst trade |")
    print("|---|---|---|---|---|---|---|")
    net, tr, rc, ns = fade_sim8(tf=tf)
    ns2, worst, _ = trade_stats(tr, rc)
    line("no stop (v7 @5x)", net, len(rc),
         f" {ns2} | {worst:+.1%} |")
    for sa in (3.0, 4.0, 5.0, 6.0):
        net, tr, rc, ns = fade_sim8(tf=tf, stop_atr=sa)
        ns2, worst, adv = trade_stats(tr, rc)
        line(f"stop +{sa:.0f}xATR", net, len(rc),
             f" {ns2} | {worst:+.1%} |")
    for sp in (0.06, 0.08, 0.10, 0.12, 0.15):
        net, tr, rc, ns = fade_sim8(tf=tf, stop_pct=sp)
        ns2, worst, adv = trade_stats(tr, rc)
        line(f"stop +{sp:.0%}", net, len(rc),
             f" {ns2} | {worst:+.1%} |")


# ---------------------------------------------------------------- battery b
def battery_b(tf):
    print("## Round 31b — CLIM v1 @5x + hard floor-stop sweep (24m)")
    print("| variant | SR(bar) | daily-block CI | ret | maxDD | stops | worst trade |")
    print("|---|---|---|---|---|---|---|")
    for sp in (None, 0.05, 0.06, 0.08, 0.10):
        kw = {} if sp is None else {"stop_pct": sp}
        net, tr, ns = cont_sim8(tf=tf, **kw)
        ns2, worst, adv = trade_stats(tr, [])
        lab = "ATR trail only (v1 @5x)" if sp is None else f"trail + stop -{sp:.0%}"
        line(lab, net, len(tr), f" {ns2} | {worst:+.1%} |")


# ---------------------------------------------------------------- battery c
def battery_c():
    print("## Round 31c — SQUEEZE @5x close-stop sweep (24m, FS.sim native)")
    print("| variant | SR(bar) | daily-block CI | ret | maxDD | stops | worst trade |")
    print("|---|---|---|---|---|---|---|")
    old = FS.STOP
    try:
        for sp in (0.05, 0.08, 0.12):
            FS.STOP = sp
            r = FS.sim(squeeze2(), False, None, SQ_OVER)
            net5 = r["net"] * LEV
            tr = r.get("trades", [])
            nstop = sum(1 for t in tr if t.get("reason") == "stop")
            worst = min([t["w"] * ((t["exit_px"] / t["px0"] - 1.0)
                        if t["dir"] == "LONG" else
                        (t["px0"] / max(t["exit_px"], 1e-12) - 1.0))
                        for t in tr] or [0.0]) * LEV
            line(f"STOP {sp:.0%} @5x", net5, r["n"],
                 f" {nstop} | {worst:+.1%} |")
    finally:
        FS.STOP = old


# ---------------------------------------------------------------- battery d
def battery_d(tf, fade_kw, clim_kw, sq_stop):
    print(f"## Round 31d — 5x trio assembly (fade {fade_kw}, clim {clim_kw},"
          f" sq STOP {sq_stop:.0%})")
    fade_net, ftr, frc, _ = fade_sim8(tf=tf, **fade_kw)
    clim_net, ctr, _ = cont_sim8(tf=tf, **clim_kw)
    old = FS.STOP
    try:
        FS.STOP = sq_stop
        sq = FS.sim(squeeze2(), False, None, SQ_OVER)
    finally:
        FS.STOP = old
    sq5 = sq["net"] * LEV
    combo = 0.60 * fade_net + 0.25 * sq5 + 0.15 * clim_net
    print("| variant | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    line("FADE @5x + stop", fade_net, len(frc))
    line("SQUEEZE @5x + stop", sq5, sq["n"])
    line("CLIM @5x + stop", clim_net, len(ctr))
    line("TRIO 60/25/15 @5x", combo)
    half = len(combo) // 2
    line("  H1", combo.iloc[:half])
    line("  H2", combo.iloc[half:])
    cut = int(len(combo) * 0.7)
    line("  train70", combo.iloc[:cut])
    line("  test30", combo.iloc[cut:])
    m = combo.resample("MS").sum()
    print(f"  months: {int((m > 0).sum())}/{len(m)} positive | worst "
          f"{m.min():+.1%} | median {m.median():+.1%}", flush=True)
    g = gross_series(ftr) + gross_series(ctr) \
        + gross_series(sq.get("trades", [])) * LEV
    print(f"  gross NAV: mean {g.mean():.2f} p95 {g.quantile(0.95):.2f} "
          f"max {g.max():.2f} (cap {GROSS_CAP})", flush=True)
    ns_f, worst_f, adv_f = trade_stats(ftr, frc)
    ns_c, worst_c, adv_c = trade_stats(ctr, [])
    tr_sq = sq.get("trades", [])
    adv_sq = max([((t["px0"] / max(t["exit_px"], 1e-12) - 1.0)
                   if t["dir"] == "SHORT" else
                   (t["exit_px"] / t["px0"] - 1.0)) for t in tr_sq] or [0.0])
    print(f"  stops: fade {ns_f}/{len(ftr)} clim {ns_c}/{len(ctr)} "
          f"sq {sum(1 for t in tr_sq if t.get('reason') == 'stop')}/{len(tr_sq)}",
          flush=True)
    print(f"  worst trade NAV: fade {worst_f:+.1%} clim {worst_c:+.1%} "
          f"(margin per max slot = {0.625 / LEV:.1%}; liq needs 20% adverse)",
          flush=True)
    print(f"  max adverse: fade {adv_f:.1%} clim {adv_c:.1%} sq {adv_sq:.1%} "
          f"(all << 20% -> no liquidation)", flush=True)
    return combo, fade_net, sq5, clim_net


# ---------------------------------------------------------------- battery e
def fade_sim9(k=5.0, cap_h=42, tf=None, tf_max=0.60,
              lev=LEV, stop_pct=0.15, gross_cap=GROSS_CAP,
              max_slots=SLOTS, cluster_gate=None, breaker=None,
              stop_cool=12):
    """FADE @5x + fixed-% stop + cluster defenses:
    max_slots   cap concurrent fade positions (cluster exposure cap)
    cluster_gate skip entries when >= N simultaneous new spikes (squeeze day)
    breaker     no new entries while trailing-24h book return < -breaker
    stop_cool   per-symbol cooldown after a STOP (vs 12h after any close)"""
    flow = tf
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades, recs = [], [], []
    last_close_i, last_stop_i = {}, {}

    def accrue_to(i):
        nonlocal last_acc
        if i > last_acc:
            for t in open_tr:
                s = t["sym"]
                sgn = -1.0
                lo = max(last_acc + 1, t["i0"] + 1)
                if lo <= i:
                    net_arr[lo:i + 1] += (t["w"] * sgn *
                                          (pc_np[s][lo:i + 1]
                                           - fund_np[s][lo:i + 1]))
            last_acc = i

    last_acc = 30 * bpd - 1

    def weight(symv, i):
        v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            return float(np.clip(mv / v, 0.5, 1.0)) * SLOT_FRAC * lev
        return SLOT_FRAC * lev

    def realize(t, i, frac, px):
        net_arr[i] += frac * t["w0"] * -1.0 * (px / t["px0"] - 1.0)
        net_arr[i] -= frac * t["w0"] * fee_rt
        t["w"] -= frac * t["w0"]

    def close_trade(t, i, reason, px=None):
        c_i = float(closes[t["sym"]].iloc[i]) if px is None else px
        t.update(j0=i, exit_px=c_i, reason=reason)
        trades.append(t)
        last_close_i[t["sym"]] = i
        if reason == "stop":
            last_stop_i[t["sym"]] = i
        recs.append({"sym": t["sym"], "i0": t["i0"], "j0": i,
                     "ret": -(c_i / t["px0"] - 1.0)})

    V5_TR = [(0.5, 0.5), (0.75, 0.25)]
    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            expired = (i - t["i0"]) >= cap_h
            o, px0 = t["origin"], t["px0"]
            span = px0 - o
            lo_i, hi_i = low_np[s][i], high_np[s][i]
            spx = px0 * (1.0 + stop_pct)
            if t["w"] > 1e-12 and hi_i >= spx:
                fill = max(float(closes[s].iloc[i - 1]), spx)
                realize(t, i, t["w"], fill)
                close_trade(t, i, "stop", px=fill)
                continue
            if t["w"] > 1e-12:
                for ti, (rf, sf) in enumerate(V5_TR):
                    if ti in t["done"]:
                        continue
                    lvl = px0 - rf * span
                    if lo_i <= lvl:
                        realize(t, i, sf, lvl)
                        t["done"].add(ti)
            if expired:
                close_trade(t, i, "time")
            elif lo_i <= o:
                close_trade(t, i, "origin", px=o)
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        gross_now = sum(t["w"] for t in open_tr)
        blocked = len(open_tr) >= max_slots
        if breaker is not None and i >= 24 \
                and net_arr[i - 24:i].sum() < -breaker:
            blocked = True
        cands = []
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv not in pc_np:
                continue
            if symv in last_close_i and i - last_close_i[symv] < 12:
                continue
            if symv in last_stop_i and i - last_stop_i[symv] < stop_cool:
                continue
            if symv not in flow:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a or r < 0:
                continue
            v0, vma = vol_raw[symv][i], vol_ma[symv][i]
            if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                    and v0 >= 3.0 * vma):
                continue
            tfe = flow[symv][i]
            if tf_max is not None and not (np.isfinite(tfe)
                                           and tfe <= tf_max):
                continue
            cands.append((symv, a))
        if cluster_gate is not None and len(cands) >= cluster_gate:
            cands = []
        if cands and not blocked:
            room = max_slots - len(open_tr)
            for symv, a in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                w = weight(symv, i)
                w = min(w, gross_cap - gross_now)
                if w < 0.5 * SLOT_FRAC * lev:
                    continue
                gross_now += w
                net_arr[i] -= w * 2 * fee_rt
                open_tr.append({"sym": symv, "dir": "SHORT", "i0": i,
                                "px0": float(closes[symv].iloc[i]),
                                "origin": origin, "atr0": a, "w": w,
                                "w0": w, "done": set()})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, recs


# ---------------------------------------------------------------- battery f
def sq_stop_hybrid(stop_h, lev=LEV):
    """SQUEEZE @5x with the daily book's trades truncated by an HOURLY
    intrabar stop (fills gap-aware: min/max(prev close, stop))."""
    r = FS.sim(squeeze2(), False, None, SQ_OVER)
    net = pd.Series(0.0, index=idx)
    out_tr = []
    for t in r["trades"]:
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        s, i0, j0, w = t["sym"], t["i0"], t["j0"], t["w"]
        px0 = t["px0"]
        stop_px = px0 * (1.0 - stop_h) if sgn > 0 else px0 * (1.0 + stop_h)
        lo = i0 + 1
        hit = None
        for k in range(lo, min(j0, len(idx) - 1) + 1):
            l_i = TR.panel.frames[s]["Low"].iloc[k]
            h_i = TR.panel.frames[s]["High"].iloc[k]
            if (sgn > 0 and l_i <= stop_px) or (sgn < 0 and h_i >= stop_px):
                hit = k
                break
        # mark-to-market path up to the bar BEFORE the stop bar; the stop
        # bar is realized at the gap-aware fill (avoids double counting)
        end = (hit - 1) if hit is not None else j0
        if lo <= end:
            net.iloc[lo:end + 1] += (w * sgn * pc_np[s][lo:end + 1]
                                     - w * fund_np[s][lo:end + 1])
        if hit is not None:
            prev_c = float(closes[s].iloc[hit - 1])
            fill = min(prev_c, stop_px) if sgn > 0 else max(prev_c, stop_px)
            net.iloc[hit] += w * sgn * (fill / prev_c - 1.0) - w * 2 * fee_rt
        else:
            net.iloc[i0] += -w * 2 * fee_rt
        t2 = dict(t)
        if hit is not None:
            t2["j0"], t2["exit_px"], t2["reason"] = hit, fill, "stop"
        out_tr.append(t2)
    net *= lev
    for t in out_tr:
        t["w"] = t["w"] * lev
    return net, out_tr, r["n"]


def battery_e(tf):
    print("## Round 31e — FADE @5x stop 15% + cluster defenses (24m)")
    print("| variant | SR(bar) | daily-block CI | ret | maxDD | trades | stops | worst trade |")
    print("|---|---|---|---|---|---|---|---|")
    net, tr, rc = fade_sim9(tf=tf)
    ns, worst, adv = trade_stats(tr, rc)
    line("baseline slots8 (from 31a)", net, len(rc), f" {ns} | {worst:+.1%} |")
    for ms in (4, 6):
        net, tr, rc = fade_sim9(tf=tf, max_slots=ms)
        ns, worst, adv = trade_stats(tr, rc)
        line(f"max_slots {ms}", net, len(rc), f" {ns} | {worst:+.1%} |")
    for cg in (3, 5):
        net, tr, rc = fade_sim9(tf=tf, cluster_gate=cg)
        ns, worst, adv = trade_stats(tr, rc)
        line(f"cluster_gate {cg}", net, len(rc), f" {ns} | {worst:+.1%} |")
    for bk in (0.04, 0.06):
        net, tr, rc = fade_sim9(tf=tf, breaker=bk)
        ns, worst, adv = trade_stats(tr, rc)
        line(f"breaker -{bk:.0%}/24h", net, len(rc),
             f" {ns} | {worst:+.1%} |")
    net, tr, rc = fade_sim9(tf=tf, stop_cool=48)
    ns, worst, adv = trade_stats(tr, rc)
    line("stop_cool 48h", net, len(rc), f" {ns} | {worst:+.1%} |")
    net, tr, rc = fade_sim9(tf=tf, max_slots=4, cluster_gate=5)
    ns, worst, adv = trade_stats(tr, rc)
    line("slots4 + cluster5", net, len(rc), f" {ns} | {worst:+.1%} |")


def battery_f():
    print("## Round 31f — SQUEEZE @5x + hourly intrabar stop (24m)")
    print("| variant | SR(bar) | daily-block CI | ret | maxDD | stops | worst trade |")
    print("|---|---|---|---|---|---|---|")
    for sh in (0.05, 0.08, 0.12):
        net, tr, n = sq_stop_hybrid(sh)
        nstop = sum(1 for t in tr if t.get("reason") == "stop")
        worst = min([t["w"] * ((t["exit_px"] / t["px0"] - 1.0)
                     if t["dir"] == "LONG" else
                     (t["px0"] / max(t["exit_px"], 1e-12) - 1.0))
                     for t in tr] or [0.0])
        line(f"hourly stop {sh:.0%} @5x", net, n, f" {nstop} | {worst:+.1%} |")


def battery_g(tf, slots=8, cg=None, sh=0.05, sq_lev=2.0):
    print(f"## Round 31g — levered trio assembly (fade slots {slots},"
          f" cluster {cg}, sq hourly stop {sh:.0%} @ {sq_lev}x,"
          f" fade/clim @ {LEV}x)")
    fade_net, ftr, frc = fade_sim9(tf=tf, max_slots=slots, cluster_gate=cg)
    clim_net, ctr, _ = cont_sim8(tf=tf, stop_pct=0.06)
    sq5, str_, sqn = sq_stop_hybrid(sh, lev=sq_lev)
    combo = 0.60 * fade_net + 0.25 * sq5 + 0.15 * clim_net
    print("| variant | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    line("FADE @5x stop15", fade_net, len(frc))
    line(f"SQUEEZE @{sq_lev}x hourly-stop", sq5, sqn)
    line("CLIM @5x trail+floor6", clim_net, len(ctr))
    line("TRIO 60/25/15 levered", combo)
    half = len(combo) // 2
    line("  H1", combo.iloc[:half])
    line("  H2", combo.iloc[half:])
    cut = int(len(combo) * 0.7)
    line("  train70", combo.iloc[:cut])
    line("  test30", combo.iloc[cut:])
    m = combo.resample("MS").sum()
    print(f"  months: {int((m > 0).sum())}/{len(m)} positive | worst "
          f"{m.min():+.1%} | median {m.median():+.1%}", flush=True)
    g = gross_series(ftr) + gross_series(ctr) + gross_series(str_)
    print(f"  gross NAV: mean {g.mean():.2f} p95 {g.quantile(0.95):.2f} "
          f"max {g.max():.2f}", flush=True)
    _, worst_f, adv_f = trade_stats(ftr, frc)
    _, worst_c, adv_c = trade_stats(ctr, [])
    worst_s = min([t["w"] * ((t["exit_px"] / t["px0"] - 1.0)
                   if t["dir"] == "LONG" else
                   (t["px0"] / max(t["exit_px"], 1e-12) - 1.0))
                   for t in str_] or [0.0])
    adv_s = max([((1.0 - t["exit_px"] / t["px0"]) if t["dir"] == "LONG" else
                  (t["exit_px"] / t["px0"] - 1.0))
                 for t in str_] or [0.0])
    print(f"  worst trade NAV: fade {worst_f:+.1%} clim {worst_c:+.1%} "
          f"sq {worst_s:+.1%} | max adverse: fade {adv_f:.1%} "
          f"clim {adv_c:.1%} sq {adv_s:.1%} (liq needs 20%)", flush=True)
    return combo


def battery_h(tf):
    print("## Round 31h — leverage sweep 1x..5x (v9 config: FADE stop15% +"
          " cluster3, SQUEEZE hourly stop 5%, CLIM trail + floor 6%)")
    print("| lev | book | SR(bar) | daily-block CI | ret | maxDD | stops |"
          " worst trade |")
    print("|---|---|---|---|---|---|---|---|")
    trios = {}
    for lev in (1.0, 2.0, 3.0, 4.0, 5.0):
        f_net, f_tr, f_rc = fade_sim9(tf=tf, cluster_gate=3, lev=lev)
        c_net, c_tr, _ = cont_sim8(tf=tf, stop_pct=0.06, lev=lev)
        sq, s_tr, sq_n = sq_stop_hybrid(0.05, lev=lev)
        trio = 0.60 * f_net + 0.25 * sq + 0.15 * c_net
        trios[lev] = trio
        ns_f, worst_f, adv_f = trade_stats(f_tr, f_rc)
        ns_c, worst_c, adv_c = trade_stats(c_tr, [])
        ns_s = sum(1 for t in s_tr if t.get("reason") == "stop")
        worst_s = min([t["w"] * ((t["exit_px"] / t["px0"] - 1.0)
                       if t["dir"] == "LONG" else
                       (t["px0"] / max(t["exit_px"], 1e-12) - 1.0))
                       for t in s_tr] or [0.0])
        line(f"{lev:.0f}x FADE", f_net, len(f_rc),
             f" {ns_f} | {worst_f:+.1%} |")
        line(f"{lev:.0f}x SQUEEZE", sq, sq_n, f" {ns_s} | {worst_s:+.1%} |")
        line(f"{lev:.0f}x CLIM", c_net, len(c_tr),
             f" {ns_c} | {worst_c:+.1%} |")
        line(f"{lev:.0f}x TRIO", trio)
        g = gross_series(f_tr) + gross_series(c_tr) + gross_series(s_tr)
        m = trio.resample("MS").sum()
        print(f"  gross mean {g.mean():.2f} p95 {g.quantile(0.95):.2f} max "
              f"{g.max():.2f} | months {int((m > 0).sum())}/{len(m)} "
              f"worst {m.min():+.1%} | max adverse fade {adv_f:.1%} "
              f"clim {adv_c:.1%}", flush=True)
    print()
    print("## Round 31h summary — TRIO across leverage")
    print("| lev | SR(bar) | ret | maxDD | ret/DD |")
    print("|---|---|---|---|---|")
    for lev, trio in trios.items():
        dd = max_dd_of(trio)
        days = len(trio) / 24
        r = trio.sum() * 365 / days
        lo, hi, pn = bootstrap_sharpe_ci(trio)
        print(f"| {lev:.0f}x | {sharpe_of(trio):.2f} [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} | {r:+.0%}/yr | {dd:.1%} | {r / -dd:.1f} |",
              flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("c", "all"):
        battery_c()
    if mode in ("e", "all"):
        battery_e(load_flow())
    if mode in ("f", "all"):
        battery_f()
    if mode in ("g", "all"):
        battery_g(load_flow(), slots=8, cg=3)
    if mode in ("h", "all"):
        battery_h(load_flow())


if __name__ == "__main__":
    main()
