"""Round 17: post-v5 exploration battery (24m, production v5 baseline).

Battery A — FADE microstructure (base = production v5):
  A0 production baseline (close-based tranche fills)
  A1 intrabar limit fills: tranches + origin as resting limit orders
     (realize on wick touch at the limit price, not on hourly close)
  A2 re-entry control: cooldown 6h/12h since last close on sym;
     new-high re-entry (close beyond previous entry price)
  A3 hour-of-day entry windows (UTC): 0-12 / 12-24 / 18-24
  A4 slots 4 / 12 (production 8)

Battery B — SQUEEZE conditioning (base = production squeeze v2):
  B1 compression duration >= 6 / 18 4h-bars before trigger
  B2 bandwidth tightness percentile <= 30% (trailing 180 bars)
  B3 volume dry-up (vol <= 0.8x trailing 36-bar mean at trigger)
  B4 stop sweep 4% / 6% (FS.STOP) and cap 10d / 14d (FS.CAP_BARS)
  B5 new family: failed-breakout fade (squeeze -> break -> close back
     inside -> fade the failure, mid-band target)

Battery C — book allocation (post-hoc on production books):
  C1 pair vol targeting (scale to full-sample median vol, 1d lag)
  C2 inverse-vol dynamic weights (60d trailing, 1d lag)

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate17.py [a|b|c]
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import (idx, closes, bpd, in_u_all, pc_np, fund_np, SLOTS,
                       SLOT_FRAC, BTC, fee_rt, atr1_np, vol_raw, vol_ma, report)
from iterate14 import fade_sim2, squeeze2, SQ_OVER, low_np, high_np
import family_scan_honest as FS
from pair_scout.research_ta_families import _states_to_signals
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

V5_TR = [(0.5, 0.5), (0.75, 0.25)]


# ---------------------------------------------------------------- battery A
def fade_sim3(k=5.0, ivol=True, direction_filter="spike", gate=None,
              vol_min=3.0, cap_h=42, slots=None, tranches=None,
              hour_win=None, cooldown_h=None, newx=False, touch=False):
    """Production v5 sim + round-17 microstructure knobs.

    hour_win: set of UTC hours allowed for NEW entries.
    cooldown_h: min hours between a symbol's trade close and next entry.
    newx: re-entry only if close is beyond the previous entry price
    (short: above it; long: below it).
    touch: tranche levels and origin exit are RESTING LIMITS filled on
    intrabar wick touch at the limit price (instead of hourly close)."""
    slots = slots or SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades, recs = [], [], []
    last_close_i, last_px0 = {}, {}

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
        if ivol:
            v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
            if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
                return float(np.clip((mv / v) * SLOT_FRAC,
                                     0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
        return SLOT_FRAC

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
        last_px0[t["sym"]] = t["px0"]
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        recs.append({"sym": t["sym"], "dir": t["dir"], "i0": t["i0"], "j0": i,
                     "ret": sgn * (c_i / t["px0"] - 1.0)})

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
            if t.get("plan") and t["w"] > 1e-12:
                for ti, (rf, sf) in enumerate(t["plan"]):
                    if ti in t["done"]:
                        continue
                    lvl = px0 - rf * span if t["dir"] == "SHORT" \
                        else px0 + rf * span
                    hit = ((lo_i <= lvl) if t["dir"] == "SHORT"
                           else (hi_i >= lvl)) if touch else \
                          ((c_i <= lvl) if t["dir"] == "SHORT"
                           else (c_i >= lvl))
                    if hit:
                        realize(t, i, sf, lvl if touch else c_i)
                        t["done"].add(ti)
            o_hit = ((lo_i <= o) if t["dir"] == "SHORT" else (hi_i >= o)) \
                if touch else ((c_i <= o) if t["dir"] == "SHORT"
                               else (c_i >= o))
            if expired:
                close_trade(t, i, "time")
            elif o_hit:
                close_trade(t, i, "origin", px=o if touch else None)
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        cands = []
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv not in pc_np:
                continue
            if hour_win is not None and idx[i].hour not in hour_win:
                continue
            if cooldown_h is not None and symv in last_close_i \
                    and i - last_close_i[symv] < cooldown_h:
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
                v0, vma = vol_raw[symv][i], vol_ma[symv][i]
                if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                        and v0 >= vol_min * vma):
                    continue
            direction = "SHORT" if r > 0 else "LONG"
            if newx and symv in last_px0:
                px_prev = last_px0[symv]
                c_now = float(closes[symv].iloc[i])
                if direction == "SHORT" and not c_now > px_prev:
                    continue
                if direction == "LONG" and not c_now < px_prev:
                    continue
            if gate is not None:
                sc = TR.score_at(symv, i, direction)
                if sc is None or sc <= gate:
                    continue
            if direction_filter == "spike" and direction != "SHORT":
                continue
            cands.append((symv, direction, a))
        if cands:
            room = slots - len(open_tr)
            for symv, direction, a in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                w = weight(symv, i)
                net_arr[i] -= w * 2 * fee_rt
                open_tr.append({"sym": symv, "dir": direction, "i0": i,
                                "px0": float(closes[symv].iloc[i]),
                                "origin": origin, "atr0": a, "w": w,
                                "w0": w, "done": set(),
                                "plan": list(V5_TR if tranches is None
                                             else tranches)})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, recs


def battery_a():
    print("## Round 17a — FADE microstructure (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, _, rc = fade_sim3()
    report("PRODUCTION v5 (close-based fills)", net, len(rc))
    net, _, rc = fade_sim3(touch=True)
    report("A1 resting-limit fills (tranches+origin on wick)", net, len(rc))
    for ch in (6, 12):
        net, _, rc = fade_sim3(touch=True, cooldown_h=ch)
        report(f"A2 cooldown {ch}h after sym close", net, len(rc))
    net, _, rc = fade_sim3(touch=True, newx=True)
    report("A2 new-high re-entry only", net, len(rc))
    for hw in (range(0, 12), range(12, 24), range(18, 24)):
        net, _, rc = fade_sim3(touch=True, hour_win=set(hw))
        report(f"A3 entries {hw.start:02d}-{hw.stop:02d} UTC", net, len(rc))
    for sl in (4, 12):
        net, _, rc = fade_sim3(touch=True, slots=sl)
        report(f"A4 slots {sl}", net, len(rc))


# ---------------------------------------------------------------- battery B
def squeeze3(kc_mult=2.0, bb_len_d=20, exit_kind="mid", min_sq_bars=None,
             bw_pct=None, vol_dry=None):
    """squeeze2 + trigger conditioning:
    min_sq_bars: compression must have been ON >= N consecutive 4h bars.
    bw_pct: bandwidth must sit in bottom pct of trailing 180 bars.
    vol_dry: volume at trigger <= vol_dry x trailing 36-bar mean."""
    h4, l4, c4, v4 = FS._resample_ohlcv(FS.panel)
    mid = FS._ema(c4, 20 * 6)
    atr = FS._atr(h4, l4, c4, 14 * 6)
    up, dn = mid + kc_mult * atr, mid - kc_mult * atr
    sd = c4.rolling(bb_len_d * 6).std(ddof=0)
    bup, bdn = mid + 2 * sd, mid - 2 * sd
    sq = (bup < up) & (bdn > dn)
    sq1 = sq.shift(1).fillna(False)
    if min_sq_bars is not None:
        run = sq1.astype(float).rolling(min_sq_bars, min_periods=min_sq_bars)
        sq1 = sq1 & (run.min() == 1.0)
    if bw_pct is not None:
        bw = (bup - bdn) / mid
        pct = bw.rolling(180, min_periods=60).rank(pct=True)
        sq1 = sq1 & (pct <= bw_pct)
    if vol_dry is not None:
        vr = v4 / v4.rolling(36).mean()
        sq1 = sq1 & (vr <= vol_dry)
    la_trig = (c4 > bup) & sq1
    sa_trig = (c4 < bdn) & sq1
    if exit_kind == "mid":
        hold_l, hold_s = c4 >= mid, c4 <= mid
    else:
        hold_l, hold_s = c4 > bdn, c4 < bup

    def persist(trig, hold):
        seg = trig.cumsum()
        ever_off = (~hold.fillna(False)).groupby(seg).cummax()
        return trig | ((seg > 0) & ~ever_off)

    la = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    sa = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    for col in c4.columns:
        la[col] = persist(la_trig[col], hold_l[col])
        sa[col] = persist(sa_trig[col], hold_s[col])
    return _states_to_signals(la, ~la, sa, ~sa, idx, c4=c4)


def battery_b():
    print("## Round 17b — SQUEEZE conditioning (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    r0 = FS.sim(squeeze2(), False, None, SQ_OVER)
    report("PRODUCTION squeeze v2", r0["net"], r0["n"])
    for nb in (6, 18):
        r = FS.sim(squeeze3(min_sq_bars=nb), False, None, SQ_OVER)
        report(f"B1 compression >= {nb} bars before trigger", r["net"], r["n"])
    r = FS.sim(squeeze3(bw_pct=0.3), False, None, SQ_OVER)
    report("B2 tightness pct <= 30%", r["net"], r["n"])
    r = FS.sim(squeeze3(vol_dry=0.8), False, None, SQ_OVER)
    report("B3 volume dry-up <= 0.8x at trigger", r["net"], r["n"])
    for st in (0.04, 0.06):
        FS.STOP = st
        r = FS.sim(squeeze2(), False, None, SQ_OVER)
        report(f"B4 stop {st:.0%}", r["net"], r["n"])
    FS.STOP = 0.05
    for cb in (240, 336):
        FS.CAP_BARS = cb
        r = FS.sim(squeeze2(), False, None, SQ_OVER)
        report(f"B4 cap {cb // 24}d", r["net"], r["n"])
    FS.CAP_BARS = 168
    # B5 failed-breakout fade family
    h4, l4, c4, v4 = FS._resample_ohlcv(FS.panel)
    mid = FS._ema(c4, 20 * 6)
    atr = FS._atr(h4, l4, c4, 14 * 6)
    up, dn = mid + 2 * atr, mid - 2 * atr
    sd = c4.rolling(20 * 6).std(ddof=0)
    bup, bdn = mid + 2 * sd, mid - 2 * sd
    sq = (bup < up) & (bdn > dn)
    up_trig = (c4 > bup) & sq.shift(1).fillna(False)
    dn_trig = (c4 < bdn) & sq.shift(1).fillna(False)
    rec_up = up_trig.rolling(6, min_periods=1).max().fillna(0) > 0
    rec_dn = dn_trig.rolling(6, min_periods=1).max().fillna(0) > 0
    fail_s = rec_up & (c4 < bup) & (c4 > mid)
    fail_l = rec_dn & (c4 > bdn) & (c4 < mid)
    age = c4.copy()
    age[:] = np.arange(len(c4))[:, None]
    age_s = age - age.where(fail_s).ffill()
    age_l = age - age.where(fail_l).ffill()
    hold_s = (c4 >= mid) & (age_s <= 60).fillna(False)
    hold_l = (c4 <= mid) & (age_l <= 60).fillna(False)
    la = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    sa = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    for col in c4.columns:
        la[col] = persist_state(fail_l[col], hold_l[col])
        sa[col] = persist_state(fail_s[col], hold_s[col])
    sig = _states_to_signals(la, ~la, sa, ~sa, idx, c4=c4)
    r = FS.sim(sig, False, None, SQ_OVER)
    report("B5 failed-breakout fade (new family)", r["net"], r["n"])


def persist_state(trig, hold):
    seg = trig.cumsum()
    ever_off = (~hold.fillna(False)).groupby(seg).cummax()
    return trig | ((seg > 0) & ~ever_off)


# ---------------------------------------------------------------- battery C
def battery_c():
    print("## Round 17c — book allocation on production books (24m)")
    print("| variant | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    f_net, _, rc_f = fade_sim2(cap_h=42, tranches=V5_TR)
    sq = FS.sim(squeeze2(), False, None, SQ_OVER)
    s_net = sq["net"]
    combo = 0.5 * s_net + 0.5 * f_net

    def rep(label, net):
        lo, hi, pn = bootstrap_sharpe_ci(net)
        days = len(net) / 24
        print(f"| {label} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
              f"{max_dd_of(net):.1%} |", flush=True)

    rep("PRODUCTION pair 50/50", combo)
    rv = combo.rolling(480).std()
    scale = (rv.median() / rv).shift(24).clip(0.25, 1.5).fillna(1.0)
    rep("C1 vol-targeted (20d, lag 1d)", combo * scale)
    vf = f_net.rolling(1440).std()
    vs = s_net.rolling(1440).std()
    wf = (1 / vf) / (1 / vf + 1 / vs)
    wf = wf.shift(24).clip(0.25, 0.75).fillna(0.5)
    rep("C2 inverse-vol weights (60d, lag 1d)", wf * f_net + (1 - wf) * s_net)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "a"
    if mode == "a":
        battery_a()
    elif mode == "b":
        battery_b()
    elif mode == "c":
        battery_c()
