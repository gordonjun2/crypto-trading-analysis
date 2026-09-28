"""Round 14: continuous-improvement battery on the production pair.

Battery A (FADE v3 hourly refinements, 24m, production config as base):
- limit-entry depth sweep (0.5/1.0/1.5 xATR) and window (6h/12h)
- next-bar execution (enter at bar+1 close; execution-realism check)
- k x vol_min plateau fill (4..6 x 2.5..4)
- scale-out exit: realize half at 50% retracement, half at origin (cap 24h)

Battery B (SQUEEZE daily validation, 24m):
- parameter plateau: KC mult 1.5/2.0/2.5, BB len 20/30
- exit variants: mid-band (current) | opposite band | 10d time stop
- direction split: long-only | short-only | both
- flash interaction: skip squeeze breakouts whose trigger bar was a fade flash

Battery C (new daily families, 24m):
- funding momentum (trade WITH sustained funding extremes)
- volume breakout (vol z>3 WITH price direction, not against)
- cross-sectional momentum (7d, dollar-neutral deciles)
- BTC lead-lag (BTC 24h sign -> alts next bar)

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate14.py [a|b|c]
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import (idx, closes, bpd, in_u_all, pc_np, fund_np, SLOTS,
                       SLOT_FRAC, BTC, fee_rt, atr1_np, vol_raw, vol_ma, report)
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

low_np = {s: TR.panel.frames[s]["Low"].values for s in TR.panel.pairs}
high_np = {s: TR.panel.frames[s]["High"].values for s in TR.panel.pairs}


def fade_sim2(k=5.0, ivol=True, exit_mode="origin", entry_mode="market",
              direction_filter="spike", gate=None, vol_min=3.0,
              lim_depth=1.0, lim_h=6, scale_at=None, w_cap=2.0,
              cap_h=24, slots=None, tranches=None, adaptive=None,
              fund_min=None, btc_quiet=False):
    """Extended fade sim (production defaults = FADE v4 when
    scale_at=0.75, cap_h=42).

    entry_mode: market | limit (lim_depth xATR deeper, lim_h window) |
    next (enter at the NEXT bar close, event must still hold).
    scale_at: fraction of the flash move to retrace before realizing half
    the position (None = no scale-out).
    tranches: list of (retrace_frac, size_frac) realized in order —
    supersedes scale_at.
    adaptive: 'volx' -> scale level by entry vol excess bucket
    (3-5x: 50%, 5-10x: 75%, >10x: 90%).
    fund_min: last funding >= fund_min to enter. btc_quiet: |BTC ret| <
    1x BTC ATR (idiosyncratic events only)."""
    if tranches is None and scale_at is not None:
        tranches = [(scale_at, 0.5)]
    slots = slots or SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades, recs = [], [], []
    pend = []  # (expire_i, sym, dir, kind, price_or_None, meta)
    last_acc = 30 * bpd - 1

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

    def weight(symv, i, r, a):
        if ivol:
            v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
            if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
                return float(np.clip((mv / v) * SLOT_FRAC,
                                     0.5 * SLOT_FRAC, w_cap * SLOT_FRAC))
        return SLOT_FRAC

    def open_pos(symv, direction, px, i, meta):
        w = weight(symv, i, meta["r"], meta["a"])
        net_arr[i] -= w * 2 * fee_rt
        open_tr.append({"sym": symv, "dir": direction, "i0": i, "px0": px,
                        "origin": meta["origin"], "atr0": meta["a"], "w": w,
                        "w0": w, "done": set()})

    def realize(t, i, frac):
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        c_i = float(closes[t["sym"]].iloc[i])
        net_arr[i] += frac * t["w0"] * sgn * (c_i / t["px0"] - 1.0)
        net_arr[i] -= frac * t["w0"] * fee_rt
        t["w"] -= frac * t["w0"]

    def close_trade(t, i, reason):
        c_i = float(closes[t["sym"]].iloc[i])
        t.update(j0=i, exit_px=c_i, reason=reason)
        trades.append(t)
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
            if t.get("plan"):
                for ti, (rf, sf) in enumerate(t["plan"]):
                    if ti in t["done"] or t["w"] <= 1e-12:
                        continue
                    lvl = px0 - rf * span if t["dir"] == "SHORT" \
                        else px0 + rf * span
                    if (c_i <= lvl) if t["dir"] == "SHORT" else (c_i >= lvl):
                        realize(t, i, sf)
                        t["done"].add(ti)
            hit = ((c_i >= o) if t["dir"] == "LONG" else (c_i <= o))
            if expired:
                close_trade(t, i, "time")
            elif hit:
                close_trade(t, i, "origin")
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        pend_syms = {p[1] for p in pend}
        # pending order fills
        fill, keep = [], []
        for exp_i, symv, direction, kind, lim, meta in pend:
            if i > exp_i or symv in held or len(open_tr) >= slots:
                continue
            c_i = float(closes[symv].iloc[i])
            if kind == "next":
                r2 = pc_np[symv][i]
                ok = ((r2 > 0) if direction == "SHORT" else (r2 < 0))
                if ok:
                    fill.append((symv, direction, c_i, meta))
            else:
                hitp = (low_np[symv][i] <= lim if direction == "LONG"
                        else high_np[symv][i] >= lim)
                if hitp:
                    fill.append((symv, direction, lim, meta))
                else:
                    keep.append((exp_i, symv, direction, kind, lim, meta))
        pend = keep
        for symv, direction, px, meta in fill:
            open_pos(symv, direction, px, i, meta)
            held.add(symv)
        # flash events
        cands = []
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv in pend_syms \
                    or symv not in pc_np:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a:
                continue
            v0 = vma = np.nan
            if vol_min is not None:
                v0, vma = vol_raw[symv][i], vol_ma[symv][i]
                if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                        and v0 >= vol_min * vma):
                    continue
            if fund_min is not None:
                lf = TR.last_fund[symv].iloc[i] if symv in TR.last_fund.columns \
                    else 0.0
                if not (np.isfinite(lf) and lf >= fund_min):
                    continue
            if btc_quiet:
                br = pc_np[BTC][i]
                if not (np.isfinite(br) and abs(br) < atr1_np[BTC][i]):
                    continue
            direction = "SHORT" if r > 0 else "LONG"
            if gate is not None:
                sc = TR.score_at(symv, i, direction)
                if sc is None or sc <= gate:
                    continue
            if direction_filter == "dump" and direction != "LONG":
                continue
            if direction_filter == "spike" and direction != "SHORT":
                continue
            cands.append((symv, direction, r, a, v0, vma))
        if cands:
            room = slots - len(open_tr)
            for symv, direction, r, a, v0, vma in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                meta = {"r": r, "a": a, "origin": origin}
                plan = None
                if adaptive == "volx" and np.isfinite(v0) and \
                        np.isfinite(vma) and vma > 0:
                    vx = v0 / vma
                    lvl = 0.5 if vx < 5 else (0.75 if vx < 10 else 0.9)
                    plan = [(lvl, 0.5)]
                elif tranches:
                    plan = list(tranches)
                if entry_mode == "limit":
                    lim = (float(closes[symv].iloc[i]) * (1.0 + lim_depth * a)
                           if direction == "SHORT"
                           else float(closes[symv].iloc[i]) * (1.0 - lim_depth * a))
                    pend.append((i + lim_h, symv, direction, "limit", lim,
                                 dict(meta, plan=plan)))
                elif entry_mode == "next":
                    pend.append((i + 1, symv, direction, "next", None,
                                 dict(meta, plan=plan)))
                else:
                    w = weight(symv, i, r, a)
                    net_arr[i] -= w * 2 * fee_rt
                    open_tr.append({"sym": symv, "dir": direction, "i0": i,
                                    "px0": float(closes[symv].iloc[i]),
                                    "origin": origin, "atr0": a, "w": w,
                                    "w0": w, "done": set(), "plan": plan})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, recs


def battery_a():
    print("## Round 14a — FADE v3 refinements (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net0, _, rc0 = fade_sim2()
    report("PRODUCTION base (market, origin, ivol)", net0, len(rc0))
    for d in (0.5, 1.0, 1.5):
        net, _, rc = fade_sim2(entry_mode="limit", lim_depth=d, lim_h=6)
        report(f"limit {d}xATR deeper, 6h window", net, len(rc))
    net, _, rc = fade_sim2(entry_mode="limit", lim_depth=1.0, lim_h=12)
    report("limit 1xATR deeper, 12h window", net, len(rc))
    net, _, rc = fade_sim2(entry_mode="next")
    report("next-bar execution", net, len(rc))
    for kk, vv in ((4, 3.0), (5, 2.5), (5, 4.0), (6, 3.0)):
        net, _, rc = fade_sim2(k=float(kk), vol_min=vv)
        report(f"plateau k={kk}xATR, vol>={vv}x", net, len(rc))
    for sa in (0.5, 0.75):
        net, _, rc = fade_sim2(scale_at=sa)
        report(f"scale-out at {sa:.0%} retracement", net, len(rc))


# ---------------------------------------------------------------- battery B
import family_scan_honest as FS
from pair_scout.research_ta_families import _states_to_signals

SQ_OVER = frozenset(("ivol", "voltarget", "riskoff"))


def squeeze2(kc_mult=2.0, bb_len_d=20, exit_kind="mid", long_only=False,
             short_only=False, skip_flash=False):
    """Parametrized squeeze family on the 4h grid -> hourly signals.

    NOTE: FS.sim exits on state DEACTIVATION (plus stop/cap), so the exit
    rule is embedded in the state: the long state persists while the
    position thesis holds and drops on the exit trigger."""
    h4, l4, c4, v4 = FS._resample_ohlcv(FS.panel)
    mid = FS._ema(c4, 20 * 6)
    atr = FS._atr(h4, l4, c4, 14 * 6)
    up, dn = mid + kc_mult * atr, mid - kc_mult * atr
    sd = c4.rolling(bb_len_d * 6).std(ddof=0)
    bup, bdn = mid + 2 * sd, mid - 2 * sd
    sq = (bup < up) & (bdn > dn)
    r4 = c4.pct_change()
    la_trig = (c4 > bup) & sq.shift(1).fillna(False)
    sa_trig = (c4 < bdn) & sq.shift(1).fillna(False)
    if skip_flash:
        a4 = FS._atr(h4, l4, c4, 24)
        flash = ((r4.abs() > 5 * a4)
                 & (v4 > 3 * v4.rolling(36).mean())).fillna(False)
        la_trig, sa_trig = la_trig & ~flash, sa_trig & ~flash
    def persist(trig, hold):
        """Entry state stays on until `hold` turns False (exit embedded)."""
        seg = trig.cumsum()
        ever_off = (~hold.fillna(False)).groupby(seg).cummax()
        return trig | ((seg > 0) & ~ever_off)

    if exit_kind == "mid":
        hold_l, hold_s = c4 >= mid, c4 <= mid
    elif exit_kind == "opp":
        hold_l, hold_s = c4 > bdn, c4 < bup
    else:  # time stop: 10 days = 60 x 4h bars
        age = c4.copy()
        age[:] = np.arange(len(c4))[:, None]
        age_l = age - age.where(la_trig).ffill()
        age_s = age - age.where(sa_trig).ffill()
        hold_l = (c4 >= mid) & (age_l <= 60).fillna(False)
        hold_s = (c4 <= mid) & (age_s <= 60).fillna(False)
    la = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    sa = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    for col in c4.columns:
        la[col] = persist(la_trig[col], hold_l[col])
        sa[col] = persist(sa_trig[col], hold_s[col])
    if long_only:
        sa = sa & False
    if short_only:
        la = la & False
    le = ~la
    se = ~sa
    return _states_to_signals(la, le, sa, se, idx, c4=c4)


def battery_b():
    print("## Round 14b — SQUEEZE validation (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    r0 = FS.sim(squeeze2(), False, None, SQ_OVER)
    report("PRODUCTION squeeze (mid exit, KC2, BB20)", r0["net"], r0["n"])
    for m in (1.5, 2.5):
        r = FS.sim(squeeze2(kc_mult=m), False, None, SQ_OVER)
        report(f"KC mult {m}", r["net"], r["n"])
    r = FS.sim(squeeze2(bb_len_d=30), False, None, SQ_OVER)
    report("BB len 30d", r["net"], r["n"])
    for ek in ("opp", "time"):
        r = FS.sim(squeeze2(exit_kind=ek), False, None, SQ_OVER)
        report(f"exit = {ek}", r["net"], r["n"])
    r = FS.sim(squeeze2(long_only=True), False, None, SQ_OVER)
    report("long-only", r["net"], r["n"])
    r = FS.sim(squeeze2(short_only=True), False, None, SQ_OVER)
    report("short-only", r["net"], r["n"])
    r = FS.sim(squeeze2(skip_flash=True), False, None, SQ_OVER)
    report("skip entries that are fade-flashes", r["net"], r["n"])


# ---------------------------------------------------------------- battery C
def battery_c():
    print("## Round 14c — new daily families (24m)")
    print("| family | variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    h4, l4, c4, v4 = FS._resample_ohlcv(FS.panel)

    # funding momentum: sustained funding extreme -> trade WITH the carry
    lf4 = FS.last_fund.reindex(c4.index, method="ffill")
    la, le = lf4 >= 0.0015, lf4 < 0
    sa, se = lf4 <= -0.0015, lf4 > 0
    r = FS.sim(_states_to_signals(la, le, sa, se, idx, c4=c4), False, None)
    report("funding_mom | dy/nogate", r["net"], r["n"])
    r = FS.sim(_states_to_signals(la, le, sa, se, idx, c4=c4), False, 80)
    report("funding_mom | dy/gated", r["net"], r["n"])

    # volume breakout: vol explosion WITH the price direction (fade is against)
    volx = (v4 / v4.rolling(36).mean()).where(v4.rolling(36).mean() > 0)
    la = (volx > 3) & (c4 > c4.shift(1))
    sa = (volx > 3) & (c4 < c4.shift(1))
    le, se = c4 < c4.shift(1), c4 > c4.shift(1)
    r = FS.sim(_states_to_signals(la, le, sa, se, idx, c4=c4), False, None)
    report("vol_breakout | dy/nogate", r["net"], r["n"])

    # cross-sectional momentum: 7d ranks, hold while in decile/quartile
    r7 = c4.pct_change(42)
    rk = r7.rank(axis=1, pct=True)
    la, le = rk >= 0.9, rk >= 0.25
    sa, se = rk <= 0.1, rk <= 0.75
    r = FS.sim(_states_to_signals(la, le, sa, se, idx, c4=c4), False, None)
    report("xs_momentum | dy/nogate", r["net"], r["n"])

    # BTC lead-lag: BTC trailing-24h sign broadcast to alts
    b = c4["BTCUSDT"].pct_change(6)
    la = pd.DataFrame(np.repeat((b > 0).values[:, None], c4.shape[1], 1),
                      index=c4.index, columns=c4.columns)
    sa = pd.DataFrame(np.repeat((b < 0).values[:, None], c4.shape[1], 1),
                      index=c4.index, columns=c4.columns)
    le, se = b < 0, b > 0
    le = pd.DataFrame(np.repeat((b < 0).values[:, None], c4.shape[1], 1),
                      index=c4.index, columns=c4.columns)
    se = pd.DataFrame(np.repeat((b > 0).values[:, None], c4.shape[1], 1),
                      index=c4.index, columns=c4.columns)
    sig = _states_to_signals(la, le, sa, se, idx, c4=c4)
    if "BTCUSDT" in sig:
        sig.pop("BTCUSDT")
    r = FS.sim(sig, False, None)
    report("btc_lag | dy/nogate", r["net"], r["n"])


def battery_d():
    """Round 15: FADE v4 refinements (24m)."""
    print("## Round 15d — FADE v4 refinements (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, _, rc = fade_sim2(scale_at=0.75, cap_h=42)
    report("PRODUCTION v4 (single tranche 75%)", net, len(rc))
    for plan, lab in (
            (((0.5, 0.5), (0.75, 0.25)), "3-tranche 50%@50%+25%@75%+25%@org"),
            (((0.5, 0.25), (0.75, 0.25)), "25%@50%+25%@75%+50%@org"),
            (((0.9, 0.5),), "single 90%"),
            (((0.6, 0.5),), "single 60%"),
    ):
        net, _, rc = fade_sim2(cap_h=42, tranches=list(plan))
        report(lab, net, len(rc))
    net, _, rc = fade_sim2(cap_h=42, adaptive="volx")
    report("adaptive level by vol bucket", net, len(rc))
    for fm in (0.0005, 0.001):
        net, _, rc = fade_sim2(scale_at=0.75, cap_h=42, fund_min=fm)
        report(f"funding >= {fm:.2%} at entry", net, len(rc))
    net, _, rc = fade_sim2(scale_at=0.75, cap_h=42, btc_quiet=True)
    report("BTC-quiet (idiosyncratic only)", net, len(rc))
    for wc in (1.0, 1.5):
        net, _, rc = fade_sim2(scale_at=0.75, cap_h=42, w_cap=wc)
        report(f"ivol w_cap {wc}x", net, len(rc))
    net, _, rc = fade_sim2(scale_at=0.75, cap_h=42, entry_mode="limit",
                           lim_depth=0.5, lim_h=6)
    report("limit 0.5xATR + v4 exit", net, len(rc))


def battery_e():
    """Round 15: SQUEEZE refinements (24m)."""
    print("## Round 15e — SQUEEZE refinements (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    r0 = FS.sim(squeeze2(), False, None, SQ_OVER)
    report("PRODUCTION squeeze v2", r0["net"], r0["n"])
    # 1h confirmation: first hourly close after 4h activation must agree
    sig = squeeze2()
    pc = closes.pct_change()
    conf = {}
    for s, (la, le, sa, se) in sig.items():
        r1 = pc[s].values
        la2 = la.values & (r1 > 0)
        sa2 = sa.values & (r1 < 0)
        # state persistence: keep active while exit-state says hold
        conf[s] = (pd.Series(la2, index=idx) | (pd.Series(le.values == False,  # noqa: E712
                                                         index=idx)
                                                & pd.Series(la2, index=idx).ffill()),
                   le, pd.Series(sa2, index=idx), se)
    r = FS.sim(conf, False, None, SQ_OVER)
    report("1h momentum confirmation", r["net"], r["n"])
    # short-only production candidate
    r = FS.sim(squeeze2(short_only=True), False, None, SQ_OVER)
    report("short-only", r["net"], r["n"])
    # tighter stop: FS uses 5%; simulate 2xATR(14d) stop via state cap is
    # not expressible -> skip; instead: no-overlay reference
    r = FS.sim(squeeze2(), False, None)
    report("bare (no overlays) reference", r["net"], r["n"])
    r = FS.sim(squeeze2(), False, None, SQ_OVER | {"longonly"})
    report("long-only", r["net"], r["n"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "a"
    if mode == "a":
        battery_a()
    elif mode == "b":
        battery_b()
    elif mode == "c":
        battery_c()
    elif mode == "d":
        battery_d()
    elif mode == "e":
        battery_e()
