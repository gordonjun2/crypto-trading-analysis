"""Round 25: ORDERFLOW (taker-buy fraction) — genuinely new data ground.

Data: saved_data_24m_flow (7-col pkls with 'Taker Buy USDT' = aggressive
buy quote volume), same grid as saved_data_24m (17,520 hourly bars).

Battery A — fade event conditioning on flow:
  tf = taker_buy_quote / total_quote of the event bar (and pre-pump mean).
  Does the pump bar's aggressive-buy share predict the fade outcome?
Battery B — new daily families built on flow climaxes/divergences.
Battery C — production fade trades bucketed by event-bar tf (edge gradient).

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate25.py [a|b|c]
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import technique_research as TR
from pair_scout.research_sizing import (bootstrap_sharpe_ci, max_dd_of,
                                        sharpe_of)
from iterate10 import (idx, closes, bpd, in_u_all, pc_np, fund_np, SLOTS,
                       SLOT_FRAC, BTC, fee_rt, atr1_np, vol_raw, vol_ma, report)
from iterate14 import low_np, high_np

FLOW_DIR = Path("saved_data_24m_flow/binance/1h")


def load_flow():
    """{sym: taker_frac array aligned to idx} for pairs with flow data."""
    out = {}
    for sym in TR.panel.pairs:
        files = list(FLOW_DIR.glob(f"{sym}_*.pkl"))
        if not files:
            continue
        with open(files[0], "rb") as fh:
            df = pickle.load(fh)["dataframe"]
        s = df.set_index("Open Time")
        vol = s["Volume in USDT"].reindex(idx)
        tb = s["Taker Buy USDT"].reindex(idx)
        with np.errstate(invalid="ignore", divide="ignore"):
            tf = (tb / vol).where(vol > 0)
        out[sym] = tf.values.astype(float)
    return out


# ---------------------------------------------------------------- battery A
def fade_sim7(k=5.0, cap_h=42, tf=None, tf_min=None, tf_max=None,
              tf_pre=None, tf_pre_max=None, gate=None):
    """FADE v6 + taker-fraction filters at the event bar.
    tf_min/tf_max bound the EVENT bar's aggressive-buy share;
    tf_pre/tf_pre_max bound the mean share of the 3 bars BEFORE it."""
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
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        recs.append({"sym": t["sym"], "dir": t["dir"], "i0": t["i0"], "j0": i,
                     "ret": sgn * (c_i / t["px0"] - 1.0)})

    V5_TR = [(0.5, 0.5), (0.75, 0.25)]
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
            if t["w"] > 1e-12:
                for ti, (rf, sf) in enumerate(V5_TR):
                    if ti in t["done"]:
                        continue
                    lvl = px0 - rf * span if t["dir"] == "SHORT" \
                        else px0 + rf * span
                    if (lo_i <= lvl) if t["dir"] == "SHORT" \
                            else (hi_i >= lvl):
                        realize(t, i, sf, lvl)
                        t["done"].add(ti)
            o_hit = (lo_i <= o) if t["dir"] == "SHORT" else (hi_i >= o)
            if expired:
                close_trade(t, i, "time")
            elif o_hit:
                close_trade(t, i, "origin", px=o)
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
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
            if tf_min is not None and not (np.isfinite(tfe) and tfe >= tf_min):
                continue
            if tf_max is not None and not (np.isfinite(tfe) and tfe <= tf_max):
                continue
            if gate is not None:
                sc = TR.score_at(symv, i, "SHORT")
                if sc is None or sc <= gate:
                    continue
            if tf_pre is not None or tf_pre_max is not None:
                pre = flow[symv][max(i - 3, 0):i]
                pre = pre[np.isfinite(pre)]
                if len(pre) < 2:
                    continue
                pm = float(pre.mean())
                if tf_pre is not None and pm < tf_pre:
                    continue
                if tf_pre_max is not None and pm > tf_pre_max:
                    continue
            cands.append((symv, a, tfe))
        if cands:
            room = slots - len(open_tr)
            for symv, a, tfe in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                w = weight(symv, i)
                net_arr[i] -= w * 2 * fee_rt
                open_tr.append({"sym": symv, "dir": "SHORT", "i0": i,
                                "px0": float(closes[symv].iloc[i]),
                                "origin": origin, "atr0": a, "w": w,
                                "w0": w, "done": set()})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, recs


def battery_a(tf):
    print("## Round 25a — fade x taker-fraction conditioning (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, _, rc = fade_sim7(tf=tf)
    report("PRODUCTION v6 (no flow filter)", net, len(rc))
    for lo, hi, lab in ((0.5, None, "tf >= 0.50 (buy-driven pump)"),
                        (0.6, None, "tf >= 0.60 (buy climax)"),
                        (None, 0.5, "tf <= 0.50 (thin pump)"),
                        (None, 0.4, "tf <= 0.40 (no aggression)")):
        net, _, rc = fade_sim7(tf=tf, tf_min=lo, tf_max=hi)
        report(lab, net, len(rc))
    for lo, hi, lab in ((0.55, None, "pre-pump tf >= 0.55"),
                        (None, 0.45, "pre-pump tf <= 0.45")):
        net, _, rc = fade_sim7(tf=tf, tf_pre=lo, tf_pre_max=hi)
        report(lab, net, len(rc))


def battery_c(tf):
    """Trade-level gradient: production fade trades by event-bar tf decile."""
    def rep(label, net, n=None):
        lo, hi, pn = bootstrap_sharpe_ci(net)
        days = len(net) / 24
        print(f"| {label} [{n}] | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
              f"{max_dd_of(net):.1%} |", flush=True)

    net, trades, recs = fade_sim7(tf=tf)
    tfs, rets = [], []
    for t in trades:
        tfe = tf[t["sym"]][t["i0"]]
        if np.isfinite(tfe):
            tfs.append(tfe)
            rets.append(1.0 - t["exit_px"] / t["px0"])
    tfs = np.array(tfs)
    rets = np.array(rets)
    print("## Round 25c — fade trade outcome by event-bar taker fraction")
    print("| tf bucket | trades | avg trade ret | win rate |")
    print("|---|---|---|---|")
    qs = np.quantile(tfs, np.linspace(0, 1, 6))
    for j in range(5):
        m = (tfs >= qs[j]) & (tfs <= qs[j + 1] if j == 4 else tfs < qs[j + 1])
        if m.sum():
            rr = rets[m]
            print(f"| [{qs[j]:.2f},{qs[j + 1]:.2f}) | {int(m.sum())} | "
                  f"{rr.mean():+.2%} | {(rr > 0).mean():.0%} |")


def load_flow_df():
    """(taker_buy_df, vol_df) hourly DataFrames aligned to idx."""
    tbs, vols = {}, {}
    for sym in TR.panel.pairs:
        files = list(FLOW_DIR.glob(f"{sym}_*.pkl"))
        if not files:
            continue
        with open(files[0], "rb") as fh:
            df = pickle.load(fh)["dataframe"]
        s = df.set_index("Open Time")
        tbs[sym] = s["Taker Buy USDT"].reindex(idx)
        vols[sym] = s["Volume in USDT"].reindex(idx)
    return pd.DataFrame(tbs), pd.DataFrame(vols)


# ---------------------------------------------------------------- battery B
def battery_b():
    """New daily families from flow: climax fades + CVD divergence."""
    import family_scan_honest as FS
    from pair_scout.research_ta_families import _states_to_signals

    tb1, v1 = load_flow_df()
    print(f"flow df: {tb1.shape[1]} pairs")
    tb4 = tb1.resample("4h").sum()
    v4 = v1.resample("4h").sum()
    h4, l4, c4, _ = FS._resample_ohlcv(FS.panel)
    tf4 = tb4 / v4.where(v4 > 0)
    r4 = c4.pct_change()

    print("## Round 25b — flow families (24m)")
    print("| family | variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")

    def persist(trig, hold):
        seg = trig.cumsum()
        ever_off = (~hold.fillna(False)).groupby(seg).cummax()
        return trig | ((seg > 0) & ~ever_off)

    # buy-climax fade: aggressive-buy-driven up bar -> SHORT until share cools
    la_trig = (tf4 <= 0.4) & (r4 < 0)
    sa_trig = (tf4 >= 0.6) & (r4 > 0)
    hold_l, hold_s = tf4 <= 0.5, tf4 >= 0.5
    la = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    sa = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    for col in c4.columns:
        la[col] = persist(la_trig[col], hold_l[col])
        sa[col] = persist(sa_trig[col], hold_s[col])
    r = FS.sim(_states_to_signals(la, ~la, sa, ~sa, idx, c4=c4), False, None)
    report("climax_fade | dy/nogate", r["net"], r["n"])
    r = FS.sim(_states_to_signals(la, ~la, sa, ~sa, idx, c4=c4), False, 80)
    report("climax_fade | dy/gated", r["net"], r["n"])

    # CVD divergence: 7d price up on NET aggressive selling -> SHORT (mirror)
    delta = 2 * tb4 - v4
    d7 = delta.rolling(42).sum()
    p7 = c4.pct_change(42)
    sa_trig = (p7 > 0.2) & (d7 < 0)
    la_trig = (p7 < -0.2) & (d7 > 0)
    hold_s = (p7 > 0) & (d7 < 0)
    hold_l = (p7 < 0) & (d7 > 0)
    la = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    sa = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    for col in c4.columns:
        la[col] = persist(la_trig[col], hold_l[col])
        sa[col] = persist(sa_trig[col], hold_s[col])
    r = FS.sim(_states_to_signals(la, ~la, sa, ~sa, idx, c4=c4), False, None)
    report("cvd_divergence | dy/nogate", r["net"], r["n"])

    # flow momentum WITH: sustained aggressive buying -> LONG
    la_trig = d7 > d7.rolling(252).quantile(0.95)
    sa_trig = d7 < d7.rolling(252).quantile(0.05)
    le = la_trig.shift(1).fillna(False)
    se = sa_trig.shift(1).fillna(False)
    r = FS.sim(_states_to_signals(la_trig, ~la_trig, sa_trig, ~sa_trig,
                                  idx, c4=c4), False, None)
    report("cvd_mom | dy/nogate", r["net"], r["n"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "a"
    if mode == "b":
        battery_b()
        sys.exit(0)
    tf = load_flow()
    print(f"flow loaded for {len(tf)}/{len(TR.panel.pairs)} panel pairs")
    all_tf = np.concatenate([v[np.isfinite(v)] for v in tf.values()])
    print(f"taker_frac: mean {all_tf.mean():.3f} median "
          f"{np.median(all_tf):.3f} p5 {np.quantile(all_tf, 0.05):.3f} "
          f"p95 {np.quantile(all_tf, 0.95):.3f}")
    if mode == "a":
        battery_a(tf)
    elif mode == "c":
        battery_c(tf)
