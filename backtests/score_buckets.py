"""Validate the probe Score: bucket historical PSAR-flip trades by entry Score
and measure forward performance (net = main + global hedge 50/50, probe exits:
SAR flip / 5% stop / 7d cap). Tests whether Score >90 is really better.
Run from repo root:  ./venv/bin/python backtests/score_buckets.py
"""
import sys
import logging

sys.path.insert(0, "/root/crypto-trading-analysis")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

from pair_scout.config import load_config
from pair_scout.data.loader import load_panel
from pair_scout.research_ta_families import psar_signals_panel

BPD = 24
CAP_BARS = 7 * BPD
STOP = 0.05
UNIV = 200
BTC = "BTCUSDT"

cfg = load_config(None)
panel = load_panel(
    cex=cfg.data.cex, interval=cfg.data.interval, data_dir="saved_data_12m",
    top_n_volume=658, min_history_bars=cfg.data.min_history_bars,
    max_nan_fraction=cfg.data.max_nan_fraction,
    trailing_volume_days=cfg.data.trailing_volume_days,
)
idx = panel.index
closes = panel.closes
sig = psar_signals_panel(panel)
bpd = panel.bars_per_day

# precompute per-pair arrays
dollar = pd.DataFrame(
    {s: panel.frames[s]["Volume"] * panel.frames[s]["Close"] for s in panel.pairs})
rank_df = dollar.rolling(30 * bpd).mean().rank(axis=1, ascending=False)
mom7_df = closes / closes.shift(7 * bpd) - 1.0
ret1d_df = closes / closes.shift(bpd) - 1.0

atr_rows = {}
for s in panel.pairs:
    f = panel.frames[s]
    pc = f["Close"].shift(1)
    tr = pd.concat([f["High"] - f["Low"], (f["High"] - pc).abs(),
                    (f["Low"] - pc).abs()], axis=1).max(axis=1)
    atr_rows[s] = (tr.rolling(bpd).mean() / f["Close"]).values
atr_arr = atr_rows  # keep NaN: score_at rejects only non-finite (parity with probe)


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


# global hedge (weakest/strongest 7d in-universe) per bar
mom7_all = mom7_df
in_u_all = (rank_df <= UNIV) & rank_df.notna()
mom_masked = mom7_all.where(in_u_all)
valid = mom_masked.notna().any(axis=1)
hedge_weak_sym = pd.Series(index=mom_masked.index, dtype=object)
hedge_weak_sym[valid] = mom_masked[valid].idxmin(axis=1)
hedge_weak_px = mom_masked[valid].min(axis=1).reindex(mom_masked.index)
hedge_strong_sym = pd.Series(index=mom_masked.index, dtype=object)
hedge_strong_sym[valid] = mom_masked[valid].idxmax(axis=1)
hedge_strong_px = mom_masked[valid].max(axis=1).reindex(mom_masked.index)


def leg_ret(direction, entry_px, px):
    return px / entry_px - 1.0 if direction == "LONG" else entry_px / max(px, 1e-12) - 1.0


trades = []
for sym in sig:
    if sym == BTC or sym not in closes.columns:
        continue
    la, _, sa, _ = sig[sym]
    la_v = la.values.astype(bool)
    sa_v = sa.values.astype(bool)
    c = closes[sym].values
    n = len(c)
    i = 30 * bpd
    while i < n:
        # fresh long flip?
        if la_v[i] and not la_v[i - 1] and bool(in_u_all[sym].iloc[i]):
            direction, act, other = "LONG", la_v, sa_v
        elif sa_v[i] and not sa_v[i - 1] and bool(in_u_all[sym].iloc[i]):
            direction, act, other = "SHORT", sa_v, la_v
        else:
            i += 1
            continue
        sc = score_at(sym, i, direction)
        entry_px = float(c[i])
        sign = 1.0 if direction == "LONG" else -1.0
        j = i
        reason = "open"
        while j < n:
            if (direction == "LONG" and not la_v[j]) or (direction == "SHORT" and not sa_v[j]):
                reason = "flip"; break
            if sign * (c[j] / entry_px - 1.0) <= -STOP:
                reason = "stop"; break
            if j - i >= CAP_BARS:
                reason = "time"; break
            j += 1
        exit_px = float(c[min(j, n - 1)])
        main = leg_ret(direction, entry_px, exit_px)
        # hedge: global extreme coin at entry, 50/50 dollar-neutral
        if direction == "LONG":
            hs = hedge_weak_sym.iloc[i]
            h_entry = closes[hs].iloc[i] if hs in closes.columns else np.nan
            h_exit = closes[hs].iloc[min(j, n - 1)] if hs in closes.columns else np.nan
            hret = leg_ret("SHORT", float(h_entry), float(h_exit)) if hs and np.isfinite(h_entry) and np.isfinite(h_exit) else 0.0
        else:
            hs = hedge_strong_sym.iloc[i]
            h_entry = closes[hs].iloc[i] if hs in closes.columns else np.nan
            h_exit = closes[hs].iloc[min(j, n - 1)] if hs in closes.columns else np.nan
            hret = leg_ret("LONG", float(h_entry), float(h_exit)) if hs and np.isfinite(h_entry) and np.isfinite(h_exit) else 0.0
        trades.append({
            "date": idx[i], "score": sc, "net": 0.5 * main + 0.5 * hret,
            "hold_d": (min(j, n - 1) - i) / bpd, "reason": reason,
            "month": idx[i].strftime("%Y-%m"),
            "i0": i, "j0": min(j, n - 1), "sym": sym, "direction": direction,
            "hsym": hs if hs in closes.columns else None,
            "hside": ("SHORT" if direction == "LONG" else "LONG"),
        })
        i = max(j, i + 1)

df = pd.DataFrame(trades).dropna(subset=["score"])
df["bucket"] = pd.cut(df["score"], [0, 50, 60, 70, 80, 90, 101],
                      labels=["<50", "50-60", "60-70", "70-80", "80-90", ">90"],
                      right=False)
print(f"{len(df)} signals, {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
g = df.groupby("bucket", observed=True)["net"]
out = pd.DataFrame({
    "trades": g.count(),
    "win": df.groupby("bucket", observed=True)["net"].apply(lambda x: (x > 0).mean()),
    "avg_net": g.mean(),
    "sum_net": g.sum(),
    "hold": df.groupby("bucket", observed=True)["hold_d"].mean(),
})
print(out.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).to_string())
# monthly consistency of the two top buckets
for b in (">90", "80-90"):
    sub = df[df["bucket"] == b]
    m = sub.groupby("month")["net"].agg(["mean", "count"])
    pos = (m["mean"] > 0).mean()
    print(f"\n{b}: positive months {pos:.0%} ({len(m)} months), "
          f"monthly avg-of-avg {m['mean'].mean():+.2%}")
    print(m.round(3).T.to_string())

# ---- portfolio-style Sharpe per bucket (daily accrual, net = main+hedge 50/50)
buckets_order = ["<50", "50-60", "60-70", "70-80", "80-90", ">90"]
pc = closes.pct_change()
recs = []
# rebuild trade records with bars for accrual
acc = {b: np.zeros(len(idx)) for b in buckets_order}
for t in trades:
    sc = t.get("score")
    if sc is None or sc != sc:
        continue
    b = (">90" if sc >= 90 else "80-90" if sc >= 80 else "70-80" if sc >= 70
         else "60-70" if sc >= 60 else "50-60" if sc >= 50 else "<50")
    i0, j0 = int(t["i0"]), int(t["j0"])
    sgn = 1.0 if t["direction"] == "LONG" else -1.0
    acc[b][i0 + 1:j0 + 1] += 0.5 * sgn * pc[t["sym"]].iloc[i0 + 1:j0 + 1].fillna(0).values
    if t["hsym"] and t["hsym"] in pc.columns:
        hsgn = 1.0 if t["hside"] == "LONG" else -1.0
        acc[b][i0 + 1:j0 + 1] += 0.5 * hsgn * pc[t["hsym"]].iloc[i0 + 1:j0 + 1].fillna(0).values

print("\n-- bucket Sharpe (daily accrual, sqrt(365)) --")
print("| bucket | trades | avg/trade | per-trade SR | daily SR | ann ret | maxDD |")
print("|---|---|---|---|---|---|---|")
for b in buckets_order:
    sub = df[df["bucket"] == b]
    if not len(sub):
        continue
    daily = pd.Series(acc[b], index=idx).resample("1D").sum()
    sr_d = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    pts = sub["net"]
    sr_t = pts.mean() / pts.std() if pts.std() > 0 else 0
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    ann = daily.sum() / len(daily) * 365
    print(f"| {b} | {len(sub)} | {pts.mean():+.2%} | {sr_t:.2f} | {sr_d:.2f} | "
          f"{ann:+.0%} | {dd:.1%} |")
