"""Daily-decision backtest: mirror the Telegram probe's real cadence.

One decision point per day at 10:00 SGT (02:00 UTC bar close):
- ENTER: fresh PSAR flips (state changed within last 24h), in-universe top-200,
  Score > gate, not already open -> enter at the 10:00 SGT bar close
- EXIT: checked ONLY at the daily bar -> signal flip / 5% stop / 7d cap
- net = main + global hedge 50/50 (as the journal tracks)

Compare vs the hourly-decision results.

Run from repo root:  ./venv/bin/python backtests/daily_gate_sim.py
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
GATE = 80
DECISION_HOUR_UTC = 2  # 10:00 SGT

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

dollar = pd.DataFrame(
    {s: panel.frames[s]["Volume"] * panel.frames[s]["Close"] for s in panel.pairs})
rank_df = dollar.rolling(30 * bpd).mean().rank(axis=1, ascending=False)
in_u_all = (rank_df <= UNIV) & rank_df.notna()
mom7_df = closes / closes.shift(7 * bpd) - 1.0
ret1d_df = closes / closes.shift(bpd) - 1.0
pc = closes.pct_change()

atr_raw = {}
for s in panel.pairs:
    f = panel.frames[s]
    p = f["Close"].shift(1)
    tr = pd.concat([f["High"] - f["Low"], (f["High"] - p).abs(),
                    (f["Low"] - p).abs()], axis=1).max(axis=1)
    atr_raw[s] = (tr.rolling(bpd).mean() / f["Close"]).values

mom_masked = mom7_df.where(in_u_all)
valid = mom_masked.notna().any(axis=1)
hedge_weak_sym = pd.Series(index=idx, dtype=object)
hedge_weak_sym[valid] = mom_masked[valid].idxmin(axis=1)
hedge_strong_sym = pd.Series(index=idx, dtype=object)
hedge_strong_sym[valid] = mom_masked[valid].idxmax(axis=1)


def score_at(sym, i, direction):
    px = closes[sym].iloc[i]
    if not np.isfinite(px) or px <= 0 or i < 7 * bpd:
        return None
    m = mom7_df[sym].iloc[i]
    if not np.isfinite(m):
        return None
    aligned = (direction == "LONG") == (m > 0)
    trend = 40.0 * min(abs(m) / 0.30, 1.0) if aligned else 0.0
    a = float(atr_raw[sym][i])
    if not np.isfinite(a) or a <= 0:
        return None
    r1 = ret1d_df[sym].iloc[i]
    if not np.isfinite(r1):
        return None
    thrust = 30.0 * min(abs(r1) / (2.0 * a), 1.0)
    rk = rank_df[sym].iloc[i]
    liq = 30.0 * max(0.0, 1.0 - (float(rk) - 1.0) / 300.0) if np.isfinite(rk) else 0.0
    return round(trend + thrust + liq)


def leg_ret(direction, entry_px, px):
    return px / entry_px - 1.0 if direction == "LONG" else entry_px / max(px, 1e-12) - 1.0


dec_idx = [i for i in range(30 * bpd, len(idx))
           if idx[i].hour == DECISION_HOUR_UTC]

trades = []
for sym in sig:
    if sym == BTC or sym not in closes.columns:
        continue
    la, _, sa, _ = sig[sym]
    la_v = la.values.astype(bool)
    sa_v = sa.values.astype(bool)
    c = closes[sym].values
    open_tr = None  # {"dir", "i0", "px0"}
    for k, i in enumerate(dec_idx):
        # 1) exit check on the open trade (daily only)
        if open_tr:
            d = open_tr["dir"]
            flipped = (not la_v[i]) if d == "LONG" else (not sa_v[i])
            adverse = ((c[i] / open_tr["px0"] - 1.0 <= -STOP) if d == "LONG"
                       else (c[i] / open_tr["px0"] - 1.0 >= STOP))
            expired = (i - open_tr["i0"]) >= CAP_BARS
            if flipped or adverse or expired:
                reason = "flip" if flipped else ("stop" if adverse else "time")
                open_tr["j0"], open_tr["exit_px"], open_tr["reason"] = i, float(c[i]), reason
                trades.append(open_tr)
                open_tr = None
        # 2) entry: fresh flip in last 24h (state changed within window)
        if open_tr is None and i >= 24:
            prev = dec_idx[k - 1] if k else i - 24
            lookback = i - 24
            if la_v[i] != la_v[lookback]:
                direction = "LONG" if la_v[i] else "SHORT"
            elif sa_v[i] != sa_v[lookback]:
                direction = "SHORT" if sa_v[i] else "LONG"
            else:
                continue
            if not bool(in_u_all[sym].iloc[i]):
                continue
            sc = score_at(sym, i, direction)
            if sc is None or sc <= GATE:
                continue
            open_tr = {"sym": sym, "dir": direction, "i0": i,
                       "j0": None, "px0": float(c[i]), "score": sc,
                       "exit_px": None, "reason": "open",
                       "hsym": (hedge_weak_sym.iloc[i] if direction == "LONG"
                                else hedge_strong_sym.iloc[i]),
                       "hside": "SHORT" if direction == "LONG" else "LONG",
                       "month": idx[i].strftime("%Y-%m")}
    if open_tr:
        open_tr["j0"], open_tr["exit_px"], open_tr["reason"] = len(idx) - 1, float(c[-1]), "open"
        trades.append(open_tr)

df = pd.DataFrame(trades)
df["net"] = [
    0.5 * leg_ret(t["dir"], t["px0"], t["exit_px"])
    + 0.5 * (leg_ret(t["hside"],
                     float(closes[t["hsym"]].iloc[t["i0"]]),
                     float(closes[t["hsym"]].iloc[t["j0"]]))
             if t["hsym"] in closes.columns else 0.0)
    for t in trades
]
df["hold_d"] = (df["j0"] - df["i0"]) / bpd
df["bucket"] = pd.cut(df["score"], [0, 80, 90, 101],
                      labels=["<=80", "80-90", ">90"], right=False)
print(f"{len(df)} gated trades (daily decisions, 10:00 SGT), "
      f"span {idx[min(df['i0'])].date()} -> {idx[max(df['j0'])].date()}")
g = df.groupby("bucket", observed=True)["net"]
out = pd.DataFrame({
    "trades": g.count(),
    "win": df.groupby("bucket", observed=True)["net"].apply(lambda x: (x > 0).mean()),
    "avg_net": g.mean(),
    "hold": df.groupby("bucket", observed=True)["hold_d"].mean(),
})
print(out.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).to_string())

# portfolio daily SR per bucket + gated book overall
acc = {}
for t in trades:
    b = t["score"] and (">90" if t["score"] > 90 else "80-90" if t["score"] > 80 else "<=80")
    if b not in acc:
        acc[b] = np.zeros(len(idx))
    sgn = 1.0 if t["dir"] == "LONG" else -1.0
    i0, j0 = t["i0"], t["j0"]
    acc[b][i0 + 1:j0 + 1] += 0.5 * sgn * pc[t["sym"]].iloc[i0 + 1:j0 + 1].fillna(0).values
    if t["hsym"] in pc.columns:
        hsgn = 1.0 if t["hside"] == "LONG" else -1.0
        acc[b][i0 + 1:j0 + 1] += 0.5 * hsgn * pc[t["hsym"]].iloc[i0 + 1:j0 + 1].fillna(0).values

print("\n-- daily-decision bucket Sharpe (vs hourly numbers from before) --")
print("| bucket | trades | avg/trade | per-trade SR | daily SR |")
print("|---|---|---|---|---|")
for b in (">90", "80-90", "<=80"):
    if b not in acc:
        continue
    daily = pd.Series(acc[b], index=idx).resample("1D").sum()
    sub = df[(df["bucket"] == b) if b != "<=80" else (df["score"] <= 80)]
    if b == "<=80":
        continue  # journal never tracks <=80; skip noise
    sr_d = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    pts = sub["net"]
    sr_t = pts.mean() / pts.std() if len(pts) and pts.std() > 0 else 0
    print(f"| {b} | {len(sub)} | {pts.mean():+.2%} | {sr_t:.2f} | {sr_d:.2f} |")

daily_all = pd.Series(acc[">90"], index=idx).resample("1D").sum() + \
    pd.Series(acc["80-90"], index=idx).resample("1D").sum()
sr = daily_all.mean() / daily_all.std() * np.sqrt(365) if daily_all.std() > 0 else 0
print(f"\nGATED BOOK (Score>80, daily decisions): daily SR {sr:.2f}, "
      f"ann ret {daily_all.sum() / 365 * 365:+.0%}, "
      f"maxDD {(daily_all.cumsum() - daily_all.cumsum().cummax()).min():.1%}")
