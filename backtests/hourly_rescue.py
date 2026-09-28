"""Hourly rescue research — VERDICT (2026-09-26): hourly is NOT rescueable.
- pullback limits 1/2/3%: all worse than market entries (-6 to -14%/yr);
  the post-flip dip keeps dipping (buying knives)
- JEV-style walk-forward classifier (logistic on 10 causal features, 120d
  train window, refit daily): selects WORSE trades (310 kept, -38%/yr) —
  features carry no out-of-sample signal for hourly-flip outcomes
- intraday-reversion family: fees (18bps RT) > edge; dead (see
  family_scan_honest.py full intraday_rev)
- hybrid (daily entries + hourly exits): SR 1.46 -> 0.65; exits intraday
  also lose
Kept for reference; do not re-run expecting different results.

Hourly rescue research tools:
(1) pullback-limit entries  (2) a JEV-style walk-forward classifier.

JEV-style layer (emulating TypeSafe's Jev: state + typed question -> typed
judgment + probability): causal features at flip confirmation -> logistic
p(win), trained ONLY on prior trades (60d window, refit every 24 bars,
min 300 samples) -> trade only if p >= threshold. Fully walk-forward.

Pullback entries: at flip confirmation (gated there) place a limit at
close * (1 -/+ pb); fills within WINDOW bars if low/high crosses (fill at
min/max(open, limit)); cancelled if the 4h state invalidates first.

Run:  ./venv/bin/python backtests/hourly_rescue.py
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
vol_mat, med_vol_bar, vol_np = TR.vol_mat, TR.med_vol_bar, TR.vol_np
pc_np, fund_np = TR.pc_np, TR.fund_np
SLOTS, SLOT_FRAC = TR.SLOTS, TR.BASE_W
STOP, CAP_BARS = TR.STOP_FIX, TR.CAP_BARS
FEE, SLIP_BPS = TR.FEE, TR.SLIP_BPS
BTC = TR.BTC
btc_mom = TR.btc_mom
last_fund = TR.last_fund

open_np, low_np, high_np = {}, {}, {}
for s in TR.panel.pairs:
    f = TR.panel.frames[s]
    low_np[s] = f["Low"].values
    high_np[s] = f["High"].values
# frames have no Open column; fills assumed AT the limit when touched
# (conservative: gap-throughs would fill better in reality)


def feats(sym, i, direction):
    m = TR.mom_np[sym][i]
    a = TR.atr_np[sym][i]
    r1 = r1_np[sym][i]
    rk = TR.rank_np[sym][i]
    lf = float(last_fund[sym].iloc[i]) if sym in last_fund.columns else 0.0
    hr = idx[i].hour
    aligned = 1.0 if (direction == "LONG") == (m > 0) else 0.0
    thrust = min(abs(r1) / (2.0 * a), 1.0) if (np.isfinite(a) and a > 0) else 0.0
    fbr = float(closes[sym].iloc[i] / closes[sym].iloc[i - 1] - 1.0) if i > 0 else 0.0
    return np.array([
        aligned,
        min(abs(m) / 0.30, 1.0) if np.isfinite(m) else 0.0,
        thrust,
        1.0 - (float(rk) - 1.0) / 300.0 if np.isfinite(rk) else 0.0,
        np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
        float(np.clip(lf, -0.003, 0.003)),
        1.0 if btc_mom[i] > 0 else (-1.0 if np.isfinite(btc_mom[i]) else 0.0),
        np.clip(fbr * 10, -3, 3),
        min(abs(fbr) / a, 3.0) if (np.isfinite(a) and a > 0) else 0.0,
    ])


def fit_logit(X, y, iters=300, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xz = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xz.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xz @ w, -30, 30)))
        w -= lr * (Xz.T @ (p - y)) / len(y)
    return {"w": w, "mu": mu, "sd": sd}


def predict(model, x):
    z = (x - model["mu"]) / model["sd"]
    z = np.append(z, 1.0)
    return float(1.0 / (1.0 + np.exp(-np.clip(z @ model["w"], -30, 30))))


def walk_forward(stream, thr, train_days=120, min_n=150, refit=24):
    """include-mask aligned to stream. Trains only on labeled (filled) trades
    from the prior train_days window; refits every refit bars."""
    W = train_days * 24
    mask, model, last_fit = [], None, -10 ** 9
    for ti, t in enumerate(stream):
        if model is None or t["i0"] - last_fit >= refit:
            Xtr = [s["x"] for s in stream[:ti]
                   if s.get("y") is not None and t["i0"] - s["i0"] <= W]
            ytr = [s["y"] for s in stream[:ti]
                   if s.get("y") is not None and t["i0"] - s["i0"] <= W]
            if len(Xtr) >= min_n:
                model = fit_logit(np.array(Xtr), np.array(ytr, dtype=float))
                last_fit = t["i0"]
        if model is None or t.get("y") is None:
            mask.append(True)
        else:
            mask.append(predict(model, t["x"]) >= thr)
    return mask


def sim(gate=80, pb_pct=None, window=12, include=None, collect=None,
        tech=frozenset(("blowoff", "ivol"))):
    """Hourly-only sim with optional pullback-limit entries and an include
    mask (by stream index). collect: list to append the candidate stream."""
    open_tr, trades = [], []
    net_arr = np.zeros(len(idx))
    fee_cost = FEE + SLIP_BPS * 1e-4
    last_acc = 30 * bpd - 1
    pend_lim = []
    stream_n = 0

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
        v, mv = vol_np[symv][i], med_vol_bar[i]
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            return float(np.clip((mv / v) * SLOT_FRAC,
                                 0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
        return SLOT_FRAC

    for i in range(30 * bpd, len(idx)):
        accrue_to(i)
        still = []
        for tr in open_tr:
            d = tr["dir"]
            la_v, sa_v = sig_arr[tr["sym"]]
            flipped = (not la_v[i]) if d == "LONG" else (not sa_v[i])
            c_i = float(closes[tr["sym"]].iloc[i])
            adverse = ((c_i / tr["px0"] - 1.0 <= -STOP) if d == "LONG"
                       else (c_i / tr["px0"] - 1.0 >= STOP))
            expired = (i - tr["i0"]) >= CAP_BARS
            if flipped or adverse or expired:
                reason = "flip" if flipped else ("stop" if adverse else "time")
                tr.update(j0=i, exit_px=c_i, reason=reason)
                trades.append(tr)
            else:
                still.append(tr)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        pend_syms = {p[1] for p in pend_lim}
        # 1) limit fills (candidates were gated at confirmation)
        if pend_lim:
            fill_cands, keep = [], []
            for exp_i, symv, direction, lim, sidx in pend_lim:
                if i > exp_i or symv in held:
                    continue
                la_v, sa_v = sig_arr[symv]
                if not (la_v[i] if direction == "LONG" else sa_v[i]):
                    continue
                hit = (low_np[symv][i] <= lim if direction == "LONG"
                       else high_np[symv][i] >= lim)
                if not hit:
                    keep.append((exp_i, symv, direction, lim, sidx))
                    continue
                fill = lim
                fill_cands.append((symv, direction, float(fill), sidx))
            pend_lim = keep
            for symv, direction, fill, sidx in fill_cands:
                if include is not None and sidx not in include:
                    continue
                if len(open_tr) >= SLOTS:
                    continue
                w = weight(symv, i)
                c_i = float(closes[symv].iloc[i])
                sgn = 1.0 if direction == "LONG" else -1.0
                net_arr[i] += w * sgn * (c_i / fill - 1.0) - w * 2 * fee_cost
                open_tr.append({"sym": symv, "dir": direction, "i0": i,
                                "j0": None, "px0": fill, "exit_px": None,
                                "reason": "open", "w": w, "sidx": sidx})
                held.add(symv)
        # 2) fresh flip confirmations: gate, stream, then market or limit
        for symv, direction in FLIPS_AT_BAR.get(i, ()):
            if symv == BTC or symv in held or symv in pend_syms:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            if "blowoff" in tech:
                r1 = r1_np[symv][i]
                if np.isfinite(r1) and abs(r1) > TR.BLOWOFF_MAX:
                    continue
            sc = TR.score_at(symv, i, direction)
            if sc is None or (gate is not None and sc <= gate):
                continue
            sidx = stream_n
            stream_n += 1
            if collect is not None:
                collect.append({"sym": symv, "dir": direction, "i0": i,
                                "sidx": sidx, "x": feats(symv, i, direction),
                                "y": None})
            if pb_pct is None:
                if include is not None and sidx not in include:
                    continue
                if len(open_tr) >= SLOTS:
                    continue
                w = weight(symv, i)
                net_arr[i] -= w * 2 * fee_cost
                open_tr.append({"sym": symv, "dir": direction, "i0": i,
                                "j0": None, "px0": float(closes[symv].iloc[i]),
                                "exit_px": None, "reason": "open", "w": w,
                                "sidx": sidx})
                held.add(symv)
            else:
                lim = (float(closes[symv].iloc[i]) * (1.0 - pb_pct)
                       if direction == "LONG"
                       else float(closes[symv].iloc[i]) * (1.0 + pb_pct))
                pend_lim.append((i + window, symv, direction, lim, sidx))
    accrue_to(len(idx) - 1)
    for t in open_tr:
        t["j0"], t["exit_px"], t["reason"] = len(idx) - 1, float(closes[t["sym"]].iloc[-1]), "open"
        trades.append(t)

    net = pd.Series(net_arr, index=idx)
    expct = [(t["exit_px"] / t["px0"] - 1.0 if t["dir"] == "LONG"
              else t["px0"] / max(t["exit_px"], 1e-12) - 1.0) for t in trades]
    wins = sum(1 for e in expct if e > 0)
    return {"net": net, "n": len(trades), "trades": trades,
            "expct": float(np.mean(expct)) if expct else 0.0,
            "win": wins / len(trades) if trades else 0.0}


def label_stream(stream, run):
    y_of = {}
    for t in run["trades"]:
        sidx = t.get("sidx")
        if sidx is None:
            continue
        y_of[sidx] = 1.0 if (
            t["exit_px"] / t["px0"] - 1.0 if t["dir"] == "LONG"
            else t["px0"] / max(t["exit_px"], 1e-12) - 1.0) > 0 else 0.0
    for s in stream:
        s["y"] = y_of.get(s["sidx"])


from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def report(label, r):
    if r["n"] == 0:
        print(f"| {label} | 0 | - | - | - | - | - | - |", flush=True)
        return
    lo, hi, pn = bootstrap_sharpe_ci(r["net"])
    days = len(r["net"]) / 24
    print(f"| {label} | {r['n']} | {sharpe_of(r['net']):.2f} | "
          f"[{lo:.2f},{hi:.2f}] P0 {pn:.0%} | "
          f"{r['net'].sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(r['net']):.1%} | {r['win']:.0%} | "
          f"{r['expct']:+.2%} |", flush=True)


if __name__ == "__main__":
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD | win | expct |")
    print("|---|---|---|---|---|---|---|---|")
    stream_g = []
    r = sim(gate=80, collect=stream_g)
    label_stream(stream_g, r)
    report("base hr/gated", r)
    for pb in (0.01, 0.02, 0.03):
        report(f"pb{int(pb * 1000) / 10:.0f}% hr/gated", sim(gate=80, pb_pct=pb))
    for thr in (0.50, 0.55):
        mask = walk_forward(stream_g, thr)
        inc = {s["sidx"] for s, m in zip(stream_g, mask) if m}
        report(f"jev>={thr} hr/gated", sim(gate=80, include=inc))
    # combination: pullback + JEV (mask from the pb run's own outcomes)
    stream_pb = []
    r_pb = sim(gate=80, pb_pct=0.02, collect=stream_pb)
    label_stream(stream_pb, r_pb)
    mask = walk_forward(stream_pb, 0.55)
    inc = {s["sidx"] for s, m in zip(stream_pb, mask) if m}
    report("pb2%+jev>=.55 hr/gated", sim(gate=80, pb_pct=0.02, include=inc))
