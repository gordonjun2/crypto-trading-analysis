"""Crypto ladder implied-vol signal (Track C pivot): the strike ladders
are professionally priced (Deribit-anchored) — no value-betting edge.
Their VALUE is as a real-time gauge: implied forward + implied sigma
solved from two legs per ladder, archived hourly for the VRP /
SQUEEZE-conditioning study (gate batteries once archive matures).

  snapshot()  solve (F, sigma*sqrt(tau)) per sym per nearest expiry,
              append vol_signal.jsonl   (cron :40 daily-hour)
Run: ./venv/bin/python pair_scout/vol_signal.py snapshot
"""
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

BASE = Path("saved_data_live/prediction")
SNAP = BASE / "snapshots.jsonl"
OUT = BASE / "vol_signal.jsonl"


def latest_snapshot():
    with open(SNAP) as fh:
        last = None
        for line in fh:
            last = line
    return json.loads(last)


def ladder_points(legs, sym_key):
    pts = []
    for l in legs:
        q = l.get("q") or ""
        if sym_key not in q.lower():
            continue
        m = re.search(r"above\s+\$?([\d,]+)", q)
        if not m:
            continue
        k = float(m.group(1).replace(",", ""))
        if l.get("bid") is not None and l.get("ask") is not None:
            mid = (l["bid"] + l["ask"]) / 2
            if 0.02 < mid < 0.98:
                pts.append((k, mid))
    pts.sort()
    return pts


def solve_F_sigma(pts):
    """Lognormal: p = Phi(ln(F/K)/w - w/2). Solve (lnF, w) from the two
    legs bracketing p=0.5 (fallback: closest pair straddling 0.3)."""
    if len(pts) < 2:
        return None
    pair = None
    for a, b in zip(pts, pts[1:]):
        if a[1] >= 0.5 >= b[1]:
            pair = (a, b)
            break
    if pair is None:
        pair = (min(pts, key=lambda x: abs(x[1] - 0.35)),
                max(pts, key=lambda x: abs(x[1] - 0.35)))
    (k1, p1), (k2, p2) = pair

    def res(x):
        lnF, w = x
        return [norm.cdf((lnF - math_log(k1)) / w - w / 2) - p1,
                norm.cdf((lnF - math_log(k2)) / w - w / 2) - p2]
    try:
        sol, info, ier, _ = fsolve(
            res, [math_log((k1 + k2) / 2), 0.02], full_output=True)
        if ier != 1 or sol[1] <= 0:
            return None
        return {"F": float(np.exp(sol[0])),
                "sig_sqrt_tau": float(sol[1]),
                "k1": k1, "p1": p1, "k2": k2, "p2": p2}
    except Exception:
        return None


def math_log(x):
    return float(np.log(x))


def spot_from_klines(sym):
    p = BASE / "strike_klines_h1.json"
    try:
        kl = json.load(open(p))["klines"][sym]
        return float(kl[-1][4])
    except Exception:
        return None


def realized_sigma(sym):
    p = BASE / "strike_klines_h1.json"
    try:
        kl = json.load(open(p))["klines"][sym]
        c = [x[4] for x in kl]
        rets = np.diff(np.log(c))
        lam = 0.5 ** (1 / 24)
        v = rets[:24].var()
        for r in rets:
            v = lam * v + (1 - lam) * r * r
        return float(np.sqrt(v))
    except Exception:
        return None


def snapshot():
    r = latest_snapshot()
    strikes = r.get("poly_strikes", [])
    rec = {"ts": r.get("ts"), "built":
           datetime.now(timezone.utc).isoformat(), "ladders": []}
    for sym in ("btc", "eth"):
        pts = ladder_points(strikes, sym)
        sol = solve_F_sigma(pts)
        if sol is None:
            continue
        spot = spot_from_klines(sym.upper())
        rs = realized_sigma(sym.upper())
        l = {"sym": sym.upper(), **sol, "spot": spot, "sig_h_real": rs}
        if spot:
            l["F_over_S"] = sol["F"] / spot
        rec["ladders"].append(l)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    for l in rec["ladders"]:
        print(f"{l['sym']}: F {l['F']:.0f} (F/S {l.get('F_over_S', 0):.5f}) "
              f"sig*sqrt(tau) {l['sig_sqrt_tau']:.4f} | realized sig_h "
              f"{l.get('sig_h_real') or 0:.4f}")


if __name__ == "__main__":
    snapshot()
