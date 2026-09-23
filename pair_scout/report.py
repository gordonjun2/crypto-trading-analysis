"""Telegram-safe report builder (HTML, escaped, 4096-chunked). Plan §6/§9."""

from __future__ import annotations

from datetime import datetime
from html import escape

from pair_scout.jev.ranker import Assessment

TELEGRAM_LIMIT = 4096


def _fmt(v: float, nd: int = 2, na: str = "n/a") -> str:
    if v != v:  # NaN
        return na
    return f"{v:.{nd}f}"


def build_report(
    now: datetime,
    universe_count: int,
    scanned_count: int,
    passed_count: int,
    scored_count: int,
    ranked: list[Assessment],
    rejected_summary: dict[str, int],
    jev_used: bool,
    watch_count: int = 0,
    jev_degraded_reason: str | None = None,
    regime_prob: float | None = None,
    entry_z: float = 1.5,
    stop_z: float = 3.5,
) -> list[str]:
    """Render the report and return it as <=4096-char HTML chunks."""
    lines: list[str] = []
    lines.append(
        f"📊 <b>PairScout</b> — {now:%d %b %Y %H:%M} UTC"
    )
    lines.append(
        f"Universe: {universe_count} pairs · scanned: {scanned_count} · "
        f"passed filters: {passed_count} · watch-tier: {watch_count} · "
        f"JEV scored: {scored_count}"
    )
    if not jev_used:
        note = jev_degraded_reason or "JEV disabled"
        lines.append(f"⚠️ JEV unavailable — rule-based ranking only ({escape(note)})")
    if regime_prob is not None:
        mood = "favorable" if regime_prob >= 0.6 else ("mixed" if regime_prob >= 0.45 else "hostile")
        lines.append(
            f"🧭 JEV regime read: {_fmt(regime_prob)} ({mood}) — context only, not a gate"
        )
    lines.append("")

    enters = [a for a in ranked if a.action == "ENTER"]
    for i, a in enumerate(ranked, 1):
        c = a.candidate
        flag = "🟢" if a.action == "ENTER" else "🟡"
        tag = "DIV" if c.strategy == "divergence" else "MR"
        lines.append(
            f"{flag} <b>{a.action}</b> [{tag}]  LONG {escape(c.asset_long)} / "
            f"SHORT {escape(c.asset_short)}"
        )
        score_src = "JEV" if a.jev_used else "Rule"
        if c.strategy == "divergence":
            stats = (
                f"mom {c.momentum_long:+.0%}/{c.momentum_short:+.0%} "
                f"(spread {c.momentum_spread:+.0%}) · βb {c.combo_beta:+.2f} · "
                f"corr {_fmt(c.return_correlation)} · vol ratio {_fmt(c.vol_ratio, 1)}"
            )
        else:
            stats = (
                f"p={_fmt(c.pvalue, 3)} · HL={_fmt(c.half_life_days, 1)}d · "
                f"z={_fmt(c.spread_zscore, 1)} · vol ratio {_fmt(c.vol_ratio, 1)}"
            )
        lines.append(
            f"   {score_src} {_fmt(a.composite)} (conf {_fmt(a.confidence)}) · {stats}"
        )
        if a.action == "ENTER":
            long_pct = round((c.weight_long if c.weight_long == c.weight_long else c.long_notional_share) * 100)
            short_pct = 100 - long_pct
            basis = "beta-balanced" if c.strategy == "divergence" else "vol-balanced"
            lines.append(
                f"   Sizing hint: {long_pct}% notional long / {short_pct}% short "
                f"({basis})"
            )
            lines.append(
                f"   Reason: {escape(a.reason)}."
            )
            if c.strategy == "divergence":
                lines.append(
                    f"   Invalidate: spread momentum &le;0 exit; re-check ranking weekly."
                )
            else:
                lines.append(
                    f"   Invalidate: |z|&gt;{_fmt(stop_z, 1)} stop; re-check p-value weekly."
                )
        else:
            lines.append(f"   Reason: {escape(a.reason)}.")
        lines.append("")

    if not enters:
        lines.append(
            "<b>🚫 NO TRADE</b> — no candidate cleared all filters/gates. "
            "Capital stays flat; nothing meets the evidence bar today."
        )
        lines.append("")
    if rejected_summary:
        top = sorted(rejected_summary.items(), key=lambda kv: -kv[1])[:5]
        joined = ", ".join(f"{escape(r)} ({n})" for r, n in top)
        lines.append(f"Most common rejections: {joined}")
        lines.append("")
    lines.append(
        "⚠️ Not financial advice. Analysis only — no orders are placed. "
        "Funding rates of the short perp leg are not modeled."
    )
    return chunk_html("\n".join(lines))


def chunk_html(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split on line boundaries so no chunk exceeds the Telegram hard limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            if current:
                chunks.append(current)
            while len(line) > limit:  # pathological single-line overflow
                chunks.append(line[:limit])
                line = line[limit:]
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
