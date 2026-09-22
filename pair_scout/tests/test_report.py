from datetime import datetime, timezone

from pair_scout.config import JevConfig
from pair_scout.jev.ranker import Assessment, jev_assessment, rule_based_assessment
from pair_scout.report import build_report, chunk_html

from .test_ranker import make_answers, make_candidate

NOW = datetime(2026, 9, 22, 1, 15, tzinfo=timezone.utc)


def _ranked():
    cfg = JevConfig()
    good = jev_assessment(make_candidate(), make_answers(), cfg)
    watch = jev_assessment(
        make_candidate(asset_long="XRPUSDT"), make_answers(conf=0.2), cfg
    )
    return [good, watch]


def test_short_report_single_chunk():
    chunks = build_report(
        now=NOW, universe_count=44, scanned_count=946, passed_count=7,
        scored_count=7, ranked=_ranked(), rejected_summary={"p > 0.05 (12)": 12},
        jev_used=True,
    )
    assert len(chunks) == 1
    assert all(len(c) <= 4096 for c in chunks)
    assert "PairScout" in chunks[0]
    assert "ENTER" in chunks[0]
    assert "WATCH" in chunks[0]
    assert "Sizing hint" in chunks[0]


def test_html_escaping():
    evil = '<b>&"inject">'
    cfg = JevConfig()
    c = make_candidate(asset_long=evil)
    a = jev_assessment(c, make_answers(), cfg)
    chunks = build_report(
        now=NOW, universe_count=1, scanned_count=1, passed_count=1,
        scored_count=1, ranked=[a], rejected_summary={}, jev_used=True,
    )
    joined = "\n".join(chunks)
    assert "<b>&\"inject\">" not in joined
    assert "&lt;b&gt;" in joined


def test_no_trade_block_when_no_enter():
    chunks = build_report(
        now=NOW, universe_count=10, scanned_count=45, passed_count=2,
        scored_count=2, ranked=[], rejected_summary={}, jev_used=True,
    )
    joined = "\n".join(chunks)
    assert "NO TRADE" in joined
    assert "Not financial advice" in joined


def test_degraded_mode_marker():
    chunks = build_report(
        now=NOW, universe_count=10, scanned_count=45, passed_count=2,
        scored_count=0, ranked=[rule_based_assessment(make_candidate())],
        rejected_summary={}, jev_used=False,
        jev_degraded_reason="TYPESAFE_API_KEY missing",
    )
    joined = "\n".join(chunks)
    assert "rule-based ranking only" in joined
    assert "TYPESAFE_API_KEY missing" in joined


def test_chunking_on_line_boundaries():
    lines = [f"line {i:04d} " + "x" * 60 for i in range(200)]
    text = "\n".join(lines)
    chunks = chunk_html(text, limit=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "\n".join(chunks) == text  # lossless


def test_determinism():
    kwargs = dict(
        now=NOW, universe_count=44, scanned_count=946, passed_count=7,
        scored_count=7, ranked=_ranked(), rejected_summary={"p > 0.05 (3)": 3},
        jev_used=True,
    )
    assert build_report(**kwargs) == build_report(**kwargs)
