"""Invariants that must hold on every batch, not just the demo seed.

These are deliberately property-style rather than golden-output tests. A golden
file would lock in today's numbers; what actually matters is that the engine
never loses a record, never books an unexplained difference, and never rewrites
its own audit trail - and those must hold for every seed, including ones nobody
has looked at.

Run: python -m pytest tests -q     (or: python tests/test_controller.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledgerguard.audit import AuditLog
from ledgerguard.engine import Controller, Policy
from ledgerguard.evaluate import evaluate
from ledgerguard.forecast import CashForecast
from ledgerguard.generate import generate
from ledgerguard.similarity import counterparty_score
from ledgerguard.tools import fee_explains, fx_convert, infer_tax_gross, subset_sum

SEEDS = list(range(20260827, 20260847))


def _run(seed: int):
    b = generate(seed=seed)
    c = Controller(b.records, policy=Policy(), audit=AuditLog()).run()
    return b, c, evaluate(b, c)


# --- the invariant that matters most ---------------------------------------
def test_no_record_is_silently_dropped():
    """Every record must end up either matched, held, or on the exception list.

    This is the whole promise of the exception ledger. An engine that quietly
    ignores what it cannot handle looks more accurate than one that reports it,
    which is exactly the incentive this test exists to remove.
    """
    for seed in SEEDS:
        b, c, _ = _run(seed)
        accounted = {i for m in c.matches for i in m.settlement_ids + m.ledger_ids}
        accounted |= {i for m in c.escalated for i in m.settlement_ids + m.ledger_ids}
        accounted |= {i for e in c.exceptions for i in e.record_ids}
        missing = {r.id for r in b.records} - accounted
        assert not missing, f"seed {seed} lost records: {sorted(missing)[:5]}"


def test_no_false_positives():
    for seed in SEEDS:
        _, _, ev = _run(seed)
        assert ev["accuracy"]["false_positives"] == 0, f"seed {seed}"
        assert ev["accuracy"]["held_wrong"] == 0, f"seed {seed}"


def test_no_record_double_counted():
    """One document may settle exactly once, or the ledger is overstated."""
    for seed in SEEDS:
        _, c, _ = _run(seed)
        seen: set[str] = set()
        for m in c.matches + c.escalated:
            for i in m.settlement_ids + m.ledger_ids:
                assert i not in seen, f"seed {seed}: {i} booked twice"
                seen.add(i)


def test_every_booked_match_is_arithmetically_closed():
    for seed in SEEDS:
        _, c, _ = _run(seed)
        for m in c.matches:
            assert abs(m.residual) <= 2, f"{m.match_id} residual {m.residual}"
            assert m.rationale.strip(), f"{m.match_id} booked with no rationale"


def test_audit_chain_is_tamper_evident():
    _, c, _ = _run(SEEDS[0])
    intact, broken = c.audit.verify()
    assert intact and broken is None
    # forge a historical entry: the chain must notice
    victim = len(c.audit.entries) // 2
    c.audit.entries[victim] = {**c.audit.entries[victim], "confidence": 0.99}
    intact, broken = c.audit.verify()
    assert not intact and broken == victim


def test_exceptions_are_actionable():
    for seed in SEEDS[:5]:
        _, c, _ = _run(seed)
        for e in c.exceptions:
            assert e.reason and e.suggested_action, e.category
            assert e.severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            assert e.record_ids


def test_unresolved_settlements_are_always_reported():
    """The residual failure mode must still surface to a human."""
    for seed in SEEDS:
        b, c, ev = _run(seed)
        flagged = {i for e in c.exceptions for i in e.record_ids}
        for u in ev["errors"]["unresolved"]:
            for sid in u["settlement"]:
                assert sid in flagged, f"seed {seed}: {sid} vanished"


def test_output_is_invariant_under_input_order():
    """The same book must reconcile the same way regardless of file order.

    Bank exports arrive sorted differently by rail, by date, by whoever hit the
    button. If a permutation of the input changes which documents get matched,
    the same month closes two different ways on two different days - so every
    tie-break in the engine is ordered by a stable key, and this test is what
    keeps it that way.
    """
    import random
    for seed in SEEDS[:8]:
        b = generate(seed=seed)
        baseline = None
        for perm in range(3):
            recs = list(b.records)
            random.Random(perm).shuffle(recs)
            c = Controller(recs, policy=Policy(), audit=AuditLog()).run()
            got = sorted((tuple(sorted(m.settlement_ids)), tuple(sorted(m.ledger_ids)))
                         for m in c.matches)
            if baseline is None:
                baseline = got
            else:
                assert got == baseline, f"seed {seed}: permutation {perm} changed the close"


# --- tools ------------------------------------------------------------------
def test_tools_refuse_rather_than_guess():
    assert fx_convert(100, "JPY", "USD", "2026-03-01")["ok"] is False
    assert infer_tax_gross(-100000, -100001)["ok"] is False
    assert fee_explains(-100000, -50000)["ok"] is False
    assert subset_sum([], -1000)["ok"] is False


def test_fx_and_tax_round_trip():
    r = fx_convert(-100000, "EUR", "USD", "2026-03-15")
    assert r["ok"] and r["rate"] == 1.0842 and r["converted"] == -108420
    t = infer_tax_gross(-100000, -118000)
    assert t["ok"] and t["rate_pct"] == 18.0


def test_similarity_separates_aliases_from_strangers():
    aliases = [("NW TRADERS EU", "Northwind Traders GmbH"),
               ("CIRRUSCLOUD.IO", "Cirrus Cloud Services"),
               ("SQ *BLUE BOTTLE 4471", "Blue Bottle Coffee Inc"),
               ("ACME LOGISTICS L.L.C.", "Acme Logistics LLC")]
    strangers = [("SABLEMEDIA", "Vantage Insurance Corp"),
                 ("ACME LOGISTIC", "Halberd Legal LLP"),
                 ("CIRRUSCLOUD.IO", "Kestrel Analytics Ltd")]
    worst_alias = min(counterparty_score(a, b) for a, b in aliases)
    best_stranger = max(counterparty_score(a, b) for a, b in strangers)
    assert worst_alias >= 0.85, worst_alias
    assert best_stranger <= 0.75, best_stranger
    assert worst_alias - best_stranger > 0.10, "alias/stranger margin too thin"


# --- forecast ---------------------------------------------------------------
def test_forecast_band_is_ordered_and_conservative():
    for seed in SEEDS[:5]:
        _, c, _ = _run(seed)
        p = CashForecast(c, "2026-04-05", 25_000_000).project()
        for w in p["weeks"]:
            assert w["low_balance"] <= w["expected_balance"] <= w["high_balance"]
            assert w["committed_outflow"] <= 0


def test_forecast_excludes_settled_receivables():
    """The double-count guard: a matched invoice must leave the AR forecast."""
    _, c, _ = _run(SEEDS[0])
    fc = CashForecast(c, "2026-04-05", 25_000_000)
    ap, ar = fc._open_items()
    assert not ({r.id for r in ap + ar} & c.consumed)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
    print(f"\n{'all green' if not fails else f'{fails} failing'}")
    raise SystemExit(1 if fails else 0)
