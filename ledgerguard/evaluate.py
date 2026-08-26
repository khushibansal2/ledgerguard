"""Grading against the hidden truth key.

A match rate on its own is a vanity metric - an engine that books everything
scores 100% throughput and destroys the ledger. So this module reports the
three numbers that actually trade off against each other:

    precision  of what we booked, how much was right      (error rate)
    recall     of what was matchable, how much we caught  (throughput)
    autonomy   how much needed no human                   (cost)

plus a calibration table, because a confidence score that does not track
observed accuracy is decoration. If items booked at 0.90 are right 60% of the
time, the number is a lie and the policy gate built on it is meaningless.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .engine import Controller
from .generate import Batch


def _key(settlement: list[str], ledger: list[str]) -> tuple:
    return (tuple(sorted(settlement)), tuple(sorted(ledger)))


def evaluate(batch: Batch, ctrl: Controller) -> dict[str, Any]:
    truth = {_key(t["settlement"], t["ledger"]): t["case"] for t in batch.truth}
    truth_settlements = {s for t in batch.truth for s in t["settlement"]}
    unmatchable_ids = {i for o in batch.unmatchable for i in o["ids"]}

    tp: list[dict] = []      # booked and exactly correct
    fp: list[dict] = []      # booked and wrong
    partial: list[dict] = []  # right counterparty pairing, wrong document set

    # index truth by settlement id so we can distinguish "wrong link" from
    # "right link, incomplete document set" - they need different fixes.
    truth_by_settlement: dict[str, tuple[tuple, str]] = {}
    for t in batch.truth:
        for s in t["settlement"]:
            truth_by_settlement[s] = (_key(t["settlement"], t["ledger"]), t["case"])

    def economically_identical(got: list[str], expected: list[str]) -> bool:
        """True when two document sets are indistinguishable in the ledger.

        When a vendor raises two identical bills and settles them with two
        identical payments, FIFO application may pick the opposite pairing to
        the one the generator recorded. Every posting - vendor, amount,
        currency, period - is the same either way, so calling that an error
        would be measuring the label rather than the ledger.

        The test is strict and provable, not a tolerance: same multiset of
        (counterparty, amount, currency, period). Anything else is a real miss
        and is still scored as one. Matches passing this test are reported
        separately as `cohort_equivalent` so the headline precision is never
        quietly inflated by them.
        """
        if len(got) != len(expected) or set(got) == set(expected):
            return False

        def sig(i: str):
            r = ctrl.by_id[i]
            return (r.counterparty.lower(), r.amount, r.currency, r.txn_date[:7])

        return sorted(map(sig, got)) == sorted(map(sig, expected))

    equivalent: list[dict] = []

    for m in ctrl.matches:
        k = _key(m.settlement_ids, m.ledger_ids)
        sid = m.settlement_ids[0]
        rec = {"match_id": m.match_id, "layer": m.layer, "confidence": m.confidence,
               "settlement": m.settlement_ids, "ledger": m.ledger_ids,
               "case": truth.get(k) or (truth_by_settlement.get(sid, (None, None))[1]),
               "rationale": m.rationale}
        if k in truth:
            tp.append(rec)
        elif sid in truth_by_settlement:
            exp_key = truth_by_settlement[sid][0]
            rec["expected_ledger"] = list(exp_key[1])
            if economically_identical(m.ledger_ids, list(exp_key[1])):
                rec["note"] = "different document, identical posting (closed cohort)"
                equivalent.append(rec)
                tp.append(rec)
            elif set(m.ledger_ids) & set(exp_key[1]):
                partial.append(rec)
            else:
                fp.append(rec)
        else:
            rec["note"] = "settlement had no true ledger counterpart (should be an exception)"
            fp.append(rec)

    # Items the policy gate resolved but withheld for sign-off are graded too.
    # Scoring a deliberate hold as a miss would reward an engine that books
    # material items on thin evidence - exactly the behaviour the gate exists
    # to prevent. Autonomy and correctness are reported as separate numbers.
    def _held_ok(m) -> bool:
        if _key(m.settlement_ids, m.ledger_ids) in truth:
            return True
        exp = truth_by_settlement.get(m.settlement_ids[0])
        return bool(exp) and economically_identical(m.ledger_ids, list(exp[0][1]))

    held_correct = [m for m in ctrl.escalated if _held_ok(m)]
    held_wrong = [m for m in ctrl.escalated if not _held_ok(m)]

    booked_settlements = {s for m in ctrl.matches for s in m.settlement_ids}
    resolved_settlements = booked_settlements | {
        s for m in held_correct for s in m.settlement_ids}
    missed = [{"settlement": t["settlement"], "ledger": t["ledger"], "case": t["case"]}
              for t in batch.truth
              if not set(t["settlement"]) & booked_settlements]
    truly_missed = [{"settlement": t["settlement"], "ledger": t["ledger"], "case": t["case"]}
                    for t in batch.truth
                    if not set(t["settlement"]) & resolved_settlements]

    # --- exception quality: did we escalate the right things? ---------------
    flagged_ids = {i for e in ctrl.exceptions for i in e.record_ids}
    correctly_escalated = sorted(unmatchable_ids & flagged_ids)
    missed_escalation = sorted(unmatchable_ids - flagged_ids)

    n_booked = len(ctrl.matches)
    n_truth = len(batch.truth)
    precision = len(tp) / n_booked if n_booked else 0.0
    recall = len(tp) / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # --- calibration --------------------------------------------------------
    buckets = [(1.0, 1.01), (0.95, 1.0), (0.90, 0.95), (0.85, 0.90), (0.0, 0.85)]
    tp_ids = {r["match_id"] for r in tp}
    calib = []
    for lo, hi in buckets:
        rows = [m for m in ctrl.matches if lo <= m.confidence < hi]
        if not rows:
            continue
        correct = sum(1 for m in rows if m.match_id in tp_ids)
        calib.append({"bucket": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": len(rows),
                      "accuracy": correct / len(rows),
                      "mean_confidence": sum(m.confidence for m in rows) / len(rows)})

    # --- per-case difficulty breakdown --------------------------------------
    by_case: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for t in batch.truth:
        by_case[t["case"]]["total"] += 1
    for r in tp:
        if r["case"]:
            by_case[r["case"]]["correct"] += 1

    layer_mix = defaultdict(int)
    for m in ctrl.matches:
        layer_mix[m.layer] += 1

    return {
        "totals": {
            "records": len(batch.records),
            "settlement_events": len(ctrl.settlements),
            "ledger_documents": len(ctrl.ledger),
            "true_matchable": n_truth,
            "booked": n_booked,
            "held_for_review": len(ctrl.escalated),
            "exceptions": len(ctrl.exceptions),
        },
        "accuracy": {
            # of what was booked without a human, how much was right
            "precision": precision,
            # of what was matchable, how much was booked without a human
            "recall": recall,
            "f1": f1,
            # of what was matchable, how much the engine resolved correctly
            # whether or not policy let it book unattended
            "resolution_recall": ((len(tp) + len(held_correct)) / n_truth) if n_truth else 0.0,
            "true_positives": len(tp), "false_positives": len(fp),
            "partial_matches": len(partial),
            "cohort_equivalent": len(equivalent),
            "missed": len(missed), "unresolved": len(truly_missed),
            "held_correct": len(held_correct), "held_wrong": len(held_wrong),
            "straight_through_rate": n_booked / len(ctrl.settlements) if ctrl.settlements else 0.0,
        },
        "exception_quality": {
            "should_escalate": len(unmatchable_ids),
            "correctly_escalated": len(correctly_escalated),
            "missed_escalation": missed_escalation,
            "escalation_recall": (len(correctly_escalated) / len(unmatchable_ids)
                                  if unmatchable_ids else 1.0),
        },
        "calibration": calib,
        "by_case": {k: v for k, v in sorted(by_case.items())},
        "layer_mix": dict(layer_mix),
        "tool_calls": ctrl.stats["tool_calls"],
        "errors": {"false_positives": fp, "partial": partial, "missed": missed,
                   "unresolved": truly_missed,
                   "held_wrong": [m.to_dict() for m in held_wrong]},
    }
