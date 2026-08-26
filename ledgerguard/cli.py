"""Command-line entry point: close the loop and report it honestly."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .agent import HeuristicPlanner, LLMPlanner
from .audit import AuditLog
from .engine import Controller, Policy
from .evaluate import evaluate
from .forecast import CashForecast
from .generate import generate
from .models import fmt

BAR = "=" * 78
SUB = "-" * 78


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _bar(x: float, width: int = 28) -> str:
    filled = int(round(x * width))
    return "#" * filled + "." * (width - filled)


def run_once(seed: int, as_of: str, opening: int, use_llm: bool,
             audit_path: Path | None) -> tuple:
    batch = generate(seed=seed)
    planner = LLMPlanner() if use_llm else HeuristicPlanner()
    ctrl = Controller(batch.records, policy=Policy(),
                      audit=AuditLog(audit_path), planner=planner)
    t0 = time.perf_counter()
    ctrl.run(as_of=as_of)
    elapsed = time.perf_counter() - t0
    return batch, ctrl, evaluate(batch, ctrl), elapsed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ledgerguard",
                                 description="Autonomous reconciliation & cash controller")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--as-of", default="2026-04-05")
    ap.add_argument("--opening", type=float, default=250_000.0,
                    help="opening bank balance in major units")
    ap.add_argument("--llm", action="store_true",
                    help="use the LLM planner for L3 hypothesis ordering "
                         "(set LEDGERGUARD_MODEL and the provider API key)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="run N consecutive seeds and aggregate (robustness mode)")
    ap.add_argument("--json", type=str, default="", help="write full results to a JSON file")
    ap.add_argument("--audit", type=str, default="out/audit.jsonl")
    ap.add_argument("--csv", type=str, default="",
                    help="write the exception ledger to CSV for the controller's "
                         "worklist (severity-ordered, one row per item)")
    args = ap.parse_args(argv)

    opening = int(round(args.opening * 100))

    if args.seeds > 1:
        return _robustness(args, opening)

    batch, ctrl, ev, elapsed = run_once(args.seed, args.as_of, opening,
                                        args.llm, Path(args.audit))
    fc = CashForecast(ctrl, args.as_of, opening)
    pos, proj, age = fc.position(), fc.project(), fc.ageing()

    t, a, q = ev["totals"], ev["accuracy"], ev["exception_quality"]
    print(f"\n{BAR}\n  LEDGERGUARD - autonomous close, seed {args.seed}, as of {args.as_of}\n{BAR}")
    print(f"  {t['records']} records | {t['settlement_events']} settlement events "
          f"| {t['ledger_documents']} ledger documents")
    print(f"  wall clock {elapsed * 1000:.0f} ms | {ev['tool_calls']} tool calls "
          f"| {ev['tool_calls'] / max(t['settlement_events'], 1):.2f} calls/event")

    print(f"\n  THROUGHPUT\n{SUB}")
    print(f"  auto-booked, no human      {_bar(a['straight_through_rate'])} "
          f"{_pct(a['straight_through_rate'])}  ({t['booked']}/{t['settlement_events']})")
    print(f"  held for controller sign-off                              "
          f"{t['held_for_review']}")
    print(f"  exceptions raised                                         "
          f"{t['exceptions']}")

    print(f"\n  ACCURACY (graded against a hidden truth key)\n{SUB}")
    print(f"  precision   {_pct(a['precision']):>7}   of what it booked, how much was correct")
    print(f"  recall      {_pct(a['recall']):>7}   of what was matchable, how much it booked alone")
    print(f"  resolution  {_pct(a['resolution_recall']):>7}   correctly resolved, incl. items held for review")
    print(f"  F1          {a['f1']:.3f}")
    print(f"  false positives {a['false_positives']}  |  partial {a['partial_matches']}"
          f"  |  cohort-equivalent {a['cohort_equivalent']}"
          f"  |  unresolved {a['unresolved']}  |  wrongly held {a['held_wrong']}")

    print(f"\n  DIFFICULTY BREAKDOWN\n{SUB}")
    for case, v in ev["by_case"].items():
        got, tot = v["correct"], v["total"]
        flag = "ok " if got == tot else "MISS"
        print(f"  {flag} {case:<32} {got}/{tot}")

    print(f"\n  CONFIDENCE CALIBRATION\n{SUB}")
    print(f"  {'bucket':<12}{'n':>4}{'mean conf':>12}{'observed acc':>14}")
    for c in ev["calibration"]:
        print(f"  {c['bucket']:<12}{c['n']:>4}{c['mean_confidence']:>12.3f}"
              f"{c['accuracy'] * 100:>13.1f}%")

    print(f"\n  LAYER MIX (cost profile)\n{SUB}")
    for layer, n in sorted(ev["layer_mix"].items()):
        print(f"  {layer:<20}{n:>4}   {_bar(n / max(t['booked'], 1))}")

    print(f"\n  EXCEPTION LEDGER - {t['exceptions']} items the controller would not book"
          f"\n{SUB}")
    by_sev = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
    for e in ctrl.exceptions:
        by_sev[e.severity].append(e)
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        items = by_sev[sev]
        if not items:
            continue
        print(f"\n  [{sev}] {len(items)} item(s)")
        for e in items[:4]:
            print(f"    - {e.category} {','.join(e.record_ids)} "
                  f"{fmt(abs(e.amount), e.currency)}")
            print(f"      {e.reason}")
            print(f"      -> {e.suggested_action}")
        if len(items) > 4:
            print(f"    ... and {len(items) - 4} more of the same class")
    print(f"\n  escalation recall {_pct(q['escalation_recall'])} "
          f"({q['correctly_escalated']}/{q['should_escalate']} truly-unmatchable "
          f"records correctly refused)")

    print(f"\n  CASH POSITION\n{SUB}")
    print(f"  opening balance          {fmt(pos['opening_balance']):>16}")
    print(f"  cleared movements        {fmt(pos['cleared_movements']):>16}")
    print(f"  closing bank balance     {fmt(pos['closing_bank_balance']):>16}")
    print(f"  open receivables         {fmt(pos['open_receivables']):>16}")
    print(f"  open payables            {fmt(pos['open_payables']):>16}")
    print(f"  net working capital      {fmt(pos['net_working_capital']):>16}")
    print(f"  book reconciled          {_pct(pos['reconciled_pct']):>16}"
          f"   ({pos['unreconciled_count']} settlements unexplained, "
          f"{fmt(pos['unreconciled_value'])})")

    print(f"\n  {proj['horizon_days']}-DAY FORECAST (expected / worst case)\n{SUB}")
    print(f"  {'week':<6}{'window':<26}{'inflow':>13}{'outflow':>13}"
          f"{'expected':>14}{'worst':>14}")
    for w in proj["weeks"]:
        print(f"  {w['week']:<6}{w['from']} - {w['to']:<10}"
              f"{fmt(w['expected_inflow']):>13}{fmt(w['committed_outflow']):>13}"
              f"{fmt(w['expected_balance']):>14}{fmt(w['low_balance']):>14}")
    print(f"\n  worst-case trough {fmt(proj['worst_case_trough'])}"
          + ("   *** LIQUIDITY WARNING ***" if proj["liquidity_warning"] else "   (funded)"))

    intact, broken = ctrl.audit.verify()
    print(f"\n  AUDIT TRAIL\n{SUB}")
    print(f"  {len(ctrl.audit.entries)} hash-chained entries -> {args.audit}")
    print(f"  chain integrity: {'VERIFIED' if intact else f'BROKEN at entry {broken}'}")
    print(f"  head commitment: {ctrl.audit.head[:32]}...")
    print(f"{BAR}\n")

    if args.csv:
        _write_exceptions_csv(ctrl, Path(args.csv))
        print(f"  exception worklist -> {args.csv}\n")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "evaluation": ev, "position": pos, "forecast": proj, "ageing": age,
            "matches": [m.to_dict() for m in ctrl.matches],
            "held": [m.to_dict() for m in ctrl.escalated],
            "exceptions": [e.to_dict() for e in ctrl.exceptions],
            "records": [r.to_dict() for r in batch.records],
            "audit_head": ctrl.audit.head, "elapsed_ms": elapsed * 1000,
            "seed": args.seed, "as_of": args.as_of,
        }, indent=2), encoding="utf-8")
        print(f"  full results -> {args.json}\n")
    return 0


def _write_exceptions_csv(ctrl, path: Path) -> None:
    """Export the exception ledger as a worklist.

    Severity-ordered and value-ordered within severity, because a controller
    works the list top-down and the largest exposure should be the first thing
    they touch. Every column is something a human needs to act - what it is,
    what it is worth, why it stopped, and what to do next.
    """
    rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    rows = sorted(ctrl.exceptions,
                  key=lambda e: (rank.get(e.severity, 9), -abs(e.amount)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["severity", "category", "record_ids", "amount", "currency",
                    "reason", "suggested_action"])
        for e in rows:
            w.writerow([e.severity, e.category, " ".join(e.record_ids),
                        f"{e.amount / 100:.2f}", e.currency,
                        e.reason, e.suggested_action])


def _robustness(args, opening: int) -> int:
    """Many seeds, aggregate stats. One good batch proves nothing."""
    print(f"\n{BAR}\n  ROBUSTNESS RUN - {args.seeds} independent synthetic batches\n{BAR}")
    print(f"  {'seed':>10}{'events':>8}{'booked':>8}{'prec':>8}{'recall':>8}"
          f"{'resol':>8}{'FP':>4}{'unres':>7}{'ms':>8}")
    agg = {"prec": [], "rec": [], "res": [], "fp": 0, "unres": 0, "held_wrong": 0,
           "events": 0, "booked": 0, "records": 0, "ms": [], "calls": 0,
           "esc": [], "cal": {}}
    for i in range(args.seeds):
        seed = args.seed + i
        batch, ctrl, ev, el = run_once(seed, args.as_of, opening, args.llm, None)
        a, t = ev["accuracy"], ev["totals"]
        agg["prec"].append(a["precision"]); agg["rec"].append(a["recall"])
        agg["res"].append(a["resolution_recall"]); agg["fp"] += a["false_positives"]
        agg["unres"] += a["unresolved"]; agg["held_wrong"] += a["held_wrong"]
        agg["events"] += t["settlement_events"]; agg["booked"] += t["booked"]
        agg["records"] += t["records"]; agg["ms"].append(el * 1000)
        agg["calls"] += ev["tool_calls"]
        agg["esc"].append(ev["exception_quality"]["escalation_recall"])
        for c in ev["calibration"]:
            b = agg["cal"].setdefault(c["bucket"], {"n": 0, "correct": 0})
            b["n"] += c["n"]; b["correct"] += round(c["accuracy"] * c["n"])
        print(f"  {seed:>10}{t['settlement_events']:>8}{t['booked']:>8}"
              f"{a['precision'] * 100:>7.1f}%{a['recall'] * 100:>7.1f}%"
              f"{a['resolution_recall'] * 100:>7.1f}%{a['false_positives']:>4}"
              f"{a['unresolved']:>7}{el * 1000:>8.0f}")

    n = args.seeds
    mean = lambda xs: sum(xs) / len(xs)
    print(f"\n{SUB}\n  AGGREGATE over {n} batches / {agg['records']} records "
          f"/ {agg['events']} settlement events\n{SUB}")
    print(f"  precision           mean {_pct(mean(agg['prec']))}   worst {_pct(min(agg['prec']))}")
    print(f"  recall (unattended) mean {_pct(mean(agg['rec']))}   worst {_pct(min(agg['rec']))}")
    print(f"  resolution recall   mean {_pct(mean(agg['res']))}   worst {_pct(min(agg['res']))}")
    print(f"  escalation recall   mean {_pct(mean(agg['esc']))}   worst {_pct(min(agg['esc']))}")
    print(f"  false positives     {agg['fp']} across all batches")
    print(f"  unresolved          {agg['unres']}   wrongly held {agg['held_wrong']}")
    print(f"  throughput          {agg['records'] / (sum(agg['ms']) / 1000):,.0f} records/sec "
          f"({mean(agg['ms']):.0f} ms per batch)")
    print(f"  tool calls          {agg['calls'] / agg['events']:.2f} per settlement event")
    print(f"\n  POOLED CALIBRATION\n{SUB}")
    print(f"  {'bucket':<12}{'n':>6}{'observed accuracy':>20}")
    for b in sorted(agg["cal"], reverse=True):
        v = agg["cal"][b]
        print(f"  {b:<12}{v['n']:>6}{v['correct'] / v['n'] * 100:>19.1f}%")
    print(f"{BAR}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
