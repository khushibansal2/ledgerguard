"""Turn the README's accuracy claims into a build gate.

A number in a README is a claim about the past; a number checked in CI is a
property of the code. Every headline figure this project publishes is asserted
here, so a change that quietly regresses precision, starts dropping records, or
inflates the naive baseline fails the build instead of shipping.

Thresholds sit slightly below the measured figures. The gap is deliberate: it
absorbs legitimate variation from a different seed range without absorbing a
real regression. Run:

    python -m ledgerguard.verify_claims [n_seeds]
"""
from __future__ import annotations

import sys

from .ablate import run_ablation
from .audit import AuditLog
from .engine import Controller, Policy
from .evaluate import evaluate
from .generate import generate

START = 20260827


class Failure(Exception):
    pass


def _check(label: str, actual: float, threshold: float, results: list) -> None:
    ok = actual >= threshold - 1e-9
    results.append((label, actual, threshold, ok))
    if not ok:
        raise Failure(f"{label}: measured {actual:.4%}, floor {threshold:.4%}")


def main(n_seeds: int = 60) -> int:
    seeds = [START + i for i in range(n_seeds)]
    prec, rec, res, esc, dwp = [], [], [], [], []
    fps = drops = 0
    value_settled = value_mis = 0

    for seed in seeds:
        batch = generate(seed=seed)
        ctrl = Controller(batch.records, policy=Policy(), audit=AuditLog()).run()
        ev = evaluate(batch, ctrl)
        a = ev["accuracy"]
        prec.append(a["precision"])
        rec.append(a["recall"])
        res.append(a["resolution_recall"])
        esc.append(ev["exception_quality"]["escalation_recall"])
        dwp.append(ev["value"]["dollar_weighted_precision"])
        value_settled += ev["value"]["total_settled"]
        value_mis += ev["value"]["mis_booked"]
        fps += a["false_positives"]

        # the honesty invariant: nothing may vanish
        accounted = {i for m in ctrl.matches for i in m.settlement_ids + m.ledger_ids}
        accounted |= {i for m in ctrl.escalated for i in m.settlement_ids + m.ledger_ids}
        accounted |= {i for e in ctrl.exceptions for i in e.record_ids}
        drops += len({r.id for r in batch.records} - accounted)

        intact, _ = ctrl.audit.verify()
        if not intact:
            raise Failure(f"seed {seed}: audit chain broken")

    m = lambda xs: sum(xs) / len(xs)
    results: list = []
    print(f"verifying published claims over {n_seeds} batches\n")

    _check("precision >= 99.5%", m(prec), 0.995, results)
    _check("dollar-weighted precision >= 99.5%", m(dwp), 0.995, results)
    # Deliberately below the measured figure and deliberately not the headline.
    # Automation is the metric that should *fall* when authority tightens: the
    # exposure fix that gates multi-line matches on the whole event, not the
    # first line, moved this from 92.5% to 89.5% and improved precision. A
    # ceiling here would create pressure to book material items on thin
    # evidence, which is the opposite of the point.
    _check("straight-through >= 88%", m(rec), 0.88, results)
    _check("resolution recall >= 98.5%", m(res), 0.985, results)
    _check("escalation recall == 100%", m(esc), 1.0, results)

    # Exposure, not frequency. A handful of wrong rows is survivable; a wrong
    # row carrying real money is not, so the binding constraint is the share of
    # settled value that was booked against the wrong document.
    mis_share = value_mis / value_settled if value_settled else 0.0
    results.append(("mis-booked value <= 0.10% of settled", 1 - mis_share, 0.999,
                    mis_share <= 0.001))
    if mis_share > 0.001:
        raise Failure(f"mis-booked value regressed: {mis_share:.4%} of settled "
                      f"value across {n_seeds} batches")

    if drops:
        raise Failure(f"{drops} records were silently dropped")
    results.append(("no records dropped", 0.0, 0.0, True))

    # The architecture claim: the full pipeline must beat every naive baseline
    # on precision. If a baseline ever matches it, the layering is not earning
    # its complexity and the README should stop saying that it does.
    ab = run_ablation(seeds[:20])
    full = next(r for r in ab if r["kind"] == "full")
    for row in ab:
        if row["kind"] != "baseline":
            continue
        if row["precision"] >= full["precision"]:
            raise Failure(
                f"baseline '{row['config']}' matched the full pipeline "
                f"({row['precision']:.1%} vs {full['precision']:.1%}) - "
                f"the layered design no longer justifies itself")
        results.append((f"beats baseline: {row['config']}", full["precision"],
                        row["precision"], True))

    for label, actual, floor, ok in results:
        mark = "PASS" if ok else "FAIL"
        detail = f"{actual:.2%}" if actual else ""
        print(f"  {mark}  {label:<42}{detail:>9}")

    print(f"\nall published claims verified over {n_seeds} batches")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    try:
        raise SystemExit(main(n))
    except Failure as exc:
        print(f"\nCLAIM VERIFICATION FAILED\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
