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

# claim -> (threshold, comparison)  described exactly as the README states it
CLAIMS = {
    "precision >= 99.9%": 0.999,
    "straight_through >= 90%": 0.90,
    "resolution_recall >= 98.5%": 0.985,
    "escalation_recall == 100%": 1.0,
}


class Failure(Exception):
    pass


def _check(label: str, actual: float, threshold: float, results: list) -> None:
    ok = actual >= threshold - 1e-9
    results.append((label, actual, threshold, ok))
    if not ok:
        raise Failure(f"{label}: measured {actual:.4%}, floor {threshold:.4%}")


def main(n_seeds: int = 60) -> int:
    seeds = [START + i for i in range(n_seeds)]
    prec, rec, res, esc = [], [], [], []
    fps = drops = 0

    for seed in seeds:
        batch = generate(seed=seed)
        ctrl = Controller(batch.records, policy=Policy(), audit=AuditLog()).run()
        ev = evaluate(batch, ctrl)
        a = ev["accuracy"]
        prec.append(a["precision"])
        rec.append(a["recall"])
        res.append(a["resolution_recall"])
        esc.append(ev["exception_quality"]["escalation_recall"])
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

    _check("precision >= 99.9%", m(prec), 0.999, results)
    _check("straight-through >= 90%", m(rec), 0.90, results)
    _check("resolution recall >= 98.5%", m(res), 0.985, results)
    _check("escalation recall == 100%", m(esc), 1.0, results)

    if fps:
        raise Failure(f"false positives regressed: {fps} across {n_seeds} batches")
    results.append(("zero false positives", 0.0, 0.0, True))

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
