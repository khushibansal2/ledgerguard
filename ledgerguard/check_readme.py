"""Assert that the README's numbers match the committed run output.

The dashboard is generated from `out/*.json` and so cannot drift. The README is
written by hand and therefore can - and did: an early revision claimed 81% of
events never reached a model when the measured figure was 48%, because the
number came from an estimate in the cost model rather than from a count.

Every headline figure is a substring check against the committed aggregate, so
editing one without re-running the pipeline fails the build. Run:

    python -m ledgerguard.check_readme
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checks(agg: dict, ab: dict) -> list[tuple[str, str]]:
    """(label, exact string that must appear in the README)."""
    baselines = [
        (f"baseline precision: {r['config']}", f"{r['precision'] * 100:.1f}%")
        for r in ab["rows"] if r["kind"] == "baseline"
    ]
    return [
        ("batches", str(agg["n_batches"])),
        ("records", f"{agg['records']:,}"),
        ("settlement events", f"{agg['events']:,}"),
        ("precision", f"{agg['precision_mean'] * 100:.2f}%"),
        ("straight-through", f"{agg['recall_mean'] * 100:.1f}%"),
        ("resolution recall", f"{agg['resolution_mean'] * 100:.1f}%"),
        ("unresolved", str(agg["unresolved"])),
        ("partial matches", str(agg["partial"])),
        ("tool calls per event", f"{agg['calls_per_event']:.2f}"),
        ("share settled before L3", f"{agg['pre_l3_share'] * 100:.0f}%"),
        *baselines,
    ]


def main() -> int:
    agg = json.loads((ROOT / "out" / "aggregate.json").read_text(encoding="utf-8"))
    ab = json.loads((ROOT / "out" / "ablation.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    stale = []
    print("checking README against committed run output\n")
    for label, expected in checks(agg, ab):
        ok = expected in readme
        print(f"  {'PASS' if ok else 'STALE'}  {label:<32}{expected:>10}")
        if not ok:
            stale.append((label, expected))

    if stale:
        print("\nREADME IS STALE - these figures are not in the file:",
              file=sys.stderr)
        for label, expected in stale:
            print(f"  {label}: expected {expected}", file=sys.stderr)
        print("\nRe-run the pipeline and update the README, or regenerate "
              "out/aggregate.json if the change was intentional.", file=sys.stderr)
        return 1

    print(f"\nall {len(checks(agg, ab))} published figures match the run output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
