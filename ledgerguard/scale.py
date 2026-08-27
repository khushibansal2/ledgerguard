"""How does the close behave as the book grows?

A demo that reconciles 136 records tells you nothing about a company that
settles 50,000 a month. Two of the layers are quadratic by construction - L2
compares every open settlement against every open document, and candidate
retrieval scores every counterparty pair - so the interesting question is not
whether it is fast but *where the curve bends*, and whether accuracy survives
the density.

Density is the real test, not row count. Concatenating batches puts many more
same-vendor, same-amount documents in one pool, so every layer has more chances
to confuse itself. If precision holds as the book thickens, the discrimination
is real rather than an artifact of small inputs.

    python -m ledgerguard.scale
"""
from __future__ import annotations

import sys
import time

from .audit import AuditLog
from .engine import Controller, Policy
from .evaluate import evaluate
from .generate import Batch, generate, vendor_universe


def compose(n: int, start_seed: int = 20260827,
            distinct_vendors: bool = False) -> Batch:
    """Merge n independent batches into one book, keeping the truth key intact.

    Two modes, because they answer different questions.

    With `distinct_vendors=False` the batches share one pool of ten
    counterparties, so doubling the book doubles the number of same-vendor,
    same-amount documents competing for every payment. That is a density test,
    and a brutal one - closer to one enormous vendor account than to a company's
    whole ledger.

    With `distinct_vendors=True` each batch gets its own namespaced vendors, so
    the book grows in *size* while counterparty cardinality grows with it, which
    is what actually happens as a company scales. Running both separates a slow
    algorithm from a hard dataset - and they turn out to answer very differently.
    """
    merged = Batch()
    for i in range(n):
        # Each batch draws from its own slice of a large synthetic universe, so
        # counterparty cardinality grows with the book. Two earlier attempts at
        # this were broken fixtures rather than results: namespacing ids gave
        # every record the same initial and collapsed every block into one, and
        # prefixing vendor names with a shared word made *different* vendors
        # inside a batch more similar to each other.
        vendors = None
        if distinct_vendors:
            universe = vendor_universe(10 * n, seed=start_seed)
            vendors = universe[i * 10:(i + 1) * 10]
        b = generate(seed=start_seed + i, vendors=vendors)
        tag = f"B{i}-"
        for r in b.records:
            r.id = tag + r.id
            merged.records.append(r)
        for t in b.truth:
            merged.truth.append({
                "settlement": [tag + x for x in t["settlement"]],
                "ledger": [tag + x for x in t["ledger"]],
                "case": t["case"]})
        for o in b.unmatchable:
            merged.unmatchable.append({**o, "ids": [tag + x for x in o["ids"]]})
    return merged


def run(sizes: tuple[int, ...] = (1, 2, 4, 8, 16), distinct: bool = False) -> int:
    w = 82
    mode = ("distinct vendors per batch - size grows, density constant"
            if distinct else "shared vendor pool - density grows with size")
    print("\n" + "=" * w)
    print("  SCALE - does the close hold up as the book grows?")
    print(f"  {mode}")
    print("=" * w)
    print(f"  {'records':>9}{'events':>9}{'seconds':>10}{'rec/sec':>10}"
          f"{'tool calls':>12}{'precision':>11}{'$ mis-booked':>16}")
    print("  " + "-" * (w - 4))

    prev = None
    rows = []
    for n in sizes:
        batch = compose(n, distinct_vendors=distinct)
        ctrl = Controller(batch.records, policy=Policy(), audit=AuditLog())
        t0 = time.perf_counter()
        ctrl.run()
        elapsed = time.perf_counter() - t0
        ev = evaluate(batch, ctrl)

        recs = len(batch.records)
        events = len(ctrl.settlements)
        rows.append((recs, elapsed))
        print(f"  {recs:>9,}{events:>9,}{elapsed:>10.2f}{recs / elapsed:>10,.0f}"
              f"{ctrl.stats['tool_calls']:>12,}"
              f"{ev['accuracy']['precision'] * 100:>10.2f}%"
              f"{ev['value']['mis_booked'] / 100:>15,.0f}")
        prev = elapsed

    print("  " + "-" * (w - 4))

    # Empirical growth exponent: t = k * n^p, so p = log(t2/t1) / log(n2/n1).
    # Reported rather than asserted - the number is what it is, and a quadratic
    # layer should show up here as an exponent near 2.
    if len(rows) >= 2:
        import math
        (n1, t1), (n2, t2) = rows[0], rows[-1]
        p = math.log(t2 / t1) / math.log(n2 / n1)
        print(f"\n  empirical growth exponent p = {p:.2f}  (t proportional to n^p)")
        if p < 1.3:
            verdict = "effectively linear over this range"
        elif p < 1.7:
            verdict = "sub-quadratic - the indexed layers are carrying most of the load"
        else:
            verdict = ("quadratic - the all-pairs layers dominate, and a real "
                       "deployment would need blocking by counterparty")
        print(f"  {verdict}")
        est = t2 * (100_000 / n2) ** p
        print(f"  extrapolated to 100,000 records: {est:,.0f} s "
              f"({est / 60:,.1f} min) - extrapolation, not a measurement")
    print("=" * w + "\n")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    distinct = "--distinct" in sys.argv
    if args:
        raise SystemExit(run(tuple(int(x) for x in args), distinct))
    raise SystemExit(run(distinct=distinct))
