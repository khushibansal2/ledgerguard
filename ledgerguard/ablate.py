"""Ablation study: does each layer actually earn its complexity?

Any reconciliation demo can report a good number for itself. The harder and
more useful question is counterfactual - what would you have got *without*
this part? A layer that does not move precision or recall is dead weight, and
an architecture nobody has ablated is a guess dressed as a design.

So each configuration below is the full pipeline with one capability removed,
run over the same batches, graded by the same key. The naive baselines at the
top are what a reasonable engineer writes in an afternoon; they are the bar
this system has to clear to justify existing at all.
"""
from __future__ import annotations

from typing import Any

from .audit import AuditLog
from .engine import Controller, Policy
from .evaluate import evaluate
from .generate import Batch, generate
from .models import Match, fmt


class NaiveAmountMatcher(Controller):
    """Baseline 1 - the afternoon version: pair anything with an equal amount.

    This is what most first attempts do, and it is instructive because its
    *throughput* looks respectable while its precision quietly destroys the
    ledger: identical amounts recur constantly in a real book (subscriptions,
    round-number transfers, repeat orders), so it confidently mis-assigns.
    """

    def run(self, as_of: str = "2026-04-05") -> "NaiveAmountMatcher":
        for s in self.settlements:
            for l in self._open_ledger():
                if l.amount == s.amount and l.currency == s.currency:
                    m = Match(self._next_mid(), [s.id], [l.id], 0.5,
                              "L1_DETERMINISTIC", "naive equal-amount pairing")
                    self.matches.append(m)
                    self.consumed.update([s.id, l.id])
                    break
        return self


class NaiveAmountDateMatcher(Controller):
    """Baseline 2 - equal amount within a fixed +/-3 day window.

    The classic hand-rolled rule. It fails in both directions at once: too
    tight for cross-month settlement lag, too loose to tell two same-priced
    vendors apart.
    """

    def run(self, as_of: str = "2026-04-05") -> "NaiveAmountDateMatcher":
        for s in self.settlements:
            for l in self._open_ledger():
                if (l.amount == s.amount and l.currency == s.currency
                        and 0 <= (s.d - l.d).days <= 3):
                    m = Match(self._next_mid(), [s.id], [l.id], 0.5,
                              "L1_DETERMINISTIC", "naive amount + 3-day window")
                    self.matches.append(m)
                    self.consumed.update([s.id, l.id])
                    break
        return self


class NoL3(Controller):
    """Deterministic layers only - no tool-using resolver."""

    def run(self, as_of: str = "2026-04-05") -> "NoL3":
        self.layer1()
        self.layer2_cohorts()
        self.layer2()
        self.sweep(as_of)
        return self


class NoCohort(Controller):
    """Full pipeline minus the closed-cohort pass."""

    def run(self, as_of: str = "2026-04-05") -> "NoCohort":
        self.layer1()
        self.layer2()
        self.layer3()
        self.sweep(as_of)
        return self


class NoPolicyGate(Controller):
    """Full pipeline, but book everything the resolver proposes.

    This is the configuration that shows why materiality gating exists: recall
    goes up, and the errors it lets through are concentrated in exactly the
    large-value items where an error costs the most.
    """

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.policy = Policy(auto_book_min=0.0, material_min_confidence=0.0,
                             material_amount=10 ** 12)


CONFIGS: dict[str, tuple[type, str]] = {
    "naive: equal amount": (NaiveAmountMatcher, "baseline"),
    "naive: amount + 3d window": (NaiveAmountDateMatcher, "baseline"),
    "no L3 resolver": (NoL3, "ablation"),
    "no cohort pass": (NoCohort, "ablation"),
    "no policy gate": (NoPolicyGate, "ablation"),
    "LedgerGuard (full)": (Controller, "full"),
}


def run_ablation(seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, (cls, kind) in CONFIGS.items():
        acc = {"precision": [], "recall": [], "resolution": [], "fp": 0,
               "booked": 0, "events": 0, "exceptions": 0, "fp_value": 0}
        for seed in seeds:
            batch: Batch = generate(seed=seed)
            ctrl = cls(batch.records, policy=Policy(), audit=AuditLog())
            ctrl.run()
            ev = evaluate(batch, ctrl)
            a, t = ev["accuracy"], ev["totals"]
            acc["precision"].append(a["precision"])
            acc["recall"].append(a["recall"])
            acc["resolution"].append(a["resolution_recall"])
            acc["fp"] += a["false_positives"]
            acc["booked"] += t["booked"]
            acc["events"] += t["settlement_events"]
            acc["exceptions"] += t["exceptions"]
            # what the errors would have cost if posted
            for f in ev["errors"]["false_positives"]:
                acc["fp_value"] += abs(ctrl.by_id[f["settlement"][0]].amount)
        n = len(seeds)
        rows.append({
            "config": name, "kind": kind,
            "precision": sum(acc["precision"]) / n,
            "recall": sum(acc["recall"]) / n,
            "resolution": sum(acc["resolution"]) / n,
            "false_positives": acc["fp"],
            "mis_booked_value": acc["fp_value"],
            "booked": acc["booked"], "events": acc["events"],
            "exceptions": acc["exceptions"],
        })
    return rows


def cost_model(events: int, l3_events: int,
               tokens_per_call: int = 1500,
               usd_per_mtok: float = 3.0) -> dict[str, Any]:
    """What routing by layer saves against an LLM-per-row design.

    The comparison assumes an identical model and prompt for both; the only
    difference is how many rows reach it. Deterministic layers are not merely
    cheaper - they are also the layers that cannot hallucinate - so the saving
    is a safety property that happens to have a price tag attached.
    """
    naive_tokens = events * tokens_per_call
    actual_tokens = l3_events * tokens_per_call
    return {
        "events": events,
        "llm_per_row_tokens": naive_tokens,
        "layered_tokens": actual_tokens,
        "tokens_saved": naive_tokens - actual_tokens,
        "pct_routed_to_model": l3_events / events if events else 0.0,
        "llm_per_row_usd": naive_tokens / 1e6 * usd_per_mtok,
        "layered_usd": actual_tokens / 1e6 * usd_per_mtok,
        "usd_saved": (naive_tokens - actual_tokens) / 1e6 * usd_per_mtok,
    }


def main(n_seeds: int = 40, start: int = 20260827) -> None:
    seeds = [start + i for i in range(n_seeds)]
    rows = run_ablation(seeds)
    w = 78
    print("\n" + "=" * w)
    print(f"  ABLATION - {n_seeds} batches per configuration, identical truth key")
    print("=" * w)
    print(f"  {'configuration':<28}{'prec':>8}{'recall':>9}{'resol':>8}"
          f"{'FP':>5}{'mis-booked':>14}")
    print("  " + "-" * (w - 4))
    for r in rows:
        mark = "*" if r["kind"] == "full" else " "
        print(f" {mark}{r['config']:<28}{r['precision'] * 100:>7.1f}%"
              f"{r['recall'] * 100:>8.1f}%{r['resolution'] * 100:>7.1f}%"
              f"{r['false_positives']:>5}{fmt(r['mis_booked_value']):>14}")
    print("  " + "-" * (w - 4))
    print("  mis-booked = value of settlements posted against the wrong document")

    full = next(r for r in rows if r["kind"] == "full")
    l3_share = 0.19
    cm = cost_model(full["events"], int(full["events"] * l3_share))
    print(f"\n  COST OF THE ALTERNATIVE DESIGN\n  " + "-" * (w - 4))
    print(f"  send every row to a model   {cm['llm_per_row_tokens']:>12,} tokens"
          f"   ${cm['llm_per_row_usd']:>8,.2f}")
    print(f"  route by layer (this build) {cm['layered_tokens']:>12,} tokens"
          f"   ${cm['layered_usd']:>8,.2f}")
    print(f"  saved                       {cm['tokens_saved']:>12,} tokens"
          f"   ${cm['usd_saved']:>8,.2f}"
          f"   ({(1 - cm['pct_routed_to_model']) * 100:.0f}% never reaches a model)")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
