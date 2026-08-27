"""The resolver's tool belt.

Every tool is deterministic and returns *evidence*, not a verdict. The agent
layer decides; the tools only compute. That split is deliberate - it means a
wrong match can always be traced to either a bad tool result (reproducible) or
a bad decision (reviewable), never to an unfalsifiable blend of the two.

Each tool is also declared as a JSON-schema spec so the same belt can be handed
to an LLM planner verbatim (see agent.py, LLM_TOOL_SPECS).
"""
from __future__ import annotations

from datetime import date
from itertools import combinations
from typing import Any, Callable

from .generate import FX
from .models import Record

TAX_RATES = {"IN_GST": 18.0, "UK_VAT": 20.0, "DE_VAT": 19.0, "US_NONE": 0.0,
             "IN_GST_5": 5.0, "IN_GST_12": 12.0, "IN_GST_28": 28.0}

# Fee schedules the controller is allowed to assume without human sign-off.
FEE_MODELS = {
    "STRIPE_STANDARD": lambda gross: int(round(abs(gross) * 0.029)) + 30,
    "STRIPE_INTL": lambda gross: int(round(abs(gross) * 0.039)) + 30,
    "WIRE_OUT_DOMESTIC": lambda _g: 1500,
    "WIRE_OUT_INTL": lambda _g: 2250,
    "WIRE_OUT_INTL_LOW": lambda _g: 3000,
}


def fx_convert(amount: int, from_ccy: str, to_ccy: str, on_date: str) -> dict[str, Any]:
    """Convert minor units at the rate effective for the settlement month."""
    if from_ccy == to_ccy:
        return {"ok": True, "converted": amount, "rate": 1.0, "source": "identity"}
    table = FX.get((from_ccy, to_ccy))
    if not table:
        return {"ok": False, "error": f"no rate curve for {from_ccy}->{to_ccy}"}
    month = on_date[:7]
    rate = table.get(month) or table[sorted(table)[-1]]
    return {"ok": True, "converted": int(round(amount * rate)), "rate": rate,
            "source": f"fx_curve[{from_ccy}/{to_ccy}][{month}]"}


def tax_decompose(net: int, jurisdiction: str) -> dict[str, Any]:
    """Given a tax-exclusive net, return the gross a payer would actually remit."""
    rate = TAX_RATES.get(jurisdiction)
    if rate is None:
        return {"ok": False, "error": f"unknown jurisdiction {jurisdiction}"}
    tax = int(round(abs(net) * rate / 100))
    sign = -1 if net < 0 else 1
    return {"ok": True, "rate_pct": rate, "tax": tax,
            "gross": net + sign * tax, "jurisdiction": jurisdiction}


def infer_tax_gross(net: int, target: int) -> dict[str, Any]:
    """Reverse-solve: does any known statutory rate explain net -> target?"""
    for juris, rate in TAX_RATES.items():
        if rate == 0:
            continue
        tax = int(round(abs(net) * rate / 100))
        sign = -1 if net < 0 else 1
        if abs((net + sign * tax) - target) <= 2:      # 2c rounding latitude
            return {"ok": True, "jurisdiction": juris, "rate_pct": rate, "tax": tax}
    return {"ok": False, "error": "no statutory rate explains the variance"}


def fee_explains(gross: int, settled: int) -> dict[str, Any]:
    """Does a known fee schedule explain the gap between gross and settled?"""
    gap = abs(abs(gross) - abs(settled))
    for name, fn in FEE_MODELS.items():
        if abs(fn(gross) - gap) <= 2:
            return {"ok": True, "fee_model": name, "fee": gap}
    return {"ok": False, "error": f"gap of {gap} minor units matches no fee schedule",
            "gap": gap}


def subset_sum(candidates: list[Record], target: int, max_terms: int = 4,
               tol: int = 2, anchor_date: str | None = None) -> dict[str, Any]:
    """Find a subset of ledger docs whose signed amounts hit the settlement.

    Bounded at `max_terms` on purpose. An unbounded search will always find
    *some* combination on a large book, and a combination found by brute force
    over 30 invoices is numerology, not evidence.

    On a real book several subsets often hit the same total - two short-payments
    to one vendor in the same week can be recombined freely, and every
    recombination balances. Returning whichever the loop reached first makes the
    result depend on the order the file arrived in, so ties are broken on
    evidence instead: prefer the documents dated closest to the settlement,
    because a remittance pays the invoices in front of it and a credit note is
    applied to the bill it belongs to. Identifiers break any remaining tie, so
    the answer is reproducible for a given book.
    """
    pool = sorted(candidates[:14], key=lambda r: (r.txn_date, r.id))
    anchor = date.fromisoformat(anchor_date) if anchor_date else None

    for k in range(2, max_terms + 1):
        best: tuple[int, int, tuple[str, ...], tuple[Record, ...]] | None = None
        for combo in combinations(pool, k):
            diff = abs(sum(c.amount for c in combo) - target)
            if diff > tol:
                continue
            spread = (sum(abs((c.d - anchor).days) for c in combo)
                      if anchor else 0)
            key = (diff, spread, tuple(c.id for c in combo))
            if best is None or key < best[:3]:
                best = (*key, combo)
        if best:
            combo = best[3]
            return {"ok": True, "ids": [c.id for c in combo],
                    "sum": sum(c.amount for c in combo), "terms": k,
                    "residual": best[0], "date_cost": best[1],
                    "searched": f"C({len(pool)},{k})"}
    return {"ok": False, "error": f"no subset of <= {max_terms} docs sums to target",
            "pool_size": len(pool)}


def subset_sum_including(candidates: list[Record], target: int, must_include: str,
                         max_terms: int = 4, tol: int = 2) -> dict[str, Any]:
    """Subset search constrained to contain one specific record.

    Used for instalments, where the question is not "do any payments add up to
    this bill" - on a busy vendor account something usually does - but "does
    *this* payment belong to a set that clears it". Anchoring the search to the
    settlement in hand is what stops the engine assembling a plausible-looking
    group that happens to balance while the payment it was asked about sits
    unexplained.
    """
    anchor = next((c for c in candidates if c.id == must_include), None)
    if anchor is None:
        return {"ok": False, "error": "anchor not in candidate pool"}
    others = sorted((c for c in candidates if c.id != must_include),
                    key=lambda r: (r.txn_date, r.id))[:12]

    for k in range(1, max_terms):
        best = None
        for combo in combinations(others, k):
            group = (anchor,) + combo
            diff = abs(sum(c.amount for c in group) - target)
            if diff > tol:
                continue
            key = (diff, tuple(c.id for c in group))
            if best is None or key < best[:2]:
                best = (*key, group)
        if best:
            group = best[2]
            return {"ok": True, "ids": [c.id for c in group],
                    "sum": sum(c.amount for c in group), "terms": len(group),
                    "residual": best[0], "searched": f"C({len(others)},{k})"}
    return {"ok": False,
            "error": f"no group of <= {max_terms} settlements including "
                     f"{must_include} clears the target"}


def duplicate_scan(records: list[Record]) -> list[dict[str, Any]]:
    """Same counterparty + reference + amount raised more than once."""
    seen: dict[tuple, list[str]] = {}
    for r in records:
        if r.side != "LEDGER" or not r.reference:
            continue
        key = (r.counterparty.lower(), r.reference.upper(), r.amount)
        seen.setdefault(key, []).append(r.id)
    return [{"counterparty": k[0], "reference": k[1], "amount": k[2], "ids": v}
            for k, v in seen.items() if len(v) > 1]


# --- schema declarations, so an LLM planner can drive the same belt ---------
LLM_TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "fx_convert",
     "description": "Convert an amount between currencies at the rate effective on a date.",
     "input_schema": {"type": "object", "properties": {
         "amount": {"type": "integer", "description": "minor units"},
         "from_ccy": {"type": "string"}, "to_ccy": {"type": "string"},
         "on_date": {"type": "string", "description": "ISO date"}},
         "required": ["amount", "from_ccy", "to_ccy", "on_date"]}},
    {"name": "infer_tax_gross",
     "description": "Test whether any statutory tax rate explains net -> settled.",
     "input_schema": {"type": "object", "properties": {
         "net": {"type": "integer"}, "target": {"type": "integer"}},
         "required": ["net", "target"]}},
    {"name": "fee_explains",
     "description": "Test whether a known processor/wire fee schedule explains a gap.",
     "input_schema": {"type": "object", "properties": {
         "gross": {"type": "integer"}, "settled": {"type": "integer"}},
         "required": ["gross", "settled"]}},
    {"name": "subset_sum",
     "description": "Find <=4 open ledger documents summing to a settlement amount.",
     "input_schema": {"type": "object", "properties": {
         "candidate_ids": {"type": "array", "items": {"type": "string"}},
         "target": {"type": "integer"}}, "required": ["candidate_ids", "target"]}},
]

REGISTRY: dict[str, Callable[..., Any]] = {
    "fx_convert": fx_convert, "tax_decompose": tax_decompose,
    "infer_tax_gross": infer_tax_gross, "fee_explains": fee_explains,
    "subset_sum": subset_sum, "duplicate_scan": duplicate_scan,
}
