"""Layer 3: the tool-using resolver.

This is the only place a model is permitted to influence a booking, and even
here it is fenced. The resolver works by *falsifiable hypothesis*: it proposes
a named economic reason the settlement differs from the document (FX, fee, tax,
split, short-pay), calls a deterministic tool to test it, and books only if the
arithmetic closes to within two cents.

The LLM's job is candidate selection and hypothesis ordering - judgement about
which vendor and which story is plausible. The LLM never does arithmetic and
never has the final say: `Policy.gate` and the residual check run afterwards
regardless of what it proposed. With no API key present the built-in heuristic
planner runs the identical hypothesis set, so the pipeline is fully reproducible
offline and the model is an accelerator, not a dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import Exception_, Match, Record, fmt
from .similarity import counterparty_score
from . import tools

CANDIDATE_FLOOR = 0.80          # entity similarity needed to even consider a doc


@dataclass
class Outcome:
    match: Optional[Match] = None
    exception: Optional[Exception_] = None
    tool_calls: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


class Resolver:
    def __init__(self, controller, planner=None) -> None:
        self.c = controller
        self.planner = planner or HeuristicPlanner()

    # -- candidate retrieval ----------------------------------------------
    def candidates(self, s: Record) -> list[Record]:
        """Entity-first retrieval. Amount is useless as a filter here (that is
        precisely what failed at L1/L2), so identity and direction carry it."""
        out = []
        for l in self.c._open_ledger():
            if (l.amount > 0) != (s.amount > 0) and l.doc_type != "CREDIT_NOTE":
                continue                              # never net a payable off a receipt
            cp = counterparty_score(s.counterparty or s.description, l.counterparty)
            if cp >= CANDIDATE_FLOOR:
                out.append((cp, l))
        # Sort by score, then id: ties must not resolve differently just
        # because the source file arrived in a different order.
        out.sort(key=lambda x: (-x[0], x[1].id))
        return [l for _cp, l in out[:14]]

    # -- main entry --------------------------------------------------------
    def resolve(self, s: Record) -> Outcome:
        oc = Outcome()
        cands = self.candidates(s)
        self.c.audit.emit("L3_RETRIEVAL", settlement=s.id,
                          amount=s.amount, currency=s.currency,
                          candidates=[c.id for c in cands])
        if not cands:
            oc.exception = Exception_(
                [s.id], "UNIDENTIFIED_OUTFLOW" if s.amount < 0 else "UNIDENTIFIED_RECEIPT",
                "HIGH",
                f"{fmt(abs(s.amount), s.currency)} settled on {s.txn_date} as "
                f"'{s.description}', but no open ledger document resembles this "
                f"counterparty above the {CANDIDATE_FLOOR:.2f} identity threshold.",
                "Investigate against card/vendor master; possible unrecorded expense "
                "or unauthorised debit.",
                s.amount, s.currency, {"description": s.description})
            return oc

        plan = self.planner.plan(s, cands)
        self.c.audit.emit("L3_PLAN", settlement=s.id, hypotheses=plan,
                          planner=type(self.planner).__name__)

        attempted: list[str] = []
        for hypothesis in plan:
            fn = getattr(self, f"_try_{hypothesis}", None)
            if fn is None:
                continue
            attempted.append(hypothesis)
            m, calls, ev = fn(s, cands)
            oc.tool_calls += calls
            self.c.audit.emit("L3_HYPOTHESIS", settlement=s.id,
                              hypothesis=hypothesis, accepted=bool(m), evidence=ev)
            if m:
                oc.match = m
                return oc

        oc.exception = self._diagnose(s, cands, attempted)
        return oc

    # -- hypotheses --------------------------------------------------------
    def _try_fx(self, s: Record, cands: list[Record]):
        calls = 0
        for l in cands:
            if l.currency == s.currency:
                continue
            r = tools.fx_convert(l.amount, l.currency, s.currency, s.txn_date)
            calls += 1
            if not r["ok"]:
                continue
            gap = abs(abs(s.amount) - abs(r["converted"]))
            if gap <= 2:
                return self._mk(s, [l], 0.94, "L3_AGENTIC", ["fx_convert"], 0,
                                f"{fmt(abs(l.amount), l.currency)} converted at "
                                f"{r['rate']} ({r['source']}) = "
                                f"{fmt(abs(r['converted']), s.currency)}, matching the "
                                f"settlement exactly.",
                                {"fx": r}), calls, {"gap": gap}
            fee = tools.fee_explains(r["converted"], s.amount)
            calls += 1
            if fee["ok"]:
                return self._mk(s, [l], 0.92, "L3_AGENTIC", ["fx_convert", "fee_explains"], 0,
                                f"{fmt(abs(l.amount), l.currency)} converted at "
                                f"{r['rate']} ({r['source']}) = "
                                f"{fmt(abs(r['converted']), s.currency)}; the remaining "
                                f"{fmt(fee['fee'])} is an unbooked "
                                f"{fee['fee_model'].replace('_', ' ').lower()} charge. "
                                f"Post the fee to bank charges, not to the vendor.",
                                {"fx": r, "fee": fee}), calls, {"fee": fee}
        return None, calls, {"reason": "no foreign-currency candidate reconciles"}

    def _try_fee(self, s: Record, cands: list[Record]):
        calls = 0
        for l in cands:
            if l.currency != s.currency or abs(l.amount) <= abs(s.amount):
                continue
            fee = tools.fee_explains(l.amount, s.amount)
            calls += 1
            if fee["ok"]:
                model = fee["fee_model"].replace("_", " ").lower()
                return self._mk(s, [l], 0.93, "L3_AGENTIC", ["fee_explains"], 0,
                                f"Gross {fmt(abs(l.amount), l.currency)} settled net at "
                                f"{fmt(abs(s.amount), s.currency)}; the "
                                f"{fmt(fee['fee'])} difference is exactly the {model} "
                                f"schedule. Revenue is recognised gross, the fee is an "
                                f"expense - netting it would understate both.",
                                {"fee": fee, "gross": l.amount, "net": s.amount}), calls, {"fee": fee}
        return None, calls, {"reason": "no fee schedule reconciles a candidate"}

    def _try_tax(self, s: Record, cands: list[Record]):
        calls = 0
        # Several open bills can each be explained by *some* statutory rate, so
        # taking the first that works would settle against a coincidence while a
        # better-evidenced document sat further down the list. Collect the
        # candidates that reconcile and prefer the one whose own declared
        # jurisdiction produces the rate - that is corroboration rather than a
        # rate that merely happens to fit.
        hits: list[tuple[float, Record, dict]] = []
        for l in cands:
            if l.currency != s.currency or abs(s.amount) <= abs(l.amount):
                continue
            t = tools.infer_tax_gross(l.amount, s.amount)
            calls += 1
            if t["ok"]:
                corroborated = l.meta.get("jurisdiction") == t["jurisdiction"]
                hits.append((0.93 if corroborated else 0.86, l, t))

        if hits:
            hits.sort(key=lambda h: (-h[0], h[1].id))
            conf, l, t = hits[0]
            declared = l.meta.get("jurisdiction")
            note = ("consistent with the jurisdiction on the document"
                    if declared == t["jurisdiction"]
                    else f"inferred - the document declares {declared or 'no jurisdiction'}")
            return self._mk(s, [l], conf, "L3_AGENTIC", ["infer_tax_gross"], 0,
                            f"Document is tax-exclusive at "
                            f"{fmt(abs(l.amount), l.currency)}; settlement of "
                            f"{fmt(abs(s.amount), s.currency)} is explained by "
                            f"{t['rate_pct']}% {t['jurisdiction']} ({note}). "
                            f"Recoverable input tax of {fmt(t['tax'])} should be "
                            f"posted to the tax control account.",
                            {"tax": t, "declared_jurisdiction": declared,
                             "competing_candidates": len(hits) - 1}), calls, {"tax": t}
        return None, calls, {"reason": "no statutory rate reconciles a candidate"}

    def _try_split(self, s: Record, cands: list[Record]):
        """One remittance against several documents - including short-pays,
        where a credit note carries the opposite sign and nets down the bill."""
        same_party = [c for c in cands
                      if counterparty_score(s.counterparty or s.description,
                                            c.counterparty) >= 0.88]
        if len(same_party) < 2:
            return None, 0, {"reason": "fewer than two same-party documents open"}
        r = tools.subset_sum(same_party, s.amount, anchor_date=s.txn_date)
        if not r["ok"]:
            return None, 1, {"reason": r["error"], "pool": len(same_party)}
        docs = [self.c.by_id[i] for i in r["ids"]]
        has_credit = any(d.doc_type == "CREDIT_NOTE" for d in docs)
        kind = "short-payment net of an open credit note" if has_credit else "batch remittance"
        lines = ", ".join(f"{d.id} {fmt(d.amount, d.currency)}" for d in docs)
        conf = 0.88 if has_credit else 0.90
        if r["terms"] >= 4:
            conf -= 0.04       # more terms, more chance of coincidence
        return self._mk(s, docs, conf, "L3_AGENTIC", ["subset_sum"], r["residual"],
                        f"Single settlement of {fmt(abs(s.amount), s.currency)} is a "
                        f"{kind} covering {r['terms']} documents: {lines}. Searched "
                        f"{r['searched']} same-counterparty combinations; this is the "
                        f"only subset that closes.",
                        {"subset_sum": r, "credit_note_applied": has_credit}), 1, r

    # -- refusal -----------------------------------------------------------
    def _diagnose(self, s: Record, cands: list[Record], attempted: list[str]) -> Exception_:
        near = min(cands, key=lambda l: abs(abs(l.amount) - abs(s.amount)))
        gap = abs(s.amount) - abs(near.amount)
        pct = abs(gap) * 100.0 / max(abs(near.amount), 1)
        severity = "CRITICAL" if abs(gap) >= 500_00 else "HIGH" if abs(gap) >= 50_00 else "MEDIUM"
        return Exception_(
            [s.id, near.id], "AMOUNT_VARIANCE_UNEXPLAINED", severity,
            f"Settlement {fmt(abs(s.amount), s.currency)} on {s.txn_date} is the "
            f"closest fit to {near.id} ({fmt(abs(near.amount), near.currency)}, "
            f"'{near.counterparty}'), a variance of {fmt(abs(gap))} ({pct:.2f}%). "
            f"Tested and rejected: {', '.join(attempted) or 'no applicable hypothesis'}. "
            f"No FX rate, fee schedule, statutory tax rate or document subset "
            f"explains the difference.",
            "Do not book. Confirm the remitted amount against the vendor statement - "
            "an unexplained overpayment is the signature of both keying error and "
            "invoice fraud.",
            s.amount, s.currency,
            {"nearest_candidate": near.id, "variance": gap, "variance_pct": round(pct, 3),
             "hypotheses_rejected": attempted,
             "other_candidates": [c.id for c in cands[:5]]})

    # -- construction ------------------------------------------------------
    def _mk(self, s: Record, docs: list[Record], conf: float, layer: str,
            used: list[str], residual: int, rationale: str, ev: dict) -> Match:
        return Match(self.c._next_mid(), [s.id], [d.id for d in docs],
                     round(conf, 4), layer, rationale, ev, residual, used)


class HeuristicPlanner:
    """Orders hypotheses by the cheapest discriminating signal available.

    Reading the *shape* of the discrepancy before testing it keeps L3 to ~2
    tool calls per item instead of brute-forcing all four hypotheses. Order
    matters for correctness too: split-sum is tried last because with enough
    open documents it can coincidentally close on a difference that FX or tax
    explains exactly.
    """

    def plan(self, s: Record, cands: list[Record]) -> list[str]:
        plan: list[str] = []
        if any(c.currency != s.currency for c in cands):
            plan.append("fx")
        smaller = [c for c in cands if abs(c.amount) < abs(s.amount) and c.currency == s.currency]
        larger = [c for c in cands if abs(c.amount) > abs(s.amount) and c.currency == s.currency]
        if larger:
            plan.append("fee")          # settled less than billed -> deduction
        if smaller:
            plan.append("tax")          # settled more than billed -> add-on
        plan.append("split")
        return plan


class LLMPlanner:
    """Optional LLM-backed planner. Same tool belt, same gates.

    It sees the settlement line and candidate documents and returns an ordered
    hypothesis list. It cannot book, cannot compute, and cannot widen the
    residual tolerance - a bad plan costs a few extra tool calls and then falls
    through to the same refusal path.
    """

    # No vendor or model is hardcoded: a controller should be able to swap the
    # planner without a code change, and pinning a model id in the source is how
    # a system quietly stops working when that id is retired.
    DEFAULT_MODEL = os.environ.get("LEDGERGUARD_MODEL", "")

    def __init__(self, model: str | None = None) -> None:
        self.model = model or self.DEFAULT_MODEL
        self.fallback = HeuristicPlanner()
        try:
            import anthropic  # noqa: F401
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.available = True
        except Exception:
            self.client = None
            self.available = False

    def plan(self, s: Record, cands: list[Record]) -> list[str]:
        if not self.available:
            return self.fallback.plan(s, cands)
        lines = "\n".join(
            f"- {c.id} {c.doc_type} {fmt(c.amount, c.currency)} {c.currency} "
            f"'{c.counterparty}' dated {c.txn_date} meta={c.meta}" for c in cands)
        prompt = (
            "You are a financial controller triaging one unreconciled settlement.\n"
            f"SETTLEMENT: {s.id} {fmt(s.amount, s.currency)} {s.currency} on "
            f"{s.txn_date}, descriptor '{s.description}'.\n"
            f"OPEN LEDGER DOCUMENTS:\n{lines}\n\n"
            "Available hypotheses: fx (foreign-currency conversion, possibly plus a "
            "wire fee), fee (processor/wire fee deducted from a gross amount), "
            "tax (document is tax-exclusive, settlement is tax-inclusive), "
            "split (one remittance covering several documents, possibly net of a "
            "credit note).\n"
            "Return ONLY a JSON array of hypothesis names, most likely first. "
            "Do not perform arithmetic; a deterministic tool will verify each one.")
        try:
            import json
            resp = self.client.messages.create(
                model=self.model, max_tokens=200,
                messages=[{"role": "user", "content": prompt}])
            text = resp.content[0].text.strip()
            text = text[text.index("["):text.rindex("]") + 1]
            got = [h for h in json.loads(text) if h in {"fx", "fee", "tax", "split"}]
            return got or self.fallback.plan(s, cands)
        except Exception:
            return self.fallback.plan(s, cands)
