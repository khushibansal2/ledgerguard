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
        for l in self.c.candidate_documents(s):
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
                s.amount, s.currency, {"description": s.description},
                missing_evidence=(
                    "A vendor-master entry mapping the descriptor "
                    f"'{s.description}' to a known counterparty, or the invoice "
                    "itself if this expense was never entered."))
            return oc

        # A planner is an untrusted input. It may be a model, and a model can be
        # rate-limited, wedged, or simply wrong - none of which is a reason to
        # abandon a close. Fall back to the built-in ordering and carry on.
        try:
            plan = list(self.planner.plan(s, cands))
        except Exception as exc:
            self.c.audit.emit("PLANNER_FAILED", settlement=s.id,
                              planner=type(self.planner).__name__,
                              error=f"{type(exc).__name__}: {exc}",
                              fallback="HeuristicPlanner")
            plan = HeuristicPlanner().plan(s, cands)

        # The competing-allocation check is a safety property, so it cannot be
        # left to the planner to enable. A plan that proposes `split` without
        # `instalment` (or the reverse) would otherwise book the one reading it
        # was told about and never discover that another reading closes just as
        # exactly - silently disabling the check by omission. Red-teaming found
        # this: a planner emitting mostly-invalid hypothesis names mis-booked
        # value the built-in planner did not.
        if "split" in plan and "instalment" not in plan:
            plan.append("instalment")
        elif "instalment" in plan and "split" not in plan:
            plan.append("split")

        self.c.audit.emit("L3_PLAN", settlement=s.id, hypotheses=plan,
                          planner=type(self.planner).__name__)

        attempted: list[str] = []
        winner: tuple[str, Match] | None = None
        for hypothesis in plan:
            fn = getattr(self, f"_try_{hypothesis}", None)
            if fn is None:
                continue
            attempted.append(hypothesis)
            m, calls, ev = fn(s, cands)
            oc.tool_calls += calls
            self.c.audit.emit("L3_HYPOTHESIS", settlement=s.id,
                              hypothesis=hypothesis, accepted=bool(m), evidence=ev)
            if not m:
                continue
            if winner is None:
                winner = (hypothesis, m)
                # Allocation hypotheses are the ones that can both be true of the
                # same figures - a vendor running a short-payment and an
                # instalment plan at once produces two groupings that each
                # balance to the cent. Keep testing rather than taking the first.
                if hypothesis not in ("split", "instalment"):
                    break
                continue
            if set(m.ledger_ids) != set(winner[1].ledger_ids):
                # Two arithmetically closed readings, different documents. The
                # bank file cannot say which is real, and booking either would
                # be a coin flip dressed as a reconciliation.
                self.c.audit.emit("AMBIGUITY_DETECTED", settlement=s.id,
                                  competing=[winner[0], hypothesis],
                                  option_a=winner[1].ledger_ids,
                                  option_b=m.ledger_ids)
                oc.exception = self._ambiguous(s, winner, (hypothesis, m))
                return oc

        if winner:
            oc.match = winner[1]
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
                                {"fx": r, "fee": fee}), calls, {"fx": r, "fee": fee}
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

    def _try_instalment(self, s: Record, cands: list[Record]):
        """Several payments clearing one bill - the mirror of a split remittance.

        Harder than a split, because no single settlement will ever tie out and
        the evidence for the grouping is weaker: any two payments to a vendor
        can be added together. The search is therefore anchored on the
        settlement in hand, capped at four terms, and confidence is set below
        the split case because a coincidental sum is easier to find on the
        payment side than on the document side.
        """
        siblings = [x for x in self.c.settlements
                    if x.id not in self.c.consumed
                    and x.currency == s.currency
                    and (x.amount > 0) == (s.amount > 0)
                    and counterparty_score(s.counterparty or s.description,
                                           x.counterparty or x.description) >= 0.88]
        if len(siblings) < 2:
            return None, 0, {"reason": "no sibling settlement to the same party"}

        calls = 0
        for l in cands:
            if l.currency != s.currency or abs(l.amount) <= abs(s.amount):
                continue
            r = tools.subset_sum_including(siblings, l.amount, s.id)
            calls += 1
            if not r["ok"]:
                continue
            pays = [self.c.by_id[i] for i in r["ids"]]
            # You cannot pay an instalment against a bill that does not exist
            # yet. Without this the search happily assembles payments that
            # straddle the invoice date and produces a group that balances but
            # never happened.
            if any(p.txn_date < l.txn_date for p in pays):
                continue
            lines = ", ".join(f"{p.id} {fmt(p.amount, p.currency)} on {p.txn_date}"
                              for p in pays)
            conf = 0.88 if len(pays) == 2 else 0.84
            m = Match(self.c._next_mid(), [p.id for p in pays], [l.id],
                      round(conf, 4), "L3_AGENTIC",
                      f"{l.id} for {fmt(abs(l.amount), l.currency)} is cleared by "
                      f"{len(pays)} instalments rather than one payment: {lines}. "
                      f"Together they settle the bill exactly; individually none "
                      f"of them ties out, which is why no earlier layer could "
                      f"match them.",
                      {"instalments": r, "bill": l.id}, r["residual"],
                      ["subset_sum_including"])
            return m, calls, r
        return None, calls, {"reason": "no bill is cleared by a group including this payment"}

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

    def _ambiguous(self, s: Record, a: tuple[str, Match],
                   b: tuple[str, Match]) -> Exception_:
        """Refuse a settlement that has more than one true-looking allocation."""
        def describe(name: str, m: Match) -> str:
            docs = ", ".join(f"{i} {fmt(self.c.by_id[i].amount, self.c.by_id[i].currency)}"
                             for i in m.ledger_ids)
            return f"{name}: {docs}"

        return Exception_(
            [s.id] + sorted(set(a[1].ledger_ids) | set(b[1].ledger_ids)),
            "AMBIGUOUS_ALLOCATION", "MEDIUM",
            f"{fmt(abs(s.amount), s.currency)} settled on {s.txn_date} for "
            f"'{s.counterparty}' can be allocated two ways, both closing to the "
            f"cent - {describe(*a)}; {describe(*b)}. The arithmetic cannot "
            f"choose between them and neither can this engine.",
            "Obtain the remittance advice and allocate manually; the correct "
            "answer exists but is not present in the source records.",
            s.amount, s.currency,
            {"option_a": {"hypothesis": a[0], "ledger": a[1].ledger_ids},
             "option_b": {"hypothesis": b[0], "ledger": b[1].ledger_ids}},
            missing_evidence=(
                "Remittance advice showing which invoices this payment was "
                "applied to. Both allocations balance, so no amount, date or "
                "vendor field in the current data can separate them."))

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
             "other_candidates": [c.id for c in cands[:5]]},
            missing_evidence=(
                f"A remittance advice for the {fmt(abs(s.amount), s.currency)} "
                f"payment, or a vendor statement showing what the extra "
                f"{fmt(abs(gap))} was for - a credit, a rebilled cost or an "
                f"error. Nothing in the current records distinguishes these."))

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
        # Split before instalment, always. Both can close on the same figures
        # when a vendor has a short-payment and an instalment plan running at
        # once, but a split is evidenced entirely by documents already on the
        # book, whereas an instalment additionally assumes that other payments
        # belong to this one. Prefer the reading that assumes less.
        plan.append("split")
        if larger:
            plan.append("instalment")
        return plan


class LLMPlanner:
    """Optional LLM-backed planner. Same tool belt, same gates.

    It sees the settlement and the candidate documents and returns an ordered
    hypothesis list. It cannot book, cannot compute, and cannot widen the
    residual tolerance - a bad plan costs a few extra tool calls and then falls
    through to the same refusal path. `redteam.py` demonstrates that against
    six deliberately hostile planners rather than asserting it.

    Spoken over plain HTTP in the OpenAI chat-completions dialect rather than
    through a vendor SDK. Three reasons, in order of importance:

      1. It keeps the project dependency-free. urllib is standard library, so
         the whole controller still installs with nothing.
      2. Every provider worth using speaks this dialect - including the free
         tiers and a locally-run Ollama - so demonstrating the agentic path
         does not require anybody's paid account.
      3. A vendor SDK here would quietly make one company's uptime a
         dependency of closing the books, which is the precise thing this
         architecture exists to avoid.

    Configured entirely by environment (see README for free providers):

        LEDGERGUARD_API_BASE   e.g. https://api.groq.com/openai/v1
        LEDGERGUARD_MODEL      e.g. llama-3.3-70b-versatile
        LEDGERGUARD_API_KEY    omit entirely for a local Ollama

    No vendor or model id is hardcoded: pinning one in source is how a system
    quietly stops working the day that id is retired.
    """

    TIMEOUT = 12.0
    VALID = {"fx", "fee", "tax", "split", "instalment"}

    def __init__(self, model: str | None = None, api_base: str | None = None,
                 api_key: str | None = None) -> None:
        self.model = model or os.environ.get("LEDGERGUARD_MODEL", "")
        self.api_base = (api_base or os.environ.get("LEDGERGUARD_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LEDGERGUARD_API_KEY", "")
        self.fallback = HeuristicPlanner()
        self.available = bool(self.model and self.api_base)
        self.calls = 0
        self.failures = 0

    def _prompt(self, s: Record, cands: list[Record]) -> str:
        lines = "\n".join(
            f"- {c.id} {c.doc_type} {fmt(c.amount, c.currency)} "
            f"'{c.counterparty}' dated {c.txn_date}"
            + (f" meta={c.meta}" if c.meta else "")
            for c in cands)
        return (
            "You are a financial controller triaging one unreconciled "
            "settlement. Decide which explanations are worth testing, in order.\n\n"
            f"SETTLEMENT: {s.id}, {fmt(s.amount, s.currency)} on {s.txn_date}, "
            f"bank descriptor '{s.description}'.\n\n"
            f"OPEN LEDGER DOCUMENTS:\n{lines}\n\n"
            "HYPOTHESES:\n"
            "  fx          the document is in another currency\n"
            "  fee         a processor or wire fee was deducted from a gross amount\n"
            "  tax         the document is tax-exclusive, the payment tax-inclusive\n"
            "  split       one payment covers several documents, maybe net of a credit note\n"
            "  instalment  this payment is one of several clearing a single larger bill\n\n"
            "Return ONLY a JSON array of hypothesis names, most likely first. "
            "Do not do arithmetic and do not explain - a deterministic tool "
            "verifies each hypothesis and a policy gate decides what is booked.")

    def _post(self, prompt: str) -> str:
        import json
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 120,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.api_base}/chat/completions",
                                     data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]

    def plan(self, s: Record, cands: list[Record]) -> list[str]:
        """Never raises. A planner is an accelerator, so if it is slow, wedged,
        rate-limited or talking nonsense the close continues on the built-in
        ordering and the only thing that changes is the tool-call count."""
        if not self.available:
            return self.fallback.plan(s, cands)
        import json
        self.calls += 1
        try:
            text = self._post(self._prompt(s, cands))
            text = text[text.index("["):text.rindex("]") + 1]
            got = [h for h in json.loads(text)
                   if isinstance(h, str) and h in self.VALID]
            # Deduplicate while keeping the model's ordering - that ordering is
            # the entire contribution it is permitted to make.
            seen: set[str] = set()
            plan = [h for h in got if not (h in seen or seen.add(h))]
            return plan or self.fallback.plan(s, cands)
        except Exception:
            self.failures += 1
            return self.fallback.plan(s, cands)
