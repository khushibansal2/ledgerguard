"""Reconstruct one settlement's full decision path from the audit trail.

The most common fair question about a system like this is "where is the agent,
exactly?" - because most of the pipeline is deliberately deterministic. This
answers it for a single record, using nothing but the hash-chained log the run
already wrote:

    which layers tried and failed, what the planner proposed, which tool was
    called with what arguments, the arithmetic that closed, and the policy
    decision that let it be booked.

Nothing here re-runs the engine or re-derives anything. If the trace says a
tool returned $179, that is the value the tool returned during the run, and the
chain hash covers it.

    python -m ledgerguard.trace BANK-5042
"""
from __future__ import annotations

import sys

from .audit import AuditLog
from .engine import Controller, Policy
from .generate import generate
from .models import fmt

W = 74


def _rule(ch: str = "-") -> str:
    return "  " + ch * (W - 4)


def render(ctrl: Controller, sid: str) -> str:
    rec = ctrl.by_id.get(sid)
    if rec is None:
        return f"no such record: {sid}"

    ev = ctrl.audit.entries
    out: list[str] = []
    out.append("=" * W)
    out.append(f"  DECISION TRACE  {sid}")
    out.append("=" * W)
    out.append(f"  {fmt(rec.amount, rec.currency)} on {rec.txn_date}"
               f"  [{rec.source}]")
    out.append(f"  descriptor: {rec.description!r}")
    out.append(f"  counterparty as banked: {rec.counterparty!r}")
    out.append("")

    # --- which layers declined it -----------------------------------------
    booked = next((e for e in ev if e["event"] == "MATCH_PROPOSED"
                   and sid in e.get("settlement", [])), None)
    reached_l3 = any(e["event"] == "L3_RETRIEVAL" and e.get("settlement") == sid
                     for e in ev)
    layer = booked["layer"] if booked else None

    out.append("  LAYER PATH")
    out.append(_rule())
    ladder = [("L0", "reversal pairing", "L0_REVERSAL"),
              ("L1", "exact reference + amount", "L1_DETERMINISTIC"),
              ("L2", "entity + timing similarity", "L2_SIMILARITY"),
              ("L3", "tool-using resolver", "L3_AGENTIC")]
    for tag, label, key in ladder:
        if layer == key:
            out.append(f"  {tag}  {label:<34} MATCHED")
            break
        if key == "L3_AGENTIC" and reached_l3 and layer != key:
            out.append(f"  {tag}  {label:<34} investigated, refused")
        else:
            out.append(f"  {tag}  {label:<34} no match")
    out.append("")

    if not reached_l3:
        out.append("  Settled by deterministic arithmetic; no model was consulted.")
        if booked:
            out.append("")
            out.append("  RATIONALE")
            out.append(_rule())
            for line in _wrap(booked["rationale"]):
                out.append(f"  {line}")
        out.append("=" * W)
        return "\n".join(out)

    # --- retrieval ---------------------------------------------------------
    ret = next(e for e in ev if e["event"] == "L3_RETRIEVAL" and e["settlement"] == sid)
    out.append("  CANDIDATE RETRIEVAL")
    out.append(_rule())
    n_c = len(ret["candidates"])
    out.append(f"  {n_c} open document{'s' if n_c != 1 else ''} "
               f"resemble{'' if n_c != 1 else 's'} this counterparty")
    for cid in ret["candidates"][:6]:
        c = ctrl.by_id[cid]
        out.append(f"    {cid:<12}{fmt(c.amount, c.currency):>14}  "
                   f"{c.counterparty[:28]}")
    if len(ret["candidates"]) > 6:
        out.append(f"    ... {len(ret['candidates']) - 6} more")
    out.append("")

    # --- plan --------------------------------------------------------------
    plan = next((e for e in ev if e["event"] == "L3_PLAN" and e["settlement"] == sid), None)
    if plan:
        out.append(f"  HYPOTHESES  (ordered by {plan['planner']})")
        out.append(_rule())
        meaning = {
            "fx": "the document is in another currency",
            "fee": "a processor or wire fee was deducted",
            "tax": "the document is net of tax, the payment is gross",
            "split": "one payment covers several documents",
            "instalment": "this payment is one of several clearing one document",
        }
        for i, h in enumerate(plan["hypotheses"], 1):
            out.append(f"    H{i}  {h:<12} {meaning.get(h, '')}")
        out.append("")

    # --- tool execution ----------------------------------------------------
    tried = [e for e in ev if e["event"] == "L3_HYPOTHESIS" and e["settlement"] == sid]
    out.append("  TOOL EXECUTION")
    out.append(_rule())
    for t in tried:
        verdict = "CLOSED" if t["accepted"] else "rejected"
        out.append(f"    {t['hypothesis']:<12} {verdict}")
        for line in _evidence_lines(t["evidence"]):
            out.append(f"      {line}")
    out.append("")

    # --- outcome -----------------------------------------------------------
    if booked:
        decision = next((e for e in ev if e["event"] in ("MATCH_BOOKED", "MATCH_HELD")
                         and e["match_id"] == booked["match_id"]), None)
        out.append("  ARITHMETIC")
        out.append(_rule())
        for line in _wrap(booked["rationale"]):
            out.append(f"  {line}")
        out.append(f"  residual {fmt(booked['residual'])} "
                   f"(tolerance {fmt(ctrl.policy.max_residual)})")
        out.append("")
        out.append("  POLICY GATE")
        out.append(_rule())
        amt = abs(rec.amount)
        out.append(f"    band            {ctrl.policy.band(amt)} ({fmt(amt)})")
        out.append(f"    requires        {ctrl.policy.required_confidence(amt):.2f}")
        out.append(f"    confidence      {booked['confidence']:.2f}")
        out.append("")
        verdict = "BOOK" if decision and decision["event"] == "MATCH_BOOKED" else "HOLD FOR APPROVAL"
        out.append(f"  DECISION: {verdict}")
        if decision:
            for line in _wrap(decision["policy"]):
                out.append(f"    {line}")
    else:
        exc = next((e for e in ev if e["event"] == "EXCEPTION_RAISED"
                    and sid in e.get("records", [])), None)
        out.append("  DECISION: REFUSE")
        out.append(_rule())
        if exc:
            out.append(f"    {exc['category']} / {exc['severity']}")
            for line in _wrap(exc["reason"]):
                out.append(f"    {line}")
            out.append("")
            out.append("    next:")
            for line in _wrap(exc["action"]):
                out.append(f"      {line}")
    out.append("=" * W)
    return "\n".join(out)


def _evidence_lines(evidence: dict) -> list[str]:
    """Render a tool result as the arithmetic a reviewer would check."""
    lines: list[str] = []
    if not isinstance(evidence, dict):
        return lines
    if "fx" in evidence:
        f = evidence["fx"]
        lines.append(f"fx_convert -> {fmt(abs(f['converted']))} at {f['rate']} "
                     f"[{f['source']}]")
    if "fee" in evidence and isinstance(evidence["fee"], dict):
        f = evidence["fee"]
        if f.get("ok"):
            lines.append(f"fee_explains -> {fmt(f['fee'])} matches {f['fee_model']}")
        else:
            lines.append(f"fee_explains -> no schedule fits a gap of "
                         f"{fmt(f.get('gap', 0))}")
    if "tax" in evidence and isinstance(evidence["tax"], dict):
        t = evidence["tax"]
        lines.append(f"infer_tax_gross -> {t['rate_pct']}% {t['jurisdiction']}, "
                     f"tax {fmt(t['tax'])}")
    if "ids" in evidence:
        lines.append(f"subset -> {' + '.join(evidence['ids'])} "
                     f"= {fmt(evidence.get('sum', 0))} (searched {evidence.get('searched', '?')})")
    if "reason" in evidence:
        lines.append(str(evidence["reason"]))
    return lines


def _wrap(text: str, width: int = W - 6) -> list[str]:
    words, line, out = str(text).split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def _utf8_stdout() -> None:
    """Currency symbols are data, not decoration - a mangled pound sign in a
    finance trace reads as a bug. Windows consoles default to cp1252, so ask
    for UTF-8 and fall back quietly where the stream cannot be reconfigured."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    sid = argv[0]
    seed = int(argv[1]) if len(argv) > 1 else 20260827
    batch = generate(seed=seed)
    ctrl = Controller(batch.records, policy=Policy(), audit=AuditLog()).run()
    print(render(ctrl, sid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
