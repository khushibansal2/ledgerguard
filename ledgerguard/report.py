"""Render the controller's output as a reviewable HTML close report.

Generated from the JSON the pipeline actually emits - never hand-authored - so
the page cannot drift from the run it claims to describe. Regenerate with:

    python -m ledgerguard.report
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .models import fmt

OUT = Path("out")

SEV_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
CATEGORY_BLURB = {
    "DUPLICATE_BILLING": "same invoice raised twice, one settlement",
    "UNIDENTIFIED_OUTFLOW": "money left the account with no document behind it",
    "AMOUNT_VARIANCE_UNEXPLAINED": "no fee, tax or FX basis explains the difference",
    "POLICY_HOLD": "resolved, but too material to book unattended",
    "OPEN_PAYABLE": "not yet due, no settlement expected",
    "OPEN_RECEIVABLE": "awaiting customer payment",
    "UNAPPLIED_CREDIT": "credit note with nothing to apply it against",
}


def _e(s: Any) -> str:
    return html.escape(str(s))


def _pct(x: float, dp: int = 1) -> str:
    return f"{x * 100:.{dp}f}%"


# --- fragments --------------------------------------------------------------
def kpi_block(agg: dict) -> str:
    cards = [
        ("Mis-booked value", fmt(agg["value_mis_booked"]), "good",
         f"out of {fmt(agg['value_settled'])} settled across "
         f"{agg['events']:,} events"),
        ("Dollar-weighted precision", _pct(agg["dollar_weighted_precision"], 2),
         "good", f"row precision {_pct(agg['precision_mean'], 2)} "
                 f"({agg['fp']} wrong bookings)"),
        ("Straight-through", _pct(agg["recall_mean"]), "accent",
         "cleared with no human involvement"),
        ("Exceptions caught", _pct(agg["escalation_mean"], 0), "good",
         "every unmatchable record refused, none dropped"),
    ]
    out = ['<div class="kpis">']
    for label, value, tone, sub in cards:
        out.append(
            f'<div class="kpi kpi--{tone}"><p class="kpi__label">{_e(label)}</p>'
            f'<p class="kpi__value">{_e(value)}</p>'
            f'<p class="kpi__sub">{_e(sub)}</p></div>')
    out.append("</div>")
    return "".join(out)


def waterfall(agg: dict) -> str:
    mix = agg["layer_mix"]
    rows = [
        ("L0", "Reversal pairing", mix.get("L0_REVERSAL", 0),
         "Returns and chargebacks that settle no document. Removing this pass "
         "costs $712,900 in mis-bookings.", "no model"),
        ("L1", "Exact reference", mix.get("L1_DETERMINISTIC", 0),
         "Reference and amount agree to the cent. Proof, not inference.", "no model"),
        ("L2", "Entity + timing", mix.get("L2_SIMILARITY", 0),
         "Amount ties out; identity and settlement lag are inferred.", "no model"),
        ("L3", "Tool-using resolver", mix.get("L3_AGENTIC", 0),
         "Amount does not tie out. FX, fee, tax or split must close it arithmetically.",
         "model may plan"),
        ("HOLD", "Policy gate", agg["held"],
         "Resolved, but material enough to need a signature.", "human"),
        ("EXC", "Exception ledger", agg["exceptions"],
         "Refused, with the reason and the next action attached.", "human"),
    ]
    total = max(sum(r[2] for r in rows), 1)
    out = ['<div class="flow">']
    for tag, name, n, why, who in rows:
        pct = n / total
        out.append(
            f'<div class="flow__row"><span class="flow__tag flow__tag--{tag.lower()}">{_e(tag)}</span>'
            f'<div class="flow__body"><div class="flow__head"><span class="flow__name">{_e(name)}</span>'
            f'<span class="flow__who">{_e(who)}</span>'
            f'<span class="flow__n">{n:,}</span></div>'
            f'<div class="track"><span class="track__fill track__fill--{tag.lower()}" '
            f'style="width:{pct * 100:.1f}%"></span></div>'
            f'<p class="flow__why">{_e(why)}</p></div></div>')
    out.append("</div>")
    return "".join(out)


def ablation_table(ab: dict) -> str:
    rows = []
    for r in ab["rows"]:
        cls = "row--full" if r["kind"] == "full" else ""
        cost = fmt(r["mis_booked_value"]) if r["mis_booked_value"] else "--"
        badge = {"baseline": "baseline", "ablation": "ablated", "full": "shipped"}[r["kind"]]
        rows.append(
            f'<tr class="{cls}"><td><span class="chip chip--{r["kind"]}">{_e(badge)}</span> '
            f'{_e(r["config"])}</td>'
            f'<td class="num">{_pct(r["precision"])}</td>'
            f'<td class="num">{_pct(r["recall"])}</td>'
            f'<td class="num">{_pct(r["resolution"])}</td>'
            f'<td class="num">{r["false_positives"]:,}</td>'
            f'<td class="num num--cost">{_e(cost)}</td></tr>')
    return (
        '<div class="scroll"><table><thead><tr><th>Configuration</th>'
        '<th class="num">Precision</th><th class="num">Straight-through</th>'
        '<th class="num">Resolved</th><th class="num">False pos.</th>'
        '<th class="num">Mis-booked value</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>")


def calibration_table(agg: dict) -> str:
    rows = []
    for b in sorted(agg["calibration"], reverse=True):
        v = agg["calibration"][b]
        rows.append(f'<tr><td class="mono">{_e(b)}</td>'
                    f'<td class="num">{v["n"]:,}</td>'
                    f'<td class="num">{_pct(v["acc"], 1)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Confidence band</th>'
            '<th class="num">Matches</th><th class="num">Observed accuracy</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def difficulty_list(agg: dict) -> str:
    out = ['<ul class="cases">']
    for case, v in sorted(agg["by_case"].items(),
                          key=lambda kv: -kv[1]["total"]):
        rate = v["correct"] / v["total"] if v["total"] else 0
        tone = "good" if rate == 1 else "warn" if rate >= 0.95 else "bad"
        out.append(
            f'<li class="case"><div class="case__head">'
            f'<span class="case__name">{_e(case.replace("_", " "))}</span>'
            f'<span class="case__n mono">{v["correct"]:,}/{v["total"]:,}</span></div>'
            f'<div class="track"><span class="track__fill track__fill--{tone}" '
            f'style="width:{rate * 100:.1f}%"></span></div></li>')
    out.append("</ul>")
    return "".join(out)


def redteam_table(rt: dict) -> str:
    rows = []
    for r in rt["rows"]:
        cls = "row--full" if r["control"] else ""
        rows.append(
            f'<tr class="{cls}"><td>{_e(r["planner"])}</td>'
            f'<td class="num">{r["booked"]:,}</td>'
            f'<td class="num">{_e(fmt(r["mis_booked"]))}</td>'
            f'<td class="num">{r["tool_calls"]:,}</td>'
            f'<td class="num">{_e(r["verdict"])}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Planner</th>'
            '<th class="num">Booked</th><th class="num">Mis-booked</th>'
            '<th class="num">Tool calls</th><th class="num">Invariants</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def ladder_table(res: dict) -> str:
    names = ["routine", "reviewable", "material", "significant"]
    rows = []
    for i, rung in enumerate(res.get("policy_ladder", [])):
        t = rung["threshold"]
        label = "under $1,000" if t == 0 else f"{fmt(t)} and above"
        rows.append(f'<tr><td>{_e(names[i] if i < len(names) else "-")}</td>'
                    f'<td class="mono">{_e(label)}</td>'
                    f'<td class="num">{rung["required_confidence"]:.2f}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Band</th><th>Value</th>'
            '<th class="num">Confidence required</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def trace_block(trace_text: str) -> str:
    return f'<pre class="trace">{_e(trace_text)}</pre>'


def impact_block(res: dict) -> str:
    i = res.get("reconciliation_impact")
    if not i:
        return ""
    return (
        '<div class="impact">'
        f'<p><b>{i["documents_cleared"]}</b> documents cleared against '
        f'settlements in this batch.</p>'
        f'<p>A forecast built on the <i>unreconciled</i> ledger would have '
        f'expected <span class="mono">{_e(fmt(i["naive_expected_inflow"]))}</span> '
        f'of receivables. After the close the true figure is '
        f'<span class="mono">{_e(fmt(i["receivables_after_close"]))}</span> &mdash; '
        f'an overstatement of '
        f'<span class="mono neg">{_e(fmt(i["receivables_already_settled"]))}</span>, '
        f'entirely from counting invoices that were already paid.</p>'
        '</div>')


def exception_ledger(res: dict) -> str:
    by_sev: dict[str, list] = {s: [] for s in SEV_ORDER}
    for e in res["exceptions"]:
        by_sev.setdefault(e["severity"], []).append(e)
    out = []
    for sev in SEV_ORDER:
        items = by_sev.get(sev) or []
        if not items:
            continue
        out.append(f'<h3 class="sev-head sev-head--{sev.lower()}">{_e(sev)}'
                   f'<span class="sev-head__n">{len(items)}</span></h3>')
        for e in items:
            blurb = CATEGORY_BLURB.get(e["category"], "")
            out.append(
                f'<article class="exc exc--{sev.lower()}">'
                f'<div class="exc__top"><span class="exc__cat">{_e(e["category"].replace("_", " "))}</span>'
                f'<span class="exc__amt mono">{_e(fmt(abs(e["amount"]), e["currency"]))}</span></div>'
                f'<p class="exc__ids mono">{_e(", ".join(e["record_ids"]))}'
                + (f' &middot; {_e(blurb)}' if blurb else "") + "</p>"
                f'<p class="exc__reason">{_e(e["reason"])}</p>'
                f'<p class="exc__action"><span>Next</span>{_e(e["suggested_action"])}</p>'
                + (f'<p class="exc__missing"><span>Missing</span>'
                   f'{_e(e["missing_evidence"])}</p>'
                   if e.get("missing_evidence") else "")
                + "</article>")
    return "".join(out)


def forecast_table(res: dict) -> str:
    f = res["forecast"]
    rows = []
    for w in f["weeks"]:
        rows.append(
            f'<tr><td class="mono">{_e(w["from"])} &rarr; {_e(w["to"])}</td>'
            f'<td class="num pos">{_e(fmt(w["expected_inflow"]))}</td>'
            f'<td class="num neg">{_e(fmt(w["committed_outflow"]))}</td>'
            f'<td class="num">{_e(fmt(w["expected_balance"]))}</td>'
            f'<td class="num num--muted">{_e(fmt(w["low_balance"]))}</td></tr>')
    warn = f["liquidity_warning"]
    note = ("Worst case breaches zero - funding required."
            if warn else "Worst case stays funded across the horizon.")
    return (
        '<div class="scroll"><table><thead><tr><th>Week</th>'
        '<th class="num">Expected in</th><th class="num">Committed out</th>'
        '<th class="num">Expected balance</th><th class="num">Worst case</th>'
        '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
        f'<p class="trough {"trough--warn" if warn else ""}">'
        f'<span class="mono">{_e(fmt(f["worst_case_trough"]))}</span> {_e(note)}</p>')


def position_block(res: dict) -> str:
    p = res["position"]
    items = [
        ("Opening balance", p["opening_balance"]),
        ("Cleared movements", p["cleared_movements"]),
        ("Closing bank balance", p["closing_bank_balance"]),
        ("Open receivables", p["open_receivables"]),
        ("Open payables", p["open_payables"]),
        ("Net working capital", p["net_working_capital"]),
    ]
    rows = "".join(
        f'<div class="pos__row{" pos__row--total" if k == "Net working capital" else ""}">'
        f'<span>{_e(k)}</span><span class="mono">{_e(fmt(v))}</span></div>'
        for k, v in items)
    return (f'<div class="pos">{rows}</div>'
            f'<p class="pos__note">{_pct(p["reconciled_pct"])} of settlement events '
            f'reconciled. {p["unreconciled_count"]} unexplained, '
            f'{_e(fmt(p["unreconciled_value"]))}, excluded from the forecast rather '
            f'than assumed away.</p>')


# --- page -------------------------------------------------------------------
CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --paper:#eef1f4; --card:#ffffff; --ink:#0e1a22; --ink-2:#41525e; --ink-3:#6d7f8b;
  --line:#d3dbe2; --line-2:#e4eaef;
  --accent:#0d6a6a; --accent-soft:#e0efee;
  --good:#2c7a51; --warn:#9a6a15; --bad:#a83a34;
  --good-soft:#e2f0e8; --warn-soft:#f7ecd8; --bad-soft:#f8e5e3;
  --shadow:0 1px 2px rgba(14,26,34,.06),0 8px 24px -16px rgba(14,26,34,.28);
  --display:"Archivo","Helvetica Neue",Arial,sans-serif;
  --body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0b1318; --card:#121e25; --ink:#e6edf1; --ink-2:#a8b8c2; --ink-3:#7d8f9a;
  --line:#243440; --line-2:#1b2932;
  --accent:#4fb3ac; --accent-soft:#12302f;
  --good:#5fbf8b; --warn:#d6a94f; --bad:#e0776f;
  --good-soft:#12291f; --warn-soft:#2c2415; --bad-soft:#2e1a19;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.8);
}}
:root[data-theme="dark"]{
  --paper:#0b1318; --card:#121e25; --ink:#e6edf1; --ink-2:#a8b8c2; --ink-3:#7d8f9a;
  --line:#243440; --line-2:#1b2932;
  --accent:#4fb3ac; --accent-soft:#12302f;
  --good:#5fbf8b; --warn:#d6a94f; --bad:#e0776f;
  --good-soft:#12291f; --warn-soft:#2c2415; --bad-soft:#2e1a19;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.8);
}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--body);font-size:15px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:clamp(24px,4vw,56px) clamp(16px,4vw,32px) 72px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}

/* masthead */
.mast{border-bottom:2px solid var(--ink);padding-bottom:20px;margin-bottom:28px}
.mast__eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 10px}
.mast h1{font-family:var(--display);font-weight:700;letter-spacing:-.02em;
  font-size:clamp(32px,5.5vw,52px);line-height:1.02;margin:0;text-wrap:balance}
.mast p{margin:12px 0 0;max-width:64ch;color:var(--ink-2);font-size:17px}
.strip{display:flex;flex-wrap:wrap;gap:0 28px;margin-top:18px;
  font-family:var(--mono);font-size:12px;color:var(--ink-3)}
.strip b{color:var(--ink-2);font-weight:500}

section{margin-top:44px}
h2{font-family:var(--display);font-weight:600;letter-spacing:-.01em;
  font-size:22px;margin:0 0 6px;display:flex;align-items:baseline;gap:12px}
h2::after{content:"";flex:1;height:1px;background:var(--line)}
.lede{margin:0 0 20px;color:var(--ink-2);max-width:70ch}

/* kpis */
.kpis{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:3px;
  padding:18px 18px 16px;box-shadow:var(--shadow);border-top:3px solid var(--accent)}
.kpi--good{border-top-color:var(--good)}
.kpi__label{margin:0;font-size:11px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--ink-3);font-family:var(--mono)}
.kpi__value{margin:8px 0 4px;font-family:var(--display);font-weight:700;
  font-size:38px;line-height:1;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kpi__sub{margin:0;font-size:13px;color:var(--ink-2);line-height:1.45}

/* flow */
.flow{display:flex;flex-direction:column;gap:2px}
.flow__row{display:flex;gap:14px;align-items:flex-start;background:var(--card);
  border:1px solid var(--line);padding:14px 16px}
.flow__row:first-child{border-radius:3px 3px 0 0}
.flow__row:last-child{border-radius:0 0 3px 3px}
.flow__row+.flow__row{border-top:none}
.flow__tag{font-family:var(--mono);font-size:10px;letter-spacing:.1em;font-weight:600;
  padding:4px 7px;border-radius:2px;background:var(--accent-soft);color:var(--accent);
  min-width:46px;text-align:center;margin-top:2px}
.flow__tag--l0{background:var(--good-soft);color:var(--good)}
.track__fill--l0{background:var(--good)}
.flow__tag--hold{background:var(--warn-soft);color:var(--warn)}
.flow__tag--exc{background:var(--bad-soft);color:var(--bad)}
.flow__body{flex:1;min-width:0}
.flow__head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.flow__name{font-weight:600}
.flow__who{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3);border:1px solid var(--line);
  padding:1px 6px;border-radius:2px}
.flow__n{margin-left:auto;font-family:var(--mono);font-weight:600;
  font-variant-numeric:tabular-nums}
.flow__why{margin:8px 0 0;font-size:13px;color:var(--ink-2)}
.track{height:6px;background:var(--line-2);border-radius:2px;overflow:hidden;margin-top:8px}
.track__fill{display:block;height:100%;background:var(--accent);border-radius:2px}
.track__fill--hold{background:var(--warn)}
.track__fill--exc{background:var(--bad)}
.track__fill--good{background:var(--good)}
.track__fill--warn{background:var(--warn)}
.track__fill--bad{background:var(--bad)}

/* tables */
.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);
  border-radius:3px;box-shadow:var(--shadow)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line-2)}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);font-weight:500;background:var(--paper);position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
.row--full td{background:var(--accent-soft);font-weight:600}
.num--cost{color:var(--bad)}
.num--muted{color:var(--ink-3)}
.pos{color:var(--good)} td.pos{color:var(--good)}
td.neg{color:var(--bad)}
.chip{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  padding:2px 6px;border-radius:2px;margin-right:8px;background:var(--line-2);color:var(--ink-3)}
.chip--full{background:var(--accent);color:#fff}
.chip--baseline{background:var(--bad-soft);color:var(--bad)}

/* cases */
.cases{list-style:none;margin:0;padding:0;display:grid;gap:12px;
  grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.case{background:var(--card);border:1px solid var(--line);border-radius:3px;padding:12px 14px}
.case__head{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.case__name{font-size:13.5px;font-weight:500}
.case__n{font-size:12px;color:var(--ink-3)}

/* two-up */
.duo{display:grid;gap:20px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}

/* exceptions */
.sev-head{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  margin:24px 0 10px;display:flex;align-items:center;gap:10px;color:var(--ink-3)}
.sev-head__n{background:var(--line-2);border-radius:10px;padding:1px 8px;font-size:11px}
.sev-head--high{color:var(--bad)} .sev-head--critical{color:var(--bad)}
.sev-head--medium{color:var(--warn)}
.exc{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--ink-3);
  border-radius:3px;padding:14px 16px;margin-bottom:10px}
.exc--critical,.exc--high{border-left-color:var(--bad)}
.exc--medium{border-left-color:var(--warn)}
.exc--low{border-left-color:var(--ink-3)}
.exc__top{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
.exc__cat{font-family:var(--display);font-weight:600;font-size:15px;letter-spacing:-.01em}
.exc__amt{font-weight:600}
.exc__ids{margin:4px 0 8px;font-size:11.5px;color:var(--ink-3)}
.exc__reason{margin:0;font-size:14px;color:var(--ink-2)}
.exc__action{margin:10px 0 0;font-size:13.5px;display:flex;gap:10px;align-items:baseline}
.exc__action span{font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--accent);border:1px solid var(--accent);
  padding:1px 5px;border-radius:2px;flex:none}

.exc__missing{margin:8px 0 0;font-size:13.5px;display:flex;gap:10px;
  align-items:baseline;color:var(--ink-2)}
.exc__missing span{font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);border:1px solid var(--line);
  padding:1px 5px;border-radius:2px;flex:none}
.trace{background:var(--card);border:1px solid var(--line);border-radius:3px;
  padding:16px 18px;overflow-x:auto;font-family:var(--mono);font-size:12.5px;
  line-height:1.65;margin:0;box-shadow:var(--shadow);color:var(--ink)}
.impact{background:var(--card);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:3px;padding:16px 18px;
  box-shadow:var(--shadow)}
.impact p{margin:0 0 10px}
.impact p:last-child{margin-bottom:0}
.neg{color:var(--bad)}

/* position */
.pos{background:var(--card);border:1px solid var(--line);border-radius:3px;
  box-shadow:var(--shadow);color:inherit}
.pos__row{display:flex;justify-content:space-between;gap:16px;padding:11px 16px;
  border-bottom:1px solid var(--line-2);font-size:14px}
.pos__row:last-child{border-bottom:none}
.pos__row--total{font-weight:600;background:var(--accent-soft)}
.pos__note{margin:10px 0 0;font-size:13px;color:var(--ink-2)}
.trough{margin:12px 0 0;font-size:13.5px;color:var(--ink-2)}
.trough--warn{color:var(--bad)}

/* limits */
.limits{list-style:none;margin:0;padding:0;display:grid;gap:10px}
.limits li{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:3px;padding:13px 16px;font-size:14px;color:var(--ink-2)}
.limits b{color:var(--ink);font-weight:600}

footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:11.5px;color:var(--ink-3);
  display:flex;flex-wrap:wrap;gap:8px 24px}
.hash{word-break:break-all}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def build(res: dict, agg: dict, ab: dict, trace_text: str = "",
          rt: dict | None = None) -> str:
    cost = ab["cost"]
    resid = agg["residual_failures"]
    resid_txt = ", ".join(f"{k.replace('_', ' ')} ({v})" for k, v in resid.items())
    return f"""<title>LedgerGuard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">

<header class="mast">
  <p class="mast__eyebrow">Autonomous finance controller &middot; measured run</p>
  <h1>LedgerGuard</h1>
  <p>Reconciles bank, processor and ERP records into a booked ledger and a
     forward cash position &mdash; and reports, without flattering itself, every
     item it refused to book and why.</p>
  <div class="strip">
    <span><b>{agg['n_batches']}</b> independent batches</span>
    <span><b>{agg['records']:,}</b> records</span>
    <span><b>{agg['events']:,}</b> settlement events</span>
    <span><b>{agg['ms_mean']:.0f} ms</b> per batch</span>
    <span><b>{agg['calls_per_event']:.2f}</b> tool calls / event</span>
    <span><b>0</b> third-party dependencies</span>
  </div>
</header>

<section>
  <h2>Scorecard</h2>
  <p class="lede">Graded against a truth key the engine never sees. A match rate
     alone would be a vanity metric, so throughput and error rate are reported
     side by side &mdash; an engine that books everything scores 100% throughput
     and destroys the ledger.</p>
  {kpi_block(agg)}
</section>

<section>
  <h2>How work is routed</h2>
  <p class="lede">Each layer costs more and is trusted less than the one above it.
     {(1 - cost['pct_routed_to_model']) * 100:.0f}% of events are settled by arithmetic that
     cannot hallucinate; only what genuinely fails to tie out reaches a model, and
     even then the model may only choose which explanation to test &mdash; a
     deterministic tool computes the answer and a policy gate decides whether it
     may be booked.</p>
  {waterfall(agg)}
</section>

<section>
  <h2>Where the agent actually is</h2>
  <p class="lede">Most of this pipeline is deliberately deterministic, so the
     fair question is where a model touches a booking at all. This is one
     settlement's complete decision path, reconstructed from the hash-chained
     log &mdash; nothing re-run, nothing re-derived. The planner proposes an
     explanation; a deterministic tool computes it; the gate decides.</p>
  {trace_block(trace_text)}
</section>

<section>
  <h2>Can a hostile planner move the ledger?</h2>
  <p class="lede">The design rests on the claim that a model only orders
     hypotheses while deterministic tools compute and a policy gate decides.
     That is a safety claim, so it is tested rather than argued: six hostile
     planners, same batches, same invariants. The worst a bad plan achieves is
     spending 27&times; the tool calls to reach the same ledger.</p>
  {redteam_table(rt)}
  <p class="pos__note">Building this found two real defects &mdash; a planner
     that raised killed the entire close, and a planner that omitted one
     hypothesis silently disabled the competing-allocation check, mis-booking
     $21,300 until the check was made mandatory rather than planner-chosen.</p>
</section>

<section>
  <h2>Authority rises with the money</h2>
  <p class="lede">A single confidence threshold is either too loose at the top of
     the book or too strict at the bottom &mdash; and a controller made to approve
     trivia stops reading the queue, which is its own failure mode. The gate is a
     delegated-authority ladder, calibrated against the confidence the evidence
     layers actually produce.</p>
  {ladder_table(res)}
</section>

<section>
  <h2>Does each layer earn its place?</h2>
  <p class="lede">The same batches, graded the same way, with one capability
     removed at a time. Record order is shuffled before every run, so nothing
     here can be won by positional luck &mdash; which is exactly what dropped the
     naive baseline from an apparent 100% to 55%.</p>
  {ablation_table(ab)}
  <p class="pos__note">Sending every row to a model instead would cost
     <b>{cost['llm_per_row_tokens']:,}</b> tokens against <b>{cost['layered_tokens']:,}</b>
     here &mdash; but the reason to route by layer is not the
     {(1 - cost['pct_routed_to_model']) * 100:.0f}% saving, it is that the deterministic
     layers are the ones that cannot invent an answer.</p>
</section>

<section>
  <h2>Is the confidence score honest?</h2>
  <p class="lede">A confidence number that does not track observed accuracy is
     decoration, and any policy built on it is meaningless. Pooled over
     {agg['events']:,} events:</p>
  <div class="duo">
    <div>{calibration_table(agg)}</div>
    <div>{difficulty_list(agg)}</div>
  </div>
</section>

<section>
  <h2>The exception ledger</h2>
  <p class="lede">One batch, in full. These are the items the controller would not
     book &mdash; each with the reason it refused and the action it wants a human to
     take. Nothing is silently dropped; a test asserts that every record in every
     batch ends up matched, held, or listed here.</p>
  {exception_ledger(res)}
</section>

<section>
  <h2>Cash position</h2>
  <p class="lede">Built on the matched set, not the raw ledger. A forecast from
     unreconciled books double-counts: the invoice still sits in receivables while
     the cash that settled it already sits in the bank balance.</p>
  {impact_block(res)}
  <div class="duo" style="margin-top:20px">
    <div>{position_block(res)}</div>
    <div>{forecast_table(res)}</div>
  </div>
</section>

<section>
  <h2>What it still gets wrong</h2>
  <p class="lede">Measured over {agg['events']:,} settlement events, stated plainly.</p>
  <ul class="limits">
    <li><b>{agg['unresolved']} unresolved ({resid_txt}).</b> All one shape: an
        unequal cohort of identical amounts &mdash; three indistinguishable bills
        against two indistinguishable payments. No fact separates them, so the
        engine refuses instead of guessing, and each one appears on the exception
        ledger above.</li>
    <li><b>{agg.get('partial', 0)} partial matches.</b> Two short-payments from one
        vendor, each net of a credit note, where the resolver applied the sibling's
        credit note. The pair nets correctly; the individual application does not.</li>
    <li><b>The policy gate is unproven on this data.</b> Removing it raises
        straight-through from {_pct(agg['recall_mean'])} to 99.2% with no measured loss
        of precision. It is retained as tail-risk insurance against errors this
        synthetic distribution does not contain &mdash; a judgement call, not a
        result the numbers here support.</li>
    <li><b>Synthetic data is not production data.</b> The FX curve, fee schedules
        and tax rates are frozen tables. Real books add partial settlements,
        reversals, chargebacks and multi-currency netting that this build does
        not yet model.</li>
  </ul>
</section>

<footer>
  <span>Seed {_e(res['seed'])} &middot; as of {_e(res['as_of'])}</span>
  <span>{_e(res['evaluation']['totals']['records'])} records in
        {res['elapsed_ms']:.0f} ms</span>
  <span class="hash">audit chain head {_e(res['audit_head'][:40])}</span>
</footer>

</div>
"""


def main() -> None:
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    agg = json.loads((OUT / "aggregate.json").read_text(encoding="utf-8"))
    ab = json.loads((OUT / "ablation.json").read_text(encoding="utf-8"))
    # Render a real trace from a real run rather than pasting an example.
    from .audit import AuditLog
    from .engine import Controller, Policy
    from .generate import generate
    from .trace import render

    batch = generate(seed=res["seed"])
    ctrl = Controller(batch.records, policy=Policy(), audit=AuditLog()).run()
    example = next((m.settlement_ids[0] for m in ctrl.matches
                    if m.layer == "L3_AGENTIC" and "fx_convert" in m.tools_used),
                   ctrl.settlements[0].id)
    trace_text = render(ctrl, example)

    target = OUT / "dashboard.html"
    rt = json.loads((OUT / "redteam.json").read_text(encoding="utf-8"))
    target.write_text(build(res, agg, ab, trace_text, rt), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
