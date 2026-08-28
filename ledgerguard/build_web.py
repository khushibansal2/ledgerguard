"""Build the interactive browser demo.

The engine is pure standard library, which means it can run unmodified in the
browser under Pyodide - so the demo executes the *real* controller rather than a
JavaScript reimplementation or a recording of a previous run. Click a settlement
and the trace you get back was produced by the same code the tests exercise.

That also means no server, no database and no API key to demo: the page is a
static file, deployable to GitHub Pages, and it cannot drift from the library
because the module sources are inlined from disk at build time.

    python -m ledgerguard.build_web      ->  web/index.html
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path("web")

# Everything the browser needs. report/build_web are excluded: one writes files,
# the other is this script.
MODULES = ["__init__", "models", "similarity", "generate", "audit", "tools",
           "agent", "engine", "evaluate", "forecast", "trace"]

PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js"

DRIVER = '''
import json, time
from ledgerguard.generate import generate
from ledgerguard.engine import Controller, Policy
from ledgerguard.evaluate import evaluate
from ledgerguard.forecast import CashForecast
from ledgerguard.audit import AuditLog
from ledgerguard.trace import render
from ledgerguard.models import fmt

_state = {}

def close_batch(seed):
    batch = generate(seed=int(seed))
    ctrl = Controller(batch.records, policy=Policy(), audit=AuditLog())
    t0 = time.perf_counter()
    ctrl.run()
    ms = (time.perf_counter() - t0) * 1000
    ev = evaluate(batch, ctrl)
    fc = CashForecast(ctrl, "2026-04-05", 25_000_000)
    _state["ctrl"] = ctrl

    intact, _ = ctrl.audit.verify()
    rows = []
    for m in ctrl.matches:
        rows.append({"id": m.match_id, "layer": m.layer, "conf": m.confidence,
                     "settlement": m.settlement_ids, "ledger": m.ledger_ids,
                     "amount": fmt(ctrl._exposure(m)), "why": m.rationale,
                     "status": "booked"})
    for m in ctrl.escalated:
        rows.append({"id": m.match_id, "layer": m.layer, "conf": m.confidence,
                     "settlement": m.settlement_ids, "ledger": m.ledger_ids,
                     "amount": fmt(ctrl._exposure(m)), "why": m.rationale,
                     "status": "held"})

    return json.dumps({
        "totals": ev["totals"], "accuracy": ev["accuracy"], "value": ev["value"],
        "layers": ev["layer_mix"], "by_case": ev["by_case"],
        "calibration": ev["calibration"],
        "exception_quality": ev["exception_quality"],
        "position": fc.position(), "forecast": fc.project(),
        "impact": fc.reconciliation_impact(),
        "ladder": [{"threshold": t, "conf": c} for t, c in ctrl.policy.ladder],
        "matches": rows,
        "exceptions": [e.to_dict() for e in ctrl.exceptions],
        "settlements": [{"id": r.id, "date": r.txn_date,
                         "amount": fmt(r.amount, r.currency),
                         "desc": r.description, "party": r.counterparty}
                        for r in ctrl.settlements],
        "audit": {"entries": len(ctrl.audit.entries), "intact": intact,
                  "head": ctrl.audit.head},
        "ms": ms, "tool_calls": ctrl.stats["tool_calls"],
    })

def trace_for(sid):
    ctrl = _state.get("ctrl")
    if ctrl is None:
        return "run a close first"
    return render(ctrl, sid)
'''


def build() -> Path:
    sources = {}
    for name in MODULES:
        p = ROOT / f"{name}.py"
        sources[f"ledgerguard/{name}.py"] = p.read_text(encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__SOURCES__", json.dumps(sources))
    html = html.replace("__DRIVER__", json.dumps(DRIVER))
    html = html.replace("__PYODIDE__", PYODIDE)
    target = OUT / "index.html"
    target.write_text(html, encoding="utf-8")
    return target


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LedgerGuard - autonomous finance controller</title>
<meta name="description" content="Reconciles bank, processor and ERP records into a booked ledger and a forward cash position, and reports every item it refused to book.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
*,*::before,*::after{box-sizing:border-box}
:root{
  --paper:#eef1f4; --card:#fff; --ink:#0e1a22; --ink-2:#41525e; --ink-3:#6d7f8b;
  --line:#d3dbe2; --line-2:#e4eaef;
  --accent:#0d6a6a; --accent-soft:#e0efee;
  --good:#2c7a51; --warn:#9a6a15; --bad:#a83a34;
  --good-soft:#e2f0e8; --warn-soft:#f7ecd8; --bad-soft:#f8e5e3;
  --shadow:0 1px 2px rgba(14,26,34,.06),0 8px 24px -16px rgba(14,26,34,.28);
  --display:"Archivo","Helvetica Neue",Arial,sans-serif;
  --body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0b1318; --card:#121e25; --ink:#e6edf1; --ink-2:#a8b8c2; --ink-3:#7d8f9a;
  --line:#243440; --line-2:#1b2932; --accent:#4fb3ac; --accent-soft:#12302f;
  --good:#5fbf8b; --warn:#d6a94f; --bad:#e0776f;
  --good-soft:#12291f; --warn-soft:#2c2415; --bad-soft:#2e1a19;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.8);
}}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:clamp(20px,4vw,48px) clamp(14px,4vw,28px) 80px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}

.mast{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:24px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 8px}
h1{font-family:var(--display);font-weight:700;letter-spacing:-.02em;
  font-size:clamp(28px,5vw,46px);line-height:1.03;margin:0;text-wrap:balance}
.sub{margin:10px 0 0;max-width:66ch;color:var(--ink-2);font-size:16.5px}

.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:20px}
button{font-family:var(--body);font-size:14px;font-weight:600;cursor:pointer;
  border-radius:3px;border:1px solid var(--accent);background:var(--accent);
  color:#fff;padding:10px 20px;transition:opacity .15s}
button:hover:not(:disabled){opacity:.88}
button:disabled{opacity:.45;cursor:not-allowed}
button.ghost{background:transparent;color:var(--accent)}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=number]{font-family:var(--mono);font-size:14px;padding:9px 12px;width:130px;
  border:1px solid var(--line);border-radius:3px;background:var(--card);color:var(--ink)}
label{font-size:12px;font-family:var(--mono);letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3)}

#status{margin-top:14px;font-family:var(--mono);font-size:12.5px;color:var(--ink-3);
  display:flex;align-items:center;gap:9px;min-height:20px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--warn);flex:none}
.dot.ready{background:var(--good)} .dot.err{background:var(--bad)}
.dot.busy{animation:pulse 1.1s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
@media (prefers-reduced-motion:reduce){.dot.busy{animation:none}}

section{margin-top:38px}
h2{font-family:var(--display);font-weight:600;font-size:21px;letter-spacing:-.01em;
  margin:0 0 6px;display:flex;align-items:baseline;gap:12px}
h2::after{content:"";flex:1;height:1px;background:var(--line)}
.lede{margin:0 0 18px;color:var(--ink-2);max-width:72ch;font-size:14.5px}
.hidden{display:none}

.kpis{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(195px,1fr))}
.kpi{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--accent);
  border-radius:3px;padding:16px;box-shadow:var(--shadow)}
.kpi.good{border-top-color:var(--good)}
.kpi .k{margin:0;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--ink-3);font-family:var(--mono)}
.kpi .v{margin:7px 0 3px;font-family:var(--display);font-weight:700;font-size:34px;
  line-height:1;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kpi .s{margin:0;font-size:12.5px;color:var(--ink-2);line-height:1.4}

.flow{display:flex;flex-direction:column;gap:2px}
.frow{display:flex;gap:13px;background:var(--card);border:1px solid var(--line);padding:13px 15px}
.frow+.frow{border-top:none}
.frow:first-child{border-radius:3px 3px 0 0}.frow:last-child{border-radius:0 0 3px 3px}
.tag{font-family:var(--mono);font-size:10px;letter-spacing:.09em;font-weight:600;
  padding:4px 7px;border-radius:2px;background:var(--accent-soft);color:var(--accent);
  min-width:44px;text-align:center;height:fit-content;margin-top:2px}
.tag.hold{background:var(--warn-soft);color:var(--warn)}
.tag.exc{background:var(--bad-soft);color:var(--bad)}
.fbody{flex:1;min-width:0}
.fhead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.fname{font-weight:600;font-size:14.5px}
.fwho{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-3);border:1px solid var(--line);padding:1px 6px;border-radius:2px}
.fn{margin-left:auto;font-family:var(--mono);font-weight:600}
.fwhy{margin:7px 0 0;font-size:12.5px;color:var(--ink-2)}
.track{height:6px;background:var(--line-2);border-radius:2px;overflow:hidden;margin-top:7px}
.fill{display:block;height:100%;background:var(--accent);border-radius:2px;
  transition:width .5s cubic-bezier(.2,.7,.3,1)}
.fill.hold{background:var(--warn)}.fill.exc{background:var(--bad)}
.fill.good{background:var(--good)}.fill.bad{background:var(--bad)}

.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);
  border-radius:3px;box-shadow:var(--shadow)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:10px 13px;text-align:left;border-bottom:1px solid var(--line-2)}
th{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);font-weight:500;background:var(--paper);position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
tr.clickable{cursor:pointer}
tr.clickable:hover td{background:var(--accent-soft)}
tr.sel td{background:var(--accent-soft);font-weight:600}
.pill{font-family:var(--mono);font-size:9.5px;letter-spacing:.07em;text-transform:uppercase;
  padding:2px 6px;border-radius:2px;background:var(--good-soft);color:var(--good)}
.pill.held{background:var(--warn-soft);color:var(--warn)}

.duo{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
pre.trace{background:var(--card);border:1px solid var(--line);border-radius:3px;
  padding:15px 17px;overflow-x:auto;font-family:var(--mono);font-size:12px;
  line-height:1.6;margin:0;box-shadow:var(--shadow);color:var(--ink);min-height:120px}

.sev{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  margin:20px 0 9px;display:flex;align-items:center;gap:9px;color:var(--ink-3)}
.sev .n{background:var(--line-2);border-radius:10px;padding:1px 8px}
.sev.high,.sev.critical{color:var(--bad)}.sev.medium{color:var(--warn)}
.exc{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--ink-3);
  border-radius:3px;padding:13px 15px;margin-bottom:9px}
.exc.critical,.exc.high{border-left-color:var(--bad)}
.exc.medium{border-left-color:var(--warn)}
.etop{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
.ecat{font-family:var(--display);font-weight:600;font-size:14.5px}
.eids{margin:3px 0 7px;font-size:11px;color:var(--ink-3);font-family:var(--mono)}
.ereason{margin:0;font-size:13.5px;color:var(--ink-2)}
.eact,.emiss{margin:8px 0 0;font-size:13px;display:flex;gap:9px;align-items:baseline}
.eact span,.emiss span{font-family:var(--mono);font-size:9.5px;letter-spacing:.09em;
  text-transform:uppercase;padding:1px 5px;border-radius:2px;flex:none;
  border:1px solid var(--accent);color:var(--accent)}
.emiss span{border-color:var(--line);color:var(--ink-3)}
.impact{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:3px;padding:15px 17px;box-shadow:var(--shadow)}
.impact p{margin:0 0 9px}.impact p:last-child{margin:0}
.neg{color:var(--bad)}.pos{color:var(--good)}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:11.5px;color:var(--ink-3);
  display:flex;flex-wrap:wrap;gap:8px 22px}
a{color:var(--accent)}
.hash{word-break:break-all}
</style>
</head>
<body>
<div class="wrap">

<header class="mast">
  <p class="eyebrow">Autonomous finance controller</p>
  <h1>LedgerGuard</h1>
  <p class="sub">Reconciles bank, processor and ERP records into a booked ledger
     and a forward cash position &mdash; and reports every item it refused to
     book, with the reason and the evidence needed to close it.</p>
  <div class="controls">
    <label for="seed">Batch seed</label>
    <input type="number" id="seed" value="20260827">
    <button id="run" disabled>Close the batch</button>
    <button id="shuffle" class="ghost" disabled>Random batch</button>
  </div>
  <div id="status"><span class="dot busy"></span><span id="statustext">loading the Python engine&hellip;</span></div>
</header>

<div id="results" class="hidden">

  <section>
    <h2>Scorecard</h2>
    <p class="lede">Graded against a truth key the engine never sees. Row counts
       hide what a controller is accountable for, so exposure is reported
       alongside frequency.</p>
    <div class="kpis" id="kpis"></div>
  </section>

  <section>
    <h2>How the work was routed</h2>
    <p class="lede">Each layer costs more and is trusted less than the one above
       it. Nothing that earlier layers can settle is ever passed to a model.</p>
    <div class="flow" id="flow"></div>
  </section>

  <section>
    <h2>Every decision, and why</h2>
    <p class="lede">Select any settlement to replay its full decision path from
       the hash-chained audit log &mdash; layers tried, hypotheses proposed, tools
       called, the arithmetic that closed, and the policy gate that allowed it.</p>
    <div class="duo">
      <div class="scroll" style="max-height:460px">
        <table>
          <thead><tr><th>Match</th><th>Layer</th><th class="num">Value</th>
            <th class="num">Conf.</th><th></th></tr></thead>
          <tbody id="matches"></tbody>
        </table>
      </div>
      <pre class="trace" id="trace">Select a match to replay its decision.</pre>
    </div>
  </section>

  <section>
    <h2>The exception queue</h2>
    <p class="lede">What happened, why the engine could not resolve it, and what
       evidence would. Nothing is silently dropped &mdash; a property test asserts
       every record ends up matched, held, or listed here.</p>
    <div id="exceptions"></div>
  </section>

  <section>
    <h2>Cash position</h2>
    <p class="lede">Built on the matched set. A forecast from an unreconciled
       ledger double-counts: the invoice sits in receivables while the cash that
       settled it already sits in the bank.</p>
    <div id="impact"></div>
    <div class="duo" style="margin-top:16px">
      <div class="scroll"><table id="position"></table></div>
      <div class="scroll"><table id="forecast"></table></div>
    </div>
  </section>

  <footer id="foot"></footer>
</div>
</div>

<script src="__PYODIDE__"></script>
<script>
const SOURCES = __SOURCES__;
const DRIVER = __DRIVER__;
let py = null;

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = (x, d=1) => (x*100).toFixed(d) + '%';

function status(text, kind) {
  $('statustext').textContent = text;
  const d = document.querySelector('.dot');
  d.className = 'dot' + (kind ? ' ' + kind : ' busy');
}

async function boot() {
  try {
    py = await loadPyodide();
    // Write the library into the virtual filesystem exactly as it exists on
    // disk, so the browser runs the same code the test suite does.
    py.FS.mkdir('/lg'); py.FS.mkdir('/lg/ledgerguard');
    for (const [path, src] of Object.entries(SOURCES)) {
      py.FS.writeFile('/lg/' + path, src);
    }
    py.runPython(`import sys; sys.path.insert(0, '/lg')`);
    py.runPython(DRIVER);
    status('engine ready - pure standard library, running locally in your browser', 'ready');
    $('run').disabled = false; $('shuffle').disabled = false;
  } catch (e) {
    status('could not start the engine: ' + e.message, 'err');
  }
}

function runClose() {
  $('run').disabled = true; $('shuffle').disabled = true;
  status('closing the batch…', 'busy');
  // Yield to the browser so the status paints before the synchronous run.
  setTimeout(() => {
    try {
      const seed = parseInt($('seed').value || '20260827', 10);
      const data = JSON.parse(py.runPython(`close_batch(${seed})`));
      render(data);
      status(`closed ${data.totals.records} records in ${data.ms.toFixed(0)} ms `
             + `- ${data.tool_calls} tool calls, audit chain `
             + (data.audit.intact ? 'verified' : 'BROKEN'), 'ready');
    } catch (e) {
      status('run failed: ' + e.message, 'err');
    }
    $('run').disabled = false; $('shuffle').disabled = false;
  }, 30);
}

function render(d) {
  $('results').classList.remove('hidden');
  const a = d.accuracy, v = d.value, t = d.totals;

  $('kpis').innerHTML = [
    ['Mis-booked value', v.mis_booked === 0 ? '$0.00' : fmtCents(v.mis_booked),
     'good', `of ${fmtCents(v.total_settled)} settled in this batch`],
    ['Precision', pct(a.precision, 2), 'good',
     `${a.false_positives} wrong of ${t.booked} booked`],
    ['Straight-through', pct(a.straight_through_rate), '',
     `${t.held_for_review} held for a signature`],
    ['Exceptions', String(t.exceptions), '',
     `${pct(d.exception_quality.escalation_recall,0)} of unmatchables refused`],
  ].map(([k, val, cls, s]) =>
    `<div class="kpi ${cls}"><p class="k">${esc(k)}</p>
     <p class="v">${esc(val)}</p><p class="s">${esc(s)}</p></div>`).join('');

  const L = d.layers;
  const rows = [
    ['L0','Reversal pairing',L.L0_REVERSAL||0,'no model','Returns and chargebacks that settle no document.'],
    ['L1','Exact reference',L.L1_DETERMINISTIC||0,'no model','Reference and amount agree to the cent. Proof, not inference.'],
    ['L2','Entity + timing',L.L2_SIMILARITY||0,'no model','Amount ties out; identity and settlement lag are inferred.'],
    ['L3','Tool-using resolver',L.L3_AGENTIC||0,'model may plan','Amount does not tie out - FX, fee, tax or a split must close it.'],
    ['hold','Policy gate',t.held_for_review,'human','Resolved, but material enough to need a signature.'],
    ['exc','Exception queue',t.exceptions,'human','Refused, with the reason and next action attached.'],
  ];
  const total = Math.max(rows.reduce((s, r) => s + r[2], 0), 1);
  $('flow').innerHTML = rows.map(([tag,name,n,who,why]) => {
    const k = (tag==='hold'||tag==='exc') ? tag : '';
    return `<div class="frow"><span class="tag ${k}">${esc(tag.toUpperCase())}</span>
      <div class="fbody"><div class="fhead"><span class="fname">${esc(name)}</span>
      <span class="fwho">${esc(who)}</span><span class="fn">${n}</span></div>
      <div class="track"><span class="fill ${k}" style="width:${(n/total*100).toFixed(1)}%"></span></div>
      <p class="fwhy">${esc(why)}</p></div></div>`;
  }).join('');

  $('matches').innerHTML = d.matches.map(m =>
    `<tr class="clickable" data-sid="${esc(m.settlement[0])}">
      <td class="mono">${esc(m.id)}</td>
      <td>${esc(m.layer.replace('_',' ').toLowerCase())}</td>
      <td class="num">${esc(m.amount)}</td>
      <td class="num">${m.conf.toFixed(2)}</td>
      <td><span class="pill ${m.status==='held'?'held':''}">${esc(m.status)}</span></td>
    </tr>`).join('');
  document.querySelectorAll('#matches tr').forEach(tr =>
    tr.addEventListener('click', () => {
      document.querySelectorAll('#matches tr').forEach(x => x.classList.remove('sel'));
      tr.classList.add('sel');
      $('trace').textContent = py.runPython(`trace_for(${JSON.stringify(tr.dataset.sid)})`);
    }));

  const order = ['CRITICAL','HIGH','MEDIUM','LOW'];
  const bySev = {};
  d.exceptions.forEach(e => (bySev[e.severity] = bySev[e.severity] || []).push(e));
  $('exceptions').innerHTML = order.filter(s => bySev[s]).map(s => {
    const items = bySev[s].map(e => `
      <article class="exc ${s.toLowerCase()}">
        <div class="etop"><span class="ecat">${esc(e.category.replace(/_/g,' '))}</span>
          <span class="mono" style="font-weight:600">${esc(fmtCents(Math.abs(e.amount)))}</span></div>
        <p class="eids">${esc(e.record_ids.join(', '))}</p>
        <p class="ereason">${esc(e.reason)}</p>
        <p class="eact"><span>Next</span>${esc(e.suggested_action)}</p>
        ${e.missing_evidence ? `<p class="emiss"><span>Missing</span>${esc(e.missing_evidence)}</p>` : ''}
      </article>`).join('');
    return `<h3 class="sev ${s.toLowerCase()}">${s}<span class="n">${bySev[s].length}</span></h3>${items}`;
  }).join('');

  const p = d.position;
  $('position').innerHTML = `<thead><tr><th>Cash position</th><th class="num">Amount</th></tr></thead><tbody>` +
    [['Opening balance',p.opening_balance],['Cleared movements',p.cleared_movements],
     ['Closing bank balance',p.closing_bank_balance],['Open receivables',p.open_receivables],
     ['Open payables',p.open_payables],['Net working capital',p.net_working_capital]]
    .map(([k,val]) => `<tr><td>${esc(k)}</td><td class="num">${esc(fmtCents(val))}</td></tr>`).join('') +
    `</tbody>`;

  $('forecast').innerHTML = `<thead><tr><th>Week</th><th class="num">In</th>
    <th class="num">Out</th><th class="num">Expected</th><th class="num">Worst</th></tr></thead><tbody>` +
    d.forecast.weeks.map(w => `<tr><td class="mono">${esc(w.from)}</td>
      <td class="num pos">${esc(fmtCents(w.expected_inflow))}</td>
      <td class="num neg">${esc(fmtCents(w.committed_outflow))}</td>
      <td class="num">${esc(fmtCents(w.expected_balance))}</td>
      <td class="num" style="color:var(--ink-3)">${esc(fmtCents(w.low_balance))}</td></tr>`).join('') +
    `</tbody>`;

  const im = d.impact;
  $('impact').innerHTML = `<div class="impact">
    <p><b>${im.documents_cleared}</b> documents cleared against settlements.</p>
    <p>Forecasting on the <i>unreconciled</i> ledger would have expected
    <span class="mono">${esc(fmtCents(im.naive_expected_inflow))}</span> of receivables.
    After the close the true figure is
    <span class="mono">${esc(fmtCents(im.receivables_after_close))}</span> &mdash; an
    overstatement of <span class="mono neg">${esc(fmtCents(im.receivables_already_settled))}</span>,
    entirely from counting invoices that were already paid.</p></div>`;

  $('foot').innerHTML =
    `<span>${t.records} records &middot; ${t.settlement_events} settlement events</span>
     <span>${d.audit.entries} hash-chained audit entries &middot;
     ${d.audit.intact ? 'chain verified' : 'CHAIN BROKEN'}</span>
     <span class="hash">head ${esc(d.audit.head.slice(0,40))}</span>
     <span><a href="https://github.com/khushibansal2/ledgerguard">source</a></span>`;
}

function fmtCents(c) {
  const neg = c < 0; c = Math.abs(c);
  const s = '$' + Math.floor(c/100).toLocaleString('en-US') + '.' + String(c%100).padStart(2,'0');
  return (neg ? '-' : '') + s;
}

$('run').addEventListener('click', runClose);
$('shuffle').addEventListener('click', () => {
  $('seed').value = 20260827 + Math.floor(Math.random() * 400);
  runClose();
});
boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"wrote {build()}")
