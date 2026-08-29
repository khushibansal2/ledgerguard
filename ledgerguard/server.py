"""HTTP service: the demo page plus a JSON API over the same controller.

Standard library only, like the rest of the project - `http.server` is enough
for an endpoint whose work is CPU-bound reconciliation rather than concurrency,
and adding a framework would mean the deployment carries dependencies the
library itself refuses to.

Two things are served:

    GET  /                     the browser demo (runs the engine client-side)
    GET  /health               liveness, for the platform's health check
    POST /api/close            {"seed": int}  -> the full close as JSON
    GET  /api/trace?id=...     one settlement's decision path as text

The demo page does not depend on the API: it runs the engine in the browser
under Pyodide, so the page keeps working while a free-tier instance is cold.
The API exists for the case the page cannot serve - scripting a close, wiring
the controller into something else, or simply proving the same numbers come out
of a server as out of the CLI.

    python -m ledgerguard.server            # http://localhost:8000
    PORT=10000 python -m ledgerguard.server # honours the platform's port
"""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .audit import AuditLog
from .engine import Controller, Policy
from .evaluate import evaluate
from .forecast import CashForecast
from .agent import LLMPlanner
from .generate import generate
from .models import fmt
from .trace import render

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web" / "index.html"

# A close is deterministic for a given seed, so the result can be cached and a
# repeated request costs nothing. Bounded because this is a demo service, not a
# store - the point is to avoid recomputing the same batch for every visitor.
_CACHE: dict[int, dict] = {}
_CACHE_MAX = 32
_CTRLS: dict[int, Controller] = {}


def close_batch(seed: int, use_llm: bool = False) -> dict:
    """Close one batch. `use_llm` routes L3 hypothesis ordering through the
    configured provider.

    Opt-in per request rather than on by default, for three reasons: the
    deterministic path is the honest default and should be what a visitor sees
    first; a model call adds latency to a demo that is otherwise instant; and
    leaving it switchable lets someone compare the two closes side by side and
    see for themselves that the ledger does not change - only the tool-call
    count does.
    """
    key = (seed, use_llm)
    if key in _CACHE:
        return _CACHE[key]

    planner = None
    if use_llm:
        planner = LLMPlanner()
        if not planner.available:
            planner = None

    batch = generate(seed=seed)
    ctrl = Controller(batch.records, policy=Policy(), audit=AuditLog(),
                      planner=planner)
    t0 = time.perf_counter()
    ctrl.run()
    ms = (time.perf_counter() - t0) * 1000
    ev = evaluate(batch, ctrl)
    fc = CashForecast(ctrl, "2026-04-05", 25_000_000)
    intact, _ = ctrl.audit.verify()

    result = {
        "seed": seed,
        # Reported so the model path can be audited from the outside: whether a
        # planner was actually used, how often it answered, and how often it
        # failed and fell back. A demo that claims to be agentic should be able
        # to show its working.
        "planner": {
            "used": planner is not None,
            "requested": use_llm,
            "kind": type(planner).__name__ if planner else "HeuristicPlanner",
            "model": getattr(planner, "model", "") or None,
            "calls": getattr(planner, "calls", 0),
            "fallbacks": getattr(planner, "failures", 0),
        },
        "totals": ev["totals"],
        "accuracy": ev["accuracy"],
        "value": ev["value"],
        "layers": ev["layer_mix"],
        "exception_quality": ev["exception_quality"],
        "position": fc.position(),
        "forecast": fc.project(),
        "impact": fc.reconciliation_impact(),
        "matches": [
            {"id": m.match_id, "layer": m.layer, "confidence": m.confidence,
             "settlement": m.settlement_ids, "ledger": m.ledger_ids,
             "exposure": ctrl._exposure(m), "rationale": m.rationale,
             "status": "held" if m in ctrl.escalated else "booked"}
            for m in ctrl.matches + ctrl.escalated],
        "exceptions": [e.to_dict() for e in ctrl.exceptions],
        "audit": {"entries": len(ctrl.audit.entries), "intact": intact,
                  "head": ctrl.audit.head},
        "elapsed_ms": round(ms, 1),
        "tool_calls": ctrl.stats["tool_calls"],
    }

    if len(_CACHE) >= _CACHE_MAX:
        oldest = next(iter(_CACHE))
        _CACHE.pop(oldest, None)
        _CTRLS.pop(oldest, None)
    _CACHE[key] = result
    _CTRLS[key] = ctrl
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "LedgerGuard"
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, indent=2).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, fmt_: str, *args) -> None:
        print(f"{self.address_string()} {fmt_ % args}", flush=True)

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:
        route = urlparse(self.path)
        path = route.path.rstrip("/") or "/"

        if path == "/health":
            planner = LLMPlanner()
            return self._json(200, {
                "status": "ok",
                "cached_batches": len(_CACHE),
                # Whether a model is configured, never what the key is.
                "llm_configured": planner.available,
                "llm_model": planner.model or None,
            })

        if path == "/":
            if not PAGE.exists():
                return self._json(503, {
                    "error": "demo page not built",
                    "fix": "run python -m ledgerguard.build_web"})
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/trace":
            q = parse_qs(route.query)
            sid = (q.get("id") or [""])[0]
            seed = int((q.get("seed") or ["20260827"])[0])
            if not sid:
                return self._json(400, {"error": "missing ?id=<settlement id>"})
            llm = (q.get("llm") or ["0"])[0] in ("1", "true", "yes")
            close_batch(seed, llm)     # ensures the controller exists
            text = render(_CTRLS[(seed, llm)], sid)
            return self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")

        if path == "/api/close":
            q = parse_qs(route.query)
            seed = int((q.get("seed") or ["20260827"])[0])
            llm = (q.get("llm") or ["0"])[0] in ("1", "true", "yes")
            return self._json(200, close_batch(seed, llm))

        self._json(404, {"error": "not found",
                         "routes": ["/", "/health", "/api/close", "/api/trace"]})

    def do_POST(self) -> None:
        if urlparse(self.path).path.rstrip("/") != "/api/close":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            seed = int(body.get("seed", 20260827))
            llm = bool(body.get("llm", False))
        except Exception as exc:
            return self._json(400, {"error": f"bad request: {exc}"})
        self._json(200, close_batch(seed, llm))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main() -> int:
    # Platforms assign the port; 0.0.0.0 because a container's loopback is not
    # reachable from outside it.
    port = int(os.environ.get("PORT", "8000"))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"LedgerGuard serving on http://0.0.0.0:{port}", flush=True)
    print(f"  demo   GET  /", flush=True)
    print(f"  api    POST /api/close   {{\"seed\": 20260827}}", flush=True)
    print(f"  trace  GET  /api/trace?id=BANK-5024", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
