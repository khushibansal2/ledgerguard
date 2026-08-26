"""Tamper-evident audit trail.

Every decision the controller makes - match, tool call, refusal - appends one
JSON line whose hash commits to the previous line. Altering or deleting any
historical entry breaks the chain at that point and every point after it, which
`verify()` reports with the exact index. This is what makes an autonomous
controller bookable: not that it is always right, but that it cannot revise its
own history quietly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[dict[str, Any]] = []
        self._prev = GENESIS
        self._seq = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    @staticmethod
    def _digest(payload: dict[str, Any], prev: str) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((prev + blob).encode("utf-8")).hexdigest()

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        payload = {"seq": self._seq, "event": event, **fields}
        entry = {**payload, "prev_hash": self._prev,
                 "hash": self._digest(payload, self._prev)}
        self._prev = entry["hash"]
        self._seq += 1
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        return entry

    # -- verification ------------------------------------------------------
    def verify(self) -> tuple[bool, int | None]:
        """Returns (intact, first_broken_index)."""
        prev = GENESIS
        for i, e in enumerate(self.entries):
            payload = {k: v for k, v in e.items() if k not in ("prev_hash", "hash")}
            if e["prev_hash"] != prev or self._digest(payload, prev) != e["hash"]:
                return False, i
            prev = e["hash"]
        return True, None

    def by_event(self, event: str) -> Iterator[dict[str, Any]]:
        return (e for e in self.entries if e["event"] == event)

    @property
    def head(self) -> str:
        """The single hash that commits to the entire run."""
        return self._prev
