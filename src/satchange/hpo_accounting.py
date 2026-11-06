"""Append-only GPU allocation ledger for the HPO v3 controller (standard library only).

Monitoring category (D-022): this module measures and records; it never decides what the
scheduler does. ``hpo_core`` reads the numbers it returns and makes all enforcement decisions.

GPU-hours = sum over allocations of (end - start) * number_of_devices / 3600, measured by the
controller from job dispatch to observed completion (D-003). Queue time is not an allocation.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

Row = dict[str, Any]
LEDGER_NAME = "gpu_ledger.jsonl"
SCHEMA_VERSION = 1


def now() -> float:
    """Wall-clock seconds; a module-level function so tests can inject a clock."""
    return time.time()


class GpuLedger:
    """Single-writer ledger; the experiment lock guarantees one controller per search."""

    def __init__(self, root: Path, *, heartbeat_seconds: float = 60.0) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive.")
        self.path = Path(root) / LEDGER_NAME
        self.heartbeat_seconds = heartbeat_seconds
        self.invalid_lines = 0
        self.allocations: dict[str, Row] = {}
        self.budgets: list[Row] = []
        self.overshoots: list[Row] = []
        self._replay()

    def _append(self, event: Row) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False)
        torn = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                torn = existing.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            # Never append onto a torn final line: the new event must stay parseable.
            handle.write(("\n" if torn else "") + line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(event)

    def _apply(self, event: Row) -> None:
        kind = event.get("event")
        if kind == "start":
            if event["allocation_id"] not in self.allocations:
                self.allocations[event["allocation_id"]] = {
                    **event, "last_seen": event["time"], "end": None,
                }
        elif kind in ("heartbeat", "end"):
            for allocation_id in event.get("allocation_ids", [event.get("allocation_id")]):
                allocation = self.allocations.get(allocation_id)
                if allocation is None or allocation["end"] is not None:
                    continue  # Unknown or already closed: counted once, never twice.
                allocation["last_seen"] = max(allocation["last_seen"], event["time"])
                if kind == "end":
                    allocation["end"] = event
        elif kind == "budget":
            self.budgets.append(event)
        elif kind == "overshoot":
            self.overshoots.append(event)

    def _replay(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or "event" not in event:
                    raise ValueError
            except ValueError:
                self.invalid_lines += 1  # A torn final write loses at most one event.
                continue
            self._apply(event)

    # Recording -----------------------------------------------------------------------------

    def start(self, *, devices: Sequence[str], **fields: Any) -> str:
        if not devices:
            raise ValueError("An allocation needs at least one device.")
        allocation_id = uuid4().hex
        self._append({
            "event": "start", "schema_version": SCHEMA_VERSION, "allocation_id": allocation_id,
            "time": now(), "devices": list(devices), **fields,
        })
        return allocation_id

    def heartbeat(self, allocation_ids: Sequence[str]) -> None:
        if allocation_ids:
            self._append({"event": "heartbeat", "allocation_ids": list(allocation_ids),
                          "time": now()})

    def end(self, allocation_id: str, status: str, **fields: Any) -> Row:
        allocation = self.allocations.get(allocation_id)
        if allocation is None:
            raise KeyError(f"Unknown GPU allocation {allocation_id}")
        if allocation["end"] is not None:
            return cast(Row, allocation["end"])
        moment = now()
        self._append({
            "event": "end", "allocation_id": allocation_id, "time": moment, "status": status,
            "seconds": max(0.0, moment - allocation["time"]), **fields,
        })
        return cast(Row, allocation["end"])

    def close_orphans(self) -> list[str]:
        """Close allocations left open by a dead controller at their last heartbeat."""
        closed = []
        for allocation_id, allocation in list(self.allocations.items()):
            if allocation["end"] is not None:
                continue
            seconds = max(0.0, allocation["last_seen"] - allocation["time"])
            self._append({
                "event": "end", "allocation_id": allocation_id, "time": allocation["last_seen"],
                "status": "interrupted", "seconds": seconds,
                "uncertain_seconds": self.heartbeat_seconds,
                "note": "controller exited without closing; charged up to the last heartbeat",
            })
            closed.append(allocation_id)
        return closed

    def set_budget(self, gpu_hours: float, *, reason: str) -> None:
        self._append({"event": "budget", "gpu_hours": float(gpu_hours), "reason": reason,
                      "time": now()})

    def record_overshoot(self, gpu_hours: float, *, cause: str) -> None:
        self._append({"event": "overshoot", "gpu_hours": float(gpu_hours), "cause": cause,
                      "time": now()})

    # Reading -------------------------------------------------------------------------------

    @property
    def budget(self) -> float | None:
        return float(self.budgets[-1]["gpu_hours"]) if self.budgets else None

    @staticmethod
    def hours(allocation: Row, until: float | None = None) -> float:
        end = allocation["end"]
        seconds = end["seconds"] if end is not None else max(
            0.0, (until if until is not None else now()) - allocation["time"]
        )
        return float(seconds) * len(allocation["devices"]) / 3600

    def spent(self, *, include_active: bool = True, ids: Sequence[str] | None = None) -> float:
        moment = now()
        selected = self.allocations.values() if ids is None else (
            self.allocations[i] for i in ids if i in self.allocations
        )
        return sum(
            self.hours(a, moment) for a in selected if include_active or a["end"] is not None
        )

    def seconds_per_epoch(self, *, model_name: str | None = None) -> float | None:
        """Average allocated seconds per trained epoch over completed allocations."""
        seconds = epochs = 0.0
        for allocation in self.allocations.values():
            end = allocation["end"]
            if end is None or not end.get("epochs_trained"):
                continue
            if model_name is not None and allocation.get("model_name") != model_name:
                continue
            seconds += end["seconds"]
            epochs += end["epochs_trained"]
        return seconds / epochs if epochs else None

    def summary(self) -> Row:
        budget = self.budget
        spent = self.spent()
        closed = self.spent(include_active=False)
        return {
            "definition": "sum(allocation seconds x devices)/3600, dispatch to completion",
            "ledger": str(self.path),
            "budget_gpu_hours": budget,
            "spent_gpu_hours": spent,
            "closed_gpu_hours": closed,
            "remaining_gpu_hours": None if budget is None else budget - spent,
            "allocations": len(self.allocations),
            "open_allocations": sum(a["end"] is None for a in self.allocations.values()),
            "uncertain_seconds": sum(
                a["end"].get("uncertain_seconds", 0.0)
                for a in self.allocations.values() if a["end"] is not None
            ),
            "overshoot_gpu_hours": sum(event["gpu_hours"] for event in self.overshoots),
            "invalid_ledger_lines": self.invalid_lines,
        }
