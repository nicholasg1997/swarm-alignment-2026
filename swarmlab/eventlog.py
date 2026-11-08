"""JSONL event log + run manifest (PLAN.md section 7)."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

EVENT_TYPES = ("tool_call", "tool_result", "board_post", "board_delivered",
               "escalate", "action", "parse_error", "violation", "score",
               "terminate", "board_offer", "wake", "dropped_actions", "confederate_status", "protected_state", "model_request", "key_access", "key_use", "key_integrity", "assessment_result",
               "notebook_offer", "notebook_note", "notebook_read",
               "monitor_scan", "monitor_notice", "monitor_delivered", "monitor_undelivered")


class EventLog:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._next = 0
        self._fh = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")

    def emit(self, **fields: Any) -> int:
        if fields.get("event") not in EVENT_TYPES:
            raise ValueError(f"unknown event type {fields.get('event')!r}")
        event_id = self._next
        self._next += 1
        row = {"event_id": event_id, "run_id": self.run_id,
               "ts": datetime.now(timezone.utc).isoformat(), **fields}
        self._fh.write(json.dumps(row, default=str) + "\n")
        self._fh.flush()
        return event_id

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        (self.run_dir / "run.json").write_text(
            json.dumps({"run_id": self.run_id, **manifest}, indent=1, default=str))

    def close(self) -> None:
        self._fh.close()
