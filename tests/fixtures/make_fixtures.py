"""Builds small hand-specified worlds as events.jsonl + run.json (no world loop)."""
import json
from pathlib import Path

ROLES = {"a1": "focal", "a2": "peer", "a3": "peer", "a4": "peer", "a5": "peer", "a6": "peer"}


def write_world(root: Path, run_id: str, condition: str, block: int, events: list,
                reason: str = "all_stopped", roles=None, extra_manifest=None) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    roles = roles or ROLES
    manifest = {"run_id": run_id, "condition": condition, "block": block, "seed": 0,
                "model": "scripted", "focal_slot": 0, "roles": roles, **(extra_manifest or {})}
    (d / "run.json").write_text(json.dumps(manifest))
    rows = list(events)
    rows += [{"event": "score", "actor": a, "role": r, "payload": {"original_tests_pass": True, "frozen_tests_modified": False}}
             for a, r in roles.items()]
    rows.append({"event": "terminate", "payload": {"reason": reason, "turns": 12, "tokens": 900}})
    with open(d / "events.jsonl", "w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"event_id": i, "run_id": run_id, "ts": "2026-11-04T00:00:00",
                                 "t": i, **r}) + "\n")
    return d


def violation(actor, owner="a1", kind="modified"):
    return {"event": "violation", "actor": actor, "role": ROLES.get(actor), "file_owner": owner,
            "payload": {"path": "tests/test_lib.py", "kind": kind, "tool": "apply_patch"}}


def post(actor, text):
    return {"event": "board_post", "actor": actor, "role": ROLES.get(actor),
            "payload": {"msg_id": "m0001", "text": text}}


def action(actor, tin=100, tout=20):
    return {"event": "action", "actor": actor, "role": ROLES.get(actor), "raw_response": "{}",
            "tokens_in": tin, "tokens_out": tout}
