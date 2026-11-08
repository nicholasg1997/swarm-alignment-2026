import json

import pytest

from swarmlab.eventlog import EventLog


def test_emit_monotonic_and_schema(tmp_path):
    log = EventLog(tmp_path, run_id="r1")
    assert log.emit(event="action", actor="a1") == 0
    assert log.emit(event="terminate") == 1
    log.close()
    rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["event_id"] for r in rows] == [0, 1]
    for r in rows:
        assert {"event_id", "run_id", "ts", "event"} <= set(r) and r["run_id"] == "r1"


def test_emit_rejects_unknown_event(tmp_path):
    log = EventLog(tmp_path, run_id="r1")
    with pytest.raises(ValueError):
        log.emit(event="banana")


def test_write_manifest(tmp_path):
    log = EventLog(tmp_path, run_id="r1")
    log.write_manifest({"condition": "C1"})
    assert json.loads((tmp_path / "run.json").read_text())["condition"] == "C1"
