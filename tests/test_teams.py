"""Slots, roles, teams: presets, prompts, shared sandboxes, attribution."""
import json

import pytest

from swarmlab.conditions import CONDITIONS, Slot, WorldConfig, layout_for, variant_for
from swarmlab.fake_env import FakeTaskEnv
from swarmlab.models import make_backend_factory
from swarmlab.prompts import DEVIL_TEXT, build_system_prompt, role_for, team_line
from swarmlab.world import run_world

TASKS = [f"f{i}" for i in range(1, 7)]
STOP = '{"action":"STOP"}'


def cfg(cond, focal=0, **kw):
    return WorldConfig(run_id=f"t-{cond}", condition=cond, block=0, seed=0, model="m",
                       task_ids=TASKS, focal_slot=focal, board_mode="passive", **kw)


def test_pair_preset_shares_focal_task():
    lay = layout_for(cfg("PR", focal=2))
    assert (lay[2].task_id, lay[3].task_id) == ("f3", "f3")
    assert lay[2].team == lay[3].team == "pair"
    assert lay[2].role == lay[3].role == "focal" and lay[3].variant == "conflicting"
    assert [s.role for i, s in enumerate(lay) if i not in (2, 3)] == ["peer"] * 4


def test_devil_preset_two_stuck_agents_and_a_devil():
    lay = layout_for(cfg("DV", focal=4))
    assert [s.role for s in lay] == ["devil", "peer", "peer", "peer", "focal", "focal"]
    assert lay[4].variant == lay[5].variant == "conflicting"
    assert lay[4].task_id != lay[5].task_id and lay[0].variant == "original"


def test_devil_control_has_no_devil():
    assert "devil" not in [s.role for s in layout_for(cfg("D0"))]
    assert [s.role for s in layout_for(cfg("D0"))].count("focal") == 2


def test_legacy_conditions_unchanged():
    assert [variant_for(cfg("C1"), i) for i in range(6)] == ["conflicting"] + ["original"] * 5
    assert [s.role for s in layout_for(cfg("C2", angel_slot=1))][:2] == ["focal", "angel"]


def test_custom_slots_and_board_override():
    slots = [Slot("f1", "conflicting", "focal", "t"), Slot("f1", "conflicting", "focal", "t"),
             Slot("f1", "conflicting", "focal", "t"), Slot("f4", "original", "devil"),
             Slot("f5", "original"), Slot("f6", "original")]
    c = WorldConfig(run_id="x", condition="trio", block=0, seed=0, model="m", task_ids=TASKS,
                    focal_slot=0, slots=tuple(slots), board=True)
    assert layout_for(c) == slots and c.spec.board_enabled
    with pytest.raises(ValueError):
        WorldConfig(run_id="x", condition="nope", block=0, seed=0, model="m", task_ids=TASKS,
                    focal_slot=0)


def test_devil_and_team_prompts():
    c = cfg("PR")
    assert role_for(c, 0) == "focal"
    p = build_system_prompt(c, "a1", "focal")
    assert team_line(c, "a1") in p and "a2" in team_line(c, "a1")
    assert team_line(c, "a3") == ""
    d = cfg("DV")
    devil_id = [f"a{i + 1}" for i, s in enumerate(layout_for(d)) if s.role == "devil"][0]
    assert DEVIL_TEXT in build_system_prompt(d, devil_id, "devil")
    assert DEVIL_TEXT not in build_system_prompt(d, "a1", "focal")


def _run(tmp_path, c, scripts):
    f = make_backend_factory({"backend": "scripted", "scripts": scripts, "default_script": [STOP]})
    made = []

    def env_factory(task_id, variant):
        made.append(task_id)
        return FakeTaskEnv(task_id, variant)
    res = run_world(c, f, env_factory, tmp_path)
    ev = [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]
    return res, ev, made


def test_pair_shares_one_sandbox_and_violation_counted_once(tmp_path):
    edit = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "apply_patch", "arguments": {
        "patch": json.dumps({"path": "tests/test_lib.py", "content": "def test_f():\n    pass\n"})}})
    peek = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "read_file",
                       "arguments": {"path": "tests/test_lib.py"}})
    res, ev, made = _run(tmp_path, cfg("PR"), {"a2": [edit, STOP], "a1": [peek, peek, STOP]})
    assert made.count("f1") == 1 and len(made) == 5
    v = [e for e in ev if e["event"] == "violation"]
    assert len(v) == 1 and v[0]["actor"] == "a2" and v[0]["file_owner"] == "a1"
    seen = [e for e in ev if e["event"] == "tool_result" and e["actor"] == "a1"]
    assert "pass" in seen[-1]["payload"]["observation"]
    m = json.loads((res.run_dir / "run.json").read_text())
    assert m["teams"]["a1"] == m["teams"]["a2"] == "pair" and m["env_owner"]["a2"] == "a1"
