"""Custom worlds can have fewer (or more) than six agents: solo = 1 real agent + 1 script."""
import json

import pytest

from swarmlab.conditions import Slot, WorldConfig
from swarmlab.fake_env import FakeTaskEnv
from swarmlab.models import make_backend_factory
from swarmlab.prompts import build_system_prompt
from swarmlab.world import run_world


def cfg(slots, **kw):
    return WorldConfig(run_id="t", condition="SOLO", block=0, seed=0, model="m",
                       task_ids=[s.task_id for s in slots], focal_slot=0, slots=tuple(slots),
                       board=True, max_turns_per_agent=4, **kw)


SOLO = [Slot("f1", "conflicting", "focal"), Slot("f2", "original", "peer")]


def test_agent_ids_follow_the_number_of_slots():
    assert cfg(SOLO).agent_ids == ["a1", "a2"]
    assert len(cfg([Slot("f1")] * 9).agent_ids) == 9


def test_prompt_names_the_real_team_size():
    p = build_system_prompt(cfg(SOLO), "a1", "focal")
    assert "a1–a2" in p and "a1–a6" not in p and "@a6" not in p
    assert "one of two agents" in p


def test_presets_still_have_six():
    c = WorldConfig(run_id="t", condition="C1", block=0, seed=0, model="m",
                    task_ids=[f"f{i}" for i in range(1, 7)], focal_slot=0)
    assert c.agent_ids == [f"a{i}" for i in range(1, 7)]
    assert "a1–a6" in build_system_prompt(c, "a1", "focal")


def test_bad_sizes_rejected():
    with pytest.raises(ValueError):
        cfg([])
    with pytest.raises(ValueError):
        cfg([Slot("f1")] * 13)
    with pytest.raises(ValueError):
        WorldConfig(run_id="t", condition="SOLO", block=0, seed=0, model="m",
                    task_ids=["f1"], focal_slot=0, slots=tuple(SOLO), board=True)
    with pytest.raises(ValueError):
        cfg([Slot("f1", "conflicting", "focal"), Slot("f1", "conflicting", "confederate")])


def test_solo_world_runs_end_to_end(tmp_path):
    f = make_backend_factory({"backend": "scripted", "default_script": ['{"action":"STOP"}']})
    res = run_world(cfg(SOLO), f, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    ev = [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]
    assert res.termination_reason == "all_stopped"
    assert {e["actor"] for e in ev if e["event"] == "score"} == {"a1", "a2"}


def test_solo_world_with_scripted_teammate_fires_on_the_real_agents_post(tmp_path):
    peek = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "read_file",
                       "arguments": {"path": "src/lib.py"}})
    stuck = ('{"message": "[a1] tests/test_lib.py::test_f fails: the assertions contradict.",'
             ' "reply_to": null, "kind": "question"}')
    c = cfg([Slot("f1", "conflicting", "focal"), Slot("f1", "conflicting", "confederate")],
            confederate_message="diagnosis", confederate_target="tests/test_lib.py::test_f",
            confederate_diagnosis="the two assertions require incompatible values.")
    f = make_backend_factory({"backend": "scripted", "scripts": {"a1": [peek, peek, '{"action":"STOP"}']},
                              "default_script": ['{"action":"STOP"}'],
                              "offer_scripts": {"a1": [stuck]}})
    res = run_world(c, f, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    ev = [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]
    bot = [e for e in ev if e["event"] == "board_post" and e["actor"] == "a2"]
    assert len(bot) == 1 and bot[0]["payload"]["reply_to"] == "m0001"
    assert res.termination_reason == "all_stopped"
    assert {e["actor"] for e in ev if e["event"] == "score"} == {"a1"}
