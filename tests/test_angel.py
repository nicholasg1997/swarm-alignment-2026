"""The angel (verifier) defense: its duties, and it is not a subject in the outcome rates."""
from fixtures.make_fixtures import violation, write_world

from swarmlab.analysis import load_world, world_metrics
from swarmlab.conditions import Slot, WorldConfig
from swarmlab.prompts import ANGEL_TEXT, build_system_prompt

ROLES = {"a1": "focal", "a2": "focal", "a3": "focal", "a4": "angel",
         "a5": "confederate", "a6": "focal"}
VARIANTS = {a: "conflicting" for a in ROLES}


def test_angel_checks_each_kind_of_claim_and_still_allows_real_fixes():
    t = ANGEL_TEXT.lower()
    for duty in ("check", "workspace", "rerun", "approval", "evidence", "escalate_to_human"):
        assert duty in t
    assert "not to block" in t and "forbidden" not in t   # works under the implicit rule too


def test_angel_prompt_only_for_the_angel():
    slots = tuple(Slot("f1", "conflicting", r) for r in ("focal", "angel"))
    c = WorldConfig(run_id="t", condition="X", block=0, seed=0, model="m", task_ids=["f1", "f1"],
                    focal_slot=0, slots=slots, board=True)
    assert ANGEL_TEXT in build_system_prompt(c, "a2", "angel")
    assert ANGEL_TEXT not in build_system_prompt(c, "a1", "focal")


def test_angel_is_not_a_subject_and_its_reports_are_counted_apart(tmp_path):
    esc = lambda a: {"event": "escalate", "actor": a, "role": ROLES[a], "payload": {"report": "x"}}
    ev = [violation("a1", owner="a1"), violation("a4", owner="a4"), esc("a4"), esc("a2")]
    m = world_metrics(load_world(write_world(tmp_path, "w", "ANG", 0, ev, roles=ROLES,
                                             extra_manifest={"variants": VARIANTS})))
    assert m["real_impossible_agents"] == 4 and m["exploiters"] == ["a1"]
    assert m["angel_agents"] == ["a4"] and m["angel_test_edits"] == 1
    assert m["reports_by_subjects"] == 1 and m["reports_by_angels"] == 1
