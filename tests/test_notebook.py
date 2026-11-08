"""Solo notebook: somewhere to keep ideas, without other voices.

The team chat doubles as memory, but the board never re-shows an agent its own posts.
A notebook re-shows all of them on request, so solo / solo+notebook / team separate
memory from the mixing of ideas. Three wordings: bare (exists), matched (the team chat's
purpose hint), instructed (told to use it to come back to ideas).
"""
import json

import pytest

from swarmlab.conditions import Slot, WorldConfig
from swarmlab.fake_env import FakeTaskEnv
from swarmlab.models import ScriptedBackend
from swarmlab.prompts import NOTEBOOK_WORDING, build_system_prompt
from swarmlab.world import run_world

LEVELS = ("bare", "matched", "instructed")
TEAM_WORDS = ("team", "agents", "teammate", "board", "chat", "workspace")


def solo(**options):
    values = dict(run_id="nb", condition="SOLO_NOTEBOOK", block=0, seed=0, model="scripted",
                  task_ids=["t"], focal_slot=0, board=False, scope_rule="soft",
                  max_turns_per_agent=3, slots=(Slot("t", "conflicting", "peer"),))
    return WorldConfig(**{**values, **options})


def test_off_by_default():
    assert "notebook" not in build_system_prompt(solo(), "a1", "peer").lower()


@pytest.mark.parametrize("level", LEVELS)
def test_each_wording_mentions_the_notebook_and_no_team(level):
    system, purpose = NOTEBOOK_WORDING[level]
    prompt = build_system_prompt(solo(notebook=level), "a1", "peer")
    assert system and system in prompt and "notebook" in system.lower()
    for word in TEAM_WORDS:
        assert word not in (system + " " + purpose).lower()


def test_only_bare_gives_no_purpose():
    assert NOTEBOOK_WORDING["bare"][1] == ""
    assert NOTEBOOK_WORDING["matched"][1] and NOTEBOOK_WORDING["instructed"][1]


def test_rejects_unknown_level_and_a_team_board():
    with pytest.raises(ValueError, match="notebook"):
        solo(notebook="diary")
    with pytest.raises(ValueError, match="notebook"):
        solo(notebook="bare", board=True)


def test_notes_are_saved_reread_in_full_and_logged(tmp_path):
    read_tool = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "read_file",
                            "arguments": {"path": "lib.py"}})
    sent = []

    class Recording(ScriptedBackend):
        def generate(self, system, messages, max_tokens=1024):
            sent.append(messages[-1]["content"])
            return super().generate(system, messages, max_tokens)

    backend = Recording([read_tool, read_tool, read_tool],
                        ['{"message": "NOTE-ONE the test contradicts itself"}',
                         '{"read": true}', '{"message": "NOTE-TWO"}',
                         '{"read": true}', '{"message": null}'])
    run = run_world(solo(notebook="matched"), lambda _a: backend, FakeTaskEnv, tmp_path).run_dir
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()]
    notes = [e for e in events if e["event"] == "notebook_note"]
    reads = [e for e in events if e["event"] == "notebook_read"]
    assert [n["payload"]["text"] for n in notes] == ["NOTE-ONE the test contradicts itself", "NOTE-TWO"]
    assert [r["payload"]["n_notes"] for r in reads] == [1, 2]
    assert not [e for e in events if e["event"] == "board_post"]
    task_turns = [m for m in sent if "This is NOT a turn" not in m]
    assert "NOTE-ONE" in task_turns[-1] and "NOTE-TWO" in task_turns[-1]  # whole notebook
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["notebook"] == "matched"
