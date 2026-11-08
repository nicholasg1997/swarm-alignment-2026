"""Framing switches: pair without chat, offer wording, autonomy and stakes lines."""
import json

import pytest

from swarmlab.conditions import CONDITIONS, WorldConfig, layout_for
from swarmlab.fake_env import FakeTaskEnv
from swarmlab.models import make_backend_factory
from swarmlab.prompts import (AUTONOMY_LINE, DELEGATE_LINE, PAIR_LINE_NO_CHAT,
                              STAKES_HIGH_LINE, TEAM_CHAT_INTRO, WAKE_REACH_LINE,
                              build_system_prompt, team_line)
from swarmlab.world import run_world

TASKS = [f"f{i}" for i in range(1, 7)]
STOP = '{"action":"STOP"}'


def cfg(cond, **kw):
    return WorldConfig(run_id=f"t-{cond}", condition=cond, block=0, seed=0, model="m",
                       task_ids=TASKS, focal_slot=0, **kw)


def test_p0_is_a_pair_without_chat():
    assert not CONDITIONS["P0"].board_enabled and CONDITIONS["P0"].pair
    c = cfg("P0")
    lay = layout_for(c)
    assert lay[0].team == lay[1].team == "pair" and lay[1].task_id == "f1"
    p = build_system_prompt(c, "a1", "focal")
    assert PAIR_LINE_NO_CHAT.format(others="a2") in p
    assert TEAM_CHAT_INTRO not in p and "chat" not in team_line(c, "a1")


def test_pair_with_chat_keeps_chat_line():
    p = build_system_prompt(cfg("PR"), "a1", "focal")
    assert "team chat" in team_line(cfg("PR"), "a1") and team_line(cfg("PR"), "a1") in p


def test_autonomy_line_default_on_and_switchable():
    assert AUTONOMY_LINE in build_system_prompt(cfg("C1"), "a3", "peer")
    assert AUTONOMY_LINE in build_system_prompt(cfg("B0"), "a1", "focal")
    assert AUTONOMY_LINE not in build_system_prompt(cfg("C1", autonomy_framing=False),
                                                    "a3", "peer")


def test_autonomy_line_does_not_say_work_alone():
    # Rerun 2026-11-05: chat fell ~4x after "make the right choices on your own" was added.
    assert "on your own" not in AUTONOMY_LINE
    assert "no one is monitoring you" in AUTONOMY_LINE


def test_delegate_line_only_with_chat_and_switchable():
    assert DELEGATE_LINE in build_system_prompt(cfg("C1"), "a3", "peer")
    assert DELEGATE_LINE in build_system_prompt(cfg("PR"), "a1", "focal")
    for cond in ("B0", "P0"):  # no chat, so no claim that teammates can take work
        assert DELEGATE_LINE not in build_system_prompt(cfg(cond), "a1", "focal")
    assert DELEGATE_LINE not in build_system_prompt(cfg("C1", helper_framing=False),
                                                    "a3", "peer")


def test_finished_teammates_reachable_only_when_wake_is_on():
    assert WAKE_REACH_LINE in build_system_prompt(cfg("C1"), "a3", "peer")
    assert WAKE_REACH_LINE not in build_system_prompt(cfg("C1", wake_on_mention=False),
                                                      "a3", "peer")


def test_stakes_levels():
    assert STAKES_HIGH_LINE not in build_system_prompt(cfg("C1"), "a1", "focal")
    hi = cfg("C1", stakes="high")
    for slot in range(6):
        assert STAKES_HIGH_LINE in build_system_prompt(hi, f"a{slot + 1}", "peer")
    with pytest.raises(ValueError):
        cfg("C1", stakes="extreme")
    with pytest.raises(ValueError):
        cfg("C1", offer_style="shout")


def _offers(tmp_path, c):
    post = '{"message": "hi", "reply_to": null, "kind": "update"}'
    peek = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "read_file",
                       "arguments": {"path": "src/lib.py"}})
    f = make_backend_factory({"backend": "scripted", "default_script": [STOP],
                              "scripts": {"a1": [peek, STOP]},
                              "offer_scripts": {"a1": [post]}})
    res = run_world(c, f, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    ev = [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]
    return res, ev


@pytest.mark.parametrize("style,marker", [("notification", "Automatic team-chat notification"),
                                          ("pause", "PAUSE.")])
def test_offer_style_wording_reaches_the_model(tmp_path, style, marker, monkeypatch):
    seen = []
    import swarmlab.models as models
    orig = models.ScriptedBackend.generate

    def spy(self, system, messages, max_tokens=1024):
        seen.append(messages[-1]["content"])
        return orig(self, system, messages, max_tokens)
    monkeypatch.setattr(models.ScriptedBackend, "generate", spy)
    res, ev = _offers(tmp_path, cfg("C1", offer_style=style))
    offers = [s for s in seen if "This is NOT a turn" in s]
    assert offers and all(marker in s for s in offers)
    assert any(e["event"] == "board_post" for e in ev)
    m = json.loads((res.run_dir / "run.json").read_text())
    assert m["offer_style"] == style and m["autonomy_framing"] is True
