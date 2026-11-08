import pytest

from swarmlab.conditions import CONDITIONS, WorldConfig, variant_for
from swarmlab.prompts import (ANGEL_TEXT, BOARD_POST_LINE, HELP_LINE, PC_TEXT,
                              build_system_prompt, role_for)

TASKS = [f"f{i}" for i in range(1, 7)]


def cfg(cond, angel=None, mode="passive"):
    return WorldConfig(run_id="r", condition=cond, block=0, seed=0, model="m",
                       task_ids=TASKS, focal_slot=0, angel_slot=angel, board_mode=mode)


@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_help_line_in_every_board_condition_and_role(cond):
    c = cfg(cond, angel=1 if CONDITIONS[cond].angel else None)
    for slot in range(6):
        p = build_system_prompt(c, f"a{slot + 1}", role_for(c, slot))
        assert (HELP_LINE in p) == CONDITIONS[cond].board_enabled
        sign = f"Your agent ID is a{slot + 1}. Start every board post with [a{slot + 1}]."
        assert (sign in p) == CONDITIONS[cond].board_enabled  # no chat: no board mentioned
        assert "ESCALATE_TO_HUMAN" in p


def test_b0_has_no_board_text():
    p = build_system_prompt(cfg("B0"), "a1", "focal")
    assert "message board" not in p and BOARD_POST_LINE not in p
    assert BOARD_POST_LINE in build_system_prompt(cfg("C1"), "a1", "focal")


def test_angel_text_only_for_angel_role_in_c2():
    c = cfg("C2", angel=1)
    assert role_for(c, 1) == "angel"
    assert ANGEL_TEXT in build_system_prompt(c, "a2", "angel")
    assert ANGEL_TEXT not in build_system_prompt(c, "a3", "peer")


def test_pc_text_only_for_focal_in_pc():
    c = cfg("PC")
    assert PC_TEXT in build_system_prompt(c, "a1", role_for(c, 0))
    assert PC_TEXT not in build_system_prompt(c, "a2", role_for(c, 1))
    assert PC_TEXT not in build_system_prompt(cfg("C1"), "a1", "focal")


def test_variants():
    assert [variant_for(cfg("C0"), s) for s in range(6)] == ["original"] * 6
    assert [variant_for(cfg("C1"), s) for s in range(6)] == ["conflicting"] + ["original"] * 5


def test_angel_cannot_be_focal():
    with pytest.raises(ValueError):
        cfg("C2", angel=0)


def test_c2_requires_angel_slot():
    with pytest.raises(ValueError):
        cfg("C2")


def test_posting_is_optional_and_read_board_documented():
    from swarmlab.prompts import POSTING_OPTIONAL_LINE, READ_BOARD_LINE
    p = build_system_prompt(cfg("C1"), "a1", "focal")
    assert POSTING_OPTIONAL_LINE in p and READ_BOARD_LINE in p
    assert READ_BOARD_LINE not in build_system_prompt(cfg("B0"), "a1", "focal")


def test_team_framing_and_offer_line():
    from swarmlab.prompts import OFFER_LINE, TEAM_LINE
    p = build_system_prompt(cfg("C1", mode="offer"), "a1", "focal")
    assert TEAM_LINE in p and OFFER_LINE in p and BOARD_POST_LINE not in p
    b0 = build_system_prompt(cfg("B0", mode="offer"), "a1", "focal")
    assert TEAM_LINE not in b0 and OFFER_LINE not in b0 and HELP_LINE not in b0


def test_team_framing_can_be_switched_off():
    from swarmlab.prompts import TEAM_LINE
    c = WorldConfig(run_id="r", condition="C1", block=0, seed=0, model="m", task_ids=TASKS,
                    focal_slot=0, team_framing=False)
    assert TEAM_LINE not in build_system_prompt(c, "a1", "focal")


def test_offer_is_default_mode():
    assert WorldConfig(run_id="r", condition="C1", block=0, seed=0, model="m",
                       task_ids=TASKS, focal_slot=0).board_mode == "offer"


def test_team_chat_intro_replaces_troy_board_text():
    from swarmlab.prompts import TEAM_CHAT_INTRO
    p = build_system_prompt(cfg("C1", mode="offer"), "a1", "focal")
    assert TEAM_CHAT_INTRO in p and "message board for agents" not in p
