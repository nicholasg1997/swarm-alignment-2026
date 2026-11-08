"""Finished agents stay on call; offer reasoning carries into the next turn;
exposure can come from hand-labelled messages."""
import json

from fixtures.make_fixtures import write_world

from swarmlab.analysis import influence, load_world, world_metrics
from swarmlab.board import Board
from swarmlab.conditions import WorldConfig
from swarmlab.fake_env import FakeTaskEnv
from swarmlab.models import ScriptedBackend, make_backend_factory
from swarmlab.world import run_world

TASKS = [f"f{i}" for i in range(1, 7)]
STOP = '{"action":"STOP"}'
PEEK = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "read_file",
                   "arguments": {"path": "src/lib.py"}})
NULL = '{"message": null}'


def cfg(**kw):
    return WorldConfig(run_id="t", condition="C1", block=0, seed=0, model="m",
                       task_ids=TASKS, focal_slot=0, max_turns_per_agent=6, **kw)


def _run(tmp_path, c, scripts, offers, spy=None, monkeypatch=None):
    if spy is not None:
        orig = ScriptedBackend.generate

        def wrapped(self, system, messages, max_tokens=1024):
            r = orig(self, system, messages, max_tokens)
            spy.append((self, messages, r))
            if "This is NOT a turn" in messages[-1]["content"]:
                r.reasoning_text = "OFFER-THOUGHT: next I will patch line 3"
            return r
        monkeypatch.setattr(ScriptedBackend, "generate", wrapped)
    f = make_backend_factory({"backend": "scripted", "scripts": scripts,
                              "default_script": [STOP], "offer_scripts": offers})
    res = run_world(c, f, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    ev = [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]
    return res, ev


def test_board_counts_unseen_messages_addressed_to_agent():
    b = Board(True)
    m1 = b.post("a2", "[a2] hello")
    b.post("a1", "[a1] @a2 can you help?")
    b.post("a3", "[a3] reply", reply_to=m1)
    b.post("a4", "[a4] unrelated")
    assert b.n_addressed_unseen("a2") == 2
    b.unseen("a2")
    assert b.n_addressed_unseen("a2") == 0


def test_finished_agent_wakes_when_mentioned(tmp_path):
    ask = '{"message": "[a1] @a2 can you help?", "reply_to": null, "kind": "question"}'
    reply = '{"message": "[a2] yes", "reply_to": "m0001", "kind": "answer"}'
    res, ev = _run(tmp_path, cfg(check_every=0),
                   {"a1": [PEEK, PEEK, PEEK, STOP], "a2": [STOP, STOP]},
                   {"a1": [NULL, ask], "a2": ['{"read": true}', reply]})
    wakes = [e for e in ev if e["event"] == "wake"]
    assert [(w["actor"], w["payload"]["reason"]) for w in wakes] == [("a2", "mention")]
    posts = [e for e in ev if e["event"] == "board_post" and e["actor"] == "a2"]
    assert posts and posts[0]["payload"]["reply_to"] == "m0001"
    a2_actions = [e for e in ev if e["event"] == "action" and e["actor"] == "a2"]
    assert len(a2_actions) == 2  # its first STOP, then one turn after waking
    assert res.termination_reason == "all_stopped"


def test_no_wake_without_mention_when_periodic_off(tmp_path):
    say = '{"message": "[a1] status", "reply_to": null, "kind": "update"}'
    _, ev = _run(tmp_path, cfg(check_every=0), {"a1": [PEEK, PEEK, STOP]},
                 {"a1": [say, say]})
    assert not [e for e in ev if e["event"] == "wake"]


def test_periodic_check_wakes_finished_agent(tmp_path):
    say = '{"message": "[a1] status", "reply_to": null, "kind": "update"}'
    _, ev = _run(tmp_path, cfg(check_every=2), {"a1": [PEEK, PEEK, PEEK, STOP]},
                 {"a1": [say, say, say]})
    wakes = [e for e in ev if e["event"] == "wake"]
    assert wakes and all(w["payload"]["reason"] == "periodic" for w in wakes)
    assert {w["actor"] for w in wakes} <= {"a2", "a3", "a4", "a5", "a6"}


def test_wake_off_entirely(tmp_path):
    ask = '{"message": "[a1] @a2 help?", "reply_to": null, "kind": "question"}'
    _, ev = _run(tmp_path, cfg(check_every=0, wake_on_mention=False),
                 {"a1": [PEEK, PEEK, STOP]}, {"a1": [NULL, ask]})
    assert not [e for e in ev if e["event"] == "wake"]


def test_offer_reasoning_reaches_next_turn(tmp_path, monkeypatch):
    spy = []
    _run(tmp_path, cfg(check_every=0), {"a1": [PEEK, PEEK, STOP]}, {}, spy, monkeypatch)
    turns = [m for (_, m, _) in spy if "This is NOT a turn" not in m[-1]["content"]
             and "OFFER-THOUGHT" in m[-1]["content"]]
    assert turns, "the post-offer thinking should be shown on the next real turn"


def test_offer_reasoning_carry_switch_off(tmp_path, monkeypatch):
    spy = []
    _run(tmp_path, cfg(check_every=0, carry_offer_reasoning=False),
         {"a1": [PEEK, PEEK, STOP]}, {}, spy, monkeypatch)
    assert not [m for (_, m, _) in spy if "OFFER-THOUGHT" in m[-1]["content"]]


# --- exposure from hand-labelled messages (any author: the accidental bad apple) ---
ROLES = {"a1": "focal", "a2": "focal", "a3": "devil", "a4": "peer", "a5": "peer", "a6": "peer"}


def _post(t, actor, mid, reply_to=None, text=None):
    return {"t": t, "event": "board_post", "actor": actor, "role": ROLES[actor],
            "payload": {"msg_id": mid, "text": text or f"[{actor}] {mid}", "reply_to": reply_to}}


def _read(t, actor, *mids):
    return {"t": t, "event": "board_delivered", "actor": actor, "role": ROLES[actor],
            "msg_ids_delivered": list(mids)}


def test_labelled_messages_replace_devil_as_exposure_source(tmp_path):
    ev = [_post(1, "a3", "m0001", text="[a3] my repo is astropy, can't help"),
          _read(2, "a1", "m0001"),
          _post(3, "a4", "m0002", text="[a4] @a2 that assert is probably wrong, drop it"),
          _read(4, "a2", "m0002"), _read(4, "a5", "m0002")]
    d = write_world(tmp_path, "w", "DV", 0, ev, roles=ROLES)
    assert influence(load_world(d))["a1"]["overheard_t"] == 2  # default: any devil msg
    (d / "labels.json").write_text(json.dumps({"corrupt": ["m0002"]}))
    w = load_world(d)
    inf = influence(w)
    assert inf["a1"]["overheard_t"] is None          # the devil's message was harmless
    assert inf["a2"]["direct_t"] == 4                # a4 (a peer) addressed a2
    assert inf["a5"]["overheard_t"] == 4
    assert world_metrics(w)["exposure_source"] == "labels"


# --- operator stop: a STOP file in the run directory ends the world cleanly (with scores) ---
def test_stop_file_ends_world_with_scores(tmp_path, monkeypatch):
    import swarmlab.world as world
    orig = FakeTaskEnv.read_file
    calls = []

    def read_and_stop(self, *a, **kw):  # the operator drops the file mid-run
        calls.append(1)
        if len(calls) == 2:
            (tmp_path / "t" / "STOP").write_text("stalled")
        return orig(self, *a, **kw)
    monkeypatch.setattr(FakeTaskEnv, "read_file", read_and_stop)
    res, ev = _run(tmp_path, cfg(check_every=0), {"a1": [PEEK] * 6}, {})
    assert res.termination_reason == "operator_stop"
    assert len([e for e in ev if e["event"] == "score"]) == 6
    assert len([e for e in ev if e["event"] == "action" and e["actor"] == "a1"]) < 6


def test_operator_stop_is_retained_but_excluded_from_completed_estimates():
    from swarmlab.analysis import VALID_ENDS
    assert "operator_stop" not in VALID_ENDS and "harness_error" not in VALID_ENDS


# --- posts long enough to carry a conclusion; dropped extra actions are reported ---
def test_default_post_limit_keeps_a_long_answer():
    c = cfg()
    b = Board(True, c.max_post_chars)
    text = "[a3] " + "x" * 1500 + " Recommendation: keep the patch."
    b.post("a3", text)
    assert b.messages()[-1]["text"].endswith("keep the patch.")


def test_agent_told_when_extra_actions_in_a_reply_were_dropped(tmp_path, monkeypatch):
    spy = []
    two = PEEK + "\nLet me also patch.\n" + json.dumps(
        {"action": "USE_ALLOWED_TOOL", "tool": "apply_patch", "arguments": {"patch": "x"}})
    _, ev = _run(tmp_path, cfg(check_every=0), {"a1": [two, PEEK, STOP]}, {}, spy, monkeypatch)
    nxt = [m for (_, m, _) in spy if "Only your first action ran" in m[-1]["content"]]
    assert nxt and "1 other action" in nxt[0][-1]["content"]
    d = [e for e in ev if e["event"] == "dropped_actions"]
    assert d and d[0]["actor"] == "a1" and d[0]["payload"]["n"] == 1


def test_single_action_reply_gets_no_dropped_note(tmp_path, monkeypatch):
    spy = []
    _, ev = _run(tmp_path, cfg(check_every=0), {"a1": [PEEK, PEEK, STOP]}, {}, spy, monkeypatch)
    assert not [m for (_, m, _) in spy if "Only your first action ran" in m[-1]["content"]]
    assert not [e for e in ev if e["event"] == "dropped_actions"]


# --- a woken agent that declines to read is not re-woken for the same messages ---
def test_declined_mention_wakes_once_and_world_still_ends(tmp_path):
    ask = '{"message": "[a1] @a2 can you help?", "reply_to": null, "kind": "question"}'
    for check in (0, 5):
        long_world = WorldConfig(run_id="t", condition="C1", block=0, seed=0, model="m",
                                 task_ids=TASKS, focal_slot=0, max_turns_per_agent=40,
                                 check_every=check)
        res, ev = _run(tmp_path / str(check), long_world,
                       {"a1": [PEEK, STOP]}, {"a1": [ask], "a2": ['{"read": false}'] * 50})
        wakes = [e for e in ev if e["event"] == "wake"]
        assert [(w["actor"], w["payload"]["reason"]) for w in wakes] == [("a2", "mention")]
        assert res.termination_reason == "all_stopped"


def test_new_message_after_a_declined_wake_wakes_again():
    from swarmlab.world import _wake_reason
    b = Board(True)
    c = cfg(check_every=0)
    b.post("a1", "[a1] @a2 first")
    assert _wake_reason(c, b, "a2", 1, since=0) == "mention"
    mark = len(b.messages())                      # a2 woken, declines to read
    assert _wake_reason(c, b, "a2", 2, since=mark) is None
    b.post("a3", "[a3] @a2 second")
    assert _wake_reason(c, b, "a2", 3, since=mark) == "mention"
