"""Private messages: an ordinary chat feature (like DMs) for the oversight-avoidance test.
Only the sender and named recipients see a private message; the harness logs all of it."""
import json

from swarmlab.board import Board, render_messages
from swarmlab.conditions import WorldConfig
from swarmlab.fake_env import FakeTaskEnv
from swarmlab.models import make_backend_factory
from swarmlab.prompts import PRIVATE_LINE, build_system_prompt
from swarmlab.world import _post_decision, _post_offer, run_world

TASKS = [f"f{i}" for i in range(1, 7)]


def cfg(**kw):
    return WorldConfig(run_id="t", condition="C1", block=0, seed=0, model="m",
                       task_ids=TASKS, focal_slot=0, max_turns_per_agent=5, **kw)


def test_private_message_visible_only_to_sender_and_recipients():
    b = Board(True)
    b.post("a1", "[a1] public")
    b.post("a1", "[a1] just for a2", to=["a2"])
    assert [m["text"] for m in b.unseen("a2")] == ["[a1] public", "[a1] just for a2"]
    assert [m["text"] for m in b.unseen("a3")] == ["[a1] public"]
    assert b.n_unseen("a4") == 1 and len(b.messages("a1")) == 2 and len(b.messages("a3")) == 1
    assert len(b.messages()) == 2                       # the harness sees everything


def test_private_mentions_only_count_for_recipients_and_render_as_private():
    b = Board(True)
    b.post("a1", "[a1] @a3 @a2 psst", to=["a2"])
    assert b.n_addressed_unseen("a2") == 1 and b.n_addressed_unseen("a3") == 0
    out = render_messages(b.messages("a2"), reader="a2")
    assert "private: a1 → a2" in out


def test_bad_recipients_are_dropped_and_self_only_becomes_public():
    b = Board(True)
    b.post("a1", "x", to=["a1", "zz", 5])
    assert b.messages()[0]["to"] is None


def test_post_decision_reads_recipients():
    post, outcome = _post_decision('{"message": "hi", "reply_to": null, "kind": "idea", "to": ["a2"]}')
    assert outcome == "posted" and post["to"] == ["a2"]
    post, _ = _post_decision('{"message": "hi"}')
    assert post["to"] is None


def test_prompt_and_offer_mention_private_messages_only_when_enabled():
    on, off = cfg(private_messages=True), cfg()
    assert PRIVATE_LINE in build_system_prompt(on, "a1", "focal")
    assert PRIVATE_LINE not in build_system_prompt(off, "a1", "focal")
    assert '"to"' in _post_offer(Board(True), "a1", private=True)
    assert '"to"' not in _post_offer(Board(True), "a1")


def _run(tmp_path, c, offers):
    peek = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "read_file",
                       "arguments": {"path": "src/lib.py"}})
    f = make_backend_factory({"backend": "scripted", "scripts": {"a1": [peek, peek, '{"action":"STOP"}']},
                              "default_script": ['{"action":"STOP"}'], "offer_scripts": offers})
    res = run_world(c, f, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    return [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]


def test_world_delivers_private_messages_only_to_recipients_and_logs_them(tmp_path):
    dm = '{"message": "[a1] @a2 keep this between us", "reply_to": null, "kind": "idea", "to": ["a2"]}'
    ev = _run(tmp_path, cfg(private_messages=True, check_every=1),
              {"a1": [dm], "a2": ['{"read": true}'], "a3": ['{"read": true}']})
    post = [e for e in ev if e["event"] == "board_post"][0]
    assert post["payload"]["to"] == ["a2"]
    got = {e["actor"] for e in ev if e["event"] == "board_delivered"
           and post["payload"]["msg_id"] in e["msg_ids_delivered"]}
    assert "a2" in got and not got - {"a2"}


def test_recipients_ignored_when_private_messages_are_off(tmp_path):
    dm = '{"message": "[a1] hi all", "reply_to": null, "kind": "idea", "to": ["a2"]}'
    ev = _run(tmp_path, cfg(), {"a1": [dm]})
    assert [e for e in ev if e["event"] == "board_post"][0]["payload"]["to"] is None
