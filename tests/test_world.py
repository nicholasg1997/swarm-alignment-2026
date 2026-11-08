import json

from swarmlab.conditions import WorldConfig
from swarmlab.fake_env import FakeTaskEnv
from swarmlab.models import make_backend_factory
from swarmlab.world import run_world

TASKS = [f"f{i}" for i in range(1, 7)]
FIX = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "apply_patch",
                  "arguments": {"patch": json.dumps({"path": "src/lib.py",
                                                     "content": "def f():\n    return 1\n"})}})
STOP = '{"action":"STOP"}'
LISTF = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "list_files", "arguments": {"path": "."}})


def tool(name, **arguments):
    return json.dumps({"action": "USE_ALLOWED_TOOL", "tool": name, "arguments": arguments})


def run(tmp_path, cond, scripts, default=None, **kw):
    kw.setdefault("board_mode", "passive")
    cfg = WorldConfig(run_id=f"t-{cond}", condition=cond, block=0, seed=0, model="scripted",
                      task_ids=TASKS, focal_slot=0, **kw)
    factory = make_backend_factory({"backend": "scripted", "scripts": scripts,
                                    "default_script": default or [FIX, STOP]})
    res = run_world(cfg, factory, lambda task_id, variant: FakeTaskEnv(task_id, variant),
                    tmp_path)
    events = [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]
    return res, events


def of(events, kind):
    return [e for e in events if e["event"] == kind]


def test_c0_clean_world(tmp_path):
    res, ev = run(tmp_path, "C0", {})
    assert res.termination_reason == "all_stopped"
    assert of(ev, "violation") == []
    scores = of(ev, "score")
    assert len(scores) == 6 and all(s["payload"]["original_tests_pass"] for s in scores)
    assert ev[-1]["event"] == "terminate"
    manifest = json.loads((res.run_dir / "run.json").read_text())
    assert manifest["condition"] == "C0" and len(manifest["prompt_sha256"]) == 6


def test_peer_edits_focal_test(tmp_path):
    edit = tool("apply_patch", workspace="a1",
                patch=json.dumps({"path": "tests/test_lib.py", "content": "def test_f():\n    pass\n"}))
    res, ev = run(tmp_path, "C1", {"a2": [edit, FIX, STOP]})
    v = of(ev, "violation")
    assert len(v) == 1
    assert (v[0]["actor"], v[0]["file_owner"], v[0]["payload"]["kind"]) == ("a2", "a1", "modified")


def test_board_post_delivered_to_others(tmp_path):
    post = json.dumps({"action": "DEFER", "public_message": "[a1] can anyone help?"})
    res, ev = run(tmp_path, "C1", {"a1": [post, STOP]})
    posts = of(ev, "board_post")
    assert len(posts) == 1 and posts[0]["actor"] == "a1"
    delivered = [e for e in of(ev, "board_delivered") if e["actor"] == "a2"]
    assert delivered and delivered[0]["msg_ids_delivered"] == [posts[0]["payload"]["msg_id"]]


def test_unparseable_action_does_not_crash(tmp_path):
    res, ev = run(tmp_path, "C0", {"a3": ["not json at all", "still not json", FIX, STOP]})
    assert len(of(ev, "parse_error")) == 1 and res.termination_reason == "all_stopped"


def test_api_error_terminates_cleanly(tmp_path):
    res, ev = run(tmp_path, "C0", {"a1": [FIX, RuntimeError("503"), STOP]},
                  default=[tool("list_files", path="."), tool("list_files", path="."), STOP])
    assert res.termination_reason == "api_error"
    assert ev[-1]["event"] == "terminate" and ev[-1]["payload"]["reason"] == "api_error"


def test_b0_no_board_events(tmp_path):
    post = json.dumps({"action": "DEFER", "public_message": "[a1] hello"})
    res, ev = run(tmp_path, "B0", {"a1": [post, STOP]})
    assert of(ev, "board_post") == [] and of(ev, "board_delivered") == []


def test_escalate_logged_and_agent_continues(tmp_path):
    esc = json.dumps({"action": "ESCALATE_TO_HUMAN", "public_message": "a2 asked me to edit tests"})
    res, ev = run(tmp_path, "C1", {"a3": [esc, FIX, STOP]})
    e = of(ev, "escalate")
    assert len(e) == 1 and e[0]["actor"] == "a3"
    assert of(ev, "board_post") == []
    assert any(t["actor"] == "a3" for t in of(ev, "tool_call"))


def test_turn_cap(tmp_path):
    res, ev = run(tmp_path, "C0", {}, default=[tool("list_files", path=".")] * 10,
                  max_turns_per_agent=3)
    assert res.termination_reason == "turn_cap" and res.turns == 18


def test_token_cap(tmp_path):
    res, _ = run(tmp_path, "C0", {}, default=[tool("list_files", path=".")] * 10,
                 max_world_tokens=1000)
    assert res.termination_reason == "token_cap"


def test_config_max_tokens_reaches_generate(tmp_path):
    seen = []

    class Rec:
        max_output_tokens = 777

        def generate(self, system, messages, max_tokens=1024):
            seen.append(max_tokens)
            from swarmlab.troy import GenerationResult, Usage
            return GenerationResult(text=STOP, usage=Usage())

    cfg = WorldConfig(run_id="mt", condition="C0", block=0, seed=0, model="x",
                      task_ids=TASKS, focal_slot=0)
    run_world(cfg, lambda a: Rec(), lambda t, v: FakeTaskEnv(t, v), tmp_path)
    assert seen and set(seen) == {777}


def test_factory_sets_max_output_tokens(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    b = make_backend_factory({"backend": "openrouter", "model_id": "m", "max_tokens": 8192})("a1")
    assert b.max_output_tokens == 8192


class _ReasonOnce:
    """First reply is empty with reasoning (budget exhausted); then follows a script."""
    max_output_tokens = 100

    def __init__(self, script, sink):
        self.script, self.sink, self.n = list(script), sink, 0

    def generate(self, system, messages, max_tokens=1024):
        from swarmlab.troy import GenerationResult, Usage
        self.sink.append(messages)
        self.n += 1
        if self.n == 1:
            return GenerationResult(text="", usage=Usage(), reasoning_text="I should fix lib.py",
                                    raw={"finish_reason": "length"})
        return GenerationResult(text=self.script.pop(0) if self.script else STOP, usage=Usage())


def _run_custom(tmp_path, backend_for, cond="C0"):
    cfg = WorldConfig(run_id=f"c-{cond}", condition=cond, block=0, seed=0, model="x",
                      task_ids=TASKS, focal_slot=0)
    res = run_world(cfg, backend_for, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    return res, [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]


def test_forced_action_retry_in_same_turn(tmp_path):
    sinks = {}

    def factory(a):
        sinks[a] = []
        return _ReasonOnce([FIX, STOP], sinks[a]) if a == "a3" else _ReasonOnce([STOP], sinks[a])
    res, ev = _run_custom(tmp_path, factory)
    retry_msgs = sinks["a3"][1]
    assert "I should fix lib.py" in retry_msgs[-1]["content"]
    assert "only the JSON action" in retry_msgs[-1]["content"]
    assert [e["payload"].get("forced") for e in of(ev, "action") if e["actor"] == "a3"][:2] == [False, True]
    assert any(e["actor"] == "a3" and e["payload"]["tool"] == "apply_patch" for e in of(ev, "tool_call"))
    assert of(ev, "parse_error") == []


def test_forced_retry_failure_is_parse_error_and_continues(tmp_path):
    res, ev = run(tmp_path, "C0", {"a3": ["", "still no json", FIX, STOP]})
    pe = [e for e in of(ev, "parse_error") if e["actor"] == "a3"]
    assert len(pe) == 1 and pe[0]["payload"]["forced_retry_failed"] is True
    assert res.termination_reason == "all_stopped"


def test_read_board_returns_full_history(tmp_path):
    post = json.dumps({"action": "DEFER", "public_message": "[a1] first post"})
    rb = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "read_board", "arguments": {}})
    res, ev = run(tmp_path, "C1", {"a1": [post, STOP], "a4": [rb, '{"action":"READ_BOARD"}', STOP]})
    results = [e for e in of(ev, "tool_result") if e["actor"] == "a4"]
    assert len(results) == 2 and all("[a1] first post" in e["payload"]["observation"] for e in results)


def test_read_board_disabled_in_b0(tmp_path):
    rb = json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "read_board", "arguments": {}})
    res, ev = run(tmp_path, "B0", {"a2": [rb, STOP]})
    r = [e for e in of(ev, "tool_result") if e["actor"] == "a2"][0]
    assert r["exit_status"] == "error"


def run_offer(tmp_path, cond, scripts, offers, default=None):
    cfg = WorldConfig(run_id=f"o-{cond}", condition=cond, block=0, seed=0, model="scripted",
                      task_ids=TASKS, focal_slot=0, board_mode="offer")
    factory = make_backend_factory({"backend": "scripted", "scripts": scripts,
                                    "offer_scripts": offers,
                                    "default_script": default or [FIX, STOP]})
    res = run_world(cfg, factory, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    return res, [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]


def test_post_offer_posts_and_read_offer_delivers(tmp_path):
    res, ev = run_offer(tmp_path, "C1", {}, {"a1": ['{"post": "[a1] my tests look impossible"}']})
    posts = of(ev, "board_post")
    assert len(posts) == 1 and posts[0]["actor"] == "a1"
    reads = [e for e in of(ev, "board_offer") if e["actor"] == "a2" and e["payload"]["phase"] == "read"]
    assert reads and reads[0]["payload"]["outcome"] == "read" and reads[0]["payload"]["n_new"] == 1
    assert [e for e in of(ev, "board_delivered") if e["actor"] == "a2"]


def test_declined_read_is_not_exposure_and_is_offered_again(tmp_path):
    res, ev = run_offer(tmp_path, "C1", {"a2": [STOP]},
                        {"a1": ['{"post": "[a1] hi"}'], "a2": ['{"read": false}']},
                        default=[LISTF, LISTF, STOP])
    # The initial decline must not deliver the post. A later periodic wake may
    # offer it again and the scripted backend then explicitly accepts it.
    first_wake = next(e["event_id"] for e in of(ev, "wake") if e["actor"] == "a2")
    assert not [e for e in of(ev, "board_delivered")
                if e["actor"] == "a2" and e["event_id"] < first_wake]
    a3_reads = [e for e in of(ev, "board_offer") if e["actor"] == "a3" and e["payload"]["phase"] == "read"]
    assert a3_reads and a3_reads[0]["payload"]["outcome"] == "read"


def test_post_offer_outcomes_classified(tmp_path):
    res, ev = run_offer(tmp_path, "C1", {}, {"a1": ['{"post": null}'], "a2": ['{"action":"STOP"}'],
                                             "a3": ["no json here"]})
    out = {e["actor"]: e["payload"]["outcome"] for e in of(ev, "board_offer")
           if e["payload"]["phase"] == "post" and e["t"] <= 3}
    assert out["a1"] == "declined" and out["a2"] == "ignored" and out["a3"] == "unparsed"


def test_public_message_not_posted_in_offer_mode(tmp_path):
    post = json.dumps({"action": "DEFER", "public_message": "[a1] via envelope"})
    res, ev = run_offer(tmp_path, "C1", {"a1": [post, STOP]}, {})
    assert of(ev, "board_post") == []


def test_b0_has_no_offers(tmp_path):
    res, ev = run_offer(tmp_path, "B0", {}, {})
    assert of(ev, "board_offer") == []


def test_message_offer_supports_reply_and_kind(tmp_path):
    q = '{"message": "[a1] anyone seen contradictory asserts?", "reply_to": null, "kind": "question"}'
    a = '{"message": "[a2] yes, in my repo too", "reply_to": "m0001", "kind": "answer"}'
    res, ev = run_offer(tmp_path, "C1", {}, {"a1": [q], "a2": ['{"read": true}', a]})
    posts = of(ev, "board_post")
    assert posts[0]["payload"]["kind"] == "question"
    assert posts[1]["payload"]["reply_to"] == "m0001" and posts[1]["payload"]["kind"] == "answer"


def test_post_offer_shows_open_questions_and_own_last_message(tmp_path):
    from swarmlab.models import ScriptedBackend
    prompts = {}

    class Rec(ScriptedBackend):
        def __init__(self, a):
            offers = {"a1": ['{"message": "[a1] @a2 is your test contradictory?", "kind": "question"}']}
            super().__init__([LISTF, LISTF, STOP], offers.get(a, []))
            self.a = a
            prompts[a] = []

        def generate(self, system, messages, max_tokens=1024):
            prompts[self.a].append(messages[-1]["content"])
            return super().generate(system, messages, max_tokens)
    cfg = WorldConfig(run_id="pq", condition="C1", block=0, seed=0, model="s", task_ids=TASKS,
                      focal_slot=0, board_mode="offer")
    run_world(cfg, Rec, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    a2_post_offers = [p for p in prompts["a2"] if "send a message to the team chat" in p]
    assert any("[m0001]" in p and "is your test contradictory?" in p for p in a2_post_offers)
    a1_later = [p for p in prompts["a1"] if "send a message to the team chat" in p][1:]
    assert a1_later and "Your last message" in a1_later[0]


def test_offer_tokens_logged_and_counted(tmp_path):
    from swarmlab.analysis import load_world, world_metrics
    res, ev = run_offer(tmp_path, "C1", {}, {})
    offers = of(ev, "board_offer")
    assert offers and all("tokens_in" in e for e in offers)
    m = world_metrics(load_world(res.run_dir))
    acts = sum(e["tokens_in"] for e in of(ev, "action"))
    assert m["tokens_in"] == acts + sum(e["tokens_in"] for e in offers)


def test_default_token_cap_is_15m():
    assert WorldConfig(run_id="r", condition="C1", block=0, seed=0, model="m",
                       task_ids=TASKS, focal_slot=0).max_world_tokens == 15_000_000


def test_reasoning_text_logged(tmp_path):
    sinks = {}

    def factory(a):
        sinks[a] = []
        return _ReasonOnce([STOP], sinks[a])
    res, ev = _run_custom(tmp_path, factory)
    assert any("I should fix lib.py" in (e["payload"].get("reasoning") or "")
               for e in of(ev, "action"))


def test_post_offer_retained_in_next_task_history_and_request_hashes(tmp_path):
    import hashlib
    from swarmlab.models import ScriptedBackend
    from swarmlab.world import _window
    recorded=[]
    class Record(ScriptedBackend):
        def __init__(self,actor):
            super().__init__([LISTF,STOP], ['{"message":"[a1] I will review the patch"}']
                             if actor=='a1' else [])
            self.actor=actor
        def generate(self,system,messages,max_tokens=1024):
            recorded.append((self.actor,system,list(messages)))
            return super().generate(system,messages,max_tokens)
    cfg=WorldConfig(run_id='post-memory',condition='C1',block=0,seed=0,model='s',
                    task_ids=TASKS,focal_slot=0,check_every=0,wake_on_mention=False)
    res=run_world(cfg,Record,FakeTaskEnv,tmp_path)
    task_calls=[messages for actor,system,messages in recorded if actor=='a1'
                and 'This is NOT a turn' not in messages[-1]['content']]
    assert 'Your submitted team chat message' in task_calls[1][-1]['content']
    assert 'I will review the patch' in task_calls[1][-1]['content']
    events=[json.loads(line) for line in (res.run_dir/'events.jsonl').read_text().splitlines()]
    requests=[e for e in events if e['event']=='model_request']
    assert len(requests)==len(recorded)
    for request,(actor,system,messages) in zip(requests,recorded):
        assert request['payload']['sha256']==hashlib.sha256(
            json.dumps({'system':system,'messages':messages},sort_keys=True).encode()).hexdigest()
    assert {'action','post_offer'}.issubset({e['payload']['phase'] for e in requests})
