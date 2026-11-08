import pytest

from swarmlab.models import ScriptedBackend, load_model_config, make_backend_factory


def test_scripted_sequence_then_stop():
    b = ScriptedBackend(['{"action":"DEFER"}'])
    assert b.generate("s", []).text == '{"action":"DEFER"}'
    assert b.generate("s", []).text == '{"action":"STOP"}'


def test_scripted_raises():
    b = ScriptedBackend([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        b.generate("s", [])


def test_scripted_factory_gives_per_agent_scripts():
    f = make_backend_factory({"backend": "scripted",
                              "scripts": {"a1": ['{"action":"DEFER"}']}})
    assert f("a1").generate("s", []).text == '{"action":"DEFER"}'
    assert f("a2").generate("s", []).text == '{"action":"STOP"}'


def test_factory_openrouter_sets_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    b = make_backend_factory({"backend": "openrouter", "model_id": "vendor/m",
                              "temperature": 0.7})("a1")
    assert b.base_url == "https://openrouter.ai/api/v1"
    assert b.api_key_env == "OPENROUTER_API_KEY" and b.model_id == "vendor/m"


def test_load_model_config(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("backend: scripted\n")
    assert load_model_config(p)["backend"] == "scripted"


def test_load_dotenv_sets_missing_vars_only(tmp_path, monkeypatch):
    from swarmlab.models import load_dotenv
    (tmp_path / ".env").write_text("# comment\nSWARMLAB_T1=from_file\nSWARMLAB_T2='quoted'\n")
    monkeypatch.setenv("SWARMLAB_T1", "already")
    monkeypatch.delenv("SWARMLAB_T2", raising=False)
    load_dotenv(tmp_path / ".env")
    import os
    assert os.environ["SWARMLAB_T1"] == "already" and os.environ["SWARMLAB_T2"] == "quoted"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    from swarmlab.models import load_dotenv
    load_dotenv(tmp_path / "nope.env")


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeClient:
    def __init__(self, content, reasoning, finish="stop"):
        msg = _Obj(content=content, reasoning=reasoning)
        resp = _Obj(choices=[_Obj(message=msg, finish_reason=finish)],
                    usage=_Obj(prompt_tokens=10, completion_tokens=20,
                               prompt_tokens_details=None,
                               completion_tokens_details=_Obj(reasoning_tokens=15)))
        self.calls = []
        self.chat = _Obj(completions=_Obj(create=lambda **kw: self.calls.append(kw) or resp))


def test_openrouter_backend_captures_reasoning_and_truncation():
    from swarmlab.models import OpenRouterBackend
    fake = _FakeClient("", "thinking hard", finish="length")
    b = OpenRouterBackend("vendor/m", client=fake)
    r = b.generate("sys", [{"role": "user", "content": "hi"}], max_tokens=50)
    assert r.reasoning_text == "thinking hard" and r.raw["finish_reason"] == "length"
    assert r.usage.input_tokens == 10 and r.usage.reasoning_tokens == 15
    sent = fake.calls[0]
    assert sent["max_tokens"] == 50 and sent["messages"][0] == {"role": "system", "content": "sys"}


def test_reasoning_effort_sent_only_when_configured():
    from swarmlab.models import OpenRouterBackend
    fake = _FakeClient("{}", None)
    OpenRouterBackend("vendor/m", client=fake, reasoning_effort="medium").generate(
        "sys", [{"role": "user", "content": "hi"}])
    assert fake.calls[0]["extra_body"] == {"reasoning": {"effort": "medium"}}
    fake2 = _FakeClient("{}", None)
    OpenRouterBackend("vendor/m", client=fake2).generate("sys", [{"role": "user", "content": "hi"}])
    assert "extra_body" not in fake2.calls[0]


def test_factory_passes_reasoning_effort(monkeypatch):
    from swarmlab.models import make_backend_factory
    b = make_backend_factory({"backend": "openrouter", "model_id": "vendor/m",
                              "reasoning_effort": "low"})("a1")
    assert b.reasoning_effort == "low"


def test_action_event_logs_hidden_reasoning_tokens(tmp_path):
    import json
    from swarmlab.conditions import WorldConfig
    from swarmlab.fake_env import FakeTaskEnv
    from swarmlab.world import run_world
    c = WorldConfig(run_id="t", condition="C1", block=0, seed=0, model="m",
                    task_ids=[f"f{i}" for i in range(1, 7)], focal_slot=0, max_turns_per_agent=1)
    f = make_backend_factory({"backend": "scripted", "default_script": ['{"action":"STOP"}']})
    res = run_world(c, f, lambda t, v: FakeTaskEnv(t, v), tmp_path)
    ev = [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]
    acts = [e for e in ev if e["event"] == "action"]
    assert acts and all("reasoning_tokens" in e["payload"] for e in acts)


def test_mix_backend_gives_each_agent_its_own_model(tmp_path):
    from swarmlab.models import OpenRouterBackend
    (tmp_path / "s.yaml").write_text("name: s\nbackend: scripted\ndefault_script: ['{\"action\":\"STOP\"}']\n")
    (tmp_path / "o.yaml").write_text("name: o\nbackend: openrouter\nmodel_id: vendor/big\n")
    f = make_backend_factory({"backend": "mix", "name": "mix",
                              "agents": {"a1": str(tmp_path / "o.yaml")},
                              "default": str(tmp_path / "s.yaml")})
    assert isinstance(f("a1"), OpenRouterBackend) and f("a1").model_id == "vendor/big"
    assert isinstance(f("a2"), ScriptedBackend)


def test_mix_backend_without_default_rejects_unlisted_agent(tmp_path):
    (tmp_path / "s.yaml").write_text("name: s\nbackend: scripted\n")
    f = make_backend_factory({"backend": "mix", "agents": {"a1": str(tmp_path / "s.yaml")}})
    with pytest.raises(ValueError):
        f("a2")


def test_manifest_records_each_agents_model(tmp_path):
    import json
    from swarmlab.conditions import WorldConfig
    from swarmlab.fake_env import FakeTaskEnv
    from swarmlab.world import run_world
    (tmp_path / "s.yaml").write_text("name: scripted_a\nbackend: scripted\ndefault_script: ['{\"action\":\"STOP\"}']\n")
    (tmp_path / "t.yaml").write_text("name: scripted_b\nbackend: scripted\ndefault_script: ['{\"action\":\"STOP\"}']\n")
    f = make_backend_factory({"backend": "mix", "agents": {"a1": str(tmp_path / "t.yaml")},
                              "default": str(tmp_path / "s.yaml")})
    c = WorldConfig(run_id="t", condition="C1", block=0, seed=0, model="mix",
                    task_ids=[f"f{i}" for i in range(1, 7)], focal_slot=0, max_turns_per_agent=1)
    res = run_world(c, f, lambda t, v: FakeTaskEnv(t, v), tmp_path / "runs")
    m = json.loads((res.run_dir / "run.json").read_text())
    assert m["agent_models"]["a1"] == "scripted_b" and m["agent_models"]["a2"] == "scripted_a"


class _FakeAdaption:
    """Shaped like adaption 0.19 Responses: reasoning items carry reasoning_text parts."""

    def __init__(self, text, reasoning, status="completed", reason=None):
        usage = {"input_tokens": 100, "output_tokens": 40,
                 "input_tokens_details": {"cached_tokens": 60},
                 "output_tokens_details": {"reasoning_tokens": 30}, "cost": 1.5e-05}
        self.resp = _Obj(status=status, output_text=text,
                         incomplete_details=_Obj(reason=reason) if reason else None,
                         output=[_Obj(type="reasoning", summary=[],
                                      content=[_Obj(type="reasoning_text", text=reasoning)]),
                                 _Obj(type="message", content=[])],
                         usage=_Obj(to_dict=lambda: usage))
        self.calls = []
        self.responses = _Obj(create=lambda **kw: self.calls.append(kw) or self.resp)


def test_adaption_backend_maps_responses_api():
    from swarmlab.models import AdaptionBackend
    fake = _FakeAdaption('{"action":"STOP"}', "visible thinking")
    b = AdaptionBackend("deepseek-v4-flash", client=fake, temperature=0.7)
    r = b.generate("sys", [{"role": "user", "content": "hi"}], max_tokens=50)
    assert r.text == '{"action":"STOP"}' and r.reasoning_text == "visible thinking"
    assert (r.usage.input_tokens, r.usage.cached_input_tokens, r.usage.output_tokens,
            r.usage.reasoning_tokens) == (100, 60, 40, 30)
    assert r.raw["finish_reason"] == "stop" and r.raw["cost"] == 1.5e-05
    sent = fake.calls[0]
    assert sent["instructions"] == "sys" and sent["input"] == [{"role": "user", "content": "hi"}]
    assert sent["max_output_tokens"] == 50 and sent["temperature"] == 0.7
    assert b.provider == "adaption"


def test_adaption_backend_reports_truncation_as_length():
    from swarmlab.models import AdaptionBackend
    fake = _FakeAdaption("", "ran out", status="incomplete", reason="max_output_tokens")
    r = AdaptionBackend("m", client=fake).generate("sys", [{"role": "user", "content": "hi"}])
    assert r.raw["finish_reason"] == "length" and r.reasoning_text == "ran out"


def test_factory_builds_adaption_backend():
    from swarmlab.models import make_backend_factory
    b = make_backend_factory({"backend": "adaption", "model_id": "qwen3.8-27b",
                              "max_tokens": 4096, "temperature": 0.5})("a1")
    assert (b.provider, b.model_id, b.max_output_tokens, b.temperature) == (
        "adaption", "qwen3.8-27b", 4096, 0.5)


def test_world_totals_provider_cost_including_offers(tmp_path):
    import json
    from swarmlab.conditions import WorldConfig
    from swarmlab.fake_env import FakeTaskEnv
    from swarmlab.world import run_world

    class Costed(ScriptedBackend):
        def generate(self, system, messages, max_tokens=1024):
            r = super().generate(system, messages, max_tokens)
            r.raw = {"finish_reason": "stop", "cost": 0.25}
            return r

    c = WorldConfig(run_id="cost", condition="C1", block=0, seed=0, model="m",
                    task_ids=[f"f{i}" for i in range(1, 7)], focal_slot=0, max_turns_per_agent=1)
    res = run_world(c, lambda a: Costed(['{"action":"STOP"}']), FakeTaskEnv, tmp_path)
    ev = [json.loads(l) for l in (res.run_dir / "events.jsonl").read_text().splitlines()]
    calls = sum(1 for e in ev if e["event"] in ("action", "board_offer"))
    end = ev[-1]["payload"]
    assert calls and end["cost"] == 0.25 * calls
    assert sum(end["cost_by_agent"].values()) == end["cost"]


def test_adaption_requests_are_explicitly_stateless_and_tool_free():
    from swarmlab.models import AdaptionBackend
    fake = _FakeAdaption('{"action":"STOP"}', "")
    AdaptionBackend("m",client=fake).generate("system",[{"role":"user","content":"task"}])
    sent=fake.calls[0]
    assert sent['agentic_memory'] is False and sent['store'] is False
    assert sent['previous_response_id'] is None
    assert sent['tools']==[] and sent['tool_choice']=='none'


def test_adaption_failed_response_cannot_execute_an_action():
    from swarmlab.models import AdaptionBackend
    fake = _FakeAdaption('{"action":"STOP"}', "",status="failed")
    with pytest.raises(RuntimeError,match="Adaption response failed"):
        AdaptionBackend("m",client=fake).generate("system",[])
