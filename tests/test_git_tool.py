import json
import subprocess
from types import SimpleNamespace

import pytest

from swarmlab.git_tool import execute_git, git_argv, git_script
from swarmlab.troy_env import TroyTaskEnv
from swarmlab.workspace import SharedWorkspace


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout
    git("init", "-q")
    (tmp_path / "code.py").write_text("original\n")
    git("add", "code.py")
    git("-c", "user.name=Maintainer", "-c", "user.email=m@example.test",
        "commit", "-qm", "Initial source")
    (tmp_path / "code.py").write_text("changed\n")
    return tmp_path


@pytest.mark.parametrize("command,args,expected", [
    ("log", ["--oneline", "-n", "1"], "Initial source"),
    ("show", ["HEAD:code.py"], "original"),
    ("diff", ["--", "code.py"], "+changed"),
])
def test_real_history_and_diff_leave_repo_unchanged(repo, command, args, expected):
    before = {p.relative_to(repo): p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    text, ok = execute_git(str(repo), git_argv(command, args))
    assert ok and expected in text
    assert before == {p.relative_to(repo): p.read_bytes() for p in repo.rglob("*") if p.is_file()}


@pytest.mark.parametrize("command,args", [
    ("commit", []), ("reset", ["--hard"]),
    ("log", ["--output=written"]), ("show", ["--output", "written"]),
    ("diff", ["--ext-diff"]), ("show", ["--textconv"]),
    ("diff", ["--no-index", "/etc/passwd", "code.py"]),
    ("log", ["-c", "alias.foo=!echo bad"]),
    ("log", ["--git-dir=/tmp/other"]),
    ("diff", ["--", "../outside"]), ("diff", ["--", "/etc/passwd"]),
    ("log", "--oneline"), ("show", [None]),
])
def test_rejects_mutations_external_helpers_and_bad_arguments(command, args):
    with pytest.raises(ValueError):
        git_argv(command, args)


def test_optional_value_cannot_turn_into_an_output_option(repo):
    text, _ = execute_git(str(repo), git_argv("log", ["--pretty", "--output=written"]))
    assert not (repo / "written").exists()
    assert "--pretty=--output=written" in git_argv("log", ["--pretty", "--output=written"])


def test_disables_repository_diff_helpers(repo):
    marker = repo / "helper-ran"
    helper = repo / "helper.sh"
    helper.write_text(f"#!/bin/sh\ntouch '{marker}'\necho external-output\n")
    helper.chmod(0o755)
    subprocess.run(["git", "config", "diff.external", str(helper)], cwd=repo, check=True)
    subprocess.run(["git", "config", "diff.custom.textconv", str(helper)], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.fsmonitor", str(helper)], cwd=repo, check=True)
    (repo / ".gitattributes").write_text("*.py diff=custom\n")
    text, ok = execute_git(str(repo), git_argv("diff"))
    assert ok and "+changed" in text and not marker.exists()


def test_remote_script_quotes_arguments_and_bounds_output(repo):
    payload = "%s $(touch should-not-exist); `touch also-not`"
    proc = subprocess.run(["bash", "-c", git_script(str(repo), "log", ["--format=" + payload])],
                          capture_output=True, text=True, check=True)
    assert "Initial source $(touch should-not-exist); `touch also-not`" in proc.stdout
    assert not (repo / "should-not-exist").exists()
    assert not (repo / "also-not").exists()
    (repo / "code.py").write_text("x" * 40000)
    text, ok = execute_git(str(repo), git_argv("diff"))
    assert ok and "Output truncated" in text and len(text) < 20200


def test_troy_adapter_and_workspace_route_to_requested_repository(repo):
    calls = []
    def run(phase, script, **kwargs):
        calls.append((phase, kwargs))
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        return dict(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    env = object.__new__(TroyTaskEnv)
    env.env = SimpleNamespace(session=SimpleNamespace(run=run),
                              spec=SimpleNamespace(repo_dir=str(repo)))
    ws = SharedWorkspace({"a1": object(), "a2": env}, git_enabled=True)
    result, owner = ws.call("a1", "git", {"command": "show", "args": ["HEAD:code.py"],
                                           "workspace": "a2"})
    assert result.ok and "original" in result.observation and owner == "a2"
    assert calls == [("read", {"capture": 24000})]
    blocked, _ = ws.call("a1", "git", {"command": "reset", "workspace": "a2"})
    assert not blocked.ok and len(calls) == 1


def test_fixtures_do_not_invent_history_or_expose_host_repo():
    from swarmlab.assessment import AssessmentEnv
    from swarmlab.fake_env import FakeTaskEnv
    for env in (FakeTaskEnv("f", "original"), AssessmentEnv()):
        result = env.git("log", ["--oneline"])
        assert not result.ok and "no Git repository" in result.observation


def test_git_tool_is_off_by_default_and_on_when_configured():
    from swarmlab.conditions import WorldConfig
    from swarmlab.prompts import GIT_TOOL_LINE, build_system_prompt, with_git_tool
    from swarmlab.troy import parse_action
    base = dict(run_id="git", condition="C1", block=0, seed=0, model="m", focal_slot=0,
                task_ids=[f"f{i}" for i in range(6)])
    off, on = WorldConfig(**base), WorldConfig(**base, git_tool=True)
    for role in ("peer", "focal", "angel", "devil"):
        assert GIT_TOOL_LINE not in build_system_prompt(off, "a1", role)
        assert GIT_TOOL_LINE in build_system_prompt(on, "a1", role)
    assert with_git_tool(with_git_tool("old prompt")).count(GIT_TOOL_LINE) == 1
    action = parse_action(json.dumps({"action": "USE_ALLOWED_TOOL", "tool": "git",
                                     "arguments": {"command": "log", "args": ["--oneline"]}}))
    assert action.tool == "git"


def test_disabled_workspace_rejects_git_with_the_old_tool_list():
    from swarmlab.fake_env import FakeTaskEnv
    ws = SharedWorkspace({"a1": FakeTaskEnv("f", "original")})
    result, _ = ws.call("a1", "git", {"command": "log"})
    assert not result.ok and "unknown tool 'git'" in result.observation
    assert result.observation.endswith(
        "read_file, list_files, search_files, run_tests, apply_patch, restore_tests")


def test_world_prompts_follow_the_flag(tmp_path):
    from swarmlab.conditions import Slot, WorldConfig
    from swarmlab.fake_env import FakeTaskEnv
    from swarmlab.models import make_backend_factory
    from swarmlab.prompts import GIT_TOOL_LINE
    from swarmlab.world import run_world
    factory = make_backend_factory({"backend": "scripted", "scripts": {}})
    for flag in (False, True):
        cfg = WorldConfig(run_id=f"g{flag}", condition="G", block=0, seed=0, model="scripted",
                          task_ids=["t"], focal_slot=0, max_turns_per_agent=1, git_tool=flag,
                          board=False, slots=(Slot("t", "conflicting", "peer"),))
        run = run_world(cfg, factory, FakeTaskEnv, tmp_path / str(flag)).run_dir
        manifest = json.loads((run / "run.json").read_text())
        assert (GIT_TOOL_LINE in manifest["system_prompts"]["a1"]) is flag
        assert manifest["git_tool"] is flag


def test_a_one_agent_world_is_not_told_about_other_agents():
    from swarmlab.conditions import Slot, WorldConfig
    from swarmlab.prompts import build_system_prompt
    solo = WorldConfig(run_id="s", condition="SOLO", block=0, seed=0, model="m", task_ids=["t"],
                       focal_slot=0, board=False, scope_rule="soft",
                       slots=(Slot("t", "conflicting", "peer"),))
    prompt = build_system_prompt(solo, "a1", "peer")
    for leak in ("Other agents", "board", "workspace", "team chat"):
        assert leak not in prompt
    assert "no one is monitoring you while you work" in prompt  # rest of the autonomy line stays
    assert "ESCALATE_TO_HUMAN" in prompt and "high priority for the next release" in prompt


def test_a_team_without_chat_is_not_told_about_a_board():
    from swarmlab.conditions import Slot, WorldConfig
    from swarmlab.prompts import build_system_prompt
    slots = tuple(Slot("t", "conflicting", "peer", "project") for _ in range(6))
    cfg = WorldConfig(run_id="n", condition="SHARED_NOCHAT", block=0, seed=0, model="m",
                      task_ids=["t"] * 6, focal_slot=0, board=False, scope_rule="soft", slots=slots)
    prompt = build_system_prompt(cfg, "a1", "peer")
    assert "board" not in prompt and "chat" not in prompt
    assert "Other agents are working in the same project" in prompt
    assert "see each other's changes" in prompt
    assert "Your agent ID is a1." in prompt


def test_a_lone_agent_with_git_is_not_told_about_workspaces():
    from swarmlab.conditions import Slot, WorldConfig
    from swarmlab.prompts import build_system_prompt
    cfg = WorldConfig(run_id="s", condition="SOLO_GIT", block=0, seed=0, model="m", task_ids=["t"],
                      focal_slot=0, board=False, scope_rule="soft", git_tool=True,
                      slots=(Slot("t", "conflicting", "peer"),))
    prompt = build_system_prompt(cfg, "a1", "peer")
    assert '"tool":"git"' in prompt and "workspace" not in prompt
