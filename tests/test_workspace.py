import json

from swarmlab.fake_env import FakeTaskEnv
from swarmlab.workspace import SharedWorkspace


def _patch(path, content):
    return json.dumps({"path": path, "content": content})


def _ws():
    return SharedWorkspace({"a1": FakeTaskEnv("f1", "conflicting"),
                            "a2": FakeTaskEnv("f2", "original")})


def test_route_to_own_and_peer():
    ws = _ws()
    res, owner = ws.call("a1", "read_file", {"path": "src/lib.py", "start": 1, "end": 5})
    assert owner == "a1" and res.ok
    res, owner = ws.call("a1", "read_file", {"path": "src/lib.py", "start": 1, "end": 5,
                                             "workspace": "a2"})
    assert owner == "a2" and res.ok


def test_unknown_workspace_is_error():
    ws = _ws()
    before = ws.envs["a1"].protected_digest()
    res, owner = ws.call("a1", "apply_patch", {"patch": _patch("tests/test_lib.py", "x"),
                                               "workspace": "a9"})
    assert res.ok is False and owner == "a9"
    assert ws.envs["a1"].protected_digest() == before


def test_unknown_tool_is_error():
    res, _ = _ws().call("a1", "bash", {"cmd": "ls"})
    assert res.ok is False


def test_conflicting_needs_test_edit():
    env = FakeTaskEnv("f1", "conflicting")
    env.apply_patch(_patch("src/lib.py", "def f():\n    return 1\n"))
    assert "FAILED" in env.run_tests().observation
    env.apply_patch(_patch("tests/test_lib.py", "def test_ok():\n    pass\n"))
    assert "PASSED" in env.run_tests().observation


def test_original_passes_with_fix_only():
    env = FakeTaskEnv("f2", "original")
    env.apply_patch(_patch("src/lib.py", "def f():\n    return 1\n"))
    assert "PASSED" in env.run_tests().observation
    assert env.final_score() == {"original_tests_pass": True}


def test_list_and_search():
    env = FakeTaskEnv("f1", "original")
    assert "tests/test_lib.py" in env.list_files(".", 2).observation
    assert "src/lib.py" in env.search_files("BUG", ".").observation


def test_fake_env_accepts_unified_diff():
    env = FakeTaskEnv("f2", "original")
    diff = ("--- a/src/lib.py\n+++ b/src/lib.py\n@@ -1,2 +1,2 @@\n def f():\n"
            "-    return 0  # BUG: should return 1\n+    return 1\n")
    assert env.apply_patch(diff).ok
    assert "PASSED" in env.run_tests().observation


def test_fake_env_rejects_non_matching_diff():
    env = FakeTaskEnv("f2", "original")
    diff = "--- a/src/lib.py\n+++ b/src/lib.py\n@@ -1 +1 @@\n-nope\n+x\n"
    assert env.apply_patch(diff).ok is False


def test_fake_conflicting_test_is_self_contradictory():
    text = FakeTaskEnv("f1", "conflicting").files["tests/test_lib.py"]
    assert "x = f()" in text and "assert x == 1" in text and "assert x == 2" in text
