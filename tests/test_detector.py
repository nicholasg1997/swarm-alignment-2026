import json

from swarmlab.detector import WRITE_TOOLS, Change, diff_digests
from swarmlab.fake_env import FakeTaskEnv


def _patch(path, content):
    return json.dumps({"path": path, "content": content})


def test_run_tests_side_effect_detected():
    env = FakeTaskEnv("f1", "conflicting", sabotage_on_run=True)
    base = env.protected_digest()
    env.run_tests()
    assert diff_digests(base, env.protected_digest(), base) == [
        Change("tests/test_lib.py", "modified")]


def test_revert_detected():
    env = FakeTaskEnv("f1", "conflicting")
    base = env.protected_digest()
    original = env.files["tests/test_lib.py"]
    env.apply_patch(_patch("tests/test_lib.py", "changed"))
    mid = env.protected_digest()
    assert diff_digests(base, mid, base)[0].kind == "modified"
    env.apply_patch(_patch("tests/test_lib.py", original))
    assert diff_digests(mid, env.protected_digest(), base) == [
        Change("tests/test_lib.py", "reverted")]


def test_created_and_deleted():
    assert diff_digests({"t": "ABSENT"}, {"t": "h"}, {"t": "ABSENT"})[0].kind == "created"
    assert diff_digests({"t": "h"}, {"t": "ABSENT"}, {"t": "h"})[0].kind == "deleted"


def test_no_change_no_events():
    assert diff_digests({"t": "h"}, {"t": "h"}, {"t": "h"}) == []


def test_write_tools():
    assert WRITE_TOOLS == ("apply_patch", "run_tests", "restore_tests")
