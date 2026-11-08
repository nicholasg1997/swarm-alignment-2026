"""Live Modal tests (slow, cost cents). Run with SWARMLAB_LIVE=1."""
import os

import pytest

from swarmlab.detector import diff_digests

pytestmark = pytest.mark.skipif(os.environ.get("SWARMLAB_LIVE") != "1",
                                reason="set SWARMLAB_LIVE=1 to run live Modal tests")
TASK = "pytest-dev__pytest-10051"
TEST_FILE = "testing/logging/test_fixture.py"



def test_deliberate_protected_edit_detected():
    from swarmlab.troy_env import TroyTaskEnv
    env = TroyTaskEnv(TASK, "conflicting")
    try:
        assert "pytest" in env.task_prompt().lower()
        base = env.protected_digest()
        assert TEST_FILE in base and base[TEST_FILE] != "ABSENT"
        assert "import logging" in env.read_file(TEST_FILE, 1, 1).observation
        res = env.apply_patch(f"--- a/{TEST_FILE}\n+++ b/{TEST_FILE}\n@@ -1,1 +1,2 @@\n"
                              "+# edited by swarmlab live test\n import logging\n")
        assert res.ok, res.observation
        changes = diff_digests(base, env.protected_digest(), base)
        assert [c.kind for c in changes] == ["modified"], res.observation
    finally:
        env.close()


def test_original_variant_reference_fix_passes():
    from pp.task_environments import fetch_gold_patch
    from swarmlab.troy_env import TroyTaskEnv
    env = TroyTaskEnv(TASK, "original")
    try:
        env.apply_patch(fetch_gold_patch(TASK))
        assert env.final_score()["original_tests_pass"] is True
    finally:
        env.close()
