from swarmlab.troy_env import normalize_search_path

import pytest


def test_repo_root_spellings_map_to_empty():
    # Troy's search turns "." into an empty grep target and silently finds nothing
    for p in (".", "./", "/", "", "  .  "):
        assert normalize_search_path(p) == ""


def test_real_paths_untouched():
    assert normalize_search_path("testing/logging") == "testing/logging"


def test_sandbox_lifetime_is_passed_to_troy(monkeypatch):
    """Troy's default sandbox lifetime (1800 s) killed 34-minute worlds."""
    from swarmlab import troy_env
    seen = {}

    class FakeSession:
        def run(self, *a, **k):
            return {}

    class FakeEnv:
        def __init__(self):
            self.session = FakeSession()

        def reset(self):
            return "prompt"

    def for_task(task_id, **kw):
        seen.update(kw)
        return FakeEnv()
    monkeypatch.setattr(troy_env.SweBenchPerTaskSandboxEnv, "for_task", staticmethod(for_task))
    troy_env.TroyTaskEnv("t", "original", timeout_s=9000)
    assert seen["timeout_s"] == 9000
    troy_env.TroyTaskEnv("t", "original")
    assert seen["timeout_s"] >= 3600 + 1800


def test_cli_sandbox_outlives_wall_clock():
    from swarmlab.__main__ import _sandbox_timeout
    assert _sandbox_timeout(3600) == 3600 + 1800
    assert _sandbox_timeout(4 * 3600) == 4 * 3600 + 1800


# --- a rejected patch says which format is expected (Luna wrote "*** Begin Patch") ---
V4A = "*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 2\n*** End Patch"
REJECTED = "apply_patch ok=False: PP_APPLY_RC=128\n\nerror: unrecognized input"


def test_unrecognised_patch_gets_format_hint():
    from swarmlab.troy_env import patch_hint
    h = patch_hint(V4A, REJECTED)
    assert "unified diff" in h and "--- a/" in h and "*** Begin Patch" in h


def test_corrupt_unified_diff_gets_generic_hint():
    from swarmlab.troy_env import patch_hint
    h = patch_hint("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b", "error: corrupt patch at line 5")
    assert "unified diff" in h and "*** Begin Patch" not in h


def test_other_failures_get_no_hint():
    from swarmlab.troy_env import patch_hint
    assert patch_hint("--- a/x.py", "error: patch failed: x.py:3\nerror: x.py: patch does not apply") == ""
    assert patch_hint(V4A, "apply_patch ok=True") == ""


# --- hunk line counts are recomputed from the body (git apply --recount), any model ---
MISCOUNTED = ("--- a/m.py\n+++ b/m.py\n@@ -211,7 +211,7 @@ def _cstack(left, right):\n"
              "     else:\n         c = 0\n-        c = 1\n+        c = right\n")


def test_recount_fixes_hunk_header_counts():
    from swarmlab.troy_env import recount_hunks
    out = recount_hunks(MISCOUNTED)
    assert "@@ -211,3 +211,3 @@ def _cstack(left, right):" in out
    assert out.endswith("+        c = right\n")


def test_recount_handles_several_hunks_and_files_and_missing_newline():
    from swarmlab.troy_env import recount_hunks
    p = ("--- a/x.py\n+++ b/x.py\n@@ -1,9 +1,9 @@\n a\n-b\n+B\n@@ -10 +10,4 @@\n c\n+d\n"
         "--- a/y.py\n+++ b/y.py\n@@ -5,2 +5 @@\n-e\n\\ No newline at end of file")
    out = recount_hunks(p)
    assert "@@ -1,2 +1,2 @@" in out and "@@ -10,1 +10,2 @@" in out and "@@ -5,1 +5,0 @@" in out
    assert out.endswith("\\ No newline at end of file\n")


def test_recount_leaves_non_diffs_alone():
    from swarmlab.troy_env import recount_hunks
    assert recount_hunks(V4A) == V4A
    assert recount_hunks("") == ""


def test_recounted_patch_applies_with_real_git(tmp_path):
    import shutil
    import subprocess
    import pytest
    if not shutil.which("git"):
        pytest.skip("git not installed")
    from swarmlab.troy_env import recount_hunks
    (tmp_path / "m.py").write_text("def f():\n    if x:\n        c = 0\n    else:\n"
                                   "        c = 1\n    return c\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    bad = ("--- a/m.py\n+++ b/m.py\n@@ -3,9 +3,9 @@\n         c = 0\n     else:\n"
           "-        c = 1\n+        c = 2\n     return c\n")  # header miscounted: 9 vs 4
    run = lambda p: subprocess.run(["git", "apply", "--check", "-"], cwd=tmp_path,
                                   input=p, text=True, capture_output=True)
    assert run(bad).returncode != 0          # what Troy's git saw
    assert run(recount_hunks(bad)).returncode == 0


@pytest.mark.parametrize("stdout", ["", "PP_FT_COUNT=0\n", "PP_FT_COUNT=2\n",
                                    "PP_FT_0=broken  tests/t.py\nPP_FT_COUNT=1\n"])
def test_bad_fingerprint_fails_instead_of_recording_a_clean_detector(monkeypatch, stdout):
    from types import SimpleNamespace
    from swarmlab import troy_env

    env = troy_env.TroyTaskEnv.__new__(troy_env.TroyTaskEnv)
    env.variant = "original"
    env.env = SimpleNamespace(spec=None, session=SimpleNamespace(
        run=lambda *args, **kwargs: {"stdout": stdout}))
    monkeypatch.setattr(troy_env, "frozen_tests_digest_script", lambda *args: "fingerprint")
    with pytest.raises(RuntimeError, match="fingerprint"):
        env.protected_digest()


def test_complete_fingerprint_accepts_missing_optional_files(monkeypatch):
    from types import SimpleNamespace
    from swarmlab import troy_env

    digest = "a" * 64
    env = troy_env.TroyTaskEnv.__new__(troy_env.TroyTaskEnv)
    env.variant = "original"
    env.env = SimpleNamespace(spec=None, session=SimpleNamespace(run=lambda *a, **k: {
        "stdout": f"PP_FT_0={digest}  tests/t.py\nPP_FT_1=ABSENT  conftest.py\nPP_FT_COUNT=2\n"}))
    monkeypatch.setattr(troy_env, "frozen_tests_digest_script", lambda *args: "fingerprint")
    assert env.protected_digest() == {"tests/t.py": digest, "conftest.py": "ABSENT"}
