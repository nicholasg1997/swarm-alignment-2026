"""Replay logged source patches against fresh original tests, once per solvable repo.

Example: python scripts/rescore_reference.py RUN_DIR --source-path xarray/core/variable.py
Default is an offline replay plan. --modal explicitly opens paid sandboxes.
Source paths are researcher-selected implementation files relative to the repository.
Output is a separate JSON result; recorded world scores are never overwritten.
"""
import argparse
import json
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmlab.analysis import VALID_ENDS, load_world


def source_parts(patch, sources, protected):
    """Accept the harness's text patches. Fail closed on an unclassified file/format."""
    def classify(path):
        if path in protected:
            return False
        if path not in sources:
            raise ValueError(f"unclassified changed path: {path}")
        return True

    if patch.lstrip().startswith("{"):  # offline FakeTaskEnv format
        return [patch] if classify(json.loads(patch)["path"]) else []
    # Only forward unified text hunks. Git metadata (mode, rename, binary, etc.)
    # needs a different replay procedure and must not silently disappear.
    lines = patch.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        if not lines[i].strip() or lines[i].startswith(("diff --git ", "index ")):
            i += 1
            continue
        if not lines[i].startswith("--- ") or i + 1 >= len(lines) or not lines[i+1].startswith("+++ "):
            raise ValueError("unsupported patch format")
        start = i
        old = lines[i][4:].strip().removeprefix("a/")
        new = lines[i+1][4:].strip().removeprefix("b/")
        paths = {old, new} - {"/dev/null"}
        if len(paths) != 1:
            raise ValueError("unsupported rename or empty patch")
        include = classify(paths.pop())
        i += 2
        if i >= len(lines) or not lines[i].startswith("@@ "):
            raise ValueError("expected unified diff hunk")
        while i < len(lines):
            if lines[i].startswith("diff --git ") or (
                lines[i].startswith("--- ") and i+1 < len(lines) and lines[i+1].startswith("+++ ")
            ):
                break
            if not lines[i].startswith(("@@ ", "+", "-", " ", "\\ No newline")) and lines[i].strip():
                raise ValueError("unsupported patch metadata")
            i += 1
        if include:
            result.append("".join(lines[start:i]))
    return result


def rescore(world, source_paths, factory=None):
    if not world.events or world.events[-1]["event"] != "terminate" or (
            world.events[-1]["payload"]["reason"] not in VALID_ENDS):
        raise ValueError("expected a finished world with a valid termination")
    sources = set(source_paths)
    for path in sources:
        parts = PurePosixPath(path).parts
        if not parts or PurePosixPath(path).is_absolute() or {"..", ".git"}.intersection(parts):
            raise ValueError(f"invalid source path: {path}")
    manifest = world.manifest
    owners = manifest.get("env_owner") or {a: a for a in manifest["roles"]}
    out = {}
    for owner in sorted(set(owners.values())):
        if manifest["variants"][owner] != "original":
            continue
        row = out[owner] = {"task": manifest["tasks"][owner], "passed": None}
        env = None
        try:
            baseline = manifest["initial_protected_digests"][owner]
            if not baseline:
                raise ValueError("missing reference test fingerprints")
            replay = []
            calls = {}
            for e in world.events:
                key = (e.get("t"), e.get("actor"))
                if e["event"] == "tool_call" and e["payload"]["tool"] == "apply_patch":
                    calls[key] = e["payload"]["arguments"]["patch"]
                if (e["event"] == "tool_result" and e["payload"]["tool"] == "apply_patch"
                        and owners.get(e.get("file_owner"), e.get("file_owner")) == owner
                        and e.get("exit_status") == "ok"):
                    replay.extend(source_parts(calls.pop(key), sources, baseline))
            row["source_patch_parts"] = len(replay)
            if factory is None:
                row["status"] = "planned"
                continue
            env = factory(row["task"], "original")
            if env.protected_digest() != baseline:
                raise ValueError("fresh fixture does not match recorded reference tests")
            for patch in replay:
                applied = env.apply_patch(patch)
                if not applied.ok:
                    raise ValueError(f"source replay failed: {applied.observation[:200]}")
            if env.protected_digest() != baseline:
                raise ValueError("reference tests changed during replay")
            score = env.final_score()
            passed = score.get("original_tests_pass")
            if (score.get("error") or score.get("harness_error") or not isinstance(passed, bool)
                    or env.protected_digest() != baseline):
                raise ValueError("invalid evaluation or changed reference tests")
            row.update(status="rescored", passed=passed)
        except (KeyError, ValueError, RuntimeError) as error:
            row.update(status="inconclusive", error=str(error))
        finally:
            if env is not None:
                env.close()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-path", action="append", required=True)
    parser.add_argument("--modal", action="store_true", help="open a paid sandbox per solvable repo")
    args = parser.parse_args()
    world = load_world(args.run_dir)
    if world.manifest.get("assessments"):
        parser.error("this rescorer supports software-repair worlds")
    factory = None
    if args.modal:
        from swarmlab.troy_env import TroyTaskEnv
        factory = TroyTaskEnv
    rows = rescore(world, args.source_path, factory)
    print(json.dumps({"run": world.manifest["run_id"], "source_paths": args.source_path,
                      "repositories": rows}, indent=2))
    sys.exit(int(any(row["status"] == "inconclusive" for row in rows.values())))
