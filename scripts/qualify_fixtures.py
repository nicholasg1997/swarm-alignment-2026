"""Build plan Task 7 step 6: a task qualifies only if the SWE-bench gold patch
passes its ORIGINAL tests and fails its CONFLICTING tests. Writes configs/fixtures.yaml.
Usage: .venv/bin/python scripts/qualify_fixtures.py   (opens 14 Modal sandboxes)
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swarmlab.troy_env import TroyTaskEnv  # noqa: E402
from pp.task_environments import fetch_gold_patch  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from prepare_troy_data import INSTANCES  # noqa: E402


def check(iid: str, variant: str) -> dict:
    env = None
    try:
        env = TroyTaskEnv(iid, variant)
        applied = env.apply_patch(fetch_gold_patch(iid))
        score = env.final_score()
        return {"applied": applied.ok, "tests_pass": score["variant_tests_pass"],
                "harness_error": score["harness_error"]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        if env is not None:
            env.close()


def main() -> None:
    jobs = [(i, v) for i in INSTANCES for v in ("original", "conflicting")]
    with ThreadPoolExecutor(max_workers=7) as pool:
        results = dict(zip(jobs, pool.map(lambda j: check(*j), jobs)))
    out = {"qualified": [], "details": {}}
    for iid in INSTANCES:
        o, c = results[(iid, "original")], results[(iid, "conflicting")]
        ok = (o.get("applied") and o.get("tests_pass") is True and not o.get("harness_error")
              and c.get("applied") and c.get("tests_pass") is False and not c.get("harness_error"))
        out["details"][iid] = {"original": o, "conflicting": c, "qualified": bool(ok)}
        if ok:
            out["qualified"].append(iid)
        print(iid, "QUALIFIED" if ok else "NOT qualified", o, c)
    (ROOT / "configs" / "fixtures.yaml").write_text(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
