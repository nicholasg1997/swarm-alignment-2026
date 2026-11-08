"""One-screen progress check for running worlds (used for check-ins on long runs).

    python scripts/progress.py runs/real/PR-... [runs/real/DV-...]

Per agent: turns taken, patches (applied / attempted), test runs (and the last result), chat posts, and two
stall signals over its last 10 tool calls: how many were distinct (low = re-reading the
same things) and rounds since its last patch. To end a stalled world cleanly, keeping
its scores: touch <run_dir>/STOP.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarise(run_dir: Path) -> str:
    ev = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l]
    man = json.loads((run_dir / "run.json").read_text())
    rnd = ev[-1]["t"] if ev else 0
    ok_patches = defaultdict(int)
    acts, calls, posts, patch_t, stopped, last_test = (defaultdict(int), defaultdict(list),
                                                       defaultdict(int), {}, set(), {})
    for e in ev:
        a = e.get("actor")
        if e["event"] == "action":
            acts[a] += 1
            stopped.discard(a)
            if '"STOP"' in (e.get("raw_response") or ""):
                stopped.add(a)
        elif e["event"] == "tool_call":
            p = e["payload"]
            calls[a].append(json.dumps([p.get("tool"), p.get("arguments")], sort_keys=True))
            if p.get("tool") == "apply_patch":
                patch_t[a] = e["t"]
        elif e["event"] == "tool_result" and e["payload"].get("tool") == "apply_patch":
            ok_patches[a] += e.get("exit_status") == "ok"
        elif e["event"] == "tool_result" and e["payload"].get("tool") == "run_tests":
            last_test[a] = e["payload"].get("observation", "")[:40].replace("\n", " ")
        elif e["event"] == "board_post":
            posts[a] += 1
        elif e["event"] == "wake":
            stopped.discard(a)
    end = next((e["payload"]["reason"] for e in ev if e["event"] == "terminate"), None)
    lines = [f"{run_dir.name}  round {rnd}  {'ENDED: ' + end if end else 'running'}  "
             f"posts {sum(posts.values())}  wakes {sum(e['event'] == 'wake' for e in ev)}"]
    for a, role in man["roles"].items():
        last10 = calls[a][-10:]
        since = rnd - patch_t[a] if a in patch_t else None
        lines.append(
            f"  {a} {role:6} turns {acts[a]:3}  patches ok/tried {ok_patches[a]:2}/{sum('apply_patch' in c for c in calls[a]):<2}"
            f"  tests {sum('run_tests' in c for c in calls[a]):2}  posts {posts[a]:2}"
            f"  distinct/last10 {len(set(last10)):2}  since-patch {since if since is not None else '-':>4}"
            f"  {'STOPPED' if a in stopped else ''}  last test: {last_test.get(a, '-')}")
    return "\n".join(lines)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(summarise(Path(d)))
