"""Doubt before vs after the first green, per world (claim C16).

    .venv/bin/python scripts/green_dissent.py RUN_DIR [RUN_DIR ...] [--window 30] [--json] [--hits]

--hits prints the doubt posts after green so they can be read before quoting.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmlab.green_dissent import analyse, is_doubt, load


def pct(x):
    return "  -  " if x is None else f"{100 * x:4.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--hits", action="store_true")
    a = ap.parse_args()
    rows = [analyse(r, a.window) for r in a.runs if (r / "events.jsonl").exists()]
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    print(f"{'world':44} {'green':>5} {'edit':>5} | public doubt  before→after | private doubt before→after | rechecks before→after (agents)")
    for r in rows:
        if r["first_green"] is None:
            print(f"{r['run'][:44]:44} {'none':>5} {str(r['first_edit']):>5} | no green")
            continue
        b, f = r["before"], r["after"]
        print(f"{r['run'][:44]:44} {r['first_green']:>5} {str(r['first_edit']):>5} | "
              f"{b['doubt_posts']:>2}/{b['posts']:<3}{pct(b['doubt_share_public'])} → {f['doubt_posts']:>2}/{f['posts']:<3}{pct(f['doubt_share_public'])} | "
              f"{pct(b['doubt_share_private'])} → {pct(f['doubt_share_private'])}             | "
              f"{b['rechecks']:>2} ({len(b['recheck_agents'])}) → {f['rechecks']:>2} ({len(f['recheck_agents'])})")
        if a.hits:
            for e in load(next(p for p in a.runs if p.name == r["run"])):
                if e["event"] == "board_post" and r["first_green"] < e["t"] <= r["first_green"] + a.window \
                        and is_doubt(e["payload"].get("text", "")):
                    print(f"      r{e['t']} {e['actor']}: {e['payload']['text'][:220]}")


if __name__ == "__main__":
    main()
