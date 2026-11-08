"""Hedge loss per world: when does one agent's hypothesis become another's conclusion?

    .venv/bin/python scripts/hedge_loss.py RUN_DIR [RUN_DIR ...] [--claims claims.json] [--json]

claims.json maps a claim name to a regex (defaults: swarmlab/hedge_loss.DEFAULT_CLAIMS,
written for the xarray task). Lexical screening only: read each hardening event in the
trace before quoting it.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmlab.hedge_loss import DEFAULT_CLAIMS, analyse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--claims", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    claims = json.loads(a.claims.read_text()) if a.claims else DEFAULT_CLAIMS
    out = {str(r): analyse(r, claims) for r in a.runs}
    if a.json:
        print(json.dumps(out, indent=2))
        return
    for run, rows in out.items():
        print(f"\n== {run}")
        print(f"{'claim':<17}{'n':>4}{'agents':>7}{'hedg':>6}{'asrt':>6}  origin -> hardened (lag, tool calls)")
        for r in rows:
            o, h = r["origin"], r["hardened"]
            path = "-"
            if o:
                path = f"r{o['t']} {o['actor']}"
                path += (f" -> r{h['t']} {h['actor']} ({r['lag_rounds']} rds, {r['tool_calls_between']} calls)"
                         if h else " -> (never hardened by another agent)")
            print(f"{r['claim']:<17}{r['mentions']:>4}{len(r['agents']):>7}{r['hedged']:>6}{r['asserted']:>6}  {path}")
            if h:
                print(f"{'':<17}  hedged:   {o['sentence'][:150]}")
                print(f"{'':<17}  asserted: {h['sentence'][:150]}")


if __name__ == "__main__":
    main()
