"""Justification graph per world: what each test edit's stated reason rests on.

    .venv/bin/python scripts/justification_graph.py RUN_DIR [RUN_DIR ...] [--claim REGEX] [--svg OUT_DIR]

Prints, for every agent that edited a protected test, the post it justified the edit
with and the root posts its citation chain bottoms out in (hedged/asserted). With --svg,
also writes a swimlane figure per world: one lane per agent, x = round, arrows = "cites",
hollow = hedged, filled = asserted, diamonds = protected-test edits. Lexical screen: read
each chain in the trace before quoting it.
"""
import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmlab.hedge_loss import classify
from swarmlab.justification import analyse, roots

DEFAULT_CLAIM = r"misplac|one block too early|after the CustomIndexable|normaliz|first_orig|inject|planted|tamper"

# Reference palette (dataviz skill), light mode: text tokens + slot 1 (blue) + slot 2 (orange).
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def svg(res: dict, title: str) -> str:
    by_id = res["_by_id"]
    traced: set[str] = set()
    for c in res["edits"]:
        if c["justification"]:
            traced |= roots(c["justification"]["msg_id"], by_id)[1]
    edits = res["edits"]
    if not edits:
        return ""
    ts = [by_id[m].t for m in traced] + [e["t"] for e in edits]
    t0, t1 = min(ts) - 4, max(ts) + 4
    agents = sorted({p.actor for p in by_id.values()} | {e["actor"] for e in edits})
    W, L, R, T, LANE = 900, 54, 24, 70, 46
    H = T + LANE * len(agents) + 46
    x = lambda t: L + (t - t0) / max(1, t1 - t0) * (W - L - R)
    y = {a: T + LANE * i + LANE / 2 for i, a in enumerate(agents)}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         '<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="{INK2}"/></marker></defs>',
         f'<text x="{L}" y="22" font-size="14" font-weight="600" fill="{INK}">{html.escape(title)}</text>']
    # legend
    lx, ly = L, 44
    o += [f'<circle cx="{lx+5}" cy="{ly}" r="5" fill="{SURF}" stroke="{BLUE}" stroke-width="2"/>',
          f'<text x="{lx+16}" y="{ly+4}" font-size="11" fill="{INK2}">hedged post</text>',
          f'<circle cx="{lx+105}" cy="{ly}" r="5" fill="{BLUE}"/>',
          f'<text x="{lx+116}" y="{ly+4}" font-size="11" fill="{INK2}">asserted post</text>',
          f'<path d="M{lx+215},{ly-6} l6,6 l-6,6 l-6,-6 z" fill="{ORANGE}"/>',
          f'<text x="{lx+228}" y="{ly+4}" font-size="11" fill="{INK2}">protected-test edit</text>',
          f'<line x1="{lx+345}" y1="{ly}" x2="{lx+375}" y2="{ly}" stroke="{INK2}" stroke-width="1.2" marker-end="url(#ah)"/>',
          f'<text x="{lx+382}" y="{ly+4}" font-size="11" fill="{INK2}">cites as its reason</text>',
          f'<circle cx="{lx+505}" cy="{ly}" r="3" fill="{MUTED}"/>',
          f'<text x="{lx+514}" y="{ly+4}" font-size="11" fill="{INK2}">other post</text>']
    # lanes + round axis
    for a in agents:
        o.append(f'<line x1="{L}" y1="{y[a]}" x2="{W-R}" y2="{y[a]}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{L-10}" y="{y[a]+4}" font-size="12" text-anchor="end" fill="{INK2}">{a}</text>')
    step = 10 if t1 - t0 > 40 else 5
    for t in range((t0 // step + 1) * step, t1 + 1, step):
        o.append(f'<text x="{x(t):.1f}" y="{H-22}" font-size="10" text-anchor="middle" fill="{INK2}">{t}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-6}" font-size="11" text-anchor="middle" fill="{INK2}">round</text>')
    # citation edges (traced subgraph only), curved
    for m in traced:
        p = by_id[m]
        for c in p.cites:
            q = by_id[c]
            x1, y1, x2, y2 = x(p.t), y[p.actor], x(q.t), y[q.actor]
            bend = -18 if y1 == y2 else 0
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + bend
            # Start and stop outside the dots (radius ~6 + ring), along the curve's end tangents.
            def pull(ax, ay, bx, by, d=9.0):
                dx, dy = bx - ax, by - ay
                n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                return ax + dx / n * d, ay + dy / n * d
            x1, y1 = pull(x1, y1, mx, my)
            x2, y2 = pull(x2, y2, mx, my, 11.0)
            o.append(f'<path d="M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}" fill="none" '
                     f'stroke="{INK2}" stroke-width="1.2" stroke-opacity="0.75" marker-end="url(#ah)"/>')
    # posts
    for p in by_id.values():
        if not (t0 <= p.t <= t1):
            continue
        cx, cy = x(p.t), y[p.actor]
        tip = html.escape(f"r{p.t} {p.actor} {p.msg_id}: {p.clean[:200]}")
        if p.msg_id in traced:
            if classify(p.clean) == "hedged":
                o.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="5.5" fill="{SURF}" stroke="{BLUE}" stroke-width="2.2"><title>{tip}</title></circle>')
            else:
                o.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="5.5" fill="{BLUE}" stroke="{SURF}" stroke-width="2"><title>{tip}</title></circle>')
        else:
            o.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="2.6" fill="{MUTED}"><title>{tip}</title></circle>')
    # edits
    for e in edits:
        cx, cy = x(e["t"]), y[e["actor"]] + 13
        o.append(f'<path d="M{cx:.1f},{cy-7} l7,7 l-7,7 l-7,-7 z" fill="{ORANGE}" stroke="{SURF}" stroke-width="1.5">'
                 f'<title>r{e["t"]} {e["actor"]} edited {html.escape(str(e["path"]))}</title></path>')
    # label the most-shared non-own root
    if res["shared_roots"]:
        m = max(res["shared_roots"], key=res["shared_roots"].get)
        p = by_id[m]
        o.append(f'<text x="{x(p.t)-4:.1f}" y="{y[p.actor]+22}" font-size="11" fill="{INK}">'
                 f'r{p.t} {p.actor}: hedged root of {res["shared_roots"][m]} editors\' chains</text>')
    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--claim", default=DEFAULT_CLAIM)
    ap.add_argument("--svg", type=Path)
    a = ap.parse_args()
    for run in a.runs:
        r = analyse(run, a.claim)
        print(f"\n== {run}\nposts {r['posts']}, citing a teammate {r['posts_citing']}, "
              f"citation circles {r['circles']}, roots shared by >1 editor {r['shared_roots']}")
        for c in r["edits"]:
            j = c["justification"]
            if not j:
                print(f"  {c['actor']} edit r{c['t']}: no post near the edit")
                continue
            how = ("cites teammates" if c["cites_teammate"] else "") + (" + follows the group" if c["follows_group"] else "")
            print(f"  {c['actor']} edit r{c['t']}: reason {j['msg_id']} (r{j['t']}) {how or 'cites no one'}; chain {c['depth_posts']} posts")
            for x in c["roots"]:
                print(f"     root r{x['t']} {x['actor']}{' (own)' if x['own'] else ''} [{x['stance']}]: {x['text'][:100]}")
        if a.svg and r["edits"]:
            a.svg.mkdir(parents=True, exist_ok=True)
            out = a.svg / f"{run.name}.svg"
            out.write_text(svg(r, f"Who cited whom before each test edit — {run.name.split('-')[0]}"))
            print(f"  figure: {out}")


if __name__ == "__main__":
    main()
