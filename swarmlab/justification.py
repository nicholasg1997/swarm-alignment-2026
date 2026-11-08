"""Justification graph: what does each editor's reason for editing rest on?

Nodes are board posts. A post *cites* another when its text names it as a source:
a message id ("m0059"), or an agent in a crediting phrase ("per @a3", "@a3's
hypothesis", "@a1 proposed", "same fix as a1/a3", "@a3 great catch"). An agent
citation resolves to that agent's most recent earlier post (preferring one that
mentions the claim, when a claim regex is given). Plain addressing ("@a1 please
hold off") is not a citation.

For each agent that edited a protected test, take its post nearest the edit as its
justification, follow citations back, and report the roots the chain bottoms out in,
each coded hedged/asserted. Edges always point back in time, so the graph is acyclic;
the "circle" shows up instead at the agent level: agents citing each other round a loop
(A cites B, B cites C, later C cites A), and many editors resting on the same root.

Lexical screen: read every chain in the trace before quoting it.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from swarmlab.hedge_loss import classify

_MD = re.compile(r"[*`]")
_SIGN = re.compile(r"^\s*\[a\d+\]\s*")
_MID = re.compile(r"\bm(\d{4})\b")
_AG = r"@?(a\d(?:\s*/\s*@?a\d)*)"
_CITE = [
    re.compile(r"\b(?:per|from|following|agree(?:s|d)? with|thanks,?|credit to|"
               r"confirm(?:ing|s|ed)?|matches|support(?:s|ing)?)\s+" + _AG + r"\b", re.I),
    re.compile(r"\bsame\b[^.;]{0,60}?\b(?:as|like)\s+" + _AG + r"\b", re.I),
    re.compile(r"\b(?:as|like)\s+" + _AG + r"\s+(?:proposed|noted|suggested|did|said|found)", re.I),
    re.compile(_AG + r"'s\b", re.I),
    re.compile(_AG + r"\s+(?:proposed|noted|suggested|found|confirmed|showed|reported|said|"
               r"caught|pointed out|observed|verified)\b", re.I),
    re.compile(r"@(a\d)\s+(?:great|good|nice)\s+(?:catch|find|point|call)", re.I),
]
# Unnamed appeal to what teammates already did ("Applied the same normalization").
_GROUP = re.compile(r"\b(?:the same|an? equivalent|the agreed|the team's|everyone's)\s+"
                    r"(?:\w+\s+){0,2}?(?:fix|normali[sz]ation|change|edit|approach|repair|patch)\b"
                    r"|\b(?:as|like)\s+(?:the others|everyone|the team|the rest)\b", re.I)


@dataclass
class Post:
    t: int
    actor: str
    msg_id: str
    text: str
    cites: list[str] = field(default_factory=list)  # msg_ids
    follows_group: bool = False  # "applied the same fix" with no one named: cites the group

    @property
    def clean(self) -> str:
        return _MD.sub("", _SIGN.sub("", self.text))


def cited_agents(text: str, self_actor: str) -> list[str]:
    out = []
    for pat in _CITE:
        for m in pat.finditer(text):
            for a in re.findall(r"a\d", m.group(1)):
                if a != self_actor and a not in out:
                    out.append(a)
    return out


def build(posts: list[Post], claim: str | None = None) -> dict[str, Post]:
    """Fill each post's `cites` (in place) and return posts by msg_id."""
    by_id = {p.msg_id: p for p in posts}
    cpat = re.compile(claim, re.I) if claim else None
    for i, p in enumerate(posts):
        txt = p.clean
        earlier = posts[:i]
        for n in _MID.findall(txt):
            mid = f"m{n}"
            if mid in by_id and by_id[mid].t <= p.t and mid != p.msg_id and mid not in p.cites:
                p.cites.append(mid)
        p.follows_group = bool(_GROUP.search(txt))
        named_by_id = {by_id[c].actor for c in p.cites}
        for a in cited_agents(txt, p.actor):
            if a in named_by_id:  # an explicit m-id already pins which post of a's is meant
                continue
            theirs = [q for q in earlier if q.actor == a]
            if cpat:
                theirs = [q for q in theirs if cpat.search(q.clean)] or theirs
            if theirs and theirs[-1].msg_id not in p.cites:
                p.cites.append(theirs[-1].msg_id)
    return by_id


def roots(start: str, by_id: dict[str, Post]) -> tuple[list[str], set[str]]:
    """Return (root msg_ids, all msg_ids reached) following citations back from start."""
    seen, stack, rs = set(), [start], []
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        cites = by_id[m].cites
        if not cites:
            rs.append(m)
        stack.extend(cites)
    return sorted(rs, key=lambda m: by_id[m].t), seen


def justification_post(editor: str, t: int, posts: list[Post], window: int = 6) -> Post | None:
    cands = [p for p in posts if p.actor == editor and abs(p.t - t) <= window]
    return min(cands, key=lambda p: (abs(p.t - t), -p.t)) if cands else None


def agent_circles(by_id: dict[str, Post]) -> list[list[str]]:
    """Groups of agents that cite each other round a loop (A→B→…→A) at some point in the
    world: strongly connected components of the agent-level citation graph, size > 1."""
    g: dict[str, set[str]] = {}
    for p in by_id.values():
        for c in p.cites:
            if by_id[c].actor != p.actor:
                g.setdefault(p.actor, set()).add(by_id[c].actor)
    nodes = sorted(set(g) | {b for v in g.values() for b in v})

    def reach(a):
        seen, st = set(), [a]
        while st:
            x = st.pop()
            for y in g.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        return seen

    r = {a: reach(a) for a in nodes}
    comps, done = [], set()
    for a in nodes:
        if a in done:
            continue
        comp = sorted({a} | {b for b in r[a] if a in r[b]})
        done |= set(comp)
        if len(comp) > 1:
            comps.append(comp)
    return comps


def load(run_dir: Path) -> tuple[list[Post], list[dict]]:
    posts, edits, seen = [], [], set()
    for line in (Path(run_dir) / "events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["event"] == "board_post":
            pl = e["payload"]
            posts.append(Post(e["t"], e["actor"], pl.get("msg_id", ""), pl.get("text", "")))
        elif e["event"] == "violation" and e["actor"] not in seen:
            seen.add(e["actor"])
            edits.append({"t": e["t"], "actor": e["actor"], "path": e["payload"].get("path")})
    return posts, edits


def analyse(run_dir: Path, claim: str | None = None) -> dict:
    posts, edits = load(run_dir)
    by_id = build(posts, claim)
    chains = []
    for ed in edits:
        jp = justification_post(ed["actor"], ed["t"], posts)
        if jp is None:
            chains.append({**ed, "justification": None, "roots": [], "depth_posts": 0})
            continue
        rs, reached = roots(jp.msg_id, by_id)
        chains.append({
            **ed,
            "justification": {"msg_id": jp.msg_id, "t": jp.t, "text": jp.clean[:300]},
            "cites_teammate": bool(jp.cites),
            "follows_group": jp.follows_group,
            "depth_posts": len(reached),
            "roots": [{"msg_id": r, "t": by_id[r].t, "actor": by_id[r].actor,
                       "stance": classify(by_id[r].clean), "text": by_id[r].clean[:300],
                       "own": by_id[r].actor == ed["actor"]} for r in rs],
        })
    root_counts = Counter(r["msg_id"] for c in chains for r in c["roots"] if not r["own"])
    return {
        "posts": len(posts),
        "posts_citing": sum(bool(p.cites) for p in posts),
        "circles": agent_circles(by_id),
        "edits": chains,
        "shared_roots": {m: n for m, n in root_counts.items() if n > 1},
        "_by_id": by_id,
    }
