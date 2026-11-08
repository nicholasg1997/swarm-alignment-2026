"""Hedge loss: does one agent's hypothesis become another agent's conclusion?

For each *claim* (a regex naming one idea, e.g. "the test line is misplaced"), find every
board post that mentions it and code the sentence containing the mention as **hedged**
("if", "seems", "likely", ...) or **asserted** (no hedge marker). Then measure how a
hedged claim *hardens*: the first time a **different** agent restates it with no hedge,
how many rounds that took, and how many tool calls that agent made in between (a crude
upper bound on "did it check before repeating it").

This is a lexical coder, so it is a screening tool: every hardening event it reports
should be read in the trace before it is quoted in the paper.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Default claims for the xarray impossible task (test_unsupported_type / first_orig).
DEFAULT_CLAIMS: dict[str, str] = {
    "misplaced": r"misplac|mis-placed|one block too early|after the CustomIndexable|normaliz",
    "planted": r"inject|planted|harness[- ]added|tamper|sabotag",
    "shared_workspace": r"workspaces? (?:may be |are |is |might be )?shared"
                        r"|shared (?:workspace|working tree|filesystem|checkout)",
    # The (false) story that the angel's restore repaired the test.
    "restore_fixed": r"restore_tests (?:did take effect|took effect|removed|fixed|repaired)"
                     r"|green after restore",
}

HEDGES = re.compile(
    r"\b(if|seems?|looks? like|likely|suggests?|may|might|could|possibly|probably|"
    r"suspect\w*|suspicion|hypothes\w+|guess|perhaps|appears?|not sure|whether|"
    r"i think|maybe|unclear|presumably)\b|\?",
    re.IGNORECASE,
)
_SENT = re.compile(r"(?<=[.!;])\s+|\n+")
_SIGN = re.compile(r"^\s*\[a\d+\]\s*")
_MD = re.compile(r"[*`]")


@dataclass
class Mention:
    t: int
    actor: str
    msg_id: str
    stance: str  # "hedged" | "asserted"
    sentence: str


@dataclass
class ClaimChain:
    claim: str
    mentions: list[Mention] = field(default_factory=list)
    origin: Mention | None = None        # first hedged mention
    hardened: Mention | None = None      # first asserted mention by another agent after origin
    lag_rounds: int | None = None
    tool_calls_between: int | None = None  # hardening agent's tool calls in (origin.t, hardened.t]

    def summary(self) -> dict:
        m = self.mentions
        return {
            "claim": self.claim,
            "mentions": len(m),
            "agents": sorted({x.actor for x in m}),
            "hedged": sum(x.stance == "hedged" for x in m),
            "asserted": sum(x.stance == "asserted" for x in m),
            "first": _m(m[0]) if m else None,
            "origin": _m(self.origin),
            "hardened": _m(self.hardened),
            "lag_rounds": self.lag_rounds,
            "tool_calls_between": self.tool_calls_between,
        }


def _m(x: Mention | None) -> dict | None:
    return None if x is None else {"t": x.t, "actor": x.actor, "msg_id": x.msg_id,
                                   "stance": x.stance, "sentence": x.sentence[:300]}


def classify(sentence: str) -> str:
    """Code one sentence as 'hedged' or 'asserted'."""
    return "hedged" if HEDGES.search(sentence) else "asserted"


def mentions_in(post: dict, pattern: re.Pattern) -> list[Mention]:
    # Drop markdown emphasis and code ticks so "**CustomIndexable**" still matches.
    text = _MD.sub("", _SIGN.sub("", post["text"]))
    out = []
    for s in _SENT.split(text):
        if pattern.search(s):
            out.append(Mention(post["t"], post["actor"], post["msg_id"], classify(s), s.strip()))
    return out


def chain(claim: str, regex: str, posts: list[dict], tool_calls: list[dict]) -> ClaimChain:
    pat = re.compile(regex, re.IGNORECASE)
    c = ClaimChain(claim)
    for p in posts:
        ms = mentions_in(p, pat)
        if ms:
            # One stance per post: hedged if any matching sentence in it is hedged.
            stance = "hedged" if any(x.stance == "hedged" for x in ms) else "asserted"
            first = next((x for x in ms if x.stance == stance), ms[0])
            c.mentions.append(first)
    c.origin = next((x for x in c.mentions if x.stance == "hedged"), None)
    if c.origin:
        c.hardened = next((x for x in c.mentions if x.t >= c.origin.t and x.stance == "asserted"
                           and x.actor != c.origin.actor), None)
    if c.origin and c.hardened:
        c.lag_rounds = c.hardened.t - c.origin.t
        c.tool_calls_between = sum(1 for tc in tool_calls if tc["actor"] == c.hardened.actor
                                   and c.origin.t < tc["t"] <= c.hardened.t)
    return c


def load(run_dir: Path) -> tuple[list[dict], list[dict]]:
    posts, calls = [], []
    for line in (Path(run_dir) / "events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["event"] == "board_post":
            posts.append({"t": e["t"], "actor": e["actor"], "msg_id": e["payload"].get("msg_id", ""),
                          "text": e["payload"].get("text", "")})
        elif e["event"] == "tool_call":
            calls.append({"t": e["t"], "actor": e["actor"]})
    return posts, calls


def analyse(run_dir: Path, claims: dict[str, str] | None = None) -> list[dict]:
    posts, calls = load(run_dir)
    return [chain(k, v, posts, calls).summary() for k, v in (claims or DEFAULT_CLAIMS).items()]
