"""Messages left in the repository: do agents without a chat talk through files?

A *message* is text an agent adds to the shared repo that is for reading rather than
running: every added line of a new non-code file, or an added comment line in a code
file. It is *addressed* when it names an agent or the team, or tells others what to do.
A *receipt* is a later tool result shown to a different agent that contains the line.

Lexical screen: read every hit in the trace before quoting it.
"""
from __future__ import annotations

import re
from typing import Dict, List

CODE_SUFFIXES = (".py", ".pyx", ".pyi", ".js", ".ts", ".c", ".h", ".cfg", ".toml", ".json")
ADDRESSED = re.compile(
    r"\ba[1-9]\b|teammate|other agents?|\bagents?\b|everyone|\bteam\b|\bplease\b|"
    r"\b(do not|don't|dont)\s+(edit|change|touch|modify|revert|remove)|"
    r"\bNOTE\b|\bFYI\b|\bTODO\b|\bWARNING\b|heads[- ]up|coordinat", re.I)
_MIN_CHARS = 12  # shorter lines match too much test output to count as a receipt


def _files(patch: str):
    """(path, is_new, added lines) per file in a unified diff."""
    out, cur = [], None
    for line in patch.splitlines():
        if line.startswith("--- "):
            cur = {"new": line[4:].strip() == "/dev/null", "path": None, "added": [], "removed": set()}
            out.append(cur)
        elif line.startswith("+++ ") and cur is not None:
            path = line[4:].strip()
            cur["path"] = path[2:] if path.startswith("b/") else path
        elif cur is not None and line.startswith("+"):
            cur["added"].append(line[1:])
        elif cur is not None and line.startswith("-"):
            cur["removed"].add(line[1:].strip())
    return [f for f in out if f["path"]]


def file_messages(events: List[Dict]) -> List[Dict]:
    """Messages agents left in files, in order, each with who later saw it."""
    results = {}
    for i, e in enumerate(events):
        if e["event"] == "tool_result" and e["payload"].get("tool") == "apply_patch":
            results.setdefault(e["actor"], []).append(i)
    msgs = []
    for i, e in enumerate(events):
        if e["event"] != "tool_call" or e["payload"].get("tool") != "apply_patch":
            continue
        res = next((events[j] for j in results.get(e["actor"], []) if j > i), None)
        if res is None or "ok=True" not in res["payload"].get("observation", "")[:60]:
            continue
        patch = (e["payload"].get("arguments") or {}).get("patch") or ""
        for f in _files(patch):
            code = f["path"].endswith(CODE_SUFFIXES)
            if f["new"] and not code:
                kind, lines = "new_file", [l for l in f["added"] if l.strip()]
            else:
                kind, lines = "comment", [l for l in f["added"] if l.strip().startswith("#")
                                          and l.strip() not in f["removed"]]  # moved, not new
            if not lines:
                continue
            text = "\n".join(lines)
            probes = [l.strip() for l in lines if len(l.strip()) >= _MIN_CHARS]
            seen = []
            for later in events[i + 1:]:
                if (later["event"] == "tool_result" and later["actor"] != e["actor"]
                        and any(p in later["payload"].get("observation", "") for p in probes)):
                    seen.append((later["t"], later["actor"]))
            msgs.append({"t": e["t"], "actor": e["actor"], "path": f["path"], "kind": kind,
                         "addressed": bool(ADDRESSED.search(text)), "text": text,
                         "read_by": seen})
    return msgs
