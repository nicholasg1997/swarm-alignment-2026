"""Protected-file change detection from before/after digests."""
from dataclasses import dataclass
from typing import Dict, List, Literal

#: tools after which the target repo's protected digest is recomputed
#: (run_tests executes agent-written code, which can write files).
WRITE_TOOLS = ("apply_patch", "run_tests", "restore_tests")

ABSENT = "ABSENT"


@dataclass(frozen=True)
class Change:
    path: str
    kind: Literal["modified", "created", "deleted", "reverted"]


def diff_digests(before: Dict[str, str], after: Dict[str, str],
                 baseline: Dict[str, str]) -> List[Change]:
    changes = []
    for path in sorted(set(before) | set(after)):
        b, a = before.get(path, ABSENT), after.get(path, ABSENT)
        if a == b:
            continue
        if a == baseline.get(path, ABSENT):
            kind = "reverted"
        elif b == ABSENT:
            kind = "created"
        elif a == ABSENT:
            kind = "deleted"
        else:
            kind = "modified"
        changes.append(Change(path, kind))
    return changes
