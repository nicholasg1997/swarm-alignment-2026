"""Turns raw pytest output into the failure detail an agent is shown.

Troy shows agents only a one-line summary. We add which target tests failed (from
Troy's parsed per-test statuses) and each one's own traceback block, found by its
pytest section header so tracebacks from nested pytest runs are not mistaken for it.
"""
import re
from typing import Dict

_FAIL = ("FAILED", "ERROR")
_SECTION_END = re.compile(r"^(_{3,} .+ _{3,}|={3,} .* ={3,})\s*$", re.M)


def _block_for(stdout: str, test_id: str) -> str:
    name = re.escape(test_id.split("::")[-1])
    header = re.compile(rf"^_{{3,}} (?:[\w.]+\.)?{name}(?:\[[^\]]*\])? _{{3,}}\s*$", re.M)
    m = header.search(stdout)
    if not m:
        return ""
    end = _SECTION_END.search(stdout, m.end())
    return stdout[m.start(): end.start() if end else len(stdout)].rstrip()


def failure_detail(stdout: str, statuses: Dict[str, str], limit: int = 3000) -> str:
    failed = [t for t, s in statuses.items() if s in _FAIL]
    if not failed:
        return ""
    parts = ["Failing tests: " + ", ".join(failed)]
    parts += [b for b in (_block_for(stdout or "", t) for t in failed) if b]
    return "\n\n".join(parts)[:limit]
