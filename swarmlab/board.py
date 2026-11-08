"""Append-only team chat: messages can reply to another message, carry a kind
(question / answer / idea / update) and mention agents as @aN. A message may be
private (`to`: named recipients); only the sender and recipients see it. The harness
log always has every message."""
import re
from typing import Dict, List, Optional

KINDS = ("question", "answer", "idea", "update")
_AGENT = re.compile(r"a\d{1,3}")


class Board:
    DIRECT = {"reviewer"}  # posts delivered by the harness, not through read offers

    def __init__(self, enabled: bool, max_post_chars: int = 2000) -> None:
        self.enabled = enabled
        self.max_post_chars = max_post_chars
        self._messages: List[Dict] = []
        self._seen: Dict[str, int] = {}

    def post(self, actor: str, text: str, reply_to: Optional[str] = None,
             kind: Optional[str] = None, to: Optional[List[str]] = None) -> Optional[str]:
        """Append a message; returns its msg_id, or None when the board is off.
        An unknown reply_to is dropped; an unknown kind becomes "update"; recipients
        that aren't agent ids (or are the sender) are dropped, and none left = public."""
        if not self.enabled:
            return None
        msg_id = f"m{len(self._messages) + 1:04d}"
        known = {m["msg_id"] for m in self._messages if self._visible(m, actor)}
        rcpt = sorted({r for r in (to or []) if isinstance(r, str)
                       and _AGENT.fullmatch(r) and r != actor})
        self._messages.append({"msg_id": msg_id, "actor": actor,
                               "text": text[: self.max_post_chars],
                               "reply_to": reply_to if reply_to in known else None,
                               "kind": kind if kind in KINDS else "update",
                               "to": rcpt or None})
        return msg_id

    @staticmethod
    def _visible(m: Dict, reader: Optional[str]) -> bool:
        return (reader is None or m.get("to") is None or m["actor"] == reader
                or reader in m["to"])

    def unseen(self, agent_id: str) -> List[Dict]:
        """Posts by others that agent_id has not been shown yet; marks them seen."""
        start = self._seen.get(agent_id, 0)
        self._seen[agent_id] = len(self._messages)
        return [dict(m) for m in self._messages[start:] if m["actor"] != agent_id
                and m["actor"] not in self.DIRECT and self._visible(m, agent_id)]

    def n_unseen(self, agent_id: str, since: int = 0) -> int:
        """How many posts by others agent_id has not seen, without marking them.
        `since` ignores posts before that index (already offered to a woken agent)."""
        start = max(self._seen.get(agent_id, 0), since)
        return sum(1 for m in self._messages[start:] if m["actor"] != agent_id
                   and m["actor"] not in self.DIRECT and self._visible(m, agent_id))

    def n_addressed_unseen(self, agent_id: str, since: int = 0) -> int:
        """Unseen posts that @mention agent_id or reply to one of its messages."""
        start = max(self._seen.get(agent_id, 0), since)
        mine = {m["msg_id"] for m in self._messages if m["actor"] == agent_id}
        tag = re.compile(rf"@{re.escape(agent_id)}\b")
        return sum(1 for m in self._messages[start:] if m["actor"] != agent_id
                   and m["actor"] not in self.DIRECT and self._visible(m, agent_id)
                   and (m["reply_to"] in mine or tag.search(m["text"])))

    def open_questions(self, exclude_actor: str, reader: Optional[str] = None) -> List[Dict]:
        """Unanswered questions, optionally limited to a reader's delivered posts."""
        visible = (self._messages if reader is None
                   else self._messages[:self._seen.get(reader, 0)])
        answered = {m["reply_to"] for m in visible
                    if m["reply_to"] and self._visible(m, reader)}
        return [dict(m) for m in visible if m["kind"] == "question"
                and m["actor"] != exclude_actor and m["msg_id"] not in answered
                and self._visible(m, reader)]

    def last_by(self, actor: str) -> Optional[Dict]:
        mine = [m for m in self._messages if m["actor"] == actor]
        return dict(mine[-1]) if mine else None

    def messages(self, reader: Optional[str] = None) -> List[Dict]:
        """All messages (the harness's view), or only those `reader` can see."""
        return [dict(m) for m in self._messages if self._visible(m, reader)]


def render_messages(msgs: List[Dict], reader: str) -> str:
    lines = []
    for m in msgs:
        tags = [m.get("kind") or "update"]
        if m.get("reply_to"):
            tags.append(f"reply to {m['reply_to']}")
        if m.get("to"):
            tags.append(f"private: {m['actor']} → {', '.join(m['to'])}")
        if re.search(rf"@{re.escape(reader)}\b", m["text"]):
            tags.append("to you")
        lines.append(f"[{m['msg_id']}] ({', '.join(tags)}) {m['text']}")
    return "\n".join(lines)
