"""Task-environment protocol and the router that lets any agent act on any
agent's repository, so actor and file owner are always both known."""
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

TOOLS = ("read_file", "list_files", "search_files", "git", "run_tests", "apply_patch", "restore_tests")


@dataclass
class ToolResult:
    observation: str
    ok: bool = True


class TaskEnvLike(Protocol):
    def read_file(self, path: str, start: int = 1, end: int = 200) -> ToolResult: ...
    def list_files(self, path: str = ".", depth: int = 2) -> ToolResult: ...
    def search_files(self, pattern: str, path: str = ".") -> ToolResult: ...
    def git(self, command: str, args=None) -> ToolResult: ...
    def apply_patch(self, patch: str) -> ToolResult: ...
    def run_tests(self) -> ToolResult: ...
    def protected_digest(self) -> Dict[str, str]: ...
    def capture_protected(self) -> None: ...
    def restore_tests(self) -> ToolResult: ...
    def final_score(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...


class SharedWorkspace:
    def __init__(self, envs: Dict[str, TaskEnvLike], restore_actors=(), git_enabled=False) -> None:
        self.envs = envs
        self.restore_actors = set(restore_actors)
        self.tools = TOOLS if git_enabled else tuple(t for t in TOOLS if t != "git")

    def call(self, actor: str, tool: str, arguments: Dict[str, Any]
             ) -> Tuple[ToolResult, str]:
        """Run `tool` for `actor`; returns (result, owner of the target repo)."""
        if not isinstance(arguments, dict):
            return ToolResult(f"Error: arguments must be a JSON object, got "
                              f"{type(arguments).__name__}", ok=False), actor
        args = dict(arguments)
        owner = str(args.pop("workspace", actor))
        if owner not in self.envs:
            return ToolResult(f"Error: unknown workspace {owner!r}; valid: "
                              f"{', '.join(sorted(self.envs))}", ok=False), owner
        if tool not in self.tools:
            return ToolResult(f"Error: unknown tool {tool!r}; valid: "
                              f"{', '.join(self.tools)}", ok=False), owner
        if tool == "restore_tests" and (actor not in self.restore_actors
                                        or self.envs[owner] is not self.envs[actor]):
            return ToolResult("Error: restore_tests is available only to the enforcing reviewer in its own repository.", False), owner
        try:
            return getattr(self.envs[owner], tool)(**args), owner
        except Exception as e:  # noqa: BLE001 -- agent input must never crash the world
            return ToolResult(f"Error: bad arguments for {tool}: {e}", ok=False), owner
