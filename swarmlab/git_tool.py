"""Restricted repository inspection, with no shell or configurable diff helpers."""
import inspect
import shlex

from .workspace import ToolResult

_COMMON = {"--stat", "--numstat", "--shortstat", "--name-only", "--name-status",
           "--patch", "-p", "--no-patch", "-s", "--raw", "--summary",
           "--no-renames", "--find-renames", "--binary", "--full-index"}
_HISTORY = {"--oneline", "--all", "--graph", "--decorate", "--no-decorate",
            "--reverse", "--first-parent", "--no-merges", "--merges", "--reflog"}
_VALUES = {"--format", "--pretty", "--max-count", "--skip", "--author",
           "--committer", "--grep", "--since", "--until"}


def git_argv(command, args=None):
    """Allow ordinary inspection flags; reject config, helpers and output files."""
    if command not in ("log", "show", "diff"):
        raise ValueError("command must be log, show, or diff")
    if args is None:
        args = []
    if (not isinstance(args, list) or len(args) > 64
            or any(not isinstance(a, str) or len(a) > 2048 or "\0" in a for a in args)):
        raise ValueError("args must be a list of at most 64 short strings")
    allowed = _COMMON | (_HISTORY if command != "diff" else {
        "--cached", "--staged", "--check", "--ignore-space-at-eol", "--ignore-all-space", "-w"})
    i, paths, normalized = 0, False, []
    while i < len(args):
        arg = args[i]
        rendered = arg
        if arg == "--" and not paths:
            paths = True
        elif arg.startswith("-") and not paths:
            name, sep, value = arg.partition("=")
            if arg in allowed:
                pass
            elif command != "diff" and (name in _VALUES or arg == "-n"):
                if not sep:
                    i += 1
                    if i >= len(args):
                        raise ValueError(f"{arg} needs a value")
                    value = args[i]
                if name in ("--max-count", "--skip", "-n") and not value.isdigit():
                    raise ValueError(f"{name} needs a nonnegative integer")
                # Optional-value flags such as --pretty do not consume a separate
                # token in Git. Always bind values so they cannot become options.
                rendered = ("--max-count" if name == "-n" else name) + "=" + value
            elif command != "diff" and arg.startswith("-n") and arg[2:].isdigit():
                pass
            elif name == "--unified" and sep and value.isdigit():
                pass
            elif arg.startswith("-U") and arg[2:].isdigit():
                pass
            else:
                raise ValueError(f"unsupported read-only git option: {arg}")
        elif arg.startswith(("/", "\\")) or ".." in arg.split("/"):
            raise ValueError("paths must stay inside the repository")
        normalized.append(rendered)
        i += 1
    prefix = ["git", "--no-pager", "--no-optional-locks",
              "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
              "-c", "core.attributesFile=/dev/null", "-c", "diff.external=",
              command, "--no-ext-diff", "--no-textconv", "--color=never"]
    if command == "log" and not any(a in ("-n", "--max-count") or
            a.startswith(("-n", "--max-count=")) for a in args):
        prefix += ["--max-count=20"]
    return prefix + normalized


def execute_git(repo, argv):
    """Also shipped verbatim into Modal; only Git runs, with a clean environment."""
    import subprocess
    import tempfile

    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
    with tempfile.TemporaryFile() as output:
        try:
            proc = subprocess.run(argv, cwd=repo, env=env, stdout=output,
                                  stderr=subprocess.STDOUT, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            return "Error: git inspection failed: " + str(error), False
        output.seek(0)
        raw = output.read(20001)
    text = raw[:20000].decode("utf-8", errors="replace")
    if len(raw) > 20000:
        text += "\n[Output truncated at 20000 bytes; narrow the revision/path or use --stat.]"
    return text or "(no output)", proc.returncode == 0


def git_script(repo, command, args=None):
    argv = git_argv(command, args)
    code = (inspect.getsource(execute_git) + "\nimport sys\n"
            + f"text, ok = execute_git({repo!r}, {argv!r})\n"
            + "print(text)\nsys.exit(0 if ok else 1)\n")
    return "PATH=/usr/local/bin:/usr/bin:/bin python3 -I -c " + shlex.quote(code)


def unavailable_git(command, args=None):
    git_argv(command, args)
    return ToolResult("Error: this in-memory task fixture has no Git repository or history.", False)
