import json

from swarmlab.file_channel import file_messages


def call(t, actor, tool, **arguments):
    return {"event": "tool_call", "t": t, "actor": actor, "payload": {"tool": tool, "arguments": arguments}}


def result(t, actor, tool, observation):
    return {"event": "tool_result", "t": t, "actor": actor,
            "payload": {"tool": tool, "observation": observation}}


NOTE = ("--- /dev/null\n+++ b/NOTES_a2.md\n@@ -0,0 +1,2 @@\n"
        "+a2: the test asserts both ndarray and CustomArray\n+Please don't edit test_variable.py yet\n")
COMMENT = ("--- a/xarray/core/variable.py\n+++ b/xarray/core/variable.py\n@@ -1,2 +1,3 @@\n x = 1\n"
           "+# NOTE for teammates: do not revert this guard\n+y = 2\n")
CODE = "--- a/lib.py\n+++ b/lib.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"


def test_finds_new_files_and_addressed_comments_and_who_read_them():
    events = [
        call(1, "a2", "apply_patch", patch=NOTE), result(1, "a2", "apply_patch", "apply_patch ok=True"),
        call(2, "a3", "apply_patch", patch=COMMENT), result(2, "a3", "apply_patch", "apply_patch ok=True"),
        call(3, "a4", "apply_patch", patch=CODE), result(3, "a4", "apply_patch", "apply_patch ok=True"),
        call(4, "a5", "apply_patch", patch=NOTE.replace("a2", "a5")),
        result(4, "a5", "apply_patch", "apply_patch ok=False: does not apply"),
        call(5, "a1", "read_file", path="NOTES_a2.md"),
        result(5, "a1", "read_file", "a2: the test asserts both ndarray and CustomArray\n"
                                      "Please don't edit test_variable.py yet"),
        result(6, "a2", "read_file", "Please don't edit test_variable.py yet"),  # own note: no receipt
    ]
    msgs = file_messages(events)
    assert [(m["t"], m["actor"], m["path"], m["kind"]) for m in msgs] == [
        (1, "a2", "NOTES_a2.md", "new_file"), (2, "a3", "xarray/core/variable.py", "comment")]
    assert msgs[0]["addressed"] and msgs[1]["addressed"]
    assert msgs[0]["read_by"] == [(5, "a1")] and msgs[1]["read_by"] == []


def test_a_moved_existing_comment_is_not_a_message():
    moved = ("--- a/lib.py\n+++ b/lib.py\n@@ -1,3 +1,3 @@\n"
             "-    # we don't want nested arrays\n-a = 1\n+a = 2\n+    # we don't want nested arrays\n")
    events = [call(1, "a1", "apply_patch", patch=moved), result(1, "a1", "apply_patch", "apply_patch ok=True")]
    assert file_messages(events) == []
