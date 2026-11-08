from swarmlab.board import Board


def test_post_and_unseen():
    b = Board(enabled=True)
    mid = b.post("a1", "[a1] hello")
    assert mid == "m0001"
    got = b.unseen("a2")
    assert [m["msg_id"] for m in got] == ["m0001"] and got[0]["actor"] == "a1"
    assert b.unseen("a2") == []


def test_poster_does_not_receive_own_post():
    b = Board(enabled=True)
    b.post("a1", "x")
    assert b.unseen("a1") == []


def test_disabled_board_returns_none():
    b = Board(enabled=False)
    assert b.post("a1", "hi") is None
    assert b.unseen("a2") == []


def test_post_truncated():
    b = Board(enabled=True)
    b.post("a1", "x" * 3000)
    assert len(b.messages()[0]["text"]) == 2000


def test_reply_and_kind_stored():
    b = Board(enabled=True)
    q = b.post("a1", "[a1] anyone seen this?", kind="question")
    r = b.post("a2", "[a2] yes", reply_to=q, kind="answer")
    m = b.messages()[1]
    assert (m["reply_to"], m["kind"]) == (q, "answer") and r == "m0002"


def test_bad_reply_to_and_kind_are_dropped():
    b = Board(enabled=True)
    b.post("a1", "x", reply_to="m0999", kind="banana")
    m = b.messages()[0]
    assert m["reply_to"] is None and m["kind"] == "update"


def test_open_questions_are_unanswered_questions_by_others():
    b = Board(enabled=True)
    q1 = b.post("a1", "q1?", kind="question")
    b.post("a3", "q2?", kind="question")
    b.post("a2", "ans", reply_to=q1, kind="answer")
    assert [m["text"] for m in b.open_questions(exclude_actor="a2")] == ["q2?"]
    assert b.open_questions(exclude_actor="a3") == []


def test_open_questions_do_not_leak_to_a_reader_who_declined_delivery():
    from swarmlab.world import _post_offer

    board = Board(enabled=True)
    board.post("a1", "[a1] UNREAD_QUESTION", kind="question")
    assert "UNREAD_QUESTION" not in _post_offer(board, "a2")
    board.unseen("a2")
    assert "UNREAD_QUESTION" in _post_offer(board, "a2")


def test_last_by():
    b = Board(enabled=True)
    b.post("a1", "one")
    b.post("a1", "two")
    assert b.last_by("a1")["text"] == "two" and b.last_by("a2") is None


def test_render_marks_replies_and_mentions():
    from swarmlab.board import render_messages
    b = Board(enabled=True)
    q = b.post("a1", "[a1] @a3 can you check?", kind="question")
    b.post("a3", "[a3] on it", reply_to=q, kind="answer")
    out = render_messages(b.messages(), reader="a3")
    assert "[m0001] (question, to you) [a1] @a3 can you check?" in out
    assert "[m0002] (answer, reply to m0001) [a3] on it" in out
