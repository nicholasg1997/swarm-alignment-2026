from swarmlab.testoutput import failure_detail

OUT = """collected 16 items
testing/logging/test_fixture.py ....F...

=================================== FAILURES ===================================
_______________________ test_log_report_captures_upon_failure _______________________
----------------------------- Captured stdout call -----------------------------
============================= test session starts ==============================
=================================== FAILURES ===================================
________________________________ test_that_fails _______________________________
>       assert False
E       assert False
_________________________ test_clear_for_call_stage __________________________

caplog = <LogCaptureFixture>

    def test_clear_for_call_stage(caplog):
>       assert caplog.get_records("call") == []
E       AssertionError: assert [<LogRecord>] == []
testing/logging/test_fixture.py:180: AssertionError
=========================== short test summary info ============================
PASSED testing/logging/test_fixture.py::test_change_level
FAILED testing/logging/test_fixture.py::test_clear_for_call_stage - AssertionError
========================= 1 failed, 15 passed in 0.4s =========================
"""
TARGET = "testing/logging/test_fixture.py::test_clear_for_call_stage"


def test_detail_names_failing_targets_and_their_traceback():
    d = failure_detail(OUT, {TARGET: "FAILED", "testing/logging/test_fixture.py::test_change_level": "PASSED"})
    assert d.startswith("Failing tests: " + TARGET)
    assert "AssertionError: assert [<LogRecord>] == []" in d
    assert "test_that_fails" not in d and "PASSED testing" not in d


def test_detail_capped():
    assert len(failure_detail(OUT, {TARGET: "FAILED"}, limit=200)) <= 200


def test_detail_block_missing_still_names_test():
    d = failure_detail("no tracebacks here", {"a.py::test_x": "ERROR"})
    assert "a.py::test_x" in d


def test_detail_nothing_failed():
    assert failure_detail(OUT, {TARGET: "PASSED"}) == ""
