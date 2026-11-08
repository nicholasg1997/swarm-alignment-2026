from fixtures.make_fixtures import action, post, violation, write_world

from swarmlab.report import render_dashboard, render_run_page


def test_run_page_contains_lanes(tmp_path):
    d = write_world(tmp_path, "w1", "C1", 0, [action("a1"), post("a2", "[a2] hi"), violation("a2")])
    html = render_run_page(d).read_text()
    for i in range(1, 7):
        assert f'data-agent="a{i}"' in html
    assert "violation" in html and "[a2] hi" in html


def test_partial_world_renders(tmp_path):
    d = write_world(tmp_path, "w1", "C1", 0, [action("a1")], reason="api_error")
    assert "api_error" in render_run_page(d).read_text()


def test_dashboard_lists_all_runs(tmp_path):
    for b in range(3):
        write_world(tmp_path, f"w{b}", "C1" if b else "C0", b, [])
    html = render_dashboard(tmp_path).read_text()
    assert all(f"w{b}/index.html" in html for b in range(3))
    assert "C0" in html and "C1" in html


def test_no_secrets_in_html(tmp_path):
    d = write_world(tmp_path, "w1", "C1", 0, [],
                    extra_manifest={"env": {"OPENROUTER_API_KEY": "sk-or-secret-123"}})
    assert "sk-or-secret-123" not in render_run_page(d).read_text()
    assert "sk-or-secret-123" not in render_dashboard(tmp_path).read_text()


def test_html_escapes_agent_text(tmp_path):
    d = write_world(tmp_path, "w1", "C1", 0, [post("a2", "[a2] <script>alert(1)</script>")])
    assert "<script>alert(1)" not in render_run_page(d).read_text()


def _devil_world(tmp_path):
    from test_influence import EVENTS, ROLES
    return write_world(tmp_path, "dv", "DV", 0, EVENTS, roles=ROLES,
                       extra_manifest={"teams": {"a1": None, "a2": None, "a3": None,
                                                 "a4": "pair", "a5": "pair", "a6": None}})


def test_run_page_shows_roles_teams_devil_and_influence(tmp_path):
    html = render_run_page(_devil_world(tmp_path)).read_text()
    assert 'data-role="devil"' in html and "team pair" in html
    assert "Influence" in html and "second_hand" in html and "direct" in html
    assert 'class="devil-msg"' in html


def test_dashboard_has_exposure_columns(tmp_path):
    _devil_world(tmp_path)
    html = render_dashboard(tmp_path).read_text()
    assert "Direct" in html and "Overheard" in html and "Second-hand" in html
    assert "Devil msgs" in html


def test_all_focal_world_renders_without_a_fabricated_peer_rate(tmp_path):
    roles = {f"a{i}": "focal" for i in range(1, 7)}
    run_dir = write_world(tmp_path, "sp0", "SP0", 0, [violation("a2")], roles=roles)
    page = render_run_page(run_dir).read_text()
    assert "Peer violation fraction: –" in page
    assert "focal violation fraction:\n0.17" in page
    assert "SP0" in render_dashboard(tmp_path).read_text()


def test_unavailable_scores_are_not_reported_as_failures():
    from swarmlab.report import _score_label

    assert _score_label({"original_tests_pass": None}) == "not scored"
    assert _score_label({"original_tests_pass": None, "error": "timeout"}) == "scoring error"
    assert _score_label({"original_tests_pass": True}) == "pass"
    assert _score_label({"original_tests_pass": False}) == "fail"
