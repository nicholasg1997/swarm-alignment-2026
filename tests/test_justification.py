from swarmlab.justification import Post, build, cited_agents, agent_circles, justification_post, roots


def test_crediting_phrases_count_but_addressing_does_not():
    assert cited_agents("per @a3 the line is misplaced", "a1") == ["a3"]
    assert cited_agents("exactly @a3's hypothesis", "a1") == ["a3"]
    assert cited_agents("the same minimal test fix as a1/a3/a4", "a2") == ["a1", "a3", "a4"]
    assert cited_agents("@a3 great catch on the subclass angle", "a2") == ["a3"]
    assert cited_agents("@a1 please hold off editing", "a2") == []
    assert cited_agents("per @a2 and my own run", "a2") == []  # no self-citation


def _world():
    return [
        Post(10, "a3", "m0001", "If the line were misplaced, both hold."),
        Post(12, "a2", "m0002", "@a3 great catch: misplaced."),
        Post(14, "a1", "m0003", "Same fix as a2, moved it."),
        Post(16, "a3", "m0004", "Confirming a1, green now."),
    ]


def test_chain_bottoms_out_in_the_hedged_origin():
    ps = _world()
    by_id = build(ps, claim="misplac")
    rs, reached = roots("m0003", by_id)
    assert rs == ["m0001"] and reached == {"m0001", "m0002", "m0003"}


def test_message_id_citation_and_agent_circle():
    ps = _world() + [Post(18, "a2", "m0005", "see m0004")]
    by_id = build(ps)
    assert by_id["m0005"].cites == ["m0004"]
    assert agent_circles(by_id) == [["a1", "a2", "a3"]]  # a1->a2->a3->a1


def test_justification_post_is_nearest_to_edit():
    ps = _world()
    assert justification_post("a1", 15, ps).msg_id == "m0003"
    assert justification_post("a6", 15, ps) is None


def test_unnamed_appeal_to_the_group_is_flagged():
    ps = [Post(1, "a5", "m0001", "Applied the same normalization: moved the line.")]
    build(ps)
    assert ps[0].follows_group and ps[0].cites == []


def test_explicit_msg_id_blocks_guessing_another_post_by_that_agent():
    ps = [Post(1, "a3", "m0001", "If misplaced..."), Post(2, "a3", "m0002", "misplaced? workspaces shared"),
          Post(3, "a2", "m0003", "exactly @a3's hypothesis (m0001)")]
    by_id = build(ps, claim="misplac")
    assert by_id["m0003"].cites == ["m0001"]
