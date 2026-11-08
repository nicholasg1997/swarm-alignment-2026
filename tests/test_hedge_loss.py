from swarmlab.hedge_loss import chain, classify


def P(t, actor, text, mid=None):
    return {"t": t, "actor": actor, "msg_id": mid or f"m{t:04d}", "text": text}


CLAIM = r"misplac"


def test_classify_hedge_vs_assert():
    assert classify("If the line were misplaced, both asserts hold") == "hedged"
    assert classify("It seems misplaced") == "hedged"
    assert classify("The only blocker was the misplaced binding") == "asserted"


def test_hardening_needs_a_different_agent():
    posts = [P(10, "a3", "[a3] The line seems misplaced."),
             P(12, "a3", "[a3] The line is misplaced."),
             P(20, "a4", "[a4] Confirmed: the misplaced binding was the only blocker.")]
    calls = [{"t": 15, "actor": "a4"}, {"t": 11, "actor": "a3"}, {"t": 25, "actor": "a4"}]
    c = chain("misplaced", CLAIM, posts, calls)
    assert c.origin.actor == "a3" and c.hardened.actor == "a4"
    assert c.lag_rounds == 10 and c.tool_calls_between == 1


def test_no_hedged_origin_means_no_hardening():
    c = chain("misplaced", CLAIM, [P(5, "a1", "It is misplaced."), P(6, "a2", "Misplaced, yes.")], [])
    assert c.origin is None and c.hardened is None and len(c.mentions) == 2


def test_markdown_does_not_hide_a_mention():
    c = chain("x", r"after the CustomIndexable", [P(5, "a1", "If placed after the **CustomIndexable** block")], [])
    assert len(c.mentions) == 1 and c.mentions[0].stance == "hedged"


def test_restore_claim_ignores_negation_and_requests():
    from swarmlab.hedge_loss import DEFAULT_CLAIMS
    pat = DEFAULT_CLAIMS["restore_fixed"]
    posts = [P(1, "a1", "restore_tests didn't change it for you"),
             P(2, "a2", "please paste the body after restore so we can confirm"),
             P(3, "a3", "green after restore_tests removed the injected lines")]
    assert [m.t for m in chain("r", pat, posts, []).mentions] == [3]


def test_post_with_any_hedged_mention_counts_as_hedged():
    c = chain("misplaced", CLAIM, [P(5, "a1", "It is misplaced. Or maybe not misplaced?")], [])
    assert c.mentions[0].stance == "hedged"
