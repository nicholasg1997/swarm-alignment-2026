from swarmlab.troy import ActionType, parse_action


def test_parse_action_roundtrip():
    a = parse_action('{"action":"USE_ALLOWED_TOOL","tool":"read_file",'
                     '"arguments":{"path":"x.py","start":1,"end":5}}')
    assert a.action == ActionType.USE_ALLOWED_TOOL and a.tool == "read_file"
