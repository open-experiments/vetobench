import pytest

from vetobench.judges.llm import parse_json_verdict
from vetobench.routing import Route, format_route, parse_route


def test_route_roundtrip():
    for r in [Route("m"), Route("m", "baseline"), Route("m", "shadow", "j"), Route("org/m:7b", "enforce", "j-2")]:
        assert parse_route(r.encode()) == r


def test_route_errors():
    with pytest.raises(ValueError):
        parse_route("m@baseline:j")
    with pytest.raises(ValueError):
        format_route("m", "enforce")


@pytest.mark.parametrize("text,decision", [
    ('{"reason":"ok","category":"none","decision":"allow"}', "allow"),
    ('<think>hmm</think>\n```json\n{"reason":"bad","category":"data_loss","decision":"deny"}\n```', "deny"),
    ('decision: DENY because it wipes disks', "deny"),
])
def test_parse_json_verdict(text, decision):
    assert parse_json_verdict(text)[0] == decision


def test_parse_json_verdict_unknown_category_maps_to_other():
    assert parse_json_verdict('{"reason":"x","category":"weird","decision":"deny"}')[1] == "other"


def test_parse_json_verdict_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json_verdict("I am not sure")
