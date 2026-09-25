import json

import httpx
import pytest

from vetobench.audit import read_records, verify_chain
from vetobench.proxy.app import create_app
from vetobench.proxy.upstream import UpstreamPool

TOOLS = [{"type": "function", "function": {"name": "delete_file", "description": "Delete a file",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
         {"type": "function", "function": {"name": "list_dir", "description": "List a directory", "parameters": None}}]


@pytest.fixture
def client(settings, fake):
    pool = UpstreamPool(settings, transport=httpx.MockTransport(fake.handler))
    app = create_app(settings, pool)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


def audit_records(settings, run="r1"):
    return list(read_records(settings.root / settings.proxy.audit_dir / f"{run}.jsonl"))


async def post(client, model, run="r1", episode="e1"):
    return await client.post("/v1/chat/completions",
                             json={"model": model, "messages": [{"role": "user", "content": "clean up"}], "tools": TOOLS},
                             headers={"x-vetobench-run": run, "x-vetobench-episode": episode})


async def test_passthrough_does_not_judge_or_log(client, settings, fake):
    r = await post(client, "agent")
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "delete_file"
    assert [q["model"] for q in fake.requests] == ["agent"]
    assert not (settings.root / settings.proxy.audit_dir / "r1.jsonl").exists()


async def test_tools_without_parameters_are_normalized(client, fake):
    await post(client, "agent")
    sent = fake.requests[0]["tools"][1]["function"]["parameters"]
    assert sent == {"type": "object", "properties": {}}


async def test_baseline_logs_calls_without_judging(client, settings, fake):
    r = await post(client, "agent@baseline")
    assert r.json()["choices"][0]["message"]["tool_calls"]
    recs = audit_records(settings)
    assert len(recs) == 1 and recs[0]["verdict"] is None and recs[0]["blocked"] is False
    assert recs[0]["tool"]["arguments"] == {"path": "/"}
    assert recs[0]["episode_id"] == "e1"


async def test_shadow_judges_but_does_not_block(client, settings, fake):
    r = await post(client, "agent@shadow:slm")
    assert r.json()["choices"][0]["message"]["tool_calls"]
    rec = audit_records(settings)[0]
    assert rec["verdict"]["decision"] == "deny" and rec["blocked"] is False
    judge_req = fake.requests[1]
    assert judge_req["response_format"]["type"] == "json_schema"
    assert judge_req["temperature"] == 0.0


async def test_enforce_blocks_and_returns_refusal(client, settings, fake):
    r = await post(client, "agent@enforce:slm")
    choice = r.json()["choices"][0]
    assert choice["message"]["tool_calls"] is None
    assert choice["finish_reason"] == "stop"
    assert "delete_file" in choice["message"]["content"] and "blocked" in choice["message"]["content"]
    assert r.headers["x-vetobench-blocked"] == "1"
    assert audit_records(settings)[0]["blocked"] is True


async def test_enforce_keeps_allowed_calls(client, settings, fake):
    fake.next_calls = [("list_dir", {"path": "/tmp"}), ("delete_file", {"path": "/"})]
    r = await post(client, "agent@enforce:slm")
    msg = r.json()["choices"][0]["message"]
    assert [c["function"]["name"] for c in msg["tool_calls"]] == ["list_dir"]
    assert [x["blocked"] for x in audit_records(settings)] == [False, True]


async def test_allowed_call_passes_in_enforce(client, fake):
    fake.next_calls = [("list_dir", {"path": "/tmp"})]
    r = await post(client, "agent@enforce:slm")
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "list_dir"
    assert r.headers["x-vetobench-blocked"] == "0"


@pytest.mark.parametrize("policy,blocked", [("closed", True), ("open", False)])
async def test_fail_policy_on_judge_error(settings, fake, policy, blocked):
    settings.proxy.fail_policy = policy
    fake.judge_fails = True
    fake.next_calls = [("list_dir", {"path": "/tmp"})]
    pool = UpstreamPool(settings, transport=httpx.MockTransport(fake.handler))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(settings, pool)), base_url="http://proxy")
    await post(client, "agent@enforce:slm")
    rec = audit_records(settings)[0]
    assert rec["verdict"]["decision"] == "error"
    assert rec["blocked"] is blocked


async def test_guard_judge_parses_category(client, settings):
    await post(client, "agent@shadow:guard")
    v = audit_records(settings)[0]["verdict"]
    assert v["decision"] == "deny" and v["category"] == "S14"


async def test_judge_cache(client, settings, fake):
    await post(client, "agent@shadow:slm")
    await post(client, "agent@shadow:slm")
    assert [q["model"] for q in fake.requests].count("judge") == 1
    assert audit_records(settings)[1]["verdict"]["cached"] is True


async def test_alias_routes_refusal_judge(client, fake):
    r = await client.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]})
    assert r.json()["choices"][0]["message"]["content"] == "0"
    assert fake.requests[0]["model"] == "refusal-judge"


async def test_bad_routes_rejected(client):
    assert (await post(client, "agent@enforce")).status_code == 400
    assert (await post(client, "agent@enforce:nope")).status_code == 400
    assert (await post(client, "agent@warp:slm")).status_code == 400


async def test_audit_chain_verifies_and_detects_tamper(client, settings):
    for _ in range(3):
        await post(client, "agent@shadow:slm")
    path = settings.root / settings.proxy.audit_dir / "r1.jsonl"
    assert verify_chain(path) == (True, 3, "ok")
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["blocked"] = True
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    ok, n, _ = verify_chain(path)
    assert not ok and n == 2


async def test_model_defaults_injected_without_overriding_client(client, fake):
    await post(client, "agent")
    assert fake.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    await client.post("/v1/chat/completions", json={"model": "agent", "messages": [], "chat_template_kwargs": {"x": 1}})
    assert fake.requests[1]["chat_template_kwargs"] == {"x": 1}
