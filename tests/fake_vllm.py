"""A deterministic stand-in for the vLLM endpoint, for end-to-end tests without GPUs.

Agent models: call the first offered tool once (or the tool an injected prompt names, like a
vulnerable model would), then answer in text. ASB plan requests get a two-step JSON plan.
Judge model: denies calls whose tool/arguments contain obviously malicious keywords.
ShieldAgent: pseudo-random but deterministic label per prompt.

    uvicorn tests.fake_vllm:app --port 18000
"""

from __future__ import annotations

import hashlib
import json
import re

from fastapi import FastAPI, Request

AGENTS = {"qwen3.8-27b", "muse-glimmer-30b", "gemma4-31b"}
SMALL = {"judge-small", "granite-guardian", "llama-guard", "shieldagent"}
BAD = re.compile(r"stealth|unnoticed|without detection|without leaving|credential|harvest|rm -rf|wipe|"
                 r"backdoor|hijack|exfiltrat|malware|delete_all|ransom", re.I)

app = FastAPI()


def _chat(content=None, tool_calls=None, model="x"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "fake", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _fill(schema: dict | None) -> dict:
    out = {}
    for name, spec in ((schema or {}).get("properties") or {}).items():
        t = (spec or {}).get("type")
        out[name] = {"integer": 1, "number": 1.0, "boolean": True, "array": [], "object": {}}.get(t, "test")
    return out


def _call(name: str, args: dict, i: int = 0) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def agent_reply(body: dict) -> dict:
    msgs, tools = body["messages"], body.get("tools") or []
    text = " ".join(str(m.get("content") or "") for m in msgs)
    if not tools and "Generate a plan of steps" in text:
        return _chat('[{"message": "Gather the needed information", "tool_use": ["x"]}, '
                     '{"message": "Summarize the result", "tool_use": []}]')
    already_called = any(m.get("role") == "tool" or "[Action]" in str(m.get("content") or "") for m in msgs)
    if tools and not already_called:
        names = [t["function"]["name"] for t in tools]
        m = re.search(r'by using the "([^"]+)" tool', text)
        target = m.group(1) if m and m.group(1) in names else names[0]
        schema = next(t["function"].get("parameters") for t in tools if t["function"]["name"] == target)
        return _chat(None, [_call(target, _fill(schema))])
    return _chat("Task finished.")


def judge_reply(body: dict) -> dict:
    sys_msg = body["messages"][0].get("content") or ""
    if sys_msg.startswith("You are a helpful assistant. Your goal is to judge"):  # ASB refusal judge
        return _chat("1")
    user = body["messages"][-1]["content"]
    if "Reply with the digit 1" in user:
        return _chat("1")
    deny = bool(BAD.search(user))
    return _chat(json.dumps({"reason": "malicious intent" if deny else "routine call",
                             "category": "security_compromise" if deny else "none",
                             "decision": "deny" if deny else "allow"}))


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in sorted(AGENTS | SMALL)]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body["model"]
    if model in AGENTS:
        return agent_reply(body)
    if model == "judge-small":
        return judge_reply(body)
    if model in ("granite-guardian", "llama-guard"):
        unsafe = bool(BAD.search(body["messages"][-1]["content"]))
        return _chat(("Yes" if unsafe else "No") if model == "granite-guardian" else ("unsafe\nS2" if unsafe else "safe"))
    return {"error": {"message": f"model {model} not served"}}


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    h = int(hashlib.sha256(body["prompt"].encode()).hexdigest(), 16)
    label = "unsafe" if h % 5 < 2 else "safe"
    return {"id": "fake", "object": "text_completion",
            "choices": [{"index": 0, "text": f"[Answer] {label}\n[Analysis]", "finish_reason": "length"}]}
