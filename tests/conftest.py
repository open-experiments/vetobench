import json
from pathlib import Path

import httpx
import pytest

from vetobench.config import ProxySettings, Settings, Upstream


def tool_call_response(calls: list[tuple[str, dict]], content: str | None = None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "agent",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
                    for i, (name, args) in enumerate(calls)
                ],
            },
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def text_response(text: str) -> dict:
    return {"id": "x", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


class FakeUpstream:
    """Agent model proposes whatever `next_calls` says; judge denies calls whose args mention rm -rf."""

    def __init__(self):
        self.next_calls: list[tuple[str, dict]] = [("delete_file", {"path": "/"})]
        self.requests: list[dict] = []
        self.judge_fails = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        model = body["model"]
        if model == "agent":
            return httpx.Response(200, json=tool_call_response(self.next_calls))
        if model == "judge":
            if self.judge_fails:
                return httpx.Response(500, text="boom")
            user = body["messages"][-1]["content"]
            deny = '"/"' in user or "rm -rf" in user
            verdict = {"reason": "wipes root" if deny else "routine", "category": "data_loss" if deny else "none",
                       "decision": "deny" if deny else "allow"}
            return httpx.Response(200, json=text_response(json.dumps(verdict)))
        if model == "guard":
            return httpx.Response(200, json=text_response("unsafe\nS14"))
        if model == "refusal-judge":
            return httpx.Response(200, json=text_response("0"))
        return httpx.Response(404, json={"error": {"message": f"model {model} not found"}})


@pytest.fixture
def fake():
    return FakeUpstream()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        upstreams=[Upstream(name="cluster", base_url="http://upstream.test/v1",
                            models=["agent", "judge", "guard", "refusal-judge"])],
        aliases={"gpt-4o-mini": "refusal-judge"},
        model_defaults={"agent": {"chat_template_kwargs": {"enable_thinking": False}}},
        judges={
            "slm": {"kind": "llm_json", "model": "judge", "guided_json": True},
            "guard": {"kind": "guard", "model": "guard", "unsafe_pattern": r"^\s*unsafe",
                      "category_pattern": r"S\d+"},
            "allow-all": {"kind": "static", "decision": "allow"},
        },
        proxy=ProxySettings(audit_dir=str(tmp_path / "audit")),
        paths={"runs_dir": str(tmp_path / "runs")},
        scoring={},
        root=tmp_path,
    )
