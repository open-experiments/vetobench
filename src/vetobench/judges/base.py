"""Judge interface: rule on one proposed tool call, seeing only the call itself."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

MAX_ARG_CHARS = 4000

CATEGORIES = (
    "none", "data_loss", "financial_loss", "privacy_leak", "security_compromise",
    "physical_harm", "harmful_content", "illegal_or_unethical", "availability", "other",
)


@dataclass
class ToolCall:
    """A tool call as the judge sees it: name, schema, and the argument values."""

    name: str
    description: str = ""
    parameters: dict | None = None
    arguments: Any = None  # parsed dict when the model produced valid JSON, raw string otherwise

    @classmethod
    def from_openai(cls, call: dict, tools: list[dict] | None) -> "ToolCall":
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        except json.JSONDecodeError:
            args = raw
        desc, params = "", None
        for t in tools or []:
            f = t.get("function") or {}
            if f.get("name") == name:
                desc, params = f.get("description") or "", f.get("parameters")
                break
        return cls(name=name, description=desc, parameters=params, arguments=args)

    def arguments_text(self) -> str:
        text = self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments, ensure_ascii=False)
        return text if len(text) <= MAX_ARG_CHARS else text[:MAX_ARG_CHARS] + " …[truncated]"

    def schema_text(self) -> str:
        return json.dumps(self.parameters, ensure_ascii=False) if self.parameters else "(none)"

    def cache_key(self, judge: str) -> str:
        blob = json.dumps(
            [judge, self.name, self.description, self.parameters, self.arguments],
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    decision: str  # "allow" | "deny" | "error"
    category: str = "none"
    reason: str = ""
    raw: str = ""
    latency_ms: float = 0.0
    cached: bool = False
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ChatBackend(Protocol):
    """Anything that can POST an OpenAI chat-completions payload for a served model."""

    async def chat(self, model: str, payload: dict, timeout_s: float | None = None) -> dict: ...


class Judge:
    name: str

    async def judge(self, call: ToolCall) -> Verdict:
        raise NotImplementedError
