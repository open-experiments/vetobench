"""Request normalisation and response rewriting for the proxy."""

from __future__ import annotations

import copy

EMPTY_PARAMS = {"type": "object", "properties": {}}


def normalize_tools(tools: list[dict] | None) -> list[dict] | None:
    """Give every function a JSON-schema ``parameters`` object.

    ASB declares tools with ``parameters: None`` or no parameters at all, which vLLM rejects.
    This is applied in every arm, including pass-through, so arms stay comparable.
    """
    if not tools:
        return tools
    out = copy.deepcopy(tools)
    for t in out:
        fn = t.get("function")
        if isinstance(fn, dict) and not isinstance(fn.get("parameters"), dict):
            fn["parameters"] = dict(EMPTY_PARAMS)
    return out



def merge_leading_system(messages: list[dict] | None) -> list[dict] | None:
    """Merge the system messages that open a conversation into one.

    ASB starts every conversation with several system messages (role, tool list, planning
    instructions). Some chat templates, e.g. Qwen3.5/3.8, reject a system message anywhere but
    first. Contents are joined in order, so nothing is reordered or dropped. Applied in every
    arm, like ``normalize_tools``.
    """
    if not messages:
        return messages
    n = 0
    while n < len(messages) and messages[n].get("role") == "system" \
            and isinstance(messages[n].get("content"), str):
        n += 1
    if n < 2:
        return messages
    merged = {"role": "system", "content": "\n\n".join(m["content"] for m in messages[:n])}
    return [merged, *messages[n:]]


def refusal_text(template: str, tool_name: str, category: str, reason: str) -> str:
    return template.format(tool_name=tool_name, category=category, reason=reason or "no reason given")


def strip_denied_calls(message: dict, denied: dict[int, str]) -> dict:
    """Remove denied tool calls (by index) from an assistant message.

    ``denied`` maps call index -> refusal sentence. If no calls remain, the message becomes a
    plain assistant reply containing the refusal(s), so the agent loop ends the turn the
    same way it would after a model refusal. Returns the new ``finish_reason``.
    """
    calls = message.get("tool_calls") or []
    kept = [c for i, c in enumerate(calls) if i not in denied]
    notes = " ".join(denied[i] for i in sorted(denied))
    if kept:
        message["tool_calls"] = kept
        return {"finish_reason": "tool_calls", "note": notes}
    message["tool_calls"] = None
    prior = (message.get("content") or "").strip()
    message["content"] = f"{prior}\n\n{notes}".strip() if prior else notes
    return {"finish_reason": "stop", "note": notes}
