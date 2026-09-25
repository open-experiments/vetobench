"""Judge dispatch shared by the live proxy and offline replay: caching, timeouts, errors."""

from __future__ import annotations

import asyncio

from ..config import Settings
from ..judges import Judge, ToolCall, Verdict, build_judge
from ..judges.base import ChatBackend


class JudgeRunner:
    def __init__(self, settings: Settings, backend: ChatBackend):
        self.settings = settings
        self.backend = backend
        self._judges: dict[str, Judge] = {}
        self._cache: dict[str, Verdict] = {}
        self._sem = asyncio.Semaphore(settings.proxy.max_concurrent_judges)

    def judge(self, name: str) -> Judge:
        if name not in self._judges:
            if name not in self.settings.judges:
                raise KeyError(f"unknown judge {name!r}; configured: {sorted(self.settings.judges)}")
            spec = {"timeout_s": self.settings.proxy.judge_timeout_s, **self.settings.judges[name]}
            if spec.get("kind") == "static":
                spec.pop("timeout_s")
            self._judges[name] = build_judge(name, spec, self.backend)
        return self._judges[name]

    async def rule(self, judge_name: str, call: ToolCall) -> Verdict:
        judge = self.judge(judge_name)
        key = call.cache_key(judge_name)
        if self.settings.proxy.judge_cache and key in self._cache:
            hit = self._cache[key]
            return Verdict(hit.decision, hit.category, hit.reason, hit.raw, 0.0, True, {})
        try:
            async with self._sem:
                verdict = await asyncio.wait_for(judge.judge(call), self.settings.proxy.judge_timeout_s)
        except asyncio.TimeoutError:
            verdict = Verdict("error", "other", f"judge timed out after {self.settings.proxy.judge_timeout_s}s")
        except Exception as e:  # upstream down, bad response, ...: the fail policy decides
            verdict = Verdict("error", "other", f"{type(e).__name__}: {e}"[:500])
        if verdict.decision != "error" and self.settings.proxy.judge_cache:
            self._cache[key] = verdict
        return verdict

    def effective(self, verdict: Verdict) -> str:
        """allow/deny after applying the fail policy to judge errors."""
        if verdict.decision == "error":
            return "deny" if self.settings.proxy.fail_policy == "closed" else "allow"
        return verdict.decision
