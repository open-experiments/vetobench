"""veto-proxy: OpenAI-compatible proxy that puts an SLM judge in front of every tool call.

Benchmarks point ``base_url`` here. The proxy forwards the request to the real model, and when
the response proposes tool calls it asks the judge about each call (shadow/enforce arms),
removes the denied ones (enforce arm), and writes a hash-chained audit record per call.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..audit import AuditLog
from ..config import Settings
from ..judges import ToolCall, Verdict
from ..routing import Route, parse_route
from .gate import JudgeRunner
from .rewrite import normalize_tools, refusal_text, strip_denied_calls
from .upstream import UpstreamPool

RUN_HEADER = "x-vetobench-run"
EPISODE_HEADER = "x-vetobench-episode"


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "vetobench_error"}}, status_code=status)


def create_app(settings: Settings, pool: UpstreamPool | None = None) -> FastAPI:
    pool = pool or UpstreamPool(settings)
    runner = JudgeRunner(settings, pool)
    audit_dir = Path(settings.proxy.audit_dir)
    audit = AuditLog(audit_dir if audit_dir.is_absolute() else settings.root / audit_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await pool.aclose()

    app = FastAPI(title="veto-proxy", version=__version__, lifespan=lifespan)
    app.state.pool, app.state.runner, app.state.audit = pool, runner, audit

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/v1/models")
    async def models() -> dict:
        ids = sorted({m for u in settings.upstreams for m in u.models} | set(settings.aliases))
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "vetobench"} for m in ids]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        model = settings.resolve_alias(parse_route(body.get("model", "")).model)
        r = await pool.post(model, "completions", settings.apply_model_defaults(model, body))
        return JSONResponse(r.json(), status_code=r.status_code)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        try:
            route = parse_route(str(body.get("model", "")))
        except ValueError as e:
            return _error(400, str(e))
        if route.judged and route.judge not in settings.judges:
            return _error(400, f"unknown judge {route.judge!r}; configured: {sorted(settings.judges)}")
        if body.get("stream"):
            return _error(400, "veto-proxy does not support stream=true")

        target = settings.resolve_alias(route.model)
        body["model"] = target
        settings.apply_model_defaults(target, body)
        if body.get("tools"):
            body["tools"] = normalize_tools(body["tools"])

        try:
            r = await pool.post(target, "chat/completions", body)
        except KeyError as e:
            return _error(404, str(e))
        except Exception as e:
            return _error(502, f"upstream error: {type(e).__name__}: {e}")
        if r.status_code >= 300 or not route.logged:
            return JSONResponse(r.json() if r.headers.get("content-type", "").startswith("application/json")
                                else {"error": {"message": r.text}}, status_code=r.status_code)

        resp = r.json()
        blocked = await _gate_response(
            resp, body.get("tools"), route, target,
            run_id=request.headers.get(RUN_HEADER),
            episode_id=request.headers.get(EPISODE_HEADER),
        )
        return JSONResponse(resp, headers={"x-vetobench-blocked": str(blocked)})

    async def _gate_response(resp: dict, tools: list | None, route: Route, target: str,
                             run_id: str | None, episode_id: str | None) -> int:
        request_id = uuid.uuid4().hex
        blocked_total = 0
        for ci, choice in enumerate(resp.get("choices") or []):
            msg = choice.get("message") or {}
            calls = msg.get("tool_calls") or []
            if not calls:
                continue
            views = [ToolCall.from_openai(c, tools) for c in calls]
            if route.judged:
                t0 = time.perf_counter()
                verdicts: list[Verdict | None] = list(
                    await asyncio.gather(*(runner.rule(route.judge, v) for v in views)))
                gate_ms = (time.perf_counter() - t0) * 1000
            else:
                verdicts, gate_ms = [None] * len(views), 0.0

            denied: dict[int, str] = {}
            for i, (view, verdict) in enumerate(zip(views, verdicts)):
                effective = runner.effective(verdict) if verdict else "allow"
                blocked = route.arm == "enforce" and effective == "deny"
                if blocked:
                    denied[i] = refusal_text(settings.proxy.refusal_template, view.name,
                                             verdict.category, verdict.reason)
                audit.append(run_id, {
                    "request_id": request_id,
                    "episode_id": episode_id,
                    "arm": route.arm,
                    "agent_model": target,
                    "judge": route.judge,
                    "choice_index": ci,
                    "call_index": i,
                    "tool_call_id": calls[i].get("id"),
                    "tool": view.to_dict(),
                    "verdict": verdict.to_dict() if verdict else None,
                    "effective_decision": effective if verdict else None,
                    "blocked": blocked,
                    "fail_policy": settings.proxy.fail_policy,
                    "gate_ms": round(gate_ms, 3),
                })
            if denied:
                blocked_total += len(denied)
                result = strip_denied_calls(msg, denied)
                choice["finish_reason"] = result["finish_reason"]
        return blocked_total

    return app
