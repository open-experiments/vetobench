"""Run Agent-SafetyBench cases through veto-proxy.

This is a faithful port of ``evaluation/eval.py`` (the agent loop) using the benchmark's own
environments (``environments/``) and its OpenAI-path system prompt, with three additions:
a case-id subset, thread-level concurrency, and headers so the proxy can tie every audit
record to its case. Output ``gen_res.json`` has the same shape ``score/eval_with_shield.py``
expects, so their scorer and ours both work on it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from openai import OpenAI

MAX_ROUNDS = 10  # as in eval.py


def load_cases(asbench_dir: Path, ids: list[int] | None = None) -> list[dict]:
    data = json.loads((asbench_dir / "data" / "released_data.json").read_text(encoding="utf-8"))
    if ids is None:
        return data
    wanted = set(ids)
    return [d for d in data if d["id"] in wanted]


def _system_prompt(asbench_dir: Path) -> str:
    # model_api/__init__.py imports modules that are missing from the repo, so load BaseAPI directly.
    spec = importlib.util.spec_from_file_location(
        "asbench_BaseAPI", asbench_dir / "evaluation" / "model_api" / "BaseAPI.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # OpenaiAPI (the native tool-calling path) uses this variant.
    return mod.BaseAPI().without_strict_jsonformat_sys_prompt


class AgentSafetyBenchRunner:
    def __init__(self, asbench_dir: Path, base_url: str, model: str, run_id: str,
                 temperature: float = 0.0, max_tokens: int = 2048, request_retries: int = 10):
        self.dir = Path(asbench_dir).resolve()
        env_dir = str(self.dir / "environments")
        if env_dir not in sys.path:
            sys.path.insert(0, env_dir)
        from EnvManager import EnvManager  # type: ignore  # noqa: E402  (benchmark module)

        self.env_manager = EnvManager()
        self.system_prompt = _system_prompt(self.dir)
        self.model, self.run_id = model, run_id
        self.gen = {"temperature": temperature, "max_tokens": max_tokens}
        self.retries = request_retries
        self.client = OpenAI(base_url=base_url, api_key="vetobench", max_retries=0, timeout=900,
                             default_headers={"x-vetobench-run": run_id})

    def _envs(self, d: dict):
        envs, tool_descs = [], []
        if not d["environments"] or d["environments"][0]["name"] == "":
            return envs, tool_descs
        for env_info in d["environments"]:
            env = self.env_manager.init_env(env_info["name"], env_info.get("parameters") or None)
            if env is None:
                raise ValueError(f"environment {env_info['name']} not found")
            envs.append(env)
            tool_descs.extend(env.get_tool_descs(env_info["tools"]))
        return envs, [{"type": "function", "function": deepcopy(t)} for t in tool_descs]

    def _complete(self, messages: list[dict], tools: list[dict], episode: str):
        """One agent step. Returns (completion dict or None, blocked_count)."""
        for attempt in range(self.retries):
            try:
                raw = self.client.chat.completions.with_raw_response.create(
                    model=self.model, messages=messages, tools=tools or None,
                    extra_headers={"x-vetobench-episode": episode}, **self.gen)
                blocked = int(raw.headers.get("x-vetobench-blocked", "0") or 0)
                c = raw.parse()
                if not c.choices:
                    continue
                msg = c.choices[0].message
                if msg.tool_calls:
                    call = msg.tool_calls[0]  # eval.py executes only the first call per turn
                    try:
                        args = json.loads(call.function.arguments) if call.function.arguments else {}
                    except json.JSONDecodeError as e:
                        return {"type": "error", "message": f"bad tool arguments: {e}"}, blocked
                    return {"type": "tool", "tool_call_id": call.id, "tool_name": call.function.name,
                            "arguments": args}, blocked
                return {"type": "content", "content": msg.content}, blocked
            except Exception as e:  # network/5xx: back off and retry like eval.py
                print(f"[{episode}] request failed ({attempt + 1}/{self.retries}): {e}", file=sys.stderr)
                time.sleep(min(2 ** attempt, 30))
        return None, 0

    def evaluate(self, d: dict) -> dict:
        episode = f"asbench:{d['id']}"
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(d["dialog"] if "dialog" in d else [{"role": "user", "content": d["instruction"]}])
        envs, tools = self._envs(d)
        rounds, blocked_total, final = 0, 0, None
        while rounds < MAX_ROUNDS:
            completion, blocked = self._complete(messages, tools, episode)
            blocked_total += blocked
            rounds += 1
            if completion is None or completion["type"] == "error":
                continue
            if completion["type"] == "tool":
                name, args = completion["tool_name"], completion["arguments"]
                result = None
                for env_idx, env in enumerate(envs):
                    if name not in d["environments"][env_idx]["tools"] or not env.has_tool(name):
                        continue
                    try:
                        result = env.call_tool(name, deepcopy(args))
                    except Exception as e:  # buggy env/tool args; eval.py would drop the whole case
                        result = {"success": False, "message": f"Tool error: {e}"}
                    break
                if result is None:
                    result = {"success": False, "message": f"Tool {name} doesn't exist."}
                messages.append({"role": "assistant", "tool_calls": [{
                    "id": completion["tool_call_id"], "type": "function",
                    "function": {"arguments": json.dumps(args), "name": name}}]})
                messages.append({"role": "tool", "content": json.dumps(result),
                                 "tool_call_id": completion["tool_call_id"], "name": name})
            else:
                messages.append({"role": "assistant", "content": completion["content"]})
                final = messages
                break

        if final is None:
            status = "exceeded" if rounds == MAX_ROUNDS else "error"
            final = messages
        elif not (final[-1].get("content") or "").strip():
            status = "error"  # eval.py discards empty final answers
        else:
            status = "ok"
        out = dict(d)
        out["output"] = final
        out["vetobench"] = {"status": status, "rounds": rounds, "blocked_calls": blocked_total,
                            "model": self.model, "run_id": self.run_id}
        return out


def run(asbench_dir: Path, out_dir: Path, base_url: str, model: str, run_id: str,
        case_ids: list[int] | None = None, workers: int = 16, **runner_kwargs) -> Path:
    """Run (or resume) a set of cases; returns the path of gen_res.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "gen_res.jsonl"
    done: dict[int, dict] = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["vetobench"]["status"] != "error":  # retry errors on resume
                    done[rec["id"]] = rec
    runner = AgentSafetyBenchRunner(asbench_dir, base_url, model, run_id, **runner_kwargs)
    todo = [d for d in load_cases(runner.dir, case_ids) if d["id"] not in done]
    print(f"[asbench] {model}: {len(done)} done, {len(todo)} to run", file=sys.stderr)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool, open(partial, "a", encoding="utf-8") as f:
        futures = {pool.submit(runner.evaluate, d): d["id"] for d in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"[asbench] case {futures[fut]} crashed: {e}", file=sys.stderr)
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[rec["id"]] = rec
            if i % 25 == 0 or i == len(todo):
                print(f"[asbench] {model}: {i}/{len(todo)}", file=sys.stderr)
    final = out_dir / "gen_res.json"
    final.write_text(json.dumps(sorted(done.values(), key=lambda r: r["id"]), ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return final
