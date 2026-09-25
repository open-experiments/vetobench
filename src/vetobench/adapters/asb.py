"""Drive ASB (Agent Security Bench) runs through veto-proxy.

ASB's FIFO scheduler serves one LLM request at a time per process, so we split the attacker
tool list into shards and run one ``main_attacker.py`` process per shard in parallel. Each
process runs under ASB's own virtualenv via ``asb_launch.py``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

LAUNCHER = Path(__file__).with_name("asb_launch.py")

ATTACK_TOOL_FILES = {
    "all": "all_attack_tools.jsonl",
    "agg": "all_attack_tools_aggressive.jsonl",
    "non-agg": "all_attack_tools_non_aggressive.jsonl",
}
METHOD_FLAGS = {
    "direct_prompt_injection": ["--direct_prompt_injection"],
    "observation_prompt_injection": ["--observation_prompt_injection"],
    "mixed_attack": ["--direct_prompt_injection", "--observation_prompt_injection"],
    "clean": ["--clean"],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def attacker_tool_names(asb_dir: Path) -> set[str]:
    return {t["Attacker Tool"] for t in read_jsonl(asb_dir / "data" / ATTACK_TOOL_FILES["all"])}


def normal_tool_names(asb_dir: Path) -> set[str]:
    return {t["Tool Name"] for t in read_jsonl(asb_dir / "data" / "all_normal_tools.jsonl")}


def limit_per_agent(tools: list[dict], k: int) -> list[dict]:
    count: dict[str, int] = {}
    out = []
    for t in tools:
        a = t["Corresponding Agent"]
        if count.get(a, 0) < k:
            count[a] = count.get(a, 0) + 1
            out.append(t)
    return out


def prepare_shards(asb_dir: Path, out_dir: Path, attack_tools: str, shards: int, clean: bool,
                   tools_per_agent: int | None = None) -> list[Path]:
    """Write attacker-tool shard files. For the clean (no-attack) run, main_attacker still loops
    over attacker tools, so we keep a single tool per agent to avoid repeating identical tasks."""
    tools = read_jsonl(asb_dir / "data" / ATTACK_TOOL_FILES[attack_tools])
    if clean:
        tools = limit_per_agent(tools, 1)
    elif tools_per_agent:
        tools = limit_per_agent(tools, tools_per_agent)
    shards = max(1, min(shards, len(tools)))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(shards):
        part = tools[i::shards]
        p = out_dir / f"shard{i:02d}.tools.jsonl"
        p.write_text("".join(json.dumps(t) + "\n" for t in part), encoding="utf-8")
        paths.append(p)
    return paths


def csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_attack(*, asb_dir: Path, asb_python: Path, out_dir: Path, model_route: str, run_id: str,
               proxy_url: str, method: str, attack_type: str = "naive", attack_tools: str = "all",
               shards: int = 8, task_num: int = 1, max_new_tokens: int = 1024,
               tools_per_agent: int | None = None) -> list[Path]:
    """Run one ASB attack configuration; returns the per-shard result CSVs. Shards whose CSV is
    already complete are skipped, so an interrupted run can be resumed."""
    if method not in METHOD_FLAGS:
        raise ValueError(f"unknown ASB method {method!r}; expected one of {sorted(METHOD_FLAGS)}")
    clean = method == "clean"
    if clean:
        task_num = max(task_num, 5)  # every agent has 5 tasks; use them all for the utility run
    shard_files = prepare_shards(asb_dir, out_dir, attack_tools, shards, clean, tools_per_agent)
    env = {**os.environ, "OPENAI_BASE_URL": proxy_url.rstrip("/"), "OPENAI_API_KEY": "vetobench",
           "PYTHONUNBUFFERED": "1"}
    procs: list[tuple[subprocess.Popen, Path, object]] = []
    results = []
    for shard in shard_files:
        res = shard.with_name(shard.name.replace(".tools.jsonl", ".csv"))
        results.append(res)
        expected = len(read_jsonl(shard)) * task_num
        if len(csv_rows(res)) >= expected:
            continue
        cmd = [
            str(asb_python), str(LAUNCHER), "--asb-dir", str(asb_dir), "--run-id", run_id, "--",
            "--llm_name", model_route,
            "--attack_type", attack_type,
            "--attacker_tools_path", str(shard.resolve()),
            "--res_file", str(res.resolve()),
            "--database", str((out_dir / "no_memory_db").resolve()),  # nonexistent: disables memory
            "--task_num", str(task_num),
            "--max_new_tokens", str(max_new_tokens),
            *METHOD_FLAGS[method],
        ]
        log = open(res.with_suffix(".log"), "w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), shard, log))
    failed = []
    for p, shard, log in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            failed.append(f"{shard.name} (exit {rc}, see {shard.with_suffix('').with_suffix('.log')})")
    if failed:
        raise RuntimeError("ASB shards failed: " + "; ".join(failed))
    return results
