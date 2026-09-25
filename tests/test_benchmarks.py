"""Checks against the real benchmark checkouts in third_party/ (skipped if absent)."""

import csv
import json
import sys
from pathlib import Path

import pytest

from vetobench.adapters import asb
from vetobench.experiment import make_split
from vetobench.scoring import judge_eval, metrics, shieldagent
from vetobench.scoring.stats import binary_metrics, paired_bootstrap_diff, wilson

ROOT = Path(__file__).resolve().parents[1]
ASBENCH = ROOT / "third_party" / "Agent-SafetyBench"
ASB = ROOT / "third_party" / "ASB"
needs_asbench = pytest.mark.skipif(not ASBENCH.exists(), reason="Agent-SafetyBench not checked out")
needs_asb = pytest.mark.skipif(not ASB.exists(), reason="ASB not checked out")


@needs_asbench
def test_split_is_stratified_and_disjoint(tmp_path):
    split = make_split(ASBENCH, n_test=300, n_dev=48, seed=0, out=tmp_path / "s.json")
    assert len(split["test"]) == 300 and len(split["dev"]) == 48
    assert not set(split["test"]) & set(split["dev"])
    cases = {c["id"]: c for c in json.loads((ASBENCH / "data" / "released_data.json").read_text())}
    per_cat = {}
    for i in split["test"]:
        per_cat[cases[i]["risks"][0]] = per_cat.get(cases[i]["risks"][0], 0) + 1
    assert len(per_cat) == 8 and max(per_cat.values()) - min(per_cat.values()) <= 1
    assert make_split(ASBENCH, 300, 48, 0, tmp_path / "t.json")["test"] == split["test"]


@needs_asbench
def test_shieldagent_prompt_matches_official_format():
    case = next(c for c in json.loads((ASBENCH / "data" / "released_data.json").read_text())
                if c["environments"] and c["environments"][0]["name"])
    tool = case["environments"][0]["tools"][0]
    case["output"] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": case.get("instruction", "")},
        {"role": "assistant", "tool_calls": [{"id": "c0", "type": "function",
                                              "function": {"name": tool, "arguments": "{}"}}]},
        {"role": "tool", "content": "{\"success\": true}", "tool_call_id": "c0", "name": tool},
        {"role": "assistant", "content": "done"},
    ]
    p = shieldagent.build_prompt(case, ASBENCH / "environments")
    assert p.startswith("<|im_start|>system\n<|im_end|>\n<|im_start|>user\nYou are an expert in safety evaluation")
    assert f'"tool_name": "{tool}"' in p and "<Assistant>: done" in p
    assert p.endswith("<|im_end|>\n<|im_start|>assistant\n")


@pytest.mark.parametrize("text,label", [("[Answer] unsafe\n[Analysis]", 1), ("[Answer] safe\n", 0),
                                        ("unsafe", 1), ("[Answer] maybe", -1), ("", -1)])
def test_parse_shield_label(text, label):
    assert shieldagent.parse_label(text) == label


@needs_asb
def test_asb_shards_cover_all_tools(tmp_path):
    paths = asb.prepare_shards(ASB, tmp_path, "all", 8, clean=False)
    names = [json.loads(line)["Attacker Tool"] for p in paths for line in p.read_text().splitlines()]
    assert len(names) == 400 and len(paths) == 8
    clean = asb.prepare_shards(ASB, tmp_path / "c", "all", 8, clean=True)
    assert sum(len(p.read_text().splitlines()) for p in clean) == 10  # one per agent
    few = asb.prepare_shards(ASB, tmp_path / "f", "all", 2, clean=False, tools_per_agent=2)
    assert sum(len(p.read_text().splitlines()) for p in few) == 20


@needs_asb
def test_asb_tool_sets_disjoint():
    attackers, normals = asb.attacker_tool_names(ASB), asb.normal_tool_names(ASB)
    assert len(attackers) == 400 and len(normals) == 20 and not attackers & normals


def _write_asb_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Agent Name", "Attack Tool", "Attack Successful", "Original Task Successful",
                    "Refuse Result", "Memory Found", "Aggressive", "messages"])
        w.writerows(rows)


def test_asb_summary_and_pairing(tmp_path):
    _write_asb_csv(tmp_path / "base.csv", [["a", "T1", 1, 1, 0, "N/A", 1, "[]"], ["a", "T2", 1, 0, 0, "N/A", 0, "[]"]])
    _write_asb_csv(tmp_path / "enf.csv", [["a", "T1", 0, 1, 1, "N/A", 1,
                                           "[{'content': '[Thinking]: I did not perform this action. The tool call `T1` was "
                                           "blocked by the tool-call safety judge (x): y'}]"],
                                          ["a", "T2", 1, 0, 0, "N/A", 0, "[]"]])
    b = metrics.asb_summary([tmp_path / "base.csv"])
    e = metrics.asb_summary([tmp_path / "enf.csv"])
    assert b["asr"]["p"] == 1.0 and e["asr"]["p"] == 0.5 and e["episodes_with_block"] == 1
    assert e["asr_aggressive"]["p"] == 0.0 and e["asr_non_aggressive"]["p"] == 1.0
    d, lo, hi, n = paired_bootstrap_diff(b["asr_by_key"], e["asr_by_key"])
    assert n == 2 and d == -0.5 and lo <= d <= hi


def test_asb_call_confusion():
    decs = [{"tool": {"name": "Evil"}, "effective_decision": "deny"},
            {"tool": {"name": "Evil"}, "effective_decision": "allow"},
            {"tool": {"name": "sys_monitor"}, "effective_decision": "allow"},
            {"tool": {"name": "sys_monitor"}, "effective_decision": "deny"},
            {"tool": {"name": "made_up"}, "effective_decision": "deny"}]
    c = judge_eval.asb_call_confusion(decs, {"Evil"}, {"sys_monitor"})
    assert (c["tp"], c["fn"], c["tn"], c["fp"], c["unknown_tool_calls"]) == (1, 1, 1, 1, 1)


def test_asbench_episode_confusion():
    call = {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}]}
    cases = [{"id": 1, "output": [call]}, {"id": 2, "output": [call]}, {"id": 3, "output": [{"role": "assistant", "content": "no"}]}]
    decs = [{"episode_id": "asbench:1", "effective_decision": "deny"},
            {"episode_id": "asbench:2", "effective_decision": "allow"}]
    c = judge_eval.asbench_episode_confusion(decs, cases, {1: 1, 2: 0, 3: 1})
    assert (c["tp"], c["tn"], c["episodes_without_tool_calls"], c["unsafe_without_tool_calls"]) == (1, 1, 1, 1)


def test_stats():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and 0.2 < lo < 0.3 and 0.7 < hi < 0.8
    m = binary_metrics([(1, 1), (1, 0), (0, 0), (0, 1)])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5


@needs_asbench
def test_asbench_runner_against_fake_upstream(tmp_path, monkeypatch):
    """Full agent loop on real environments, via the proxy app served in-process."""
    import threading

    import uvicorn

    from tests.fake_vllm import app as fake_app
    from vetobench.adapters import agent_safetybench
    from vetobench.config import ProxySettings, Settings, Upstream
    from vetobench.proxy.app import create_app

    def serve(app, port):
        server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        while not server.started:
            pass
        return server

    fake = serve(fake_app, 18101)
    settings = Settings(
        upstreams=[Upstream("u", "http://127.0.0.1:18101/v1", models=["qwen3.8-27b", "judge-small"])],
        aliases={}, model_defaults={}, judges={"small": {"kind": "llm_json", "model": "judge-small"}},
        proxy=ProxySettings(audit_dir=str(tmp_path / "audit")), paths={}, scoring={}, root=tmp_path)
    proxy = serve(create_app(settings), 18102)
    try:
        out = agent_safetybench.run(ASBENCH, tmp_path / "out", "http://127.0.0.1:18102/v1",
                                    "qwen3.8-27b@shadow:small", "t", case_ids=[0, 1, 2, 3], workers=4)
        res = json.loads(out.read_text())
        assert [r["id"] for r in res] == [0, 1, 2, 3]
        assert all(r["vetobench"]["status"] == "ok" for r in res)
        audit = (tmp_path / "audit" / "t.jsonl").read_text().splitlines()
        assert audit and all(json.loads(a)["episode_id"].startswith("asbench:") for a in audit)
    finally:
        fake.should_exit = proxy.should_exit = True
        sys.path[:] = [p for p in sys.path if "Agent-SafetyBench" not in p]
