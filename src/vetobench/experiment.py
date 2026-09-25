"""Experiment matrix: models x variants x benchmarks, with idempotent run/score/replay/report steps.

Layout under ``run_dir``::

    asbench/<model>/<variant>/gen_res.json(l), shield.jsonl
    asb/<model>/<variant>/<attack>/shardNN.csv
    audit/<run_id>.jsonl              (written by the proxy; run_id = <exp>.<bench>.<model>.<variant>)
    replay/<bench>.<model>.jsonl      (offline judge rulings on baseline calls)
    report.md, summary.json
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .adapters import agent_safetybench as asbench_adapter
from .adapters import asb as asb_adapter
from .audit import read_records
from .config import Settings, load_yaml
from .proxy.upstream import UpstreamPool
from .routing import format_route
from .scoring import judge_eval, metrics, shieldagent
from .scoring.stats import paired_bootstrap_diff

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def slug(s: str) -> str:
    return _SAFE.sub("_", s)


@dataclass
class Variant:
    arm: str
    judge: str | None = None

    @property
    def name(self) -> str:
        return self.arm if not self.judge else f"{self.arm}-{self.judge}"


@dataclass
class Experiment:
    name: str
    run_dir: Path
    models: list[str]
    judges: list[str]
    variants: list[Variant]
    proxy_url: str
    asbench: dict = field(default_factory=dict)
    asb: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, settings: Settings) -> "Experiment":
        raw = load_yaml(path)
        judges = raw.get("judges", [])
        for j in judges:
            if j not in settings.judges:
                raise ValueError(f"experiment judge {j!r} is not configured in the settings file")
        variants = []
        for arm in raw.get("arms", ["baseline", "enforce"]):
            if arm in ("shadow", "enforce"):
                variants += [Variant(arm, j) for j in judges]
            else:
                variants.append(Variant(arm))
        run_dir = Path(raw.get("run_dir", f"runs/{raw['name']}"))
        return cls(
            name=raw["name"],
            run_dir=run_dir if run_dir.is_absolute() else settings.root / run_dir,
            models=raw["models"],
            judges=judges,
            variants=variants,
            proxy_url=raw.get("proxy_url", f"http://{settings.proxy.host}:{settings.proxy.port}/v1"),
            asbench=raw.get("agent_safetybench") or {},
            asb=raw.get("asb") or {},
        )

    def run_id(self, bench: str, model: str, variant: Variant) -> str:
        return f"{self.name}.{bench}.{slug(model)}.{variant.name}"

    def audit_path(self, settings: Settings, run_id: str) -> Path:
        d = Path(settings.proxy.audit_dir)
        d = d if d.is_absolute() else settings.root / d
        return d / f"{slug(run_id)}.jsonl"

    def select(self, models: list[str] | None, variants: list[str] | None):
        ms = [m for m in self.models if not models or m in models]
        vs = [v for v in self.variants if not variants or v.name in variants]
        return ms, vs


# ------------------------------------------------------------------ helpers

def check_proxy(proxy_url: str) -> None:
    base = proxy_url.rstrip("/").removesuffix("/v1")
    try:
        httpx.get(f"{base}/healthz", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"veto-proxy is not reachable at {base} ({e}). Start it with: vetobench proxy -c <config>")


def split_ids(exp: Experiment, settings: Settings) -> list[int] | None:
    split = exp.asbench.get("split")
    if not split:
        return None
    p = Path(split)
    p = p if p.is_absolute() else settings.root / p
    data = json.loads(p.read_text())
    return data[exp.asbench.get("split_part", "test")]


def make_split(asbench_dir: Path, n_test: int, n_dev: int, seed: int, out: Path) -> dict:
    """Stratified dev/test split of Agent-SafetyBench by risk category (disjoint)."""
    cases = asbench_adapter.load_cases(asbench_dir)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for c in cases:
        by_cat[(c.get("risks") or ["(none)"])[0]].append(c["id"])
    rng = random.Random(seed)
    cats = sorted(by_cat)
    dev, test = [], []
    for i, cat in enumerate(cats):
        ids = sorted(by_cat[cat])
        rng.shuffle(ids)
        # spread the remainder over the first categories so totals are exact
        nd = n_dev // len(cats) + (1 if i < n_dev % len(cats) else 0)
        nt = n_test // len(cats) + (1 if i < n_test % len(cats) else 0)
        dev += ids[:nd]
        test += ids[nd:nd + nt]
    split = {"seed": seed, "dev": sorted(dev), "test": sorted(test),
             "note": "dev is for judge prompt development only; report numbers on test"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1))
    return split


# ------------------------------------------------------------------ steps

def run(exp: Experiment, settings: Settings, benches: list[str], models=None, variants=None) -> None:
    check_proxy(exp.proxy_url)
    ms, vs = exp.select(models, variants)
    for model in ms:
        for v in vs:
            route = format_route(model, v.arm, v.judge)
            if "asbench" in benches and exp.asbench.get("enabled", True):
                rid = exp.run_id("asbench", model, v)
                print(f"== Agent-SafetyBench | {model} | {v.name}")
                asbench_adapter.run(
                    settings.path("asbench_dir"), exp.run_dir / "asbench" / slug(model) / v.name,
                    base_url=exp.proxy_url, model=route, run_id=rid, case_ids=split_ids(exp, settings),
                    workers=exp.asbench.get("workers", 16),
                    max_tokens=exp.asbench.get("max_tokens", 2048))
            if "asb" in benches and exp.asb.get("enabled", True):
                rid = exp.run_id("asb", model, v)
                for atk in exp.asb.get("attacks", []):
                    tag = atk["method"] if atk["method"] == "clean" else f"{atk['method']}-{atk['attack_type']}"
                    print(f"== ASB | {model} | {v.name} | {tag}")
                    asb_adapter.run_attack(
                        asb_dir=settings.path("asb_dir"), asb_python=settings.path("asb_python"),
                        out_dir=exp.run_dir / "asb" / slug(model) / v.name / tag,
                        model_route=route, run_id=rid, proxy_url=exp.proxy_url,
                        method=atk["method"], attack_type=atk.get("attack_type", "naive"),
                        attack_tools=exp.asb.get("attack_tools", "all"),
                        shards=exp.asb.get("shards", 8), task_num=exp.asb.get("task_num", 1),
                        max_new_tokens=exp.asb.get("max_new_tokens", 1024),
                        tools_per_agent=exp.asb.get("tools_per_agent"))


def score(exp: Experiment, settings: Settings, models=None, variants=None) -> None:
    model = settings.scoring.get("shieldagent_model")
    if not model:
        sys.exit("set scoring.shieldagent_model in the config")
    ms, vs = exp.select(models, variants)
    env_dir = settings.path("asbench_dir") / "environments"

    async def go():
        pool = UpstreamPool(settings)
        try:
            for m in ms:
                for v in vs:
                    d = exp.run_dir / "asbench" / slug(m) / v.name
                    if (d / "gen_res.json").exists():
                        print(f"== ShieldAgent | {m} | {v.name}")
                        await shieldagent.score_file(pool, model, d / "gen_res.json", env_dir, d / "shield.jsonl",
                                                     concurrency=settings.scoring.get("concurrency", 32))
        finally:
            await pool.aclose()

    asyncio.run(go())


def replay(exp: Experiment, settings: Settings, models=None, judges=None) -> None:
    ms, _ = exp.select(models, None)
    judges = judges or exp.judges
    base = Variant("baseline")
    for m in ms:
        for bench in ("asbench", "asb"):
            audit = exp.audit_path(settings, exp.run_id(bench, m, base))
            if audit.exists():
                print(f"== replay | {bench} | {m} | judges={judges}")
                asyncio.run(judge_eval.replay(settings, [audit], judges,
                                              exp.run_dir / "replay" / f"{bench}.{slug(m)}.jsonl"))


# ------------------------------------------------------------------ report

def _pct(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def _ci(d: dict, key: str) -> str:
    return f"{_pct(d[key])} [{_pct(d['lo'])}, {_pct(d['hi'])}]"


def _diff(t: tuple) -> str:
    d, lo, hi, n = t
    return "–" if n == 0 else f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] (n={n})"


def report(exp: Experiment, settings: Settings) -> Path:
    lines = [f"# vetobench report: {exp.name}", ""]
    summary: dict = {"asbench": {}, "asb": {}, "judges": {}}
    base = Variant("baseline")

    # Agent-SafetyBench
    lines += ["## Agent-SafetyBench (ShieldAgent safety score, % safe, 95% CI)", "",
              "| model | variant | n | safe % | Δ vs baseline (paired) | episodes w/ block |",
              "|---|---|---|---|---|---|"]
    for m in exp.models:
        base_s = None
        for v in exp.variants:
            d = exp.run_dir / "asbench" / slug(m) / v.name
            if not (d / "shield.jsonl").exists():
                continue
            s = metrics.asbench_summary(d / "gen_res.json", d / "shield.jsonl")
            if v.name == base.name:
                base_s = s
            diff = paired_bootstrap_diff(base_s["safe_by_id"], s["safe_by_id"]) if base_s and v.name != base.name else (0, 0, 0, 0)
            lines.append(f"| {m} | {v.name} | {s['n_scored']} | {_ci(s['overall'], 'safe')} | {_diff(diff)} | {s['episodes_with_block']} |")
            summary["asbench"].setdefault(m, {})[v.name] = {k: s[k] for k in s if k != "safe_by_id"}
    lines += ["", "### Per risk category (safe %)", ""]
    cats = sorted({c for m in summary["asbench"].values() for v in m.values() for c in v["by_category"]})
    if cats:
        lines += ["| model | variant | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
        for m, vs in summary["asbench"].items():
            for vn, s in vs.items():
                lines.append(f"| {m} | {vn} | " + " | ".join(_pct(s["by_category"].get(c, {}).get("safe", math.nan)) for c in cats) + " |")

    # ASB
    lines += ["", "## ASB (attack success rate, original-task success, refusal rate; %, 95% CI)", "",
              "| model | attack | variant | n | ASR | ΔASR vs baseline (paired) | task success | refusal |",
              "|---|---|---|---|---|---|---|---|"]
    for m in exp.models:
        for atk in exp.asb.get("attacks", []):
            tag = atk["method"] if atk["method"] == "clean" else f"{atk['method']}-{atk['attack_type']}"
            base_a = None
            for v in exp.variants:
                d = exp.run_dir / "asb" / slug(m) / v.name / tag
                csvs = sorted(d.glob("shard*.csv"))
                if not csvs:
                    continue
                a = metrics.asb_summary(csvs)
                if v.name == base.name:
                    base_a = a
                diff = paired_bootstrap_diff(base_a["asr_by_key"], a["asr_by_key"]) if base_a and v.name != base.name else (0, 0, 0, 0)
                lines.append(f"| {m} | {tag} | {v.name} | {a['n']} | {_ci(a['asr'], 'p')} | {_diff(diff)} | "
                             f"{_ci(a['original_task_success'], 'p')} | {_ci(a['refusal_rate'], 'p')} |")
                summary["asb"].setdefault(m, {}).setdefault(tag, {})[v.name] = {
                    k: a[k] for k in a if not k.endswith("_by_key")}

    # Judge quality (offline replay on baseline calls)
    lines += ["", "## Judge quality on baseline tool calls (offline replay)", "",
              "| model | bench | judge | precision | recall | F1 | FPR | p50 ms | p95 ms | errors | notes |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    asb_dir = settings.path("asb_dir")
    attackers, normals = asb_adapter.attacker_tool_names(asb_dir), asb_adapter.normal_tool_names(asb_dir)
    for m in exp.models:
        for bench in ("asbench", "asb"):
            rp = exp.run_dir / "replay" / f"{bench}.{slug(m)}.jsonl"
            if not rp.exists():
                continue
            recs = judge_eval.read_records_plain(rp)
            for j in sorted({r["judge"] for r in recs}):
                decs = judge_eval.decisions(recs, j)
                if not decs:
                    continue
                lat = judge_eval.latency_summary(decs)
                if bench == "asb":
                    c = judge_eval.asb_call_confusion(decs, attackers, normals)
                    note = f"per call; {c['unknown_tool_calls']} unknown-tool calls skipped"
                else:
                    bd = exp.run_dir / "asbench" / slug(m) / "baseline"
                    if not (bd / "shield.jsonl").exists():
                        continue
                    cases = json.loads((bd / "gen_res.json").read_text())
                    c = judge_eval.asbench_episode_confusion(decs, cases, metrics.load_scores(bd / "shield.jsonl"))
                    note = (f"per episode; {c['unsafe_without_tool_calls']}/{c['episodes_without_tool_calls']} "
                            "no-call episodes unsafe (unreachable)")
                lines.append(f"| {m} | {bench} | {j} | {_pct(c['precision'])} | {_pct(c['recall'])} | {_pct(c['f1'])} | "
                             f"{_pct(c['fpr'])} | {lat['p50_ms']:.0f} | {lat['p95_ms']:.0f} | {lat['errors']} | {note} |")
                summary["judges"].setdefault(m, {}).setdefault(bench, {})[j] = {**c, **lat}

    # Live gate latency (enforce arms)
    lat_lines = []
    for m in exp.models:
        for v in exp.variants:
            if not v.judge:
                continue
            for bench in ("asbench", "asb"):
                p = exp.audit_path(settings, exp.run_id(bench, m, v))
                if p.exists():
                    lat = judge_eval.latency_summary([r for r in read_records(p) if r.get("verdict")])
                    lat_lines.append(f"| {m} | {bench} | {v.name} | {lat['n_calls']} | {lat['p50_ms']:.0f} | "
                                     f"{lat['p95_ms']:.0f} | {lat['errors']} |")
    if lat_lines:
        lines += ["", "## Live judge latency (uncached calls)", "",
                  "| model | bench | variant | calls | p50 ms | p95 ms | errors |", "|---|---|---|---|---|---|---|",
                  *lat_lines]

    exp.run_dir.mkdir(parents=True, exist_ok=True)
    out = exp.run_dir / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (exp.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return out
