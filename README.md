# vetobench

Measures how much a **small-language-model (SLM) tool-call judge** reduces unsafe agent actions
on two industry benchmarks, **Agent-SafetyBench** (THU-CoAI, 2,000 cases, 349 environments) and
**ASB / Agent Security Bench** (400 attacker tools, 10 agents, prompt-injection attacks).

The judge sits in **veto-proxy**, an OpenAI-compatible proxy between the agent and its model
endpoint. Every tool call the agent model proposes is shown to the judge, which sees **only the
call** (name, description, parameter schema, argument values) and rules allow/deny. In the
enforce arm, denied calls are removed before the benchmark executes them. Each ruling is written
to a hash-chained audit log.

```
 workstation                                             OpenShift (endpoint only)
┌──────────────────────────────────────────────┐        ┌──────────────────────────────┐
│ Agent-SafetyBench / ASB                      │        │ GPU0  agent model (vLLM)     │
│   base_url = http://127.0.0.1:8080/v1        │        │ GPU1  judge SLMs, ShieldAgent│
│   model    = qwen3.8-27b@enforce:granite     │        └──────────────────────────────┘
│              │                               │                    ▲
│              ▼                               │                    │
│ veto-proxy ── forward ───────────────────────┼────────────────────┤
│   tool_calls? ── judge each call ────────────┼────────────────────┘
│   enforce: strip denied calls, return refusal│
│   audit: runs/audit/<run>.jsonl (hash chain) │
└──────────────────────────────────────────────┘
```

## Arms

The arm is part of the model name, so benchmarks need nothing but a `base_url` change:

| model name | arm | effect |
|---|---|---|
| `M` | pass-through | nothing logged |
| `M@baseline` | baseline | calls logged, never judged |
| `M@shadow:J` | shadow | judge J rules, ruling logged, nothing changed |
| `M@enforce:J` | enforce | judge J rules, denied calls removed |

Judge quality is measured by **replaying** every judge offline over the calls logged in the
baseline run. This is equivalent to shadow mode but needs only one agent run. **Enforce** runs
measure the end-to-end effect, since blocking changes the trajectory. The `allow-all` judge is a
built-in control: its enforce run must reproduce the baseline.

## What gets measured

| | metric | source |
|---|---|---|
| Agent-SafetyBench | safety score (% of cases ShieldAgent labels safe), per risk category, paired Δ vs baseline | ShieldAgent served on GPU1 |
| ASB | attack success rate (ASR), original-task success, refusal rate, paired ΔASR | ASB's own checks; its GPT refusal judge is aliased to a local model |
| over-blocking | ASB `clean` runs (no attack): task success must not drop; judge FPR on normal tools | ASB + replay |
| judge quality | ASB: per call (attacker tool = unsafe, normal tool = safe). Agent-SafetyBench: per episode vs ShieldAgent label; unsafe episodes with no tool call are reported as unreachable for a call-only judge | replay |
| cost | judge latency p50/p95, errors, blocks | audit log |

All proportions are reported with Wilson 95% CIs; arm differences with paired bootstrap CIs.

## Setup

```bash
git clone https://github.com/thu-coai/Agent-SafetyBench third_party/Agent-SafetyBench
git -C third_party/Agent-SafetyBench checkout 74feea8de601b3a1449a93fcf70017fe61556f73
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
./scripts/setup_asb_env.sh            # clones ASB (pinned) + its own venv at third_party/ASB/.venv
cp configs/vetobench.example.yaml configs/vetobench.yaml   # fill in routes; export VETOBENCH_API_KEY
```

Ask the cluster admin to serve the models in [`docs/cluster-request.md`](docs/cluster-request.md).

## Running

```bash
vetobench smoke -e configs/pilot.yaml        # served models, tool-call parsing, judges, ShieldAgent
vetobench proxy                              # keep running in its own terminal
vetobench split --n-test 300 --n-dev 48      # stratified Agent-SafetyBench split (dev = prompt work only)

# 1. sanity: 16 cases, allow-all must equal baseline
vetobench split --n-test 16 --n-dev 0 --out runs/splits/asbench-sanity.json
vetobench run -e configs/sanity.yaml && vetobench score -e configs/sanity.yaml && vetobench report -e configs/sanity.yaml

# 2. pilot, one agent model at a time (the admin swaps the model on GPU0 between runs)
vetobench run    -e configs/pilot.yaml --model qwen3.8-27b
vetobench score  -e configs/pilot.yaml --model qwen3.8-27b    # ShieldAgent labels
vetobench replay -e configs/pilot.yaml --model qwen3.8-27b    # judges on baseline calls
vetobench report -e configs/pilot.yaml                        # runs/pilot/report.md + summary.json

vetobench verify-audit runs/audit/*.jsonl    # hash chain intact?
```

Every step is resumable: finished cases, shards, scores and replays are skipped on re-run.
Useful filters: `--variant baseline enforce-granite`, `--bench asbench`.

## Faithfulness to the benchmarks

* **Agent-SafetyBench**: `vetobench.adapters.agent_safetybench` ports `evaluation/eval.py`'s
  loop: the benchmark's own environments, the OpenAI-path system prompt, greedy decoding,
  2,048 max tokens, 10 rounds, only the first tool call executed per turn. Differences: case
  subset, 16-way concurrency, a per-case header for audit correlation, and the non-standard
  `type: object` key that `eval.py` adds to function specs is not added. Scoring ports
  `score/eval_with_shield.py`'s prompt verbatim; ShieldAgent runs on vLLM instead of HF
  transformers.
* **ASB**: `main_attacker.py` runs unmodified (pinned commit) under `adapters/asb_launch.py`.
  The launcher registers an OpenAI-compatible backend (ASB's GPT backend minus its name check and
  a fixed 2 s sleep), disables the memory DB, and shards attacker tools across parallel
  processes. ASB's `gpt-4o-mini` refusal judge is routed to `judge-small` via `aliases`, so
  refusal rates are not directly comparable to the ASB paper's.
* **Proxy normalisation** is applied in *every* arm: tools without a JSON-schema `parameters`
  (ASB) get an empty object schema, since vLLM rejects them otherwise.

## Judges

Configured in `configs/vetobench.yaml`, all call-only:

* `llm_json`: any instruct SLM plus `src/vetobench/judges/prompts/call_only_v1.txt`; the verdict
  is `{"reason","category","decision"}`, enforced with vLLM guided JSON.
* `guard`: classifier models (Granite Guardian, Llama Guard). The call is rendered as the
  assistant's action after a neutral user turn; `unsafe_pattern` decides.
* `static`: allow-all / deny-all controls.

The judge prompt must be developed on the `dev` split only. ASB is never used for tuning, so it
stays fully held out.

`fail_policy: closed` (the default) denies a call when the judge errors or times out; run with
`open` as a sensitivity check.

## Layout

```
src/vetobench/
  proxy/        app.py (FastAPI), gate.py (judge dispatch/cache), rewrite.py, upstream.py
  judges/       base.py (ToolCall, Verdict), llm.py (llm_json, guard, static), prompts/
  adapters/     agent_safetybench.py, asb.py (orchestration), asb_launch.py (runs in ASB's venv)
  scoring/      shieldagent.py, metrics.py, judge_eval.py (replay + confusion), stats.py
  audit.py      hash-chained JSONL      routing.py   model@arm:judge
  experiment.py run/score/replay/report  smoke.py    day-1 endpoint checks   cli.py
configs/        vetobench.example.yaml, pilot.yaml, sanity.yaml
tests/          unit + proxy tests, fake_vllm.py (GPU-free end-to-end stand-in)
```

GPU-free end-to-end check: `uvicorn tests.fake_vllm:app --port 18000`, then point a copy of the
config at `http://127.0.0.1:18000/v1` and run `configs/sanity.yaml`.
