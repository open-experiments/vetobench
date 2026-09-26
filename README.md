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
                                                           # (venice cluster: see "Cluster" below)
```

Ask the cluster admin to serve the models in [`docs/cluster-request.md`](docs/cluster-request.md).

## Cluster: `venice` OpenShift

The harness and proxy run on a workstation; the cluster only serves models as OpenAI-compatible
vLLM endpoints. We are admins of the `vetobench` project and nothing else (no nodes, no
cluster-scoped objects, no operators), so only ever touch that namespace.

| | |
|---|---|
| console | https://console-openshift-console.apps.venice.narlabs.io |
| API server | `https://api.venice.narlabs.io:6443` (log in with the token from the console, user menu → *Copy login command*) |
| project | `vetobench`, user `vetobench-admin` |
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell, 96 GB each, on one node |
| quota `vetobench-guardrail` | 2 GPUs; requests 24 CPU / 96Gi, limits 48 CPU / 192Gi; 1Ti storage, 30 PVCs, 60 pods |
| LimitRange | container max 24 CPU / 96Gi; **every container must set explicit cpu and memory limits** (the default limit is 1 CPU, so larger requests are rejected) |
| storage | StorageClass `lvms-vg1` (default, RWO, local LVM) |
| routes | `https://<route>-vetobench.apps.venice.narlabs.io`, edge TLS, 15 min timeout |

**TLS.** The router certificate is signed by the cluster's self-signed ingress CA
(`CN=ingress-operator@1784587897`, valid until 2028-07-19), so plain `curl` fails with
*self-signed certificate in certificate chain*. Use the CA bundle `venice-ca.crt` (ask the
project owner for it); it also covers the API server, so it works for `oc login` too. Don't
disable verification. In `configs/vetobench.yaml`, `verify_tls` takes the bundle path
(relative to the repo root, or absolute); the venice config defaults to `../venice-ca.crt`
and can be overridden with `VETOBENCH_CA_BUNDLE`.

**Secrets.** `Secret vllm-secrets` holds `VLLM_API_KEY` (bearer token for all routes) and
`HF_TOKEN` (for gated models). Never print, log or commit them. Get the key into your shell
without echoing it:

```bash
oc login --server=https://api.venice.narlabs.io:6443 --certificate-authority=venice-ca.crt --token=...
export VETOBENCH_API_KEY=$(oc get secret vllm-secrets -n vetobench -o jsonpath='{.data.VLLM_API_KEY}' | base64 -d)
```

### What is deployed

| GPU | object | served model | route | status |
|---|---|---|---|---|
| 0 | `Deployment vllm-agent` | `qwen3-8b-test` = `Qwen/Qwen3-8B`, tool parser `hermes` | `https://vllm-agent-vetobench.apps.venice.narlabs.io/v1` | running, verified 2026-09-25 |
| 1 | `vllm-small` (one pod, four `vllm serve` processes) | `judge-small`, `granite-guardian`, `llama-guard`, `shieldagent` | `vllm-<served name>-vetobench…` | **not deployed yet** |

`vllm-agent`: image `docker.io/vllm/vllm-openai:latest` (vLLM 0.30.0), 1 GPU, `strategy:
Recreate`, cpu 4/12, memory 32Gi/64Gi, `/dev/shm` 16Gi, model cache on `PVC model-cache`
(300Gi, mounted at `/models`, `HF_HOME=/models/hf`). The model is chosen by env vars
`MODEL_ID`, `SERVED_NAME`, `TOOL_PARSER`; flags `--enable-auto-tool-choice --tool-call-parser
$TOOL_PARSER --max-model-len 32768 --gpu-memory-utilization 0.92 --dtype bfloat16 --api-key
$VLLM_API_KEY`. With Qwen3-8B it gets a 68 GiB KV cache (about 15× concurrency at 32k context).
Exposed by `Service vllm-agent` (port 8000) and `Route vllm-agent`. Known cosmetic warning
"Unknown vLLM environment variable VLLM_AGENT_*" comes from service links
(`enableServiceLinks: false` removes it).

Swap the agent model under test (the served name must match the experiment config exactly,
and the tool parser must fit the model family):

```bash
oc set env deploy/vllm-agent MODEL_ID=<hf id> SERVED_NAME=<served name> TOOL_PARSER=<parser>
```

The 27–31B agent models are expected to fit in BF16; if they run out of memory at 32k context
and 16 parallel episodes, add `--quantization fp8` and record it here.

GPU1 plan: pods can't share a GPU (no fractional requests), so one pod requests
`nvidia.com/gpu: 1` and runs four `vllm serve` processes on ports 8001–8004, each with
`--gpu-memory-utilization ~0.2` and a smaller `--max-model-len`, under a supervisor that exits
if any child dies. One Service (4 ports), four Routes. The two pods together must stay within
96Gi of memory requests. Manifests go in `deploy/openshift/`.

### Checking an endpoint

```bash
export VETOBENCH_AGENT_URL=https://vllm-agent-vetobench.apps.venice.narlabs.io/v1
curl -s --cacert venice-ca.crt $VETOBENCH_AGENT_URL/models -H "Authorization: Bearer $VETOBENCH_API_KEY"
vetobench smoke --agents qwen3-8b-test --judges allow-all
```

The route returns 401 without the key; that is expected.

### Status log

* **2026-09-25** `vllm-agent` verified from the workstation: `/v1/models` lists
  `qwen3-8b-test`, and `vetobench smoke` gets a parsed tool call
  (`get_weather(city="Paris")`, ~1.8 s). GPU1 is empty, so ShieldAgent scoring, the real
  judges and ASB's refusal judge are not available yet. The real agent models
  (`qwen3.8-27b`, `muse-glimmer-30b`, `gemma4-31b`) still need Hugging Face ids.
  TODO: add the `vllm-agent` manifest to `deploy/openshift/`.

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
