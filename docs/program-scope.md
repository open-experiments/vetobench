# VetoBench program scope and directions

VetoBench measures how much a gate on each proposed tool call reduces unsafe agent actions,
what it wrongly blocks, what it costs per call, and what record it leaves. It is a test bench
for the tool-call trust boundary. As of September 2026 it measures one layer (a call-only judge);
the directions below add the other layers of a gate so the bench can report what each one adds.

## Scope

In scope:

* Gates that see only the call (tool name, description, parameter schema, argument values) and
  rule allow or deny: SLM judges with a JSON verdict, guard models (Granite Guardian, Llama Guard),
  and static controls (allow-all, deny-all).
* Agent-SafetyBench (2,000 cases, 349 environments, scored by ShieldAgent) and ASB (400 attacker
  tools, 10 agents, four injection variants plus a clean arm for over-blocking).
* Four arms selected through the model name (pass-through, baseline, shadow, enforce); judges
  compared offline by replay over baseline calls; enforce runs for trajectory effects.
* Safety score, attack success rate, clean-run task success, per-call judge confusion and
  latency p50/p95, with Wilson or paired bootstrap 95% intervals.
* Any OpenAI-compatible endpoint. The bench runs on a workstation and needs only routes.

Out of scope for now: training judges, model routing and tool retrieval (VetoBench consumes
their output), and gates that read the conversation.

## Where it sits

The Govern leg of the
[Telco AI Grid](https://lfnetworking.org/wp-content/uploads/sites/7/2026/09/LFN-Red-Hat-Whitepaper.pdf)
calls for deterministic policy enforcement around probabilistic synthesis, least-privilege tool
access, bounded action spaces, risk-based approval gates and structured decision logging. Its
reference AI Gateway authenticates, meters and routes, and tool arguments pass through it unread.
VetoBench measures at that point. The target is a layered gate on every tools/call, cheapest layer first:

| Layer | What decides | Nature | Candidate components |
| --- | --- | --- | --- |
| 0. Catalog filter | Which tools the model sees | Identity + retrieval | IBAC at the gateway, bind-time retrieval (ToolScope, k=10) |
| 1. Injection guards | Was the request or an argument injected | Classifier, ~0.1B | vLLM-SR `toolcall-sentinel`, `toolcall-verifier` |
| 2. Value gate | Do the argument values pass named rules | Deterministic | Agent Exchange tool-call gate (`aex-toolgate`) |
| 3. Residual judge | Is the call unsafe beyond the rules | Probabilistic | SLM judge, Granite Guardian, Llama Guard, Vela Shield, Decision 1.0 (Noul) |
| 4. Hold | Escalate to a person | Human | Explicit `hold` outcome, released under the approver's identity |

On the reference cluster (one node, 2 x RTX PRO 6000 96 GB), 1B to 8B judges and agents stand in
for the grid's user-edge tier and 27B to 31B agents for the provider-edge tier. The cards are
larger than edge hardware, so latency results are a best case.

## Related work and reuse

VetoBench plugs each of these in where it fits.

| Effort | Owns | Reuse in VetoBench |
| --- | --- | --- |
| [Agent Exchange](https://open-experiments.github.io/agent-exchange/) tool-call gate | Deterministic value gate, hash-chained decision record, hold and release | `rule` judge kind wrapping `aex-toolgate` (layer 2) |
| ToolScope + BFCL harness | Bind-time tool retrieval; BFCL Multiple (443 tools) name and AST accuracy | BFCL as a third suite; retrieval-only numbers as the no-gate baseline |
| MLflow agent tracing (proposed) | Traces, human review, judge improvement | Trace and review layer; the VetoBench audit log stays the scored record |
| [vLLM Semantic Router](https://huggingface.co/llm-semantic-router) tool-call guards | Injection detection per request and per argument token | `classifier` judge kind (layer 1) |
| vLLM-SR Vela Shield, [Decision 1.0](https://vllm-sr.ai/decision-paper.pdf) | Safety taxonomy (34 categories); typed Choice, Noul, Score decisions | Residual judges; Decision needs a System One adapter and an NVIDIA test |
| [TelcoAIBench](https://open-experiments.github.io/telcoaibench/) | Right model per workload and footprint | Agent model selection per tier |
| [Quantization guard-rails](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7450801) | Acceptance of quantized artifacts per rung | Agent runs per rung; gate refusal rate as a runtime fitness signal |

Notes on the external assets: the vLLM-SR tool-call guards were trained mainly on prompt-injection
data, so they cover injected arguments and leave business-rule violations to the value gate. Decision 1.0
reports 5.56% calibration error for its 9B model and no candidate-order invariance, so a
threshold needs its own calibration and the record must carry the candidate order.

## Strengths to keep

1. The arm in the model name (`M@enforce:J`) is the only integration contract.
2. Replay separated from enforce: one baseline run scores every judge offline.
3. One hash-chained audit record per proposed call, checked by `vetobench verify-audit`.
4. Controls: allow-all enforce must equal baseline; `fail_policy` closed versus open.
5. Benchmarks at pinned commits with documented deviations; ASB never used for tuning.
6. Over-blocking reported beside attack success.

## Companion study: the decision record

A companion study, under submission, fixes a nine-field audit artifact and five scoring rules
for tool-call authorization and measures them on a fixed set of accounts-payable calls with the
model's outputs held constant. The two efforts feed each other.

The study gives VetoBench:

* the record specification: timestamp, agent identity (agent, principal, session), tool name,
  argument values, decision, rule or scope with its policy version, approval status (automatic
  or held), outcome, and record integrity;
* five scoring rules (action recoverability, lifecycle coverage, policy checkability,
  responsibility attribution, evidence integrity) computable from a record alone;
* a deterministic value gate and twelve calls that serve as a regression suite;
* the requirement that the component that enforced a decision writes its record.

VetoBench gives the study live small models in place of held-fixed outputs; refusal, hold and
pass-through counts per agent model, tier and quantization rung; probabilistic judges scored on
the same fields; and public suites beside the synthetic set.

Record gap in `vetobench/v1` (from `proxy/app.py` and `audit.py`):

| # | Field | `vetobench/v1` | Needed for v2 |
| --- | --- | --- | --- |
| 1 | Timestamp | `ts` | none |
| 2 | Agent identity | `agent_model`, `episode_id` | principal (tenant or user) |
| 3 | Tool name | `tool.name` | none |
| 4 | Argument values | `tool` arguments | none |
| 5 | Decision | `effective_decision`, `blocked` | none |
| 6 | Rule or scope | `verdict.category`, `verdict.reason` | rule id and policy version; for probabilistic judges, model revision, prompt version, probability, threshold, temperature, candidate order |
| 7 | Approval status | absent | `auto` or `held` |
| 8 | Outcome | absent | benchmark execution result joined on `tool_call_id` |
| 9 | Record integrity | `seq`, `prev_hash`, `hash` | none |

## Roadmap

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 0. Prepare (no GPUs) | `vetobench/v2` record; `rule` and `classifier` judge kinds; the twelve calls as a regression suite on `tests/fake_vllm.py` | The twelve calls reproduce the companion study's decisions and scores |
| 1. Align | One runner (VetoBench), layer order, BFCL in or out | Written decisions on the open items |
| 2. Serve and pilot | Agent and judge models served; sanity run, then pilot per layer | Pilot report with per-layer deltas and scored records |
| 3. Joint experiment | BFCL through bind-time retrieval, the layered gate, agents at two tiers and several quantization rungs | Per-layer, per-tier, per-rung results |

Open items: harness base, gate order, BFCL inclusion, Decision 1.0 on NVIDIA, packing four small
models on one GPU, record schema v2.

## Progress (as of 2026-09-26)

Phase 2 (serve and pilot) has started on the reference cluster; Phase 0 and Phase 1 work has
not. Details, commands and the dated status log are in the README section *Cluster: `venice`
OpenShift*.

* **Serving.** The `vetobench` project serves the agent model on GPU0 as a vLLM Deployment
  behind an HTTPS route. Manifests are in `deploy/openshift/`;
  `scripts/swap-agent-model.sh <served name>` swaps the agent model and waits until it is
  served. GPU1 is still empty.
* **Harness to cluster.** The routes use the cluster's self-signed ingress CA, so an
  upstream's `verify_tls` now takes a CA bundle path and verification stays on.
  `vetobench smoke` passes the agent tool-call check for both models served so far.
* **Agent models.** `qwen3.8-27b` is `Qwen/Qwen3.8-27B`, served in BF16 (FP8 not needed) with
  tool parser `qwen3_xml`, text only, and `--max-num-seqs 64` (vLLM's default does not fit its
  linear-attention state). Typed tool arguments parse correctly and thinking is off.
  `Qwen/Qwen3-8B` (`qwen3-8b-test`) stays available for pipeline tests. The Hugging Face ids
  for `muse-glimmer-30b` and `gemma4-31b` are still open.
* **First measurements** (Qwen3-8B, sanity-sized, no real judge yet; `configs/baseline-8b.yaml`):
  * The allow-all control made the same tool calls as baseline in 16/16 Agent-SafetyBench
    cases. Four differ only in the wording of the final text answer (vLLM greedy decoding is
    not bit-exact across batch compositions).
  * ASB direct prompt injection (naive), 20 attacker tools: attack success 100% (20/20, 95% CI
    83.9–100); original-task success 0%.
  * ASB clean, 50 tasks: task success 42% baseline and 38% allow-all; attack success 0%.
  * Not yet available: the Agent-SafetyBench safety score (needs ShieldAgent). The ASB refusal
    rate is not usable, because Qwen3-8B, standing in as ASB's refusal judge, marks completed
    tasks as refusals.
* **Qwen3.8-27B sanity run** (`configs/sanity.yaml`, baseline vs allow-all):
  * Agent-SafetyBench, 16 cases: allow-all made the same tool calls as baseline in 16/16.
  * ASB direct prompt injection (naive), 20 attacker tools: attack success 45% (9/20, 95% CI
    25.8–65.8), against 100% for Qwen3-8B; original-task success 0%.
  * ASB clean, 50 tasks: task success 82% (Qwen3-8B: 42%); attack success 0%. Allow-all
    matches baseline on attack and task success.
  * Refusal is again measured with a stand-in judge (Qwen3.8-27B itself), so it is provisional.
* **Fixes on the way.** The ASB launcher no longer needs conda and no longer hangs at exit. The
  proxy merges the system messages that open a conversation into one (every arm), because
  Qwen3.8's chat template rejects ASB's second system message; before this fix every ASB
  request to Qwen3.8 failed.

## What's next

1. **GPU1 small models**: one pod requesting one GPU runs `judge-small`, `granite-guardian`,
   `llama-guard` and `shieldagent` as four vLLM servers (ports 8001–8004, about 0.2 of GPU
   memory each), behind one Service and four routes. `llama-guard` is gated and needs
   `HF_TOKEN`. The two pods together must stay within the 96Gi memory-request quota.
2. **Score and re-measure**: score the Agent-SafetyBench runs with ShieldAgent (finished runs are
   scored without re-running), point ASB's refusal judge back to `judge-small`, and re-measure
   refusal.
3. **Remaining agent models**: get the Hugging Face ids for `muse-glimmer-30b` and `gemma4-31b`,
   add them to the swap script with their tool parsers, and run the smoke check.
4. **Pilot**: the 300-case Agent-SafetyBench split and all 400 ASB attacker tools across the
   five attack settings, per agent model; baseline plus enforce per judge; replay all judges on
   the baseline calls; report.
5. **Phase 0 (no GPUs)**: the `vetobench/v2` record, the `rule` and `classifier` judge kinds, and
   the twelve-call regression suite. Not started.
