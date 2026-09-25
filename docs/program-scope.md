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
