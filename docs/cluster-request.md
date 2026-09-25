# Model serving request (OpenShift, 2× RTX PRO 6000 96 GB)

vetobench only needs OpenAI-compatible endpoints (vLLM). Please serve the following and send
back the route URL(s), a token, and the output of `GET /v1/models` for each route.

## GPU0: agent model under test (one at a time; we'll ask for swaps)

| served name (`--served-model-name`) | HF model id | notes |
|---|---|---|
| `qwen3.8-27b` | _TBD, from the TelcoAIBench leaderboard_ | BF16 |
| `muse-glimmer-30b` | _TBD_ | BF16 |
| `gemma4-31b` | _TBD_ | BF16 |

vLLM flags (all three):

```
vllm serve <hf-id> --served-model-name <name> \
  --enable-auto-tool-choice --tool-call-parser <parser for the model family> \
  --max-model-len 32768 --gpu-memory-utilization 0.92 --dtype bfloat16
```

The tool-call parser is **required**: without `--enable-auto-tool-choice` and the right
`--tool-call-parser`, the model's tool calls come back as plain text and both benchmarks break.
Use the parser vLLM documents for each model family (e.g. `hermes` for Qwen-family models). We
check this on day 1 with `vetobench smoke`, which fails if `tool_calls` do not parse.

A 27–31B model in BF16 is ~55–62 GB of weights, which leaves ~25–30 GB of KV cache on one card
at 0.92 utilisation. If that proves too tight at 32k context and our concurrency (16 parallel
episodes), FP8 (`--quantization fp8`) is acceptable, but please tell us so we record it.

## GPU1: small models (all resident at the same time)

| served name | HF model id | size | purpose |
|---|---|---|---|
| `judge-small` | a 4–8B instruct model, e.g. Qwen3-8B-class | ~16 GB | SLM judge (JSON verdict) + ASB refusal judge |
| `granite-guardian` | `ibm-granite/granite-guardian-3.3-8b` (or 3.2) | ~16 GB | guard-model judge |
| `llama-guard` | `meta-llama/Llama-Guard-3-8B` | ~16 GB | guard-model judge |
| `shieldagent` | `thu-coai/ShieldAgent` | ~15 GB | Agent-SafetyBench scorer (needs `/v1/completions`) |

Four vLLM instances with `--gpu-memory-utilization 0.2` each (≈19 GB each) fit on one 96 GB card.
`judge-small` needs structured outputs (on by default in recent vLLM). No tool parser is needed
on GPU1.

## Network

The harness runs on a workstation outside the cluster, so the routes must be reachable from
there over HTTPS. Timeouts on the route should be at least 10 minutes (long agent turns).
