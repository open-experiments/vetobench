#!/usr/bin/env bash
# Swap the agent model under test on GPU0 (deploy/vllm-agent in project vetobench), wait until
# the new model is served, and check it through the route.
#
#   scripts/swap-agent-model.sh <served name>      e.g. qwen3.8-27b
#
# The served name must match the experiment configs exactly. To add a model, add a case below
# with its Hugging Face id, the vLLM tool-call parser for its family, and any extra vLLM flags.
# The first start of a new model downloads its weights into PVC model-cache (≈2 GB per 1B
# params in BF16), so it can take a while; later starts load from the cache.
#
# Needs: oc (logged in), curl, python3. VETOBENCH_API_KEY is read from the cluster secret when
# not set; it is never printed.
set -euo pipefail

NS=vetobench
ROUTE_URL=${VETOBENCH_AGENT_URL:-https://vllm-agent-vetobench.apps.venice.narlabs.io/v1}
CA=${VETOBENCH_CA_BUNDLE:-$(dirname "$0")/../../venice-ca.crt}

SERVED_NAME=${1:-}
case "$SERVED_NAME" in
  qwen3-8b-test)     # pipeline tests
    MODEL_ID=Qwen/Qwen3-8B
    TOOL_PARSER=hermes
    EXTRA_ARGS="" ;;
  qwen3.8-27b)       # https://huggingface.co/Qwen/Qwen3.8-27B, recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
    MODEL_ID=Qwen/Qwen3.8-27B
    TOOL_PARSER=qwen3_xml                               # XML <function=...> tool calls, not hermes
    # --language-model-only: text only, skip the vision encoder. --max-num-seqs: its linear-attention
    # layers need one state block per running sequence; vLLM's default 1024 does not fit (708 do).
    EXTRA_ARGS="--language-model-only --reasoning-parser qwen3 --max-num-seqs 64" ;;
  # muse-glimmer-30b)  TBD: Hugging Face id from Fatih
  # gemma4-31b)        TBD: Hugging Face id from Fatih
  *)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    echo "known models: $(grep -oE '^  [a-z0-9.-]+\)' "$0" | tr -d ' )' | tr '\n' ' ')" >&2
    exit 2 ;;
esac

current=$(oc -n "$NS" get deploy/vllm-agent -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="SERVED_NAME")].value}')
echo "vllm-agent: $current -> $SERVED_NAME ($MODEL_ID, parser $TOOL_PARSER${EXTRA_ARGS:+, $EXTRA_ARGS})"

oc -n "$NS" set env deploy/vllm-agent \
  MODEL_ID="$MODEL_ID" SERVED_NAME="$SERVED_NAME" TOOL_PARSER="$TOOL_PARSER" EXTRA_ARGS="$EXTRA_ARGS"
sleep 30  # let the Recreate rollout remove the old pod before checking the new one's state

if [[ -z "${VETOBENCH_API_KEY:-}" ]]; then
  VETOBENCH_API_KEY=$(oc -n "$NS" get secret vllm-secrets -o jsonpath='{.data.VLLM_API_KEY}' | base64 -d)
fi

# The pod can be Running before vLLM has loaded the weights; wait for /v1/models to list the model.
echo "waiting for $ROUTE_URL/models to list $SERVED_NAME (weights may be downloading)..."
for _ in $(seq 360); do
  state=$(oc -n "$NS" get pod -l app=vllm-agent -o jsonpath='{.items[*].status.containerStatuses[0].state.waiting.reason}')
  if [[ "$state" == *CrashLoopBackOff* ]]; then
    echo "vllm-agent is crash-looping; last error:" >&2
    oc -n "$NS" logs deploy/vllm-agent --tail=300 | grep -E 'Error|error:' | grep -v 'See root cause' | tail -3 >&2
    exit 1
  fi
  if curl -sf -m 10 --cacert "$CA" -H "Authorization: Bearer $VETOBENCH_API_KEY" "$ROUTE_URL/models" \
      | python3 -c 'import json,sys; sys.exit(0 if sys.argv[1] in [m["id"] for m in json.load(sys.stdin)["data"]] else 1)' "$SERVED_NAME" 2>/dev/null; then
    echo "serving $SERVED_NAME. Next: .venv/bin/vetobench smoke --agents $SERVED_NAME --judges allow-all"
    exit 0
  fi
  sleep 10
done
echo "timed out after 60 min; check: oc -n $NS logs deploy/vllm-agent --tail=50" >&2
exit 1
