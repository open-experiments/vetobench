"""Run ASB's main_attacker.py with an OpenAI-compatible backend pointed at veto-proxy.

Executed with ASB's own Python (its heavy deps live there), so this file imports only the
stdlib, ``openai`` and ASB modules. It registers the requested model name in ASB's
MODEL_REGISTRY with ``VetoLLM`` (ASB's GPT backend without the ``gpt`` name assertion and the
fixed 2 s sleep, plus vetobench headers), then runs main_attacker.py unchanged.

ASB's refusal judge calls ``OpenAI()`` with model ``gpt-4o-mini``; with OPENAI_BASE_URL set to
the proxy, the proxy's alias table sends that to a locally served model.

usage: <asb-python> asb_launch.py --asb-dir DIR --run-id ID -- <main_attacker.py args>
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asb-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    rest = ns.rest[1:] if ns.rest[:1] == ["--"] else ns.rest

    asb_dir = os.path.abspath(ns.asb_dir)
    os.chdir(asb_dir)
    sys.path.insert(0, asb_dir)
    if "OPENAI_BASE_URL" not in os.environ:
        sys.exit("OPENAI_BASE_URL must point at veto-proxy (e.g. http://127.0.0.1:8080/v1)")
    os.environ.setdefault("OPENAI_API_KEY", "vetobench")

    import openai
    from openai import OpenAI

    from aios.llm_core.llm_classes import model_registry
    from aios.llm_core.llm_classes.base_llm import BaseLLM
    from pyopenagi.utils.chat_template import Response

    run_id = ns.run_id

    class VetoLLM(BaseLLM):
        def load_llm_and_tokenizer(self) -> None:
            self.model = OpenAI(default_headers={"x-vetobench-run": run_id}, max_retries=5, timeout=900)
            self.tokenizer = None

        @staticmethod
        def _parse_tool_calls(tool_calls):
            if not tool_calls:
                return None
            parsed = []
            for tc in tool_calls:
                raw = tc.function.arguments
                try:
                    args = json.loads(raw) if raw and raw.strip() else {}
                except json.JSONDecodeError:
                    args = {}
                parsed.append({"name": tc.function.name, "parameters": args})
            return parsed

        def process(self, agent_process, temperature=0.0):
            agent_process.set_status("executing")
            agent_process.set_start_time(time.time())
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=agent_process.query.messages,
                    tools=agent_process.query.tools,
                    max_tokens=self.max_new_tokens,
                    seed=0,
                    temperature=temperature,
                    extra_headers={"x-vetobench-episode": f"asb:{agent_process.agent_name}"},
                )
                msg = response.choices[0].message
                agent_process.set_response(Response(response_message=msg.content,
                                                    tool_calls=self._parse_tool_calls(msg.tool_calls)))
            except openai.APIError as e:
                agent_process.set_response(Response(response_message=f"LLM API error: {e}"))
            except Exception as e:
                agent_process.set_response(Response(response_message=f"An unexpected error occurred: {e}"))
            agent_process.set_status("done")
            agent_process.set_end_time(time.time())

    try:
        llm_name = rest[rest.index("--llm_name") + 1]
    except (ValueError, IndexError):
        sys.exit("pass --llm_name <model route> to main_attacker")
    model_registry.MODEL_REGISTRY[llm_name] = VetoLLM  # llms.py holds the same dict object

    sys.argv = ["main_attacker.py", *rest]
    runpy.run_path(os.path.join(asb_dir, "main_attacker.py"), run_name="__main__")


if __name__ == "__main__":
    main()
