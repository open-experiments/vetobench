"""Arm selection encoded in the model name.

Clients address the proxy with ``<served-model>@<arm>[:<judge>]``:

    qwen3.8-27b                      -> pass-through, nothing logged
    qwen3.8-27b@baseline             -> pass-through, every proposed tool call is logged
    qwen3.8-27b@shadow:granite       -> judge rules on each call, ruling logged, nothing changed
    qwen3.8-27b@enforce:granite      -> judge rules; denied calls are removed before the agent sees them

Keeping the arm in the model name means benchmarks need no code change beyond ``base_url``
and several arms can run concurrently through one proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

ARMS = ("passthrough", "baseline", "shadow", "enforce")


@dataclass(frozen=True)
class Route:
    model: str
    arm: str = "passthrough"
    judge: str | None = None

    @property
    def judged(self) -> bool:
        return self.arm in ("shadow", "enforce")

    @property
    def logged(self) -> bool:
        return self.arm != "passthrough"

    def encode(self) -> str:
        return format_route(self.model, self.arm, self.judge)


def format_route(model: str, arm: str = "passthrough", judge: str | None = None) -> str:
    if arm == "passthrough":
        return model
    if arm in ("shadow", "enforce"):
        if not judge:
            raise ValueError(f"arm {arm!r} needs a judge")
        return f"{model}@{arm}:{judge}"
    if arm == "baseline":
        return f"{model}@baseline"
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def parse_route(name: str) -> Route:
    if "@" not in name:
        return Route(model=name)
    model, _, spec = name.rpartition("@")
    arm, _, judge = spec.partition(":")
    if arm not in ARMS or arm == "passthrough":
        raise ValueError(f"unknown arm {arm!r} in model {name!r}")
    if arm in ("shadow", "enforce") and not judge:
        raise ValueError(f"arm {arm!r} needs a judge: use {model}@{arm}:<judge>")
    if arm == "baseline" and judge:
        raise ValueError("the baseline arm takes no judge")
    if not model:
        raise ValueError(f"empty model in {name!r}")
    return Route(model=model, arm=arm, judge=judge or None)
