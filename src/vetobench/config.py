"""Load the vetobench YAML config (endpoints, judges, proxy) and experiment files."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} and ${VAR:-default} in all strings of a parsed YAML tree."""
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(f"environment variable {name} is not set (referenced in config)")
        return _ENV_RE.sub(sub, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return _expand_env(yaml.safe_load(f) or {})


@dataclass
class Upstream:
    name: str
    base_url: str
    api_key: str = "EMPTY"
    models: list[str] = field(default_factory=list)
    verify_tls: bool = True
    timeout_s: float = 600.0


@dataclass
class ProxySettings:
    host: str = "127.0.0.1"
    port: int = 8080
    audit_dir: str = "runs/audit"
    fail_policy: str = "closed"  # closed: judge error => deny in enforce mode; open: => allow
    judge_timeout_s: float = 60.0
    judge_cache: bool = True
    max_concurrent_judges: int = 32
    refusal_template: str = (
        "I did not perform this action. The tool call `{tool_name}` was blocked by the "
        "tool-call safety judge ({category}): {reason}"
    )


@dataclass
class Settings:
    upstreams: list[Upstream]
    aliases: dict[str, str]
    judges: dict[str, dict]
    model_defaults: dict[str, dict]
    proxy: ProxySettings
    paths: dict[str, str]
    scoring: dict[str, Any]
    root: Path

    def upstream_for(self, model: str) -> Upstream:
        for up in self.upstreams:
            if model in up.models:
                return up
        # A single upstream without a model list acts as a catch-all.
        catch_all = [u for u in self.upstreams if not u.models]
        if catch_all:
            return catch_all[0]
        raise KeyError(f"no upstream serves model {model!r}; add it to an upstream's `models` list")

    def resolve_alias(self, model: str) -> str:
        return self.aliases.get(model, model)

    def apply_model_defaults(self, model: str, body: dict) -> dict:
        """Fill request fields configured for a served model (e.g. chat_template_kwargs) that the
        client did not set itself."""
        for k, v in (self.model_defaults.get(model) or {}).items():
            body.setdefault(k, v)
        return body

    def path(self, key: str) -> Path:
        p = Path(self.paths[key])
        return p if p.is_absolute() else (self.root / p)


def load_settings(path: str | Path) -> Settings:
    path = Path(path).resolve()
    raw = load_yaml(path)
    ups = [Upstream(**u) for u in raw.get("upstreams", [])]
    if not ups:
        raise ValueError("config needs at least one entry under `upstreams`")
    # Paths in the config are relative to the repo root (the config's parent dir's parent).
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    return Settings(
        upstreams=ups,
        aliases=raw.get("aliases", {}) or {},
        model_defaults=raw.get("model_defaults", {}) or {},
        judges=raw.get("judges", {}) or {},
        proxy=ProxySettings(**(raw.get("proxy") or {})),
        paths={
            "asbench_dir": "third_party/Agent-SafetyBench",
            "asb_dir": "third_party/ASB",
            "asb_python": "third_party/ASB/.venv/bin/python",
            "runs_dir": "runs",
            **(raw.get("paths") or {}),
        },
        scoring=raw.get("scoring", {}) or {},
        root=root,
    )
