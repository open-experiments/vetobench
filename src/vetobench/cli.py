"""vetobench command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/vetobench.yaml"


def _settings(args):
    from .config import load_settings
    return load_settings(args.config)


def _exp(args, settings):
    from .experiment import Experiment
    return Experiment.load(args.experiment, settings)


def cmd_proxy(args) -> None:
    import uvicorn

    from .proxy.app import create_app
    s = _settings(args)
    uvicorn.run(create_app(s), host=args.host or s.proxy.host, port=args.port or s.proxy.port,
                log_level=args.log_level)


def cmd_smoke(args) -> None:
    from .smoke import run_smoke
    s = _settings(args)
    agents = args.agents
    if agents is None and args.experiment:
        agents = _exp(args, s).models
    judges = args.judges if args.judges is not None else list(s.judges)
    sys.exit(0 if run_smoke(s, agents or [], judges) else 1)


def cmd_split(args) -> None:
    from .experiment import make_split
    s = _settings(args)
    out = Path(args.out)
    split = make_split(s.path("asbench_dir"), args.n_test, args.n_dev, args.seed, out)
    print(f"wrote {out}: {len(split['test'])} test, {len(split['dev'])} dev cases")


def cmd_run(args) -> None:
    from .experiment import run
    s = _settings(args)
    run(_exp(args, s), s, args.bench, args.model, args.variant)


def cmd_score(args) -> None:
    from .experiment import score
    s = _settings(args)
    score(_exp(args, s), s, args.model, args.variant)


def cmd_replay(args) -> None:
    from .experiment import replay
    s = _settings(args)
    replay(_exp(args, s), s, args.model, args.judges)


def cmd_report(args) -> None:
    from .experiment import report
    s = _settings(args)
    print(f"wrote {report(_exp(args, s), s)}")


def cmd_verify_audit(args) -> None:
    from .audit import verify_chain
    bad = 0
    for p in args.files:
        ok, n, msg = verify_chain(p)
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {p}: {n} records – {msg}")
    sys.exit(1 if bad else 0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vetobench", description=__doc__)
    ap.add_argument("-c", "--config", default=DEFAULT_CONFIG, help=f"settings file (default {DEFAULT_CONFIG})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("proxy", help="run veto-proxy (OpenAI-compatible, judges tool calls)")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--log-level", default="warning")
    p.set_defaults(func=cmd_proxy)

    p = sub.add_parser("smoke", help="check endpoint: served models, agent tool calling, judges, scorer")
    p.add_argument("-e", "--experiment", help="take agent models from this experiment file")
    p.add_argument("--agents", nargs="*", help="agent models to probe")
    p.add_argument("--judges", nargs="*", help="judges to probe (default: all configured)")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("split", help="make a stratified dev/test split of Agent-SafetyBench")
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-dev", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/splits/asbench-pilot.json")
    p.set_defaults(func=cmd_split)

    for name, func, help_ in [("run", cmd_run, "run agents (needs the proxy running)"),
                              ("score", cmd_score, "score Agent-SafetyBench outputs with ShieldAgent"),
                              ("replay", cmd_replay, "run judges offline on baseline tool calls"),
                              ("report", cmd_report, "write report.md and summary.json")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("-e", "--experiment", required=True)
        if name in ("run", "score", "replay"):
            p.add_argument("--model", nargs="*", help="restrict to these agent models")
        if name in ("run", "score"):
            p.add_argument("--variant", nargs="*", help="restrict to variants, e.g. baseline enforce-granite")
        if name == "run":
            p.add_argument("--bench", nargs="*", default=["asbench", "asb"], choices=["asbench", "asb"])
        if name == "replay":
            p.add_argument("--judges", nargs="*")
        p.set_defaults(func=func)

    p = sub.add_parser("verify-audit", help="verify the hash chain of audit files")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_verify_audit)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
