"""Command line: `skillrc-gate {report,validate,convert,schema}`.

Exit codes of `report` (for CI): 0 PASS, 1 INCONCLUSIVE, 2 BLOCK, 3 INVALID, 4 DESCRIPTIVE, 64 usage error.
Only exit code 0 should let a release through.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import __version__


def _cmd_report(a) -> int:
    from .engine import run_gate
    from .report import write_reports
    from .spec import load_spec

    spec = load_spec(a.spec)
    result = run_gate(spec)
    name, verdict = result.selected(a.target)
    out = Path(a.out) if a.out else Path(a.spec).with_suffix("").parent / "gate_report"
    paths = write_reports(result.to_dict(), out, selected=name, formats=tuple(a.formats.split(",")))
    for n, v in result.targets.items():
        mark = "*" if n == name else " "
        print(f"{mark} {n:<24} {v.state.value}")
    if name is None or name not in result.targets:
        print("; ".join(verdict.reasons), file=sys.stderr)
    for k, p in paths.items():
        print(f"  {k}: {p}")
    print(f"exit {verdict.exit_code} ({verdict.state.value}) for target {name}")
    return verdict.exit_code


def _cmd_validate(a) -> int:
    from .ingest import load

    opts = json.loads(a.options) if a.options else {}
    eps, man = load(a.format, [Path(p) for p in a.logs], a.arm, Path(a.manifest) if a.manifest else None, opts)
    n_scored = sum(e.scored for e in eps)
    reps = Counter(e.repetition for e in eps)
    print(f"{len(eps)} attempts, {n_scored} scored, {len({e.task_uid for e in eps})} tasks, "
          f"repetitions {dict(reps)}, manifest: {'yes' if man else 'no'}")
    missing = [f for f in ("repetition", "latency_s") if any(getattr(e, f) is None for e in eps)]
    if missing or man is None:
        print("release-eligible: no (" + ", ".join(missing + ([] if man else ["manifest"])) + " missing)")
        return 4
    print("release-eligible: yes")
    return 0


def _cmd_convert(a) -> int:
    from .ingest import load

    eps, _ = load(a.format, [Path(p) for p in a.logs], a.arm, None, json.loads(a.options) if a.options else {})
    with open(a.out, "w") as f:
        for e in eps:
            f.write(e.model_dump_json(exclude_none=False) + "\n")
    print(f"wrote {len(eps)} episodes to {a.out}")
    return 0


def _cmd_schema(a) -> int:
    from .schema import json_schemas
    from .spec import GateSpec

    s = {**json_schemas(), "GateSpec": GateSpec.model_json_schema()}
    text = json.dumps(s, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skillrc-gate", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"skillrc-gate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="compute verdicts and write result.json / summary.md / report.html")
    r.add_argument("spec")
    r.add_argument("--target", help="release target whose verdict sets the exit code (default: the default target)")
    r.add_argument("--out", help="output directory (default: gate_report/ next to the spec)")
    r.add_argument("--formats", default="json,md,html")
    r.set_defaults(fn=_cmd_report)

    v = sub.add_parser("validate", help="check that logs parse and whether they are release-eligible")
    v.add_argument("logs", nargs="+")
    v.add_argument("--format", default="skillrc", choices=["skillrc", "episodes", "inspect"])
    v.add_argument("--arm", default="arm")
    v.add_argument("--manifest")
    v.add_argument("--options", help='adapter options as JSON, e.g. \'{"value_key": "utility"}\'')
    v.set_defaults(fn=_cmd_validate)

    c = sub.add_parser("convert", help="convert logs to the canonical episodes JSONL")
    c.add_argument("logs", nargs="+")
    c.add_argument("--format", default="skillrc", choices=["skillrc", "inspect"])
    c.add_argument("--arm", default="arm")
    c.add_argument("--out", required=True)
    c.add_argument("--options")
    c.set_defaults(fn=_cmd_convert)

    s = sub.add_parser("schema", help="print the JSON Schemas of Episode, RunManifest and GateSpec")
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_schema)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3 if args.cmd == "report" else 64


if __name__ == "__main__":
    sys.exit(main())
