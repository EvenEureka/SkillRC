"""Idempotent finalization for the 32B pilot decision and the D-phase trigger.

A decision is always recomputed from the saved episodes. If a decision file exists, its core fields must equal the
fresh evaluation (else exit 3: the evidence or the code changed); the launch marker is then written if it should
exist and is missing. So a job killed between "decision written" and "marker written" can simply be resubmitted."""
import json
import os
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    """Publish a state file atomically: temp file in the same directory, flush + fsync, then rename over the target.
    An interrupted publication leaves either the old file or none, never partial JSON."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def finalize(dec_path: Path, fresh: dict, core_keys, marker_path: Path, marker_text: str, fire: bool) -> int:
    fresh = _norm(fresh)
    if dec_path.exists():
        try:
            saved = json.loads(dec_path.read_text())
        except json.JSONDecodeError:
            print(f"{dec_path}: unreadable saved decision")
            return 3
        diff = [k for k in core_keys if saved.get(k) != fresh.get(k)]
        if diff:
            print(f"{dec_path}: saved decision differs from a fresh evaluation in {diff}")
            return 3
        print(f"{dec_path}: saved decision confirmed")
    else:
        atomic_write(dec_path, json.dumps(fresh, indent=1) + "\n")
    if fire and not marker_path.exists():
        atomic_write(marker_path, marker_text)
        print(f"wrote {marker_path}")
    return 0
