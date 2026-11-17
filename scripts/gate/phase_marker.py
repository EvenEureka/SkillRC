#!/usr/bin/env python3
"""Success marker for a completed block (8B) or phase (32B): run ids and the sha256 of each run's episodes file.

Idempotent: if the marker exists and lists the same runs with the same hashes, nothing changes (exit 0); if it
differs, exit 3 (the evidence changed after the marker was written).
usage: phase_marker.py <results dir> <marker name> <job id> <chain file> [<chain file> ...]
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_io import atomic_write  # noqa: E402


def build(out: Path, chains):
    runs = [Path(c).stem for ch in chains for c in Path(ch).read_text().split()]
    return {r: hashlib.sha256((out / f"{r}.episodes.jsonl").read_bytes()).hexdigest() for r in runs}


def main(argv):
    out, name, job, chains = Path(argv[0]), argv[1], argv[2], argv[3:]
    hashes = build(out, chains)
    marker = out / name
    if marker.exists():
        try:
            old = json.loads(marker.read_text())
        except json.JSONDecodeError:
            print(f"{marker} is unreadable")
            return 3
        if old.get("episodes_sha256") != hashes:
            print(f"{marker} exists with different episode hashes")
            return 3
        print(f"{marker} already written; unchanged")
        return 0
    rec = {"marker": name, "job": job, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "episodes_sha256": hashes}
    atomic_write(marker, json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
