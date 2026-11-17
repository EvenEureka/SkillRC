"""Canonical format: one `schema.Episode` JSON object per line (what `skillrc-gate convert` writes)."""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import List, Optional, Tuple

from ..schema import Episode, RunManifest


def load_episodes(paths: List[Path], arm: str,
                  manifest_path: Optional[Path] = None) -> Tuple[List[Episode], Optional[RunManifest]]:
    out: List[Episode] = []
    for p in paths:
        p = Path(p)
        with (gzip.open(p, "rt") if p.suffix == ".gz" else p.open()) as f:
            for ln, line in enumerate(f, 1):
                if line.strip():
                    try:
                        ep = Episode.model_validate_json(line)
                    except Exception as e:
                        raise ValueError(f"{p}:{ln}: {e}") from e
                    out.append(ep.model_copy(update={"arm": arm}))
    manifest = RunManifest.model_validate_json(Path(manifest_path).read_text()) if manifest_path else None
    return out, manifest
