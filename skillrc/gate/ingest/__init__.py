"""Adapters: source log formats -> schema.Episode / schema.RunManifest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schema import Episode, RunManifest


def load(fmt: str, paths: List[Path], arm: str, manifest_path: Optional[Path] = None,
         options: Optional[Dict[str, Any]] = None) -> Tuple[List[Episode], Optional[RunManifest]]:
    options = options or {}
    if fmt == "skillrc":
        from .skillrc_jsonl import load_skillrc
        return load_skillrc(paths, arm, manifest_path)
    if fmt == "episodes":
        from .episodes_jsonl import load_episodes
        return load_episodes(paths, arm, manifest_path)
    if fmt == "inspect":
        from .inspect_log import load_inspect
        return load_inspect(paths, arm, manifest_path, **options)
    raise ValueError(f"unknown log format {fmt!r}")
