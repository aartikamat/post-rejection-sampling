"""prfs.provenance - structured logging + run manifests.

Implements FORMAL_SPEC section 10. Every reference-implementation run emits
a scope.json and per-tick structured JSON logs so consumers can trace any
number back to the tick that produced it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    prfs_version: str
    spec_version: str
    started_at_ms: str
    ended_at_ms: str | None
    input_registry_sha256: str
    input_outcomes_sha256: str
    tracker_epoch: int
    config: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def write_manifest(manifest: RunManifest, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(asdict(manifest), fh, indent=2, sort_keys=True)


def emit_log(event: dict[str, Any], sink: Path) -> None:
    """Append a single JSON-line record to sink. Every record must carry
    (event_id | None, tick_ts, kind, payload)."""
    for key in ("event_id", "tick_ts", "kind", "payload"):
        if key not in event:
            raise ValueError(f"emit_log record missing required field {key!r}")
    p = Path(sink)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
