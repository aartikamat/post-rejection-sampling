"""tests/test_provenance.py - spec section 10."""
from __future__ import annotations

import json

import pytest

from prfs.provenance import RunManifest, emit_log, write_manifest


def test_write_manifest_roundtrip(tmp_path):
    m = RunManifest(
        prfs_version="2.0.0",
        spec_version="v0.1",
        started_at_ms="2026-04-15T00:00:00.000Z",
        ended_at_ms="2026-04-15T00:05:00.000Z",
        input_registry_sha256="abc",
        input_outcomes_sha256="def",
        tracker_epoch=1,
        config={"scheduler": "fixed"},
        counts={"events": 100},
    )
    out = tmp_path / "manifest.json"
    write_manifest(m, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["prfs_version"] == "2.0.0"
    assert data["counts"]["events"] == 100


def test_emit_log_requires_fields(tmp_path):
    with pytest.raises(ValueError):
        emit_log({"event_id": "e", "tick_ts": "t"}, tmp_path / "log.ndjson")


def test_emit_log_appends_newlines(tmp_path):
    log = tmp_path / "log.ndjson"
    for i in range(3):
        emit_log({"event_id": f"e{i}", "tick_ts": "t",
                  "kind": "sample", "payload": {"i": i}}, log)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # parseable
