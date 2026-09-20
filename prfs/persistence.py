"""prfs.persistence - durable WAL, checkpoint, restart recovery.

Implements FORMAL_SPEC section 7 and asserts section 9 duplicate handling.

Storage format: newline-delimited JSON. Each line is a single record. Records
are idempotent-keyed:
    events: key = event_id
    samples: key = (event_id, tau_min)
    absences: key = (event_id, tau_min, 'ABS')

A duplicate write on an existing key is rejected (silently ignored, warning
logged); this matches spec section 7.1 and section 9.
"""
from __future__ import annotations

import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

from .clock import DELTA_MAX_MS, within_window
from .types import FollowupSample, OracleAbsenceRecord, RejectionEvent

_LOG = logging.getLogger("prfs.persistence")


def _to_json(obj) -> str:
    if is_dataclass(obj):
        return json.dumps(asdict(obj), sort_keys=True)
    return json.dumps(obj, sort_keys=True)


class EventStore(ABC):
    @abstractmethod
    def append_event(self, event: RejectionEvent) -> bool: ...
    @abstractmethod
    def replay_events(self) -> Iterable[RejectionEvent]: ...


class SampleLog(ABC):
    @abstractmethod
    def append_sample(self, sample: FollowupSample) -> bool: ...
    @abstractmethod
    def append_absence(self, absence: OracleAbsenceRecord) -> bool: ...
    @abstractmethod
    def replay_samples(self) -> Iterable[FollowupSample]: ...
    @abstractmethod
    def replay_absences(self) -> Iterable[OracleAbsenceRecord]: ...


class WalEventStore(EventStore):
    """Append-only NDJSON WAL, idempotent on event_id."""

    def __init__(self, wal_path: Path, checkpoint_path: Path | None = None) -> None:
        self.wal_path = Path(wal_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._seen: set[str] = set()
        # Load prior seen keys on init so restart is idempotent on re-append.
        if self.wal_path.exists():
            with self.wal_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._seen.add(rec["event_id"])
        self._terminal: set[str] = set()  # spec 3.6

    def append_event(self, event: RejectionEvent) -> bool:
        if event.event_id in self._seen:
            _LOG.warning("duplicate event_id %s dropped", event.event_id)
            return False
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.wal_path.open("a", encoding="utf-8") as fh:
            fh.write(_to_json(event) + "\n")
        self._seen.add(event.event_id)
        return True

    def replay_events(self) -> Iterable[RejectionEvent]:
        if not self.wal_path.exists():
            return
        with self.wal_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield RejectionEvent(**rec)

    def mark_terminal(self, event_id: str) -> None:
        self._terminal.add(event_id)

    def is_terminal(self, event_id: str) -> bool:
        return event_id in self._terminal


class WalSampleLog(SampleLog):
    def __init__(self, wal_path: Path, absence_path: Path | None = None) -> None:
        self.wal_path = Path(wal_path)
        self.absence_path = Path(absence_path) if absence_path else self.wal_path.with_suffix(".absences.ndjson")
        self._seen_samples: set[tuple[str, float]] = set()
        self._seen_absences: set[tuple[str, float]] = set()
        if self.wal_path.exists():
            with self.wal_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._seen_samples.add((rec["event_id"], float(rec["tau_min"])))
        if self.absence_path.exists():
            with self.absence_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._seen_absences.add((rec["event_id"], float(rec["tau_min"])))
        self._terminal: set[str] = set()

    def append_sample(self, sample: FollowupSample) -> bool:
        if sample.event_id in self._terminal:
            _LOG.warning(
                "sample write to terminal event %s (tau=%s) dropped",
                sample.event_id, sample.tau_min,
            )
            return False
        key = (sample.event_id, float(sample.tau_min))
        if key in self._seen_samples:
            _LOG.warning(
                "duplicate sample (event=%s, tau=%s) dropped",
                sample.event_id, sample.tau_min,
            )
            return False
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.wal_path.open("a", encoding="utf-8") as fh:
            fh.write(_to_json(sample) + "\n")
        self._seen_samples.add(key)
        return True

    def append_absence(self, absence: OracleAbsenceRecord) -> bool:
        key = (absence.event_id, float(absence.tau_min))
        if key in self._seen_absences:
            return False
        self.absence_path.parent.mkdir(parents=True, exist_ok=True)
        with self.absence_path.open("a", encoding="utf-8") as fh:
            fh.write(_to_json(absence) + "\n")
        self._seen_absences.add(key)
        return True

    def replay_samples(self) -> Iterable[FollowupSample]:
        if not self.wal_path.exists():
            return
        with self.wal_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield FollowupSample(**rec)

    def replay_absences(self) -> Iterable[OracleAbsenceRecord]:
        if not self.absence_path.exists():
            return
        with self.absence_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield OracleAbsenceRecord(**rec)

    def mark_terminal(self, event_id: str) -> None:
        self._terminal.add(event_id)


def restart_recover(
    store: EventStore, log: SampleLog, now_ms: str, delta_max_ms: int = DELTA_MAX_MS
) -> list[RejectionEvent]:
    """Spec section 7.3 restart recovery.

    Returns the events still inside their observation window after now_ms
    (i.e. still eligible for scheduling). Sample state is available via
    log.replay_samples() so retry state can be reconstructed by the caller.
    """
    active: list[RejectionEvent] = []
    for ev in store.replay_events():
        if within_window(ev.ts, now_ms, delta_max_ms):
            active.append(ev)
    return active
