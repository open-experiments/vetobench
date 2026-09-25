"""Hash-chained JSONL audit log: one record per proposed tool call.

Each record carries ``prev_hash`` (the previous record's ``hash``, or 64 zeros for the first)
and ``hash`` = sha256 over the canonical JSON of the record without ``hash``. Deleting,
reordering or editing any record breaks the chain, which ``verify_chain`` detects.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from . import AUDIT_SCHEMA

GENESIS = "0" * 64
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def canonical(record: dict) -> bytes:
    body = {k: v for k, v in record.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def record_hash(record: dict) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


def read_records(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def verify_chain(path: str | Path) -> tuple[bool, int, str]:
    """Return (ok, records_checked, message)."""
    prev = GENESIS
    n = 0
    for n, rec in enumerate(read_records(path), start=1):
        if rec.get("prev_hash") != prev:
            return False, n, f"record {n}: prev_hash does not match previous record"
        if rec.get("hash") != record_hash(rec):
            return False, n, f"record {n}: hash mismatch (record was modified)"
        prev = rec["hash"]
    return True, n, "ok"


class AuditLog:
    """Appends chained records to ``<dir>/<run_id>.jsonl``; one chain per run id."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last: dict[Path, str] = {}
        self._seq: dict[Path, int] = {}

    def path_for(self, run_id: str | None) -> Path:
        name = _SAFE_NAME.sub("_", run_id or "adhoc").strip("_") or "adhoc"
        return self.dir / f"{name}.jsonl"

    def _tail(self, path: Path) -> tuple[str, int]:
        if path not in self._last:
            last, seq = GENESIS, 0
            if path.exists():
                for rec in read_records(path):
                    last, seq = rec["hash"], rec.get("seq", seq + 1)
            self._last[path], self._seq[path] = last, seq
        return self._last[path], self._seq[path]

    def append(self, run_id: str | None, fields: dict[str, Any]) -> dict:
        path = self.path_for(run_id)
        with self._lock:
            prev, seq = self._tail(path)
            rec = {
                "schema": AUDIT_SCHEMA,
                "seq": seq + 1,
                "ts": time.time(),
                "run_id": run_id,
                **fields,
                "prev_hash": prev,
            }
            rec["hash"] = record_hash(rec)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._last[path], self._seq[path] = rec["hash"], rec["seq"]
        return rec
