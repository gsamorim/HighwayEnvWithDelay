"""Lightweight CSV + JSON run logger (no external experiment-tracking
dependency needed)."""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any


class RunLogger:
    """Buffers rows in memory and rewrites the CSV on each flush so that
    columns which only appear later (e.g. SAC's loss/alpha metrics, which are
    absent from the very first logged row during the random-action warmup
    phase) are never silently dropped by a fixed CSV header."""

    def __init__(self, log_dir: str, run_name: str, flush_every: int = 20):
        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{run_name}.csv")
        self.meta_path = os.path.join(log_dir, f"{run_name}.meta.json")
        self._rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] = []
        self._flush_every = flush_every
        self._t0 = time.time()

    def log_meta(self, meta: dict[str, Any]) -> None:
        meta = dict(meta)
        meta["logged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def log_row(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("wall_time_s", round(time.time() - self._t0, 2))
        self._rows.append(row)
        for k in row:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        if len(self._rows) % self._flush_every == 0:
            self._write_csv()

    def _write_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})

    def close(self) -> None:
        self._write_csv()
