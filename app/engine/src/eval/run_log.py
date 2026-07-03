from __future__ import annotations

import json
import os
import time
from pathlib import Path


class RunLogger:
  def __init__(self, path: str | os.PathLike):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)

  def record(self, config: dict, metrics: dict, note: str = "") -> dict:
    entry = {
      "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
      "note": note,
      "config": config,
      "metrics": metrics,
    }
    with open(self.path, "a") as f:
      f.write(json.dumps(entry) + "\n")
    return entry

  def load(self) -> list[dict]:
    if not self.path.exists():
      return []
    with open(self.path) as f:
      return [json.loads(line) for line in f if line.strip()]

  def best(self, metric: str = "psnr_db", maximize: bool = True) -> dict | None:
    runs = [r for r in self.load() if metric in r.get("metrics", {})]
    if not runs:
      return None
    return (max if maximize else min)(runs, key=lambda r: r["metrics"][metric])
