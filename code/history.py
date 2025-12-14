"""
history.py

Stores and loads past runs to provide a "History" view.
History is stored as JSON in code/results/history.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class HistoryItem:
    timestamp_iso: str
    source_type: str  # "upload" | "gallery"
    source_name: str  # filename
    kernel_size: int
    gaussian_sigma: float
    noise_mode: str
    noise_gaussian_sigma: float
    salt_pepper_prob: float
    boundary_mode: str
    outputs: Dict[str, str]  # saved filenames
    metrics: Dict[str, Dict[str, float]]

def history_path(results_dir: str) -> str:
    return os.path.join(results_dir, "history.json")


def load_history(results_dir: str) -> List[HistoryItem]:
    path = history_path(results_dir)

    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []  # empty file
            data = json.loads(content)
    except (json.JSONDecodeError, OSError):
        # corrupted or partially written file
        return []

    return [HistoryItem(**item) for item in data]


def append_history(results_dir: str, item: HistoryItem) -> None:
    items = load_history(results_dir)
    items.insert(0, item)

    path = history_path(results_dir)
    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in items], f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)  # atomic replace


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
