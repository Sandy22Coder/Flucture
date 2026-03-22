from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict


def save_report(report_dir: str, payload: Dict[str, object]) -> str:
    os.makedirs(report_dir, exist_ok=True)
    filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(report_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_latest_report(report_dir: str):
    if not os.path.exists(report_dir):
        return None
    files = sorted(
        [os.path.join(report_dir, name) for name in os.listdir(report_dir) if name.endswith(".json")],
        reverse=True,
    )
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as handle:
        return json.load(handle)
