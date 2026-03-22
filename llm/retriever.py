from __future__ import annotations

import os
from typing import Dict, List


ISSUE_TO_FILE = {
    "forward_head_posture": "forward_head.md",
    "slouching": "slouching.md",
    "shoulder_imbalance": "shoulder_imbalance.md",
    "lateral_neck_tilt": "lateral_neck_tilt.md",
    "torso_tilt": "torso_tilt.md",
}


def retrieve_knowledge(base_dir: str, issues: List[Dict[str, object]], limit: int = 3) -> str:
    chunks = []
    for issue in issues[:limit]:
        filename = ISSUE_TO_FILE.get(issue["id"])
        if not filename:
            continue
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            chunks.append(handle.read().strip())
    return "\n\n".join(chunks)
