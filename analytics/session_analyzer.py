from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from typing import Dict, List

from analytics.severity_rules import build_issue_summary


METRIC_KEYS = ["torso_fwd", "torso_lat", "neck_fwd", "neck_lat", "roll", "z_side"]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_session_rows(csv_path: str, last_n: int = 300) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-last_n:] if last_n and rows else rows


def summarize_session(csv_path: str, last_n: int = 300) -> Dict[str, object]:
    rows = load_session_rows(csv_path, last_n=last_n)
    if not rows:
        return {
            "total_frames": 0,
            "good_frames": 0,
            "poor_frames": 0,
            "good_ratio": 0.0,
            "dominant_mode": "UNKNOWN",
            "reason_counts": {},
            "metric_summary": {},
            "issues": [],
            "session_confidence": "low",
        }

    total_frames = len(rows)
    good_frames = sum(1 for row in rows if (row.get("status", "").strip().lower() == "good"))
    poor_frames = total_frames - good_frames
    dominant_mode = Counter(row.get("mode", "").strip() for row in rows if row.get("mode")).most_common(1)
    dominant_mode = dominant_mode[0][0] if dominant_mode else "UNKNOWN"

    reason_counts = Counter()
    metric_buckets = defaultdict(lambda: {"actual_total": 0.0, "count": 0, "threshold_total": 0.0})

    for row in rows:
        reasons = [item.strip() for item in (row.get("reasons") or "").split("|") if item.strip()]
        for reason in reasons:
            reason_counts[reason] += 1

        for key in METRIC_KEYS:
            if key not in row:
                continue
            metric_buckets[key]["actual_total"] += _safe_float(row.get(key))
            metric_buckets[key]["count"] += 1

    threshold_map = _estimate_thresholds(rows)
    metric_summary = {}
    for key, bucket in metric_buckets.items():
        count = max(1, bucket["count"])
        metric_summary[key] = {
            "avg_actual": round(bucket["actual_total"] / count, 2),
            "avg_threshold": round(threshold_map.get(key, 0.0), 2),
        }

    issues = build_issue_summary(dict(reason_counts), metric_summary)
    session_confidence = "high" if total_frames >= 120 else "medium" if total_frames >= 45 else "low"

    return {
        "total_frames": total_frames,
        "good_frames": good_frames,
        "poor_frames": poor_frames,
        "good_ratio": round(good_frames / max(1, total_frames), 3),
        "dominant_mode": dominant_mode,
        "reason_counts": dict(reason_counts),
        "metric_summary": metric_summary,
        "issues": issues,
        "session_confidence": session_confidence,
    }


def _estimate_thresholds(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """Back-fill rough thresholds for older CSV rows that do not store them."""
    defaults = {
        "torso_fwd": 20.0,
        "torso_lat": 10.0,
        "neck_fwd": 18.0,
        "neck_lat": 10.0,
        "roll": 10.0,
        "z_side": 0.35,
    }
    return defaults
