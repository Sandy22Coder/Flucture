from __future__ import annotations

from typing import Dict, List


ISSUE_RULES = {
    "forward_head_posture": {
        "label": "Forward head posture",
        "reason_tokens": ["Head too far forward", "Head jutting forward"],
        "metric_keys": ["neck_fwd"],
    },
    "slouching": {
        "label": "Slouching / rounded upper back",
        "reason_tokens": ["Torso leaning forward", "Slouching forward"],
        "metric_keys": ["torso_fwd"],
    },
    "shoulder_imbalance": {
        "label": "Shoulder imbalance",
        "reason_tokens": ["Shoulders uneven"],
        "metric_keys": ["roll"],
    },
    "lateral_neck_tilt": {
        "label": "Lateral neck tilt",
        "reason_tokens": ["Head tilting sideways"],
        "metric_keys": ["neck_lat"],
    },
    "torso_tilt": {
        "label": "Sideways torso tilt",
        "reason_tokens": ["Torso tilting sideways"],
        "metric_keys": ["torso_lat"],
    },
}


def severity_from_excess(excess: float) -> str:
    if excess <= 2:
        return "low"
    if excess <= 6:
        return "moderate"
    return "high"


def build_issue_summary(reason_counts: Dict[str, int], metric_summary: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    """Map raw reasons and metric excess into canonical posture issues."""
    issues: List[Dict[str, object]] = []

    for issue_id, rule in ISSUE_RULES.items():
        matched_reasons = []
        count = 0
        max_excess = 0.0

        for token in rule["reason_tokens"]:
            token_count = int(reason_counts.get(token, 0))
            if token_count > 0:
                matched_reasons.append(token)
                count += token_count

        for metric_key in rule["metric_keys"]:
            metric = metric_summary.get(metric_key)
            if not metric:
                continue
            avg_excess = max(0.0, float(metric.get("avg_actual", 0.0)) - float(metric.get("avg_threshold", 0.0)))
            max_excess = max(max_excess, avg_excess)

        if count == 0 and max_excess <= 0:
            continue

        issues.append(
            {
                "id": issue_id,
                "label": rule["label"],
                "count": count,
                "severity": severity_from_excess(max_excess),
                "avg_excess": round(max_excess, 2),
                "evidence": matched_reasons,
            }
        )

    issues.sort(key=lambda item: (item["count"], item["avg_excess"]), reverse=True)
    return issues
