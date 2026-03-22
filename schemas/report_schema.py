from __future__ import annotations


REQUIRED_FIELDS = {
    "overall_assessment": str,
    "risk_level": str,
    "what_is_wrong": list,
    "possible_consequences": list,
    "improvement_plan": list,
    "remedies": dict,
    "red_flags": list,
    "progress_score": dict,
}


def validate_report_schema(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    for key, expected_type in REQUIRED_FIELDS.items():
        if key not in payload or not isinstance(payload[key], expected_type):
            return False
    return True
