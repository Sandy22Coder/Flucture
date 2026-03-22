from __future__ import annotations


SYSTEM_PROMPT = """
You are an expert posture coach creating a grounded wellness report.
Use only the provided session evidence and retrieved knowledge.
Do not claim to diagnose medical conditions.
Be conservative, practical, and specific.
Return strict JSON with the exact schema requested.
""".strip()


def build_user_prompt(session_summary: dict, retrieved_context: str) -> str:
    return (
        "Create a posture report from this structured evidence.\n\n"
        f"Session summary:\n{session_summary}\n\n"
        f"Retrieved knowledge:\n{retrieved_context}\n\n"
        "Required JSON fields:\n"
        "{"
        "\"overall_assessment\": string, "
        "\"risk_level\": \"low|moderate|high\", "
        "\"what_is_wrong\": [{\"issue\": string, \"severity\": \"low|moderate|high\", \"evidence\": [string]}], "
        "\"possible_consequences\": [{\"issue\": string, \"risks\": [string]}], "
        "\"improvement_plan\": [{\"priority\": number, \"action\": string, \"reason\": string}], "
        "\"remedies\": {\"stretches\": [string], \"strengthening\": [string], \"daily_habits\": [string], \"ergonomic_corrections\": [string]}, "
        "\"red_flags\": [string], "
        "\"progress_score\": {\"current_score\": number, \"previous_score\": number, \"change\": \"improved|same|declined\"}"
        "}"
    )
