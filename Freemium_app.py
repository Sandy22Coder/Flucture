import atexit
import json
import os
import subprocess
import sys
import time

import requests
from flask import Flask, Response, jsonify, render_template, send_from_directory

from analytics.session_analyzer import summarize_session
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from llm.retriever import retrieve_knowledge
from schemas.report_schema import validate_report_schema
from utils.pdf_report import build_posture_pdf
from utils.storage import load_latest_report, save_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
STREAM_DIR = os.path.join(BASE_DIR, "stream")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
TRACKER_LOG_PATH = os.path.join(STREAM_DIR, "tracker.log")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(STREAM_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

MAIN3_PATH = os.path.join(BASE_DIR, "main3.py")
CSV_LOG = os.path.join(BASE_DIR, "posture_log.csv")
STREAM_PATH = os.path.join(STREAM_DIR, "latest.jpg")
STATUS_PATH = os.path.join(STREAM_DIR, "status.json")

if not os.path.exists(STATUS_PATH):
    with open(STATUS_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "posture": "Calibrating",
                "review": "Warming up the tracker...",
                "metrics": {},
                "updated": int(time.time() * 1000),
            },
            handle,
        )

tracker_process = None
tracker_log_handle = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _read_status_file():
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {
            "posture": "Calibrating",
            "review": "Waiting for tracker...",
            "metrics": {},
            "updated": int(time.time() * 1000),
        }


def tracker_running() -> bool:
    return tracker_process is not None and tracker_process.poll() is None


def _cleanup_tracker():
    global tracker_process, tracker_log_handle
    if tracker_running():
        try:
            tracker_process.terminate()
            tracker_process.wait(timeout=3)
        except Exception:
            pass
        tracker_process = None
    if tracker_log_handle:
        try:
            tracker_log_handle.close()
        except Exception:
            pass
        tracker_log_handle = None


atexit.register(_cleanup_tracker)


def _file_stream_generator():
    blank = b"\xff\xd8\xff\xff\xd9"
    while True:
        try:
            if os.path.exists(STREAM_PATH):
                with open(STREAM_PATH, "rb") as handle:
                    frame_bytes = handle.read()
            else:
                frame_bytes = blank
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + (frame_bytes or blank) + b"\r\n"
        except Exception:
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + blank + b"\r\n"
        time.sleep(1 / 10.0)


def _normalize_metrics(data):
    normalized = {}
    metrics = data.get("metrics", {}) or {}
    for key, value in metrics.items():
        actual = value.get("actual", value.get("monitor"))
        threshold = value.get("threshold", value.get("limit"))
        if actual is not None and threshold is not None:
            normalized[key] = {"actual": actual, "threshold": threshold}
    return normalized


def _summary_from_status(data):
    metrics = _normalize_metrics(data)
    issues = []
    metric_to_issue = {
        "neck_fwd": "Forward head posture",
        "neck_lat": "Lateral neck tilt",
        "torso_fwd": "Slouching / rounded upper back",
        "torso_lat": "Sideways torso tilt",
        "roll": "Shoulder imbalance",
    }

    for metric_key, label in metric_to_issue.items():
        item = metrics.get(metric_key)
        if not item:
            continue
        actual = float(item["actual"])
        threshold = float(item["threshold"])
        if actual <= threshold:
            continue
        delta = actual - threshold
        severity = "low" if delta <= 2 else "moderate" if delta <= 6 else "high"
        issues.append(
            {
                "id": metric_key,
                "label": label,
                "count": 1,
                "severity": severity,
                "avg_excess": round(delta, 2),
                "evidence": [f"{metric_key} exceeded threshold by {delta:.1f}"],
            }
        )

    return {
        "total_frames": 1 if metrics else 0,
        "good_frames": 0 if issues else (1 if metrics else 0),
        "poor_frames": 1 if issues else 0,
        "good_ratio": 0.0 if issues else (1.0 if metrics else 0.0),
        "dominant_mode": "FRONT",
        "reason_counts": {},
        "metric_summary": {
            key: {"avg_actual": float(value["actual"]), "avg_threshold": float(value["threshold"])}
            for key, value in metrics.items()
        },
        "issues": issues,
        "session_confidence": "low",
    }


def _read_tracker_log_tail():
    if not os.path.exists(TRACKER_LOG_PATH):
        return ""
    try:
        with open(TRACKER_LOG_PATH, "r", encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]

        ignored_tokens = [
            "inference_feedback_manager.cc:114",
            "SymbolDatabase.GetPrototype() is deprecated",
            "INFO: Created TensorFlow Lite XNNPACK delegate for CPU.",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR",
            "warnings.warn(",
        ]
        filtered = [line for line in lines if not any(token in line for token in ignored_tokens)]
        return filtered[-1] if filtered else ""
    except Exception:
        return ""


def _pdf_name_from_report(report_filename: str) -> str:
    stem, _ = os.path.splitext(report_filename)
    return f"{stem}.pdf"


def _score_from_summary(summary):
    good_ratio = float(summary.get("good_ratio", 0.0))
    issue_penalty = min(40, len(summary.get("issues", [])) * 8)
    return max(0, min(100, int(good_ratio * 100) - issue_penalty + 25))


def _fallback_report(session_summary, previous_score):
    issues = session_summary.get("issues", [])
    current_score = _score_from_summary(session_summary)
    change = "same"
    if current_score > previous_score:
        change = "improved"
    elif current_score < previous_score:
        change = "declined"

    if not issues:
        issues = [
            {
                "label": "No dominant issue detected",
                "severity": "low",
                "evidence": ["The recent session stayed mostly within threshold."],
            }
        ]

    top_issues = [
        {
            "issue": issue["label"],
            "severity": issue["severity"],
            "evidence": issue.get("evidence") or ["Session evidence showed repeated deviation from baseline."],
        }
        for issue in issues[:3]
    ]

    possible_consequences = [
        {
            "issue": issue["label"],
            "risks": [
                "Can increase fatigue during longer study sessions.",
                "May reinforce inefficient movement habits over time.",
            ],
        }
        for issue in issues[:3]
    ]

    improvement_plan = []
    for index, issue in enumerate(issues[:3], start=1):
        improvement_plan.append(
            {
                "priority": index,
                "action": f"Correct {issue['label'].lower()} first.",
                "reason": "This issue appeared repeatedly in the session evidence.",
            }
        )

    return {
        "overall_assessment": "Moderate posture deviation detected. Focus on the top repeated issue first.",
        "risk_level": "moderate" if session_summary.get("poor_frames", 0) else "low",
        "what_is_wrong": top_issues,
        "possible_consequences": possible_consequences,
        "improvement_plan": improvement_plan,
        "remedies": {
            "stretches": [
                "Doorway chest stretch for 30 seconds.",
                "Gentle neck mobility resets between study blocks.",
            ],
            "strengthening": [
                "Band pull-aparts with slow control.",
                "Wall slides for scapular stability.",
            ],
            "daily_habits": [
                "Reset posture every 30 to 45 minutes.",
                "Keep the screen at eye level.",
            ],
            "ergonomic_corrections": [
                "Keep shoulders stacked over hips.",
                "Place both feet flat on the floor while seated.",
            ],
        },
        "red_flags": [
            "Seek professional help if pain, numbness, or tingling persists.",
            "Do not force neck or back range when pain increases.",
        ],
        "progress_score": {
            "current_score": current_score,
            "previous_score": previous_score,
            "change": change,
        },
    }


def _openai_report(session_summary, previous_score):
    if not OPENAI_API_KEY:
        return None

    retrieved_context = retrieve_knowledge(KNOWLEDGE_DIR, session_summary.get("issues", []))
    if not retrieved_context:
        return None

    user_prompt = build_user_prompt(
        {
            "session_summary": session_summary,
            "previous_score": previous_score,
            "current_score": _score_from_summary(session_summary),
        },
        retrieved_context,
    )

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1400,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=40)
        response.raise_for_status()
        response_payload = response.json()
        content = response_payload["choices"][0]["message"]["content"]
        report = json.loads(content)
        if not validate_report_schema(report):
            return None
        return report
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    response = Response(_file_stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/realtime")
def realtime():
    data = _read_status_file()
    data["metrics"] = _normalize_metrics(data)
    return jsonify(data)


@app.route("/status")
def status():
    return jsonify({"running": tracker_running(), "last_error": _read_tracker_log_tail()})


@app.route("/start", methods=["POST"])
def start_tracking():
    global tracker_process, tracker_log_handle
    if tracker_running():
        return jsonify({"status": "already running"})
    if not os.path.isfile(MAIN3_PATH):
        return jsonify({"status": "error", "error": f"main3.py not found at {MAIN3_PATH}"}), 400

    if tracker_log_handle:
        try:
            tracker_log_handle.close()
        except Exception:
            pass
        tracker_log_handle = None

    env = os.environ.copy()
    env["STREAM_TO_FILE"] = "1"
    env["STREAM_DIR"] = STREAM_DIR
    tracker_log_handle = open(TRACKER_LOG_PATH, "w", encoding="utf-8")
    tracker_process = subprocess.Popen(
        [sys.executable, MAIN3_PATH],
        env=env,
        stdout=tracker_log_handle,
        stderr=tracker_log_handle,
    )
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop_tracking():
    global tracker_process, tracker_log_handle
    if tracker_running():
        tracker_process.terminate()
        tracker_process = None
    if tracker_log_handle:
        try:
            tracker_log_handle.close()
        except Exception:
            pass
        tracker_log_handle = None
    if os.path.exists(TRACKER_LOG_PATH):
        return jsonify({"status": "stopped", "last_error": _read_tracker_log_tail()})
    return jsonify({"status": "not running"})


@app.route("/generate_report", methods=["POST"])
def generate_report():
    session_summary = summarize_session(CSV_LOG, last_n=300)
    if session_summary.get("total_frames", 0) == 0:
        session_summary = _summary_from_status(_read_status_file())

    previous_report = load_latest_report(REPORT_DIR) or {}
    previous_score = previous_report.get("panel", {}).get("progress_score", {}).get("current_score", 0)

    report = _openai_report(session_summary, previous_score)
    generator = "openai"
    if report is None:
        report = _fallback_report(session_summary, previous_score)
        generator = "fallback"

    payload = {
        "generated_at": int(time.time() * 1000),
        "generator": generator,
        "session_summary": session_summary,
        "panel": report,
    }
    report_path = save_report(REPORT_DIR, payload)
    pdf_filename = _pdf_name_from_report(os.path.basename(report_path))
    pdf_path = os.path.join(REPORT_DIR, pdf_filename)
    build_posture_pdf(pdf_path, payload)

    return jsonify(
        {
            "status": "ok",
            "generator": generator,
            "panel": report,
            "report_file": os.path.basename(report_path),
            "pdf_file": pdf_filename,
            "pdf_url": f"/reports/{pdf_filename}",
            "message": "Report generated from live posture evidence." if session_summary.get("total_frames", 0) else "Not enough posture evidence yet.",
        }
    )


@app.route("/reports/<path:filename>")
def get_report(filename):
    return send_from_directory(REPORT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
