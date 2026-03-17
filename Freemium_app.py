import os
import sys
import csv
import time
import json
import atexit
import subprocess
from collections import Counter
from flask import Flask, render_template, jsonify, Response, send_from_directory
import requests

# --- Directories ---
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
REPORT_DIR    = os.path.join(BASE_DIR, "reports")   # kept for compat
STREAM_DIR    = os.path.join(BASE_DIR, "stream")    # main3.py writes latest.jpg + status.json here
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(STREAM_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# --- Files ---
MAIN3_PATH  = os.path.join(BASE_DIR, "main3.py")
CSV_LOG     = os.path.join(BASE_DIR, "posture_log.csv")
STREAM_PATH = os.path.join(STREAM_DIR, "latest.jpg")
STATUS_PATH = os.path.join(STREAM_DIR, "status.json")

# Ensure status file exists so /realtime has something on first load
if not os.path.exists(STATUS_PATH):
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "posture": "Calibrating",
                    "review": "Warming up the tracker…",
                    "metrics": {},
                    "updated": int(time.time() * 1000),
                },
                f,
            )
    except Exception:
        pass

# --- Process handle ---
tracker_process = None

# === OpenAI (from env var — never hardcode keys) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _read_status_file():
    try:
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    # baseline if read fails
    return {
        "posture": "Calibrating",
        "review": "Warming up the tracker…",
        "metrics": {},
        "updated": int(time.time() * 1000),
    }

def chatgpt_insights_json(summary_text: str):
    """
    Calls OpenAI and returns STRICT JSON for the report.
    No fallback. If missing key or any error -> return None and caller will 500.
    """
    if not OPENAI_API_KEY:
        print("[openai] Missing OPENAI_API_KEY env var.", flush=True)
        return None

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    system = (
        "You are a certified posture coach. Return STRICT JSON with keys: "
        "summary (string), did_well (array of strings), fix_order (array of strings), "
        "how_to_fix (array of strings), next_goal (string), tempo (string), set_target (string), "
        "workouts (array of strings), safety (array of strings). "
        "Write for a fifth-grade reading level. Use very short sentences (<= 12 words). "
        "Use simple words. No markdown. No extra keys."
    )
    user = "Make a super simple plan from this posture summary.\n\n" + summary_text

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 700,
    }

    try:
        r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=35)
        if not r.ok:
            print(f"[openai] HTTP {r.status_code}: {r.text[:300]}", flush=True)
            return None
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        # Validate required keys
        required = ["summary","did_well","fix_order","how_to_fix","next_goal","tempo","set_target"]
        if any(k not in parsed for k in required):
            print("[openai] Missing required keys in response.", flush=True)
            return None

        # Normalize types
        parsed["workouts"]   = list(map(str, parsed.get("workouts", [])))
        parsed["safety"]     = list(map(str, parsed.get("safety", [])))
        parsed["did_well"]   = list(map(str, parsed.get("did_well", [])))
        parsed["fix_order"]  = list(map(str, parsed.get("fix_order", [])))
        parsed["how_to_fix"] = list(map(str, parsed.get("how_to_fix", [])))
        for k in ["summary","next_goal","tempo","set_target"]:
            parsed[k] = str(parsed.get(k, ""))

        return parsed
    except Exception as e:
        print(f"[openai] Exception: {e}", flush=True)
        return None

def tracker_running() -> bool:
    return tracker_process is not None and tracker_process.poll() is None

def _cleanup_tracker():
    """Terminate tracker subprocess on app exit."""
    global tracker_process
    if tracker_running():
        try:
            tracker_process.terminate()
            tracker_process.wait(timeout=3)
        except Exception:
            pass
        tracker_process = None

atexit.register(_cleanup_tracker)

# ---------- CSV summarization (cached by mtime) ----------
_csv_cache = {"mtime": None, "summary": None}

def _summarize_csv(path, last_n=220):
    if not os.path.exists(path):
        return {"total": 0, "good": 0, "poor": 0, "modes": {}, "reasons": {}}

    mtime = os.path.getmtime(path)
    if _csv_cache["mtime"] == mtime and _csv_cache["summary"] is not None:
        return _csv_cache["summary"]

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows = rows[-last_n:] if rows else []
    total = len(rows)
    good  = sum(1 for r in rows if (r.get("status","").strip().lower() == "good"))
    poor  = total - good
    modes = Counter(r.get("mode","").strip() for r in rows if r.get("mode"))
    reasons = Counter()
    for r in rows:
        for t in (r.get("reasons") or "").split("|"):
            t = t.strip()
            if t: reasons[t] += 1
    result = {"total": total, "good": good, "poor": poor, "modes": dict(modes), "reasons": dict(reasons)}
    _csv_cache["mtime"] = mtime
    _csv_cache["summary"] = result
    return result

def _rule_based_panel(summary):
    # kept for compatibility (unused now that there's no fallback)
    total, good, poor = summary["total"], summary["good"], summary["poor"]
    common = [k for k,_ in sorted(summary["reasons"].items(), key=lambda kv: kv[1], reverse=True)]
    summary_line = "Good alignment. Keep it steady." if good >= poor else "You slouch or lean. Sit tall and keep chin tucked."
    panel = {
        "summary": summary_line,
        "did_well": ["Steady movement", "Consistent attempts"] if total else [],
        "fix_order": ["Show shoulders and elbows in frame", "Stack shoulders over hips", "Gentle chin tuck"],
        "how_to_fix": ["Adjust camera to see head, shoulders, hips", "Shoulder reset: up–back–down",
                       "Doorway pec stretch 30–45s", "Wall drill: head/upper-back/hips touch", "Record 10s and review"],
        "next_goal": "Keep shoulders and elbows visible throughout.",
        "tempo": "Smooth and steady.",
        "set_target": "1–2 sets × 8–12 reps focusing on form.",
        "workouts": ["Cat-cow 6–8 reps", "Thoracic extension over towel 1–2 min", "Chin tucks 10 reps, 3s hold",
                     "Band pull-aparts 12–15 reps", "Wall angels 8–10 reps"],
        "safety": ["Stop if pain or numbness.", "Move gently; no forcing ranges.", "Keep breathing; avoid bracing the neck."]
    }
    return panel

# ---------- MJPEG stream ----------
def _file_stream_generator():
    last_mtime, last_bytes = None, None
    blank = (b"\xff\xd8\xff" b"\xff\xd9")
    while True:
        try:
            if os.path.exists(STREAM_PATH):
                mtime = os.path.getmtime(STREAM_PATH)
                if mtime != last_mtime:
                    with open(STREAM_PATH, "rb") as f:
                        last_bytes = f.read()
                    last_mtime = mtime
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + (last_bytes or blank) + b"\r\n")
        except Exception:
            if last_bytes:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_bytes + b"\r\n")
        time.sleep(1/15.0)   # 15 FPS is sufficient for posture monitoring

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream")
def stream():
    return Response(_file_stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ===== Realtime with threshold logs =====
@app.route("/realtime")
def realtime():
    """
    Returns current posture + review + metrics.
    - Prefer metrics written by main3.py to status.json
    - If empty, fallback to latest CSV row with simple thresholds
    """
    data = _read_status_file()

    # Normalize metric keys to {actual, threshold} for the UI
    normalized = {}
    try:
        m = data.get("metrics", {}) or {}
        for k, v in m.items():
            actual = v.get("actual", v.get("monitor"))
            threshold = v.get("threshold", v.get("limit"))
            if actual is not None and threshold is not None:
                normalized[k] = {"actual": actual, "threshold": threshold}
    except Exception:
        normalized = {}

    # Fallback from CSV if we still have no metrics
    try:
        if not normalized and os.path.exists(CSV_LOG):
            with open(CSV_LOG, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                latest = rows[-1]
                # simple defaults
                thresholds = {"torso_fwd": 20.0, "neck_fwd": 18.0, "torso_lat": 10.0, "roll": 10.0, "z_side": 0.35}
                for k, thr in thresholds.items():
                    if k in latest:
                        val = float(latest.get(k, 0) or 0)
                        normalized[k] = {"actual": round(val, 1 if k != "z_side" else 3), "threshold": thr}
    except Exception:
        pass

    data["metrics"] = normalized
    return jsonify(data)

@app.route("/status")
def status():
    return jsonify({"running": tracker_running()})

@app.route("/start", methods=["POST"])
def start_tracking():
    global tracker_process
    if tracker_running():
        return jsonify({"status": "already running"})
    if not os.path.isfile(MAIN3_PATH):
        return jsonify({"status":"error","error":f"main3.py not found at {MAIN3_PATH}"}), 400
    env = os.environ.copy()
    env["STREAM_TO_FILE"] = "1"
    env["STREAM_DIR"]     = STREAM_DIR
    tracker_process = subprocess.Popen([sys.executable, MAIN3_PATH], env=env)
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop_tracking():
    global tracker_process
    if tracker_running():
        tracker_process.terminate()
        tracker_process = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not running"})

@app.route("/generate_report", methods=["POST"])
def generate_report():
    # Build a compact summary for the model
    summary = _summarize_csv(CSV_LOG, last_n=220)
    summary_text = (
        f"Total={summary['total']}; good={summary['good']}; poor={summary['poor']}; "
        f"reasons={summary['reasons']}"
    )

    panel = chatgpt_insights_json(summary_text)
    if panel is None:
        return jsonify({
            "status": "error",
            "error": "openai_failed_or_key_missing",
            "message": "Could not generate AI analysis. Check OPENAI_API_KEY and network."
        }), 500

    return jsonify({"status": "ok", "panel": panel})

@app.route("/reports/<path:filename>")
def get_report(filename):
    return send_from_directory(REPORT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
