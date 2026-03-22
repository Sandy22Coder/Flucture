import cv2
import numpy as np
import time
import os
import csv
import json
import sys
from collections import deque, Counter
from datetime import datetime

# ===== STREAM EXPORT =====
STREAM_TO_FILE = os.getenv("STREAM_TO_FILE", "0") == "1"
STREAM_DIR = os.getenv("STREAM_DIR", os.path.join(os.path.dirname(__file__), "stream"))
os.makedirs(STREAM_DIR, exist_ok=True)
STREAM_PATH = os.path.join(STREAM_DIR, "latest.jpg")
STATUS_PATH = os.path.join(STREAM_DIR, "status.json")

# Initialize status for UI on first load
try:
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump({"posture": "Calibrating", "review": "Hold still for a moment...",
                   "metrics": {}, "updated": int(time.time() * 1000)}, f)
except Exception:
    pass

def bootstrap_status(posture, review):
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"posture": posture, "review": review, "metrics": {}, "updated": int(time.time() * 1000)},
                f,
            )
    except Exception:
        pass

# ===== CONFIG =====
CALIBRATION_FRAMES = 20
SMOOTHING_ALPHA = 0.6
VISIBILITY_THRESHOLD = 0.15
HIP_VISIBILITY_THRESHOLD = 0.05
FRAME_HISTORY = 5
CSV_LOG = "posture_log.csv"

PAD_TORSO_FWD = 4.0
PAD_TORSO_LAT = 4.0
PAD_NECK_FWD = 6.0
PAD_NECK_LAT = 6.0
PAD_ROLL = 5.0

SIDE_Z_THR = 0.35
SIDE_LOCK_Z = 0.10
MODE_VOTE_WINDOW = 5
YAW_TO_FRONT = 17.0
YAW_TO_SIDE = 28.0
SHOULDER_SLOPE_MAX = 0.25

# Performance tuning
PROCESS_EVERY_N = 4          # Run MediaPipe every Nth frame
CSV_FLUSH_INTERVAL = 30      # Flush CSV buffer every N logged rows
MAX_CAMERA_RETRIES = 10      # Abort after N consecutive read failures
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "").strip().upper()

# Pre-computed constants
VERTICAL = np.array([0, -1, 0], dtype=np.float32)

# ===== MEDIAPIPE =====
try:
    import mediapipe as mp
except Exception as exc:
    bootstrap_status("Tracker error", f"MediaPipe import failed: {exc}")
    print(f"MediaPipe import failed: {exc}")
    sys.exit(1)

if not hasattr(mp, "solutions"):
    bootstrap_status(
        "Tracker error",
        "Installed mediapipe package is incompatible. Reinstall the official package in the same Python environment.",
    )
    print("Installed mediapipe package is incompatible: missing mp.solutions")
    sys.exit(1)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0,        # Lowest complexity for stability on student laptops
    smooth_landmarks=True
)

# ===== UTIL =====
smoothed = {}
recent_frames = deque(maxlen=FRAME_HISTORY)

def ema(prev, new, alpha=SMOOTHING_ALPHA):
    if prev is None:
        return np.array(new, dtype=np.float32)
    return alpha * np.array(new) + (1 - alpha) * np.array(prev)

def angle_between(v, w):
    v, w = np.array(v, dtype=np.float32), np.array(w, dtype=np.float32)
    nv, nw = np.linalg.norm(v), np.linalg.norm(w)
    if nv == 0 or nw == 0:
        return 0.0
    c = np.clip(np.dot(v, w) / (nv * nw), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def project_xy(vec): return np.array([vec[0], vec[1], 0.0], dtype=np.float32)
def project_yz(vec): return np.array([0.0, vec[1], vec[2]], dtype=np.float32)

def process_landmarks(landmarks):
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.NOSE
    ]
    coords = {}
    for e in required:
        lm = landmarks[e.value]
        if lm.visibility < VISIBILITY_THRESHOLD:
            return None
        coords[e] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    hip_points = []
    for e in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]:
        lm = landmarks[e.value]
        if lm.visibility >= HIP_VISIBILITY_THRESHOLD:
            coords[e] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            hip_points.append(e)

    if len(hip_points) == 1:
        visible = hip_points[0]
        missing = (
            mp_pose.PoseLandmark.RIGHT_HIP
            if visible == mp_pose.PoseLandmark.LEFT_HIP
            else mp_pose.PoseLandmark.LEFT_HIP
        )
        left_sh = coords[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_sh = coords[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_dx = right_sh[0] - left_sh[0]
        shoulder_dy = right_sh[1] - left_sh[1]
        shoulder_dz = right_sh[2] - left_sh[2]
        visible_hip = coords[visible]
        if visible == mp_pose.PoseLandmark.LEFT_HIP:
            coords[missing] = np.array(
                [visible_hip[0] + shoulder_dx, visible_hip[1] + shoulder_dy, visible_hip[2] + shoulder_dz],
                dtype=np.float32,
            )
        else:
            coords[missing] = np.array(
                [visible_hip[0] - shoulder_dx, visible_hip[1] - shoulder_dy, visible_hip[2] - shoulder_dz],
                dtype=np.float32,
            )
    elif len(hip_points) == 0:
        left_sh = coords[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_sh = coords[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = coords[mp_pose.PoseLandmark.NOSE]
        torso_drop = max(0.22, abs(((left_sh[1] + right_sh[1]) / 2.0) - nose[1]) * 2.4)
        coords[mp_pose.PoseLandmark.LEFT_HIP] = np.array(
            [left_sh[0], min(left_sh[1] + torso_drop, 0.98), left_sh[2]],
            dtype=np.float32,
        )
        coords[mp_pose.PoseLandmark.RIGHT_HIP] = np.array(
            [right_sh[0], min(right_sh[1] + torso_drop, 0.98), right_sh[2]],
            dtype=np.float32,
        )

    for k, v in coords.items():
        smoothed[k] = ema(smoothed.get(k), v)
    recent_frames.append(smoothed.copy())
    avg = {}
    for k in smoothed.keys():
        vals = [f[k] for f in recent_frames if k in f]
        avg[k] = np.mean(vals, axis=0)
    return avg

def compute_metrics(sm):
    LS = sm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    RS = sm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    LH = sm[mp_pose.PoseLandmark.LEFT_HIP]
    RH = sm[mp_pose.PoseLandmark.RIGHT_HIP]
    LE = sm[mp_pose.PoseLandmark.LEFT_EAR]
    RE = sm[mp_pose.PoseLandmark.RIGHT_EAR]
    NO = sm[mp_pose.PoseLandmark.NOSE]

    mid_sh = (LS + RS) / 2.0
    mid_hip = (LH + RH) / 2.0
    head = (LE + RE + NO) / 3.0
    neck_pt = mid_sh + (head - mid_sh) / 3.0

    spine = mid_sh - mid_hip
    head_rel = head - neck_pt

    torso_fwd = angle_between(project_yz(spine), VERTICAL)
    neck_fwd = angle_between(project_yz(head_rel), VERTICAL)
    torso_lat = angle_between(project_xy(spine), VERTICAL)
    neck_lat = angle_between(project_xy(head_rel), VERTICAL)

    dxy = RS - LS
    roll = np.degrees(np.arctan2(abs(dxy[1]), abs(dxy[0]) + 1e-6))
    z_side = float(mid_hip[2] - head[2])
    yaw_sh = np.degrees(np.arctan2(abs(RS[2] - LS[2]), abs(RS[0] - LS[0]) + 1e-6))
    yaw_hp = np.degrees(np.arctan2(abs(RH[2] - LH[2]), abs(RH[0] - LH[0]) + 1e-6))
    yaw = 0.6 * yaw_sh + 0.4 * yaw_hp
    slope_ok = (abs(RS[1] - LS[1]) < SHOULDER_SLOPE_MAX * max(abs(RS[0] - LS[0]), 1e-6))

    return {
        "mid_sh": mid_sh, "mid_hip": mid_hip, "head": head, "neck_pt": neck_pt,
        "torso_fwd": float(torso_fwd), "torso_lat": float(torso_lat),
        "neck_fwd": float(neck_fwd), "neck_lat": float(neck_lat),
        "roll": float(roll), "z_side": z_side, "yaw": float(yaw), "slope_ok": bool(slope_ok)
    }

def classify_mode(prev_mode, yaw, z_side, slope_ok):
    to_front = (yaw < YAW_TO_FRONT) and slope_ok
    to_side = (yaw > YAW_TO_SIDE)
    if prev_mode == "SIDE":
        return "FRONT" if to_front else "SIDE"
    return "SIDE" if to_side else "FRONT"


# ===== BUFFERED CSV WRITER =====
class BufferedCSVWriter:
    """Buffers CSV rows and flushes periodically to reduce I/O overhead."""

    def __init__(self, path, flush_every=CSV_FLUSH_INTERVAL):
        self.path = path
        self.flush_every = flush_every
        self._buffer = []
        self._header_written = os.path.isfile(path)

    def write(self, ts, mode, M, posture, reasons):
        self._buffer.append([
            ts, mode, f"{M['torso_fwd']:.1f}", f"{M['torso_lat']:.1f}",
            f"{M['neck_fwd']:.1f}", f"{M['neck_lat']:.1f}",
            f"{M['roll']:.1f}", f"{M['z_side']:.3f}", f"{M['yaw']:.1f}",
            posture, "|".join(reasons)
        ])
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        try:
            with open(self.path, "a", newline="") as f:
                w = csv.writer(f)
                if not self._header_written:
                    w.writerow(["timestamp", "mode", "torso_fwd", "torso_lat", "neck_fwd", "neck_lat",
                                "roll", "z_side", "yaw", "status", "reasons"])
                    self._header_written = True
                w.writerows(self._buffer)
            self._buffer.clear()
        except Exception as e:
            print(f"CSV flush error: {e}")


def write_status(posture, review, metrics=None):
    try:
        tmp = STATUS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "posture": posture,
                "review": review,
                "metrics": metrics or {},
                "updated": int(time.time() * 1000)
            }, f)
        os.replace(tmp, STATUS_PATH)
    except Exception:
        pass


def open_camera():
    """Try common Windows camera backends and indices before giving up."""

    backend_map = {
        "CAP_ANY": cv2.CAP_ANY,
        "CAP_DSHOW": cv2.CAP_DSHOW,
        "CAP_MSMF": cv2.CAP_MSMF,
    }
    if CAMERA_BACKEND in backend_map:
        backend_candidates = [(CAMERA_BACKEND, backend_map[CAMERA_BACKEND])]
    else:
        backend_candidates = [
            ("CAP_DSHOW", cv2.CAP_DSHOW),
            ("CAP_MSMF", cv2.CAP_MSMF),
            ("CAP_ANY", cv2.CAP_ANY),
        ]

    index_candidates = []
    for idx in [CAMERA_INDEX, 0, 1, 2]:
        if idx not in index_candidates:
            index_candidates.append(idx)

    for index in index_candidates:
        for name, backend in backend_candidates:
            try:
                cap = cv2.VideoCapture(index, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    cap.release()
                    continue

                good_reads = 0
                last_shape = None
                for _ in range(8):
                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        time.sleep(0.05)
                        continue
                    last_shape = frame.shape
                    if _frame_looks_valid(frame):
                        good_reads += 1
                    time.sleep(0.03)

                if good_reads >= 2:
                    write_status(
                        "Calibrating",
                        f"Camera opened with {name} on index {index}. Hold still for calibration.",
                        {},
                    )
                    print(f"Camera backend selected: {name} on index {index}, shape={last_shape}")
                    return cap

                print(f"Rejected camera candidate {name} index {index} due to unstable/corrupt frames.")
                cap.release()
            except Exception as exc:
                print(f"Camera backend {name} index {index} failed: {exc}")

    write_status(
        "Tracker error",
        f"Could not read usable frames from camera indices {index_candidates}. Try another camera or close apps using the webcam.",
        {},
    )
    return None


def _frame_looks_valid(frame):
    """Reject obviously corrupted frames with stripe/noise artifacts."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return False

    std = float(np.std(gray))
    row_diff = float(np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0))))
    col_diff = float(np.mean(np.abs(np.diff(gray.astype(np.float32), axis=1))))

    if std < 4.0:
        return False

    # Corrupted stripe frames tend to have much stronger column repetition than real images.
    if col_diff < 1.0 and row_diff > 8.0:
        return False

    return True

def draw_overlay(frame, res, M, proc):
    h, w = frame.shape[:2]
    mp_drawing.draw_landmarks(
        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
    )
    pN = tuple((M["neck_pt"][:2] * [w, h]).astype(int))
    head = tuple((M["head"][:2] * [w, h]).astype(int))
    mhip = (proc[mp_pose.PoseLandmark.LEFT_HIP] + proc[mp_pose.PoseLandmark.RIGHT_HIP]) / 2.0
    pH = tuple((mhip[:2] * [w, h]).astype(int))
    cv2.circle(frame, pN, 6, (0, 255, 255), -1)
    cv2.line(frame, pN, pH, (60, 200, 60), 2)
    ctrl = ((np.array(pN) + np.array(head)) // 2) - np.array([0, 12])
    prevp = pN
    for t in np.linspace(0, 1, 16):
        x = int((1-t)**2 * pN[0] + 2*(1-t)*t*ctrl[0] + t*t*head[0])
        y = int((1-t)**2 * pN[1] + 2*(1-t)*t*ctrl[1] + t*t*head[1])
        cv2.line(frame, prevp, (x, y), (0, 255, 255), 2)
        prevp = (x, y)


def evaluate_posture(M, mode, thresholds):
    """Evaluate posture against calibrated thresholds. Returns (label, reasons, metrics_dict)."""
    thr_torso_fwd, thr_torso_lat, thr_neck_fwd, thr_neck_lat, thr_roll = thresholds
    reasons = []

    if mode == "FRONT":
        if M["torso_fwd"] > thr_torso_fwd:
            reasons.append("Torso leaning forward")
        if M["torso_lat"] > thr_torso_lat:
            reasons.append("Torso tilting sideways")
        if M["neck_fwd"] > thr_neck_fwd:
            reasons.append("Head too far forward")
        if M["neck_lat"] > thr_neck_lat:
            reasons.append("Head tilting sideways")
        if M["roll"] > thr_roll:
            reasons.append("Shoulders uneven")
    else:  # SIDE
        if M["torso_fwd"] > thr_torso_fwd:
            reasons.append("Slouching forward")
        if M["neck_fwd"] > thr_neck_fwd:
            reasons.append("Head jutting forward")

    posture = "Good" if not reasons else "Poor"

    metrics_dict = {
        "torso_fwd": {"actual": round(M["torso_fwd"], 1), "threshold": round(thr_torso_fwd, 1)},
        "torso_lat": {"actual": round(M["torso_lat"], 1), "threshold": round(thr_torso_lat, 1)},
        "neck_fwd":  {"actual": round(M["neck_fwd"], 1),  "threshold": round(thr_neck_fwd, 1)},
        "neck_lat":  {"actual": round(M["neck_lat"], 1),  "threshold": round(thr_neck_lat, 1)},
        "roll":      {"actual": round(M["roll"], 1),       "threshold": round(thr_roll, 1)},
        "z_side":    {"actual": round(M["z_side"], 3),     "threshold": SIDE_Z_THR},
    }

    review = "Good posture - keep it up!" if posture == "Good" else "; ".join(reasons)
    return posture, reasons, metrics_dict, review


# ===== MAIN LOOP =====
cap = open_camera()
if cap is None:
    print("Could not open camera.")
    sys.exit(1)

print("Camera opened successfully - starting posture detection...")

is_calibrated = False
buf_torso_fwd, buf_torso_lat = [], []
buf_neck_fwd, buf_neck_lat = [], []
buf_roll = []

thr_torso_fwd = thr_torso_lat = thr_neck_fwd = thr_neck_lat = thr_roll = 0.0
head_tilt_baseline = 20.0
prev_mode = "FRONT"
mode_votes = deque(maxlen=MODE_VOTE_WINDOW)

csv_writer = BufferedCSVWriter(CSV_LOG)
frame_count = 0
camera_fail_count = 0
last_res = None           # Cache last MediaPipe result for frame skipping

while True:
    ok, frame = cap.read()
    if not ok:
        camera_fail_count += 1
        if camera_fail_count >= MAX_CAMERA_RETRIES:
            print(f"Camera failed {MAX_CAMERA_RETRIES} times in a row - exiting.")
            break
        wait = min(0.1 * (2 ** (camera_fail_count - 1)), 2.0)
        print(f"Frame read failed ({camera_fail_count}/{MAX_CAMERA_RETRIES}), retrying in {wait:.1f}s...")
        time.sleep(wait)
        continue

    camera_fail_count = 0   # Reset on successful read
    frame_count += 1
    frame = cv2.flip(frame, 1)

    # --- Frame skipping: run MediaPipe every Nth frame ---
    if frame_count % PROCESS_EVERY_N == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_res = pose.process(rgb)

    res = last_res
    if res is None or res.pose_landmarks is None:
        write_status("Calibrating", "No landmarks detected", {})
    else:
        proc = process_landmarks(res.pose_landmarks.landmark)
        if proc is not None:
            M = compute_metrics(proc)
            inst = classify_mode(prev_mode, M["yaw"], M["z_side"], M["slope_ok"])
            mode_votes.append(inst)
            mode = Counter(mode_votes).most_common(1)[0][0]
            prev_mode = mode

            if not is_calibrated and len(buf_torso_fwd) < CALIBRATION_FRAMES:
                buf_torso_fwd.append(M["torso_fwd"])
                buf_torso_lat.append(M["torso_lat"])
                buf_neck_fwd.append(M["neck_fwd"])
                buf_neck_lat.append(M["neck_lat"])
                buf_roll.append(M["roll"])
                write_status(
                    "Calibrating",
                    f"Body detected. Hold still... {len(buf_torso_fwd)}/{CALIBRATION_FRAMES}",
                    {},
                )
                if len(buf_torso_fwd) % 10 == 0:
                    print(f"Calibrating... {len(buf_torso_fwd)}/{CALIBRATION_FRAMES}")
            elif not is_calibrated:
                thr_torso_fwd = float(np.mean(buf_torso_fwd) + PAD_TORSO_FWD)
                thr_torso_lat = float(np.mean(buf_torso_lat) + PAD_TORSO_LAT)
                thr_neck_fwd = float(np.mean(buf_neck_fwd) + PAD_NECK_FWD)
                thr_neck_lat = float(np.mean(buf_neck_lat) + PAD_NECK_LAT)
                thr_roll = float(np.mean(buf_roll) + PAD_ROLL)
                head_tilt_baseline = max(20.0, thr_neck_fwd)
                is_calibrated = True
                write_status("Good", "Calibration complete. Begin normal posture tracking.", {})
                print("Calibration complete!")
                print(f"   Thresholds -> torso_fwd={thr_torso_fwd:.1f} deg torso_lat={thr_torso_lat:.1f} deg "
                      f"neck_fwd={thr_neck_fwd:.1f} deg neck_lat={thr_neck_lat:.1f} deg roll={thr_roll:.1f} deg")

            # --- Posture evaluation (runs every frame after calibration) ---
            if is_calibrated:
                thresholds = (thr_torso_fwd, thr_torso_lat, thr_neck_fwd, thr_neck_lat, thr_roll)
                posture, reasons, metrics_dict, review = evaluate_posture(M, mode, thresholds)

                # Write status so the Flask UI gets live data
                write_status(posture, review, metrics_dict)

                # Log to CSV (buffered)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.write(ts, mode, M, posture, reasons)

                # Draw posture label on frame
                color = (0, 255, 0) if posture == "Good" else (0, 0, 255)
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Posture: {posture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Mode: {mode}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if reasons:
                    for i, r in enumerate(reasons[:3]):
                        cv2.putText(frame, r, (10, 90 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            # Draw overlay for testing
            draw_overlay(frame, res, M, proc)

        else:
            write_status("Calibrating", "Ensure head, shoulders and hips are visible", {})

    # Save stream frame (with retry for Windows file-locking)
    ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok_enc:
        tmp = STREAM_PATH + ".tmp"
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        for _attempt in range(3):
            try:
                os.replace(tmp, STREAM_PATH)
                break
            except PermissionError:
                time.sleep(0.01)  # brief wait for Flask reader to release

    # Allow graceful exit if you run this directly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
csv_writer.flush()
cap.release()
cv2.destroyAllWindows()
print("Posture detection stopped.")
