"""
================================================================
  SAMVAAD — Flask Backend  (app.py)
  Place this file in:  D:\SAMVAAD\templates\

  HOW TO RUN:
      cd D:\SAMVAAD\templates
      python app.py

  Then open:  http://localhost:5000

  FIXES in this version:
    1. `if data is None` instead of `if not data` — empty list [] is
       a valid payload (no hands detected) and must NOT return 400.
    2. lm_array() in sign_recog.py now accepts numpy arrays directly,
       so classify_right/left_hand work correctly when called from Flask.
    3. Both fetch paths in sign.html now throw on non-ok HTTP status,
       so _requestInFlight is always cleared even on server errors.
================================================================
"""

import json
import re
import threading
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import sys, os

# ── sign_recog.py is one level up (D:\SAMVAAD\sign_recog.py) ─────────────────
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _parent)

from sign_recog import (
    classify_right_hand,
    classify_left_hand,
    StableGesture,
)

# Tuned for ~30fps landmark input — fast server-side debounce
CONFIRM_FRAMES = 3
RELEASE_FRAMES = 3

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)

# Per-client StableGesture state
_stable_states = {}
_states_lock   = threading.Lock()
_samples_dir   = os.path.join(_parent, "dataset", "gesture_samples")
_sample_matcher = None
RIGHT_HAND_GESTURES = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "HELLO", "STOP", "SPACE",
}
LEFT_HAND_GESTURES = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "YES", "NO"}
TRANSITION_ANIMATIONS = {
    "idle",
    "right_hand_raise",
    "lower_right",
    "raise_left",
    "lower_left",
    "space",
}


def _get_stable(client_id):
    with _states_lock:
        if client_id not in _stable_states:
            _stable_states[client_id] = {
                "right": StableGesture(confirm=CONFIRM_FRAMES, release=RELEASE_FRAMES),
                "left":  StableGesture(confirm=CONFIRM_FRAMES, release=RELEASE_FRAMES),
            }
        return _stable_states[client_id]


def _sanitize_label(value):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return cleaned.strip("_")[:50]


def _prepare_landmarks(raw_lms):
    return np.array(
        [[1.0 - lm["x"], lm["y"], lm["z"]] for lm in raw_lms],
        dtype=np.float32
    )


def _infer_user_side(label, lms):
    mirrored_label_side = {"left": "right", "right": "left"}.get(label, "")
    wrist_x = float(lms[0, 0])
    position_side = "right" if wrist_x >= 0.5 else "left"
    return mirrored_label_side, position_side


def _summarize_sample_files():
    summary = {}
    if not os.path.isdir(_samples_dir):
        return summary

    for name in os.listdir(_samples_dir):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(_samples_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
        except OSError:
            count = 0
        summary[os.path.splitext(name)[0]] = count

    return dict(sorted(summary.items()))


def _format_gesture_label(key):
    text = str(key).strip().replace("_", " ")
    return " ".join(part.capitalize() for part in text.split())


def _discover_common_gesture_animations():
    animations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "animations")
    discovered = {}

    if not os.path.isdir(animations_dir):
        return []

    for name in os.listdir(animations_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in {".fbx", ".glb"}:
            continue

        normalized = base.strip().lower()
        if not normalized:
            continue
        if normalized in TRANSITION_ANIMATIONS:
            continue
        if normalized in {str(i) for i in range(1, 11)}:
            continue
        if len(normalized) == 1 and normalized.isalpha():
            continue

        discovered.setdefault(normalized, _format_gesture_label(base))

    return [
        {"label": label, "key": key}
        for key, label in sorted(discovered.items(), key=lambda item: item[1].lower())
    ]


def _landmark_embedding(lms):
    wrist_xy = lms[0, :2]
    scale = float(np.linalg.norm(lms[9, :2] - lms[0, :2]) + 1e-6)
    xy = (lms[:, :2] - wrist_xy) / scale
    z = (lms[:, 2:3] / scale) * 0.3
    return np.concatenate([xy.reshape(-1), z.reshape(-1)]).astype(np.float32)


def _load_sample_matcher():
    global _sample_matcher
    if _sample_matcher is not None:
        return _sample_matcher

    matcher = {}
    if not os.path.isdir(_samples_dir):
        _sample_matcher = matcher
        return matcher

    for name in os.listdir(_samples_dir):
        if not name.endswith(".jsonl"):
            continue
        label = os.path.splitext(name)[0]
        if label not in RIGHT_HAND_GESTURES and label not in LEFT_HAND_GESTURES:
            continue

        embeddings = []
        path = os.path.join(_samples_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    lms = np.array(row.get("mirrored_landmarks", []), dtype=np.float32)
                    if lms.shape != (21, 3):
                        continue
                    embeddings.append(_landmark_embedding(lms))
        except (OSError, json.JSONDecodeError, ValueError):
            continue

        if not embeddings:
            continue

        pairwise = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                pairwise.append(float(np.linalg.norm(embeddings[i] - embeddings[j])))

        threshold = (max(pairwise) + 0.15) if pairwise else 0.45
        matcher[label] = {
            "embeddings": embeddings,
            "threshold": threshold,
        }

    _sample_matcher = matcher
    return matcher


def _reset_sample_matcher():
    global _sample_matcher
    _sample_matcher = None


def _match_recorded_gesture(lms, allowed_labels):
    matcher = _load_sample_matcher()
    if not matcher:
        return None

    emb = _landmark_embedding(lms)
    nearest = []
    for label, info in matcher.items():
        if label not in allowed_labels:
            continue
        for sample_emb in info["embeddings"]:
            dist_value = float(np.linalg.norm(emb - sample_emb))
            nearest.append((dist_value, label))

    if not nearest:
        return None

    nearest.sort(key=lambda item: item[0])
    top = nearest[:3]

    counts = {}
    best_for_label = {}
    for dist_value, label in top:
        counts[label] = counts.get(label, 0) + 1
        best_for_label[label] = min(best_for_label.get(label, dist_value), dist_value)

    label, votes = max(counts.items(), key=lambda item: (item[1], -best_for_label[item[0]]))
    if votes < 2 and len(top) >= 3:
        return None

    threshold = matcher[label]["threshold"]
    if best_for_label[label] > threshold:
        return None

    return label


# ── Static serving ────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/animations/<path:filename>")
def serve_animations(filename):
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "animations"), filename)

@app.route("/libs/<path:filename>")
def serve_libs(filename):
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs"), filename)

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


# ── PRIMARY: Landmark-based classification (maximum accuracy) ─────────────────

@app.route("/process_landmarks", methods=["POST"])
def process_landmarks():
    """
    Receives landmark coordinates directly from the browser's MediaPipe instance.

    FIX: `data is None` instead of `not data`.
    An empty list [] means "no hands detected" — it is a valid payload.
    The old `if not data` treated [] as falsy and returned 400, which
    caused _requestInFlight to get stuck True in the browser, freezing
    the camera feed permanently after the first no-hand frame.

    Expected JSON body:
    [
        {
            "label": "Right",
            "landmarks": [
                {"x": 0.52, "y": 0.73, "z": -0.04},  // landmark 0 (WRIST)
                ...                                     // landmarks 1-20
            ]
        },
        ...  // second hand if present
    ]

    Returns: { "text": "A" }
    """
    data = request.get_json(silent=True)

    # ── FIX 1: Only reject truly missing/malformed JSON, not empty list ───
    if data is None:
        return jsonify({"text": ""}), 400

    right_raw = None
    left_raw  = None

    for hand in data:
        try:
            label   = str(hand.get("label", "")).strip().lower()
            raw_lms = hand.get("landmarks", [])

            if len(raw_lms) != 21:
                continue  # MediaPipe always gives 21 landmarks; skip malformed

            # ── FIX 2: Build numpy array directly ────────────────────────
            # sign_recog.py's classify_*_hand() functions expect shape (21,3).
            # lm_array() was designed for MediaPipe landmark objects; we bypass
            # it here and construct the array ourselves from the JSON payload.
            lms = _prepare_landmarks(raw_lms)

            # Browser handedness can be flipped on selfie cameras, so use the
            # reported label as a hint, then fall back to the opposite model.
            mirrored_label_side, position_side = _infer_user_side(label, lms)
            right_guess = classify_right_hand(lms)
            left_guess  = classify_left_hand(lms)

            if position_side == "right":
                recorded_right_guess = _match_recorded_gesture(lms, RIGHT_HAND_GESTURES)
                right_raw = recorded_right_guess or (right_guess if right_guess in RIGHT_HAND_GESTURES else None)
            elif position_side == "left":
                recorded_left_guess = _match_recorded_gesture(lms, LEFT_HAND_GESTURES)
                left_raw = recorded_left_guess or (left_guess if left_guess in LEFT_HAND_GESTURES else None)

        except (KeyError, TypeError, ValueError):
            continue  # skip malformed hand data, don't crash

    stable = _get_stable(request.remote_addr)
    stable["right"].update(right_raw)
    stable["left"].update(left_raw)

    gesture = stable["right"].get() or stable["left"].get() or ""
    return jsonify({"text": gesture})


@app.route("/api/gesture-samples/summary", methods=["GET"])
def gesture_samples_summary():
    return jsonify({"samples": _summarize_sample_files()})


@app.route("/api/learn/common-gestures", methods=["GET"])
def learn_common_gestures():
    return jsonify({"gestures": _discover_common_gesture_animations()})


@app.route("/api/gesture-samples", methods=["POST"])
def record_gesture_sample():
    payload = request.get_json(silent=True) or {}
    label = _sanitize_label(payload.get("label", ""))
    raw_lms = payload.get("landmarks", [])
    overwrite = bool(payload.get("overwrite"))

    if not label:
        return jsonify({"error": "Label is required"}), 400
    if not isinstance(raw_lms, list) or len(raw_lms) != 21:
        return jsonify({"error": "Exactly 21 landmarks are required"}), 400

    try:
        lms = _prepare_landmarks(raw_lms)
        right_guess = classify_right_hand(lms)
        left_guess = classify_left_hand(lms)
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Malformed landmark payload"}), 400

    os.makedirs(_samples_dir, exist_ok=True)
    sample_path = os.path.join(_samples_dir, f"{label}.jsonl")
    sample_record = {
        "label": label,
        "handedness_hint": payload.get("handedness_hint", ""),
        "captured_at": payload.get("captured_at", ""),
        "notes": str(payload.get("notes", "")).strip(),
        "right_guess": right_guess,
        "left_guess": left_guess,
        "landmarks": raw_lms,
        "mirrored_landmarks": lms.tolist(),
    }

    mode = "w" if overwrite else "a"
    try:
        with open(sample_path, mode, encoding="utf-8") as handle:
            handle.write(json.dumps(sample_record) + "\n")
    except OSError as exc:
        return jsonify({"error": f"Could not save sample: {exc}"}), 500

    _reset_sample_matcher()

    return jsonify({
        "ok": True,
        "label": label,
        "saved_to": sample_path,
        "right_guess": right_guess or "",
        "left_guess": left_guess or "",
        "count": _summarize_sample_files().get(label, 0),
        "overwrite": overwrite,
    })


# ── Braille endpoint ──────────────────────────────────────────────────────────

@app.route("/api/recognize-braille", methods=["POST"])
def recognize_braille():
    import cv2
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        from samvaad_braille import BrailleRecognizer
        file_bytes = file.read()
        np_arr     = np.frombuffer(file_bytes, np.uint8)
        img        = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
        result = BrailleRecognizer().analyze(img)
        if not result.text.strip():
            return jsonify({"error": "No Braille detected in image"}), 200
        return jsonify({
            "text": result.text,
            "raw_text": result.raw_text,
            "confidence": round(float(result.confidence), 3),
            "meta": {
                "dots_count": result.dots_count,
                "cells_count": result.cells_count,
                "unknown_cells": result.unknown_cells,
                "used_auto_space": result.used_auto_space,
            },
        })
    except ImportError:
        return jsonify({"error": "samvaad_braille.py not found"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SAMVAAD Backend is running!")
    print("  Open your browser at:  http://localhost:5000")
    print("  Mode: Landmark-based (maximum accuracy)")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
