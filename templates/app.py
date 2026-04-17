"""SAMVAAD Flask backend."""

import json
import os
import re
import sys
import threading

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

TEMPLATES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEMPLATES_DIR)
ANIMATIONS_DIR = os.path.join(TEMPLATES_DIR, "animations")
LIBS_DIR = os.path.join(TEMPLATES_DIR, "libs")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sign_recog import StableGesture, classify_left_hand, classify_right_hand

CONFIRM_FRAMES = 3
RELEASE_FRAMES = 3

app = Flask(__name__, static_folder=TEMPLATES_DIR, template_folder=TEMPLATES_DIR)
CORS(app)

_stable_states = {}
_states_lock = threading.Lock()
_samples_dir = os.path.join(DATASET_DIR, "gesture_samples")
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
                "left": StableGesture(confirm=CONFIRM_FRAMES, release=RELEASE_FRAMES),
            }
        return _stable_states[client_id]


def _sanitize_label(value):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return cleaned.strip("_")[:50]


def _prepare_landmarks(raw_lms):
    return np.array(
        [[1.0 - lm["x"], lm["y"], lm["z"]] for lm in raw_lms],
        dtype=np.float32,
    )


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
    discovered = {}

    if not os.path.isdir(ANIMATIONS_DIR):
        return []

    for name in os.listdir(ANIMATIONS_DIR):
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


@app.route("/")
def home():
    return send_from_directory(TEMPLATES_DIR, "index.html")


@app.route("/animations/<path:filename>")
def serve_animations(filename):
    return send_from_directory(ANIMATIONS_DIR, filename)


@app.route("/libs/<path:filename>")
def serve_libs(filename):
    return send_from_directory(LIBS_DIR, filename)


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(TEMPLATES_DIR, filename)


@app.route("/process_landmarks", methods=["POST"])
def process_landmarks():
    """
    Receives browser-side MediaPipe landmark coordinates directly.
    An empty list [] means "no hands detected" and is a valid payload.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"text": ""}), 400

    right_raw = None
    left_raw = None

    for hand in data:
        try:
            raw_lms = hand.get("landmarks", [])
            if len(raw_lms) != 21:
                continue

            lms = _prepare_landmarks(raw_lms)
            right_guess = classify_right_hand(lms)
            left_guess = classify_left_hand(lms)

            position_side = "right" if float(lms[0, 0]) >= 0.5 else "left"
            if position_side == "right":
                recorded_right_guess = _match_recorded_gesture(lms, RIGHT_HAND_GESTURES)
                right_raw = recorded_right_guess or (
                    right_guess if right_guess in RIGHT_HAND_GESTURES else None
                )
            else:
                recorded_left_guess = _match_recorded_gesture(lms, LEFT_HAND_GESTURES)
                left_raw = recorded_left_guess or (
                    left_guess if left_guess in LEFT_HAND_GESTURES else None
                )
        except (KeyError, TypeError, ValueError):
            continue

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
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

