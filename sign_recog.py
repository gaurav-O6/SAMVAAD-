import time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ─────────────────────────────────────────────
CONFIRM_FRAMES     = 6   # frames to hold before locking (higher = stabler)
RELEASE_FRAMES     = 5   # frames of absence before clearing
MIN_DETECTION_CONF = 0.75
MIN_TRACKING_CONF  = 0.75
FONT               = cv2.FONT_HERSHEY_SIMPLEX
AMBIGUITY_MARGIN   = 0.04
HISTORY_SIZE       = 8
MAJORITY_RATIO     = 0.75

# ─────────────────────────────────────────────
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF,
)

# ─────────────────────────────────────────────
WRIST=0; THUMB_CMC=1; THUMB_MCP=2; THUMB_IP=3; THUMB_TIP=4
INDEX_MCP=5;  INDEX_PIP=6;  INDEX_DIP=7;  INDEX_TIP=8
MIDDLE_MCP=9; MIDDLE_PIP=10; MIDDLE_DIP=11; MIDDLE_TIP=12
RING_MCP=13;  RING_PIP=14;  RING_DIP=15;  RING_TIP=16
PINKY_MCP=17; PINKY_PIP=18; PINKY_DIP=19; PINKY_TIP=20

# ═════════════════════════════════════════════
def lm_array(landmarks):
    """Return landmarks as a numpy array."""
    if isinstance(landmarks, np.ndarray):
        return landmarks
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

def hand_scale(lms):
    return float(np.linalg.norm(lms[MIDDLE_MCP] - lms[WRIST]) + 1e-6)

def dist(lms, a, b):
    return float(np.linalg.norm(lms[a, :2] - lms[b, :2]) / hand_scale(lms))

def tip_above_pip(lms, tip, pip):
    return lms[tip, 1] < lms[pip, 1]

def is_horiz(lms, tip, mcp):
    return abs(lms[tip,0]-lms[mcp,0]) > abs(lms[tip,1]-lms[mcp,1])

def near(value, target, margin=AMBIGUITY_MARGIN):
    return abs(value - target) <= margin

def ordered_x_spread(lms, indices):
    xs = [float(lms[idx, 0]) for idx in indices]
    return xs == sorted(xs) or xs == sorted(xs, reverse=True)

def hand_is_large_enough(lms):
    return hand_scale(lms) > 0.12

# ═════════════════════════════════════════════
def finger_states(lms):
    th = lms[THUMB_TIP, 0] < lms[THUMB_CMC, 0]   # right hand CMC-based
    ix = tip_above_pip(lms, INDEX_TIP,  INDEX_PIP)
    mx = tip_above_pip(lms, MIDDLE_TIP, MIDDLE_PIP)
    rx = tip_above_pip(lms, RING_TIP,   RING_PIP)
    px = tip_above_pip(lms, PINKY_TIP,  PINKY_PIP)
    return th, ix, mx, rx, px

def finger_states_left(lms):
    th = lms[THUMB_TIP, 0] > lms[THUMB_CMC, 0]   # left hand mirrored
    ix = tip_above_pip(lms, INDEX_TIP,  INDEX_PIP)
    mx = tip_above_pip(lms, MIDDLE_TIP, MIDDLE_PIP)
    rx = tip_above_pip(lms, RING_TIP,   RING_PIP)
    px = tip_above_pip(lms, PINKY_TIP,  PINKY_PIP)
    return th, ix, mx, rx, px

# ═════════════════════════════════════════════
def classify_right_hand(lms):
    if not hand_is_large_enough(lms):
        return None

    th, ix, mx, rx, px = finger_states(lms)
    def d(a, b): return dist(lms, a, b)
    def h(t, m):  return is_horiz(lms, t, m)
    tvy = lms[THUMB_MCP, 1] - lms[THUMB_TIP, 1]   # +ve = thumb tip above MCP
    th_ix_tip = d(THUMB_TIP, INDEX_TIP)
    th_mid_tip = d(THUMB_TIP, MIDDLE_TIP)
    th_ring_tip = d(THUMB_TIP, RING_TIP)
    th_index_pip = d(THUMB_TIP, INDEX_PIP)
    th_middle_mcp = d(THUMB_TIP, MIDDLE_MCP)
    th_middle_pip = d(THUMB_TIP, MIDDLE_PIP)
    ix_mid_tip = d(INDEX_TIP, MIDDLE_TIP)
    ix_pinky_tip = d(INDEX_TIP, PINKY_TIP)
    thumb_horiz = h(THUMB_TIP, THUMB_MCP)
    index_horiz = h(INDEX_TIP, INDEX_MCP)

    # ── 4 fingers up ──────────────────────────────────────────────────────
    # HELLO: th+all4, spread>0.70 | STOP: th+all4, spread<=0.70 | B: no th
    if ix and mx and rx and px:
        if not ordered_x_spread(lms, [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]):
            return None
        if th:
            spread = d(INDEX_TIP, PINKY_TIP)
            if near(spread, 0.70, 0.08):
                return None
            return "HELLO" if spread > 0.70 else "STOP"
        return "B"

    # ── ix+mx+rx, no px ───────────────────────────────────────────────────
    # W only (th=F)
    if ix and mx and rx and not px and not th:
        return "W"

    # ── not_ix + mx+rx+px + th ────────────────────────────────────────────
    if not ix and mx and rx and px and th:
        if th_middle_mcp > 1.00:
            return "P" if tvy <= -0.14 else "Q"
        if th_ix_tip < 0.18:
            return "F"
        if th_ring_tip < 0.42 and tvy > -0.09:
            return "H"
        if tvy < -0.18 and lms[THUMB_TIP, 1] > lms[WRIST, 1]:
            return "P"
        if thumb_horiz:
            return "Q"
        if tvy < -0.04:
            return "Q"

    # ── not_ix+not_mx + rx+px ─────────────────────────────────────────────
    if not ix and not mx and rx and px:
        if th:
            if th_middle_mcp > 1.00:
                return "P" if tvy <= -0.14 else "Q"
            if th_ring_tip < 0.42 and tvy > -0.09:
                return "H"
            if tvy < -0.18 and lms[THUMB_TIP, 1] > lms[WRIST, 1]:
                return "P"
            if thumb_horiz and tvy < -0.04:
                return "Q"
            if not thumb_horiz and tvy < -0.04:
                return "H"
        else:
            if tvy < -0.10:
                return "P"

    # ── ix+mx only ────────────────────────────────────────────────────────
    # K: thumb near index PIP | R: tips very close | U: close | V: spread
    if ix and mx and not rx and not px:
        s = d(INDEX_TIP, MIDDLE_TIP)
        thumb_to_index_pip = d(THUMB_TIP, INDEX_PIP)
        if near(thumb_to_index_pip, 0.22, 0.03) or near(s, 0.13, 0.03) or near(s, 0.30, 0.04):
            return None
        if thumb_to_index_pip < 0.22:
            return "K"
        if s < 0.13:
            return "R"
        if s < 0.30:
            return "U"
        return "V"

    # ── ix only ───────────────────────────────────────────────────────────
    if ix and not mx and not rx and not px:
        thumb_to_middle = th_mid_tip
        if near(thumb_to_middle, 0.15, 0.03):
            return None
        if thumb_to_middle < 0.18:
            return "D"
        if th and lms[INDEX_TIP, 1] < lms[INDEX_MCP, 1]:
            return "L" if thumb_horiz else "J"

    # ── ix+px ─────────────────────────────────────────────────────────────
    if ix and not mx and not rx and px:
        if th:
            index_middle = ix_mid_tip
            index_pinky = ix_pinky_tip
            if near(index_middle, 0.10, 0.025) or near(index_pinky, 0.80, 0.06):
                return None
            if thumb_horiz and index_horiz and index_pinky < 0.30 and tvy < 0.05:
                return "C"
            if index_middle < 0.10:
                return "C"
            if index_pinky > 0.80:
                return "Z"
            if index_middle >= 0.12 and index_pinky <= 0.72 and th_ix_tip > 0.34 and th_mid_tip > 0.34:
                return "SPACE"
        else:
            return "SPACE"

    # ── not_ix+not_mx+not_rx+px + th ─────────────────────────────────────
    if not ix and not mx and not rx and px and th:
        s = ix_pinky_tip
        th_to_ix = th_ix_tip
        if near(s, 0.20, 0.03) or near(th_to_ix, 0.35, 0.04) or near(th_to_ix, 0.55, 0.04):
            return None
        if s < 0.20:
            return "O"
        if th_to_ix > 0.55:
            return "Y"
        if th_to_ix < 0.35:
            return "I"

    # ── all fingers down ──────────────────────────────────────────────────
    if not ix and not mx and not rx and not px:
        if th:
            if index_horiz and thumb_horiz and th_middle_mcp > 0.75 and tvy < 0.03:
                return "G"
            if (not thumb_horiz) and index_horiz and 0.32 <= th_ix_tip <= 0.58 and 0.50 <= th_middle_mcp <= 0.68 and tvy >= 0.05:
                return "E"
            if th_middle_mcp >= 0.45 and th_index_pip <= 0.28 and tvy >= 0.12:
                return "T"
            if 0.34 <= th_middle_mcp <= 0.42 and 0.42 <= th_index_pip <= 0.50 and tvy >= 0.11:
                return "N"
            if th_middle_mcp < 0.20 and th_index_pip < 0.16:
                return "S"
            if th_middle_mcp < 0.28 and th_middle_pip < 0.30 and th_ix_tip < 0.55:
                return "M"

            # O: fingers bunched AND thumb close
            if ix_mid_tip < 0.10 and th_ix_tip < 0.30:
                return "O"

            # X: thumb sits close to MIDDLE tip but far from INDEX tip.
            # Captured data: d_th_mx_tip=0.2288, d_th_ix_tip=0.5152
            if th_mid_tip < 0.26 and th_ix_tip > 0.45:
                return "X"

            # G: index pointing sideways (extended, not hooked)
            if index_horiz:
                return "G"

            # S vs T: split by palm distance at 0.32
            if th_index_pip < 0.22:
                return "S" if th_middle_mcp < 0.32 else "T"
            # A vs E: split by index_pip distance at 0.55
            if th_index_pip < 0.55:
                return "A"
            return "E"
        else:
            # S (th=F): thumb buried at palm center
            if th_middle_mcp < 0.06:
                return "S"
            # M: thumb close to palm
            if th_middle_mcp < 0.22:
                return "M"
            # N: thumb near middle PIP
            if th_middle_pip < 0.35:
                return "N"
            return "E"

    return None


# ═════════════════════════════════════════════
# LEFT HAND CLASSIFIER
# ═════════════════════════════════════════════

def classify_left_hand(lms):
    if not hand_is_large_enough(lms):
        return None

    th, ix, mx, rx, px = finger_states_left(lms)
    def d(a, b): return dist(lms, a, b)
    vec_y = lms[THUMB_MCP, 1] - lms[THUMB_TIP, 1]
    vec_x = lms[THUMB_TIP, 0] - lms[THUMB_MCP, 0]
    t_vert = abs(vec_y) > abs(vec_x)
    t_horiz = abs(vec_x) > abs(vec_y)

    # ── all fingers down: YES, NO, 10, 0 ─────────────────────────────────
    if not ix and not mx and not rx and not px:
        if near(vec_y, 0.05, 0.02) or near(vec_y, -0.05, 0.02):
            return None
        if t_vert and vec_y >  0.05: return "YES"
        if t_vert and vec_y < -0.05: return "NO"
        if t_horiz:                  return "10"
        if near(d(THUMB_TIP, INDEX_TIP), 0.22, 0.03): return None
        if d(THUMB_TIP, INDEX_TIP) < 0.22: return "0"

    # ── ix+mx+rx, no px: NO(alt), 6, 5 ───────────────────────────────────
    if ix and mx and rx and not px:
        if near(vec_y, -0.05, 0.02) or near(d(THUMB_TIP, PINKY_TIP), 0.15, 0.03):
            return None
        if th and vec_y < -0.05:           return "NO"
        if d(THUMB_TIP, PINKY_TIP) < 0.15: return "6"
        return "5"

    # ── all five up: 5 ────────────────────────────────────────────────────
    if th and ix and mx and rx and px:
        return "5"

    # ── four fingers up, no thumb: 4 ──────────────────────────────────────
    if not th and ix and mx and rx and px:
        return "4"

    # ── ix+mx+th, no rx no px: 3 ──────────────────────────────────────────
    if ix and mx and not rx and not px and th:
        return "3"

    # ── ix+mx, no thumb: 2 ────────────────────────────────────────────────
    if ix and mx and not rx and not px and not th:
        return "2"

    # ── ix+mx+px, no rx: 7 ────────────────────────────────────────────────
    if ix and mx and not rx and px:
        if d(THUMB_TIP, RING_TIP) < 0.15: return "7"

    # ── ix+not_mx+rx+px: 8 ────────────────────────────────────────────────
    if ix and not mx and rx and px:
        if d(THUMB_TIP, MIDDLE_TIP) < 0.10: return "8"

    # ── not_ix+mx+rx+px: 9 ────────────────────────────────────────────────
    if not ix and mx and rx and px:
        if d(THUMB_TIP, INDEX_TIP) < 0.22: return "9"

    # ── ix only: 1 ────────────────────────────────────────────────────────
    if ix and not mx and not rx and not px:
        return "1"

    return None


# ═════════════════════════════════════════════
# STABILITY: hold-to-confirm
# ═════════════════════════════════════════════

class StableGesture:
    def __init__(self, confirm=CONFIRM_FRAMES, release=RELEASE_FRAMES):
        self.confirm       = confirm
        self.release       = release
        self.candidate     = None
        self.cand_count    = 0
        self.locked        = None
        self.release_count = 0
        self.history       = deque(maxlen=HISTORY_SIZE)

    def update(self, raw):
        self.history.append(raw)

        if raw is not None and raw == self.candidate:
            self.cand_count += 1
        else:
            self.candidate  = raw
            self.cand_count = 1

        recent = [g for g in self.history if g is not None]
        majority = None
        majority_count = 0
        if recent:
            counts = {}
            for gesture in recent:
                counts[gesture] = counts.get(gesture, 0) + 1
            majority, majority_count = max(counts.items(), key=lambda item: item[1])

        if (
            self.candidate is not None and
            self.cand_count >= self.confirm and
            majority == self.candidate and
            majority_count >= max(2, int(np.ceil(len(recent) * MAJORITY_RATIO)))
        ):
            self.locked        = self.candidate
            self.release_count = 0
        elif raw is None:
            self.release_count += 1
            if self.release_count >= self.release:
                self.locked = None
        else:
            self.release_count = 0

    def get(self):
        return self.locked


# ═════════════════════════════════════════════
# VISUALISATION
# ═════════════════════════════════════════════

def draw_label(frame, text, pos, color=(0,255,120), scale=1.2, thickness=2):
    cv2.putText(frame, text, pos, FONT, scale, (0,0,0), thickness+2)
    cv2.putText(frame, text, pos, FONT, scale, color, thickness)

def draw_fps(frame, fps):
    draw_label(frame, f"FPS: {fps:.1f}", (10,30),
               color=(200,200,200), scale=0.7, thickness=1)

def draw_gesture_panel(frame, right_gesture, left_gesture):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-80), (w, h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    if right_gesture:
        draw_label(frame, f"Letter: {right_gesture}",
                   (20, h-25), color=(0,230,100), scale=1.4, thickness=2)
    if left_gesture:
        draw_label(frame, f"Number: {left_gesture}",
                   (w//2, h-25), color=(100,200,255), scale=1.4, thickness=2)


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    right_stable = StableGesture()
    left_stable  = StableGesture()
    prev_time    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands_detector.process(rgb)
        rgb.flags.writeable = True

        right_raw = None
        left_raw  = None

        if results.multi_hand_landmarks:
            for hand_lms, hand_info in zip(
                    results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label
                lms   = lm_array(hand_lms)  # works for MediaPipe objects

                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if label == "Right":
                    right_raw = classify_right_hand(lms)
                else:
                    left_raw  = classify_left_hand(lms)

        right_stable.update(right_raw)
        left_stable.update(left_raw)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-9)
        prev_time = curr_time

        draw_fps(frame, fps)
        draw_gesture_panel(frame, right_stable.get(), left_stable.get())
        draw_label(frame, "SAMVAAD - ASL Recognition",
                   (10, 65), color=(255,220,50), scale=0.8, thickness=2)

        cv2.imshow("SAMVAAD", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()


if __name__ == "__main__":
    main()

