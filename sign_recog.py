import cv2  
import mediapipe as mp
import math
from collections import deque, Counter

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_buffer = deque(maxlen=6)
stop_sub_buffer = deque(maxlen=10)
o_buffer = deque(maxlen=2)
left_buffer = deque(maxlen=6)

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def palm_facing_camera(lm):
    return lm[0].z < lm[9].z and abs(lm[5].z - lm[17].z) < 0.05

def fist_closed(lm):
    palm = lm[0]
    return all(distance(lm[i], palm) < 0.15 for i in [8,12,16,20])

def fingers_up(lm, hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if hand == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)

    for i in range(1, 5):
        fingers.append(1 if lm[tips[i]].y < lm[tips[i]-2].y else 0)

    return fingers

def is_H(lm, hand):
    index_ext_x = abs(lm[8].x - lm[5].x)
    index_ext_y = abs(lm[8].y - lm[5].y)
    index_horizontal = index_ext_x > index_ext_y and index_ext_x > 0.06

    middle_ext_x = abs(lm[12].x - lm[9].x)
    middle_ext_y = abs(lm[12].y - lm[9].y)
    middle_horizontal = middle_ext_x > middle_ext_y and middle_ext_x > 0.06

    tips_parallel = abs(lm[8].y - lm[12].y) < 0.055
    tips_together = distance(lm[8], lm[12]) < 0.08
    ring_curled = lm[16].y > lm[14].y
    pinky_curled = lm[20].y > lm[18].y
    thumb_tucked = distance(lm[4], lm[5]) < 0.12
    back_of_hand = not palm_facing_camera(lm)

    return (
        back_of_hand and
        index_horizontal and
        middle_horizontal and
        tips_parallel and
        tips_together and
        ring_curled and
        pinky_curled and
        thumb_tucked
    )

def is_X(lm, hand):
    index_pip_high   = (lm[5].y - lm[6].y) > 0.04
    index_tip_hooked = lm[8].y > lm[6].y
    middle_curled    = lm[12].y > lm[10].y
    ring_curled      = lm[16].y > lm[14].y
    pinky_curled     = lm[20].y > lm[18].y
    thumb_close      = distance(lm[4], lm[9]) < 0.12
    return (index_pip_high and index_tip_hooked and middle_curled and
            ring_curled and pinky_curled and thumb_close)

def is_Y(lm, hand):
    index_curled      = lm[8].y  > lm[6].y
    middle_curled     = lm[12].y > lm[10].y
    ring_curled       = lm[16].y > lm[14].y
    pinky_up          = lm[20].y < lm[18].y
    pinky_pointing_up = lm[20].y < lm[19].y
    thumb_out = lm[4].x < lm[3].x if hand == "Right" else lm[4].x > lm[3].x
    thumb_pinky_spread = distance(lm[4], lm[20]) > 0.20
    return (index_curled and middle_curled and ring_curled and
            pinky_up and pinky_pointing_up and thumb_out and thumb_pinky_spread)

def is_Z(lm, hand):
    index_up           = lm[8].y  < lm[6].y
    middle_curled      = lm[12].y > lm[10].y
    ring_curled        = lm[16].y > lm[14].y
    pinky_up           = lm[20].y < lm[18].y
    thumb_out = lm[4].x < lm[3].x if hand == "Right" else lm[4].x > lm[3].x
    index_pinky_spread = distance(lm[8], lm[20]) > 0.15
    thumb_spread       = distance(lm[4], lm[5]) > 0.09
    return (index_up and middle_curled and ring_curled and
            pinky_up and thumb_out and index_pinky_spread and thumb_spread)

def is_SPACE(lm, hand):
    index_up           = lm[8].y  < lm[6].y
    middle_curled      = lm[12].y > lm[10].y
    ring_curled        = lm[16].y > lm[14].y
    pinky_up           = lm[20].y < lm[18].y
    index_pinky_spread = distance(lm[8], lm[20]) > 0.15
    thumb_tucked       = distance(lm[4], lm[5]) < 0.09
    return (index_up and middle_curled and ring_curled and
            pinky_up and thumb_tucked and index_pinky_spread)

def is_Q(lm, hand):
    index_points_down = (lm[8].y - lm[5].y) > 0.07
    thumb_points_down = lm[4].y > lm[3].y
    pinch_close       = distance(lm[4], lm[8]) < 0.07
    wrist_above_tip   = lm[0].y < lm[8].y
    middle_curled     = lm[12].y < lm[10].y
    ring_curled       = lm[16].y < lm[14].y
    pinky_curled      = lm[20].y < lm[18].y
    back_of_hand      = not palm_facing_camera(lm)
    return (index_points_down and thumb_points_down and pinch_close
            and wrist_above_tip and middle_curled and ring_curled
            and pinky_curled and back_of_hand)

def is_T(lm, hand):
    index_curled  = lm[8].y  > lm[6].y
    middle_curled = lm[12].y > lm[10].y
    ring_curled   = lm[16].y > lm[14].y
    pinky_curled  = lm[20].y > lm[18].y
    if not (index_curled and middle_curled and ring_curled and pinky_curled):
        return False
    thumb_on_idx_pip = distance(lm[4], lm[6]) < 0.035
    return thumb_on_idx_pip

def is_STOP_PALM(lm, hand):
    """STOP: Palm forward with fingers up and together (just close HELLO fingers, no bending)"""
    # All 4 fingers must be up
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y
    
    if not (index_up and middle_up and ring_up and pinky_up):
        return False
    
    # Fingers close together (the key difference from HELLO)
    im_gap = distance(lm[8], lm[12])
    mr_gap = distance(lm[12], lm[16])
    rp_gap = distance(lm[16], lm[20])
    fingers_together = im_gap < 0.050 and mr_gap < 0.050 and rp_gap < 0.085
    
    # Palm facing camera
    palm_forward = palm_facing_camera(lm)
    
    # NOT checking finger straightness - just together vs apart
    return fingers_together and palm_forward

def classify(lm, f, hand):
    thumb, index, middle, ring, pinky = f

    # ---------- O ----------
    if (
        distance(lm[4], lm[8]) < 0.045 and
        distance(lm[8], lm[6]) < distance(lm[6], lm[5]) and
        all(distance(lm[i], lm[0]) > 0.12 for i in [12,16,20]) and
        not fist_closed(lm) and
        not palm_facing_camera(lm)
    ):
        return "O"

    # ---------- P ----------
    if distance(lm[4], lm[8]) < 0.05 and middle and ring and pinky:
        mr = abs(lm[12].x - lm[16].x)
        rp = abs(lm[16].x - lm[20].x)
        if mr < 0.03 and rp < 0.03:
            return "P"

    # ---------- F ----------
    if distance(lm[4], lm[8]) < 0.05 and middle and ring and pinky:
        return "F"

    # ---------- I ----------
    if f == [0,0,0,0,1]:
        return "I"

    # ---------- B (check before fist gestures) ----------
    if f == [0,1,1,1,1] and not palm_facing_camera(lm):
        return "B"

    # ---------- T (check before STOP fist) ----------
    if is_T(lm, hand):
        return "T"

    # ---------- STOP ----------
    if f == [0,0,0,0,0] and fist_closed(lm):
        return "STOP"

    # ---------- E ----------
    if f == [0,0,0,0,0] and not fist_closed(lm) and distance(lm[4], lm[5]) < 0.08:
        return "E"

    # ---------- STOP PALM (check BEFORE HELLO) ----------
    if is_STOP_PALM(lm, hand):
        return "STOP_PALM"

    # ---------- HELLO ----------
    if f == [1,1,1,1,1]:
        return "HELLO"

    # ---------- X ----------
    if is_X(lm, hand):
        return "X"

    # ---------- A ----------
    if f == [1,0,0,0,0]:
        return "A"

    # ---------- H ----------
    if is_H(lm, hand):
        return "H"

    # ---------- Q ----------
    if is_Q(lm, hand):
        return "Q"

    # ---------- Y ----------
    if is_Y(lm, hand):
        return "Y"

    # ---------- Z ----------
    if is_Z(lm, hand):
        return "Z"

    # ---------- SPACE ----------
    if is_SPACE(lm, hand):
        return "SPACE"

    # ---------- G ----------
    if f == [1,1,0,0,0]:
        if abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y):
            return "G"

    # ---------- K ----------
    if index and middle and not ring and not pinky:
        thumb_tip = lm[4]
        idx_mcp = lm[5]
        mid_mcp = lm[9]
        if min(idx_mcp.x, mid_mcp.x) < thumb_tip.x < max(idx_mcp.x, mid_mcp.x):
            if thumb_tip.y < idx_mcp.y:
                return "K"

    # ---------- U ----------
    if index and middle and not ring and not pinky:
        if distance(lm[8], lm[12]) < 0.04:
            return "U"

    # ---------- V ----------
    if index and middle and not ring and not pinky:
        if distance(lm[8], lm[12]) > 0.055:
            return "V"

    # ---------- W ----------
    if f == [0,1,1,1,0]:
        index_middle_dist = distance(lm[8], lm[12])
        middle_ring_dist = distance(lm[12], lm[16])
        if index_middle_dist > 0.045 and middle_ring_dist > 0.045:
            return "W"

    # ---------- R ----------
    if f == [0,1,1,1,0]:
        index_middle_dist = distance(lm[8], lm[12])
        middle_ring_dist = distance(lm[12], lm[16])
        if index_middle_dist <= 0.045 and middle_ring_dist <= 0.045:
            return "R"

    # ---------- J / L ----------
    if f == [1,1,0,0,0]:
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        if abs(thumb_tip.x - thumb_ip.x) > abs(thumb_tip.y - thumb_ip.y):
            return "L"
        else:
            return "J"

    # ---------- D ----------
    if f == [0,1,0,0,0]:
        return "D"

    # ---------- C ----------
    if f[0] and f[1]:
        return "C"

    return ""

def classify_left(lm, f, hand):
    thumb, index, middle, ring, pinky = f

    index_curled  = lm[8].y  > lm[6].y
    middle_curled = lm[12].y > lm[10].y
    ring_curled   = lm[16].y > lm[14].y
    pinky_curled  = lm[20].y > lm[18].y
    all_curled    = index_curled and middle_curled and ring_curled and pinky_curled
    fingers_closed = all(distance(lm[i], lm[0]) < 0.15 for i in [8, 12, 16, 20])

    # 10
    if all_curled:
        thumb_x_dist = abs(lm[4].x - lm[0].x)
        thumb_y_dist = abs(lm[4].y - lm[0].y)
        thumb_pointing_right = (thumb_x_dist > thumb_y_dist and thumb_x_dist > 0.10)
        thumb_extended = distance(lm[4], lm[0]) > 0.10
        if thumb_pointing_right and thumb_extended:
            return "10"

    # YES
    thumb_up = lm[4].y < lm[3].y and lm[4].y < lm[2].y
    thumb_not_right = abs(lm[4].y - lm[0].y) > abs(lm[4].x - lm[0].x) * 0.5
    if all_curled and thumb_up and thumb_not_right:
        return "YES"

    # NO
    thumb_pointing_down = lm[4].y > lm[3].y and lm[4].y > lm[9].y
    if fingers_closed and thumb_pointing_down:
        return "NO"

    # 0
    if all(distance(lm[i], lm[4]) < 0.07 for i in [8, 12, 16, 20]):
        return "0"

    # 1
    if f == [0, 1, 0, 0, 0]:
        return "1"

    # 2
    if index and middle and not ring and not pinky:
        thumb_ext_2 = distance(lm[4], lm[9]) > 0.10
        if distance(lm[8], lm[12]) > 0.04 and not thumb_ext_2:
            return "2"

    # 3
    index_up_geo  = lm[8].y < lm[6].y
    middle_up_geo = lm[12].y < lm[10].y
    ring_down_geo = lm[16].y > lm[14].y
    pinky_down_geo= lm[20].y > lm[18].y
    thumb_extended = distance(lm[4], lm[9]) > 0.10
    if (index_up_geo and middle_up_geo and ring_down_geo and
            pinky_down_geo and thumb_extended):
        return "3"

    # 4
    if not thumb and index and middle and ring and pinky:
        return "4"

    # 5
    if f == [1, 1, 1, 1, 1]:
        return "5"

    # 6
    if index and middle and ring and not pinky:
        if distance(lm[4], lm[20]) < 0.06:
            return "6"

    # 7
    if index and middle and not ring and pinky:
        if distance(lm[4], lm[16]) < 0.06:
            return "7"

    # 8
    if index and not middle and ring and pinky:
        if distance(lm[4], lm[12]) < 0.06:
            return "8"

    # 9
    if not index and middle and ring and pinky:
        if distance(lm[4], lm[8]) < 0.06:
            return "9"

    return ""

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        raw_gesture = ""
        final_gesture = ""
        left_gesture = ""
        right_active = False
        left_active = False

        if res.multi_hand_landmarks:
            for i, hand_lm in enumerate(res.multi_hand_landmarks):
                lm = hand_lm.landmark
                hand = res.multi_handedness[i].classification[0].label

                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                f = fingers_up(lm, hand)

                if hand == "Right":
                    right_active = True
                    raw_gesture = classify(lm, f, hand)

                    if raw_gesture == "O":
                        o_buffer.append("O")
                        if len(o_buffer) == 2:
                            final_gesture = "O"
                    else:
                        o_buffer.clear()

                    # T detected directly, clear sub-buffer
                    if raw_gesture == "T":
                        stop_sub_buffer.clear()
                        final_gesture = raw_gesture

                    # STOP_PALM detected directly, display as STOP
                    elif raw_gesture == "STOP_PALM":
                        stop_sub_buffer.clear()
                        final_gesture = "STOP"

                    elif raw_gesture == "STOP":
                        thumb_y = lm[4].y
                        idx_pip = lm[6].y
                        mid_pip = lm[10].y
                        ring_pip = lm[14].y

                        if thumb_y > idx_pip and thumb_y > mid_pip:
                            if abs(thumb_y - ring_pip) < 0.025:
                                stop_sub_buffer.append("S")
                            else:
                                stop_sub_buffer.append("M")
                        else:
                            stop_sub_buffer.append("N")

                        weights = Counter(stop_sub_buffer)
                        if "S" in weights:
                            weights["S"] += 2
                        final_gesture = weights.most_common(1)[0][0]

                    elif not final_gesture:
                        stop_sub_buffer.clear()
                        final_gesture = raw_gesture

                else:
                    left_active = True
                    left_gesture = classify_left(lm, f, hand)
                    left_buffer.append(left_gesture)

        if not right_active:
            gesture_buffer.clear()
            stop_sub_buffer.clear()
            final_gesture = ""
        if not left_active:
            left_buffer.clear()

        if final_gesture:
            gesture_buffer.append(final_gesture)

        right_out = Counter(gesture_buffer).most_common(1)[0][0] if gesture_buffer else ""
        left_out = Counter(left_buffer).most_common(1)[0][0] if left_buffer else ""

        if right_active and right_out:
            cv2.putText(frame, f"Letter: {right_out}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif left_active and left_out:
            cv2.putText(frame, f"Number: {left_out}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,128,0), 2)

        cv2.imshow("SAMVAAD ASL", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()