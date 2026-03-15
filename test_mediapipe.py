import cv2
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

recording = False
frames_data = []


def save_landmarks(frames, filename="samvaad_motion.csv"):

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "type", "id", "x", "y", "z"])

        for frame_idx, frame in enumerate(frames):

            if frame["pose"]:
                for i, lm in enumerate(frame["pose"].landmark):
                    writer.writerow([frame_idx, "pose", i, lm.x, lm.y, lm.z])

            if frame["hands"]:
                for hand_idx, hand in enumerate(frame["hands"]):
                    for i, lm in enumerate(hand.landmark):
                        writer.writerow([frame_idx, f"hand_{hand_idx}", i, lm.x, lm.y, lm.z])

    print(f"✅ Motion saved → {filename}")


def main():
    global recording, frames_data

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose() as pose, mp_hands.Hands(max_num_hands=2) as hands:

        print("\nSPACE → Start/Stop Recording | Q → Quit\n")

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(rgb)
            hand_results = hands.process(rgb)

            if recording:
                frames_data.append({
                    "pose": pose_results.pose_landmarks,
                    "hands": hand_results.multi_hand_landmarks
                })

            cv2.imshow("SAMVAAD Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                recording = not recording

                if not recording and frames_data:
                    print(f"✅ Recorded {len(frames_data)} frames")
                    save_landmarks(frames_data)
                    frames_data = []

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()