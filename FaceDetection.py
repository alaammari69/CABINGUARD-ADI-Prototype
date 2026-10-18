import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"

# Eye landmark indices from FaceMesh (6 points per eye)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)


def eye_aspect_ratio(points, idx, w, h):
    """EAR: small when the eye is closed, larger when open."""
    p = [np.array([points[i].x * w, points[i].y * h]) for i in idx]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / (2.0 * horizontal)


cap = cv2.VideoCapture(0)  # if no image appears, try 1
start = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from camera")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int((time.time() - start) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.face_landmarks:
        points = result.face_landmarks[0]
        for p in points:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)

        ear = (eye_aspect_ratio(points, RIGHT_EYE, w, h)
               + eye_aspect_ratio(points, LEFT_EYE, w, h)) / 2
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "NO FACE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("face test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()