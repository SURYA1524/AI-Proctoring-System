import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False
)
def detect_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)

    eye_dev = 0
    head_dev = 0
    face_box = None

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        left_eye = lm[33]
        right_eye = lm[263]
        nose = lm[1]

        eye_dev = abs(left_eye.x - right_eye.x)
        head_dev = abs(nose.x - 0.5)

        h, w, _ = frame.shape
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]

        face_box = (min(xs), min(ys), max(xs), max(ys))

    return eye_dev, head_dev, face_box