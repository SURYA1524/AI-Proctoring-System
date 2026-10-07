import cv2
import time
import os

from baseline import Baseline
from detector import detect_features
from scoring import compute_score
from audio_detector import detect_silence
from cognitive import cognitive_load
from event_buffer import EventBuffer
from report import generate_report
from api import update_status
from ultralytics import YOLO

# ================= ADD HERE =================
# Load YOLO model for object detection
model = YOLO("yolov8n.pt")
def detect_emotion(eye, head):
    if eye < 0.015:
        return "Drowsy"
    elif head > 0.15:
        return "Looking Away"
    elif eye > 0.06:
        return "Surprised"
    else:
        return "Neutral"

# ================= CONFIG =================
FLAG_LIMIT = 10
FRAME_SKIP = 1
AUDIO_INTERVAL = 1.0
ALERT_COOLDOWN = 10
BASELINE_TIME = 5
VIOLATION_TIME = 3.5
YOLO_SKIP = 10   # run YOLO every 10 frames

# Baseline-relative thresholds
EYE_DIFF_THRESHOLD = 0.9
HEAD_DIFF_THRESHOLD = 1.0

# ================= STATE =================

alert_count = 0
frame_count = 0
last_audio_time = 0
last_alert_time = 0

eye_start = None
head_start = None

smooth_eye = 0
smooth_head = 0

alert_text = ""
alert_time = 0

baseline_eye = None
baseline_head = None

baseline = Baseline()
events = EventBuffer()
scores = []

# ================= CAMERA =================

cap = cv2.VideoCapture(0)
cv2.namedWindow("AI Proctoring System", cv2.WINDOW_NORMAL)
cv2.waitKey(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

start_time = time.time()
time.sleep(1)

# ================= MAIN LOOP =================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ================= PHONE DETECTION (OPTIMIZED) =================

    phone_detected = False

    if frame_count % YOLO_SKIP == 0:
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if "phone" in label:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
                cv2.putText(frame, "PHONE DETECTED",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,0,255), 2)
                
    # ===============================================================

    current_time = time.time()

    phone_detected = False
    results = model(frame, verbose=False)[0]

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
        label = model.names[cls]

        if label == "cell phone":
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
            cv2.putText(frame, "PHONE DETECTED",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,0,255), 2)

    frame_count += 1
    current_time = time.time()

    # ---------- FRAME PROCESS ----------

    if frame_count % FRAME_SKIP == 0:
        eye, head, face_box = detect_features(frame)

        # ---------- AUDIO CHECK ----------

        if current_time - last_audio_time > AUDIO_INTERVAL:
            silence = detect_silence()
            last_audio_time = current_time
        else:
            silence = False
    else:
        silence = False

    alert_text = ""

    # ---------- BASELINE COLLECTION ----------

    if current_time - start_time < BASELINE_TIME:
        baseline.update(eye, head)
        score = 0
        emotion = "Learning"
        cog = 0
    else:
        if baseline_eye is None or baseline_head is None:
            base = baseline.get_baseline()
            baseline_eye = base["eye"]
            baseline_head = base["head"]

        # ---------- BASELINE-RELATIVE DETECTION ----------

        eye_diff = abs(eye - baseline_eye)
        head_diff = abs(head - baseline_head)
            
        # Ignore tiny noise movements

        if eye_diff < 0.05:
            eye_diff = 0
        if head_diff < 0.05:
            head_diff = 0

        # ---- HEAD MOVEMENT ----

        if head_diff > HEAD_DIFF_THRESHOLD:
            if head_start is None:
                head_start = current_time
            elif current_time - head_start >= VIOLATION_TIME:
                alert_text = "HEAD MOVEMENT DETECTED"
        else:
            head_start = None

        # ---- EYE MOVEMENT ----

        if eye_diff > EYE_DIFF_THRESHOLD:
            if eye_start is None:
                eye_start = current_time
            elif current_time - eye_start >= VIOLATION_TIME:
                alert_text = "LOOKING AWAY DETECTED"
        else:
            eye_start = None

        # ---- AUDIO ----

        if silence and (current_time - last_alert_time) > ALERT_COOLDOWN:
            alert_text = "AUDIO SILENCE DETECTED"

        base = baseline.get_baseline()
        score = compute_score(eye, head, base)
        emotion = detect_emotion(eye, head)
        cog = cognitive_load(eye, head, silence)

    # ---------- ALERT TRIGGER ----------

    if phone_detected:
        alert_text = "PHONE DETECTED"
    if alert_text and (current_time - last_alert_time) > ALERT_COOLDOWN:
        alert_count += 1
        alert_time = current_time
        last_alert_time = current_time
        events.add(alert_text)
        os.system("afplay /System/Library/Sounds/Ping.aiff &")

    scores.append(score)

    update_status({
        "score": round(score, 2),
        "emotion": emotion,
        "cognitive_load": round(cog, 2),
        "alerts": events.events[-5:]
    })

    # ---------- UI ---------- #
    
    cv2.putText(
        frame,
        f"Eye:{eye:.3f} Head:{head:.3f} Score:{score:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Warnings: {alert_count}/{FLAG_LIMIT}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )
    cv2.putText(
    frame,
    f"Emotion: {emotion}",
    (20, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 0, 0),
    2
)

    if alert_text and (current_time - alert_time) < 2:
        cv2.putText(
            frame,
            alert_text,
            (50, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    if face_box:
        x1, y1, x2, y2 = face_box
        color = (0, 0, 255) if alert_text else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if alert_count == FLAG_LIMIT - 1:
        cv2.putText(
            frame,
            "FINAL WARNING!",
            (150, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            4
        )

    if alert_count >= FLAG_LIMIT:
        cv2.putText(
            frame,
            "EXAM TERMINATED",
            (130, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )
        cv2.imshow("AI Proctoring System", frame)
        cv2.waitKey(3000)
        break

    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP ================= #
cap.release()
cv2.destroyAllWindows()
generate_report(events.events, scores)