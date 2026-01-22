import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# LOAD EMOTION MODEL
# =========================
model = load_model("emotion_model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# LOAD EMOJI PNGs
# =========================
emoji_map = {
    'Happy': cv2.imread('assets/happy.png', cv2.IMREAD_UNCHANGED),
    'Sad': cv2.imread('assets/sad.png', cv2.IMREAD_UNCHANGED),
    'Angry': cv2.imread('assets/angry.png', cv2.IMREAD_UNCHANGED),
    'Surprise': cv2.imread('assets/surprise.png', cv2.IMREAD_UNCHANGED),
    'Neutral': cv2.imread('assets/neutral.png', cv2.IMREAD_UNCHANGED)
}

# =========================
# EMOTION → COLOR (AURA)
# =========================
emotion_colors = {
    'Happy': (0, 255, 255),     # Yellow
    'Sad': (255, 0, 0),         # Blue
    'Angry': (0, 0, 255),       # Red
    'Surprise': (255, 255, 0),  # Cyan
    'Neutral': (200, 200, 200)  # Gray
}

# =========================
# EMOTION STABILITY VARIABLES 🔥
# =========================
current_emotion = None
emotion_counter = 0
STABILITY_THRESHOLD = 7   # increase for more stability

# =========================
# MEDIAPIPE FACE DETECTION
# =========================
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# =========================
# OVERLAY FUNCTION
# =========================
def overlay_transparent(bg, overlay, x, y, size):
    overlay = cv2.resize(overlay, size)

    h, w, _ = overlay.shape
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg

    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0

    bg_part = bg[y:y+h, x:x+w]
    bg[y:y+h, x:x+w] = (1 - mask) * bg_part + mask * overlay_img
    return bg

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        # ✅ ONLY MOST CONFIDENT FACE
        detection = max(results.detections, key=lambda d: d.score[0])

        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x, y = max(0, x), max(0, y)
        face_roi = frame[y:y+bh, x:x+bw]

        if face_roi.size != 0:

            # ========= PREPROCESS =========
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            normalized = resized / 255.0
            input_data = normalized.reshape(1, 64, 64, 1)

            # ========= PREDICT =========
            preds = model.predict(input_data, verbose=0)
            idx = np.argmax(preds)
            predicted_emotion = emotion_labels[idx]
            confidence = preds[0][idx]

            # ========= EMOTION SMOOTHING 🔥 =========
            if predicted_emotion == current_emotion:
                emotion_counter += 1
            else:
                current_emotion = predicted_emotion
                emotion_counter = 1

            emotion = None
            if emotion_counter >= STABILITY_THRESHOLD:
                emotion = current_emotion

            # ========= DRAW ONLY IF STABLE =========
            if emotion is not None:

                # ---- AURA ----
                overlay = frame.copy()
                color = emotion_colors.get(emotion, (255, 255, 255))

                center = (x + bw // 2, y + bh // 2)
                axes = (bw // 2 + 20, bh // 2 + 30)

                cv2.ellipse(
                    overlay,
                    center,
                    axes,
                    0, 0, 360,
                    color,
                    15
                )

                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                # ---- EMOJI ----
                emoji = emoji_map.get(emotion)
                if emoji is not None:
                    frame = overlay_transparent(
                        frame,
                        emoji,
                        x,
                        y - bh // 2,
                        (bw, bw)
                    )

                # ---- TEXT ----
                cv2.putText(
                    frame,
                    f"{emotion} ({int(confidence*100)}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        # Face box (always visible)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    cv2.imshow("AI Face Emotion + Persona Overlay", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
