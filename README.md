# 🎭 EmotiVision-AI  
### Real-Time Face Emotion Recognition with Persona Overlay

EmotiVision-AI is a real-time computer vision application that detects human facial emotions using a deep learning model and augments the face with emotion-based visual personas such as emojis and colored aura effects.

This project combines **MediaPipe**, **OpenCV**, and a **CNN-based emotion recognition model** to deliver a smooth, stable, and visually engaging AI experience.

---

## 🚀 Features

- 🎥 Real-time face detection using MediaPipe  
- 🧠 CNN-based emotion recognition (FER-2013 trained model)  
- 🎭 Emotion-stable prediction using temporal smoothing  
- 😄 Emoji overlay based on detected emotion  
- 🌈 Colored aura / glow effects around face  
- ⚡ Optimized for real-time performance  
- 🧩 Clean modular pipeline (CV → AI → AR overlay)

---

## 🧠 Emotions Supported

- Angry  
- Sad  
- Happy  
- Surprise  
- Neutral  

*(Emotion changes are stabilized to avoid flickering.)*

---

## 🛠️ Tech Stack

- **Python 3.10**
- **OpenCV**
- **MediaPipe**
- **TensorFlow / Keras**
- **NumPy**

---

## 📁 Project Structure

EmotiVision-AI/
│
├── emotion_detection_overlay.py
├── emotion_model.h5
│
├── assets/
│ ├── happy.png
│ ├── sad.png
│ ├── angry.png
│ ├── surprise.png
│ └── neutral.png
│
└── README.md


---

## ▶️ How to Run

### 1️⃣ Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
2️⃣ Install dependencies
pip install opencv-python mediapipe tensorflow numpy
3️⃣ Run the application
python emotion_detection_overlay.py
Press Q to exit.

🧠 How It Works (Pipeline)
Camera
 → Face Detection (MediaPipe)
 → Face Cropping (ROI)
 → Grayscale Conversion
 → Resize & Normalize
 → CNN Emotion Prediction
 → Emotion Stabilization
 → Emoji + Aura Overlay
