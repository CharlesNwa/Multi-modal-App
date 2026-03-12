# 🖐️ Real-Time Hand Gesture Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![License](https://img.shields.io/badge/License-Educational-yellow)

> A computer vision system that recognizes hand gestures in real time
> using MediaPipe for hand tracking and a Bidirectional LSTM for
> gesture classification — no keyboard or touchscreen required.

---

## 👤 Author
**Charles Nwa**
Nigeria 🇳🇬
Built with Claude Code & Visual Studio IDE

---

## 📌 Project Overview

This project captures hand movements through a webcam, extracts
21 hand landmarks using MediaPipe, and classifies gesture sequences
using a trained PyTorch LSTM model. The end goal is a fully
functional real-time gesture recognition application.

---

## 🎯 Real World Applications
- ♿ Accessibility tools for people with disabilities
- 🤟 Sign language recognition
- 🖥️ Hands-free computer control
- 🎮 Gaming and interactive controls
- 📊 Touchless presentation control

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming language |
| MediaPipe | Real-time hand landmark detection |
| PyTorch | LSTM model training and inference |
| OpenCV | Webcam access and video processing |
| NumPy | Data storage and manipulation |
| Scikit-learn | Data splitting and evaluation |
| Streamlit | Web application interface |

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQUENCE_LENGTH` | 30 | Number of frames per gesture sample |
| `VIDEOS_PER_GESTURE` | 20 | Training samples per gesture class |
| `CAMERA_INDEX` | 1 | Webcam device index |
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum confidence to display prediction |

---

## 🏗️ Technical Architecture

### Hand Landmark Extraction
- **MediaPipe Hands** detects up to 2 hands per frame
- **21 landmarks × 2 hands = 42 points** → 126 features (x, y, z each)
- Normalised to `[0, 1]` relative to frame dimensions

### LSTM Classifier
- **Input:** 30 frames × 126 features
- **Model:** 3-layer Bidirectional LSTM, 128 hidden dims, dropout 0.3
- **Head:** Fully-connected → 4 gesture classes
- **Loss:** CrossEntropyLoss | **Optimiser:** Adam (lr = 0.001)

---

## 🎯 Gesture Vocabulary

| Gesture | Display Output |
|---------|---------------|
| `hello_im_charles` | "Hello, I'm Charles" |
| `gesture_recognition_system` | "Real-Time Hand Gesture Recognition System" |
| `from_nigeria` | "I am from Nigeria" |
| `neutral` | *(idle)* |

---

## 📁 Project Structure
```
hand-gesture-recognition/
│
├── data/
│   ├── raw/                       # Recorded MP4 clips per gesture
│   │   ├── hello_im_charles/
│   │   ├── gesture_recognition_system/
│   │   ├── from_nigeria/
│   │   └── neutral/
│   └── processed/                 # Extracted landmark sequences (.npy)
│
├── models/
│   └── checkpoints/               # Model checkpoints during training
│
├── src/
│   ├── mediapipe_extractor.py     # Hand landmark extraction
│   └── utils.py                   # Helper functions
│
├── collect_gestures.py            # Interactive data collection tool
├── verify_setup.py                # Environment & dependency check
├── test_camera.py                 # Camera diagnostics
├── config.py                      # Project configuration
├── requirements.txt               # Required libraries
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/CharlesNwa/Multi-modal-App.git
cd Multi-modal-App/hand-gesture-recognition
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Verify Setup
```bash
python verify_setup.py
```

### 4. Collect Gesture Data
```bash
python collect_gestures.py
```

### 5. Train Model *(Phase 5)*
```bash
python train_model.py
```

### 6. Run Real-Time Demo *(Phase 6)*
```bash
python demo.py
```

---

## 📷 Data Collection Tips
- Use **good lighting** — MediaPipe is sensitive to shadows
- Keep your **full hand in frame** for the 3-second duration
- **Vary angle and distance** slightly between recordings for robustness
- Aim for at least **20 samples per gesture** before training

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| No hand detected | Improve lighting and move hand closer to camera |
| Camera won't open | Change `CAMERA_INDEX` in `config.py` to 0 or 1 |
| Low FPS in preview | Close background apps and reduce resolution |
| Low accuracy | Collect more varied samples, aim for 30+ per gesture |

---

## 🗺️ Roadmap

- [x] Phase 1 — Environment setup & MediaPipe integration
- [x] Phase 2 — Data collection pipeline
- [x] Phase 3 — Landmark extraction & storage
- [ ] Phase 4 — PyTorch Dataset & DataLoader
- [ ] Phase 5 — LSTM training pipeline
- [ ] Phase 6 — Real-time inference & demo UI
- [ ] Phase 7 — Streamlit web application
- [ ] Phase 8 — Final testing & deployment

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | In Progress |
| Validation Accuracy | In Progress |
| Inference Speed | In Progress |

---

## 📜 License
Educational use only.
Built and owned by **Charles Nwachukwu** — Nigeria.

---

## 🤝 Acknowledgements
- [MediaPipe by Google](https://mediapipe.dev)
- [PyTorch](https://pytorch.org)
- Inspired by Farah Gherir's gesture recognition project on LinkedIn
