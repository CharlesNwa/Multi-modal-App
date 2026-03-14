# 🖐️ Real-Time Hand Gesture Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![License](https://img.shields.io/badge/License-Educational-yellow)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

> A computer vision system that recognizes hand gestures in real time using
> MediaPipe for hand tracking and geometric landmark rules — no training or
> internet required. Runs directly from your webcam.

---

## 👤 Author
**Charles Nwachukwu**
Nigeria 🇳🇬
Built with Claude Code & Visual Studio Code

---

## 🎬 Demo

![Demo](demo_recording_compressed.mp4)

> Live demo: the system detects hand gestures in real time and displays the
> gesture label on screen with a confidence bar and hand skeleton overlay.

---

## 🎯 Gestures Recognized

| Gesture | Label Shown on Screen |
|---------|----------------------|
| 👍 Thumbs Up | Thumbs Up 👍 |
| 👎 Thumbs Down | Thumbs Down 👎 |
| ✊ Closed Fist | Fist ✊ |
| ✌️ Peace / Victory | Peace ✌️ |
| 🖐️ Open Palm | Stop 🖐️ |
| 👌 OK Sign | OK 👌 |

---

## 📌 How It Works

1. **Webcam captures frames** in real time
2. **MediaPipe Hands** detects 21 hand landmarks per frame (x, y, z)
3. **Geometric rules** classify the gesture from landmark positions — no ML model needed
4. **Gesture label + confidence bar** displayed on screen instantly

### Why geometric rules instead of a trained model?
- Works 100% offline — no dataset download or model file needed
- Instant results — no training pipeline required
- Reliable for clearly defined hand shapes

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| MediaPipe | Real-time hand landmark detection (21 points) |
| OpenCV | Webcam capture, drawing, video recording |
| NumPy | Landmark array operations |
| PyTorch | Custom LSTM model (optional training pipeline) |
| Scikit-learn | Data splitting for training pipeline |

---

## 📁 Project Structure

```
hand-gesture-recognition/
│
├── recognize_mp.py          # ✅ MAIN — offline gesture recognizer (run this)
├── recognize.py             # Recognizer using trained LSTM model
├── collect_gestures.py      # Interactive data collection tool
├── train.py                 # LSTM model training pipeline
├── prepare_hagrid.py        # HaGRID public dataset downloader
├── config.py                # Project configuration
├── run.bat                  # One-click menu launcher (Windows)
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── raw/                 # Recorded MP4 clips per gesture
│   └── processed/           # Extracted landmark sequences (.npy)
│
├── models/
│   └── checkpoints/         # Saved model weights (best_model.pt)
│
├── src/
│   ├── mediapipe_extractor.py
│   └── utils.py
│
├── demo_recording.mp4           # Full demo recording
└── demo_recording_compressed.mp4 # Compressed demo (GitHub preview)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/CharlesNwa/Multi-modal-App.git
cd Multi-modal-App/hand-gesture-recognition
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Run the Gesture Recognizer (No Training Needed)
```bash
python recognize_mp.py
```

Or use the interactive menu:
```bash
.\run.bat    # Windows — double-click or run in terminal
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `R` | Start / Stop recording demo video |
| `Q` or `Esc` | Quit |

---

## 📋 Run Menu Options (`run.bat`)

| Option | Description |
|--------|-------------|
| 1 | Download HaGRID public dataset & extract landmarks |
| 2 | Train custom LSTM model |
| 3 | Run recognizer with trained model |
| 4 | Collect custom gesture data |
| 5 | Run offline MediaPipe recognizer (recommended) |
| 6 | Exit |

---

## ⚙️ Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `1` | Webcam index (change to `0` for laptop camera) |
| `SEQUENCE_LENGTH` | `30` | Frames per gesture sequence |
| `CONFIDENCE_THRESHOLD` | `0.3` | Min confidence to show prediction |
| `HIDDEN_DIM` | `128` | LSTM hidden size |
| `NUM_EPOCHS` | `50` | Training epochs |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera won't open | Change `CAMERA_INDEX` in `recognize_mp.py` from `1` to `0` |
| No hand detected | Improve lighting, keep hand fully in frame |
| Wrong gesture detected | Make the gesture more clearly / slower |
| Low FPS | Close background apps, set `model_complexity=0` |

---

## 🏗️ Optional: Train a Custom LSTM Model

If you want to train your own model on custom gestures:

```bash
# Step 1 — Collect gesture data (15–20 samples per gesture)
python collect_gestures.py

# Step 2 — Train the BiLSTM model
python train.py

# Step 3 — Run with trained model
python recognize.py
```

### Model Architecture
- **Input:** 30 frames × 126 features (42 landmarks × 3 coords)
- **Model:** 3-layer Bidirectional LSTM, 128 hidden dims, dropout 0.3
- **Head:** Linear → 6 gesture classes
- **Loss:** CrossEntropyLoss | **Optimiser:** Adam (lr=0.001)

---

## 🗺️ Roadmap

- [x] Phase 1 — Environment setup & MediaPipe integration
- [x] Phase 2 — Data collection pipeline
- [x] Phase 3 — Landmark extraction & storage
- [x] Phase 4 — LSTM training pipeline
- [x] Phase 5 — Real-time inference with trained model
- [x] Phase 6 — Offline rule-based recognizer (no training needed)
- [x] Phase 7 — Demo recording with R key
- [ ] Phase 8 — Streamlit web application
- [ ] Phase 9 — Custom gesture vocabulary via fine-tuning

---

## 🎯 Real World Applications

- ♿ Accessibility tools for people with disabilities
- 🤟 Sign language recognition
- 🖥️ Hands-free computer control
- 🎮 Gaming and interactive controls
- 📊 Touchless presentation control

---

## 📜 License
Educational use only.
Built and owned by **Charles Nwachukwu** — Nigeria 🇳🇬

---

## 🤝 Acknowledgements
- [MediaPipe by Google](https://mediapipe.dev)
- [PyTorch](https://pytorch.org)
- [OpenCV](https://opencv.org)
- Inspired by Farah Gherir's gesture recognition project on LinkedIn
