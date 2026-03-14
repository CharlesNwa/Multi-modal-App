# 🖐️ Real-Time Hand Gesture Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

> A computer vision system that recognizes hand gestures in real time using MediaPipe for hand tracking and a Bidirectional LSTM for gesture classification — no keyboard or touchscreen required.

---

## 👤 Author

**Charles Nwachukwu** · Nigeria 🇳🇬
Built with Claude Code & Visual Studio Code

---

## 🎬 Demo

https://github.com/CharlesNwa/Multi-modal-App/blob/master/hand-gesture-recognition/demo_recording_compressed.mp4

> Live demo: detects hand gestures in real time and displays the label on screen with a confidence bar and hand skeleton overlay.

---

## 🎯 Gestures Recognized

| Gesture | Label |
|---------|-------|
| 👍 Thumbs Up | Thumbs Up |
| 👎 Thumbs Down | Thumbs Down |
| ✊ Closed Fist | Fist |
| ✌️ Peace / Victory | Peace |
| 🖐️ Open Palm | Stop |
| 👌 OK Sign | OK |

---

## 📌 How It Works

1. **Webcam** captures frames in real time
2. **MediaPipe Hands** detects 21 hand landmarks per frame (x, y, z)
3. **Geometric rules** classify the gesture from landmark positions — no ML model required
4. **Gesture label + confidence bar** displayed instantly on screen

> Works 100% offline — no dataset download, no internet, no model file needed.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| MediaPipe | Real-time hand landmark detection |
| OpenCV | Webcam capture, drawing, recording |
| PyTorch | BiLSTM model (optional training pipeline) |
| Streamlit | Web application UI |
| streamlit-webrtc | Live webcam stream in browser |
| NumPy | Landmark array operations |
| Scikit-learn | Data splitting for training |

---

## 📁 Project Structure

```
hand-gesture-recognition/
│
├── app.py                   # ✅ Streamlit web app (Phase 8)
├── recognize_mp.py          # ✅ Offline gesture recognizer (desktop)
├── recognize.py             # Recognizer using trained LSTM model
├── collect_gestures.py      # Data collection tool
├── train.py                 # LSTM training pipeline
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
└── demo_recording_compressed.mp4
```

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/CharlesNwa/Multi-modal-App.git
cd Multi-modal-App/hand-gesture-recognition
```

### 2. Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run (choose one)

**Desktop — no training needed:**
```bash
python recognize_mp.py
```

**Web app (Streamlit):**
```bash
pip install streamlit streamlit-webrtc av
streamlit run app.py
```

**Interactive menu (Windows):**
```bash
.\run.bat
```

---

## 🎮 Controls (Desktop)

| Key | Action |
|-----|--------|
| `R` | Start / Stop recording |
| `Q` | Quit |

---

## 📋 run.bat Menu

| Option | Description |
|--------|-------------|
| 1 | Download HaGRID dataset & extract landmarks |
| 2 | Train custom LSTM model |
| 3 | Run recognizer with trained model |
| 4 | Collect custom gesture data |
| 5 | Run offline desktop recognizer |
| 6 | Launch Streamlit web app |
| 7 | Exit |

---

## ⚙️ Configuration

Edit `config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `1` | `0` = laptop cam, `1` = USB cam |
| `SEQUENCE_LENGTH` | `30` | Frames per gesture sequence |
| `CONFIDENCE_THRESHOLD` | `0.3` | Min confidence to show prediction |
| `HIDDEN_DIM` | `128` | LSTM hidden size |
| `NUM_EPOCHS` | `50` | Training epochs |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera won't open | Change `CAMERA_INDEX` to `0` in `recognize_mp.py` |
| No hand detected | Improve lighting, keep hand fully in frame |
| Wrong gesture | Slow down, make the shape clearer |
| Low FPS | Close background apps |

---

## 🏗️ Optional: Train a Custom Model

```bash
# Collect data (15–20 samples per gesture)
python collect_gestures.py

# Train BiLSTM
python train.py

# Run with trained model
python recognize.py
```

**Model architecture:**
- Input: 30 frames × 126 features (42 landmarks × 3 coords)
- 3-layer Bidirectional LSTM · 128 hidden dims · dropout 0.3
- CrossEntropyLoss · Adam (lr=0.001)

---

## 🗺️ Roadmap

- [x] Phase 1 — Environment setup & MediaPipe integration
- [x] Phase 2 — Data collection pipeline
- [x] Phase 3 — Landmark extraction & storage
- [x] Phase 4 — LSTM training pipeline
- [x] Phase 5 — Real-time inference with trained model
- [x] Phase 6 — Offline rule-based recognizer
- [x] Phase 7 — Demo recording
- [x] Phase 8 — Streamlit web application
- [ ] Phase 9 — Custom gesture vocabulary via fine-tuning

---

## 🎯 Applications

- ♿ Accessibility tools
- 🤟 Sign language recognition
- 🖥️ Hands-free computer control
- 🎮 Gaming & interactive interfaces
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
