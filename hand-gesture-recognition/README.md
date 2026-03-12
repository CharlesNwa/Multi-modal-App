# Real-Time Hand Gesture Recognition System

> Built by **Charles** — from Nigeria

A real-time hand gesture recognition system using **MediaPipe** for hand landmark extraction and a **PyTorch Bidirectional LSTM** for temporal sequence classification. The model recognises personalised gestures that introduce Charles, his project, and his origin.

---

## Gesture Vocabulary

| Gesture | Hand Pose | Output |
|---|---|---|
| `hello_im_charles` | Wave / open palm facing camera | _"Hello, I'm Charles"_ |
| `gesture_recognition_system` | Index finger pointing up | _"Real-Time Hand Gesture Recognition System"_ |
| `from_nigeria` | Raised fist | _"I am from Nigeria"_ |
| `neutral` | Hand at rest | _(idle)_ |

---

## Project Structure

```
hand-gesture-recognition/
├── config.py                   # Central configuration (gestures, camera, model, thresholds)
├── collect_gestures.py         # Interactive data collection via webcam
├── verify_setup.py             # Environment & dependency check
├── test_camera.py              # Camera diagnostics
├── run.bat                     # Windows one-click launcher
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── mediapipe_extractor.py  # Hand landmark extraction (21 pts × 2 hands)
│   └── utils.py                # Data helpers, visualisation, file I/O
│
├── data/
│   ├── raw/                    # Recorded MP4 clips (per gesture class)
│   └── processed/              # Extracted landmark sequences (.npy)
│
└── models/
    └── checkpoints/            # Model checkpoints during training
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd hand-gesture-recognition
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python verify_setup.py
```

### 3. Collect Gesture Data

```bash
python collect_gestures.py
```

Type a gesture name (e.g. `hello_im_charles`) and hold the pose for 3 seconds. Repeat **20 times per gesture** for best accuracy.

**Available commands inside the collector:**

| Command | Action |
|---|---|
| `preview` | Open live camera preview |
| `stats` | Show collection progress |
| `quit` | Exit |

### 4. Train the Model _(coming in Phase 5)_

```bash
python train_model.py
```

### 5. Run Real-Time Demo _(coming in Phase 6)_

```bash
python demo.py
```

---

## Configuration

All settings are in [`config.py`](config.py). Key options:

| Setting | Default | Description |
|---|---|---|
| `VIDEOS_PER_GESTURE` | 20 | Target recordings per gesture |
| `VIDEO_DURATION_SECONDS` | 3 | Length of each recording |
| `SEQUENCE_LENGTH` | 30 | Frames per sequence (≈1 s at 30 fps) |
| `CAMERA_INDEX` | 1 | Camera to use (0 = laptop, 1 = USB) |
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum confidence to display a prediction |

---

## Technical Architecture

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

## Data Collection Tips

- Use **good lighting** — MediaPipe is sensitive to shadows
- Keep your **full hand in frame** for the 3-second duration
- **Vary angle and distance** slightly between recordings for robustness
- Aim for at least **20 samples per gesture** before training

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No hand detected" | Improve lighting; move hand closer to camera |
| Camera won't open | Change `CAMERA_INDEX` in `config.py` (try 0 or 1) |
| Low FPS in preview | Close background apps; reduce resolution in config |
| Low accuracy after training | Collect more varied samples (aim for 30+) |

---

## Roadmap

- [x] Phase 1 — Environment setup & MediaPipe integration
- [x] Phase 2 — Data collection pipeline
- [x] Phase 3 — Landmark extraction & storage
- [ ] Phase 4 — PyTorch Dataset & DataLoader
- [ ] Phase 5 — LSTM training pipeline
- [ ] Phase 6 — Real-time inference & demo UI

---

## License

Educational use. Built and owned by **Charles Nwa** — Nigeria.
