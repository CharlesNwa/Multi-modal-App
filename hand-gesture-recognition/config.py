"""
Project configuration and constants for hand gesture recognition.
"""

import os

# ==================== Project Paths ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== Gesture Configuration ====================
GESTURES = {
    0: "like",
    1: "dislike",
    2: "peace",
    3: "stop",
    4: "fist",
    5: "ok",
}

GESTURE_NAMES = list(GESTURES.values())
NUM_CLASSES = len(GESTURES)

# Display labels shown on screen during inference
GESTURE_LABELS = {
    "like":    "Thumbs Up",
    "dislike": "Thumbs Down",
    "peace":   "Peace",
    "stop":    "Stop",
    "fist":    "Fist",
    "ok":      "OK",
}

# ==================== MediaPipe Settings ====================
MEDIAPIPE_MODEL_COMPLEXITY = 0  # 0 (lite), 1 (full) - use 0 to prevent hangs
HAND_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to detect hand
TRACKING_CONFIDENCE = 0.5

# ==================== Frame & Sequence Settings ====================
SEQUENCE_LENGTH = 30  # Number of frames in a sequence (~1 second at 30fps)
NUM_LANDMARKS = 42  # 21 per hand × 2 hands (left + right)
LANDMARKS_DIMS = 3  # x, y, z coordinates
INPUT_FEATURES = NUM_LANDMARKS * LANDMARKS_DIMS  # 126 features

# ==================== Camera Settings ====================
CAMERA_INDEX = 1  # 0=laptop, 1=external USB camera
CAMERA_WIDTH = 640  # Reduced resolution for stability
CAMERA_HEIGHT = 480  # Reduced resolution for stability
FPS = 30

# ==================== Data Collection Settings ====================
VIDEOS_PER_GESTURE = 20  # Target videos to collect per gesture
VIDEO_DURATION_SECONDS = 3  # How long to record each gesture
MIN_FRAMES_PER_VIDEO = 30  # Minimum frames needed in a video

# ==================== LSTM Model Hyperparameters ====================
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
BIDIRECTIONAL = True

# ==================== Training Settings ====================
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==================== Inference Settings ====================
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display predicted gesture
SMOOTHING_WINDOW = 5  # Number of predictions to smooth over

# ==================== Visualization Settings ====================
LANDMARK_COLOR = (0, 255, 0)  # BGR format
CONNECTION_COLOR = (255, 0, 0)
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)
