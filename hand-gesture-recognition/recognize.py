"""
Real-time hand gesture recognition using the trained BiLSTM model.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.mediapipe_extractor import HandLandmarkExtractor
from collect_gestures import draw_hands_ui


# ==================== Model (must match train.py) ====================

class GestureLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.INPUT_FEATURES,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL,
        )
        lstm_out = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ==================== Recognizer ====================

class GestureRecognizer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Run train.py first."
            )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = GestureLSTM().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Model loaded (val_acc={checkpoint.get('val_acc', 0):.1%})")

        # MediaPipe
        self.extractor = HandLandmarkExtractor(
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=config.HAND_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )

        # Camera
        print(f"Opening camera {config.CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        time.sleep(0.5)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            time.sleep(0.5)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        print("Camera ready.")

        # Sliding window buffer
        self.frame_buffer: deque = deque(maxlen=config.SEQUENCE_LENGTH)
        # Prediction smoothing
        self.pred_history: deque = deque(maxlen=config.SMOOTHING_WINDOW)

        self.gesture_names = [config.GESTURES[i] for i in sorted(config.GESTURES.keys())]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self):
        """Run inference on current buffer. Returns (label_idx, confidence) or None."""
        if len(self.frame_buffer) < config.SEQUENCE_LENGTH:
            return None, 0.0

        seq = np.array(self.frame_buffer, dtype=np.float32)           # (seq, 42, 3)
        seq = seq.reshape(1, config.SEQUENCE_LENGTH, config.INPUT_FEATURES)  # flatten
        tensor = torch.tensor(seq, dtype=torch.float32).to(self.device)

        logits = self.model(tensor)                                    # (1, num_classes)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
        return idx, conf

    # ------------------------------------------------------------------ #
    def _smoothed_prediction(self):
        """Return the most frequent prediction in the history window."""
        if not self.pred_history:
            return None, 0.0
        idxs  = [p[0] for p in self.pred_history]
        confs = [p[1] for p in self.pred_history]
        best  = max(set(idxs), key=idxs.count)
        avg_conf = np.mean([c for i, c in zip(idxs, confs) if i == best])
        return best, avg_conf

    # ------------------------------------------------------------------ #
    def _draw_prediction_hud(self, frame, pred_idx, conf, fps):
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 70), (15, 15, 15), -1)

        if pred_idx is not None and conf >= config.CONFIDENCE_THRESHOLD:
            gesture_name = self.gesture_names[pred_idx]
            label = config.GESTURE_LABELS.get(gesture_name, "") or gesture_name
            color = (0, 255, 100)
        elif pred_idx is not None:
            # Show raw name even below threshold so user knows something is happening
            gesture_name = self.gesture_names[pred_idx]
            label = f"[{config.GESTURE_LABELS.get(gesture_name, '') or gesture_name}]"
            color = (80, 180, 80)
        else:
            label = "Loading buffer..."
            color = (100, 100, 100)

        # Scale font to fit frame width
        font_scale = 1.0
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        if tw > w - 160:
            font_scale = font_scale * (w - 160) / tw

        cv2.putText(frame, label, (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

        # Confidence bar (right side)
        bar_w = int(conf * 120)
        cv2.rectangle(frame, (w - 135, 15), (w - 15, 35), (40, 40, 40), -1)
        cv2.rectangle(frame, (w - 135, 15), (w - 135 + bar_w, 35), color, -1)
        cv2.putText(frame, f"{conf:.0%}", (w - 135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Buffer fill indicator
        buf_pct = len(self.frame_buffer) / config.SEQUENCE_LENGTH
        cv2.rectangle(frame, (0, 70), (w, 75), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, 70), (int(w * buf_pct), 75), (0, 160, 255), -1)

        # Confidence % text
        cv2.putText(frame, f"conf: {conf:.0%}", (12, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # FPS bottom-right
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 80, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # Hint
        cv2.putText(frame, "Q = quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    # ------------------------------------------------------------------ #
    def run(self):
        win = "Gesture Recognition"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 600)
        print("\nRecognizer running. Press Q to quit.\n")

        prev_time = time.time()
        failed = 0

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                failed += 1
                if failed > 20:
                    print("Camera stalled — reopening...")
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
                    time.sleep(0.5)
                    failed = 0
                continue
            failed = 0

            frame = cv2.resize(frame, (640, 480))
            left_hand, right_hand = self.extractor.extract_landmarks_from_frame(frame)

            # Build combined (42, 3) array
            combined = np.zeros((42, 3), dtype=np.float32)
            if left_hand  is not None: combined[0:21]  = left_hand
            if right_hand is not None: combined[21:42] = right_hand
            self.frame_buffer.append(combined)

            # Predict
            idx, conf = self.predict()
            if idx is not None:
                self.pred_history.append((idx, conf))
            smooth_idx, smooth_conf = self._smoothed_prediction()

            # Draw
            display = frame.copy()
            draw_hands_ui(display, left_hand, right_hand)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            self._draw_prediction_hud(display, smooth_idx, smooth_conf, fps)

            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break

        self.cleanup()

    # ------------------------------------------------------------------ #
    def cleanup(self):
        if self.cap:
            try: self.cap.release()
            except: pass
        if self.extractor:
            try: self.extractor.close()
            except: pass
        try: cv2.destroyAllWindows()
        except: pass


# ==================== Main ====================

def main():
    model_path = os.path.join(config.CHECKPOINTS_DIR, "best_model.pt")
    try:
        recognizer = GestureRecognizer(model_path)
        recognizer.run()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        print("Done.")
        input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
