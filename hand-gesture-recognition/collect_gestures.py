"""
Interactive gesture data collection script using webcam.
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.mediapipe_extractor import HandLandmarkExtractor
from src.utils import (
    draw_fps,
    pad_landmarks_sequence,
    create_gesture_directories,
    get_next_video_id,
    print_gesture_stats,
)
import config


# ==================== UI Helpers ====================

def _draw_corner_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     color: tuple, thickness: int = 2, corner_len: int = 20) -> None:
    """Draw a corner-bracket style bounding box (in-place)."""
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)


def _label_with_bg(frame: np.ndarray, text: str, x: int, y: int,
                   color: tuple, font_scale: float = 0.6, thickness: int = 2) -> None:
    """Draw text with a dark background pill for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_hands_ui(frame: np.ndarray,
                  left_hand: np.ndarray | None,
                  right_hand: np.ndarray | None) -> np.ndarray:
    """
    Draw landmarks + corner bounding boxes for both hands.
    Left = green (#00FF00), Right = cyan (#00FFFF).
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    hand_configs = [
        (left_hand,  (0, 255, 0),   (255, 80,  0),   "Left"),
        (right_hand, (0, 255, 255), (0,  165, 255),  "Right"),
    ]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for lm, dot_color, line_color, label in hand_configs:
        if lm is None:
            continue

        px = (lm[:, 0] * w).astype(int)
        py = (lm[:, 1] * h).astype(int)

        # Connections
        for s, e in connections:
            cv2.line(overlay, (px[s], py[s]), (px[e], py[e]), line_color, 2, cv2.LINE_AA)

        # Landmark dots
        for i in range(len(lm)):
            radius = 5 if i == 0 else 3   # wrist slightly larger
            cv2.circle(overlay, (px[i], py[i]), radius, dot_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (px[i], py[i]), radius, (255, 255, 255), 1, cv2.LINE_AA)

        # Corner bounding box
        pad = 18
        x1 = max(0, int(np.min(px)) - pad)
        y1 = max(0, int(np.min(py)) - pad)
        x2 = min(w - 1, int(np.max(px)) + pad)
        y2 = min(h - 1, int(np.max(py)) + pad)

        # Semi-transparent fill
        roi = overlay[y1:y2, x1:x2]
        fill = np.full_like(roi, 30)
        cv2.addWeighted(fill, 0.25, roi, 0.75, 0, roi)

        _draw_corner_box(overlay, x1, y1, x2, y2, dot_color, thickness=3, corner_len=22)
        _label_with_bg(overlay, label, x1, y1 - 10, dot_color, font_scale=0.65)

    # Blend overlay onto frame
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


def _draw_status_bar(frame: np.ndarray, gesture_name: str,
                     remaining: float, progress: int,
                     left_detected: bool, right_detected: bool) -> None:
    """Draw a clean HUD at the top and bottom of the frame."""
    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)

    # Gesture name
    cv2.putText(frame, f"REC  {gesture_name.upper()}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

    # Timer
    timer_txt = f"{remaining:.1f}s  [{progress}%]"
    (tw, _), _ = cv2.getTextSize(timer_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, timer_txt, (w - tw - 12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Progress bar
    bar_x1, bar_y1, bar_y2 = 0, 45, 50
    cv2.rectangle(frame, (bar_x1, bar_y1), (w, bar_y2), (40, 40, 40), -1)
    filled = int(w * progress / 100)
    cv2.rectangle(frame, (bar_x1, bar_y1), (filled, bar_y2), (0, 200, 100), -1)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 38), (w, h), (20, 20, 20), -1)

    # Hand status dots
    lc = (0, 255, 0) if left_detected else (60, 60, 60)
    rc = (0, 255, 255) if right_detected else (60, 60, 60)
    cv2.circle(frame, (20, h - 19), 8, lc, -1)
    cv2.putText(frame, "L", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.circle(frame, (42, h - 19), 8, rc, -1)
    cv2.putText(frame, "R", (36, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    hint = "Q = cancel"
    cv2.putText(frame, hint, (w - 90, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


# ==================== Collector ====================

class GestureCollector:
    """Interactive gesture collection system."""

    def __init__(self):
        print("Loading MediaPipe...")
        self.extractor = HandLandmarkExtractor(
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=config.HAND_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        print("MediaPipe ready.")

        print(f"Opening camera {config.CAMERA_INDEX} (DirectShow)...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        time.sleep(1)

        if not self.cap.isOpened():
            print(f"Camera {config.CAMERA_INDEX} failed, trying 0...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            time.sleep(1)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open any camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        ret, frame = self.cap.read()
        if ret:
            print(f"Camera ready — {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("Warning: first frame failed, continuing anyway.")

        self.fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = None
        self.is_recording = False
        self.recorded_landmarks = []

        create_gesture_directories()
        print("All initialized.\n")

    # ------------------------------------------------------------------
    def record_gesture(self, gesture_name: str,
                       duration: float = config.VIDEO_DURATION_SECONDS) -> bool:

        if gesture_name not in config.GESTURE_NAMES:
            print(f"Unknown gesture: {gesture_name}")
            return False

        video_id     = get_next_video_id(gesture_name)
        raw_dir      = os.path.join(config.RAW_DATA_DIR, gesture_name)
        video_fname  = f"{video_id:03d}_{gesture_name}.mp4"
        video_path   = os.path.join(raw_dir, video_fname)

        self.recorded_landmarks = []
        self.is_recording       = True

        self.video_writer = cv2.VideoWriter(
            video_path, self.fourcc, config.FPS, (640, 480)
        )

        win_name = f"Recording: {gesture_name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 600)

        print(f"\nCamera open — perform your gesture.")
        print("  SPACE or S = save  |  Q = cancel\n")

        # Flush stale frames
        for _ in range(5):
            self.cap.grab()

        start_time      = time.time()
        frames_captured = 0
        failed_reads    = 0
        hands_detected  = 0
        cancelled       = False
        saved           = False

        print("  SPACE or S = save & finish  |  Q = cancel\n")

        while True:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                failed_reads += 1
                if failed_reads > 15:
                    print("Camera stalled — reopening...")
                    self.cap.release()
                    time.sleep(0.4)
                    self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
                    time.sleep(0.4)
                    failed_reads = 0
                continue

            failed_reads = 0
            frame = cv2.resize(frame, (640, 480))

            # Extract landmarks
            left_hand, right_hand = self.extractor.extract_landmarks_from_frame(frame)

            combined = np.zeros((42, 3), dtype=np.float32)
            if left_hand  is not None: combined[0:21]  = left_hand
            if right_hand is not None: combined[21:42] = right_hand
            self.recorded_landmarks.append(combined)
            frames_captured += 1

            if left_hand is not None or right_hand is not None:
                hands_detected += 1

            if self.video_writer:
                self.video_writer.write(frame)

            # ---- Build display frame ----
            display = frame.copy()
            draw_hands_ui(display, left_hand, right_hand)

            elapsed = time.time() - start_time
            h, w = display.shape[:2]

            # Top bar
            cv2.rectangle(display, (0, 0), (w, 45), (20, 20, 20), -1)
            cv2.putText(display, f"REC  {gesture_name.upper()}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            elapsed_txt = f"{elapsed:.1f}s  [{frames_captured} frames]"
            (tw, _), _ = cv2.getTextSize(elapsed_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(display, elapsed_txt, (w - tw - 12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Bottom hint bar
            cv2.rectangle(display, (0, h - 38), (w, h), (20, 20, 20), -1)
            lc = (0, 255, 0) if left_hand is not None else (60, 60, 60)
            rc = (0, 255, 255) if right_hand is not None else (60, 60, 60)
            cv2.circle(display, (20, h - 19), 8, lc, -1)
            cv2.putText(display, "L", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.circle(display, (42, h - 19), 8, rc, -1)
            cv2.putText(display, "R", (36, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(display, "SPACE/S = Save   Q = Cancel", (65, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("Recording cancelled.")
                cancelled = True
                break
            elif key in (ord('s'), ord('S'), 32):  # S or SPACE
                saved = True
                print(f"\nSaved! {frames_captured} frames captured.")
                break

            if frames_captured % 15 == 0:
                print(f"\r  {frames_captured} frames | {elapsed:.1f}s elapsed",
                      end="", flush=True)

        print(f"\r  [100%] {frames_captured} frames | Done!          ", flush=True)

        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        cv2.destroyWindow(win_name)

        if cancelled or not saved or frames_captured == 0:
            print("No frames captured." if frames_captured == 0 else "")
            return False

        print(f"\nRecording complete — {frames_captured} frames, {hands_detected} with hands.")

        # Save landmarks
        try:
            seq = np.array(self.recorded_landmarks, dtype=np.float32)

            if seq.shape[0] < config.MIN_FRAMES_PER_VIDEO:
                need  = config.MIN_FRAMES_PER_VIDEO - seq.shape[0]
                last  = seq[-1] if seq.shape[0] > 0 else np.zeros((42, 3), dtype=np.float32)
                seq   = np.vstack([seq, np.tile(last, (need, 1, 1))])

            lm_path = os.path.join(config.PROCESSED_DATA_DIR, gesture_name,
                                   f"{video_id:03d}_landmarks.npy")
            np.save(lm_path, seq)
            print(f"Saved: data/processed/{gesture_name}/{video_id:03d}_landmarks.npy")
        except Exception as exc:
            print(f"Warning: could not save landmarks: {exc}")

        return True

    # ------------------------------------------------------------------
    def display_live_preview(self):
        win_name = "Live Preview — Hand Gesture"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 600)
        print("\nPreview open. Press Q to exit.\n")

        frame_n = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (640, 480))
            left_hand, right_hand = self.extractor.extract_landmarks_from_frame(frame)

            draw_hands_ui(frame, left_hand, right_hand)

            # Top status
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
            if left_hand is not None or right_hand is not None:
                parts = []
                if left_hand  is not None: parts.append("Left")
                if right_hand is not None: parts.append("Right")
                txt = "DETECTED: " + " + ".join(parts)
                cv2.putText(frame, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hands — show your hand(s)", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.putText(frame, "Q = exit preview", (w - 160, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

            frame_n += 1

        cv2.destroyWindow(win_name)
        print("Preview closed.")

    # ------------------------------------------------------------------
    def playback_gesture(self, gesture_name: str):
        if gesture_name not in config.GESTURE_NAMES:
            print(f"Unknown gesture: {gesture_name}")
            return

        raw_dir = os.path.join(config.RAW_DATA_DIR, gesture_name)
        videos  = sorted([f for f in os.listdir(raw_dir) if f.endswith(".mp4")]) \
                  if os.path.exists(raw_dir) else []
        if not videos:
            print(f"No videos for {gesture_name}")
            return

        video_path = os.path.join(raw_dir, videos[-1])
        print(f"Playing: {videos[-1]}")
        cap = cv2.VideoCapture(video_path)

        win_name = f"Playback: {gesture_name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 600)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            left_hand, right_hand = self.extractor.extract_landmarks_from_frame(frame)
            draw_hands_ui(frame, left_hand, right_hand)
            cv2.imshow(win_name, frame)
            if cv2.waitKey(33) & 0xFF in (ord('q'), ord('Q'), 27):
                break

        cap.release()
        cv2.destroyWindow(win_name)

    # ------------------------------------------------------------------
    def cleanup(self):
        if self.video_writer:
            try: self.video_writer.release()
            except: pass
        if self.cap:
            try: self.cap.release()
            except: pass
        if self.extractor:
            try: self.extractor.close()
            except: pass
        try: cv2.destroyAllWindows()
        except: pass

    def __del__(self):
        self.cleanup()

    # ------------------------------------------------------------------
    def interactive_mode(self):
        print("\n" + "=" * 60)
        print("  HAND GESTURE RECOGNITION — DATA COLLECTION")
        print("=" * 60)
        print(f"  Gestures : {', '.join(config.GESTURE_NAMES)}")
        print(f"  Duration : {config.VIDEO_DURATION_SECONDS}s per recording")
        print()
        print("  Commands : preview | stats | quit")
        print("  In window: SPACE or S = save  |  Q = cancel")
        print("  Record   : type a gesture name")
        print("-" * 60)

        while True:
            print_gesture_stats()
            user_input = input("\nGesture / command: ").strip().lower()

            if user_input == "quit":
                print("Exiting.")
                break
            elif user_input == "stats":
                continue
            elif user_input == "preview":
                self.display_live_preview()
            elif user_input in config.GESTURE_NAMES:
                while True:
                    ok = self.record_gesture(user_input)
                    if ok:
                        print_gesture_stats()
                        again = input(f"\nRecord another '{user_input}'? (Y to continue / N to stop): ").strip().lower()
                        if again != 'y':
                            break
                    else:
                        break
            else:
                print(f"Unknown: '{user_input}'")
                print(f"  Gestures : {', '.join(config.GESTURE_NAMES)}")
                print(f"  Commands : preview, stats, quit")


# ==================== Main ====================

def main():
    try:
        collector = GestureCollector()
        collector.interactive_mode()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        input("\nPress Enter to close...")


if __name__ == "__main__":
    main()

