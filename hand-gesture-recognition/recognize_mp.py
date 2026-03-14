"""
Real-time gesture recognition using MediaPipe hand landmarks + geometric rules.
Works 100% offline — no model download required.

Gestures detected:
  Thumbs Up, Thumbs Down, Peace/Victory, Stop/Open Palm, Fist, OK
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys

CAMERA_INDEX = 1   # change to 0 if using laptop camera


# ── Gesture classifier (rule-based, no ML model needed) ─────────────────────

def classify_gesture(lm):
    """
    lm: list of 21 NormalizedLandmark (each has .x, .y, .z)
    Returns (label_string, confidence) or ("", 0.0)
    """
    # Finger extended: tip.y significantly above pip.y (y=0 is top of frame)
    idx_ext = lm[8].y  < lm[6].y  - 0.02
    mid_ext = lm[12].y < lm[10].y - 0.02
    rng_ext = lm[16].y < lm[14].y - 0.02
    pky_ext = lm[20].y < lm[18].y - 0.02

    # Thumb: tip above / below its MCP
    thumb_raised  = lm[4].y < lm[2].y - 0.04   # thumb up
    thumb_lowered = lm[4].y > lm[2].y + 0.04   # thumb down

    four_curled  = not idx_ext and not mid_ext and not rng_ext and not pky_ext
    all_extended = idx_ext and mid_ext and rng_ext and pky_ext

    # OK: thumb tip close to index tip, other three fingers extended
    thumb_idx_dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2) ** 0.5

    if four_curled and thumb_raised:
        return "Thumbs Up 👍", 0.92
    if four_curled and thumb_lowered:
        return "Thumbs Down 👎", 0.92
    if four_curled:
        return "Fist ✊", 0.88
    if idx_ext and mid_ext and not rng_ext and not pky_ext:
        return "Peace ✌️", 0.92
    if all_extended:
        return "Stop 🖐️", 0.88
    if thumb_idx_dist < 0.08 and mid_ext and rng_ext and pky_ext:
        return "OK 👌", 0.88

    return "", 0.0


# ── Main loop ────────────────────────────────────────────────────────────────

def run():
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    print(f"Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    time.sleep(0.5)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(0.5)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Camera ready. Press Q to quit.\n")

    win = "Hand Gesture Recognition"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 600)

    label     = ""
    conf      = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            label, conf = classify_gesture(hand_lm.landmark)
        else:
            label, conf = "", 0.0

        # ── HUD ─────────────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Dark top bar
        cv2.rectangle(frame, (0, 0), (w, 75), (15, 15, 15), -1)

        display_text = label if label else "Show a hand gesture..."
        color        = (0, 255, 100) if label else (80, 80, 80)

        # Auto-scale font so text always fits
        font_scale = 1.2
        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        if tw > w - 20:
            font_scale = font_scale * (w - 20) / tw

        cv2.putText(frame, display_text, (12, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

        # Confidence bar
        if label:
            bar_w = int(conf * 160)
            cv2.rectangle(frame, (10, 62), (170, 72), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, 62), (10 + bar_w, 72), color, -1)
            cv2.putText(frame, f"{conf:.0%}", (175, 71),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # FPS + hint
        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 80, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        cv2.putText(frame, "Q = quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Done.")
        input("Press Enter to close...")
