"""
Download HaGRID public dataset (6 gesture classes) and extract
MediaPipe landmarks so train.py can use them directly.

Run once:
    python prepare_hagrid.py
"""

import os, sys, numpy as np

SAMPLES_PER_CLASS = 30  # sequences to extract per gesture


def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkg.split())


def extract_landmarks(img_rgb, hands):
    """Return (126-d landmark array, hand_detected bool)."""
    import config
    results = hands.process(img_rgb)
    landmarks = np.zeros(config.INPUT_FEATURES, dtype=np.float32)
    if results.multi_hand_landmarks:
        for i, hand_lm in enumerate(results.multi_hand_landmarks[:2]):
            offset = i * 21 * 3
            for j, lm in enumerate(hand_lm.landmark):
                landmarks[offset + j * 3]     = lm.x
                landmarks[offset + j * 3 + 1] = lm.y
                landmarks[offset + j * 3 + 2] = lm.z
    return landmarks, results.multi_hand_landmarks is not None


def frame_to_sequence(landmarks, seq_len, jitter=0.003):
    """Repeat a single-frame landmark into a seq_len sequence with tiny jitter."""
    seq = np.tile(landmarks, (seq_len, 1))
    seq += np.random.normal(0, jitter, seq.shape).astype(np.float32)
    return seq.astype(np.float32)


def main():
    # ── deps ────────────────────────────────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing HuggingFace datasets …")
        install("datasets pillow huggingface_hub")
        from datasets import load_dataset

    try:
        import mediapipe as mp
    except ImportError:
        print("Installing mediapipe …")
        install("mediapipe")
        import mediapipe as mp

    import config

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
        min_detection_confidence=config.HAND_CONFIDENCE_THRESHOLD,
    )

    print("=" * 55)
    print("  HaGRID → MediaPipe landmarks extractor")
    print("  Dataset: abhishek/hagrid (streaming - no full download)")
    print("=" * 55)

    for class_idx, gesture_name in config.GESTURES.items():
        out_dir = os.path.join(config.PROCESSED_DATA_DIR, gesture_name)
        os.makedirs(out_dir, exist_ok=True)

        existing = [f for f in os.listdir(out_dir) if f.endswith("_landmarks.npy")]
        if len(existing) >= SAMPLES_PER_CLASS:
            print(f"[SKIP] {gesture_name} — already has {len(existing)} samples")
            continue

        print(f"\n[{class_idx+1}/6] Processing: {gesture_name} …")

        try:
            ds = load_dataset(
                "abhishek/hagrid",
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            continue

        count = len(existing)
        scanned = 0
        for sample in ds:
            if count >= SAMPLES_PER_CLASS:
                break
            scanned += 1
            if scanned > 5000:  # safety limit to avoid scanning forever
                break

            # Each sample has a 'labels' list with gesture names
            sample_labels = sample.get("labels", [])
            if gesture_name not in sample_labels:
                continue

            try:
                img_pil = sample.get("image")
                if img_pil is None:
                    continue
                img_rgb = np.array(img_pil.convert("RGB"))
            except Exception:
                continue

            landmarks, detected = extract_landmarks(img_rgb, hands)
            if not detected:
                continue

            seq = frame_to_sequence(landmarks, config.SEQUENCE_LENGTH)
            save_path = os.path.join(out_dir, f"{count:03d}_landmarks.npy")
            np.save(save_path, seq)
            count += 1
            print(f"  {gesture_name}: {count}/{SAMPLES_PER_CLASS} (scanned {scanned})", end="\r")

        print(f"  {gesture_name}: saved {count} sequences (scanned {scanned} images)    ")

    hands.close()
    print("\nDone! Now run:  python train.py")


if __name__ == "__main__":
    main()
