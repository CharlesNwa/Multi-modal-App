"""
Utility functions for data processing, visualization, and file handling.
"""

import cv2
import numpy as np
import os
import json
from typing import Optional, List, Tuple
import config


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to have zero mean and unit variance across all dimensions.

    Args:
        landmarks: Array of shape (21, 3) or (N, 21, 3) for batch

    Returns:
        Normalized landmarks with same shape
    """
    if landmarks is None:
        return None

    original_shape = landmarks.shape
    landmarks_flat = landmarks.reshape(-1, config.INPUT_FEATURES)

    # Compute mean and std
    mean = np.mean(landmarks_flat, axis=0)
    std = np.std(landmarks_flat, axis=0)
    std[std == 0] = 1  # Avoid division by zero

    # Normalize
    normalized = (landmarks_flat - mean) / std
    return normalized.reshape(original_shape)


def save_landmarks_to_npy(
    landmarks_list: List[Optional[np.ndarray]], output_path: str
) -> None:
    """
    Save list of landmarks sequences to numpy file.

    Args:
        landmarks_list: List of landmarks arrays (some may be None)
        output_path: Path to save .npy file
    """
    # Pad sequences to SEQUENCE_LENGTH
    padded_seq = pad_landmarks_sequence(landmarks_list, config.SEQUENCE_LENGTH)
    np.save(output_path, padded_seq)


def load_landmarks_from_npy(file_path: str) -> np.ndarray:
    """
    Load landmarks from numpy file.

    Args:
        file_path: Path to .npy file

    Returns:
        Landmarks array of shape (SEQUENCE_LENGTH, 21, 3)
    """
    return np.load(file_path)


def pad_landmarks_sequence(
    landmarks_list: List[np.ndarray], target_length: int
) -> np.ndarray:
    """
    Pad or truncate landmarks sequence to target length.
    Each element should be a (42, 3) combined two-hand array (zeros for absent hands).

    Args:
        landmarks_list: List of (42, 3) landmark arrays
        target_length: Target sequence length

    Returns:
        Padded array of shape (target_length, 42, 3)
    """
    valid_landmarks = [lm for lm in landmarks_list if lm is not None]

    if len(valid_landmarks) == 0:
        return np.zeros((target_length, config.NUM_LANDMARKS, config.LANDMARKS_DIMS), dtype=np.float32)

    stacked = np.stack(valid_landmarks, axis=0)  # (num_frames, 42, 3)

    if len(valid_landmarks) >= target_length:
        return stacked[:target_length].astype(np.float32)
    else:
        padded = np.zeros((target_length, config.NUM_LANDMARKS, config.LANDMARKS_DIMS), dtype=np.float32)
        padded[: len(valid_landmarks)] = stacked
        return padded


def save_landmarks_to_csv(landmarks: np.ndarray, output_path: str) -> None:
    """
    Save landmarks to CSV file (for inspection/analysis).

    Args:
        landmarks: Array of shape (21, 3)
        output_path: Path to save CSV file
    """
    np.savetxt(output_path, landmarks.reshape(-1), delimiter=",")


def load_landmarks_from_csv(file_path: str) -> np.ndarray:
    """
    Load landmarks from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        Landmarks array of shape (21, 3)
    """
    data = np.loadtxt(file_path, delimiter=",")
    return data.reshape(config.NUM_LANDMARKS, config.LANDMARKS_DIMS)


def draw_fps(frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw FPS counter on frame.

    Args:
        frame: Input frame (BGR format)
        fps: Current FPS value
        position: (x, y) coordinate for text placement

    Returns:
        Frame with FPS text
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        config.TEXT_COLOR,
        config.FONT_THICKNESS,
    )
    return frame


def draw_landmarks(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    draw_box: bool = True,
) -> np.ndarray:
    """
    Draw hand landmarks on frame.

    Args:
        frame: Input frame (BGR format)
        landmarks: Combined two-hand array (42, 3) — left hand [0:21], right hand [21:42].
                   A hand is skipped if all its values are zero.
        draw_box: Whether to draw bounding box

    Returns:
        Frame with drawn landmarks
    """
    if landmarks is None:
        return frame

    h, w, c = frame.shape
    output_frame = frame.copy()

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    hand_configs = [
        (landmarks[0:21],  config.LANDMARK_COLOR,   config.CONNECTION_COLOR,  "Left"),
        (landmarks[21:42], (0, 255, 255),            (0, 165, 255),            "Right"),
    ]

    for hand_lm, dot_color, line_color, label in hand_configs:
        if np.all(hand_lm == 0):
            continue

        for x, y, _ in hand_lm:
            cv2.circle(output_frame, (int(x * w), int(y * h)), 4, dot_color, -1)

        for start, end in connections:
            x1, y1, _ = hand_lm[start]
            x2, y2, _ = hand_lm[end]
            cv2.line(output_frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), line_color, 2)

        if draw_box:
            xs = hand_lm[:, 0] * w
            ys = hand_lm[:, 1] * h
            pad = 10
            x_min = max(0, int(np.min(xs)) - pad)
            y_min = max(0, int(np.min(ys)) - pad)
            x_max = min(w, int(np.max(xs)) + pad)
            y_max = min(h, int(np.max(ys)) + pad)
            cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), dot_color, 2)
            cv2.putText(output_frame, label, (x_min, y_min - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color, 2)

    return output_frame


def create_gesture_directories() -> None:
    """Create directories for each gesture in raw and processed data folders."""
    for gesture_name in config.GESTURE_NAMES:
        raw_dir = os.path.join(config.RAW_DATA_DIR, gesture_name)
        processed_dir = os.path.join(config.PROCESSED_DATA_DIR, gesture_name)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)


def get_next_video_id(gesture_name: str) -> int:
    """
    Get the next video ID for a gesture (based on existing files).

    Args:
        gesture_name: Name of gesture

    Returns:
        Next available video ID
    """
    raw_dir = os.path.join(config.RAW_DATA_DIR, gesture_name)
    if not os.path.exists(raw_dir):
        return 0

    existing_files = os.listdir(raw_dir)
    if not existing_files:
        return 0

    # Extract video IDs from filenames
    video_ids = []
    for filename in existing_files:
        if filename.endswith(".mp4"):
            try:
                video_id = int(filename.split("_")[0])
                video_ids.append(video_id)
            except (ValueError, IndexError):
                pass

    return max(video_ids) + 1 if video_ids else 0


def get_gesture_stats() -> dict:
    """
    Get collection statistics for each gesture.

    Returns:
        Dictionary with gesture names and counts of collected videos
    """
    stats = {}
    for gesture_name in config.GESTURE_NAMES:
        raw_dir = os.path.join(config.RAW_DATA_DIR, gesture_name)
        processed_dir = os.path.join(config.PROCESSED_DATA_DIR, gesture_name)

        raw_count = len([f for f in os.listdir(raw_dir) if f.endswith(".mp4")]) if os.path.exists(raw_dir) else 0
        processed_count = len([f for f in os.listdir(processed_dir) if f.endswith(".npy")]) if os.path.exists(processed_dir) else 0

        stats[gesture_name] = {
            "raw_videos": raw_count,
            "processed_sequences": processed_count,
        }

    return stats


def print_gesture_stats() -> None:
    """Print collection statistics for all gestures."""
    stats = get_gesture_stats()
    print("\n" + "=" * 60)
    print("GESTURE COLLECTION STATISTICS")
    print("=" * 60)
    total_raw = 0
    total_processed = 0
    for gesture, counts in stats.items():
        print(f"{gesture:15} | Raw: {counts['raw_videos']:2} | Processed: {counts['processed_sequences']:2}")
        total_raw += counts["raw_videos"]
        total_processed += counts["processed_sequences"]
    print("-" * 60)
    print(f"{'TOTAL':15} | Raw: {total_raw:2} | Processed: {total_processed:2}")
    print("=" * 60 + "\n")
