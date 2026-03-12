"""
MediaPipe Hand Landmark Extractor.
Extracts 21 hand landmarks from video frames using MediaPipe Hands.
Supports up to two hands simultaneously.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class HandLandmarkExtractor:
    """
    Extracts hand landmarks from video frames using MediaPipe.

    Attributes:
        - Detects hand presence in frame
        - Extracts 21 landmarks per hand (x, y, z normalized to [0,1])
        - Supports up to two hands (left and right)
        - Returns None for a hand if not detected
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Hands detector.

        Args:
            static_image_mode: If True, detection runs on every frame. If False, detection
                runs only when tracking is lost (faster).
            max_num_hands: Maximum number of hands to detect (1 or 2).
            model_complexity: 0 (lite, ~6MB) or 1 (full, ~30MB). Full is more accurate.
            min_detection_confidence: Minimum confidence threshold for detection.
            min_tracking_confidence: Minimum confidence threshold for tracking.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2
        )
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=2
        )

    def extract_landmarks_from_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract hand landmarks from a single frame for up to two hands.

        Args:
            frame: Input frame (BGR format, numpy array, shape: (H, W, 3))

        Returns:
            Tuple of (left_hand, right_hand) where each is either:
                - numpy array of shape (21, 3) with normalized landmarks [x, y, z]
                - None if that hand was not detected
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                landmark_array = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                    dtype=np.float32,
                )
                if label == "Left":
                    left_hand = landmark_array
                else:
                    right_hand = landmark_array

        return left_hand, right_hand

    def extract_landmarks_from_video(
        self, video_path: str
    ) -> Tuple[List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]], List[int]]:
        """
        Extract hand landmarks from all frames in a video file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of:
                - List of (left_hand, right_hand) tuples per frame
                - List of frame indices where at least one hand was detected
        """
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        frame_indices_with_hands = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            left_hand, right_hand = self.extract_landmarks_from_frame(frame)
            landmarks_list.append((left_hand, right_hand))

            if left_hand is not None or right_hand is not None:
                frame_indices_with_hands.append(frame_count)

            frame_count += 1

        cap.release()
        return landmarks_list, frame_indices_with_hands

    def _draw_single_hand(
        self,
        landmark_frame: np.ndarray,
        landmarks: np.ndarray,
        w: int,
        h: int,
        dot_color: Tuple[int, int, int],
        line_color: Tuple[int, int, int],
    ) -> None:
        """Draw landmarks and connections for one hand onto landmark_frame (in-place)."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17),             # Palm
        ]
        for x, y, _ in landmarks:
            cv2.circle(landmark_frame, (int(x * w), int(y * h)), 3, dot_color, -1)
        for start, end in connections:
            x1, y1, _ = landmarks[start]
            x2, y2, _ = landmarks[end]
            cv2.line(
                landmark_frame,
                (int(x1 * w), int(y1 * h)),
                (int(x2 * w), int(y2 * h)),
                line_color, 2,
            )

    def draw_landmarks_on_frame(
        self,
        frame: np.ndarray,
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Draw hand landmarks on frame for up to two hands.

        Args:
            frame: Input frame (BGR format)
            left_hand: Left hand landmarks (21, 3) or None
            right_hand: Right hand landmarks (21, 3) or None

        Returns:
            Frame with drawn landmarks (left=green, right=cyan)
        """
        if left_hand is None and right_hand is None:
            return frame

        h, w, c = frame.shape
        landmark_frame = np.zeros((h, w, c), dtype=np.uint8)

        if left_hand is not None:
            self._draw_single_hand(landmark_frame, left_hand, w, h, (0, 255, 0), (255, 0, 0))
        if right_hand is not None:
            self._draw_single_hand(landmark_frame, right_hand, w, h, (0, 255, 255), (0, 165, 255))

        return cv2.addWeighted(frame, 0.7, landmark_frame, 0.3, 0)

    def draw_hand_presence_box(
        self,
        frame: np.ndarray,
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Draw bounding boxes around detected hands.

        Args:
            frame: Input frame (BGR format)
            left_hand: Left hand landmarks (21, 3) or None
            right_hand: Right hand landmarks (21, 3) or None

        Returns:
            Frame with bounding boxes (left=green, right=cyan)
        """
        h, w, c = frame.shape
        padding = 10

        for landmarks, color, label in [
            (left_hand, (0, 255, 0), "Left"),
            (right_hand, (0, 255, 255), "Right"),
        ]:
            if landmarks is None:
                continue
            xs = landmarks[:, 0] * w
            ys = landmarks[:, 1] * h
            x_min = max(0, int(np.min(xs)) - padding)
            y_min = max(0, int(np.min(ys)) - padding)
            x_max = min(w, int(np.max(xs)) + padding)
            y_max = min(h, int(np.max(ys)) + padding)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def close(self):
        """Close MediaPipe resources."""
        self.hands.close()

    def __del__(self):
        """Cleanup on object deletion."""
        self.close()
