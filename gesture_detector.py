"""Hand gesture detection using MediaPipe HandLandmarker (Tasks API).

Gestures:
- "pinch": thumb + index fingertips close together (pinch),
           middle + ring + pinky curled. Drawing point is the midpoint.
- "palm":  all/most fingers extended. Eraser mode.
- "none":  no hand or unrecognized gesture.
"""

import os
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from config import Config

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task"
)


@dataclass
class GestureResult:
    """Result of gesture detection for a single frame."""
    gesture: str = "none"       # "none" | "pinch" | "palm"
    draw_point: Optional[tuple[int, int]] = None    # where to draw (pinch midpoint)
    palm_center: Optional[tuple[int, int]] = None    # eraser center
    landmarks: Optional[list] = None                  # raw landmarks for debug
    handedness: Optional[str] = None


class GestureSmoother:
    """Smooths gesture classification with a confirmation window."""

    def __init__(self, window: int = 3):
        self.window = window
        self.history: deque[str] = deque(maxlen=window)
        self.current = "none"

    def update(self, raw_gesture: str) -> str:
        """Feed a raw classification and return the confirmed gesture."""
        self.history.append(raw_gesture)
        if len(self.history) == self.history.maxlen:
            if all(g == self.history[0] for g in self.history):
                self.current = self.history[0]
        return self.current


class GestureDetector:
    """Detects hand gestures using MediaPipe HandLandmarker."""

    # Landmark indices
    WRIST = 0
    # Thumb: 1(CMC), 2(MCP), 3(IP), 4(TIP)
    THUMB_MCP = 2
    THUMB_TIP = 4
    # Index: 5(MCP), 6(PIP), 7(DIP), 8(TIP)
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_TIP = 8
    # Middle: 9(MCP), 10(PIP), 11(DIP), 12(TIP)
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    # Ring: 13(MCP), 14(PIP), 15(DIP), 16(TIP)
    RING_MCP = 13
    RING_PIP = 14
    RING_TIP = 16
    # Pinky: 17(MCP), 18(PIP), 19(DIP), 20(TIP)
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_TIP = 20

    # Finger defs for palm detection: (mcp, pip, tip)
    PALM_FINGERS = [
        (INDEX_MCP, INDEX_PIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    ]

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    def __init__(self, config: Config, model_path: str | None = None):
        self.config = config
        self._model_path = model_path or _DEFAULT_MODEL_PATH

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"MediaPipe hand landmark model not found at {self._model_path}. "
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.smoother = GestureSmoother(window=config.gesture_confirmation_window)

        # Drawing point smoothing (EMA). Lower = smoother but more lag.
        self._prev_draw_point: Optional[tuple[float, float]] = None
        self._prev_palm_center: Optional[tuple[float, float]] = None
        self._ema_alpha = config.smoothing

        # Frame counter for VIDEO mode timestamp
        self._frame_counter = 0

    @staticmethod
    def _dist(lm1, lm2) -> float:
        """Euclidean distance between two landmarks (in normalized coords)."""
        return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5

    def detect(self, frame: np.ndarray) -> GestureResult:
        """Detect hand gestures in a BGR frame."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_counter += 1
        result = self.detector.detect_for_video(mp_image, self._frame_counter * 33)

        if not result.hand_landmarks:
            self.smoother.update("none")
            self._prev_draw_point = None
            self._prev_palm_center = None
            return GestureResult(gesture="none")

        landmarks = result.hand_landmarks[0]
        handedness = (
            result.handedness[0][0].category_name if result.handedness else None
        )

        raw_gesture, draw_point, palm_center = self._classify_gesture(
            landmarks, w, h
        )
        draw_point = self._smooth_point(draw_point, "_prev_draw_point")
        palm_center = self._smooth_point(palm_center, "_prev_palm_center")
        smooth_gesture = self.smoother.update(raw_gesture)

        return GestureResult(
            gesture=smooth_gesture,
            draw_point=draw_point,
            palm_center=palm_center,
            landmarks=landmarks,
            handedness=handedness,
        )

    def _smooth_point(self, point, attr_name):
        """Apply EMA smoothing to a point for jitter reduction."""
        if point is None:
            setattr(self, attr_name, None)
            return None
        prev = getattr(self, attr_name)
        if prev is None:
            setattr(self, attr_name, (float(point[0]), float(point[1])))
            return point
        sx = self._ema_alpha * point[0] + (1 - self._ema_alpha) * prev[0]
        sy = self._ema_alpha * point[1] + (1 - self._ema_alpha) * prev[1]
        smoothed = (int(sx), int(sy))
        setattr(self, attr_name, (sx, sy))
        return smoothed

    def _classify_gesture(self, landmarks, frame_w, frame_h):
        """Classify hand gesture from landmarks.

        "pinch": thumb + index fingertips close together,
                 middle + ring + pinky curled.
        "palm":  all 4 fingers extended.
        "none":  anything else.
        """
        # Hand size reference: wrist to middle MCP distance
        hand_size = self._dist(
            landmarks[self.WRIST], landmarks[self.MIDDLE_MCP]
        )
        if hand_size < 0.02:  # too small / unreliable
            palm_center = self._compute_palm_center(landmarks, frame_w, frame_h)
            return ("none", None, palm_center)

        # --- Pinch detection: thumb tip close to index tip ---
        pinch_dist = self._dist(
            landmarks[self.THUMB_TIP], landmarks[self.INDEX_TIP]
        )

        # Middle + ring + pinky curled: tip close to MCP (< 50% hand_size)
        middle_curled = (
            self._dist(landmarks[self.MIDDLE_TIP], landmarks[self.MIDDLE_MCP])
            < hand_size * 0.5
        )
        ring_curled = (
            self._dist(landmarks[self.RING_TIP], landmarks[self.RING_MCP])
            < hand_size * 0.5
        )
        pinky_curled = (
            self._dist(landmarks[self.PINKY_TIP], landmarks[self.PINKY_MCP])
            < hand_size * 0.5
        )

        if (
            pinch_dist < hand_size * 0.35
            and middle_curled
            and ring_curled
            and pinky_curled
        ):
            # Drawing point: midpoint between thumb and index tips
            mx = (landmarks[self.THUMB_TIP].x + landmarks[self.INDEX_TIP].x) / 2
            my = (landmarks[self.THUMB_TIP].y + landmarks[self.INDEX_TIP].y) / 2
            dp = (int(mx * frame_w), int(my * frame_h))
            dp = (max(0, min(dp[0], frame_w - 1)),
                  max(0, min(dp[1], frame_h - 1)))
            palm_center = self._compute_palm_center(landmarks, frame_w, frame_h)
            return ("pinch", dp, palm_center)

        # --- Palm detection: each finger tip is far from its MCP ---
        extended_count = 0
        for mcp, pip, tip in self.PALM_FINGERS:
            tip_to_mcp = self._dist(landmarks[tip], landmarks[mcp])
            if tip_to_mcp > hand_size * 0.7:
                extended_count += 1

        palm_center = self._compute_palm_center(landmarks, frame_w, frame_h)

        if extended_count >= self.config.palm_min_extended_fingers:
            return ("palm", None, palm_center)

        return ("none", None, palm_center)

    def _compute_palm_center(self, landmarks, frame_w, frame_h):
        """Compute palm center as average of wrist + MCP joints (pixel coords)."""
        indices = [
            self.WRIST, self.INDEX_MCP, self.MIDDLE_MCP,
            self.RING_MCP, self.PINKY_MCP,
        ]
        px = sum(landmarks[i].x for i in indices) / len(indices)
        py = sum(landmarks[i].y for i in indices) / len(indices)
        return (
            max(0, min(int(px * frame_w), frame_w - 1)),
            max(0, min(int(py * frame_h), frame_h - 1)),
        )

    # --- Debug drawing helpers ---

    def draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        """Draw hand landmarks and connections on the frame for debugging."""
        h, w = frame.shape[:2]
        for s, e in self.HAND_CONNECTIONS:
            x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
            x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)

    def draw_pinch_indicator(self, frame, landmarks, color, frame_w, frame_h):
        """Draw dots on thumb and index fingertips to show pinch points."""
        for idx in (self.THUMB_TIP, self.INDEX_TIP):
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_eraser_indicator(self, frame, center, radius, color):
        """Draw a semi-transparent circle showing the eraser area."""
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, center, radius, color, 2)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.detector.close()
