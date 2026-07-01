"""Shared utilities for draw-on-cam."""

import time
from collections import deque

import cv2
import numpy as np


class FPSMeter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 30):
        self.times = deque(maxlen=window)

    def tick(self) -> None:
        """Record a frame timestamp."""
        self.times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Current rolling-average FPS."""
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.6,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 4,
) -> None:
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(
        frame,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_status_panel(
    frame: np.ndarray,
    gesture: str,
    fps: float,
    drawing_active: bool,
    height: int,
) -> None:
    """Draw debug status panel in the top-left corner."""
    lines = [
        f"Gesture: {gesture}",
        f"FPS: {fps:.1f}",
        f"Drawing: {'ON' if drawing_active else 'OFF'}",
    ]
    for i, line in enumerate(lines):
        y = 25 + i * 22
        draw_text_with_background(
            frame, line, (10, y),
            text_color=(255, 255, 255) if i > 0 else (0, 255, 0),
        )
