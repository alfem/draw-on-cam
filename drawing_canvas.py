"""Drawing canvas with RGBA overlay and polyline management."""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Polyline:
    """A single continuous stroke."""
    points: list[tuple[int, int]] = field(default_factory=list)
    color: tuple[int, int, int] = (0, 0, 255)  # BGR
    thickness: int = 4


class DrawingCanvas:
    """Manages drawing state: polylines, erasing, and compositing.

    The canvas is an RGBA image painted additively as the user draws.
    Lines are stored as polylines so they can be selectively erased.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        # RGBA canvas: all channels zero = fully transparent
        self._canvas = np.zeros((height, width, 4), dtype=np.uint8)
        self._lines: list[Polyline] = []
        self._active_line: Optional[Polyline] = None
        self._prev_point: Optional[tuple[int, int]] = None

        # Limits to prevent unbounded memory growth
        self._max_lines = 500
        self._max_total_points = 50000

    @property
    def line_count(self) -> int:
        return len(self._lines)

    @property
    def total_points(self) -> int:
        return sum(len(line.points) for line in self._lines)

    def add_point(
        self,
        x: int,
        y: int,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 4,
    ) -> None:
        """Add a point to the current stroke, drawing a line segment.

        If no active stroke exists, starts a new one.
        """
        if self._active_line is None:
            self._active_line = Polyline(color=color, thickness=thickness)
            self._lines.append(self._active_line)
            self._prev_point = None

        self._active_line.points.append((x, y))

        # Draw segment from previous point
        if self._prev_point is not None:
            cv2.line(
                self._canvas,
                self._prev_point,
                (x, y),
                (*color, 255),  # RGBA
                thickness,
                cv2.LINE_AA,
            )
        else:
            # First point: draw a small dot
            cv2.circle(
                self._canvas,
                (x, y),
                thickness // 2,
                (*color, 255),
                -1,
                cv2.LINE_AA,
            )

        self._prev_point = (x, y)

        # Enforce memory limits
        self._enforce_limits()

    def end_stroke(self) -> None:
        """Finalize the current stroke. Next add_point starts a new line."""
        self._active_line = None
        self._prev_point = None

    def erase_at(self, x: int, y: int, radius: int) -> bool:
        """Erase polyline points within `radius` of (x, y).

        Polylines with erased points are split at gaps > 10px.
        Polylines with fewer than 2 points are removed.

        Returns True if anything was erased.
        """
        if not self._lines:
            return False

        radius_sq = radius * radius
        new_lines: list[Polyline] = []
        erased_any = False

        for line in self._lines:
            kept = [
                pt for pt in line.points
                if (pt[0] - x) ** 2 + (pt[1] - y) ** 2 > radius_sq
            ]

            if len(kept) < len(line.points):
                erased_any = True

            # Split at gaps and keep valid segments
            segments = self._split_at_gaps(kept, max_gap_px=10)
            for seg_points in segments:
                if len(seg_points) >= 2:
                    new_lines.append(Polyline(
                        points=seg_points,
                        color=line.color,
                        thickness=line.thickness,
                    ))

        self._lines = new_lines
        self._active_line = None
        self._prev_point = None

        # Rebuild canvas from remaining lines
        if erased_any:
            self._rebuild_canvas()

        return erased_any

    def _split_at_gaps(
        self, points: list[tuple[int, int]], max_gap_px: int = 10
    ) -> list[list[tuple[int, int]]]:
        """Split a list of points into segments where gaps exceed max_gap_px."""
        if not points:
            return []

        segments: list[list[tuple[int, int]]] = []
        current_segment = [points[0]]

        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            if dx * dx + dy * dy > max_gap_px * max_gap_px:
                # Gap detected: start new segment
                segments.append(current_segment)
                current_segment = [points[i]]
            else:
                current_segment.append(points[i])

        if current_segment:
            segments.append(current_segment)

        return segments

    def _rebuild_canvas(self) -> None:
        """Fully rebuild the RGBA canvas from the current polylines."""
        self._canvas.fill(0)
        for line in self._lines:
            if len(line.points) < 2:
                # Single point: draw a dot
                pt = line.points[0]
                cv2.circle(
                    self._canvas, pt, line.thickness // 2,
                    (*line.color, 255), -1, cv2.LINE_AA,
                )
                continue

            for i in range(1, len(line.points)):
                cv2.line(
                    self._canvas,
                    line.points[i - 1],
                    line.points[i],
                    (*line.color, 255),
                    line.thickness,
                    cv2.LINE_AA,
                )

    def _enforce_limits(self) -> None:
        """Drop oldest lines if memory limits are exceeded."""
        while len(self._lines) > self._max_lines:
            self._lines.pop(0)
        # More aggressive: drop oldest if total points exceed limit
        while self.total_points > self._max_total_points and self._lines:
            self._lines.pop(0)

    def render_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Composite the drawing canvas onto a BGR frame using alpha blending.

        Args:
            frame: BGR image (H, W, 3) — the webcam background.

        Returns:
            BGR image with drawings composited on top.
        """
        canvas_rgb = self._canvas[:, :, :3]
        canvas_alpha = self._canvas[:, :, 3].astype(np.float32) / 255.0
        alpha_3ch = np.stack([canvas_alpha] * 3, axis=-1)

        result = (
            canvas_rgb.astype(np.float32) * alpha_3ch
            + frame.astype(np.float32) * (1.0 - alpha_3ch)
        ).astype(np.uint8)

        return result

    def clear_all(self) -> None:
        """Reset the canvas to fully transparent."""
        self._canvas.fill(0)
        self._lines.clear()
        self._active_line = None
        self._prev_point = None
