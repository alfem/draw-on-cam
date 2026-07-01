#!/usr/bin/env python3
"""Draw on Cam — Real-time drawing with hand gestures or mouse.

Pinch thumb+index to draw on the webcam feed.
Open your palm to erase nearby lines.
Or use --mouse for mouse mode (left=draw, right=erase).

The output is streamed to a virtual camera (v4l2loopback) for use in
video conferencing apps like Teams, Zoom, etc.

Usage:
    python main.py
    python main.py --camera 1 --width 1280 --height 720
    python main.py --no-preview --draw-color blue
    python main.py --mouse
    python main.py --no-output  # Preview only, no virtual camera
"""

import sys
import time

import cv2
import numpy as np

from config import Config
from drawing_canvas import DrawingCanvas
from gesture_detector import GestureDetector, GestureResult
from utils import FPSMeter, draw_status_panel, load_image_fill
from virtual_camera import VirtualCamera


class DrawOnCam:
    """Main application orchestrator.

    Coordinates webcam capture, input (gestures or mouse),
    drawing canvas, and virtual camera output in a real-time loop.
    """

    def __init__(self, config: Config):
        self.config = config
        self.cap: cv2.VideoCapture | None = None
        self.gesture_detector = GestureDetector(config)
        self.drawing_canvas = DrawingCanvas(config.output_height, config.output_width)
        self.virtual_camera: VirtualCamera | None = None
        self.fps_meter = FPSMeter()

        # State
        self.current_gesture = "none"
        self.drawing_active = False
        self.frame_count = 0
        self.running = False
        self._last_gesture_result = GestureResult()
        self._show_landmarks = False

        # Mouse state
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_left_down = False
        self._mouse_right_down = False

        # Background image / PiP state
        self._background: np.ndarray | None = None
        self._pip_enabled: bool = False

    def run(self) -> None:
        """Start the main processing loop."""
        # --- Initialize camera ---
        self.cap = self._init_camera()

        # --- Initialize virtual camera ---
        if self.config.enable_output:
            self.virtual_camera = VirtualCamera(
                device=self.config.output_device,
                width=self.config.output_width,
                height=self.config.output_height,
                fps=self.config.output_fps,
            )
            try:
                self.virtual_camera.start()
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                print("[INFO] Continuing without virtual camera output...")
                self.virtual_camera = None

        # --- Controls info ---
        if self.config.use_mouse:
            print("[INFO] Controls (mouse mode):")
            print("  - Left button drag to DRAW")
            print("  - Right button drag to ERASE")
        else:
            print("[INFO] Controls (gesture mode):")
            print("  - Pinch thumb+index fingers together to DRAW")
            print("  - Open palm to ERASE")
        print("  - Press 'q' in preview window to quit")
        print("  - Press 'c' to clear all drawings")
        print("  - Press 'b' to select a background image (PiP mode)")
        print("  - Press 'x' to clear background")
        print()

        # Load background from CLI if provided
        if self.config.background_image:
            self._load_background(self.config.background_image)

        # --- Create preview window (disable QT toolbar/context menu) ---
        if self.config.display_preview:
            cv2.namedWindow("Draw on Cam", cv2.WINDOW_GUI_NORMAL)
            if self.config.use_mouse:
                cv2.setMouseCallback("Draw on Cam", self._mouse_callback)

        self.running = True

        try:
            while self.running:
                # --- Frame capture ---
                ret, frame = self.cap.read()
                if not ret:
                    print("[WARN] Frame capture failed, retrying...")
                    time.sleep(0.01)
                    continue

                # Resize to output dimensions if needed
                if frame.shape[0] != self.config.output_height or \
                   frame.shape[1] != self.config.output_width:
                    frame = cv2.resize(
                        frame,
                        (self.config.output_width, self.config.output_height),
                    )

                self.fps_meter.tick()
                self.frame_count += 1

                # --- Input: gestures or mouse ---
                if self.config.use_mouse:
                    self._handle_mouse()
                else:
                    if self.frame_count % self.config.process_every_n_frames == 0:
                        self._last_gesture_result = self.gesture_detector.detect(frame)
                    self.current_gesture = self._last_gesture_result.gesture
                    self._handle_gesture(self._last_gesture_result)

                # --- Compose base frame (background + optional PiP) ---
                if self._background is not None and self._pip_enabled:
                    base = self._background.copy()
                    # Place webcam as PiP in bottom-right corner
                    pip_w = int(self.config.output_width * self.config.pip_scale)
                    pip_h = int(pip_w * frame.shape[0] / frame.shape[1])
                    pip_frame = cv2.resize(frame, (pip_w, pip_h))
                    margin = 10
                    px = self.config.output_width - pip_w - margin
                    py = self.config.output_height - pip_h - margin
                    base[py:py + pip_h, px:px + pip_w] = pip_frame
                    # White border around PiP
                    cv2.rectangle(base, (px - 2, py - 2),
                                  (px + pip_w + 2, py + pip_h + 2),
                                  (255, 255, 255), 2)
                    output = self.drawing_canvas.render_to_frame(base)
                elif self._background is not None:
                    # Background only (no PiP) — useful for drawing on whiteboard/slides
                    output = self.drawing_canvas.render_to_frame(self._background.copy())
                else:
                    output = self.drawing_canvas.render_to_frame(frame)

                # --- Overlays (gesture indicators or mouse cursor) ---
                if self.config.use_mouse:
                    self._draw_mouse_overlay(output)
                else:
                    self._draw_overlays(output, self._last_gesture_result)

                # --- Write to virtual camera ---
                if self.virtual_camera and self.virtual_camera.is_alive():
                    try:
                        self.virtual_camera.write_frame(output)
                    except BrokenPipeError as e:
                        print(f"[WARN] Virtual camera: {e}")
                        self.virtual_camera = None

                # --- Preview window: flip for mirror effect, draw text on top ---
                # Skip mirror flip when background is active (backgrounds aren't mirrors)
                if self.config.display_preview:
                    if self.config.flip_horizontal and self._background is None:
                        preview = cv2.flip(output, 1)
                    else:
                        preview = output
                    # Draw text AFTER flip so it's readable
                    draw_status_panel(
                        preview,
                        self.current_gesture,
                        self.fps_meter.fps,
                        self.drawing_active,
                        self.config.output_height,
                    )
                    cv2.imshow("Draw on Cam", preview)
                    key = cv2.waitKey(1) & 0xFF
                    # Handle window close (X button)
                    try:
                        window_open = (
                            cv2.getWindowProperty("Draw on Cam", cv2.WND_PROP_VISIBLE) >= 1
                        )
                    except cv2.error:
                        window_open = False
                    if key == ord("q") or not window_open:
                        self.running = False
                    elif key == ord("c"):
                        self.drawing_canvas.clear_all()
                        print("[INFO] Canvas cleared")
                    elif key == ord("h"):
                        self._show_landmarks = not self._show_landmarks
                        print(f"[INFO] Hand landmarks: "
                              f"{'ON' if self._show_landmarks else 'OFF'}")
                    elif key == ord("b"):
                        path = self._pick_background_file()
                        if path:
                            self._load_background(path)
                    elif key == ord("p"):
                        if self._background is not None:
                            self._pip_enabled = not self._pip_enabled
                            print(f"[INFO] PiP: {'ON' if self._pip_enabled else 'OFF'}")
                    elif key == ord("x"):
                        self._background = None
                        self._pip_enabled = False
                        print("[INFO] Background cleared")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            raise
        finally:
            self._cleanup()

    def _init_camera(self) -> cv2.VideoCapture:
        """Initialize the webcam with optimal settings."""
        cap = cv2.VideoCapture(self.config.camera_device, cv2.CAP_V4L2)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera: {self.config.camera_device}")
            print("[INFO] Check permissions: sudo usermod -aG video $USER")
            print("[INFO] Use --list-cameras to see available cameras:")
            Config._list_cameras()
            sys.exit(1)

        # Request MJPG format for better performance
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])

        print(f"[INFO] Camera opened: {self.config.camera_device}")
        print(f"[INFO] Resolution: {actual_w}x{actual_h} @ {actual_fps:.1f}fps "
              f"({fourcc_str})")

        return cap

    def _handle_gesture(self, result: GestureResult) -> None:
        """Map gesture classification to drawing/erasing actions."""
        if result.gesture == "pinch" and result.draw_point:
            x, y = result.draw_point
            if self.config.enable_drawing:
                self.drawing_active = True
                self.drawing_canvas.add_point(
                    x, y,
                    color=self.config.drawing_color,
                    thickness=self.config.drawing_thickness,
                )

        elif result.gesture == "palm" and result.palm_center:
            x, y = result.palm_center
            if self.config.enable_erase:
                if self.drawing_active:
                    self.drawing_canvas.end_stroke()
                    self.drawing_active = False
                self.drawing_canvas.erase_at(x, y, self.config.eraser_radius)

        else:
            if self.drawing_active:
                self.drawing_canvas.end_stroke()
                self.drawing_active = False

    def _draw_overlays(self, frame: np.ndarray, result: GestureResult) -> None:
        """Draw gesture feedback overlays on the output frame."""
        if self._show_landmarks and result.landmarks:
            self.gesture_detector.draw_landmarks(frame, result.landmarks)

        if result.gesture == "pinch" and result.landmarks and result.draw_point:
            self.gesture_detector.draw_pinch_indicator(
                frame, result.landmarks, self.config.drawing_color,
                self.config.output_width, self.config.output_height,
            )
            x, y = result.draw_point
            cv2.circle(frame, (x, y), 3, self.config.drawing_color, -1, cv2.LINE_AA)

        if result.gesture == "palm" and result.palm_center:
            self.gesture_detector.draw_eraser_indicator(
                frame, result.palm_center,
                self.config.eraser_radius,
                self.config.eraser_color,
            )

    # --- Mouse input ---

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        """OpenCV mouse callback — tracks button state and position."""
        self._mouse_x = x
        self._mouse_y = y
        self._mouse_left_down = (flags & cv2.EVENT_FLAG_LBUTTON) != 0
        self._mouse_right_down = (flags & cv2.EVENT_FLAG_RBUTTON) != 0

    def _handle_mouse(self) -> None:
        """Process mouse input: draw with left button, erase with right."""
        mx, my = self._mouse_x, self._mouse_y
        w = self.config.output_width

        # Mouse coords are in preview (flipped) space.
        # Convert to unflipped output space for drawing.
        if self.config.flip_horizontal:
            x, y = w - mx - 1, my
        else:
            x, y = mx, my

        if self._mouse_left_down and self.config.enable_drawing:
            self.drawing_active = True
            self.current_gesture = "pinch"
            self.drawing_canvas.add_point(
                x, y,
                color=self.config.drawing_color,
                thickness=self.config.drawing_thickness,
            )
        elif self._mouse_right_down and self.config.enable_erase:
            if self.drawing_active:
                self.drawing_canvas.end_stroke()
                self.drawing_active = False
            self.current_gesture = "palm"
            self.drawing_canvas.erase_at(x, y, self.config.eraser_radius)
        else:
            if self.drawing_active:
                self.drawing_canvas.end_stroke()
                self.drawing_active = False
            self.current_gesture = "none"

    def _draw_mouse_overlay(self, frame: np.ndarray) -> None:
        """Draw mouse cursor indicator on the unflipped output frame."""
        mx, my = self._mouse_x, self._mouse_y
        w, h = self.config.output_width, self.config.output_height

        # Convert mouse coords (preview space) to unflipped output space
        if self.config.flip_horizontal:
            x, y = w - mx - 1, my
        else:
            x, y = mx, my

        if x < 0 or x >= w or y < 0 or y >= h:
            return

        if self._mouse_left_down:
            cv2.drawMarker(frame, (x, y), self.config.drawing_color,
                           cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        elif self._mouse_right_down:
            self.gesture_detector.draw_eraser_indicator(
                frame, (x, y), self.config.eraser_radius, self.config.eraser_color)
        else:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)

    # --- Background / PiP ---

    def _load_background(self, path: str) -> None:
        """Load and resize a background image, and enable PiP mode."""
        try:
            self._background = load_image_fill(
                path, self.config.output_width, self.config.output_height
            )
            self._pip_enabled = True
            print(f"[INFO] Background loaded: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load background: {e}")

    @staticmethod
    def _pick_background_file() -> str | None:
        """Open a file dialog to pick a background image.

        Uses zenity (Ubuntu/GNOME) if available, falls back to tkinter.
        """
        import subprocess
        # Try zenity first (no extra dependencies on Ubuntu)
        try:
            result = subprocess.run(
                ["zenity", "--file-selection",
                 "--title=Select background image",
                 "--file-filter=Images (*.png *.jpg *.jpeg *.bmp) | *.png *.jpg *.jpeg *.bmp"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback to tkinter
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Select background image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")],
            )
            root.destroy()
            if path:
                return path
        except Exception:
            pass

        print("[ERROR] No file dialog available. Use --background PATH instead.")
        return None

    def _cleanup(self) -> None:
        """Release all resources gracefully."""
        print("[INFO] Shutting down...")

        if self.virtual_camera:
            self.virtual_camera.stop()
            print("[INFO] Virtual camera stopped")

        if self.cap:
            self.cap.release()
            print("[INFO] Camera released")

        self.gesture_detector.close()

        cv2.destroyAllWindows()
        print("[INFO] Goodbye!")


def main():
    config = Config.from_args()
    app = DrawOnCam(config)
    app.run()


if __name__ == "__main__":
    main()
