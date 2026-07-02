"""Configuration for draw-on-cam application."""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Config:
    # Camera input
    camera_device: str = "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # Virtual camera output
    output_device: str = "/dev/video13"
    output_width: int = 640
    output_height: int = 480
    output_fps: int = 30
    output_pix_fmt: str = "yuyv422"

    # Gesture detection (MediaPipe)
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 1
    model_path: str | None = None  # Auto-detected: models/hand_landmarker.task

    # Gesture classification thresholds
    gesture_confirmation_window: int = 3
    palm_min_extended_fingers: int = 4

    # Drawing
    drawing_color: tuple = field(default_factory=lambda: (0, 0, 255))  # Red in BGR
    drawing_thickness: int = 4
    smoothing: float = 0.4  # EMA smoothing (0 = max smooth, 1 = no smoothing)
    eraser_radius: int = 60
    eraser_color: tuple = field(default_factory=lambda: (0, 255, 0))  # Green indicator

    # Performance
    display_preview: bool = True
    flip_horizontal: bool = True
    enable_drawing: bool = True
    enable_erase: bool = True
    enable_output: bool = True
    use_mouse: bool = False  # Mouse mode: left button=draw, right button=erase
    flip_output: bool = False  # Flip virtual camera output (for Teams self-view)
    pip_scale: float = 0.33  # Webcam PiP width as fraction of output width
    background_image: str | None = None  # Path to background image (also settable at runtime)
    process_every_n_frames: int = 1  # 1 = every frame, 2 = every other, etc.

    @classmethod
    def from_args(cls) -> "Config":
        parser = argparse.ArgumentParser(
            description="Draw on webcam with hand gestures. "
                        "Pinch thumb+index to draw, open palm to erase. "
                        "Use --mouse for mouse mode."
        )
        parser.add_argument(
            "--camera", default=cls.camera_device,
            help=f"Camera: device path (/dev/video0) or index from --list-cameras "
                 f"(default: {cls.camera_device})"
        )
        parser.add_argument(
            "--list-cameras", action="store_true",
            help="List available cameras and exit"
        )
        parser.add_argument(
            "--output", default=cls.output_device,
            help=f"Virtual camera output device (default: {cls.output_device})"
        )
        parser.add_argument(
            "--width", type=int, default=cls.camera_width,
            help=f"Frame width (default: {cls.camera_width})"
        )
        parser.add_argument(
            "--height", type=int, default=cls.camera_height,
            help=f"Frame height (default: {cls.camera_height})"
        )
        parser.add_argument(
            "--no-preview", action="store_true",
            help="Disable preview window"
        )
        parser.add_argument(
            "--no-drawing", action="store_true",
            help="Disable drawing (debug mode)"
        )
        parser.add_argument(
            "--no-erase", action="store_true",
            help="Disable erasing (debug mode)"
        )
        parser.add_argument(
            "--no-flip", action="store_true",
            help="Do not mirror the webcam horizontally"
        )
        parser.add_argument(
            "--no-output", action="store_true",
            help="Do not write to virtual camera (preview only)"
        )
        parser.add_argument(
            "--mouse", action="store_true",
            help="Use mouse instead of hand gestures (left=draw, right=erase)"
        )
        parser.add_argument(
            "--background", default=None,
            help="Path to background image (enables PiP mode)"
        )
        parser.add_argument(
            "--flip-output", action="store_true",
            help="Flip virtual camera output (corrects Teams self-view mirror)"
        )
        parser.add_argument(
            "--draw-color", default="red",
            choices=["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"],
            help="Drawing color (default: red)"
        )
        parser.add_argument(
            "--thickness", type=int, default=cls.drawing_thickness,
            help=f"Drawing line thickness (default: {cls.drawing_thickness})"
        )
        parser.add_argument(
            "--smoothing", type=float, default=cls.smoothing,
            help=f"Stroke smoothing 0.05–1.0 (default: {cls.smoothing}, lower=smoother)"
        )
        parser.add_argument(
            "--eraser-radius", type=int, default=cls.eraser_radius,
            help=f"Eraser radius in pixels (default: {cls.eraser_radius})"
        )
        parser.add_argument(
            "--fps-skip", type=int, default=cls.process_every_n_frames,
            help=f"Process gesture every N frames for performance "
                 f"(default: {cls.process_every_n_frames})"
        )

        args = parser.parse_args()

        # Handle --list-cameras
        if args.list_cameras:
            cls._list_cameras()
            parser.exit(0)

        cfg = cls()
        # Resolve camera: accept index (0, 1, ...) or path (/dev/video0)
        cfg.camera_device = cls._resolve_camera(args.camera)
        cfg.output_device = args.output
        cfg.camera_width = args.width
        cfg.camera_height = args.height
        cfg.output_width = args.width
        cfg.output_height = args.height
        cfg.display_preview = not args.no_preview
        cfg.enable_drawing = not args.no_drawing
        cfg.enable_erase = not args.no_erase
        cfg.flip_horizontal = not args.no_flip
        cfg.enable_output = not args.no_output
        cfg.use_mouse = args.mouse
        cfg.flip_output = args.flip_output
        cfg.background_image = args.background
        cfg.drawing_thickness = args.thickness
        cfg.smoothing = max(0.05, min(1.0, args.smoothing))  # clamp
        cfg.eraser_radius = args.eraser_radius
        cfg.process_every_n_frames = args.fps_skip

        # Parse color name
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        cfg.drawing_color = color_map[args.draw_color]

        return cfg

    @staticmethod
    def _discover_cameras() -> list[tuple[str, str]]:
        """Discover available cameras using v4l2-ctl.

        Returns:
            List of (device_path, description) tuples for unique physical cameras.
            Only returns the first device per physical camera.
        """
        cameras: list[tuple[str, str]] = []
        try:
            output = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cameras

        current_name = None
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            # Lines with camera name don't start with /dev/
            if not line.startswith("/dev/"):
                # Skip v4l2loopback/platform devices
                if "platform:" in line:
                    current_name = None
                    continue
                current_name = line
            elif current_name:
                # First /dev/video* under a camera name
                dev = line.strip()
                cameras.append((dev, current_name))
                current_name = None  # Only take first device per camera

        return cameras

    @staticmethod
    def _list_cameras() -> None:
        """Print available cameras and exit."""
        cameras = Config._discover_cameras()
        if not cameras:
            print("No cameras found. Install v4l2-ctl or check your webcam connection.")
            return

        print("Available cameras:")
        for i, (dev, name) in enumerate(cameras):
            default = " (default)" if i == 0 else ""
            print(f"  {i}: {dev} — {name}{default}")
        print()
        print("Use: python main.py --camera <index>")
        print("  e.g. python main.py --camera 0")
        print("  e.g. python main.py --camera /dev/video2")

    @staticmethod
    def _resolve_camera(camera_arg: str) -> str:
        """Resolve a camera argument to a device path.

        Accepts an integer index (0-based) or a device path like /dev/video0.
        """
        # If it looks like a device path, use as-is
        if camera_arg.startswith("/dev/video"):
            return camera_arg

        # Try to parse as integer index
        try:
            index = int(camera_arg)
        except ValueError:
            print(f"[ERROR] Invalid camera: '{camera_arg}'. "
                  "Use --list-cameras to see options.")
            sys.exit(1)

        cameras = Config._discover_cameras()
        if not cameras:
            print("[ERROR] No cameras detected. "
                  "Use a full device path like --camera /dev/video0")
            sys.exit(1)

        if index < 0 or index >= len(cameras):
            print(f"[ERROR] Camera index {index} out of range (0–{len(cameras)-1}).")
            print("Available cameras:")
            for i, (dev, name) in enumerate(cameras):
                print(f"  {i}: {dev} — {name}")
            sys.exit(1)

        dev, name = cameras[index]
        print(f"[INFO] Using camera {index}: {dev} — {name}")
        return dev
