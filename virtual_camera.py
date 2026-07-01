"""Virtual camera output via v4l2loopback using ffmpeg subprocess."""

import atexit
import os
import subprocess
import time

import numpy as np


class VirtualCamera:
    """Writes processed frames to a v4l2loopback device via ffmpeg.

    Pipeline: Python BGR frames -> ffmpeg stdin -> v4l2 yuyv422 -> /dev/videoN

    Cleanup is handled via atexit to prevent orphaned ffmpeg processes
    that would lock the v4l2loopback device.
    """

    def __init__(
        self,
        device: str = "/dev/video13",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        pix_fmt: str = "yuyv422",
        verbose: bool = False,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.verbose = verbose
        self.process: subprocess.Popen | None = None
        self._started = False
        self._cleaned_up = False

    def start(self) -> None:
        """Launch ffmpeg subprocess and begin streaming.

        Raises:
            RuntimeError: If ffmpeg fails to start.
        """
        if self._started:
            return

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", "pipe:0",
            "-f", "v4l2",
            "-pix_fmt", self.pix_fmt,
            self.device,
        ]

        stderr_dest = None if self.verbose else subprocess.DEVNULL

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_dest,
                preexec_fn=os.setsid,  # dies when parent dies
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg:\n"
                "  sudo apt install ffmpeg"
            )
        except PermissionError:
            raise RuntimeError(
                f"Cannot access {self.device}. Ensure you have permissions:\n"
                f"  sudo chmod 666 {self.device}\n"
                f"Or add yourself to the video group:\n"
                f"  sudo usermod -aG video $USER"
            )

        self._started = True
        self._cleaned_up = False

        # Register cleanup (atexit handles normal exit and unhandled exceptions)
        atexit.register(self._atexit_cleanup)

        # Brief pause to let ffmpeg initialize
        time.sleep(0.5)

        if self.process.poll() is not None:
            raise RuntimeError(
                f"ffmpeg exited immediately (code {self.process.returncode}). "
                f"Is {self.device} available? Try:\n"
                f"  sudo modprobe v4l2loopback video_nr=13 card_label='Draw on Cam' "
                f"exclusive_caps=1"
            )

        print(f"[INFO] Virtual camera started on {self.device}")
        print(f"[INFO] Output: {self.width}x{self.height} @ {self.fps}fps "
              f"({self.pix_fmt})")

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a BGR frame to the virtual camera.

        Args:
            frame: BGR numpy array (H, W, 3), uint8.

        Raises:
            BrokenPipeError: If ffmpeg has exited unexpectedly.
        """
        if not self._started or self.process is None:
            return

        if self.process.poll() is not None:
            raise BrokenPipeError(
                f"ffmpeg process died unexpectedly (code {self.process.returncode})"
            )

        # Ensure C-contiguous memory layout for tobytes()
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except BrokenPipeError:
            self._started = False
            raise BrokenPipeError(
                "ffmpeg stdin closed unexpectedly. The virtual camera may have been "
                "disconnected or the device may be in use by another application."
            )

    def stop(self) -> None:
        """Gracefully stop the virtual camera output."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self.process is None:
            return

        if self.process.poll() is None:
            # ffmpeg is still running — close stdin to signal end of stream
            try:
                self.process.stdin.close()
            except Exception:
                pass

            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't exit gracefully
                self.process.kill()
                self.process.wait(timeout=2)
                print("[WARN] ffmpeg was forcefully killed after timeout")

        self.process = None
        self._started = False

    def is_alive(self) -> bool:
        """Check if ffmpeg is still running."""
        return (
            self._started
            and self.process is not None
            and self.process.poll() is None
        )

    def _atexit_cleanup(self) -> None:
        """Cleanup handler for atexit."""
        self.stop()
