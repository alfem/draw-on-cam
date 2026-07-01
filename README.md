# Draw on Cam

Draw in real time on your webcam using hand gestures. The output is streamed as a virtual camera (v4l2loopback) for use in Teams, Zoom, OBS, or any video conferencing app.

## How it works

The app captures your webcam feed, detects your hand using MediaPipe, and overlays lines and strokes you draw with your fingers. The result is output as a virtual camera you can select in any app.

### Gestures

| Gesture | Action |
|---|---|
| Pinch thumb+index (fingertips together, other fingers curled) | **Draw** — the stroke follows the midpoint between both fingers |
| Open palm (all fingers extended) | **Erase** — clears strokes inside the green circle |
| Any other position / no hand | No action |

When you separate your fingertips, the stroke stops instantly.

### Keyboard shortcuts (preview window)

| Key | Action |
|---|---|
| `q` | Quit |
| `c` | Clear all drawings |
| `b` | Select a background image (enables PiP mode) |
| `p` | Toggle PiP on/off (when background is loaded) |
| `x` | Clear background, return to full webcam |
| `h` | Toggle hand landmarks (debug) |

## Requirements

- Python 3.10+
- v4l2loopback (`sudo apt install v4l2loopback-dkms`)
- ffmpeg (`sudo apt install ffmpeg`)
- Webcam

## Installation

```bash
cd draw-on-cam/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The MediaPipe model is downloaded automatically. To download it manually:

```bash
mkdir -p models
wget -O models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### Set up the virtual camera

```bash
# Load the module with exclusive_caps=1 (required for Teams/Zoom)
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=13 card_label="Draw on Cam" exclusive_caps=1

# Lock the output format
v4l2-ctl -d /dev/video13 -c keep_format=1

# Set permissions
sudo chmod 666 /dev/video13
```

Verify it works:

```bash
v4l2-ctl -d /dev/video13 --info
# Should show "Video Capture" in the capabilities
```

## Usage

```bash
source venv/bin/activate
python main.py
```

### Selecting a camera

```bash
# List available cameras
python main.py --list-cameras

# Select by index
python main.py --camera 0
python main.py --camera 1

# Select by device path
python main.py --camera /dev/video2
```

### Mouse mode

```bash
# Use mouse instead of hand gestures
python main.py --mouse
```

| Mouse button | Action |
|---|---|
| Left button drag | Draw |
| Right button drag | Erase |

### Options

```
python main.py [options]

  --camera 0|/dev/videoN   Camera: index (0, 1, ...) or path (default: /dev/video0)
  --list-cameras            List detected cameras and exit
  --output /dev/video13     Virtual camera device (default: /dev/video13)
  --width 640               Frame width (default: 640)
  --height 480              Frame height (default: 480)
  --draw-color red          Stroke color: red, green, blue, yellow, cyan, magenta, white, black
  --thickness 4             Stroke thickness in pixels (default: 4)
  --smoothing 0.4           Stroke smoothing 0.05–1.0 (default: 0.4, lower = smoother)
  --eraser-radius 60        Eraser radius in pixels (default: 60)
  --mouse                   Use mouse instead of hand gestures (left=draw, right=erase)
  --background PATH          Load a background image on startup (enables PiP)
  --fps-skip 1              Process gesture every N frames (2+ for better performance)
  --no-preview              Disable preview window
  --no-output               No virtual camera output (preview only)
  --no-flip                 Do not mirror the preview horizontally
  --no-drawing              Disable drawing (debug)
  --no-erase                Disable erasing (debug)
```

### Examples

```bash
# Thin blue stroke with heavy smoothing
python main.py --draw-color blue --thickness 2 --smoothing 0.2

# Thick green stroke, large eraser
python main.py --draw-color green --thickness 8 --eraser-radius 100

# Load a background image at startup (slide/whiteboard to annotate)
python main.py --background /path/to/slide.png

# Preview only, no virtual camera
python main.py --no-output

# High resolution (more CPU load)
python main.py --width 1280 --height 720

# Mouse mode instead of gestures
python main.py --mouse

# Secondary camera with responsive strokes
python main.py --camera 1 --smoothing 0.7
```

### Picture-in-picture mode

Press `b` while the app is running to select a background image. The webcam will shrink to a small window in the bottom-right corner, and the image will fill the screen. You can draw annotations on top of everything.

- `b` — select a background image (opens file dialog)
- `p` — toggle the webcam PiP on/off
- `x` — clear the background and return to full webcam

## Verifying the virtual camera

While the app is running, in another terminal:

```bash
ffplay /dev/video13
```

In Teams or Zoom, select **"Draw on Cam"** as your camera. If it doesn't appear, restart the video conferencing app (it caches the device list on startup).

## Troubleshooting

### Camera won't open

```bash
sudo usermod -aG video $USER
# Log out and back in
```

### Virtual camera doesn't show in Teams/Zoom

```bash
# Reload v4l2loopback with exclusive_caps=1
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=13 card_label="Draw on Cam" exclusive_caps=1
v4l2-ctl -d /dev/video13 -c keep_format=1
```

### ffmpeg exits immediately

```bash
# Check if the device is in use
sudo fuser /dev/video13

# Kill the process using it (if safe)
sudo fuser -k /dev/video13
```

### Low performance / FPS drops

```bash
# Process gestures every other frame
python main.py --fps-skip 2

# Lower resolution
python main.py --width 640 --height 480

# Disable preview
python main.py --no-preview
```

### Stroke is too jittery

```bash
# Increase smoothing
python main.py --smoothing 0.2
```

### Stroke lags behind

```bash
# Reduce smoothing
python main.py --smoothing 0.7
```

## Project structure

```
draw-on-cam/
├── main.py              # Main loop and orchestration
├── config.py            # Configuration and CLI parsing
├── gesture_detector.py  # MediaPipe Hands + gesture classification
├── drawing_canvas.py    # RGBA canvas, polylines, eraser
├── virtual_camera.py    # v4l2loopback output via ffmpeg
├── utils.py             # FPSMeter, debug helpers
├── models/
│   └── hand_landmarker.task  # MediaPipe model
├── requirements.txt     # Python dependencies
└── README.md
```
