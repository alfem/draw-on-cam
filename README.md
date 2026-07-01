# Draw on Cam

Dibuja en tiempo real sobre tu webcam usando gestos de la mano. La salida se emite como cámara virtual (v4l2loopback) para usarla en Teams, Zoom, OBS o cualquier aplicación de videoconferencia.

## Cómo funciona

La aplicación captura el vídeo de tu webcam, detecta la mano mediante MediaPipe y superpone líneas y trazos que dibujas con los dedos. El resultado se emite como una cámara virtual que puedes seleccionar en cualquier app.

### Gestos

| Gesto | Acción |
|---|---|
| Pinza pulgar+índice (yemas juntas, resto de dedos cerrados) | **Dibujar** — el trazo sigue el punto medio entre ambos dedos |
| Palma abierta (todos los dedos extendidos) | **Borrar** — elimina los trazos dentro del círculo verde |
| Cualquier otra posición / sin mano | No dibuja ni borra |

Al separar los dedos de la pinza, el trazo se corta al instante.

### Atajos de teclado en la ventana de preview

| Tecla | Acción |
|---|---|
| `q` | Salir |
| `c` | Limpiar todos los dibujos |
| `h` | Mostrar/ocultar landmarks de la mano (debug) |

## Requisitos

- Python 3.10+
- v4l2loopback (`sudo apt install v4l2loopback-dkms`)
- ffmpeg (`sudo apt install ffmpeg`)
- Cámara web

## Instalación

```bash
cd draw-on-cam/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

El modelo de MediaPipe se descarga automáticamente. Si necesitas descargarlo manualmente:

```bash
mkdir -p models
wget -O models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### Configurar la cámara virtual

```bash
# Cargar el módulo con exclusive_caps=1 (necesario para Teams/Zoom)
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=13 card_label="Draw on Cam" exclusive_caps=1

# Fijar el formato de salida
v4l2-ctl -d /dev/video13 -c keep_format=1

# Dar permisos
sudo chmod 666 /dev/video13
```

Verifica que funciona:

```bash
v4l2-ctl -d /dev/video13 --info
# Debe mostrar "Video Capture" en las capabilities
```

## Uso

```bash
source venv/bin/activate
python main.py
```

### Seleccionar cámara

```bash
# Listar cámaras disponibles
python main.py --list-cameras

# Elegir por índice
python main.py --camera 0
python main.py --camera 1

# Elegir por ruta de dispositivo
python main.py --camera /dev/video2
```

### Parámetros

```
python main.py [opciones]

  --camera 0|/dev/videoN   Cámara: índice (0, 1, ...) o ruta (default: /dev/video0)
  --list-cameras            Lista las cámaras detectadas y sale
  --output /dev/video13     Dispositivo de cámara virtual (default: /dev/video13)
  --width 640               Ancho del frame (default: 640)
  --height 480              Alto del frame (default: 480)
  --draw-color red          Color del trazo: red, green, blue, yellow, cyan, magenta, white, black
  --thickness 4             Grosor de línea en píxeles (default: 4)
  --smoothing 0.4           Suavizado del trazo 0.05–1.0 (default: 0.4, menos = más suave)
  --eraser-radius 60        Radio del borrador en píxeles (default: 60)
  --fps-skip 1              Procesar gesto cada N frames (2+ para mejor rendimiento)
  --no-preview              Sin ventana de preview
  --no-output               Sin cámara virtual (solo preview)
  --no-flip                 No reflejar el preview horizontalmente
  --no-drawing              Desactivar dibujo (debug)
  --no-erase                Desactivar borrado (debug)
```

### Ejemplos

```bash
# Trazo fino azul con mucho suavizado
python main.py --draw-color blue --thickness 2 --smoothing 0.2

# Trazo grueso verde, borrador grande
python main.py --draw-color green --thickness 8 --eraser-radius 100

# Solo preview, sin cámara virtual
python main.py --no-output

# Alta resolución (más carga de CPU)
python main.py --width 1280 --height 720

# Cámara secundaria con poco suavizado (trazo más reactivo)
python main.py --camera 1 --smoothing 0.7
```

## Verificar la cámara virtual

Mientras la app está corriendo, en otra terminal:

```bash
ffplay /dev/video13
```

En Teams o Zoom, selecciona **"Draw on Cam"** como cámara. Si no aparece, reinicia la aplicación de videoconferencia (cachea la lista de dispositivos al arrancar).

## Solución de problemas

### La cámara no se abre

```bash
sudo usermod -aG video $USER
# Cerrar sesión y volver a entrar
```

### La cámara virtual no aparece en Teams/Zoom

```bash
# Recargar v4l2loopback con exclusive_caps=1
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=13 card_label="Draw on Cam" exclusive_caps=1
v4l2-ctl -d /dev/video13 -c keep_format=1
```

### ffmpeg sale inmediatamente

```bash
# Verificar que el dispositivo no está en uso
sudo fuser /dev/video13

# Matar el proceso que lo ocupa (si es seguro)
sudo fuser -k /dev/video13
```

### Rendimiento bajo / FPS bajos

```bash
# Procesar gestos en frames alternos
python main.py --fps-skip 2

# Reducir resolución
python main.py --width 640 --height 480

# Desactivar preview
python main.py --no-preview
```

### El trazo tiembla mucho

```bash
# Aumentar el suavizado
python main.py --smoothing 0.2
```

### El trazo va con mucho retraso

```bash
# Reducir el suavizado
python main.py --smoothing 0.7
```

## Estructura del proyecto

```
draw-on-cam/
├── main.py              # Bucle principal, orquestación
├── config.py            # Configuración y CLI
├── gesture_detector.py  # MediaPipe Hands + clasificación de gestos
├── drawing_canvas.py    # Lienzo RGBA, polilíneas, borrado
├── virtual_camera.py    # Salida v4l2loopback vía ffmpeg
├── utils.py             # FPSMeter, helpers
├── models/
│   └── hand_landmarker.task  # Modelo MediaPipe
├── requirements.txt     # Dependencias Python
└── README.md
```
