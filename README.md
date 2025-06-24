# Player Re-Identification with YOLO and Kalman Filter

This project implements a robust player tracking and re-identification system for sports videos. It uses a fine-tuned YOLO model for player detection, a Kalman filter for tracking player motion, and a two-stage matching algorithm to ensure player IDs remain persistent, even when they are occluded or temporarily leave the screen.

## Features

- **Player Detection:** High-accuracy player detection using a YOLOv8 model.
- **Motion Tracking:** Smooth and predictive tracking with individual Kalman filters for each player.
- **Robust Re-Identification:** A two-stage association mechanism handles both active tracking and re-identification of lost players.
- **Persistent IDs:** Players retain their unique ID throughout the video.

## How It Works

The system processes video frames to perform the following steps:
1.  **Detect:** A YOLO model identifies all players in the current frame.
2.  **Predict:** The Kalman filter for each existing track predicts the player's new position.
3.  **Associate & Re-Identify:** A sophisticated matching algorithm associates new detections with existing tracks.
    -   **Stage 1:** Actively tracked players are matched based on a combination of location (IoU) and appearance (color histograms).
    -   **Stage 2:** If a detection is unmatched, it's compared against temporarily lost tracks, using a much higher weight for appearance similarity to re-identify the player.
4.  **Update:** The state of each matched tracker is updated with the new detection. New trackers are created for unmatched detections, and old trackers are moved to a "lost" state if they are not seen for a while.

For a more detailed technical explanation, please see `report.md`.

## Setup and Installation

### 1. Prerequisites

- Python 3.8 or higher
- `pip` for package installation

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 3. Install Dependencies

It is recommended to use a virtual environment to avoid conflicts with other projects.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### 4. Project Structure

Make sure your project is organized as follows:

```
.
├── input/
│   └── 15sec_input_720p.mp4
├── model-task2/
│   └── best.pt
├── output/
├── main3.py
├── requirements.txt
├── README.md
└── report.md
```

- **`input/`**: Place your input video files here.
- **`model-task2/`**: Contains the fine-tuned YOLO model weights (`best.pt`).
- **`output/`**: The processed video with tracked players will be saved here.

## How to Run

Execute the `main3.py` script from your terminal. You can specify the paths for the input video, output video, and model, or use the default values.

```bash
python main3.py
```

### Command-Line Arguments

-   `--input`: Path to the input video file.
    -   Default: `input/15sec_input_720p.mp4`
-   `--output`: Path to save the processed output video.
    -   Default: `output/output_video_reid.mp4`
-   `--model`: Path to the YOLO model weights file.
    -   Default: `model-task2/best.pt`

**Example:**

```bash
python main3.py --input my_videos/game1.mp4 --output results/game1_tracked.mp4
```

## Video Output Troubleshooting

If the output video isn't playable, try these solutions:

1. **Codec Compatibility**:
   - The script automatically tries these codecs in order:
     - MP4V (MPEG-4)
     - MJPG (Motion-JPEG)
     - XVID
     - PIM1 (MPEG-1)
   - On Windows, install the K-Lite Codec Pack for broader format support

2. **DirectShow Support**:
```python
import cv2
print([x for x in dir(cv2.videoio) if x.startswith('CAP_')])  # Print available backends
```

3. **Alternative Players**:
   - Try VLC or MPV for better codec support
   - Use FFmpeg to convert if needed:
```bash
ffmpeg -i input.mp4 -c:v libx264 output.mp4
```