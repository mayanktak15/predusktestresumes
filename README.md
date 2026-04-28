# Multi-Object Detection & Persistent-ID Tracking

A production-quality pipeline for detecting and tracking multiple objects in sports/event videos using **YOLOv8** (detection) and **DeepSORT** (tracking).

---

## 🎯 Features

- **Real-time detection** using YOLOv8 (Ultralytics)
- **Persistent ID tracking** with DeepSORT (appearance + motion)
- **Trajectory visualization** — trail lines per tracked ID
- **Movement heatmap** — density map of all trajectories
- **Object-count plot** — tracked objects over time
- **Sample-frame extraction** at configurable intervals
- **Configurable** — confidence, IOU, frame-skip, resize, model size
- **Video download** — built-in yt-dlp support for YouTube URLs

---

## 📁 Project Structure

```
multi_object_tracking/
├── main.py                  # Pipeline entry point (CLI)
├── config.py                # Centralised configuration
├── detector/
│   ├── __init__.py
│   └── yolo_detector.py     # YOLOv8 detection wrapper
├── tracker/
│   ├── __init__.py
│   └── tracker.py           # DeepSORT tracking wrapper
├── utils/
│   ├── __init__.py
│   ├── video_utils.py       # Video read/write/download
│   └── visualization.py     # Drawing, heatmaps, plots
├── outputs/                 # Generated outputs
│   ├── output.mp4
│   ├── heatmap.png
│   ├── object_count.png
│   └── sample_frames/
├── requirements.txt
├── README.md
└── report.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended (CPU works but is slower)

### Install

```bash
# Clone / navigate to the project
cd multi_object_tracking

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1 — Local video file

```bash
python main.py --input path/to/your/video.mp4
```

### Option 2 — Download from YouTube

```bash
python main.py --download "https://www.youtube.com/watch?v=VIDEO_ID" --input input_video.mp4
```

### Option 3 — With custom settings

```bash
python main.py \
    --input match.mp4 \
    --output outputs/tracked.mp4 \
    --model yolov8s.pt \
    --confidence 0.4 \
    --device auto \
    --frame-skip 2 \
    --resize 1280 \
    --output-fps 30 \
    --max-age 50
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | `input_video.mp4` | Input video path |
| `--output`, `-o` | `outputs/output.mp4` | Output video path |
| `--download`, `-d` | — | YouTube/URL to download first |
| `--model` | `yolov8n.pt` | YOLOv8 model variant |
| `--confidence` | `0.3` | Detection confidence threshold |
| `--device` | `auto` | Inference device: auto, cuda, or cpu |
| `--output-fps` | `input_fps / frame_skip` | Override output FPS |
| `--frame-skip` | `1` | Process every Nth frame |
| `--resize` | original | Resize width (keeps aspect ratio) |
| `--max-age` | `30` | Frames to keep a lost track alive |
| `--no-trajectory` | — | Disable trajectory trails |
| `--no-heatmap` | — | Disable heatmap output |
| `--no-count-plot` | — | Disable object-count chart |

---

## 📥 Suggested Input Videos

Use any publicly available sports video. Good options for testing:

1. **Football/Soccer** — FIFA World Cup highlights or Premier League clips
2. **Basketball** — NBA game clips showing multiple players
3. **Cricket** — IPL match highlights with batsmen, fielders in view
4. **Marathon** — Road race footage with many runners

Download example:
```bash
python main.py --download "https://www.youtube.com/watch?v=EXAMPLE" --input input_video.mp4
```

---

## 🧠 Model & Tracker Choice

### Why YOLOv8?

- State-of-the-art single-stage detector with excellent speed/accuracy
- Built-in NMS, easy model scaling (nano → extra-large)
- Pre-trained on COCO (80 classes including "person")
- Active community and maintenance by Ultralytics

### Why DeepSORT?

- Combines **motion** (Kalman filter) and **appearance** (deep embeddings)
- Handles occlusion via track memory (`max_age` parameter)
- Appearance features (MobileNet) reduce ID switches for similar-looking subjects
- Well-proven in sports analytics and surveillance

---

## 📊 Outputs

After a run, the `outputs/` directory contains:

| File | Description |
|------|-------------|
| `output.mp4` | Annotated video with bounding boxes, IDs, trajectories |
| `heatmap.png` | Movement density heatmap (JET colourmap) |
| `object_count.png` | Chart of active tracked objects vs. frame number |
| `sample_frames/` | Periodic snapshots of annotated frames |

---

## 🎥 Video Source

- Source link: https://www.youtube.com/shorts/AOSFy_nmRPs
- Local file used: `trakin.webm` (converted to `trakin.mp4` for processing)

---

## ✅ Run Summary (Local Video)

- Input: `trakin.mp4` (1080x1920 @ 30 FPS, 436 frames)
- Output: `outputs/output.mp4`
- Average processing speed: ~11.4 FPS on GPU
- Unique IDs assigned: 1
- Generated: heatmap + object count plot + sample frames

---

## 🖥️ Frontend Viewer

A lightweight local web page lets you select a video file and view a report summary on screen.

**Open locally:**

```bash
cd multi_object_tracking
python -m http.server 8000
```

Then open:

```
http://localhost:8000/web/index.html
```

---

## 🧭 Streamlit App (Run Tracking)

Launch the Streamlit UI to upload a video and run the full pipeline from the browser:

```bash
cd multi_object_tracking
streamlit run streamlit_app.py
```

Then open the URL shown in the terminal. The app will:

- Upload a video file
- Run YOLOv8 + DeepSORT
- Show the output video, heatmap, count plot, and sample frames

---

## 🎬 Demo Video

Record a 3–5 minute demo video following [demo_script.md](demo_script.md). Save it as `outputs/demo.mp4` (or upload and link it here).

---

## ⚠️ Assumptions & Limitations

### Assumptions

- Input video has reasonable resolution (360p–1080p)
- Primary tracking targets are "person" class (COCO ID 0)
- Camera is relatively stable or has slow pan/zoom

### Limitations

- **ID switches** can still occur during prolonged full occlusion or when many similar-looking subjects cluster together
- **Motion blur** degrades detection confidence — detector may miss subjects in fast-motion frames
- **Camera cuts** (scene changes) will cause all tracks to reset
- **Small/distant subjects** may not be detected with the nano model — upgrade to `yolov8s.pt` or larger
- Processing speed depends on GPU availability; CPU-only runs will be significantly slower

---

## 📄 License

This project is for educational and research purposes.

---

## 📑 Technical Report (Appended)

### 1. System Overview

This system implements a complete pipeline for detecting and tracking multiple objects (primarily persons/players) in sports and event video footage. The pipeline is designed for robustness, modularity, and ease of configuration.

**Pipeline architecture:**

```
Input Video → Frame Extraction → YOLOv8 Detection → DeepSORT Tracking
            → Annotated Output Video + Analytics (Heatmap, Count Plot)
```

### 2. Detector: YOLOv8

**Model Selection**

We use **YOLOv8** from Ultralytics as the detection backbone. The default configuration uses `yolov8n.pt` (nano) for speed, but the pipeline supports all variants (`s`, `m`, `l`, `x`) via CLI flags.

**Why YOLOv8?**

- **Speed**: Single-stage architecture; nano variant runs 100+ FPS on modern GPUs
- **Accuracy**: State-of-the-art mAP on COCO; anchor-free design reduces tuning
- **Ease of Use**: Ultralytics provides a clean Python API with built-in NMS
- **Pre-trained**: COCO pre-training covers 80 classes including "person"
- **Scalability**: Easy to swap nano → small → large for accuracy-critical use cases

**Detection Output**

For each frame, the detector returns a list of bounding boxes with:

- Coordinates `[x1, y1, x2, y2]`
- Confidence score (0–1)
- Class label (filtered to "person" by default)

### 3. Tracker: DeepSORT

**Why DeepSORT?**

DeepSORT extends the original SORT algorithm with **appearance-based re-identification**, making it significantly more robust to occlusion and ID switching.

- **Kalman Filter**: Predicts object motion between frames; bridges short gaps
- **Hungarian Algorithm**: Optimal assignment of detections to existing tracks
- **Appearance Embeddings**: MobileNet-based feature vectors distinguish visually similar subjects
- **Track Lifecycle**: Tentative → Confirmed → Deleted states suppress noise

**How ID Consistency Is Maintained**

1. **Motion prediction**: The Kalman filter predicts where each tracked object will be in the next frame. Even if detection is missed for a few frames, the predicted position is used for matching.
2. **Appearance matching**: Each detection is paired with a deep appearance embedding (128-D MobileNet feature). When a previously occluded subject reappears, its appearance vector is compared against stored track features, enabling re-identification.
3. **Cascaded matching**: DeepSORT first attempts to match detections to tracks using appearance distance, then falls back to IoU-based matching for unmatched detections. This two-stage approach balances identity preservation with spatial proximity.
4. **Max age**: Tracks are kept alive for up to `max_age` frames (default: 30) without a matching detection. This allows recovery after brief occlusions without prematurely discarding tracks.
5. **Confirmation threshold**: New tracks require `n_init` (default: 3) consecutive detections before being promoted to "confirmed" status. This suppresses false-positive detections from becoming tracked objects.

### 4. Challenges & Solutions

**Occlusion**

Players frequently overlap in team sports, causing detections to merge or disappear.

**Solution**: DeepSORT's `max_age` parameter keeps occluded tracks alive. The Kalman filter continues predicting motion during occlusion. When the subject reappears, appearance embeddings help re-associate the detection with the correct track.

**ID Switching**

When two similar-looking subjects cross paths, their IDs may swap.

**Solution**: Appearance embeddings provide discriminative features beyond spatial proximity. The cascaded matching strategy prioritises appearance similarity over IoU, reducing switches. However, this remains the hardest challenge — subjects in identical uniforms (same team) can still cause switches.

**Motion Blur**

Fast camera pans or rapid subject movement create blurred frames where detection confidence drops.

**Solution**: A lower confidence threshold (0.3) catches more blurred detections. The Kalman filter smooths over frames where detection fails entirely. Frame skipping can also help by processing only sharp frames.

**Camera Motion (Pan/Zoom)**

Camera movement shifts all bounding boxes between frames, confusing motion-based matching.

**Solution**: The appearance embedding component of DeepSORT is camera-motion agnostic — it matches based on visual features, not just position. For extreme camera motion, increasing `max_iou_distance` helps.

**Scale Variation**

Subjects appear at different sizes depending on distance from camera.

**Solution**: YOLOv8's multi-scale feature pyramid handles scale variation natively. DeepSORT's appearance model also extracts scale-invariant features.

### 5. Advanced Features Implemented

**Trajectory Visualization**

Each tracked object's centre point is recorded per frame. The last N points (configurable, default 30) are rendered as a coloured trail line on the output video. Trail opacity increases toward the current position, creating a fade-out effect.

**Movement Heatmap**

All trajectory points across all tracks are accumulated into a 2D density map. Gaussian blurring smooths the result, and a JET colourmap highlights high-activity zones. This reveals movement patterns — for example, the central midfield area in a football match.

**Object Count Over Time**

A time-series chart records the number of actively tracked objects per frame. This helps identify key moments (e.g., a cluster of players converging for a set piece) and can validate tracking consistency.

### 6. Failure Cases

1. **Prolonged full occlusion** (> 30 frames): Tracks will be deleted if a subject is hidden for longer than `max_age`. The subject will receive a new ID upon reappearance.
2. **Identical appearance**: Players on the same team wearing identical kits are nearly indistinguishable to the appearance model. ID switches between same-team players remain common.
3. **Scene cuts**: Hard cuts in broadcast footage cause all existing tracks to lose association. The system has no scene-cut detection.
4. **Very crowded scenes**: When 20+ subjects cluster tightly (e.g., marathon start), the combination of overlapping bounding boxes and similar appearances overwhelms the tracker.
5. **Tiny/distant subjects**: The nano model may fail to detect subjects occupying fewer than ~30x30 pixels. Using a larger model variant mitigates this.

### 7. Future Improvements

1. **BoT-SORT or OC-SORT**: More modern trackers that handle camera motion explicitly via camera-motion compensation.
2. **Re-identification model**: Train a domain-specific ReID model (e.g., on sports datasets) to replace the generic MobileNet embeddings.
3. **Scene-cut detection**: Automatically detect hard cuts and reset all tracks to prevent ghost tracks.
4. **Team/jersey colour classification**: Assign team labels based on dominant jersey colour within each bounding box.
5. **3D position estimation**: Use camera calibration (homography) to map pixel coordinates to real-world pitch coordinates for accurate speed/distance estimation.
6. **Online learning**: Adapt appearance features per-track during inference for better long-term re-identification.

### 8. Conclusion

The YOLOv8 + DeepSORT combination provides a strong baseline for multi-object tracking in sports video. The system successfully assigns persistent IDs across hundreds of frames, handles moderate occlusion, and provides useful analytics (trajectories, heatmaps, count plots). The modular architecture allows easy substitution of detector or tracker components as better models emerge.

---

## 📦 Submission Checklist

- [ ] GitHub repository or zipped codebase
- [ ] README.md with setup, run steps, and assumptions
- [ ] Annotated output video (`outputs/output.mp4`)
- [ ] Original public video link (see Video Source above)
- [ ] Short technical report (`report.md`)
- [ ] Sample screenshots (`outputs/sample_frames/`)
- [ ] 3–5 minute demo video (`outputs/demo.mp4`)
