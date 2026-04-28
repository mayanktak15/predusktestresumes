# Technical Report — Multi-Object Detection & Persistent-ID Tracking

---

## 1. System Overview

This system implements a complete pipeline for detecting and tracking multiple objects (primarily persons/players) in sports and event video footage. The pipeline is designed for robustness, modularity, and ease of configuration.

**Video source:** https://www.youtube.com/shorts/AOSFy_nmRPs

**Local file used:** `trakin.webm` (converted to `trakin.mp4` for processing)

**Pipeline architecture:**

```
Input Video → Frame Extraction → YOLOv8 Detection → DeepSORT Tracking
            → Annotated Output Video + Analytics (Heatmap, Count Plot)
```

---

## 2. Detector: YOLOv8

### Model Selection

We use **YOLOv8** from Ultralytics as the detection backbone. The default configuration uses `yolov8n.pt` (nano) for speed, but the pipeline supports all variants (`s`, `m`, `l`, `x`) via CLI flags.

### Why YOLOv8?

| Factor | YOLOv8 Advantage |
|--------|-----------------|
| **Speed** | Single-stage architecture; nano variant runs 100+ FPS on modern GPUs |
| **Accuracy** | State-of-the-art mAP on COCO; anchor-free design reduces tuning |
| **Ease of Use** | Ultralytics provides a clean Python API with built-in NMS |
| **Pre-trained** | COCO pre-training covers 80 classes including "person" |
| **Scalability** | Easy to swap nano → small → large for accuracy-critical use cases |

### Detection Output

For each frame, the detector returns a list of bounding boxes with:
- Coordinates `[x1, y1, x2, y2]`
- Confidence score (0–1)
- Class label (filtered to "person" by default)

---

## 3. Tracker: DeepSORT

### Why DeepSORT?

DeepSORT extends the original SORT algorithm with **appearance-based re-identification**, making it significantly more robust to occlusion and ID switching.

| Component | Purpose |
|-----------|---------|
| **Kalman Filter** | Predicts object motion between frames; bridges short gaps |
| **Hungarian Algorithm** | Optimal assignment of detections to existing tracks |
| **Appearance Embeddings** | MobileNet-based feature vectors distinguish visually similar subjects |
| **Track Lifecycle** | Tentative → Confirmed → Deleted states suppress noise |

### How ID Consistency Is Maintained

1. **Motion prediction**: The Kalman filter predicts where each tracked object will be in the next frame. Even if detection is missed for a few frames, the predicted position is used for matching.

2. **Appearance matching**: Each detection is paired with a deep appearance embedding (128-D MobileNet feature). When a previously occluded subject reappears, its appearance vector is compared against stored track features, enabling re-identification.

3. **Cascaded matching**: DeepSORT first attempts to match detections to tracks using appearance distance, then falls back to IoU-based matching for unmatched detections. This two-stage approach balances identity preservation with spatial proximity.

4. **Max age**: Tracks are kept alive for up to `max_age` frames (default: 30) without a matching detection. This allows recovery after brief occlusions without prematurely discarding tracks.

5. **Confirmation threshold**: New tracks require `n_init` (default: 3) consecutive detections before being promoted to "confirmed" status. This suppresses false-positive detections from becoming tracked objects.

---

## 4. Challenges & Solutions

### 4.1 Occlusion

**Problem**: Players frequently overlap in team sports, causing detections to merge or disappear.

**Solution**: DeepSORT's `max_age` parameter keeps occluded tracks alive. The Kalman filter continues predicting motion during occlusion. When the subject reappears, appearance embeddings help re-associate the detection with the correct track.

### 4.2 ID Switching

**Problem**: When two similar-looking subjects cross paths, their IDs may swap.

**Solution**: Appearance embeddings provide discriminative features beyond spatial proximity. The cascaded matching strategy prioritises appearance similarity over IoU, reducing switches. However, this remains the hardest challenge — subjects in identical uniforms (same team) can still cause switches.

### 4.3 Motion Blur

**Problem**: Fast camera pans or rapid subject movement create blurred frames where detection confidence drops.

**Solution**: A lower confidence threshold (0.3) catches more blurred detections. The Kalman filter smooths over frames where detection fails entirely. Frame skipping can also help by processing only sharp frames.

### 4.4 Camera Motion (Pan/Zoom)

**Problem**: Camera movement shifts all bounding boxes between frames, confusing motion-based matching.

**Solution**: The appearance embedding component of DeepSORT is camera-motion agnostic — it matches based on visual features, not just position. For extreme camera motion, increasing `max_iou_distance` helps.

### 4.5 Scale Variation

**Problem**: Subjects appear at different sizes depending on distance from camera.

**Solution**: YOLOv8's multi-scale feature pyramid handles scale variation natively. DeepSORT's appearance model also extracts scale-invariant features.

---

## 5. Advanced Features Implemented

### 5.1 Trajectory Visualization

Each tracked object's centre point is recorded per frame. The last N points (configurable, default 30) are rendered as a coloured trail line on the output video. Trail opacity increases toward the current position, creating a fade-out effect.

### 5.2 Movement Heatmap

All trajectory points across all tracks are accumulated into a 2D density map. Gaussian blurring smooths the result, and a JET colourmap highlights high-activity zones. This reveals movement patterns — e.g., which areas of a football pitch see the most player traffic.

### 5.3 Object Count Over Time

A time-series chart records the number of actively tracked objects per frame. This helps identify key moments (e.g., a cluster of players converging for a set piece) and can validate tracking consistency.

---

## 6. Failure Cases

1. **Prolonged full occlusion** (> 30 frames): Tracks will be deleted if a subject is hidden for longer than `max_age`. The subject will receive a new ID upon reappearance.

2. **Identical appearance**: Players on the same team wearing identical kits are nearly indistinguishable to the appearance model. ID switches between same-team players remain common.

3. **Scene cuts**: Hard cuts in broadcast footage cause all existing tracks to lose association. The system has no scene-cut detection.

4. **Very crowded scenes**: When 20+ subjects cluster tightly (e.g., marathon start), the combination of overlapping bounding boxes and similar appearances overwhelms the tracker.

5. **Tiny/distant subjects**: The nano model may fail to detect subjects occupying fewer than ~30x30 pixels. Using a larger model variant mitigates this.

---

## 7. Future Improvements

1. **BoT-SORT or OC-SORT**: More modern trackers that handle camera motion explicitly via camera-motion compensation.

2. **Re-identification model**: Train a domain-specific ReID model (e.g., on sports datasets) to replace the generic MobileNet embeddings.

3. **Scene-cut detection**: Automatically detect hard cuts and reset all tracks to prevent ghost tracks.

4. **Team/jersey colour classification**: Assign team labels based on dominant jersey colour within each bounding box.

5. **3D position estimation**: Use camera calibration (homography) to map pixel coordinates to real-world pitch coordinates for accurate speed/distance estimation.

6. **Online learning**: Adapt appearance features per-track during inference for better long-term re-identification.

---

## 8. Conclusion

The YOLOv8 + DeepSORT combination provides a strong baseline for multi-object tracking in sports video. The system successfully assigns persistent IDs across hundreds of frames, handles moderate occlusion, and provides useful analytics (trajectories, heatmaps, count plots). The modular architecture allows easy substitution of detector or tracker components as better models emerge.
