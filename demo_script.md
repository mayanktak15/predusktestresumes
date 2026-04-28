# Demo Video Script — Multi-Object Detection & Tracking Pipeline

**Duration:** 3–5 minutes
**Tone:** Technical but accessible — aimed at an evaluator or team lead

---

## SLIDE 1 — Title (0:00 – 0:15)

**Visual:** Project title card + sample tracked frame

**Narration:**
> "Hi, I'm presenting a multi-object detection and persistent-ID tracking pipeline built for sports video analysis. This system detects players in real-time, assigns unique IDs, and maintains those IDs consistently across hundreds of frames — even through occlusion, motion blur, and camera movement."

---

## SLIDE 2 — Pipeline Overview (0:15 – 0:45)

**Visual:** Architecture diagram:
```
Video → Frame Extraction → YOLOv8 Detection → DeepSORT Tracking → Annotated Output
```

**Narration:**
> "The pipeline follows a clean modular architecture. First, we extract frames from the input video. Each frame passes through YOLOv8 for real-time object detection — generating bounding boxes with confidence scores. These detections are then fed into DeepSORT, our multi-object tracker, which assigns and maintains persistent IDs. Finally, annotated frames are written to an output video with bounding boxes, ID labels, and trajectory trails."

---

## SLIDE 3 — Detection: YOLOv8 (0:45 – 1:30)

**Visual:** Side-by-side — raw frame vs. frame with detection boxes

**Narration:**
> "For detection, we use YOLOv8 from Ultralytics — the current state-of-the-art single-stage detector. It's anchor-free, runs at over 100 FPS on a modern GPU, and is pre-trained on COCO with 80 classes. We filter detections to the 'person' class and apply a configurable confidence threshold — set to 0.3 by default to catch partially visible or motion-blurred subjects. The detector outputs bounding box coordinates, a confidence score, and the class label for each detected person."

---

## SLIDE 4 — Tracking: DeepSORT (1:30 – 2:30)

**Visual:** Tracked frame with coloured IDs + trajectory lines

**Narration:**
> "The core of this system is DeepSORT — Deep Simple Online Realtime Tracking. It combines three components:
>
> First, a **Kalman filter** predicts where each tracked object will move next. This bridges gaps when detection temporarily fails.
>
> Second, the **Hungarian algorithm** optimally assigns new detections to existing tracks based on a combined distance metric.
>
> Third — and this is what distinguishes DeepSORT from basic SORT — **appearance embeddings**. A MobileNet model extracts a 128-dimensional feature vector from each detected person. When a player is occluded and reappears, the system compares appearance features to reconnect the right track — even if they've moved significantly.
>
> Tracks go through a lifecycle: tentative tracks require 3 consecutive detections to be confirmed, and lost tracks survive for up to 30 frames before deletion."

---

## SLIDE 5 — Advanced Analytics (2:30 – 3:15)

**Visual:** Heatmap + object count chart

**Narration:**
> "Beyond basic tracking, the pipeline generates three analytics outputs:
>
> **Trajectory visualisation**: Each tracked ID has a coloured trail showing their recent movement path directly on the video.
>
> **Movement heatmap**: All trajectory points are accumulated into a density map, revealing high-traffic zones — for example, the central midfield area in a football match.
>
> **Object count over time**: A time-series chart shows how many players are being tracked per frame. Dips can indicate occlusion events or scene transitions."

---

## SLIDE 6 — Challenges & Solutions (3:15 – 4:00)

**Visual:** Example frames showing occlusion, ID switch, blur

**Narration:**
> "Real-world sports video presents serious challenges:
>
> **Occlusion** — players overlap constantly. DeepSORT handles this through track memory and appearance-based re-identification.
>
> **ID switching** — when two similar-looking players cross paths. Our cascaded matching strategy prioritises appearance over spatial proximity to minimise this.
>
> **Motion blur** — fast camera pans degrade detection. A lower confidence threshold and the Kalman filter's prediction help maintain continuity.
>
> **Similar appearance** — same-team players in identical kits remain the toughest case. A domain-specific re-identification model would improve this significantly."

---

## SLIDE 7 — Results & Demo (4:00 – 4:30)

**Visual:** Play output video clip

**Narration:**
> "Here's the system running on a [sport] match clip. You can see consistent IDs maintained across frames, trajectory trails following player movement, and the info overlay showing active tracks and processing speed. The system processes at approximately [X] FPS on a [GPU model]."

---

## SLIDE 8 — Future Work & Closing (4:30 – 5:00)

**Visual:** Bullet points of improvements

**Narration:**
> "For future improvements, I'd consider:
> - Upgrading to BoT-SORT for explicit camera-motion compensation
> - Training a sports-specific re-identification model
> - Adding team classification based on jersey colour
> - Implementing homography-based pitch mapping for real-world speed estimation
>
> Thank you for watching. The full code, documentation, and technical report are available in the project repository."

---

*End of demo script.*
