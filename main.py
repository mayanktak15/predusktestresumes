#!/usr/bin/env python3
"""
Multi-Object Detection & Persistent-ID Tracking Pipeline
=========================================================
Pipeline flow:
    video → frame extraction → YOLOv8 detection → DeepSORT tracking
          → annotation → output video + analytics

Usage examples:
    # Run on a local video file
    python main.py --input match.mp4

    # Download a YouTube video first, then process
    python main.py --download "https://youtube.com/watch?v=..." --input input_video.mp4

    # Custom settings
    python main.py --input match.mp4 --confidence 0.4 --frame-skip 2 --model yolov8s.pt
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from config import Config
from detector.yolo_detector import YOLODetector
from tracker.tracker import MultiObjectTracker
from utils.video_utils import VideoReader, VideoWriter, download_video
from utils.visualization import (
    draw_tracked_objects,
    draw_info_overlay,
    generate_heatmap,
    generate_object_count_plot,
)


class TrackingPipeline:
    """
    End-to-end detection + tracking pipeline.

    Orchestrates:
      1. Frame reading (with optional skip / resize)
      2. YOLOv8 detection
      3. DeepSORT tracking
      4. Annotated-frame writing
      5. Analytics (heatmap, object-count plot, sample frames)
    """

    def __init__(self, config: Config):
        self.cfg = config

        self.detector = YOLODetector(
            model_name=config.DETECTOR_MODEL,
            confidence=config.DETECTION_CONFIDENCE,
            iou_threshold=config.DETECTION_IOU_THRESHOLD,
            target_classes=config.TARGET_CLASSES,
            device=config.DEVICE,
            use_half=config.USE_HALF,
        )

        self.tracker = MultiObjectTracker(
            max_age=config.MAX_AGE,
            n_init=config.N_INIT,
            max_iou_distance=config.MAX_IOU_DISTANCE,
            use_gpu=(None if config.DEVICE == "auto"
                     else config.DEVICE != "cpu"),
        )

        self.count_history: list = []  # (frame_num, active_count)
        print("\n[Pipeline] Initialisation complete.\n")

    # ------------------------------------------------------------------ #
    def run(self, input_path: str, output_path: str):
        """Execute the full pipeline on *input_path*."""

        reader = VideoReader(input_path, resize_width=self.cfg.RESIZE_WIDTH)
        if self.cfg.OUTPUT_FPS is not None:
            fps = self.cfg.OUTPUT_FPS
        else:
            fps = reader.fps
            if self.cfg.FRAME_SKIP > 1:
                fps = max(1.0, fps / self.cfg.FRAME_SKIP)

        writer = VideoWriter(
            output_path, fps, reader.out_w, reader.out_h)

        # Ensure sample-frames dir exists
        if self.cfg.SAVE_SAMPLE_FRAMES:
            os.makedirs(self.cfg.SAMPLE_FRAMES_DIR, exist_ok=True)

        processed = 0
        t_start = time.time()

        print(f"[Pipeline] Processing '{input_path}' → '{output_path}'")
        print(f"[Pipeline] Frame skip = {self.cfg.FRAME_SKIP}\n")

        for frame_num, frame in reader.frames(skip=self.cfg.FRAME_SKIP):

            t0 = time.time()

            # 1. Detect
            detections = self.detector.detect(frame)

            # 2. Track
            tracked = self.tracker.update(detections, frame)

            # 3. Annotate
            annotated = draw_tracked_objects(
                frame, tracked,
                trajectories=self.tracker.get_all_trajectories(),
                trajectory_length=self.cfg.TRAJECTORY_LENGTH,
                draw_bbox=self.cfg.DRAW_BBOXES,
                draw_label=self.cfg.DRAW_LABELS,
                draw_confidence=self.cfg.DRAW_CONFIDENCE,
                draw_trajectory=self.cfg.DRAW_TRAJECTORIES,
            )

            elapsed = time.time() - t0
            cur_fps = 1.0 / elapsed if elapsed > 0 else 0.0

            annotated = draw_info_overlay(
                annotated, frame_num, reader.total_frames,
                active_tracks=len(tracked),
                total_unique=self.tracker.get_total_unique_ids(),
                fps=cur_fps,
            )

            # 4. Write frame
            writer.write(annotated)

            # 5. Record analytics
            self.count_history.append((frame_num, len(tracked)))

            # 6. Save sample frames
            if (self.cfg.SAVE_SAMPLE_FRAMES
                    and processed % self.cfg.SAMPLE_FRAME_INTERVAL == 0):
                path = os.path.join(
                    self.cfg.SAMPLE_FRAMES_DIR,
                    f"frame_{frame_num:06d}.jpg")
                cv2.imwrite(path, annotated)

            processed += 1
            if processed % 50 == 0:
                print(f"  frame {frame_num:>6d}/{reader.total_frames}  "
                      f"tracks={len(tracked):>3d}  "
                      f"fps={cur_fps:.1f}")

        writer.release()
        total_time = time.time() - t_start

        print(f"\n[Pipeline] Done — {processed} frames in "
              f"{total_time:.1f}s  ({processed / total_time:.1f} avg FPS)")
        print(f"[Pipeline] Unique IDs assigned: "
              f"{self.tracker.get_total_unique_ids()}")

        # ---- Generate advanced analytics ---- #
        if self.cfg.ENABLE_HEATMAP:
            os.makedirs(os.path.dirname(self.cfg.HEATMAP_OUTPUT)
                        or ".", exist_ok=True)
            generate_heatmap(
                self.tracker.get_all_trajectories(),
                reader.out_w, reader.out_h,
                self.cfg.HEATMAP_OUTPUT,
            )

        if self.cfg.ENABLE_OBJECT_COUNT:
            os.makedirs(os.path.dirname(self.cfg.COUNT_PLOT_OUTPUT)
                        or ".", exist_ok=True)
            generate_object_count_plot(
                self.count_history,
                self.cfg.COUNT_PLOT_OUTPUT,
            )

        print("\n[Pipeline] All outputs saved. ✓")


# ================================================================== #
#  CLI                                                                 #
# ================================================================== #
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Object Detection & Tracking Pipeline")

    p.add_argument("--input", "-i", type=str,
                   default=Config.INPUT_VIDEO,
                   help="Path to input video file.")
    p.add_argument("--output", "-o", type=str,
                   default=Config.OUTPUT_VIDEO,
                   help="Path for annotated output video.")
    p.add_argument("--download", "-d", type=str, default=None,
                   help="Download video from URL before processing.")

    p.add_argument("--model", type=str, default=Config.DETECTOR_MODEL,
                   help="YOLOv8 model weight file.")
    p.add_argument("--confidence", type=float,
                   default=Config.DETECTION_CONFIDENCE,
                   help="Detection confidence threshold.")
    p.add_argument("--device", type=str,
                   default=Config.DEVICE,
                   help="Inference device: auto, cuda, or cpu.")
    p.add_argument("--output-fps", type=float,
                   default=Config.OUTPUT_FPS,
                   help="Override output FPS (default: input_fps / frame_skip).")
    p.add_argument("--frame-skip", type=int,
                   default=Config.FRAME_SKIP,
                   help="Process every Nth frame.")
    p.add_argument("--resize", type=int, default=None,
                   help="Resize frames to this width (keeps aspect).")

    p.add_argument("--max-age", type=int, default=Config.MAX_AGE,
                   help="Tracker max age (frames).")
    p.add_argument("--no-trajectory", action="store_true",
                   help="Disable trajectory drawing.")
    p.add_argument("--no-heatmap", action="store_true",
                   help="Disable heatmap generation.")
    p.add_argument("--no-count-plot", action="store_true",
                   help="Disable object-count plot.")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # Override config with CLI args
    cfg.INPUT_VIDEO = args.input
    cfg.OUTPUT_VIDEO = args.output
    cfg.DETECTOR_MODEL = args.model
    cfg.DETECTION_CONFIDENCE = args.confidence
    cfg.DEVICE = args.device
    cfg.OUTPUT_FPS = args.output_fps
    cfg.FRAME_SKIP = args.frame_skip
    cfg.RESIZE_WIDTH = args.resize
    cfg.MAX_AGE = args.max_age
    cfg.DRAW_TRAJECTORIES = not args.no_trajectory
    cfg.ENABLE_TRAJECTORY = not args.no_trajectory
    cfg.ENABLE_HEATMAP = not args.no_heatmap
    cfg.ENABLE_OBJECT_COUNT = not args.no_count_plot

    # Download video if requested
    if args.download:
        print(f"[Main] Downloading video …")
        cfg.INPUT_VIDEO = download_video(args.download, cfg.INPUT_VIDEO)

    if not os.path.exists(cfg.INPUT_VIDEO):
        print(f"[Error] Input video not found: {cfg.INPUT_VIDEO}")
        print("  Use --input <path> or --download <url>")
        sys.exit(1)

    pipeline = TrackingPipeline(cfg)
    pipeline.run(cfg.INPUT_VIDEO, cfg.OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
