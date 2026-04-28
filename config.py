"""
Configuration file for Multi-Object Detection and Tracking Pipeline.
All configurable parameters are centralized here for easy tuning.
"""


class Config:
    """Pipeline configuration with sensible defaults."""

    # --- Detection ---
    DETECTOR_MODEL = "yolov8n.pt"       # YOLOv8 nano (fast); use yolov8s/m/l/x for accuracy
    DETECTION_CONFIDENCE = 0.3          # Minimum confidence threshold
    DETECTION_IOU_THRESHOLD = 0.45      # IoU threshold for NMS
    TARGET_CLASSES = [0]                # COCO class IDs: 0 = person
    DEVICE = "auto"                     # "auto" | "cuda" | "cpu"
    USE_HALF = True                     # Use FP16 on CUDA for speed

    # --- Tracking ---
    MAX_AGE = 30                        # Max frames to keep a lost track alive
    N_INIT = 3                          # Min consecutive detections to confirm track
    MAX_IOU_DISTANCE = 0.7              # Max IoU distance for matching

    # --- Video ---
    INPUT_VIDEO = "input_video.mp4"
    OUTPUT_VIDEO = "outputs/output.mp4"
    OUTPUT_FPS = None                   # None = input_fps / frame_skip
    FRAME_SKIP = 1                      # Process every Nth frame (1 = all)
    RESIZE_WIDTH = None                 # None = original resolution

    # --- Visualization ---
    DRAW_BBOXES = True
    DRAW_LABELS = True
    DRAW_CONFIDENCE = True
    DRAW_TRAJECTORIES = True
    TRAJECTORY_LENGTH = 30              # Past positions to draw per track

    # --- Advanced Features ---
    ENABLE_TRAJECTORY = True
    ENABLE_OBJECT_COUNT = True
    ENABLE_HEATMAP = True
    HEATMAP_OUTPUT = "outputs/heatmap.png"
    COUNT_PLOT_OUTPUT = "outputs/object_count.png"

    # --- Sample Frames ---
    SAVE_SAMPLE_FRAMES = True
    SAMPLE_FRAME_INTERVAL = 100         # Save a sample every N frames
    SAMPLE_FRAMES_DIR = "outputs/sample_frames"
