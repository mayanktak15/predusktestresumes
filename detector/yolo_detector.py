"""
YOLOv8-based Object Detection Module.

Uses Ultralytics YOLOv8 for real-time multi-object detection.
Outputs standardized Detection objects consumed by the tracker.
"""

from ultralytics import YOLO
import numpy as np
import torch
from typing import List, Tuple, Optional


class Detection:
    """
    Represents a single detection result.

    Attributes:
        bbox:       Bounding box as [x1, y1, x2, y2] (top-left, bottom-right).
        confidence: Detection confidence score (0-1).
        class_id:   Integer COCO class ID.
        class_name: Human-readable class label.
    """

    def __init__(self, bbox: List[float], confidence: float,
                 class_id: int, class_name: str):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    @property
    def tlwh(self) -> List[float]:
        """Convert [x1, y1, x2, y2] → [top-left-x, top-left-y, width, height]."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]

    @property
    def center(self) -> Tuple[int, int]:
        """Return the centre point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def __repr__(self):
        return (f"Detection(class={self.class_name}, "
                f"conf={self.confidence:.2f}, bbox={self.bbox})")


class YOLODetector:
    """
    YOLOv8 wrapper for the tracking pipeline.

    Loads a pre-trained (or fine-tuned) YOLOv8 model and exposes a
    simple `detect(frame)` interface that returns a list of Detection
    objects filtered by confidence and target classes.
    """

    def __init__(self, model_name: str = "yolov8n.pt",
                 confidence: float = 0.3,
                 iou_threshold: float = 0.45,
                 target_classes: Optional[List[int]] = None,
                 device: str = "auto",
                 use_half: bool = True):
        """
        Args:
            model_name:     YOLOv8 weight file (n/s/m/l/x variants).
            confidence:     Minimum confidence to keep a detection.
            iou_threshold:  IoU threshold for Non-Maximum Suppression.
            target_classes: COCO class IDs to detect (None = all classes).
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.class_names = self.model.names
        self.device = device
        self.use_half = bool(use_half and self.device != "cpu")

        try:
            self.model.to(self.device)
        except Exception:
            pass

        try:
            self.model.fuse()
        except Exception:
            pass

        print(f"[Detector] Loaded model: {model_name}  device={self.device}  "
              f"fp16={self.use_half}")
        if target_classes:
            names = [self.class_names[c] for c in target_classes]
            print(f"[Detector] Filtering for classes: {names}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame.

        Returns:
            List of Detection objects sorted by confidence (descending).
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.target_classes,
            verbose=False,
            device=self.device,
            half=self.use_half,
        )

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                detections.append(Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.class_names[cls_id],
                ))

        return detections
