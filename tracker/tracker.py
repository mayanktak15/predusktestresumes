"""
Multi-Object Tracker using DeepSORT.

Maintains persistent IDs across frames via:
  - Appearance embeddings (MobileNet) for re-identification
  - Kalman filter for motion prediction through occlusions
  - Hungarian algorithm for optimal detection-to-track assignment
  - Track lifecycle management (tentative → confirmed → deleted)
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


class TrackedObject:
    """A single tracked object with a persistent ID."""

    def __init__(self, track_id: int, bbox: List[float],
                 confidence: float, class_name: str):
        self.track_id = int(track_id)
        self.bbox = bbox          # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_name = class_name

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def __repr__(self):
        return (f"TrackedObject(id={self.track_id}, "
                f"class={self.class_name}, bbox={self.bbox})")


class MultiObjectTracker:
    """
    DeepSORT-based multi-object tracker.

    Key design choices:
      • MobileNet appearance embeddings give a good speed/accuracy trade-off
        for re-identification after occlusion.
      • `max_age` controls how long a lost track survives — higher values
        tolerate longer occlusions but may cause ghost tracks.
      • `n_init` suppresses false positives by requiring several consecutive
        detections before promoting a track to "confirmed".
    """

    def __init__(self, max_age: int = 30, n_init: int = 3,
                 max_iou_distance: float = 0.7,
                 use_gpu: Optional[bool] = None):
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder="mobilenet",
            half=use_gpu,
            embedder_gpu=use_gpu,
        )

        # Trajectory history: {track_id: [(x, y), ...]}
        self.trajectories: Dict[int, List[Tuple[int, int]]] = {}

        print(f"[Tracker] DeepSORT initialised  "
              f"max_age={max_age}  n_init={n_init}  "
              f"max_iou_distance={max_iou_distance}  gpu={use_gpu}")

    # ------------------------------------------------------------------ #
    #  Core update                                                        #
    # ------------------------------------------------------------------ #
    def update(self, detections, frame: np.ndarray) -> List[TrackedObject]:
        """
        Feed new detections and the current frame to the tracker.

        Args:
            detections: List[Detection] from the detector.
            frame:      Current BGR frame (needed for appearance features).

        Returns:
            List of confirmed TrackedObject instances.
        """
        if not detections:
            tracks = self.tracker.update_tracks([], frame=frame)
        else:
            ds_dets = [
                (d.tlwh, d.confidence, d.class_name) for d in detections
            ]
            tracks = self.tracker.update_tracks(ds_dets, frame=frame)

        tracked_objects: List[TrackedObject] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            ltrb = track.to_ltrb()

            obj = TrackedObject(
                track_id=tid,
                bbox=[float(c) for c in ltrb],
                confidence=(track.det_conf
                            if track.det_conf is not None else 0.0),
                class_name=(track.det_class
                            if track.det_class is not None else "unknown"),
            )
            tracked_objects.append(obj)

            # Record trajectory
            if tid not in self.trajectories:
                self.trajectories[tid] = []
            self.trajectories[tid].append(obj.center)

        return tracked_objects

    # ------------------------------------------------------------------ #
    #  Trajectory helpers                                                 #
    # ------------------------------------------------------------------ #
    def get_trajectory(self, track_id: int,
                       max_length: Optional[int] = None
                       ) -> List[Tuple[int, int]]:
        """Return recent centre-point trajectory for one track."""
        traj = self.trajectories.get(track_id, [])
        if max_length and len(traj) > max_length:
            return traj[-max_length:]
        return traj

    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        return self.trajectories

    def get_total_unique_ids(self) -> int:
        return len(self.trajectories)
