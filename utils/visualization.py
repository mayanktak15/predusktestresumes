"""
Visualization utilities — bounding boxes, trajectories, heatmaps, count plots.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# ------------------------------------------------------------------ #
#  Colour palette (BGR for OpenCV)                                    #
# ------------------------------------------------------------------ #
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 255), (255, 128, 0), (0, 128, 255),
    (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (128, 128, 255), (128, 255, 255), (255, 128, 128),
    (255, 128, 255),
]


def get_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic colour per track ID."""
    if not isinstance(track_id, int):
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            track_id = abs(hash(track_id))
    return COLORS[track_id % len(COLORS)]


# ------------------------------------------------------------------ #
#  Frame annotation                                                   #
# ------------------------------------------------------------------ #
def draw_tracked_objects(
    frame: np.ndarray,
    tracked_objects,
    trajectories: Optional[Dict] = None,
    trajectory_length: int = 30,
    draw_bbox: bool = True,
    draw_label: bool = True,
    draw_confidence: bool = True,
    draw_trajectory: bool = True,
) -> np.ndarray:
    """
    Render bounding boxes, ID labels, confidence, and trajectory
    trails on a frame.
    """
    out = frame.copy()

    for obj in tracked_objects:
        color = get_color(obj.track_id)
        x1, y1, x2, y2 = (int(c) for c in obj.bbox)

        # --- bounding box ---
        if draw_bbox:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # --- label ---
        parts: List[str] = [f"ID:{obj.track_id}"]
        if draw_label:
            parts.append(obj.class_name)
        if draw_confidence and obj.confidence > 0:
            parts.append(f"{obj.confidence:.2f}")
        label = " | ".join(parts)

        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 5),
                      (x1 + tw + 5, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # --- trajectory trail ---
        if draw_trajectory and trajectories:
            traj = trajectories.get(obj.track_id, [])
            if trajectory_length:
                traj = traj[-trajectory_length:]
            for i in range(1, len(traj)):
                alpha = i / len(traj)
                thick = max(1, int(2 * alpha))
                cv2.line(out, traj[i - 1], traj[i],
                         color, thick, cv2.LINE_AA)

    return out


def draw_info_overlay(
    frame: np.ndarray,
    frame_num: int,
    total_frames: int,
    active_tracks: int,
    total_unique: int,
    fps: float = 0.0,
) -> np.ndarray:
    """Semi-transparent stats overlay in the top-left corner."""
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    lines = [
        f"Frame: {frame_num}/{total_frames}",
        f"Active Tracks: {active_tracks}",
        f"Total Unique IDs: {total_unique}",
        f"Processing FPS: {fps:.1f}",
    ]
    y = 30
    for line in lines:
        cv2.putText(out, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)
        y += 25

    return out


# ------------------------------------------------------------------ #
#  Advanced analytics                                                 #
# ------------------------------------------------------------------ #
def generate_heatmap(
    trajectories: Dict[int, List[Tuple[int, int]]],
    width: int,
    height: int,
    output_path: str,
) -> np.ndarray:
    """
    Movement-density heatmap over all trajectories.
    Each centre-point is rendered as a Gaussian blob; the result
    is normalised and colour-mapped (JET).
    """
    hmap = np.zeros((height, width), dtype=np.float32)

    for pts in trajectories.values():
        for x, y in pts:
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(hmap, (x, y), 15, 1, -1)

    hmap = cv2.GaussianBlur(hmap, (51, 51), 0)
    if hmap.max() > 0:
        hmap /= hmap.max()

    colored = cv2.applyColorMap(
        (hmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite(output_path, colored)
    print(f"[Vis] Heatmap saved → {output_path}")
    return colored


def generate_object_count_plot(
    count_history: List[Tuple[int, int]],
    output_path: str,
):
    """Line chart of tracked-object count over time (frames)."""
    if not count_history:
        print("[Vis] No count data — skipping plot.")
        return

    frames, counts = zip(*count_history)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(frames, counts, color="#2196F3", lw=1.5, alpha=0.8)
    ax.fill_between(frames, counts, alpha=0.2, color="#2196F3")
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Tracked Objects", fontsize=12)
    ax.set_title("Object Count Over Time",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Vis] Count plot saved → {output_path}")
