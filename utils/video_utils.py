"""
Video I/O utilities — reading frames, writing output video, downloading.
"""

import cv2
import os
import numpy as np
from typing import Generator, Tuple, Optional


class VideoReader:
    """Frame-by-frame video reader with optional resizing."""

    def __init__(self, video_path: str,
                 resize_width: Optional[int] = None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.resize_width = resize_width

        if self.fps <= 0:
            self.fps = 30.0
            print("[VideoReader] Warning: FPS not detected, defaulting to 30")

        if resize_width:
            scale = resize_width / self.width
            self.out_w = resize_width
            self.out_h = int(self.height * scale)
        else:
            self.out_w = self.width
            self.out_h = self.height

        print(f"[VideoReader] {video_path}")
        print(f"[VideoReader] {self.width}x{self.height} @ {self.fps} FPS  "
              f"| {self.total_frames} frames")

    def frames(self, skip: int = 1
               ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield (frame_number, frame) tuples.

        Args:
            skip: Process every Nth frame (1 = every frame).
        """
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % skip == 0:
                if self.resize_width:
                    frame = cv2.resize(frame, (self.out_w, self.out_h))
                yield idx, frame
            idx += 1
        self.cap.release()

    def __del__(self):
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """Write annotated frames to an MP4 file."""

    def __init__(self, output_path: str, fps: int,
                 width: int, height: int):
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (width, height))
        self.output_path = output_path

        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create writer: {output_path}")

        print(f"[VideoWriter] → {output_path}  "
              f"({width}x{height} @ {fps} FPS)")

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()
            print(f"[VideoWriter] Saved: {self.output_path}")

    def __del__(self):
        self.release()


def download_video(url: str, output_path: str) -> str:
    """
    Download a video from a URL using yt-dlp.

    Returns the path to the downloaded file.
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required.  Install with:  pip install yt-dlp")

    ydl_opts = {
        "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"[Download] {url}")
        ydl.download([url])

    if os.path.exists(output_path):
        return output_path

    for ext in (".mp4", ".mkv", ".webm"):
        candidate = output_path + ext
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Download failed — file not at {output_path}")
