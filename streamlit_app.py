import os
import shutil
import subprocess
import time
from pathlib import Path

import streamlit as st

from config import Config
from main import TrackingPipeline


APP_TITLE = "Multi-Object Tracking"
UPLOAD_DIR = Path("outputs/inputs")
SAMPLE_DIR = Path("outputs/sample_frames")
HEATMAP_PATH = Path("outputs/heatmap.png")
COUNT_PATH = Path("outputs/object_count.png")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_to_mp4(src_path: Path, dst_path: Path) -> bool:
    if not has_ffmpeg():
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(dst_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return result.returncode == 0 and dst_path.exists()


def ensure_browser_mp4(src_path: Path, dst_path: Path) -> Path:
    if not has_ffmpeg():
        return src_path
    if dst_path.exists() and dst_path.stat().st_mtime >= src_path.stat().st_mtime:
        return dst_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0 and dst_path.exists():
        return dst_path
    return src_path


def run_pipeline(input_path: Path, cfg: Config) -> None:
    pipeline = TrackingPipeline(cfg)
    pipeline.run(str(input_path), cfg.OUTPUT_VIDEO)


def build_sidebar(cfg: Config) -> Config:
    st.sidebar.header("Pipeline Settings")

    cfg.DETECTOR_MODEL = st.sidebar.selectbox(
        "YOLOv8 model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0,
    )
    cfg.DETECTION_CONFIDENCE = st.sidebar.slider(
        "Detection confidence",
        min_value=0.1,
        max_value=0.9,
        value=cfg.DETECTION_CONFIDENCE,
        step=0.05,
    )
    cfg.FRAME_SKIP = st.sidebar.slider(
        "Frame skip",
        min_value=1,
        max_value=6,
        value=cfg.FRAME_SKIP,
        step=1,
    )
    cfg.MAX_AGE = st.sidebar.slider(
        "Max age (frames)",
        min_value=5,
        max_value=90,
        value=cfg.MAX_AGE,
        step=5,
    )
    cfg.RESIZE_WIDTH = st.sidebar.number_input(
        "Resize width (0 = keep)",
        min_value=0,
        max_value=1920,
        value=0,
        step=32,
    )
    cfg.DEVICE = st.sidebar.selectbox(
        "Device",
        ["auto", "cuda", "cpu"],
        index=0,
    )
    cfg.OUTPUT_FPS = st.sidebar.number_input(
        "Output FPS (0 = auto)",
        min_value=0,
        max_value=120,
        value=0,
        step=1,
    )

    cfg.CLEAR_SAMPLES_ON_REFRESH = st.sidebar.checkbox(
        "Clear sample frames on refresh",
        value=True,
    )

    cfg.RESIZE_WIDTH = None if cfg.RESIZE_WIDTH == 0 else int(cfg.RESIZE_WIDTH)
    cfg.OUTPUT_FPS = None if cfg.OUTPUT_FPS == 0 else float(cfg.OUTPUT_FPS)

    return cfg


def clear_sample_frames() -> None:
    if not SAMPLE_DIR.exists():
        return
    for item in SAMPLE_DIR.glob("*"):
        if item.is_file():
            try:
                item.unlink()
            except FileNotFoundError:
                continue


def clear_visual_outputs() -> None:
    for path in (HEATMAP_PATH, COUNT_PATH):
        if path.exists() and path.is_file():
            try:
                path.unlink()
            except FileNotFoundError:
                continue


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("Multi-Object Detection and Persistent ID Tracking")
    st.write(
        "Upload a video to run YOLOv8 detection + DeepSORT tracking, then review the outputs."
    )

    cfg = build_sidebar(Config())

    if "cleared_samples" not in st.session_state:
        if getattr(cfg, "CLEAR_SAMPLES_ON_REFRESH", False):
            clear_sample_frames()
            clear_visual_outputs()
        st.session_state["cleared_samples"] = True

    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "webm", "mkv", "mov"],
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("Run Tracking", type="primary", disabled=uploaded is None)
    with col2:
        st.caption("Outputs will be saved to outputs/ and displayed below.")

    if run_btn and uploaded:
        ensure_dir(UPLOAD_DIR)
        src_path = UPLOAD_DIR / uploaded.name
        src_path.write_bytes(uploaded.getbuffer())

        input_path = src_path
        if src_path.suffix.lower() in {".webm", ".mkv"}:
            mp4_path = src_path.with_suffix(".mp4")
            if convert_to_mp4(src_path, mp4_path):
                input_path = mp4_path
            else:
                st.error("FFmpeg is required to convert this format. Install ffmpeg and try again.")
                return

        cfg.INPUT_VIDEO = str(input_path)
        cfg.OUTPUT_VIDEO = "outputs/output.mp4"

        start = time.time()
        with st.spinner("Running detection + tracking..."):
            run_pipeline(input_path, cfg)
        elapsed = time.time() - start

        st.success(f"Processing complete in {elapsed:.1f}s")

    st.divider()
    st.subheader("Results")

    output_video = Path("outputs/output.mp4")
    if output_video.exists():
        playable_video = ensure_browser_mp4(
            output_video, Path("outputs/output_streamlit.mp4")
        )
        video_path = playable_video.resolve()
        video_bytes = video_path.read_bytes()
        st.video(video_bytes, format="video/mp4")
        st.caption(
            f"Output video: {output_video.name} ({output_video.stat().st_size / 1024 / 1024:.1f} MB)"
        )
        st.download_button(
            label="Download output video",
            data=video_bytes,
            file_name=output_video.name,
            mime="video/mp4",
        )
    else:
        st.info("Run the pipeline to generate outputs/output.mp4")

    cols = st.columns(2)
    heatmap = HEATMAP_PATH
    count_plot = COUNT_PATH

    with cols[0]:
        st.subheader("Heatmap")
        if heatmap.exists():
            st.image(str(heatmap), use_container_width=True)
        else:
            st.caption("Heatmap not generated yet.")

    with cols[1]:
        st.subheader("Object Count")
        if count_plot.exists():
            st.image(str(count_plot), use_container_width=True)
        else:
            st.caption("Count plot not generated yet.")

    st.subheader("Sample Frames")
    sample_dir = Path("outputs/sample_frames")
    if sample_dir.exists():
        samples = sorted(sample_dir.glob("*.jpg"))[:6]
        if samples:
            grid_cols = st.columns(3)
            for idx, sample in enumerate(samples):
                with grid_cols[idx % 3]:
                    st.image(str(sample), use_container_width=True)
        else:
            st.caption("No sample frames available.")
    else:
        st.caption("Sample frames directory not found.")


if __name__ == "__main__":
    main()
