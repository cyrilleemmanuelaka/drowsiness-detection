"""
Streamlit Web Application
=========================
Interactive UI to demonstrate the drowsiness detection system.

Features
--------
- Live webcam mode (browser camera).
- Image upload with single-frame analysis.
- Video upload with frame-by-frame analysis & summary stats.
- Real-time charts of EAR / CNN scores.

Run with:
    streamlit run src/app.py
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Make sibling imports work whether run as `streamlit run src/app.py`
# from the project root or directly from inside src/.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from detector import DrowsinessDetector  # noqa: E402

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .big-status   { font-size: 2.2rem; font-weight: 700; }
      .awake        { color: #16a34a; }
      .drowsy       { color: #dc2626; }
      .metric-card  { background:#1f2937; padding:1rem; border-radius:8px; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("😴 Drowsiness Detection — Hybrid CNN + EAR")
st.caption(
    "Real-time monitoring that combines geometric eye landmarks with a CNN classifier "
    "to detect microsleeps and prolonged eye closure."
)

# --------------------------------------------------------------------------- #
# Sidebar configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")

    ear_threshold = st.slider("EAR threshold", 0.10, 0.35, 0.23, 0.01,
                              help="Lower = stricter (more easily considered closed).")
    consec_frames = st.slider("Consecutive drowsy frames for alert", 5, 60, 20, 1)
    ear_weight = st.slider("EAR weight in fusion", 0.0, 1.0, 0.5, 0.05)
    cnn_weight = 1.0 - ear_weight
    st.write(f"CNN weight (auto): **{cnn_weight:.2f}**")

    st.divider()
    landmark_path = st.text_input(
        "Landmark model path",
        value=str(PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat"),
    )
    cnn_path = st.text_input(
        "CNN model path",
        value=str(PROJECT_ROOT / "models" / "eye_state_cnn.h5"),
    )

    st.divider()
    st.markdown("### 📊 About")
    st.markdown(
        "- **EAR**: Soukupová & Čech (2016)\n"
        "- **CNN**: Lightweight LeNet-style classifier\n"
        "- **Fusion**: Late, weighted, smoothed (30-frame window)"
    )

# --------------------------------------------------------------------------- #
# Cached detector
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models...")
def load_detector(landmark_p, cnn_p, ear_t, consec, ear_w, cnn_w):
    return DrowsinessDetector(
        shape_predictor_path=landmark_p,
        cnn_model_path=cnn_p,
        ear_threshold=ear_t,
        consec_frames=consec,
        ear_weight=ear_w,
        cnn_weight=cnn_w,
    )


def get_detector():
    try:
        return load_detector(landmark_path, cnn_path, ear_threshold, consec_frames, ear_weight, cnn_weight)
    except FileNotFoundError as exc:
        st.error(f"❌ {exc}")
        st.stop()


# --------------------------------------------------------------------------- #
# Mode tabs
# --------------------------------------------------------------------------- #
tab_webcam, tab_image, tab_video = st.tabs(["📷 Webcam", "🖼️ Image upload", "🎬 Video upload"])

# ---------- Webcam --------------------------------------------------------- #
with tab_webcam:
    st.markdown("Capture a frame from your webcam to test the detector.")
    cam_input = st.camera_input("Take a photo")
    if cam_input is not None:
        detector = get_detector()
        file_bytes = np.asarray(bytearray(cam_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = detector.process_frame(frame)
        annotated = detector.annotate(frame, result)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("EAR", f"{result['ear']:.3f}" if result['ear'] is not None else "—")
        c2.metric("CNN P(closed)",
                  f"{result['cnn_closed_prob']:.2f}" if result['cnn_closed_prob'] is not None else "n/a")
        c3.metric("Status", "DROWSY" if result["is_drowsy"] else "AWAKE")

# ---------- Image upload --------------------------------------------------- #
with tab_image:
    img_file = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        detector = get_detector()
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = detector.process_frame(frame)
        annotated = detector.annotate(frame, result)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_b:
            st.markdown(
                f"<div class='big-status {'drowsy' if result['is_drowsy'] else 'awake'}'>"
                f"{'DROWSY' if result['is_drowsy'] else 'AWAKE'}</div>",
                unsafe_allow_html=True,
            )
            st.metric("Eye Aspect Ratio",
                      f"{result['ear']:.3f}" if result['ear'] is not None else "—")
            st.metric("CNN P(closed)",
                      f"{result['cnn_closed_prob']:.2f}" if result['cnn_closed_prob'] is not None else "n/a")
            st.metric("Fused score", f"{result['fused_score']:.2f}")

# ---------- Video upload --------------------------------------------------- #
with tab_video:
    vid_file = st.file_uploader("Upload a video (mp4, avi, mov)", type=["mp4", "avi", "mov"])
    if vid_file is not None:
        detector = get_detector()
        # Reset history between videos
        detector.history.clear()
        detector.frame_counter = 0
        detector.alarm_on = False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vid_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        st.info(f"Video: {total_frames} frames @ {fps:.1f} FPS")

        progress = st.progress(0.0, text="Analyzing...")
        frame_box = st.empty()
        chart_box = st.empty()

        ears, cnn_probs, fused = [], [], []
        drowsy_frames = 0
        alarm_count = 0
        idx = 0
        sample_every = max(1, int(fps // 10))  # ~10 analyses per second

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                result = detector.process_frame(frame)
                ears.append(result["ear"] if result["ear"] is not None else np.nan)
                cnn_probs.append(result["cnn_closed_prob"] if result["cnn_closed_prob"] is not None else np.nan)
                fused.append(result["fused_score"])
                if result["is_drowsy"]:
                    drowsy_frames += 1
                if result["alarm"]:
                    alarm_count += 1
                if idx % (sample_every * 5) == 0:
                    annotated = detector.annotate(frame, result)
                    frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            idx += 1
            if total_frames > 0:
                progress.progress(min(idx / total_frames, 1.0), text=f"Analyzing frame {idx}/{total_frames}")
        cap.release()
        os.unlink(tmp_path)

        progress.empty()

        # Summary
        st.subheader("📊 Analysis Summary")
        analyzed = len(fused)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames analyzed", analyzed)
        c2.metric("Drowsy frames",
                  f"{drowsy_frames} ({100 * drowsy_frames / max(analyzed,1):.1f}%)")
        c3.metric("Alarm triggers", alarm_count)
        avg_ear = np.nanmean(ears) if ears else float("nan")
        c4.metric("Avg EAR", f"{avg_ear:.3f}" if not np.isnan(avg_ear) else "—")

        df = pd.DataFrame({
            "frame": np.arange(analyzed),
            "EAR": ears,
            "CNN P(closed)": cnn_probs,
            "Fused score": fused,
        }).set_index("frame")
        chart_box.line_chart(df)
