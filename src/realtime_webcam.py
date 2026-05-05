"""
Real-Time Webcam Drowsiness Monitor
====================================
Standalone OpenCV script - useful for demos when Streamlit is not available.

Usage:
    python src/realtime_webcam.py
    python src/realtime_webcam.py --camera 1 --sound

Press 'q' to quit, 's' to save a snapshot.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

from detector import DrowsinessDetector


def parse_args():
    p = argparse.ArgumentParser(description="Real-time drowsiness detection")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument(
        "--landmark",
        default=str(PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat"),
    )
    p.add_argument(
        "--cnn",
        default=str(PROJECT_ROOT / "models" / "eye_state_cnn.h5"),
    )
    p.add_argument("--ear_threshold", type=float, default=0.23)
    p.add_argument("--consec_frames", type=int, default=20)
    p.add_argument("--sound", action="store_true", help="Play a beep on alarm")
    p.add_argument("--snapshot_dir", default="demo")
    return p.parse_args()


def play_beep():
    """Cross-platform best-effort beep."""
    try:
        import sys
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1500, 250)
        else:
            print("\a", end="", flush=True)  # terminal bell
    except Exception:
        pass


def main():
    args = parse_args()
    os.makedirs(args.snapshot_dir, exist_ok=True)

    detector = DrowsinessDetector(
        shape_predictor_path=args.landmark,
        cnn_model_path=args.cnn,
        ear_threshold=args.ear_threshold,
        consec_frames=args.consec_frames,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("[INFO] Press 'q' to quit, 's' for snapshot.")
    last_alarm_beep = 0.0
    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.process_frame(frame)
        annotated = detector.annotate(frame, result)

        # FPS overlay
        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps_display = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()
        cv2.putText(annotated, f"FPS: {fps_display:.1f}",
                    (10, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if args.sound and result["alarm"] and time.time() - last_alarm_beep > 1.0:
            play_beep()
            last_alarm_beep = time.time()

        cv2.imshow("Drowsiness Detection", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = os.path.join(args.snapshot_dir, f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg")
            cv2.imwrite(fname, annotated)
            print(f"[INFO] Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
