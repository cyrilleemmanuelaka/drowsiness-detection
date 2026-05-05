"""
Hybrid Drowsiness Detection System
Combines Eye Aspect Ratio (EAR) with a CNN classifier for robust detection.

Author: Drowsiness Detection Project
"""

import bz2
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import dlib
from collections import deque
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model


class DrowsinessDetector:
    """
    Hybrid drowsiness detector that combines:
      1. Geometric Eye Aspect Ratio (EAR) - fast, interpretable
      2. CNN binary classifier (open/closed eyes) - robust to lighting/pose
    
    The final decision is a weighted vote between both signals,
    smoothed over a temporal window to reduce false positives.
    """

    # Indices of the facial landmarks for left and right eyes (68-point model)
    LEFT_EYE_IDX = list(range(36, 42))
    RIGHT_EYE_IDX = list(range(42, 48))

    def __init__(
        self,
        shape_predictor_path: str = "models/shape_predictor_68_face_landmarks.dat",
        cnn_model_path: str = "models/eye_state_cnn.h5",
        ear_threshold: float = 0.23,
        consec_frames: int = 20,
        cnn_input_size: tuple = (24, 24),
        ear_weight: float = 0.5,
        cnn_weight: float = 0.5,
    ):
        """
        Args:
            shape_predictor_path: dlib 68-point landmark model.
            cnn_model_path:       Trained Keras CNN model file.
            ear_threshold:        EAR below this is considered "closed".
            consec_frames:        Number of consecutive closed-eye frames to trigger alert.
            cnn_input_size:       Input shape expected by the CNN (H, W).
            ear_weight, cnn_weight: Fusion weights (should sum to 1.0).
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.cnn_input_size = cnn_input_size
        self.ear_weight = ear_weight
        self.cnn_weight = cnn_weight

        # Detector + landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        shape_predictor_path = self._resolve_model_path(shape_predictor_path)
        cnn_model_path = self._resolve_model_path(cnn_model_path)

        if not os.path.exists(shape_predictor_path):
            resolved_path = Path(shape_predictor_path)
            if not self._attempt_download_shape_predictor(resolved_path):
                raise FileNotFoundError(
                    f"Landmark model not found at {shape_predictor_path}. "
                    "Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                )
        self.landmark_predictor = dlib.shape_predictor(shape_predictor_path)

        # CNN eye-state classifier (optional - falls back to EAR-only if missing)
        self.cnn = None
        if os.path.exists(cnn_model_path):
            try:
                self.cnn = load_model(cnn_model_path, compile=False)
                print(f"[INFO] Loaded CNN model from {cnn_model_path}")
            except Exception as exc:
                print(f"[WARN] Could not load CNN model: {exc}. Falling back to EAR only.")
        else:
            print(f"[WARN] CNN model not found at {cnn_model_path}. Running in EAR-only mode.")

        # Runtime state
        self.frame_counter = 0
        self.alarm_on = False
        self.history = deque(maxlen=30)  # Smoothing buffer for the fused score

    # ------------------------------------------------------------------ #
    # Geometric features
    # ------------------------------------------------------------------ #
    @staticmethod
    def eye_aspect_ratio(eye: np.ndarray) -> float:
        """
        Compute the Eye Aspect Ratio (EAR) defined by Soukupová & Čech (2016).
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        a = dist.euclidean(eye[1], eye[5])
        b = dist.euclidean(eye[2], eye[4])
        c = dist.euclidean(eye[0], eye[3])
        return (a + b) / (2.0 * c)

    @staticmethod
    def shape_to_np(shape) -> np.ndarray:
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    @classmethod
    def _resolve_model_path(cls, path: str) -> str:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate)

        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            return str(cwd_candidate)

        repo_candidate = Path(__file__).resolve().parent.parent / candidate
        if repo_candidate.exists():
            return str(repo_candidate)

        model_name_candidate = Path(__file__).resolve().parent.parent / "models" / candidate.name
        if model_name_candidate.exists():
            return str(model_name_candidate)

        return str(candidate)

    @staticmethod
    def _attempt_download_shape_predictor(dst: Path) -> bool:
        if dst.name != "shape_predictor_68_face_landmarks.dat":
            return False
        if dst.exists():
            return True

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_bz2 = dst.with_suffix(dst.suffix + ".bz2")
        try:
            print(f"[INFO] Downloading dlib shape predictor to {dst} ...")
            urllib.request.urlretrieve(
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                tmp_bz2,
            )
            with bz2.open(tmp_bz2, "rb") as compressed, open(dst, "wb") as out_file:
                out_file.write(compressed.read())
            return dst.exists()
        except Exception as exc:
            print(f"[WARN] Auto-download of shape predictor failed: {exc}")
            return False
        finally:
            try:
                if tmp_bz2.exists():
                    tmp_bz2.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # CNN feature
    # ------------------------------------------------------------------ #
    def _extract_eye_roi(self, gray: np.ndarray, eye_pts: np.ndarray) -> np.ndarray:
        """Crop a tight, square-ish ROI around an eye and resize for the CNN."""
        x, y, w, h = cv2.boundingRect(eye_pts)
        pad = int(0.3 * max(w, h))
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, gray.shape[1])
        y2 = min(y + h + pad, gray.shape[0])
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        roi = cv2.resize(roi, self.cnn_input_size)
        roi = roi.astype("float32") / 255.0
        return roi.reshape(*self.cnn_input_size, 1)

    def _cnn_eye_score(self, gray: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """
        Returns probability that BOTH eyes are CLOSED according to the CNN.
        Output is in [0, 1] where 1 = definitely closed.
        """
        if self.cnn is None:
            return None

        rois = []
        for eye in (left_eye, right_eye):
            roi = self._extract_eye_roi(gray, eye)
            if roi is not None:
                rois.append(roi)
        if not rois:
            return None

        batch = np.stack(rois, axis=0)
        preds = self.cnn.predict(batch, verbose=0)
        # Convention: model outputs P(closed). Average over both eyes.
        if preds.shape[-1] == 2:
            closed_prob = preds[:, 0].mean()  # softmax with [closed, open]
        else:
            closed_prob = preds.flatten().mean()  # sigmoid -> P(closed)
        return float(closed_prob)

    # ------------------------------------------------------------------ #
    # Main inference
    # ------------------------------------------------------------------ #
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single BGR frame and return a result dict:
            {
              'face_found': bool,
              'ear': float | None,
              'cnn_closed_prob': float | None,
              'fused_score': float in [0,1],   # 1 = drowsy
              'is_drowsy': bool,
              'alarm': bool,
              'landmarks': np.ndarray | None,
              'left_eye': np.ndarray | None,
              'right_eye': np.ndarray | None,
            }
        """
        result = {
            "face_found": False,
            "ear": None,
            "cnn_closed_prob": None,
            "fused_score": 0.0,
            "is_drowsy": False,
            "alarm": False,
            "landmarks": None,
            "left_eye": None,
            "right_eye": None,
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 0)

        if len(faces) == 0:
            self.history.append(0.0)
            return result

        # Use the largest face only
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.landmark_predictor(gray, face)
        landmarks = self.shape_to_np(shape)

        left_eye = landmarks[self.LEFT_EYE_IDX]
        right_eye = landmarks[self.RIGHT_EYE_IDX]

        ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
        ear_closed = 1.0 if ear < self.ear_threshold else 0.0

        cnn_prob = self._cnn_eye_score(gray, left_eye, right_eye)

        # ---- Late fusion -------------------------------------------------
        if cnn_prob is None:
            fused = ear_closed
        else:
            fused = self.ear_weight * ear_closed + self.cnn_weight * cnn_prob

        self.history.append(fused)
        smoothed = float(np.mean(self.history))

        is_drowsy = smoothed > 0.5
        if is_drowsy:
            self.frame_counter += 1
            if self.frame_counter >= self.consec_frames:
                self.alarm_on = True
        else:
            self.frame_counter = max(0, self.frame_counter - 2)
            if self.frame_counter == 0:
                self.alarm_on = False

        result.update(
            face_found=True,
            ear=float(ear),
            cnn_closed_prob=cnn_prob,
            fused_score=smoothed,
            is_drowsy=is_drowsy,
            alarm=self.alarm_on,
            landmarks=landmarks,
            left_eye=left_eye,
            right_eye=right_eye,
        )
        return result

    # ------------------------------------------------------------------ #
    # Visualization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def annotate(frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw bounding boxes, eye contours, and status onto the frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        if not result["face_found"]:
            cv2.putText(out, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return out

        # Eye contours
        for eye in (result["left_eye"], result["right_eye"]):
            hull = cv2.convexHull(eye)
            color = (0, 0, 255) if result["is_drowsy"] else (0, 255, 0)
            cv2.drawContours(out, [hull], -1, color, 1)

        # Status panel
        status = "DROWSY" if result["is_drowsy"] else "AWAKE"
        status_color = (0, 0, 255) if result["is_drowsy"] else (0, 200, 0)
        cv2.rectangle(out, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(out, f"Status: {status}", (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # Metrics
        ear_txt = f"EAR: {result['ear']:.3f}" if result["ear"] is not None else "EAR: --"
        cnn_txt = (f"CNN(closed): {result['cnn_closed_prob']:.2f}"
                   if result["cnn_closed_prob"] is not None else "CNN: n/a")
        cv2.putText(out, ear_txt, (w - 360, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out, cnn_txt, (w - 200, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Alarm banner
        if result["alarm"]:
            cv2.rectangle(out, (0, h - 60), (w, h), (0, 0, 200), -1)
            cv2.putText(out, "!!! DROWSINESS ALERT - WAKE UP !!!",
                        (w // 2 - 260, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out
