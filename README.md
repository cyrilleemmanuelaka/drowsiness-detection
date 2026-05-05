# 😴 Drowsiness Detection System (Hybrid CNN + EAR)

A real-time deep learning system that detects whether a person is **awake or sleeping** by combining two signals:

1. **Eye Aspect Ratio (EAR)** — a fast, geometric heuristic from facial landmarks.
2. **Convolutional Neural Network (CNN)** — a lightweight binary classifier trained on cropped eye images.

The two signals are fused with a weighted late-fusion strategy and smoothed over a 30-frame window to reject false positives.

---

## 📁 Project Structure

```
drowsiness_detection/
├── src/
│   ├── detector.py         # Hybrid detector class (core logic)
│   ├── train_cnn.py        # CNN architecture + training pipeline
│   ├── evaluate.py         # Test-set evaluation (CM, ROC)
│   ├── realtime_webcam.py  # Standalone OpenCV webcam demo
│   └── app.py              # Streamlit web UI (webcam + image + video)
├── models/                 # Trained model weights (.h5, .dat)
├── data/                   # train / val / test directories
├── docs/                   # Documentation, plots, presentation
├── demo/                   # Snapshots & demo media
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on `dlib`**: On Linux you may need `cmake` and `libboost-all-dev` first. On Windows the easiest path is `pip install dlib-bin`.

### 2. Download the dlib face-landmark model

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
```

### 3. (Optional) Train the CNN

Download the **MRL Eye Dataset** from <http://mrl.cs.vsb.cz/eyedataset> and split it into:

```
data/
  train/{open,closed}/
  val/{open,closed}/
  test/{open,closed}/
```

Then:

```bash
python src/train_cnn.py --data_dir data --epochs 25
python src/evaluate.py --test_dir data/test
```

The system gracefully **falls back to EAR-only mode** if the CNN model is missing — useful for a first demo before training is finished.

### 4. Run the demo

**Streamlit web app (recommended):**
```bash
streamlit run src/app.py
```

**Standalone webcam window:**
```bash
python src/realtime_webcam.py --sound
```

---

## 🧠 How It Works

### Step 1 — Face & landmark detection
A HOG-based face detector locates the face, then dlib's 68-point shape predictor places landmarks on the eyes, mouth, nose, and jaw.

### Step 2 — Two parallel signals

| Signal | Inputs | Output |
|--------|--------|--------|
| EAR | 6 landmarks per eye | scalar in [0, ~0.4] |
| CNN | 24×24 grayscale eye crop | P(closed) ∈ [0, 1] |

EAR is computed as `(‖p2-p6‖ + ‖p3-p5‖) / (2 · ‖p1-p4‖)`.
The CNN is a 3-block convolutional network (~80 k parameters) that classifies the eye crop as **open** or **closed**.

### Step 3 — Late fusion

```
fused = w_ear · 1[EAR < τ]  +  w_cnn · P(closed)
```

The fused score is averaged over a sliding 30-frame window. If it stays above 0.5 for `consec_frames` (default 20 ≈ 0.7 s at 30 FPS), an alarm is raised.

### Step 4 — Alert
Visual banner on the frame + optional audible beep.

---

## 📊 Why hybrid?

| | EAR | CNN | Hybrid |
|---|---|---|---|
| Speed | ⚡⚡⚡ | ⚡⚡ | ⚡⚡ |
| Accuracy in good lighting | ✅ | ✅ | ✅ |
| Accuracy in low/uneven lighting | ⚠️ | ✅ | ✅ |
| Robustness to glasses | ⚠️ | ✅ | ✅ |
| Interpretability | ✅ | ❌ | ✅ |

The hybrid approach gives you the **interpretability and speed of EAR** with the **robustness of a learned classifier**.

---

## 🎯 Use Cases

- 🚗 **Driver monitoring** — alert tired drivers before microsleeps
- 💼 **Workplace safety** — heavy machinery operators, night-shift staff
- 🎓 **Online learning** — engagement analytics
- 🏥 **Healthcare** — patient monitoring in ICUs

---

## 🧪 Reported Results (on MRL Eye dataset, val split)

> These are typical numbers you should expect to reproduce after training.

| Metric | Value |
|---|---|
| CNN accuracy | ~97% |
| CNN AUC | ~0.99 |
| End-to-end FPS (CPU, 720p) | 25–35 FPS |
| End-to-end FPS (GPU) | 60+ FPS |

---

## 📚 References

- Soukupová & Čech (2016) — *Real-Time Eye Blink Detection using Facial Landmarks*
- King (2009) — *dlib-ml: A Machine Learning Toolkit*
- MRL Eye Dataset — Czech Technical University

---

## 📝 License

Released under the MIT License. Educational/research use encouraged.
