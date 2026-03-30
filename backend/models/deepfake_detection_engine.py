"""
Deepfake Detection Engine
- build_model(): returns a compiled Keras CNN for 128x128 frame classification
- extract_frames(): pulls frames from an image or video file
- predict_media(): scores uploaded media and returns per-frame + aggregate results
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

IMG_SIZE = 128
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "saved_models", "deepfake_detector.h5"
)


def build_model():
    """Return a compiled deepfake detection CNN."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_model():
    """Load from disk if a saved model exists, otherwise return a freshly compiled model."""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return build_model()


def extract_frames(file_path, max_frames=30):
    """
    Extract frames from a video or a single frame from an image.
    Returns a list of (IMG_SIZE, IMG_SIZE, 3) numpy arrays normalised to [0, 1].
    """
    ext = os.path.splitext(file_path)[1].lower()
    frames = []

    # Image file
    if ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'):
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img.astype(np.float32) / 255.0)
        return frames

    # Video file
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample evenly across the video to stay within max_frames
    step = max(1, total_frames // max_frames) if total_frames > 0 else 1
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        frame_idx += 1

    cap.release()
    return frames


def predict_media(model, file_path):
    """
    Run deepfake detection on a media file.
    Returns dict with:
      - fake_score:  0.0 (definitely real) -> 1.0 (definitely fake)
      - label:       'Real' | 'Fake'
      - frame_count: how many frames were analysed
      - per_frame:   list of per-frame scores
    """
    frames = extract_frames(file_path)

    if not frames:
        return {
            "fake_score": 0.0,
            "label": "Unknown",
            "frame_count": 0,
            "per_frame": [],
            "error": "Could not extract any frames from the uploaded file.",
        }

    batch = np.array(frames)                      # (N, 128, 128, 3)
    predictions = model.predict(batch, verbose=0)  # (N, 1)
    per_frame = [float(p[0]) for p in predictions]
    avg_score = float(np.mean(per_frame))

    return {
        "fake_score": round(avg_score, 4),
        "label": "Fake" if avg_score > 0.5 else "Real",
        "frame_count": len(frames),
        "per_frame": [round(s, 4) for s in per_frame],
    }


# ---------------------------------------------------------------------------
# Training helper (only used by training scripts, never imported by the API)
# ---------------------------------------------------------------------------
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


def load_training_videos(path, max_frames_per_video=30):
    """
    Load labelled frames from real/ and fake/ subdirectories.
    Supports both image files (one frame each) and video files
    (up to *max_frames_per_video* evenly-sampled frames).
    """
    frames = []
    labels = []

    for category in ['real', 'fake']:
        label = 0 if category == 'real' else 1
        category_path = os.path.join(path, category)
        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            ext = os.path.splitext(filename)[1].lower()

            # ── Image file ────────────────────────────────────────────
            if ext in IMAGE_EXTS:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
                labels.append(label)

            # ── Video file ────────────────────────────────────────────
            elif ext in VIDEO_EXTS:
                cap = cv2.VideoCapture(file_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // max_frames_per_video) if total > 0 else 1
                count, sampled = 0, 0

                while cap.isOpened() and sampled < max_frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % step == 0:
                        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        labels.append(label)
                        sampled += 1
                    count += 1

                cap.release()

    if len(frames) == 0:
        return np.array([]), np.array([])

    return np.array(frames), np.array(labels)
