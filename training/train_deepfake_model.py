import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# If a local TensorFlow install exists at C:\tf_temp, prefer it over a
# potentially broken user-site install (Windows long-path issue).
_tf_local = r"C:\tf_temp"
if os.path.isdir(_tf_local) and _tf_local not in sys.path:
    sys.path.insert(0, _tf_local)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from backend.models.deepfake_detection_engine import build_model, load_training_videos, IMG_SIZE

DATA_DIR = os.path.join(project_root, "backend", "data")
SAVE_PATH = os.path.join(project_root, "backend", "saved_models", "deepfake_detector.h5")

# ── 1. Load frames from real/ and fake/ ──────────────────────────────────────
print("Loading training data...")
videos, labels = load_training_videos(DATA_DIR)

if len(videos) == 0:
    print("ERROR: No training data found.")
    print(f"Place real media files in:  {os.path.join(DATA_DIR, 'real')}")
    print(f"Place fake media files in:  {os.path.join(DATA_DIR, 'fake')}")
    sys.exit(1)

# Normalise pixel values to [0, 1]
videos = videos.astype("float32") / 255.0

print(f"Loaded {len(videos)} frames  |  Real: {int(np.sum(labels == 0))}  Fake: {int(np.sum(labels == 1))}")

# ── 2. Train / validation split ──────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    videos, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train: {len(X_train)} frames  |  Val: {len(X_val)} frames")

# ── 3. Data augmentation ─────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    brightness_range=(0.9, 1.1),
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# ── 4. Build & train ─────────────────────────────────────────────────────────
model = build_model()
model.summary()

EPOCHS = 100

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
)

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation loss: {loss:.4f}")
print(f"Validation accuracy: {acc:.4f}")

# ── 6. Save ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
model.save(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
