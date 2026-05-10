"""
Deepfake Detection Engine — ONNX Runtime edition.

Uses a quantized ONNX ViT model from HuggingFace, run locally with the very
lightweight `onnxruntime` package. Designed to fit Render's 512 MB free tier:

  - Model file:        ~89 MB on disk (INT8-quantized ViT base/32).
  - onnxruntime CPU:   ~25-40 MB resident.
  - Inference RAM:     ~60-100 MB peak per request (224x224x3 input).

Total stays well under torch+transformers (~600 MB) which OOMs on free tier.

Public API (kept compatible with backend/app.py):
  - load_model() -> object         # the detector instance
  - predict_media(model, path)     # dict result with label/fake_score/per_frame

Required env vars: none. The model is downloaded on first startup from the
HuggingFace public repo (no auth needed). Set DEEPFAKE_MODEL_PATH to a local
file to skip the download (useful in CI / offline tests).
"""

from __future__ import annotations

import io
import os
import urllib.request
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# ── Config ─────────────────────────────────────────────────────────────────
# Quantized ViT (~89 MB). Other variants in the same repo if you want
# different size/accuracy tradeoffs (model_q4.onnx ~63 MB, model.onnx ~350 MB).
DEFAULT_MODEL_URL = os.getenv(
    "DEEPFAKE_MODEL_URL",
    "https://huggingface.co/prithivMLmods/Deepfake-Detection-Exp-02-22-ONNX"
    "/resolve/main/onnx/model_quantized.onnx",
)
# Where to cache the downloaded model. Render's filesystem is ephemeral but
# persists for the life of the running container, so re-downloads only happen
# on cold starts.
DEFAULT_MODEL_DIR = os.getenv(
    "DEEPFAKE_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models"),
)
DEFAULT_MODEL_FILENAME = "deepfake_vit_quantized.onnx"

# id2label from the model config: 0 = Deepfake, 1 = Real.
LABEL_FAKE_INDEX = 0

# Frames sampled per video. Each frame is a separate ONNX forward pass.
DEFAULT_VIDEO_FRAMES = 15
HTTP_TIMEOUT = int(os.getenv("DEEPFAKE_HTTP_TIMEOUT", "120"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ViT preprocessing constants (from preprocessor_config.json).
_IMAGE_SIZE = 224
_IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)


# ── Helpers ────────────────────────────────────────────────────────────────
def _ensure_model_file(model_dir: str, filename: str, url: str) -> str:
    """Download the ONNX model if it's not already cached locally."""
    path = os.environ.get("DEEPFAKE_MODEL_PATH")
    if path:
        if not os.path.exists(path):
            raise RuntimeError(f"DEEPFAKE_MODEL_PATH points to missing file: {path}")
        return path

    os.makedirs(model_dir, exist_ok=True)
    target = os.path.join(model_dir, filename)
    if os.path.exists(target) and os.path.getsize(target) > 1_000_000:
        return target

    print(f"[Deepfake] Downloading ONNX model from {url} -> {target}")
    # Stream to a temp file then rename, so a partial download doesn't get
    # cached as a valid model.
    tmp = target + ".part"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    print(f"[Deepfake] Cached model ({os.path.getsize(target):,} bytes)")
    return target


def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize to 224x224, RGB, scale to [0,1], normalize, NCHW float32."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    # PIL.Image.BILINEAR == 2, matches preprocessor_config "resample": 2.
    image = image.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0      # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)[None, ...]                # 1xCxHxW
    arr = (arr - _IMAGE_MEAN) / _IMAGE_STD
    return arr.astype(np.float32, copy=False)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


# ── Detector ───────────────────────────────────────────────────────────────
class ONNXDeepfakeDetector:
    """Lightweight ONNX-based deepfake classifier (CPU, INT8 quantized ViT)."""

    def __init__(
        self,
        model_path: str | None = None,
        model_url: str = DEFAULT_MODEL_URL,
        model_dir: str = DEFAULT_MODEL_DIR,
        model_filename: str = DEFAULT_MODEL_FILENAME,
    ):
        if model_path is None:
            model_path = _ensure_model_file(model_dir, model_filename, model_url)

        # Single-thread CPU keeps RAM low and avoids contention with other
        # endpoints on the same Render dyno.
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _score_pil(self, image: Image.Image) -> float:
        """Return P(fake) in [0, 1] for a single PIL image."""
        x = _preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        probs = _softmax(logits)[0]
        return float(probs[LABEL_FAKE_INDEX])

    def predict_image(self, image: Image.Image) -> dict:
        score = self._score_pil(image)
        return {
            "fake_score": round(score, 4),
            "label": "Fake" if score > 0.5 else "Real",
            "frame_count": 1,
            "per_frame": [round(score, 4)],
            "average_confidence": round(score if score > 0.5 else 1.0 - score, 4),
        }

    def predict_video(self, video_path: str, num_frames: int | None = None) -> dict:
        if num_frames is None:
            num_frames = int(os.getenv("DEEPFAKE_VIDEO_FRAMES", str(DEFAULT_VIDEO_FRAMES)))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "fake_score": 0.0,
                "label": "Unknown",
                "frame_count": 0,
                "per_frame": [],
                "average_confidence": 0.0,
                "error": f"Could not open video: {os.path.basename(video_path)}",
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return {
                "fake_score": 0.0,
                "label": "Unknown",
                "frame_count": 0,
                "per_frame": [],
                "average_confidence": 0.0,
                "error": "Video contains no frames.",
            }

        step = max(1, total_frames // num_frames)
        frame_indices = [i * step for i in range(min(num_frames, total_frames))]

        per_frame: List[float] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            per_frame.append(self._score_pil(pil_image))

        cap.release()

        if not per_frame:
            return {
                "fake_score": 0.0,
                "label": "Unknown",
                "frame_count": 0,
                "per_frame": [],
                "average_confidence": 0.0,
                "error": "No frames could be decoded from the video.",
            }

        avg_score = float(np.mean(per_frame))
        is_fake = avg_score > 0.5
        confidence = avg_score if is_fake else 1.0 - avg_score
        fake_frames = int(sum(1 for s in per_frame if s > 0.5))

        return {
            "fake_score": round(avg_score, 4),
            "label": "Fake" if is_fake else "Real",
            "frame_count": len(per_frame),
            "per_frame": [round(s, 4) for s in per_frame],
            "average_confidence": round(confidence, 4),
            "fake_frames_detected": f"{fake_frames}/{len(per_frame)}",
        }


# ── Backwards-compatible module-level API expected by app.py ───────────────
def load_model() -> ONNXDeepfakeDetector:
    """Instantiate the ONNX detector (downloads model on first call)."""
    return ONNXDeepfakeDetector()


def predict_media(model: ONNXDeepfakeDetector, file_path: str) -> dict:
    """Route an uploaded file to image- or video-prediction based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in IMAGE_EXTS:
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return {
                "fake_score": 0.0,
                "label": "Unknown",
                "frame_count": 0,
                "per_frame": [],
                "average_confidence": 0.0,
                "error": f"Could not read image: {e}",
            }
        return model.predict_image(image)

    if ext in VIDEO_EXTS:
        return model.predict_video(file_path)

    return {
        "fake_score": 0.0,
        "label": "Unknown",
        "frame_count": 0,
        "per_frame": [],
        "average_confidence": 0.0,
        "error": f"Unsupported file format: {ext}",
    }
