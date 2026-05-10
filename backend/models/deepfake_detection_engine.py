"""
Deepfake Detection Engine — ViT (Vision Transformer) edition.

Uses the pre-trained `Sxhni/deepfake-detector-vit` model from HuggingFace
to score images and videos for deepfake manipulation. No local training
data or .h5 weights are required; weights are downloaded on first run
and cached under ~/.cache/huggingface.

Public API (kept compatible with backend/app.py):
  - load_model() -> object         # the detector instance
  - predict_media(model, path)     # dict result with label/fake_score/per_frame
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("DEEPFAKE_MODEL_NAME", "Sxhni/deepfake-detector-vit")
# How many frames to sample from a video for majority-vote scoring.
# Higher -> more accurate but slower.
DEFAULT_VIDEO_FRAMES = int(os.getenv("DEEPFAKE_VIDEO_FRAMES", "15"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ── Detector ───────────────────────────────────────────────────────────────
class UnifiedDeepfakeDetector:
    """Wraps a pre-trained ViT classifier for image & video deepfake scoring."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Figure out which class index corresponds to "fake".
        # The model's id2label mapping varies (e.g. {"0": "Real", "1": "Fake"}
        # or {"0": "REAL", "1": "FAKE"}), so resolve it once up-front.
        self._fake_idx = self._resolve_fake_idx()

    def _resolve_fake_idx(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if "fake" in str(label).lower():
                return int(idx)
        # Fallback: assume binary classifier with index 1 == fake.
        return 1

    @torch.no_grad()
    def _score_pil(self, image: Image.Image) -> float:
        """Return P(fake) in [0,1] for a single PIL image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        return float(probs[self._fake_idx].item())

    def predict_image(self, image: Image.Image) -> dict:
        score = self._score_pil(image)
        return {
            "fake_score": round(score, 4),
            "label": "Fake" if score > 0.5 else "Real",
            "frame_count": 1,
            "per_frame": [round(score, 4)],
            "average_confidence": round(score if score > 0.5 else 1.0 - score, 4),
        }

    def predict_video(self, video_path: str, num_frames: int = DEFAULT_VIDEO_FRAMES) -> dict:
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
        # Confidence = how strongly the average leans toward its decision.
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
def load_model() -> UnifiedDeepfakeDetector:
    """Instantiate (and warm) the detector. Heavy: downloads weights on first call."""
    return UnifiedDeepfakeDetector()


def predict_media(model: UnifiedDeepfakeDetector, file_path: str) -> dict:
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
