"""
Skin tone analysis using PyTorch + face detection.
Extracts dominant skin colors from face regions and maps them
to the Fitzpatrick scale and perceptual Lab color space.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms, models
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from typing import Optional


# ---------------------------------------------------------------------------
# Lightweight CNN head for skin-pixel classification
# (fine-tunes MobileNetV3-Small backbone; falls back to heuristic if no GPU)
# ---------------------------------------------------------------------------

class SkinSegmenter(nn.Module):
    """Binary classifier: skin pixel vs. non-skin pixel."""

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Replace classifier head for binary output
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, 2)
        self.backbone = backbone
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Face detector wrapper (uses OpenCV Haar cascade — no license restrictions)
# ---------------------------------------------------------------------------

def detect_face_roi(image: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest face crop or None if no face detected."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # Largest face by area
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    # Slight inward crop to reduce hair / background bleed
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.10)
    x1 = max(0, x + pad_x)
    y1 = max(0, y + pad_y)
    x2 = min(image.shape[1], x + w - pad_x)
    y2 = min(image.shape[0], y + h - pad_y)
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Heuristic skin-pixel mask (YCrCb range — robust, no training needed)
# ---------------------------------------------------------------------------

def skin_mask_ycrcb(image_rgb: np.ndarray) -> np.ndarray:
    """Return boolean mask of likely skin pixels."""
    img_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(img_ycrcb, lower, upper)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Dominant color extraction via K-Means
# ---------------------------------------------------------------------------

def extract_dominant_colors(pixels_rgb: np.ndarray, k: int = 3) -> list[np.ndarray]:
    """Cluster skin pixels and return k dominant RGB colors (most common first)."""
    if len(pixels_rgb) < k:
        return [pixels_rgb.mean(axis=0)]
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(pixels_rgb)
    centers = km.cluster_centers_
    counts = np.bincount(km.labels_)
    order = np.argsort(-counts)
    return [centers[i] for i in order]


# ---------------------------------------------------------------------------
# Fitzpatrick scale mapper
# ---------------------------------------------------------------------------

FITZPATRICK_SCALE = [
    # (label, L* range in CIELab)
    ("I — Very Fair",     (75, 100)),
    ("II — Fair",         (65, 75)),
    ("III — Medium",      (55, 65)),
    ("IV — Olive/Medium Dark", (45, 55)),
    ("V — Brown",         (35, 45)),
    ("VI — Dark Brown/Black", (0, 35)),
]


def rgb_to_lab(rgb: np.ndarray) -> LabColor:
    r, g, b = rgb / 255.0
    srgb = sRGBColor(r, g, b, is_upscaled=False)
    return convert_color(srgb, LabColor)


def fitzpatrick_type(dominant_rgb: np.ndarray) -> str:
    lab = rgb_to_lab(dominant_rgb)
    L = lab.lab_l
    for label, (lo, hi) in FITZPATRICK_SCALE:
        if lo <= L < hi:
            return label
    return FITZPATRICK_SCALE[-1][0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SkinAnalysisResult:
    def __init__(self, dominant_colors: list, fitzpatrick: str,
                 lab_values: list, face_crop: Optional[np.ndarray]):
        self.dominant_colors = dominant_colors        # list of np.ndarray RGB uint8
        self.fitzpatrick = fitzpatrick                # Fitzpatrick type string
        self.lab_values = lab_values                  # list of (L, a, b) tuples
        self.face_crop = face_crop                    # cropped face array or None

    def primary_rgb(self) -> np.ndarray:
        return np.clip(self.dominant_colors[0], 0, 255).astype(np.uint8)

    def primary_lab(self) -> tuple:
        return self.lab_values[0]


def analyze_skin(pil_image: Image.Image, n_colors: int = 3) -> SkinAnalysisResult:
    """
    Full pipeline: face detect → skin mask → dominant colors → Fitzpatrick type.

    Args:
        pil_image: Input PIL image (any mode).
        n_colors: Number of dominant skin tones to extract.

    Returns:
        SkinAnalysisResult
    """
    img_rgb = np.array(pil_image.convert("RGB"))

    # 1. Face region
    face_crop = detect_face_roi(img_rgb)
    region = face_crop if face_crop is not None else img_rgb

    # 2. Skin mask
    mask = skin_mask_ycrcb(region)
    skin_pixels = region[mask]

    if len(skin_pixels) < 10:
        # Fallback: use entire face/image center strip
        h, w = region.shape[:2]
        center = region[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        skin_pixels = center.reshape(-1, 3)

    # 3. Dominant colors
    dominant = extract_dominant_colors(skin_pixels.astype(float), k=n_colors)

    # 4. Lab conversion & Fitzpatrick
    lab_values = []
    for rgb in dominant:
        lab = rgb_to_lab(rgb)
        lab_values.append((lab.lab_l, lab.lab_a, lab.lab_b))

    fitzpatrick = fitzpatrick_type(dominant[0])

    return SkinAnalysisResult(
        dominant_colors=dominant,
        fitzpatrick=fitzpatrick,
        lab_values=lab_values,
        face_crop=face_crop,
    )
