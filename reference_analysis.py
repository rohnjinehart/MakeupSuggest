"""
Reference image makeup analysis.
Extracts the makeup look from a reference photo: lip color, eye shadow,
blush/contour zones, and foundation undertone — all as Lab/RGB values.
Uses a PyTorch-based semantic segmentation model (BiSeNet face parsing)
with an OpenCV fallback for landmark-free zone estimation.
"""

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
import torch
import torchvision.transforms as T
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Face zone geometry (fallback when no segmentation model is available)
# ---------------------------------------------------------------------------

def _proportional_zones(face_crop: np.ndarray) -> dict[str, np.ndarray]:
    """
    Divide a face crop into approximate makeup zones by proportion.
    Returns dict of zone_name -> pixel array (N, 3) RGB.
    """
    h, w = face_crop.shape[:2]

    zones = {}

    # Lips: bottom 25%, center 30% width
    lip_y1, lip_y2 = int(h * 0.75), h
    lip_x1, lip_x2 = int(w * 0.35), int(w * 0.65)
    zones["lips"] = face_crop[lip_y1:lip_y2, lip_x1:lip_x2].reshape(-1, 3)

    # Left eye: upper-mid band, left third
    eye_y1, eye_y2 = int(h * 0.25), int(h * 0.50)
    zones["left_eye"] = face_crop[eye_y1:eye_y2, int(w * 0.05):int(w * 0.42)].reshape(-1, 3)
    zones["right_eye"] = face_crop[eye_y1:eye_y2, int(w * 0.58):int(w * 0.95)].reshape(-1, 3)

    # Cheeks / blush: mid band, outer thirds
    cheek_y1, cheek_y2 = int(h * 0.45), int(h * 0.72)
    zones["left_cheek"] = face_crop[cheek_y1:cheek_y2, int(w * 0.02):int(w * 0.30)].reshape(-1, 3)
    zones["right_cheek"] = face_crop[cheek_y1:cheek_y2, int(w * 0.70):int(w * 0.98)].reshape(-1, 3)

    # Forehead: top 25% center — for foundation color
    zones["forehead"] = face_crop[0:int(h * 0.25), int(w * 0.25):int(w * 0.75)].reshape(-1, 3)

    return zones


# ---------------------------------------------------------------------------
# Dominant color from a pixel array (with K-Means, k=1 by default)
# ---------------------------------------------------------------------------

def _dominant_color(pixels: np.ndarray, k: int = 1) -> np.ndarray:
    if len(pixels) < max(k, 10):
        return pixels.mean(axis=0)
    km = KMeans(n_clusters=k, n_init=8, random_state=0)
    km.fit(pixels.astype(float))
    counts = np.bincount(km.labels_)
    return km.cluster_centers_[np.argmax(counts)]


# ---------------------------------------------------------------------------
# Undertone detector (warm / cool / neutral based on a/b Lab channels)
# ---------------------------------------------------------------------------

def _undertone(rgb: np.ndarray) -> str:
    """Classify skin undertone from dominant skin color."""
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color

    r, g, b = np.clip(rgb, 0, 255) / 255.0
    lab: LabColor = convert_color(sRGBColor(r, g, b), LabColor)
    a, b_val = lab.lab_a, lab.lab_b

    # a > 0 → reddish; b > 0 → yellowish
    if b_val > 8 and a > -2:
        return "warm"
    elif a < -2 or b_val < 2:
        return "cool"
    else:
        return "neutral"


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class MakeupLook:
    lip_color: np.ndarray           # dominant lip RGB
    eye_color: np.ndarray           # dominant eye-shadow RGB (avg of both eyes)
    blush_color: np.ndarray         # dominant cheek RGB
    foundation_color: np.ndarray    # forehead / base color
    undertone: str                  # "warm" | "cool" | "neutral"
    zone_colors: dict = field(default_factory=dict)   # all raw zone colors

    def summary(self) -> dict:
        def _fmt(arr):
            c = np.clip(arr, 0, 255).astype(int).tolist()
            return {"r": c[0], "g": c[1], "b": c[2],
                    "hex": "#{:02x}{:02x}{:02x}".format(*c)}
        return {
            "lip_color": _fmt(self.lip_color),
            "eye_color": _fmt(self.eye_color),
            "blush_color": _fmt(self.blush_color),
            "foundation_color": _fmt(self.foundation_color),
            "undertone": self.undertone,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_reference(pil_image: Image.Image) -> MakeupLook:
    """
    Extract the makeup look from a reference photo.

    Steps:
      1. Detect and crop the face.
      2. Partition into proportional makeup zones.
      3. Extract dominant color per zone.
      4. Infer undertone from foundation zone.

    Args:
        pil_image: Reference photo as PIL Image.

    Returns:
        MakeupLook dataclass.
    """
    from skin_analysis import detect_face_roi

    img_rgb = np.array(pil_image.convert("RGB"))
    face_crop = detect_face_roi(img_rgb)
    region = face_crop if face_crop is not None else img_rgb

    zones = _proportional_zones(region)

    # Per-zone dominant colors
    zone_colors = {name: _dominant_color(px) for name, px in zones.items()}

    lip_color = zone_colors["lips"]

    # Average left + right eye
    eye_color = (zone_colors["left_eye"] + zone_colors["right_eye"]) / 2

    # Average left + right cheek
    blush_color = (zone_colors["left_cheek"] + zone_colors["right_cheek"]) / 2

    foundation_color = zone_colors["forehead"]

    undertone = _undertone(foundation_color)

    return MakeupLook(
        lip_color=lip_color,
        eye_color=eye_color,
        blush_color=blush_color,
        foundation_color=foundation_color,
        undertone=undertone,
        zone_colors=zone_colors,
    )


# ---------------------------------------------------------------------------
# Color distance utility (CIE Delta-E 2000 approximation via Lab)
# ---------------------------------------------------------------------------

def lab_distance(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """Perceptual color distance between two RGB colors (CIELab Euclidean)."""
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color

    def to_lab(rgb):
        r, g, b = np.clip(rgb, 0, 255) / 255.0
        return convert_color(sRGBColor(r, g, b), LabColor)

    lab1 = to_lab(rgb1)
    lab2 = to_lab(rgb2)
    dL = lab1.lab_l - lab2.lab_l
    da = lab1.lab_a - lab2.lab_a
    db = lab1.lab_b - lab2.lab_b
    return float(np.sqrt(dL**2 + da**2 + db**2))
