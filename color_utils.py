"""
Shared color utility functions used across modules.
"""

import numpy as np
from PIL import Image, ImageDraw


def rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(np.clip(v, 0, 255)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb_safe(hex_str: str) -> np.ndarray:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        return np.array([128, 128, 128], dtype=np.uint8)
    try:
        return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    except ValueError:
        return np.array([128, 128, 128], dtype=np.uint8)


def make_palette_image(colors: list[np.ndarray], size: int = 50, gap: int = 4) -> Image.Image:
    """Create a horizontal palette image from a list of RGB arrays."""
    n = len(colors)
    w = n * size + (n - 1) * gap
    img = Image.new("RGB", (w, size), (240, 240, 240))
    for i, color in enumerate(colors):
        swatch = Image.new("RGB", (size, size), tuple(int(v) for v in color))
        img.paste(swatch, (i * (size + gap), 0))
    return img


def color_distance_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """Simple Euclidean RGB distance (0–441)."""
    return float(np.linalg.norm(a.astype(float) - b.astype(float)))


def closest_color(target: np.ndarray, candidates: list[np.ndarray]) -> tuple[int, float]:
    """Return index and distance of the closest color in candidates to target."""
    dists = [color_distance_rgb(target, c) for c in candidates]
    idx = int(np.argmin(dists))
    return idx, dists[idx]
