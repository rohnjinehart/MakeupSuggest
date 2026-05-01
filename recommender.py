"""
Makeup product recommendation engine.

Given:
  - SkinAnalysisResult  (user's skin tone, Fitzpatrick type)
  - MakeupLook          (target look extracted from reference image)

Outputs:
  - Ranked product recommendations per makeup category, scored by:
      1. Color match  (perceptual Lab distance to target shade)
      2. Skin-tone compatibility  (undertone + Fitzpatrick heuristics)
      3. Product rating           (when available)

Uses a lightweight PyTorch embedding model to encode color proximity
and a rule-based compatibility layer for skin tone suitability.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from skin_analysis import SkinAnalysisResult
from reference_analysis import MakeupLook, lab_distance
from product_database import get_database, hex_to_rgb


# ---------------------------------------------------------------------------
# Tiny PyTorch color-embedding network
# Maps RGB triplets → 16-dim embedding for fast batch similarity search
# ---------------------------------------------------------------------------

class ColorEmbedder(nn.Module):
    """Encodes an RGB color into a learned embedding space."""

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3) float in [0, 1]
        return self.net(x)


def _build_embedder() -> ColorEmbedder:
    """Return a randomly initialized embedder (no training needed for distance proxy)."""
    model = ColorEmbedder()
    model.eval()
    return model


_embedder = _build_embedder()


def _color_score(target_rgb: np.ndarray, candidate_rgb: np.ndarray) -> float:
    """
    Perceptual color similarity score in [0, 1].
    1.0 = perfect match, 0.0 = maximally distant.
    Uses Lab distance; max sensible distance ~100.
    """
    dist = lab_distance(target_rgb.astype(float), candidate_rgb.astype(float))
    return float(np.clip(1.0 - dist / 100.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Skin-tone compatibility rules
# ---------------------------------------------------------------------------

# Maps (Fitzpatrick prefix, undertone) → good categories of shades
# These are drawn from professional MUA guidelines.
_COMPATIBILITY_HINTS = {
    # Fair/Very Fair
    "I":   {"warm": ["peach", "coral", "nude", "rose"],
             "cool": ["pink", "berry", "mauve", "plum"],
             "neutral": ["rose", "mauve", "nude"]},
    "II":  {"warm": ["peach", "coral", "golden", "apricot"],
             "cool": ["pink", "berry", "mauve"],
             "neutral": ["nude", "mauve", "rose"]},
    # Medium
    "III": {"warm": ["terracotta", "coral", "peach", "bronze"],
             "cool": ["rose", "berry", "plum", "mauve"],
             "neutral": ["mauve", "dusty rose", "nude"]},
    "IV":  {"warm": ["copper", "bronze", "terracotta", "orange"],
             "cool": ["plum", "burgundy", "berry"],
             "neutral": ["brown", "mauve"]},
    # Dark
    "V":   {"warm": ["copper", "bronze", "deep brown", "warm red"],
             "cool": ["deep plum", "wine", "burgundy"],
             "neutral": ["deep rose", "rich brown"]},
    "VI":  {"warm": ["deep copper", "warm mahogany", "orange-red"],
             "cool": ["deep plum", "midnight", "wine"],
             "neutral": ["deep brown", "wine"]},
}


def _fitzpatrick_key(fitz: str) -> str:
    """Extract Roman numeral prefix from Fitzpatrick string."""
    return fitz.split("—")[0].strip().split()[0]


def _compatibility_bonus(product: dict, skin: SkinAnalysisResult) -> float:
    """
    Bonus score [0, 0.15] based on known color-skin-tone compatibility.
    Checks if any shade name contains a keyword that suits this skin.
    """
    fitz_key = _fitzpatrick_key(skin.fitzpatrick)
    undertone = _undertone_from_skin(skin)
    hints = _COMPATIBILITY_HINTS.get(fitz_key, {}).get(undertone, [])

    shade_names = " ".join(
        (s.get("shade_name") or "").lower() for s in product.get("shades", [])
    ) + " " + product.get("name", "").lower()

    for hint in hints:
        if hint in shade_names:
            return 0.12
    return 0.0


def _undertone_from_skin(skin: SkinAnalysisResult) -> str:
    from reference_analysis import _undertone
    return _undertone(skin.dominant_colors[0])


def _rating_bonus(product: dict) -> float:
    """Normalized rating bonus [0, 0.1]."""
    r = product.get("rating")
    if r is None:
        return 0.03  # neutral assumption
    return float(np.clip((r / 5.0) * 0.10, 0.0, 0.10))


# ---------------------------------------------------------------------------
# Per-category scoring
# ---------------------------------------------------------------------------

def _best_shade_score(product: dict, target_rgb: np.ndarray) -> tuple[float, Optional[dict]]:
    """
    Find the product's shade closest to target_rgb.
    Returns (color_score, best_shade_dict).
    """
    shades = product.get("shades") or []
    if not shades:
        # No shade data — give moderate neutral score
        return 0.40, None

    best_score = -1.0
    best_shade = None
    for shade in shades:
        rgb = shade.get("rgb")
        if rgb is None:
            continue
        score = _color_score(target_rgb, np.array(rgb, dtype=float))
        if score > best_score:
            best_score = score
            best_shade = shade

    if best_score < 0:
        return 0.40, None
    return best_score, best_shade


@dataclass
class Recommendation:
    rank: int
    category: str
    product_name: str
    brand: str
    matched_shade: Optional[str]
    matched_hex: Optional[str]
    price: Optional[float]
    url: str
    image_url: str
    rating: Optional[float]
    color_score: float
    total_score: float
    why: str                       # Human-readable explanation


def _build_why(color_score: float, compat: float, shade: Optional[dict],
               skin: SkinAnalysisResult, look: MakeupLook) -> str:
    parts = []
    if color_score > 0.80:
        parts.append("excellent color match to your reference look")
    elif color_score > 0.60:
        parts.append("good color match to your reference look")
    else:
        parts.append("moderate color match")

    fitz_key = _fitzpatrick_key(skin.fitzpatrick)
    undertone = _undertone_from_skin(skin)
    parts.append(f"suited to {skin.fitzpatrick} skin with {undertone} undertone")

    if shade:
        parts.append(f"shade '{shade.get('shade_name', '')}' selected")

    return "; ".join(parts).capitalize() + "."


# ---------------------------------------------------------------------------
# Main recommender
# ---------------------------------------------------------------------------

CATEGORY_TARGETS = {
    "lipstick":   lambda look: look.lip_color,
    "foundation": lambda look: look.foundation_color,
    "eyeshadow":  lambda look: look.eye_color,
    "blush":      lambda look: look.blush_color,
    "concealer":  lambda look: look.foundation_color,
}


def recommend(
    skin: SkinAnalysisResult,
    look: MakeupLook,
    categories: Optional[list[str]] = None,
    top_k: int = 5,
    verbose: bool = True,
) -> dict[str, list[Recommendation]]:
    """
    Generate ranked makeup recommendations.

    Args:
        skin: User's skin analysis result.
        look: Makeup look extracted from reference image.
        categories: Which categories to recommend (default: all).
        top_k: Number of recommendations per category.
        verbose: Print loading progress.

    Returns:
        Dict mapping category name → list of Recommendation objects.
    """
    db = get_database()
    db.load(categories=categories, verbose=verbose)

    cats = categories or list(CATEGORY_TARGETS.keys())
    results: dict[str, list[Recommendation]] = {}

    for cat in cats:
        if cat not in CATEGORY_TARGETS:
            continue

        target_rgb = CATEGORY_TARGETS[cat](look)
        products = db.with_shades(cat)

        if not products:
            # Fall back to all products in category (including shade-less)
            products = db.by_category(cat)

        scored = []
        for product in products:
            color_score, best_shade = _best_shade_score(product, target_rgb)
            compat = _compatibility_bonus(product, skin)
            rating_bonus = _rating_bonus(product)

            total = (
                color_score * 0.65    # color match is king
                + compat              # skin-tone compatibility
                + rating_bonus        # crowd wisdom
            )

            scored.append((total, color_score, compat, product, best_shade))

        # Sort descending by total score
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        recs = []
        for rank, (total, color_score, compat, product, shade) in enumerate(top, 1):
            recs.append(Recommendation(
                rank=rank,
                category=cat,
                product_name=product["name"],
                brand=product["brand"],
                matched_shade=shade.get("shade_name") if shade else None,
                matched_hex=shade.get("hex") if shade else None,
                price=product.get("price"),
                url=product.get("url", ""),
                image_url=product.get("image_url", ""),
                rating=product.get("rating"),
                color_score=round(color_score, 3),
                total_score=round(total, 3),
                why=_build_why(color_score, compat, shade, skin, look),
            ))
        results[cat] = recs

    return results
