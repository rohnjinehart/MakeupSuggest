"""
Product database layer.

Sources:
  1. Open Beauty Facts API  (https://world.openbeautyfacts.org)
     — open-source cosmetics product database (~1M products)
  2. Makeup API              (https://makeup-api.herokuapp.com)
     — curated makeup product catalog with hex color swatches
  3. Local CSV cache         — avoids repeated API calls

Products are normalized into a common schema and cached locally.
"""

import json
import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

MAKEUP_API_BASE = "https://makeup-api.herokuapp.com/api/v1/products.json"
OBF_API_BASE = "https://world.openbeautyfacts.org/cgi/search.pl"

# Makeup categories we care about
CATEGORY_MAP = {
    "lipstick": ["lipstick", "lip gloss", "lip liner", "liquid lipstick", "lip stain"],
    "foundation": ["foundation", "bb cream", "cc cream", "tinted moisturizer"],
    "eyeshadow": ["eyeshadow", "eye shadow", "eye palette"],
    "blush": ["blush", "blusher", "bronzer", "highlighter", "contour"],
    "concealer": ["concealer", "color corrector"],
}


# ---------------------------------------------------------------------------
# Hex → RGB conversion
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_str: str) -> Optional[np.ndarray]:
    """Convert #RRGGBB or RRGGBB to numpy uint8 array."""
    if not hex_str:
        return None
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        return None
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return np.array([r, g, b], dtype=np.uint8)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Makeup API (makeup-api.herokuapp.com) — has hex color swatches
# ---------------------------------------------------------------------------

def _fetch_makeup_api(product_type: str, page_size: int = 200) -> list[dict]:
    cache_file = CACHE_DIR / f"makeup_api_{product_type}.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400 * 7:  # 7-day cache
            with open(cache_file) as f:
                return json.load(f)

    print(f"  Fetching Makeup API: {product_type}...")
    try:
        resp = requests.get(
            MAKEUP_API_BASE,
            params={"product_type": product_type},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Warning: Makeup API fetch failed for {product_type}: {e}")
        return []

    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data


def _normalize_makeup_api(raw: list[dict], category: str) -> list[dict]:
    products = []
    for item in raw:
        name = (item.get("name") or "").strip()
        brand = (item.get("brand") or "").strip()
        price = item.get("price")
        url = item.get("product_link") or item.get("website_link") or ""
        image_url = item.get("image_link") or ""
        description = item.get("description") or ""
        rating = item.get("rating")

        colors_raw = item.get("product_colors") or []
        shades = []
        for c in colors_raw:
            hex_val = c.get("hex_value", "")
            rgb = hex_to_rgb(hex_val)
            if rgb is not None:
                shades.append({
                    "shade_name": c.get("colour_name", ""),
                    "hex": hex_val,
                    "rgb": rgb.tolist(),
                })

        if not name or not brand:
            continue

        products.append({
            "source": "makeup_api",
            "category": category,
            "name": name,
            "brand": brand,
            "price": float(price) if price else None,
            "url": url,
            "image_url": image_url,
            "description": description,
            "rating": float(rating) if rating else None,
            "shades": shades,
        })
    return products


# ---------------------------------------------------------------------------
# Open Beauty Facts API — broad cosmetics database
# ---------------------------------------------------------------------------

def _fetch_obf(search_term: str, page: int = 1) -> list[dict]:
    cache_file = CACHE_DIR / f"obf_{search_term.replace(' ', '_')}_p{page}.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400 * 3:
            with open(cache_file) as f:
                return json.load(f)

    print(f"  Fetching Open Beauty Facts: '{search_term}' page {page}...")
    try:
        resp = requests.get(
            OBF_API_BASE,
            params={
                "search_terms": search_term,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 50,
                "page": page,
            },
            timeout=20,
        )
        resp.raise_for_status()
        products = resp.json().get("products", [])
    except Exception as e:
        print(f"  Warning: Open Beauty Facts fetch failed: {e}")
        return []

    with open(cache_file, "w") as f:
        json.dump(products, f)
    return products


def _normalize_obf(raw: list[dict], category: str) -> list[dict]:
    products = []
    for item in raw:
        name = (item.get("product_name") or "").strip()
        brand = (item.get("brands") or "").strip()
        if not name:
            continue
        url = f"https://world.openbeautyfacts.org/product/{item.get('code', '')}"
        image_url = item.get("image_front_url") or item.get("image_url") or ""
        ingredients = item.get("ingredients_text") or ""

        products.append({
            "source": "open_beauty_facts",
            "category": category,
            "name": name,
            "brand": brand,
            "price": None,
            "url": url,
            "image_url": image_url,
            "description": ingredients[:300],
            "rating": None,
            "shades": [],  # OBF doesn't store shade swatches
        })
    return products


# ---------------------------------------------------------------------------
# Combined database builder
# ---------------------------------------------------------------------------

class ProductDatabase:
    """Unified, lazily-loaded product database with local caching."""

    _MAKEUP_API_TYPE_MAP = {
        "lipstick": "lipstick",
        "foundation": "foundation",
        "eyeshadow": "eyeshadow",
        "blush": "blush",
        "concealer": "concealer",
    }

    def __init__(self):
        self._products: list[dict] = []
        self._loaded = False

    def load(self, categories: Optional[list[str]] = None, verbose: bool = True):
        """Fetch and merge products from all sources."""
        if self._loaded:
            return

        cats = categories or list(CATEGORY_MAP.keys())
        all_products: list[dict] = []

        for cat in cats:
            if verbose:
                print(f"Loading category: {cat}")

            # Source 1: Makeup API (has color swatches — highest value)
            api_type = self._MAKEUP_API_TYPE_MAP.get(cat, cat)
            raw = _fetch_makeup_api(api_type)
            all_products.extend(_normalize_makeup_api(raw, cat))

            # Source 2: Open Beauty Facts (broader catalog)
            for term in CATEGORY_MAP.get(cat, [cat])[:2]:  # first 2 aliases
                raw_obf = _fetch_obf(term, page=1)
                all_products.extend(_normalize_obf(raw_obf, cat))

        self._products = all_products
        self._loaded = True
        if verbose:
            print(f"Loaded {len(self._products)} products total.")

    def by_category(self, category: str) -> list[dict]:
        return [p for p in self._products if p["category"] == category]

    def with_shades(self, category: str) -> list[dict]:
        """Only products that have at least one color swatch."""
        return [p for p in self.by_category(category) if p["shades"]]

    def all_products(self) -> list[dict]:
        return self._products

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self._products:
            if p["shades"]:
                for s in p["shades"]:
                    rows.append({**p, **s, "shades": None})
            else:
                rows.append({**p, "shade_name": "", "hex": "", "rgb": None})
        return pd.DataFrame(rows)


# Singleton
_db = ProductDatabase()


def get_database() -> ProductDatabase:
    return _db
