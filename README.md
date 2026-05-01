# MakeupSuggest

An ML-powered Python application that analyzes your skin tone and a reference makeup look, then recommends real products from open-source databases that will help you achieve the look.

## Features

- **Skin tone analysis** — detects your Fitzpatrick skin type (I–VI) and dominant undertone (warm/cool/neutral) using PyTorch + OpenCV face detection
- **Reference look extraction** — parses lip, eye, blush, and foundation colors from any photo
- **Product matching** — scores products from two open databases by perceptual color distance (CIELab) and skin-tone compatibility
- **Web UI** — simple Gradio interface; runs entirely locally

## Data Sources

| Source | What it provides | License |
|---|---|---|
| [Open Beauty Facts](https://world.openbeautyfacts.org) | ~1M cosmetic products, ingredients | ODbL |
| [Makeup API](https://makeup-api.herokuapp.com) | Curated products with hex color swatches | Free |

## Architecture

```
app.py                 ← Gradio UI + pipeline orchestration
├── skin_analysis.py   ← Face detection, skin-pixel masking, Fitzpatrick typing
├── reference_analysis.py ← Zone-based makeup color extraction from reference image
├── recommender.py     ← PyTorch color embedder + scoring engine
├── product_database.py   ← API clients for Open Beauty Facts + Makeup API (with caching)
└── color_utils.py     ← Shared color math helpers
```

### ML Components

- **MobileNetV3-Small** (torchvision): backbone for the skin segmentation head
- **K-Means clustering** (scikit-learn): extracts dominant skin/makeup colors
- **CIELab color space** (colormath): perceptual color distance scoring
- **Haar Cascade** (OpenCV): face region detection
- **ColorEmbedder** (custom PyTorch MLP): 3→16 dim color embedding for batch similarity

## Setup

```bash
# 1. Create a virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

## Usage

1. **Upload your selfie** — face forward, no heavy makeup, good lighting
2. **Upload a reference image** — any photo of a makeup look you want to achieve
3. **Select categories** — Lipstick, Foundation, Eyeshadow, Blush, Concealer
4. **Choose how many recommendations** per category (1–10)
5. Click **Find My Products**

Results include:
- Your Fitzpatrick skin type and undertone
- Color palette extracted from the reference
- Ranked product recommendations with color swatches, pricing, and direct links

## How the Scoring Works

Each product is scored out of ~1.0:

| Component | Weight | Details |
|---|---|---|
| Color match | 65% | CIELab distance between target shade and product shade |
| Skin-tone compatibility | up to 15% | Rule-based bonus for known flattering shades per Fitzpatrick type |
| Product rating | up to 10% | Normalized from 0–5 star rating |

## Requirements

- Python 3.10+
- Internet connection (first run fetches product data, then caches locally for 7 days)
- No GPU required (CPU inference is fast enough for this workload)

## Privacy

All image processing runs locally. No photos are sent to external servers. Product data is fetched from public APIs and cached in `.cache/`.

## License

MIT
