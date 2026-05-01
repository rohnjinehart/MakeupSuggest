# MakeupSuggest

A Python app that analyzes your skin tone and a reference makeup photo, then recommends real products to match the look.

## Features

- **Skin tone analysis**: detects your Fitzpatrick skin type (I-VI) and undertone (warm/cool/neutral) using PyTorch and OpenCV
- **Reference look extraction**: pulls lip, eye, blush, and foundation colors from a photo
- **Product matching**: scores products from two open databases by color distance and skin-tone fit
- **Makeup simulation**: applies the reference look to your face using BeautyGAN
- **Web UI**: runs locally in your browser via Gradio

## Data Sources

| Source | What it provides | License |
|---|---|---|
| [Open Beauty Facts](https://world.openbeautyfacts.org) | ~1M cosmetic products, ingredients | ODbL |
| [Makeup API](https://makeup-api.herokuapp.com) | Makeup products with hex color swatches | Free |

## File Structure

```
app.py                    Gradio UI and pipeline
skin_analysis.py          Face detection, skin masking, Fitzpatrick classification
reference_analysis.py     Makeup color extraction from reference image
makeup_transfer.py        BeautyGAN makeup simulation
recommender.py            Product scoring engine
product_database.py       API clients for both databases, with local caching
color_utils.py            Color math helpers
```

## ML Components

- **MobileNetV3-Small** (torchvision): skin segmentation backbone
- **BeautyGAN** (PyTorch): makeup transfer from reference to selfie
- **K-Means** (scikit-learn): dominant color extraction
- **CIELab color space** (colormath): perceptual color distance
- **Haar Cascade** (OpenCV): face detection

## Setup

```bash
# Python 3.10+ required
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

## Usage

1. Upload a selfie (face forward, minimal makeup, decent lighting)
2. Upload a reference image (any makeup look you want)
3. Pick makeup categories and number of recommendations
4. Click **Analyze and Find My Products**

Results include your skin type, the extracted color palette, a BeautyGAN simulation of the look on your face, and ranked product recommendations with links.

## Scoring

| Component | Weight | Details |
|---|---|---|
| Color match | 65% | CIELab distance between target and product shade |
| Skin-tone fit | up to 15% | Bonus for shades known to work for your Fitzpatrick type |
| Product rating | up to 10% | Normalized from 0-5 stars |

## Requirements

- Python 3.10+
- Internet connection (product data is fetched on first run and cached for 7 days)
- No GPU required

## Privacy

Images are processed locally. Nothing is sent to external servers. Product data is cached in `.cache/`.

## License

MIT
