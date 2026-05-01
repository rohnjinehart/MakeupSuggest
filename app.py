"""
MakeupSuggest — Gradio web application.

Upload a reference photo (celebrity, magazine look) and a selfie.
The app will:
  1. Analyze your skin tone and Fitzpatrick type.
  2. Extract the makeup look from the reference image.
  3. Apply BeautyGAN makeup transfer to simulate the look on your face.
  4. Query Open Beauty Facts + Makeup API for real products.
  5. Rank products by color proximity and skin-tone suitability.
  6. Display results with color swatches and buy links.

Run:
    python app.py
"""

import textwrap

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from skin_analysis import analyze_skin, SkinAnalysisResult
from reference_analysis import analyze_reference, MakeupLook
from recommender import recommend, Recommendation
from makeup_transfer import apply_makeup


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _swatch(rgb, size: int = 40) -> Image.Image:
    r, g, b = (int(np.clip(v, 0, 255)) for v in rgb)
    img = Image.new("RGB", (size, size), (r, g, b))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=(0, 0, 0), width=2)
    return img


def _build_color_strip(colors: list, labels: list,
                        w: int = 420, h: int = 60) -> Image.Image:
    strip = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(strip)
    n = len(colors)
    sw = w // max(n, 1)
    for i, (color, label) in enumerate(zip(colors, labels)):
        r, g, b = (int(np.clip(v, 0, 255)) for v in color)
        x0 = i * sw
        draw.rectangle([x0, 0, x0 + sw - 2, h - 16], fill=(r, g, b), outline=(0, 0, 0))
        draw.text((x0 + 4, h - 14), label[:14], fill=(0, 0, 0))
    return strip


def _result_card(recs: list[Recommendation], category: str) -> Image.Image:
    card_w, row_h = 700, 90
    card_h = row_h * len(recs) + 50
    card = Image.new("RGB", (card_w, card_h), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    draw.rectangle([0, 0, card_w, 40], fill=(30, 30, 30))
    draw.text((10, 10), f"  {category.upper()} RECOMMENDATIONS", fill=(255, 255, 255))

    for i, rec in enumerate(recs):
        y = 50 + i * row_h

        if rec.matched_hex:
            from product_database import hex_to_rgb
            rgb = hex_to_rgb(rec.matched_hex)
            if rgb is not None:
                card.paste(_swatch(rgb, size=row_h - 10), (8, y + 5))

        x_text = 60
        name = textwrap.shorten(rec.product_name, width=40)
        brand = rec.brand or "Unknown brand"
        price_str = f"${rec.price:.2f}" if rec.price else "Price N/A"
        rating_str = f"★ {rec.rating:.1f}" if rec.rating else ""
        shade_str = f"Shade: {rec.matched_shade}" if rec.matched_shade else ""
        score_str = f"Match: {rec.color_score * 100:.0f}%"

        draw.text((x_text, y + 4),  f"#{rec.rank}  {brand} — {name}", fill=(10, 10, 10))
        draw.text((x_text, y + 22), shade_str,                          fill=(80, 80, 80))
        draw.text((x_text, y + 38), rec.why[:80],                       fill=(100, 100, 100))
        draw.text((x_text, y + 54), f"{price_str}   {rating_str}   {score_str}", fill=(50, 50, 180))
        draw.line([(0, y + row_h - 1), (card_w, y + row_h - 1)], fill=(220, 220, 220))

    return card


def _combine_cards(cards: list[Image.Image]) -> Image.Image | None:
    if not cards:
        return None
    total_h = sum(img.height for img in cards) + 10 * len(cards)
    combined = Image.new("RGB", (cards[0].width, total_h), (240, 240, 240))
    y = 0
    for img in cards:
        combined.paste(img, (0, y))
        y += img.height + 10
    return combined


def _recs_to_markdown(recommendations: dict) -> str:
    lines = []
    for cat, recs in recommendations.items():
        lines.append(f"\n## {cat.title()}\n")
        for rec in recs:
            price = f"${rec.price:.2f}" if rec.price else "N/A"
            shade = f" — *{rec.matched_shade}*" if rec.matched_shade else ""
            hex_badge = f" `{rec.matched_hex}`" if rec.matched_hex else ""
            rating = f"★ {rec.rating:.1f}" if rec.rating else ""
            lines.append(
                f"**{rec.rank}. {rec.brand} — {rec.product_name}**{shade}{hex_badge}  \n"
                f"{rec.why}  \n"
                f"Price: {price}  {rating}  Color match: {rec.color_score*100:.0f}%  \n"
                f"[View product]({rec.url})  \n"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")


def gradio_pipeline(selfie, reference, categories, top_k):
    if selfie is None or reference is None:
        empty = (None,) * 5
        return "Please upload both images.", "", None, *empty

    selfie_pil = _to_pil(selfie)
    ref_pil    = _to_pil(reference)

    try:
        # Step 1 — skin analysis
        skin = analyze_skin(selfie_pil)
        skin_info = (
            f"Fitzpatrick Type: {skin.fitzpatrick}\n"
            f"Primary Skin Tone: RGB{tuple(skin.primary_rgb().tolist())}\n"
            f"Lab Values (L, a, b): {tuple(round(v, 1) for v in skin.primary_lab())}\n"
            f"Dominant Colors: {len(skin.dominant_colors)} extracted"
        )

        # Step 2 — reference look analysis
        look = analyze_reference(ref_pil)
        summary = look.summary()
        look_info = (
            f"Undertone detected: {look.undertone}\n"
            f"Lip color:   {summary['lip_color']['hex']}  "
            f"RGB{(summary['lip_color']['r'], summary['lip_color']['g'], summary['lip_color']['b'])}\n"
            f"Eye color:   {summary['eye_color']['hex']}\n"
            f"Blush:       {summary['blush_color']['hex']}\n"
            f"Foundation:  {summary['foundation_color']['hex']}"
        )

        color_strip = _build_color_strip(
            [look.lip_color, look.eye_color, look.blush_color, look.foundation_color],
            ["Lips", "Eyes", "Blush", "Foundation"],
        )

        # Step 3 — BeautyGAN makeup transfer
        transferred_img, method_used = apply_makeup(selfie_pil, ref_pil)
        transfer_label = f"Simulated look ({method_used})"

        # Step 4 — product recommendations
        cats = [c.lower() for c in categories] if categories else None
        recs = recommend(skin, look, categories=cats, top_k=int(top_k), verbose=True)

        # Step 5 — build outputs
        cards = [_result_card(r, cat) for cat, r in recs.items() if r]
        final_card = _combine_cards(cards)
        md_output = _recs_to_markdown(recs)

        return (
            skin_info,
            look_info,
            color_strip,
            transferred_img,
            transfer_label,
            final_card,
            md_output,
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return (f"Error: {e}", "", None, None, "", None, tb)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CATEGORY_CHOICES = ["Lipstick", "Foundation", "Eyeshadow", "Blush", "Concealer"]

with gr.Blocks(title="MakeupSuggest") as demo:
    gr.Markdown(
        """
        # MakeupSuggest
        ### AI-powered makeup simulation & product recommendations

        **How it works:**
        1. Upload a **selfie** — face forward, minimal makeup, good lighting.
        2. Upload a **reference image** — a look you want to achieve.
        3. The AI simulates the look on your face using **BeautyGAN**.
        4. Real products from **Open Beauty Facts** & **Makeup API** are ranked by color match and skin-tone suitability.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            selfie_input = gr.Image(label="Your Selfie", type="numpy", height=300)
        with gr.Column(scale=1):
            reference_input = gr.Image(label="Reference Look", type="numpy", height=300)

    with gr.Row():
        categories_input = gr.CheckboxGroup(
            choices=CATEGORY_CHOICES,
            value=CATEGORY_CHOICES,
            label="Makeup Categories",
        )
        top_k_input = gr.Slider(
            minimum=1, maximum=10, value=5, step=1,
            label="Recommendations per Category",
        )

    run_btn = gr.Button("Analyze & Find My Products", variant="primary", size="lg")

    # --- Analysis outputs ---
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Your Skin Analysis")
            skin_output = gr.Textbox(label="Skin Tone Profile", lines=5, interactive=False)
        with gr.Column():
            gr.Markdown("### Reference Look Analysis")
            look_output = gr.Textbox(label="Detected Makeup Colors", lines=5, interactive=False)

    gr.Markdown("### Reference Color Palette")
    color_strip_output = gr.Image(label="Extracted Colors", height=80)

    # --- BeautyGAN transfer ---
    gr.Markdown("### Simulated Look (BeautyGAN)")
    with gr.Row():
        with gr.Column(scale=1):
            transfer_output = gr.Image(label="Your face with reference makeup applied", height=400)
        with gr.Column(scale=1):
            transfer_method = gr.Textbox(label="Method used", lines=1, interactive=False)
            gr.Markdown(
                "_The simulation uses BeautyGAN to transfer the makeup style from "
                "the reference onto your selfie. Results are an approximation — "
                "actual product colors may vary._"
            )

    # --- Product recommendations ---
    gr.Markdown("### Product Recommendations")
    cards_output = gr.Image(label="Recommendation Cards", height=600)

    gr.Markdown("### Detailed Recommendations with Links")
    markdown_output = gr.Markdown()

    run_btn.click(
        fn=gradio_pipeline,
        inputs=[selfie_input, reference_input, categories_input, top_k_input],
        outputs=[
            skin_output,
            look_output,
            color_strip_output,
            transfer_output,
            transfer_method,
            cards_output,
            markdown_output,
        ],
    )

    gr.Markdown(
        """
        ---
        **Data sources:**
        [Open Beauty Facts](https://world.openbeautyfacts.org) (ODbL) ·
        [Makeup API](https://makeup-api.herokuapp.com) ·
        [BeautyGAN](https://dl.acm.org/doi/10.1145/3240508.3240618) (He et al., 2018)

        **Privacy:** All image processing runs locally. No photos are sent to external servers.
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, theme=gr.themes.Soft())
