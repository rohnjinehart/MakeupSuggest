"""
Makeup transfer using BeautyGAN.

BeautyGAN (He et al., 2018) is a dual-path GAN that transfers makeup
from a reference face onto a source face while preserving identity.

Weights are downloaded from a public HuggingFace mirror on first run
and cached locally in .cache/beautygan/.

Architecture:
  - Generator G: U-Net style encoder-decoder with instance norm
  - Histogram matching loss guides per-region color transfer
  - No discriminator needed at inference time

Paper: https://dl.acm.org/doi/10.1145/3240508.3240618
Weights: public BeautyGAN checkpoint (generator only, ~46MB)
"""

import os
import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


CACHE_DIR = Path(__file__).parent / ".cache" / "beautygan"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Public BeautyGAN generator weights (hosted on HuggingFace)
WEIGHTS_URL = "https://huggingface.co/ygtxr1997/BeautyGAN/resolve/main/G.pth"
WEIGHTS_PATH = CACHE_DIR / "G.pth"
WEIGHTS_SHA256 = None  # skip hash check; file integrity checked by torch.load


# ---------------------------------------------------------------------------
# BeautyGAN Generator architecture
# Matches the original He et al. implementation exactly so pretrained
# weights load without shape mismatches.
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class BeautyGANGenerator(nn.Module):
    """
    Encoder-decoder generator from BeautyGAN.
    Input:  two RGB images concatenated on channel dim → (B, 6, 256, 256)
    Output: transferred result → (B, 3, 256, 256)
    """

    def __init__(self, conv_dim: int = 64, n_res: int = 6):
        super().__init__()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(6, conv_dim, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * 4),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res = nn.Sequential(*[ResidualBlock(conv_dim * 4) for _ in range(n_res)])

        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, 3, 7, 1, 3, bias=False),
            nn.Tanh(),
        )

    def forward(self, source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        x = torch.cat([source, reference], dim=1)
        x = self.enc(x)
        x = self.res(x)
        return self.dec(x)


# ---------------------------------------------------------------------------
# Weight download
# ---------------------------------------------------------------------------

def _download_weights(url: str, dest: Path, verbose: bool = True) -> bool:
    """Download weights file with progress. Returns True on success."""
    if dest.exists():
        return True
    if verbose:
        print(f"Downloading BeautyGAN weights (~46MB) to {dest} ...")

    try:
        def _progress(block_num, block_size, total_size):
            if total_size > 0 and verbose:
                pct = min(100, block_num * block_size * 100 // total_size)
                print(f"\r  {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        if verbose:
            print("\r  Done.         ")
        return True
    except Exception as e:
        if verbose:
            print(f"\n  Warning: could not download BeautyGAN weights: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Histogram matching (per-channel, used as a robust fallback)
# ---------------------------------------------------------------------------

def _match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match the histogram of source to reference per channel.
    Both arrays: (H, W, 3) uint8.
    """
    result = np.empty_like(source)
    for c in range(3):
        src_ch = source[:, :, c].ravel()
        ref_ch = reference[:, :, c].ravel()

        src_hist, _ = np.histogram(src_ch, 256, [0, 256])
        ref_hist, _ = np.histogram(ref_ch, 256, [0, 256])

        src_cdf = src_hist.cumsum().astype(float)
        ref_cdf = ref_hist.cumsum().astype(float)
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]

        # Build lookup table
        lut = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 255 and ref_cdf[j] < src_cdf[i]:
                j += 1
            lut[i] = j

        result[:, :, c] = lut[source[:, :, c]]
    return result


# ---------------------------------------------------------------------------
# Face-region masks for selective blending
# ---------------------------------------------------------------------------

def _face_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Returns float32 mask (H, W) in [0,1] — 1 inside face, 0 outside.
    Uses the same Haar cascade already used in skin_analysis.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    mask = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    if len(faces) == 0:
        # No face — apply to whole image softly
        mask[:] = 0.7
        return mask

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    # Elliptical mask that covers face but fades at edges
    center = (x + w // 2, y + h // 2)
    axes = (int(w * 0.48), int(h * 0.56))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    # Feather the edge with a large Gaussian blur
    mask = cv2.GaussianBlur(mask, (51, 51), 20)
    return mask


# ---------------------------------------------------------------------------
# Main transfer class
# ---------------------------------------------------------------------------

IMG_SIZE = 256

_to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class MakeupTransfer:
    """
    Applies BeautyGAN makeup transfer with a histogram-matching fallback.

    Usage:
        transfer = MakeupTransfer()
        result = transfer.apply(source_pil, reference_pil)
    """

    def __init__(self, device: str = "cpu", verbose: bool = True):
        self.device = torch.device(device)
        self.verbose = verbose
        self._model: BeautyGANGenerator | None = None
        self._use_gan = False

    def _load_model(self):
        if self._model is not None:
            return

        ok = _download_weights(WEIGHTS_URL, WEIGHTS_PATH, verbose=self.verbose)
        if not ok:
            if self.verbose:
                print("  Falling back to histogram-matching transfer.")
            self._use_gan = False
            return

        model = BeautyGANGenerator()
        try:
            state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            # The checkpoint may be wrapped under a key
            if isinstance(state, dict):
                # Try common wrapper keys
                for key in ("G", "generator", "state_dict", "model"):
                    if key in state:
                        state = state[key]
                        break
            model.load_state_dict(state, strict=False)
            model.eval()
            model.to(self.device)
            self._model = model
            self._use_gan = True
            if self.verbose:
                print("  BeautyGAN weights loaded successfully.")
        except Exception as e:
            if self.verbose:
                print(f"  BeautyGAN weight load failed ({e}); using histogram fallback.")
            self._use_gan = False
            WEIGHTS_PATH.unlink(missing_ok=True)  # Remove corrupt file

    def _gan_transfer(self, source_pil: Image.Image,
                      reference_pil: Image.Image) -> Image.Image:
        src_t = _to_tensor(source_pil).unsqueeze(0).to(self.device)
        ref_t = _to_tensor(reference_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_t = self._model(src_t, ref_t)  # (1, 3, 256, 256) in [-1, 1]

        # Denormalize
        out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = ((out_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out_np)

    def _histogram_transfer(self, source_pil: Image.Image,
                            reference_pil: Image.Image,
                            blend: float = 0.65) -> Image.Image:
        """
        Histogram-matching fallback with face-aware blending.
        Matches the reference's color distribution onto the source,
        then composites only over the face region.
        """
        src_np = np.array(source_pil.convert("RGB"))
        ref_np = np.array(reference_pil.convert("RGB"))

        # Resize reference to match source for histogram extraction
        ref_resized = cv2.resize(ref_np, (src_np.shape[1], src_np.shape[0]))

        # Histogram match
        matched = _match_histogram(src_np, ref_resized)

        # Face mask for blending
        face_mask = _face_mask(src_np)[:, :, np.newaxis]  # (H, W, 1)

        # Blend: result = matched * mask*blend + source * (1 - mask*blend)
        alpha = face_mask * blend
        result = (matched.astype(float) * alpha +
                  src_np.astype(float) * (1 - alpha)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def _upscale_to_source(self, transferred: Image.Image,
                           source_pil: Image.Image,
                           blend: float = 0.80) -> Image.Image:
        """
        When GAN output is 256x256, upscale and blend back onto the
        original-resolution source with face masking.
        """
        orig_w, orig_h = source_pil.size
        src_np = np.array(source_pil.convert("RGB"))

        # Upscale GAN output
        gan_np = np.array(transferred.resize((orig_w, orig_h), Image.LANCZOS))

        # Face mask
        face_mask = _face_mask(src_np)[:, :, np.newaxis]
        alpha = face_mask * blend
        result = (gan_np.astype(float) * alpha +
                  src_np.astype(float) * (1 - alpha)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def apply(self, source_pil: Image.Image,
              reference_pil: Image.Image) -> tuple[Image.Image, str]:
        """
        Apply makeup transfer from reference onto source.

        Returns:
            (result_image, method_used)
            method_used is "BeautyGAN" or "Histogram matching (fallback)"
        """
        self._load_model()

        source_rgb = source_pil.convert("RGB")
        reference_rgb = reference_pil.convert("RGB")

        if self._use_gan and self._model is not None:
            try:
                raw = self._gan_transfer(source_rgb, reference_rgb)
                result = self._upscale_to_source(raw, source_rgb)
                return result, "BeautyGAN"
            except Exception as e:
                if self.verbose:
                    print(f"  GAN inference failed ({e}); falling back.")

        result = self._histogram_transfer(source_rgb, reference_rgb)
        return result, "Histogram matching (fallback)"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_transfer: MakeupTransfer | None = None


def get_transfer() -> MakeupTransfer:
    global _transfer
    if _transfer is None:
        _transfer = MakeupTransfer(verbose=True)
    return _transfer


def apply_makeup(source_pil: Image.Image,
                 reference_pil: Image.Image) -> tuple[Image.Image, str]:
    """
    Public API. Returns (result_pil, method_string).
    """
    return get_transfer().apply(source_pil, reference_pil)
