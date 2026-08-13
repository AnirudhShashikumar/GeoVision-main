"""FastAPI service for the repository's Pix2Pix SAR-to-optical generator."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pix2pix import Pix2Pix


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path(
    os.environ.get("SAR_COLORIZATION_CHECKPOINT", ROOT_DIR / "pix2pix_gen_180.pth")
)

MODEL_CONFIG = {
    "c_in": 3,
    "c_out": 3,
    "use_upsampling": False,
    "mode": "nearest",
}

app = FastAPI(title="SAR-to-Optical Colorization API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def load_generator() -> Pix2Pix:
    """Load the same generator and checkpoint used by inference.py."""
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(f"Generator checkpoint not found: {CHECKPOINT_PATH}")

    model = Pix2Pix(
        c_in=MODEL_CONFIG["c_in"],
        c_out=MODEL_CONFIG["c_out"],
        is_train=False,
        use_upsampling=MODEL_CONFIG["use_upsampling"],
        mode=MODEL_CONFIG["mode"],
    ).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.gen.load_state_dict(state_dict, strict=True)
    return model.eval()


generator = load_generator()


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode an uploaded image and normalize its color channels."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Images must be 20 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Upload a valid PNG, JPEG, or TIFF image.") from error
    return image.convert("RGB")


def to_model_input(image: Image.Image) -> torch.Tensor:
    """Resize and normalize an RGB image to the generator's expected tensor."""
    image = image.resize((256, 256), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(DEVICE)


def generate(image: Image.Image) -> Image.Image:
    with torch.inference_mode():
        prediction = generator(to_model_input(image))[0]
    output = ((prediction + 1) / 2).clamp(0, 1)
    output_array = (output.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(output_array, mode="RGB")


def as_png_response(
    image: Image.Image, headers: Optional[Dict[str, str]] = None
) -> StreamingResponse:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="sar-to-optical.png"', **(headers or {})},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ready",
        "device": str(DEVICE),
        "checkpoint": CHECKPOINT_PATH.name,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> StreamingResponse:
    """Translate one SAR image to an RGB optical-style image."""
    try:
        image = decode_image(await file.read())
        return as_png_response(generate(image))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error


@app.post("/evaluate")
async def evaluate(
    sar_file: UploadFile = File(...), optical_file: UploadFile = File(...)
) -> StreamingResponse:
    """Generate an image and return PSNR/SSIM against a paired optical reference."""
    try:
        predicted = generate(decode_image(await sar_file.read()))
        reference = decode_image(await optical_file.read()).resize(
            (256, 256), Image.Resampling.BICUBIC
        )
        predicted_array = np.asarray(predicted)
        reference_array = np.asarray(reference)
        headers = {
            "X-PSNR": f"{peak_signal_noise_ratio(reference_array, predicted_array, data_range=255):.2f}",
            "X-SSIM": f"{structural_similarity(reference_array, predicted_array, channel_axis=2, data_range=255):.4f}",
        }
        return as_png_response(predicted, headers)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {error}") from error
