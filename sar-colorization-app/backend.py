"""FastAPI inference service for independent SAR-to-optical models."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from sarfusionformer import SARFusionFormer, lab_to_rgb
from src.pix2pix import Pix2Pix
from provider_settings import ProviderSettingsError, ProviderSettingsStore
from vision_analysis import ImageAnalysisService, VisionAnalysisError, VisionSettings

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("sar-colorization")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIX2PIX_CHECKPOINT = Path(
    os.environ.get("PIX2PIX_CHECKPOINT", ROOT_DIR / "pix2pix_gen_180.pth")
)
SARFUSIONFORMER_CHECKPOINT = Path(
    os.environ.get(
        "SARFUSIONFORMER_CHECKPOINT",
        ROOT_DIR / "models" / "checkpoints" / "sarfusionformer_256_decoder_best.pt",
    )
)
COLOR_CORRECTOR_CHECKPOINT = Path(
    os.environ.get(
        "COLOR_CORRECTOR_CHECKPOINT",
        ROOT_DIR / "models" / "checkpoints" / "color_corrector_256_best.pt",
    )
)

app = FastAPI(title="SAR-to-Optical Colorization API", version="2.0.0")
CORS_ORIGINS = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS", "http://127.0.0.1:8520,http://localhost:8520"
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class ColorCorrectionNet(torch.nn.Module):
    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, hidden_channels, kernel_size=1),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

    def forward(self, coarse_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        correction = 0.10 * torch.tanh(self.net(coarse_rgb))
        return torch.clamp(coarse_rgb + correction, 0.0, 1.0), correction


def load_pix2pix() -> Pix2Pix:
    if not PIX2PIX_CHECKPOINT.is_file():
        raise FileNotFoundError("Pix2Pix checkpoint not found: {}".format(PIX2PIX_CHECKPOINT))
    model = Pix2Pix(c_in=3, c_out=3, is_train=False).to(DEVICE)
    model.gen.load_state_dict(
        torch.load(PIX2PIX_CHECKPOINT, map_location=DEVICE, weights_only=True),
        strict=True,
    )
    return model.eval()


def load_sarfusionformer() -> SARFusionFormer:
    if not SARFUSIONFORMER_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "SARFusionFormer checkpoint not found: {}".format(SARFUSIONFORMER_CHECKPOINT)
        )
    model = SARFusionFormer(
        input_channels=2,
        output_channels=3,
        base_channels=48,
        transformer_depth=4,
        attention_heads=6,
        window_size=8,
        dropout=0.0,
    ).to(DEVICE)
    checkpoint = torch.load(
        SARFUSIONFORMER_CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def load_color_corrector() -> ColorCorrectionNet:
    if not COLOR_CORRECTOR_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Color-corrector checkpoint not found: {}".format(COLOR_CORRECTOR_CHECKPOINT)
        )
    model = ColorCorrectionNet().to(DEVICE)
    checkpoint = torch.load(
        COLOR_CORRECTOR_CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["color_corrector"], strict=True)
    return model.eval()


def try_load(name: str, loader):
    try:
        model = loader()
        LOGGER.info("%s loaded successfully on %s", name, DEVICE)
        return model, None
    except Exception as error:
        LOGGER.warning("%s unavailable: %s", name, error)
        return None, str(error)


PIX2PIX_MODEL, PIX2PIX_ERROR = try_load("Pix2Pix", load_pix2pix)
SARFUSIONFORMER_MODEL, SARFUSIONFORMER_ERROR = try_load(
    "SARFusionFormer", load_sarfusionformer
)
COLOR_CORRECTOR, COLOR_CORRECTOR_ERROR = try_load(
    "Color corrector", load_color_corrector
)
VISION_ANALYSIS = ImageAnalysisService()
PROVIDER_SETTINGS = ProviderSettingsStore()


def sync_vision_settings() -> None:
    """Refresh the analysis service after a key is changed in Settings."""
    credentials = PROVIDER_SETTINGS.credentials()
    VISION_ANALYSIS.settings = VisionSettings(
        provider=credentials.provider,
        api_key=credentials.api_key,
        model=credentials.model or "gpt-4.1-mini",
    )


sync_vision_settings()


class ProviderConfigurationRequest(BaseModel):
    provider: str
    api_key: str


def model_status(model: Optional[torch.nn.Module], error: Optional[str], checkpoint: Path) -> Dict[str, Any]:
    return {
        "available": model is not None,
        "checkpoint": checkpoint.name,
        "error": error,
    }


def require_model(model: Optional[torch.nn.Module], error: Optional[str], name: str):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="{} is unavailable. {}".format(name, error or "Check the checkpoint path."),
        )
    return model


def read_upload(upload: UploadFile) -> bytes:
    contents = upload.file.read()
    if not contents:
        raise ValueError("The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Files must be 20 MB or smaller.")
    return contents


def decode_rgb(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Upload a valid PNG, JPEG, or TIFF image.") from error
    return image.convert("RGB")


def decode_grayscale(image_bytes: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename or "").suffix.lower()
    try:
        if suffix == ".npy":
            array = np.load(io.BytesIO(image_bytes), allow_pickle=False)
        else:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            array = np.asarray(image)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("Upload a valid .npy, TIFF, PNG, or JPEG single-channel image.") from error

    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("VV and VH inputs must be single-channel two-dimensional arrays.")
    return array.astype(np.float32, copy=False)


def load_combined_sar_npy(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, str, Tuple[int, ...]]:
    """Load a two-channel VV/VH NumPy array in channel-first or channel-last form."""
    try:
        array = np.asarray(np.load(io.BytesIO(image_bytes), allow_pickle=False))
    except (ValueError, OSError) as error:
        raise ValueError("Upload a valid combined VV/VH NumPy (.npy) file.") from error

    detected_shape = tuple(array.shape)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ValueError("Combined SAR arrays must contain numeric, non-object values.")
    if np.iscomplexobj(array):
        raise ValueError("Combined SAR arrays must contain real-valued VV and VH channels.")
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            "Expected a 3D two-channel SAR array, got shape {}".format(detected_shape)
        )

    is_chw = array.shape[0] == 2
    is_hwc = array.shape[-1] == 2
    if is_chw and is_hwc:
        raise ValueError(
            "Ambiguous two-channel SAR layout in shape {}. Use an unambiguous CHW or HWC array."
            .format(detected_shape)
        )
    if is_chw:
        sar = array
        layout = "CHW"
    elif is_hwc:
        sar = np.moveaxis(array, -1, 0)
        layout = "HWC"
    else:
        raise ValueError(
            "Could not find exactly two SAR channels in shape {}".format(detected_shape)
        )

    sar = sar.astype(np.float32, copy=False)
    if not np.isfinite(sar).any():
        raise ValueError("Combined SAR input contains no finite values.")
    return sar[0], sar[1], layout, detected_shape


def normalize_channel(array: np.ndarray) -> np.ndarray:
    """Apply the exact per-channel percentile normalization used in training."""
    low, high = np.percentile(array, [1, 99])
    if high - low <= np.finfo(np.float32).eps:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return rgb_float_to_png(array)


def rgb_float_to_png(rgb: np.ndarray) -> Image.Image:
    """Encode an RGB float image without changing its radiometric values first."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB output must have shape [height, width, 3].")
    encoded = (np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8)
    return Image.fromarray(encoded, mode="RGB")


def contrast_stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    """Global display-only 2nd/98th percentile stretch that preserves RGB balance."""
    rgb = np.asarray(rgb, dtype=np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)

    low = float(np.percentile(rgb, 2))
    high = float(np.percentile(rgb, 98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return rgb.copy()

    stretched = (rgb - low) / (high - low)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def array_diagnostics(array: np.ndarray, prefix: str) -> Dict[str, Any]:
    array = np.asarray(array, dtype=np.float32)
    return {
        "{}_min".format(prefix): float(array.min()),
        "{}_max".format(prefix): float(array.max()),
        "{}_mean".format(prefix): float(array.mean()),
        "{}_std".format(prefix): float(array.std()),
    }


def rgb_diagnostics(rgb: np.ndarray, prefix: str) -> Dict[str, Any]:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB diagnostics require a [height, width, 3] array.")
    return {
        **array_diagnostics(rgb, prefix),
        "{}_channel_mean".format(prefix): [float(value) for value in rgb.mean(axis=(0, 1))],
        "{}_channel_std".format(prefix): [float(value) for value in rgb.std(axis=(0, 1))],
    }


def display_representation(rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], Optional[str]]:
    """Return a visual-only representation and diagnostics for one raw RGB image."""
    rgb = np.asarray(rgb, dtype=np.float32)
    finite_rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    low = float(np.percentile(finite_rgb, 2))
    high = float(np.percentile(finite_rgb, 98))
    raw_std = float(finite_rgb.std())
    warning = None
    if raw_std < 0.001:
        # Do not turn a nearly constant scientific result into apparent detail.
        display_rgb = finite_rgb.copy()
        warning = (
            "Raw RGB standard deviation is below 0.001; contrast enhancement was skipped. "
            "Review the model input and checkpoint rather than interpreting amplified noise."
        )
    else:
        display_rgb = contrast_stretch_rgb(finite_rgb)
    diagnostics = {
        **rgb_diagnostics(finite_rgb, "raw_rgb"),
        **rgb_diagnostics(display_rgb, "display_rgb"),
        "stretch_low": low,
        "stretch_high": high,
        "stretch_applied": warning is None and high > low,
    }
    return display_rgb, diagnostics, warning


def channel_preview(channel: np.ndarray) -> Image.Image:
    return Image.fromarray((channel.clip(0, 1) * 255).round().astype(np.uint8), mode="L")


def prepare_sarfusionformer_input(
    vv_array: np.ndarray, vh_array: np.ndarray
) -> Tuple[torch.Tensor, Dict[str, Image.Image]]:
    """Produce the exact [1, 2, 256, 256] float32 tensor used during training."""
    if vv_array.shape != vh_array.shape:
        raise ValueError("VV and VH inputs must have matching spatial dimensions.")
    if vv_array.ndim != 2:
        raise ValueError("VV and VH inputs must be two-dimensional SAR channels.")

    # Keep this order aligned with the training pipeline: stack, replace invalid
    # values, normalize each channel independently, then resize to model resolution.
    source_sar = np.stack([vv_array, vh_array], axis=0).astype(np.float32, copy=False)
    if not np.isfinite(source_sar).any():
        raise ValueError("SAR array contains no finite values.")
    source_sar = np.nan_to_num(source_sar, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_sar = np.stack(
        [normalize_channel(source_sar[0]), normalize_channel(source_sar[1])], axis=0
    ).astype(np.float32, copy=False)

    sar_tensor = torch.from_numpy(normalized_sar).unsqueeze(0)
    sar_tensor = F.interpolate(
        sar_tensor, size=(256, 256), mode="bilinear", align_corners=False
    ).to(dtype=torch.float32)
    preview_sar = sar_tensor[0].numpy()

    # SAR is not an RGB image.  A neutral intensity composite avoids falsely
    # presenting VV/VH as red/green/blue colours.
    combined_preview = channel_preview(preview_sar.mean(axis=0)).convert("RGB")
    return sar_tensor.to(DEVICE, dtype=torch.float32), {
        "vv": channel_preview(preview_sar[0]),
        "vh": channel_preview(preview_sar[1]),
        "sar": combined_preview,
    }


def calculate_metrics(
    prediction: Union[Image.Image, np.ndarray], ground_truth: Image.Image
) -> Dict[str, Optional[float]]:
    """Calculate metrics from raw radiometric RGB, never display-enhanced RGB."""
    if isinstance(prediction, Image.Image):
        prediction_array = np.asarray(prediction.convert("RGB"), dtype=np.float32) / 255.0
    else:
        prediction_array = np.asarray(prediction, dtype=np.float32)
        if prediction_array.ndim != 3 or prediction_array.shape[-1] != 3:
            raise ValueError("Metric prediction must have shape [height, width, 3].")
        prediction_array = np.nan_to_num(
            prediction_array, nan=0.0, posinf=1.0, neginf=0.0
        ).clip(0.0, 1.0)
    target = ground_truth.convert("RGB").resize(
        (prediction_array.shape[1], prediction_array.shape[0]), Image.Resampling.BICUBIC
    )
    target_array = np.asarray(target, dtype=np.float32) / 255.0
    psnr = peak_signal_noise_ratio(target_array, prediction_array, data_range=1.0)
    return {
        "psnr": None if not np.isfinite(psnr) else float(psnr),
        "ssim": float(
            structural_similarity(target_array, prediction_array, channel_axis=2, data_range=1.0)
        ),
        "rgb_l1": float(np.mean(np.abs(target_array - prediction_array))),
    }


def pix2pix_generate(image: Image.Image) -> Tuple[Image.Image, float]:
    model = require_model(PIX2PIX_MODEL, PIX2PIX_ERROR, "Pix2Pix")
    resized = image.resize((256, 256), Image.Resampling.BICUBIC)
    input_array = np.asarray(resized, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(input_array).permute(2, 0, 1)
    input_tensor = ((input_tensor - 0.5) / 0.5).unsqueeze(0).to(DEVICE)
    LOGGER.info("Pix2Pix input shape: %s", tuple(input_tensor.shape))
    start = time.perf_counter()
    with torch.inference_mode():
        output = ((model(input_tensor)[0] + 1.0) / 2.0).clamp(0, 1)
    duration_ms = (time.perf_counter() - start) * 1000
    LOGGER.info("Pix2Pix inference duration: %.2f ms", duration_ms)
    return tensor_to_image(output), duration_ms


def sarfusionformer_generate(
    vv_array: np.ndarray, vh_array: np.ndarray, apply_color_correction: bool = False
) -> Dict[str, Any]:
    model = require_model(
        SARFUSIONFORMER_MODEL, SARFUSIONFORMER_ERROR, "SARFusionFormer"
    )
    sar, previews = prepare_sarfusionformer_input(vv_array, vh_array)
    LOGGER.info(
        "SARFusionFormer input shape=%s dtype=%s range=[%.6f, %.6f]",
        tuple(sar.shape),
        sar.dtype,
        sar.amin().item(),
        sar.amax().item(),
    )
    start = time.perf_counter()
    with torch.inference_mode():
        prediction_lab = model(sar)["lab"]
        if not torch.isfinite(prediction_lab).all():
            raise ValueError("SARFusionFormer produced non-finite LAB values.")
        raw_rgb = lab_to_rgb(prediction_lab.float()).clamp(0, 1)
        if not torch.isfinite(raw_rgb).all():
            raise ValueError("LAB-to-RGB conversion produced non-finite RGB values.")

        # This is the scientific prediction. The display path receives an
        # independent copy and is checked below so it can never overwrite it.
        raw_rgb_hwc = raw_rgb[0].detach().cpu().permute(1, 2, 0).numpy().copy()
        raw_rgb_before_display = raw_rgb_hwc.copy()
        display_rgb_hwc, diagnostics, warning = display_representation(raw_rgb_hwc.copy())
        if not np.array_equal(raw_rgb_hwc, raw_rgb_before_display):
            raise RuntimeError("Display processing modified the raw RGB prediction.")
        diagnostics.update(
            array_diagnostics(prediction_lab[0].detach().cpu().numpy(), "prediction_lab")
        )
        diagnostics.update(
            {
                "prediction_shape": list(raw_rgb.shape),
                "prediction_dtype": str(raw_rgb.dtype),
                "lab_finite": True,
                "rgb_finite": True,
                "raw_prediction_preserved": True,
                "inference_successful": True,
            }
        )
        if diagnostics["raw_rgb_max"] - diagnostics["raw_rgb_min"] < 0.1:
            narrow_range_message = (
                "Model inference succeeded. The prediction has a narrow radiometric range, "
                "so the raw image appears dark."
            )
            warning = (
                narrow_range_message if warning is None else "{} {}".format(warning, narrow_range_message)
            )
        LOGGER.info("Loaded checkpoint: %s", SARFUSIONFORMER_CHECKPOINT.name)
        LOGGER.info("Prediction shape: %s", tuple(raw_rgb.shape))
        LOGGER.info("Prediction dtype: %s", raw_rgb.dtype)
        LOGGER.info(
            "Prediction min/max: %.6f / %.6f; Prediction mean/std: %.6f / %.6f",
            diagnostics["raw_rgb_min"],
            diagnostics["raw_rgb_max"],
            diagnostics["raw_rgb_mean"],
            diagnostics["raw_rgb_std"],
        )
        LOGGER.info(
            "RGB image range: [%.6f, %.6f]; Enhanced image range: [%.6f, %.6f]",
            diagnostics["raw_rgb_min"],
            diagnostics["raw_rgb_max"],
            diagnostics["display_rgb_min"],
            diagnostics["display_rgb_max"],
        )
        LOGGER.info("Lab-to-RGB conversion: completed; Inference successful: True")
        LOGGER.info(
            "SARFusionFormer Lab min=%.6f max=%.6f mean=%.6f std=%.6f; "
            "raw RGB min=%.6f max=%.6f mean=%.6f std=%.6f; stretch=[%.6f, %.6f]",
            diagnostics["prediction_lab_min"],
            diagnostics["prediction_lab_max"],
            diagnostics["prediction_lab_mean"],
            diagnostics["prediction_lab_std"],
            diagnostics["raw_rgb_min"],
            diagnostics["raw_rgb_max"],
            diagnostics["raw_rgb_mean"],
            diagnostics["raw_rgb_std"],
            diagnostics["stretch_low"],
            diagnostics["stretch_high"],
        )
        corrected_rgb = None
        corrected_raw_hwc = None
        corrected_display_hwc = None
        corrected_diagnostics = None
        if apply_color_correction and COLOR_CORRECTOR is not None:
            corrected_rgb, _ = COLOR_CORRECTOR(raw_rgb)
            corrected_raw_hwc = corrected_rgb[0].detach().cpu().permute(1, 2, 0).numpy()
            corrected_display_hwc, corrected_diagnostics, corrected_warning = display_representation(
                corrected_raw_hwc
            )
            if corrected_warning:
                warning = corrected_warning if warning is None else "{} {}".format(warning, corrected_warning)
        elif apply_color_correction:
            correction_warning = COLOR_CORRECTOR_ERROR or "Color corrector is unavailable."
            warning = correction_warning if warning is None else "{} {}".format(warning, correction_warning)
    duration_ms = (time.perf_counter() - start) * 1000
    LOGGER.info("SARFusionFormer inference duration: %.2f ms", duration_ms)

    return {
        "raw_rgb": raw_rgb_hwc,
        "display_rgb": display_rgb_hwc,
        "corrected_raw_rgb": corrected_raw_hwc,
        "corrected_display_rgb": corrected_display_hwc,
        "previews": previews,
        "duration_ms": duration_ms,
        "warning": warning,
        "diagnostics": diagnostics,
        "corrected_diagnostics": corrected_diagnostics,
    }


def pix2pix_payload(
    input_bytes: bytes, ground_truth_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    source = decode_rgb(input_bytes)
    output, duration_ms = pix2pix_generate(source)
    target = decode_rgb(ground_truth_bytes) if ground_truth_bytes else None
    return {
        "input_preview": image_to_base64(source),
        "output": image_to_base64(output),
        "metrics": calculate_metrics(output, target) if target else None,
        "inference_time_ms": round(duration_ms, 2),
        "checkpoint": PIX2PIX_CHECKPOINT.name,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ready",
        "device": str(DEVICE),
        "models": {
            "pix2pix": model_status(PIX2PIX_MODEL, PIX2PIX_ERROR, PIX2PIX_CHECKPOINT),
            "sarfusionformer": model_status(
                SARFUSIONFORMER_MODEL, SARFUSIONFORMER_ERROR, SARFUSIONFORMER_CHECKPOINT
            ),
            "color_corrector": model_status(
                COLOR_CORRECTOR, COLOR_CORRECTOR_ERROR, COLOR_CORRECTOR_CHECKPOINT
            ),
        },
        "vision_analysis": {**VISION_ANALYSIS.status(), **PROVIDER_SETTINGS.public_status()},
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> StreamingResponse:
    """Legacy Pix2Pix endpoint retained for existing clients."""
    try:
        payload = pix2pix_payload(read_upload(file))
        output_bytes = base64.b64decode(payload["output"])
        return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/evaluate")
async def evaluate(
    sar_file: UploadFile = File(...), optical_file: UploadFile = File(...)
) -> StreamingResponse:
    """Legacy Pix2Pix evaluation endpoint retained for existing clients."""
    try:
        payload = pix2pix_payload(read_upload(sar_file), read_upload(optical_file))
        output_bytes = base64.b64decode(payload["output"])
        metrics = payload["metrics"] or {}
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers={
                "X-PSNR": "{:.2f}".format(metrics["psnr"]) if metrics["psnr"] is not None else "N/A",
                "X-SSIM": "{:.4f}".format(metrics["ssim"]) if metrics["ssim"] is not None else "N/A",
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/pix2pix/infer")
async def pix2pix_infer(
    file: UploadFile = File(...), ground_truth: Optional[UploadFile] = File(None)
) -> JSONResponse:
    try:
        target = read_upload(ground_truth) if ground_truth else None
        return JSONResponse(pix2pix_payload(read_upload(file), target))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/sarfusionformer/infer")
async def sarfusionformer_infer(
    combined_file: Optional[UploadFile] = File(None),
    vv_file: Optional[UploadFile] = File(None),
    vh_file: Optional[UploadFile] = File(None),
    ground_truth: Optional[UploadFile] = File(None),
    apply_color_correction: bool = Form(False),
) -> JSONResponse:
    try:
        if combined_file is not None:
            if Path(combined_file.filename or "").suffix.lower() != ".npy":
                raise ValueError("Combined VV/VH input must be a NumPy (.npy) file.")
            vv, vh, channel_layout, detected_shape = load_combined_sar_npy(
                read_upload(combined_file)
            )
        elif vv_file is not None and vh_file is not None:
            vv = decode_grayscale(read_upload(vv_file), vv_file.filename or "")
            vh = decode_grayscale(read_upload(vh_file), vh_file.filename or "")
            channel_layout = "Separate VV and VH files"
            detected_shape = tuple(vv.shape)
        elif vv_file is None:
            raise ValueError("Provide a combined VV/VH .npy file or upload a VV file.")
        else:
            raise ValueError("Provide a combined VV/VH .npy file or upload a VH file.")
        result = sarfusionformer_generate(
            vv, vh, apply_color_correction=apply_color_correction
        )
        target = decode_rgb(read_upload(ground_truth)) if ground_truth else None
        return JSONResponse(
            {
                "raw_output": image_to_base64(rgb_float_to_png(result["raw_rgb"])),
                "display_output": image_to_base64(rgb_float_to_png(result["display_rgb"])),
                "corrected_raw_output": (
                    image_to_base64(rgb_float_to_png(result["corrected_raw_rgb"]))
                    if result["corrected_raw_rgb"] is not None
                    else None
                ),
                "corrected_display_output": (
                    image_to_base64(rgb_float_to_png(result["corrected_display_rgb"]))
                    if result["corrected_display_rgb"] is not None
                    else None
                ),
                "vv_preview": image_to_base64(result["previews"]["vv"]),
                "vh_preview": image_to_base64(result["previews"]["vh"]),
                "sar_preview": image_to_base64(result["previews"]["sar"]),
                "metrics_raw": calculate_metrics(result["raw_rgb"], target) if target else None,
                "metrics_corrected": (
                    calculate_metrics(result["corrected_raw_rgb"], target)
                    if target and result["corrected_raw_rgb"] is not None
                    else None
                ),
                "inference_time_ms": round(result["duration_ms"], 2),
                "detected_shape": list(detected_shape),
                "channel_layout": channel_layout,
                "checkpoint": SARFUSIONFORMER_CHECKPOINT.name,
                "color_checkpoint": (
                    COLOR_CORRECTOR_CHECKPOINT.name
                    if result["corrected_raw_rgb"] is not None
                    else None
                ),
                "warning": result["warning"],
                "diagnostics": result["diagnostics"],
                "corrected_diagnostics": result["corrected_diagnostics"],
            }
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/compare")
async def compare(
    pix2pix_output: UploadFile = File(...),
    sarfusionformer_output: UploadFile = File(...),
    ground_truth: UploadFile = File(...),
) -> JSONResponse:
    try:
        target = decode_rgb(read_upload(ground_truth))
        pix_metrics = calculate_metrics(decode_rgb(read_upload(pix2pix_output)), target)
        sar_metrics = calculate_metrics(decode_rgb(read_upload(sarfusionformer_output)), target)
        return JSONResponse({"pix2pix": pix_metrics, "sarfusionformer": sar_metrics})
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/analysis/image")
async def analyze_image(
    image: UploadFile = File(...),
    analysis_type: str = Form(...),
    metadata: Optional[str] = Form(None),
) -> JSONResponse:
    """Qualitatively review a rendered image without touching inference or metrics."""
    try:
        image_bytes = read_upload(image)
        decoded = decode_rgb(image_bytes)
        metadata_object: Dict[str, Any] = {}
        if metadata:
            parsed = json.loads(metadata)
            if not isinstance(parsed, dict):
                raise ValueError("Analysis metadata must be a JSON object.")
            metadata_object = parsed
        # Re-encode accepted images so the provider sees the safe RGB rendering
        # displayed by the application, not an arbitrary uploaded container.
        buffer = io.BytesIO()
        decoded.save(buffer, format="PNG")
        result = VISION_ANALYSIS.analyze_image(
            buffer.getvalue(), "image/png", analysis_type, metadata_object
        )
        LOGGER.info(
            "AI image analysis completed: type=%s provider=%s model=%s cached=%s",
            analysis_type,
            result["provider"],
            result["model"],
            result["cached"],
        )
        return JSONResponse(result)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="Analysis metadata must be valid JSON.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except VisionAnalysisError as error:
        status_code = 503 if not VISION_ANALYSIS.settings.configured else 502
        raise HTTPException(status_code=status_code, detail=str(error)) from error


@app.get("/api/settings/provider")
def get_provider_settings() -> JSONResponse:
    """Return non-sensitive provider status for the Settings UI."""
    return JSONResponse(PROVIDER_SETTINGS.public_status())


@app.post("/api/settings/provider")
def save_provider_settings(configuration: ProviderConfigurationRequest) -> JSONResponse:
    """Store a local key in macOS Keychain; never return its raw value."""
    try:
        status = PROVIDER_SETTINGS.save(configuration.provider, configuration.api_key)
        sync_vision_settings()
        return JSONResponse({"message": "Configuration saved successfully.", **status})
    except ProviderSettingsError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/settings/provider")
def delete_provider_settings() -> JSONResponse:
    try:
        status = PROVIDER_SETTINGS.delete()
        sync_vision_settings()
        return JSONResponse({"message": "Configuration deleted.", **status})
    except ProviderSettingsError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/settings/test")
def test_provider_settings() -> JSONResponse:
    result = PROVIDER_SETTINGS.test_connection()
    sync_vision_settings()
    return JSONResponse(result)
