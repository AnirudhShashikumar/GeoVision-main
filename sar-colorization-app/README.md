# SAR Optical Reconstruction App

A Streamlit interface and FastAPI service for two independent SAR-to-optical workflows:

- **Pix2Pix** for visually realistic optical reconstruction from the repository's existing SAR image representation.
- **SARFusionFormer** for structure-preserving reconstruction from separate raw VV and VH channels.

The application compares the two results side by side but never blends them.

## Checkpoint placement

Keep model weights out of Git. The default local paths are:

```text
../pix2pix_gen_180.pth
../models/checkpoints/sarfusionformer_256_decoder_best.pt
../models/checkpoints/color_corrector_256_best.pt
```

The color-corrector is optional and is disabled by default. Verify the raw SARFusionFormer result first, then explicitly enable colour correction if desired. If its checkpoint is unavailable, raw SARFusionFormer inference remains available.

## Run locally

From this directory, start the API:

```bash
venv/bin/uvicorn backend:app --host 127.0.0.1 --port 8010
```

In a second terminal, start the interface:

```bash
venv/bin/streamlit run app.py
```

The interface connects to `http://127.0.0.1:8010` by default.

## Environment variables

- `PIX2PIX_CHECKPOINT`: path to the Pix2Pix generator checkpoint.
- `SARFUSIONFORMER_CHECKPOINT`: path to `sarfusionformer_256_decoder_best.pt`.
- `COLOR_CORRECTOR_CHECKPOINT`: path to `color_corrector_256_best.pt`.
- `SAR_COLORIZATION_API_URL`: API URL used by Streamlit.
- `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_DEFAULT_MODEL`, `GEMINI_SUPPORTED_MODELS`: optional server-side Gemini configuration. On local macOS installs, configure a Gemini key through Settings → AI Providers; it is stored in macOS Keychain and never added to Git. Do not set `NEXT_PUBLIC_GEMINI_API_KEY`.
- `GEOVISION_SECRET_ENCRYPTION_KEY`: reserved for an encrypted server-side secret-store deployment; keep it server-only and out of Git.
- `CORS_ORIGINS`: comma-separated trusted frontend origins; defaults to the local GeoVision frontend.

## API routes

- `POST /predict`: legacy Pix2Pix PNG response.
- `POST /evaluate`: legacy Pix2Pix PNG response with PSNR and SSIM headers.
- `POST /api/pix2pix/infer`: Pix2Pix JSON inference response.
- `POST /api/sarfusionformer/infer`: independent VV/VH SARFusionFormer inference response.
- `POST /api/compare`: metrics for two already-generated outputs and one common ground truth.
- `POST /api/analysis/image`: optional qualitative analysis of a rendered image; it never changes inference or metrics.
- `GET|POST|DELETE /api/settings/ai-provider`: retrieve non-sensitive Gemini status, validate and save a local secure configuration, or remove it. Legacy `/api/settings/provider` routes remain available.
- `GET /api/settings/ai-provider/models?provider=gemini`: Gemini image-capable models enabled by this backend.
- `POST /api/settings/ai-provider/test`: lightweight server-side Gemini connection test. Legacy `/api/settings/test` remains available.
- `GET /health`: independent availability for all three loaded model components.

## Input requirements

Pix2Pix accepts the same RGB image formats as the original project. SARFusionFormer defaults to one combined `.npy` input containing VV and VH as `[2, H, W]`, `[H, W, 2]`, `[1, 2, H, W]`, or `[1, H, W, 2]`. It also accepts separate VV and VH files as `.npy`, TIFF, PNG, or JPEG; those channels must have matching spatial dimensions.

SARFusionFormer replaces invalid numeric values with zero, independently percentile-normalizes VV and VH using the 1st and 99th percentiles, and resizes both channels to 256 × 256 before inference. The backend uses the training model's GroupNorm/GELU, shifted-window attention, decoder, and normalized-Lab-to-sRGB conversion exactly; it does not apply log/dB, mean/std, gamma, histogram, global, or `[-1,1]` input normalization.

## Scientific output and display output

SARFusionFormer returns both a raw radiometric PNG and an enhanced display PNG. The latter applies one global 2nd/98th-percentile stretch across the complete RGB image solely for on-screen inspection; it never changes channel balance. Metrics, raw downloads, and the optional colour corrector always use the unmodified float RGB model output.
