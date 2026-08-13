# SAR Colorization App

A Streamlit interface and FastAPI service for the repository's Pix2Pix SAR-to-optical generator.

## Features

- Upload a SAR image and generate a 256 × 256 optical-style RGB image
- Download the generated image as PNG
- Compare a generated image against a paired optical reference using PSNR and SSIM
- Display backend readiness, selected checkpoint, and runtime device

## Run locally

From this directory, start the model API:

```bash
venv/bin/uvicorn backend:app --host 127.0.0.1 --port 8010
```

In a second terminal, start the interface:

```bash
venv/bin/streamlit run app.py
```

The interface connects to `http://127.0.0.1:8010` by default.

## Configuration

- `SAR_COLORIZATION_CHECKPOINT`: optional path to a compatible generator checkpoint
- `SAR_COLORIZATION_API_URL`: optional API URL for the Streamlit interface

By default, the service loads `../pix2pix_gen_180.pth`, the working checkpoint used by the root project's inference script.

## Project structure

- `app.py`: Streamlit user interface
- `backend.py`: FastAPI inference and evaluation service
- `requirements.txt`: frontend service dependencies
