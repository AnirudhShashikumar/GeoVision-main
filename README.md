# SAR2Optical

Pix2Pix conditional GAN for translating three-channel Sentinel-1 SAR images into optical-style RGB images. The repository includes command-line inference, model training/evaluation utilities, ONNX export, and a Streamlit web interface.

## Quick start: web interface

The web interface requires a compatible generator checkpoint, which is deliberately not stored in Git.

1. Install the root project dependencies:

   ```bash
   uv sync --group train
   ```

2. Download the documented generator checkpoint to the repository root:

   ```bash
   curl -L -o pix2pix_gen_180.pth https://huggingface.co/yuulind/pix2pix-sar2rgb/resolve/main/pix2pix_gen_180.pth
   ```

3. Install the interface dependencies:

   ```bash
   python -m venv sar-colorization-app/venv
   sar-colorization-app/venv/bin/pip install -r sar-colorization-app/requirements.txt
   ```

4. Start the API from the repository root:

   ```bash
   sar-colorization-app/venv/bin/uvicorn backend:app --app-dir sar-colorization-app --host 127.0.0.1 --port 8010
   ```

5. In another terminal, start the Streamlit interface:

   ```bash
   sar-colorization-app/venv/bin/streamlit run sar-colorization-app/app.py
   ```

Open the local URL Streamlit prints. The interface supports image translation, PNG download, and paired-image PSNR/SSIM evaluation.

## Command-line inference

Set the image and checkpoint paths in [config.yaml](config.yaml), then run:

```bash
uv run python inference.py
```

The default checkpoint path is `pix2pix_gen_180.pth`. The command writes the result to `output/sample_output.jpg`.

## Training, data, and export

- Dataset: [SEN1-2 Sentinel-1 & Sentinel-2 image pairs](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain). Download it with `uv sync --group data` and `uv run python utils/data_downloader.py`.
- Training: `uv sync --group train`, then configure `config.yaml` and run `uv run python train.py`.
- Evaluation: configure the dataset and checkpoint, then run `uv run python test.py`.
- ONNX export: `uv sync --group export`, configure `config.yaml`, then run `uv run python torch2onnx.py`.

## Model files

The original pretrained model files are hosted on [Hugging Face](https://huggingface.co/yuulind/pix2pix-sar2rgb). Checkpoints, datasets, virtual environments, and generated files are ignored by Git to keep clones lightweight and prevent accidental publication of local artifacts.

## Repository structure

- `src/`: Pix2Pix architecture, dataset loader, and metrics
- `utils/`: configuration, logging, and dataset download utilities
- `sar-colorization-app/`: Streamlit UI and FastAPI inference service
- `config.yaml`: runtime, training, and export configuration
