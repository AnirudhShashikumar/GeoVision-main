"""Streamlit frontend for the project Pix2Pix SAR-to-optical model."""

from __future__ import annotations

import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("SAR_COLORIZATION_API_URL", "http://127.0.0.1:8010").rstrip("/")
IMAGE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]

st.set_page_config(page_title="SAR Colorization", page_icon="🛰️", layout="wide")
st.title("SAR → Optical Colorization")
st.caption(
    "Translate three-channel Sentinel-1 SAR imagery into an optical-style RGB image "
    "with the repository's trained Pix2Pix generator."
)


def api_error(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def predict(file, endpoint: str, extra_file=None) -> requests.Response:
    field_name = "sar_file" if endpoint == "evaluate" else "file"
    files = {field_name: (file.name, file.getvalue(), file.type or "image/png")}
    if extra_file is not None:
        files["optical_file"] = (
            extra_file.name,
            extra_file.getvalue(),
            extra_file.type or "image/png",
        )
    response = requests.post(f"{API_URL}/{endpoint}", files=files, timeout=90)
    response.raise_for_status()
    return response


with st.sidebar:
    st.subheader("Service status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        health.raise_for_status()
        status = health.json()
        st.success("Model ready")
        st.caption(f"Checkpoint: {status['checkpoint']} · Device: {status['device']}")
    except requests.RequestException:
        st.warning("Backend is offline")
        st.caption("Start the FastAPI service, then refresh this page.")

translate_tab, evaluate_tab, about_tab = st.tabs(
    ["Translate", "Evaluate", "About the model"]
)

with translate_tab:
    uploaded_sar = st.file_uploader("Upload a SAR image", type=IMAGE_TYPES, key="translate")
    if uploaded_sar is not None:
        left, right = st.columns(2)
        with left:
            st.markdown("**Input SAR**")
            st.image(
                Image.open(io.BytesIO(uploaded_sar.getvalue())).convert("RGB"),
                use_container_width=True,
            )
        if st.button("Generate optical image", type="primary", key="translate_button"):
            with st.spinner("Running the Pix2Pix generator…"):
                try:
                    response = predict(uploaded_sar, "predict")
                    output_bytes = response.content
                    with right:
                        st.markdown("**Predicted optical RGB**")
                        st.image(Image.open(io.BytesIO(output_bytes)), use_container_width=True)
                        st.download_button(
                            "Download PNG",
                            data=output_bytes,
                            file_name=f"optical_{uploaded_sar.name.rsplit('.', 1)[0]}.png",
                            mime="image/png",
                        )
                except requests.HTTPError as error:
                    st.error(api_error(error.response))
                except requests.RequestException:
                    st.error(
                        "The backend could not be reached. Confirm that it is running, then try again."
                    )

with evaluate_tab:
    st.write(
        "Compare a generated image with its paired optical reference. Both images "
        "are resized to 256 × 256 before evaluation."
    )
    input_column, target_column = st.columns(2)
    with input_column:
        evaluation_sar = st.file_uploader("Input SAR image", type=IMAGE_TYPES, key="eval_sar")
    with target_column:
        reference_optical = st.file_uploader(
            "Ground-truth optical image", type=IMAGE_TYPES, key="eval_optical"
        )
    if evaluation_sar and reference_optical and st.button(
        "Calculate PSNR and SSIM", type="primary"
    ):
        with st.spinner("Generating and measuring…"):
            try:
                response = predict(evaluation_sar, "evaluate", reference_optical)
                metric_left, metric_right = st.columns(2)
                metric_left.metric("PSNR", f"{response.headers['X-PSNR']} dB")
                metric_right.metric("SSIM", response.headers["X-SSIM"])
                display_input, display_prediction, display_reference = st.columns(3)
                display_input.image(
                    Image.open(io.BytesIO(evaluation_sar.getvalue())).convert("RGB"),
                    caption="Input SAR",
                    use_container_width=True,
                )
                display_prediction.image(
                    Image.open(io.BytesIO(response.content)),
                    caption="Predicted optical",
                    use_container_width=True,
                )
                display_reference.image(
                    Image.open(io.BytesIO(reference_optical.getvalue())).convert("RGB"),
                    caption="Ground truth",
                    use_container_width=True,
                )
            except requests.HTTPError as error:
                st.error(api_error(error.response))
            except requests.RequestException:
                st.error(
                    "The backend could not be reached. Confirm that it is running, then try again."
                )

with about_tab:
    st.markdown(
        """
        ### Current project model

        - 8-stage U-Net Pix2Pix generator trained to map 3-channel SAR inputs to RGB outputs.
        - 70 × 70 PatchGAN discriminator during training, with GAN and L1 reconstruction losses.
        - Inputs are resized to 256 × 256 and normalized to the −1 to 1 range used by the trained generator.
        - The evaluation tab reports PSNR and SSIM only when a paired optical reference is available.
        """
    )
