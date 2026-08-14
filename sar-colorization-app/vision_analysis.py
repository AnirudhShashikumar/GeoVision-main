"""Provider-neutral, server-side vision analysis for rendered SAR products.

This module deliberately operates on exported display images only.  It never
touches model tensors, checkpoints, preprocessing, or metric calculations.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


ANALYSIS_TYPES = {
    "pix2pix",
    "sarfusionformer_raw",
    "sarfusionformer_enhanced",
    "sarfusionformer_corrected",
    "ground_truth",
    "difference",
    "comparison",
}

SYSTEM_PROMPT = """You are an assistant supporting qualitative remote-sensing image review.
Analyze only visible patterns in the supplied rendered image. Do not claim that generated
SAR-to-optical imagery is real ground truth, do not infer precise geography, dates, land
ownership, activity, or identities, and do not make safety-critical decisions. Clearly
distinguish observations from plausible interpretations. Be especially cautious around
model hallucinations, synthetic colour, blur, tiling, and contrast enhancement. Return
only valid JSON matching the requested schema."""

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "executive_summary": {"type": "string"},
        "terrain": {"type": "array", "items": {"type": "string"}},
        "structural_and_human_features": {"type": "array", "items": {"type": "string"}},
        "vegetation_and_water": {"type": "array", "items": {"type": "string"}},
        "image_quality": {"type": "array", "items": {"type": "string"}},
        "possible_artifacts": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "disclaimer": {"type": "string"},
    },
    "required": [
        "executive_summary", "terrain", "structural_and_human_features",
        "vegetation_and_water", "image_quality", "possible_artifacts", "notes",
        "limitations", "recommended_actions", "confidence", "disclaimer",
    ],
}


@dataclass(frozen=True)
class VisionSettings:
    provider: str
    api_key: Optional[str]
    model: str

    @classmethod
    def from_environment(cls) -> "VisionSettings":
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.isfile(env_file):
            with open(env_file, "r", encoding="utf-8") as environment:
                for line in environment:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
        return cls(
            provider=os.getenv("VISION_PROVIDER", "").strip().lower(),
            api_key=os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY"),
            model=os.getenv("VISION_MODEL", "gpt-5.6").strip(),
        )

    @property
    def configured(self) -> bool:
        return self.provider == "openai" and bool(self.api_key) and bool(self.model)


class VisionAnalysisError(RuntimeError):
    """Expected configuration, provider, or schema error for this optional feature."""


class ImageAnalysisService:
    """Small provider interface; add providers here without exposing keys to the UI."""

    def __init__(self, settings: Optional[VisionSettings] = None) -> None:
        self.settings = settings or VisionSettings.from_environment()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.settings.configured,
            "provider": self.settings.provider or None,
            "model": self.settings.model if self.settings.configured else None,
        }

    def analyze_image(
        self, image_bytes: bytes, mime_type: str, analysis_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if analysis_type not in ANALYSIS_TYPES:
            raise VisionAnalysisError("Unsupported analysis type: {}".format(analysis_type))
        if not self.settings.configured:
            raise VisionAnalysisError(
                "AI Image Analysis is not configured. Set VISION_PROVIDER=openai, "
                "VISION_API_KEY, and VISION_MODEL on the backend server."
            )
        if self.settings.provider != "openai":
            raise VisionAnalysisError("Unsupported VISION_PROVIDER: {}".format(self.settings.provider))

        cache_key = hashlib.sha256(
            image_bytes + analysis_type.encode("utf-8") + json.dumps(metadata or {}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return {**self._cache[cache_key], "cached": True}

        encoded = base64.b64encode(image_bytes).decode("ascii")
        context = {
            "analysis_type": analysis_type,
            "metadata": metadata or {},
            "instruction": (
                "Provide a qualitative review of this rendered image. This may be a generated "
                "SAR-to-optical prediction, an enhanced display, a colour-corrected rendering, "
                "or reference imagery. Describe only visual evidence and preserve scientific caution."
            ),
        }
        payload = {
            "model": self.settings.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(context)},
                        {"type": "input_image", "image_url": "data:{};base64,{}".format(mime_type, encoded), "detail": "high"},
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "remote_sensing_image_review",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer {}".format(self.settings.api_key), "Content-Type": "application/json"},
                json=payload,
                timeout=75,
            )
        except requests.RequestException as error:
            raise VisionAnalysisError("The vision provider could not be reached.") from error
        if not response.ok:
            if response.status_code == 401:
                raise VisionAnalysisError("OpenAI rejected the saved API key. Replace it in Settings → AI Providers.")
            if response.status_code == 429:
                raise VisionAnalysisError(
                    "OpenAI has no available API quota for this project. Add billing or use a funded API key, then test the connection in Settings → AI Providers."
                )
            if response.status_code == 403:
                raise VisionAnalysisError("The saved API key does not have access to the configured model. Review the project permissions in Settings → AI Providers.")
            raise VisionAnalysisError("The vision provider could not complete this request (HTTP {}). Try again or test the connection in Settings → AI Providers.".format(response.status_code))

        try:
            response_data = response.json()
            output_text = response_data.get("output_text")
            if not output_text:
                output_text = next(
                    content["text"]
                    for item in response_data.get("output", [])
                    for content in item.get("content", [])
                    if content.get("type") == "output_text" and content.get("text")
                )
            report = json.loads(output_text)
        except (KeyError, StopIteration, TypeError, ValueError) as error:
            raise VisionAnalysisError("Vision provider returned an invalid structured analysis.") from error
        result = {"report": report, "provider": self.settings.provider, "model": self.settings.model, "cached": False}
        self._cache[cache_key] = result
        return result
