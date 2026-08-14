"""Server-side AI provider credentials and connection status.

Local GeoVision installations use the macOS Keychain for API keys.  Deployed
instances should use environment variables, which are intentionally read-only
from the web UI.  Neither mechanism returns the raw credential to the client.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


PROVIDERS = {
    "openai": {"label": "OpenAI", "model": "gpt-4.1-mini", "supported": True},
    "gemini": {"label": "Google Gemini", "model": None, "supported": False},
    "anthropic": {"label": "Anthropic Claude", "model": None, "supported": False},
}
KEYCHAIN_SERVICE = "GeoVision.AIProvider"
KEYCHAIN_ACCOUNT = "local-user"


class ProviderSettingsError(RuntimeError):
    """A recoverable provider-configuration error appropriate for the UI."""


def mask_key(api_key: str) -> str:
    """Show only a stable, non-sensitive hint of an API key."""
    if len(api_key) < 8:
        return "••••••••"
    prefix = api_key[:3] if api_key.startswith("sk-") else api_key[:2]
    return "{}{}{}".format(prefix, "*" * 32, api_key[-4:])


@dataclass(frozen=True)
class ProviderCredentials:
    provider: str
    api_key: Optional[str]
    model: Optional[str]
    source: Optional[str]


class ProviderSettingsStore:
    """Keeps provider metadata in a private state file and secrets in Keychain."""

    def __init__(self) -> None:
        self.state_path = Path.home() / ".config" / "geovision" / "provider.json"

    def _load_local_environment(self) -> None:
        """Keep compatibility with a local, gitignored server .env file."""
        env_file = Path(__file__).with_name(".env")
        if not env_file.is_file():
            return
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    def _environment_credentials(self) -> Optional[ProviderCredentials]:
        self._load_local_environment()
        provider = os.getenv("VISION_PROVIDER", "").strip().lower()
        api_key = os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not provider and api_key:
            provider = "openai"
        if not provider or not api_key:
            return None
        return ProviderCredentials(
            provider=provider,
            api_key=api_key,
            model=os.getenv("VISION_MODEL") or PROVIDERS.get(provider, {}).get("model"),
            source="environment",
        )

    def _read_state(self) -> Dict[str, Any]:
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            return state if isinstance(state, dict) else {}
        except (FileNotFoundError, OSError, ValueError):
            return {}

    def _write_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle)
        temporary.chmod(0o600)
        temporary.replace(self.state_path)

    def _keychain_available(self) -> bool:
        return platform.system() == "Darwin"

    def _read_keychain(self) -> Optional[str]:
        if not self._keychain_available():
            return None
        result = subprocess.run(
            ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None

    def _write_keychain(self, api_key: str) -> None:
        if not self._keychain_available():
            raise ProviderSettingsError(
                "Secure local key storage is unavailable on this system. Use a server environment variable instead."
            )
        result = subprocess.run(
            ["security", "add-generic-password", "-U", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE, "-w", api_key],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ProviderSettingsError("GeoVision could not save the API key in macOS Keychain.")

    def _delete_keychain(self) -> None:
        if not self._keychain_available():
            return
        subprocess.run(
            ["security", "delete-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE],
            capture_output=True,
            text=True,
            check=False,
        )

    def credentials(self) -> ProviderCredentials:
        environment = self._environment_credentials()
        if environment:
            return environment
        state = self._read_state()
        keychain_key = self._read_keychain()
        return ProviderCredentials(
            provider=str(state.get("provider", "")).lower(),
            api_key=keychain_key,
            model=state.get("model"),
            source="keychain" if keychain_key else None,
        )

    def public_status(self) -> Dict[str, Any]:
        credentials = self.credentials()
        configured = bool(credentials.api_key and credentials.provider)
        state = self._read_state()
        configured_provider = PROVIDERS.get(credentials.provider, {})
        return {
            "configured": configured,
            "provider": configured_provider.get("label") or None,
            "provider_id": credentials.provider or None,
            "model": credentials.model if configured else None,
            "masked_key": mask_key(credentials.api_key) if credentials.api_key else None,
            "source": credentials.source,
            "connection_status": state.get("connection_status", "not_configured" if not configured else "connected"),
            "supported": bool(configured_provider.get("supported")),
            "managed_by_environment": credentials.source == "environment",
        }

    def save(self, provider: str, api_key: str) -> Dict[str, Any]:
        provider = provider.strip().lower()
        if provider not in PROVIDERS:
            raise ProviderSettingsError("Choose a supported provider option.")
        if not api_key.strip():
            raise ProviderSettingsError("API key cannot be empty.")
        if self._environment_credentials():
            raise ProviderSettingsError("This deployment is configured through environment variables and cannot be changed here.")
        self._write_keychain(api_key.strip())
        self._write_state({
            "provider": provider,
            "model": PROVIDERS[provider]["model"],
            "connection_status": "configured",
        })
        return self.public_status()

    def delete(self) -> Dict[str, Any]:
        if self._environment_credentials():
            raise ProviderSettingsError("This deployment is configured through environment variables and cannot be deleted here.")
        self._delete_keychain()
        self._write_state({"connection_status": "not_configured"})
        return self.public_status()

    def test_connection(self) -> Dict[str, Any]:
        credentials = self.credentials()
        if not credentials.api_key or not credentials.provider:
            return {"success": False, "message": "AI analysis requires an API key."}
        provider = PROVIDERS.get(credentials.provider)
        if not provider or not provider["supported"]:
            self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "not_configured"})
            return {"success": False, "message": "This provider is marked for future support."}
        try:
            response = requests.get(
                "https://api.openai.com/v1/models/{}".format(credentials.model or provider["model"]),
                headers={"Authorization": "Bearer {}".format(credentials.api_key)},
                timeout=15,
            )
        except requests.RequestException:
            return {"success": False, "message": "Could not reach the AI provider."}
        if response.status_code == 401:
            self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "invalid"})
            return {"success": False, "message": "Invalid API key."}
        if response.status_code == 429:
            self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "configured"})
            return {"success": False, "message": "OpenAI has no available API quota for this project. Add billing or use a funded API key."}
        if response.status_code == 403:
            self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "configured"})
            return {"success": False, "message": "The API key is valid but does not have access to the configured model."}
        if not response.ok:
            return {"success": False, "message": "Provider request failed. Check key permissions and model access."}
        self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "connected"})
        return {
            "success": True,
            "provider": provider["label"],
            "model": credentials.model or provider["model"],
            "message": "Connection successful.",
        }
