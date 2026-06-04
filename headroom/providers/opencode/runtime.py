"""Runtime helpers for OpenCode integrations."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib import error as urllib_error
from urllib import request as urllib_request

from headroom.copilot_auth import (
    DEFAULT_API_URL as DEFAULT_COPILOT_API_URL,
)
from headroom.copilot_auth import (
    CopilotSubscriptionTokenResolution,
    copilot_api_url_from_enterprise_url,
    token_fingerprint,
)
from headroom.providers.codex import proxy_base_url as codex_proxy_base_url

DEFAULT_MODEL = "gpt-4o"
DEFAULT_PROVIDER_ID = "headroom"
DEFAULT_CONTEXT_LIMIT = 128_000
DEFAULT_OUTPUT_LIMIT = 16_384
DEFAULT_COPILOT_PROVIDER_ID = "github-copilot"
ENTERPRISE_COPILOT_PROVIDER_ID = "github-copilot-enterprise"
COPILOT_PROVIDER_IDS = (ENTERPRISE_COPILOT_PROVIDER_ID, DEFAULT_COPILOT_PROVIDER_ID)
OPENCODE_COPILOT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

WireApi = Literal["completions", "responses"]


def _model_prefers_responses(model: str) -> bool:
    """Return True when the model should use OpenAI Responses by default."""
    normalized = model.strip().lower().rsplit("/", 1)[-1]
    return normalized.startswith("gpt-5")


def resolve_wire_api(wire_api: str | None, model: str) -> WireApi:
    """Resolve OpenCode's generated Headroom provider wire API."""
    if wire_api == "responses":
        return "responses"
    if wire_api == "completions":
        return "completions"
    return "responses" if _model_prefers_responses(model) else "completions"


def _provider_model_ref(provider_id: str, model: str) -> str:
    return f"{provider_id}/{model}"


def _normalize_model_ref(provider_id: str, model_ref: str) -> tuple[str, str]:
    """Return ``(opencode_model_ref, provider_model_id)`` for a model value."""
    prefix = f"{provider_id}/"
    if model_ref.startswith(prefix):
        model = model_ref[len(prefix) :]
        return model_ref, model
    return _provider_model_ref(provider_id, model_ref), model_ref


def normalize_model_args(
    args: tuple[str, ...],
    *,
    provider_id: str = DEFAULT_PROVIDER_ID,
    default_model: str = DEFAULT_MODEL,
) -> tuple[tuple[str, ...], str, bool]:
    """Normalize raw OpenCode model args to the generated Headroom provider.

    OpenCode expects ``provider/model``. The wrapper-level ``--model`` option
    sets the generated config default, while pass-through ``--model``/``-m``
    values are rewritten to ``headroom/<model>`` when the user supplies a raw
    model id.
    """
    normalized = list(args)
    selected_model = default_model
    changed = False

    for idx, arg in enumerate(normalized):
        if arg in {"--model", "-m"} and idx + 1 < len(normalized):
            model_ref, selected_model = _normalize_model_ref(provider_id, normalized[idx + 1])
            if model_ref != normalized[idx + 1]:
                normalized[idx + 1] = model_ref
                changed = True
            return tuple(normalized), selected_model, changed

        for prefix in ("--model=", "-m="):
            if arg.startswith(prefix):
                model_ref, selected_model = _normalize_model_ref(
                    provider_id, arg.split("=", 1)[1]
                )
                rewritten = f"{prefix}{model_ref}"
                if rewritten != arg:
                    normalized[idx] = rewritten
                    changed = True
                return tuple(normalized), selected_model, changed

    return tuple(normalized), selected_model, changed


def build_config(
    *,
    port: int,
    model: str,
    wire_api: WireApi,
    provider_id: str = DEFAULT_PROVIDER_ID,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
) -> dict[str, object]:
    """Build a temporary OpenCode config that routes through Headroom."""
    base_url = codex_proxy_base_url(port)
    npm_package = (
        "@ai-sdk/openai" if wire_api == "responses" else "@ai-sdk/openai-compatible"
    )
    model_ref = _provider_model_ref(provider_id, model)
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": model_ref,
        "small_model": model_ref,
        "provider": {
            provider_id: {
                "id": provider_id,
                "name": "Headroom",
                "env": [],
                "npm": npm_package,
                "api": base_url,
                "options": {
                    "baseURL": base_url,
                    "apiKey": "headroom",
                    "timeout": 600000,
                    "chunkTimeout": 60000,
                },
                "models": {
                    model: {
                        "id": model,
                        "name": model,
                        "attachment": True,
                        "reasoning": False,
                        "temperature": True,
                        "tool_call": True,
                        "release_date": "2026-01-01",
                        "limit": {
                            "context": context_limit,
                            "output": output_limit,
                        },
                        "cost": {
                            "input": 0,
                            "output": 0,
                        },
                        "options": {},
                    }
                },
            }
        },
    }


def build_launch_env(
    *,
    port: int,
    config_dir: Path,
    model: str,
    wire_api: WireApi,
    provider_id: str = DEFAULT_PROVIDER_ID,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for OpenCode through the local proxy."""
    env = dict(environ if environ is not None else os.environ)
    env["OPENCODE_CONFIG_DIR"] = str(config_dir)
    base_url = codex_proxy_base_url(port)
    return env, [
        f"OPENCODE_CONFIG_DIR={config_dir}",
        f"OPENCODE_MODEL={_provider_model_ref(provider_id, model)}",
        f"OPENCODE_WIRE_API={wire_api}",
        f"HEADROOM_BASE_URL={base_url}",
    ]


def opencode_auth_file_candidates(environ: Mapping[str, str] | None = None) -> list[Path]:
    """Return possible OpenCode auth.json paths without reading secret data."""
    env = environ if environ is not None else os.environ
    paths: list[Path] = []

    override = env.get("HEADROOM_OPENCODE_AUTH_FILE", "").strip()
    if override:
        paths.append(Path(override).expanduser())

    xdg_data_home = env.get("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        paths.append(Path(xdg_data_home).expanduser() / "opencode" / "auth.json")

    home = Path(env.get("HOME", "") or str(Path.home())).expanduser()
    paths.append(home / ".local" / "share" / "opencode" / "auth.json")
    paths.append(home / "Library" / "Application Support" / "opencode" / "auth.json")

    local_app_data = env.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        paths.append(Path(local_app_data).expanduser() / "opencode" / "auth.json")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_domain(value: str) -> str:
    return value.replace("https://", "", 1).replace("http://", "", 1).rstrip("/")


def _entry_api_url(auth: dict[str, object]) -> str:
    enterprise_url = auth.get("enterpriseUrl")
    if isinstance(enterprise_url, str) and enterprise_url.strip():
        return copilot_api_url_from_enterprise_url(enterprise_url.strip()).rstrip("/")
    return DEFAULT_COPILOT_API_URL


def _entry_exchange_url(auth: dict[str, object]) -> str:
    enterprise_url = auth.get("enterpriseUrl")
    domain = _normalize_domain(enterprise_url.strip()) if isinstance(enterprise_url, str) else "github.com"
    return f"https://api.{domain}/copilot_internal/v2/token"


def _expires_ms(value: object) -> float | None:
    if isinstance(value, (int, float)):
        timestamp = float(value)
        return timestamp * 1000 if timestamp < 10_000_000_000 else timestamp
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.isdigit():
            return _expires_ms(float(raw))
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc).timestamp() * 1000
    return None


def _valid_access_token(auth: dict[str, object]) -> str | None:
    access = auth.get("access")
    if not isinstance(access, str) or not access.strip():
        return None
    expires_ms = _expires_ms(auth.get("expires"))
    if expires_ms is None or expires_ms <= (time.time() * 1000):
        return None
    return access.strip()


def _exchange_opencode_copilot_token(
    refresh_token: str,
    *,
    exchange_url: str,
    timeout: float = 10.0,
) -> str | None:
    headers = {
        **OPENCODE_COPILOT_HEADERS,
        "Authorization": f"Bearer {refresh_token}",
    }
    req = urllib_request.Request(exchange_url, headers=headers)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib_error.URLError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    token = payload.get("token")
    return token.strip() if isinstance(token, str) and token.strip() else None


def _opencode_auth_payload(
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, object], str] | None:
    env = environ if environ is not None else os.environ
    auth_content = env.get("OPENCODE_AUTH_CONTENT", "").strip()
    if auth_content:
        try:
            payload = json.loads(auth_content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload, "env:OPENCODE_AUTH_CONTENT"

    for path in opencode_auth_file_candidates(env):
        payload = _read_json_object(path)
        if payload is not None:
            return payload, f"file:{path}"

    return None


def resolve_opencode_copilot_subscription_token_details(
    environ: Mapping[str, str] | None = None,
) -> CopilotSubscriptionTokenResolution | None:
    """Resolve OpenCode's stored GitHub Copilot OAuth token, if available."""
    payload_with_source = _opencode_auth_payload(environ)
    if payload_with_source is None:
        return None

    payload, source = payload_with_source
    for provider_id in COPILOT_PROVIDER_IDS:
        auth = payload.get(provider_id)
        if not isinstance(auth, dict):
            auth = payload.get(f"{provider_id}/")
        if not isinstance(auth, dict) or auth.get("type") != "oauth":
            continue

        token = _valid_access_token(auth)
        token_kind = "access"
        if token is None:
            refresh_token = auth.get("refresh")
            if not isinstance(refresh_token, str) or not refresh_token.strip():
                continue
            token = _exchange_opencode_copilot_token(
                refresh_token.strip(),
                exchange_url=_entry_exchange_url(auth),
            )
            token_kind = "exchange"
            if token is None:
                token = refresh_token.strip()
                token_kind = "refresh"

        api_url = _entry_api_url(auth)
        return CopilotSubscriptionTokenResolution(
            token=token,
            source=f"opencode:{source}:{provider_id}:{token_kind}",
            confidence=f"opencode-oauth-{token_kind}",
            api_url=api_url,
            token_fingerprint=token_fingerprint(token),
        )

    return None
