#!/usr/bin/env python3
"""One-time GitHub Copilot subscription diagnostics.

This script is intentionally standalone and secret-safe:

* It never prints bearer tokens, only short SHA-256 fingerprints.
* By default it reads only environment variables and opencode's auth.json.
* It does not touch Keychain, Secret Service, Credential Manager, or gh unless
  --allow-secret-store is passed.
* Generation probes are opt-in with --probe-generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_API_URL = "https://api.githubcopilot.com"
DEFAULT_BUSINESS_API_URL = "https://api.business.githubcopilot.com"
DEFAULT_USER_INFO_URL = "https://api.github.com/copilot_internal/user"
DEFAULT_OPENCODE_USER_AGENT = "opencode/1.15.11"
DEFAULT_COPILOT_USER_AGENT = "GitHubCopilotChat/0.1"
DEFAULT_EDITOR_VERSION = "vscode/1.104.1"

API_TOKEN_ENV_VARS = (
    "GITHUB_COPILOT_API_TOKEN",
    "COPILOT_PROVIDER_BEARER_TOKEN",
)
OAUTH_TOKEN_ENV_VARS = (
    "GITHUB_COPILOT_GITHUB_TOKEN",
    "GITHUB_COPILOT_TOKEN",
    "COPILOT_GITHUB_TOKEN",
)

SAFE_RESPONSE_HEADERS = {
    "content-type",
    "retry-after",
    "x-request-id",
    "x-github-request-id",
    "x-copilot-request-id",
    "cf-ray",
}
SAFE_RESPONSE_HEADER_PREFIXES = ("x-ratelimit-", "x-copilot-")


@dataclass(frozen=True)
class TokenCandidate:
    token: str
    source: str
    enterprise_url: str | None = None

    @property
    def fingerprint(self) -> str:
        return token_fingerprint(self.token)


@dataclass(frozen=True)
class HttpResult:
    method: str
    url: str
    status: int | None
    ok: bool
    headers: dict[str, str]
    text: str
    json_body: Any
    elapsed_ms: int
    error: str | None = None


def token_fingerprint(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).hexdigest()
    return f"sha256:{digest[:12]}"


def normalize_base_url(url: str) -> str:
    return url.strip().rstrip("/")


def dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_base_url(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def redacted_preview(value: Any, tokens: list[TokenCandidate], *, limit: int = 1200) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(redact_json(value), ensure_ascii=False, default=str)
        except Exception:
            text = repr(value)

    for candidate in tokens:
        if candidate.token:
            text = text.replace(candidate.token, "[REDACTED_TOKEN]")
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def redact_json(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in {"authorization", "token", "access_token", "refresh_token"}
                or normalized.endswith("_token")
                or normalized.endswith("_key")
                or normalized.endswith("_secret")
            ):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact_json(child)
        return redacted
    if isinstance(value, list):
        return [redact_json(item) for item in value]
    return value


def safe_header_subset(headers: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in dict(headers).items():
        normalized = str(key).lower()
        if normalized in SAFE_RESPONSE_HEADERS or normalized.startswith(
            SAFE_RESPONSE_HEADER_PREFIXES
        ):
            result[str(key)] = str(value)
    return result


def http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any] | None = None,
    timeout: float,
    tokens: list[TokenCandidate],
) -> HttpResult:
    payload = None
    outbound_headers = dict(headers)
    if body is not None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        outbound_headers.setdefault("Content-Type", "application/json")

    request = urllib.request.Request(url, data=payload, headers=outbound_headers, method=method)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
            text = raw.decode("utf-8", errors="replace")
            parsed = parse_json(text)
            return HttpResult(
                method=method,
                url=url,
                status=status,
                ok=200 <= status < 300,
                headers=safe_header_subset(response.headers),
                text=redacted_preview(parsed if parsed is not None else text, tokens),
                json_body=parsed,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        text = raw.decode("utf-8", errors="replace")
        parsed = parse_json(text)
        return HttpResult(
            method=method,
            url=url,
            status=int(exc.code),
            ok=False,
            headers=safe_header_subset(exc.headers),
            text=redacted_preview(parsed if parsed is not None else text, tokens),
            json_body=parsed,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            error=str(exc),
        )
    except Exception as exc:
        return HttpResult(
            method=method,
            url=url,
            status=None,
            ok=False,
            headers={},
            text="",
            json_body=None,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def read_opencode_auth_candidates() -> list[TokenCandidate]:
    paths: list[Path] = []
    explicit = os.environ.get("OPENCODE_AUTH_JSON", "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser())

    xdg_data = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg_data:
        paths.append(Path(xdg_data).expanduser() / "opencode" / "auth.json")
    paths.extend(
        [
            Path.home() / ".local" / "share" / "opencode" / "auth.json",
            Path.home() / "Library" / "Application Support" / "opencode" / "auth.json",
            Path.home() / ".config" / "opencode" / "auth.json",
        ]
    )

    candidates: list[TokenCandidate] = []
    seen_paths: set[Path] = set()
    for path in paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except Exception as exc:
            print(f"warn: could not read opencode auth file {path}: {exc}", file=sys.stderr)
            continue

        entry = payload.get("github-copilot") if isinstance(payload, dict) else None
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "oauth":
            continue

        token = str(entry.get("refresh") or entry.get("access") or "").strip()
        if not token:
            continue
        enterprise_url = str(entry.get("enterpriseUrl") or "").strip() or None
        candidates.append(
            TokenCandidate(
                token=token,
                source=f"opencode:{path}:github-copilot",
                enterprise_url=enterprise_url,
            )
        )
    return candidates


def token_candidates(*, allow_secret_store: bool) -> list[TokenCandidate]:
    candidates: list[TokenCandidate] = []
    for env_var in (*API_TOKEN_ENV_VARS, *OAUTH_TOKEN_ENV_VARS):
        token = os.environ.get(env_var, "").strip()
        if token:
            candidates.append(TokenCandidate(token=token, source=f"env:{env_var}"))

    candidates.extend(read_opencode_auth_candidates())

    if allow_secret_store:
        try:
            from headroom import copilot_auth

            resolution = copilot_auth.resolve_subscription_bearer_token_details()
            if resolution is not None:
                candidates.append(
                    TokenCandidate(
                        token=resolution.token,
                        source=f"headroom-secret-store:{resolution.source}",
                    )
                )
        except Exception as exc:
            print(f"warn: Headroom secret-store discovery failed: {exc}", file=sys.stderr)

    deduped: list[TokenCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.token in seen:
            continue
        seen.add(candidate.token)
        deduped.append(candidate)
    return deduped


def opencode_copilot_base(enterprise_url: str | None) -> str:
    if not enterprise_url:
        return DEFAULT_API_URL
    normalized = enterprise_url.strip().replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://copilot-api.{normalized}"


def auth_headers(candidate: TokenCandidate, *, style: str, opencode_user_agent: str) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {candidate.token}",
    }
    if style == "opencode":
        headers["User-Agent"] = opencode_user_agent
        headers["Openai-Intent"] = "conversation-edits"
        headers["x-initiator"] = "user"
    elif style == "copilot-cli":
        headers["User-Agent"] = DEFAULT_COPILOT_USER_AGENT
        headers["Editor-Version"] = DEFAULT_EDITOR_VERSION
    return headers


def endpoint_url(base_url: str, path: str) -> str:
    return f"{normalize_base_url(base_url)}{path}"


def model_summary(models_payload: Any, model: str) -> dict[str, Any] | None:
    if not isinstance(models_payload, dict):
        return None
    data = models_payload.get("data")
    if not isinstance(data, list):
        return None

    normalized = model.strip().lower()
    matches = [
        item
        for item in data
        if isinstance(item, dict) and str(item.get("id") or "").strip().lower() == normalized
    ]
    if not matches:
        contains = [
            str(item.get("id"))
            for item in data
            if isinstance(item, dict) and normalized in str(item.get("id") or "").lower()
        ][:10]
        return {
            "found": False,
            "total_models": len(data),
            "near_matches": contains,
        }

    item = matches[0]
    capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
    limits = capabilities.get("limits") if isinstance(capabilities.get("limits"), dict) else {}
    supports = capabilities.get("supports") if isinstance(capabilities.get("supports"), dict) else {}
    policy = item.get("policy") if isinstance(item.get("policy"), dict) else {}
    return {
        "found": True,
        "id": item.get("id"),
        "name": item.get("name"),
        "version": item.get("version"),
        "model_picker_enabled": item.get("model_picker_enabled"),
        "supported_endpoints": item.get("supported_endpoints"),
        "policy_state": policy.get("state"),
        "family": capabilities.get("family"),
        "limits": {
            "context": limits.get("max_context_window_tokens"),
            "prompt": limits.get("max_prompt_tokens"),
            "output": limits.get("max_output_tokens"),
        },
        "supports": {
            "streaming": supports.get("streaming"),
            "tool_calls": supports.get("tool_calls"),
            "structured_outputs": supports.get("structured_outputs"),
            "reasoning_effort": supports.get("reasoning_effort"),
            "adaptive_thinking": supports.get("adaptive_thinking"),
        },
    }


def model_catalog_summary(models_payload: Any, *, limit: int = 30) -> dict[str, Any] | None:
    if not isinstance(models_payload, dict):
        return None
    data = models_payload.get("data")
    if not isinstance(data, list):
        return None

    models: list[dict[str, Any]] = []
    for item in data[:limit]:
        if not isinstance(item, dict):
            continue
        policy = item.get("policy") if isinstance(item.get("policy"), dict) else {}
        models.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "model_picker_enabled": item.get("model_picker_enabled"),
                "supported_endpoints": item.get("supported_endpoints"),
                "policy_state": policy.get("state"),
            }
        )

    return {
        "total_models": len(data),
        "shown": len(models),
        "truncated": len(data) > limit,
        "models": models,
    }


def user_info_api_url(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, dict):
        return None
    api_url = endpoints.get("api")
    return api_url.strip() if isinstance(api_url, str) and api_url.strip() else None


def generation_body(path: str, model: str) -> dict[str, Any]:
    if "responses" in path:
        return {
            "model": model,
            "input": "Reply with exactly: HEADROOM_OK",
            "max_output_tokens": 16,
            "stream": False,
        }
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: HEADROOM_OK"}],
        "max_tokens": 16,
        "stream": False,
    }


def print_http_result(label: str, result: HttpResult) -> None:
    status = result.status if result.status is not None else "error"
    print(f"{label}: status={status} elapsed_ms={result.elapsed_ms} url={result.url}")
    if result.headers:
        print(f"  headers={json.dumps(result.headers, sort_keys=True)}")
    if result.error:
        print(f"  error={result.error}")
    if not result.ok and result.text:
        print(f"  error_preview={result.text}")


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4", help="Copilot model id to inspect.")
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help="Additional Copilot API base URL to probe. Can be passed multiple times.",
    )
    parser.add_argument(
        "--header-style",
        choices=("all", "minimal", "opencode", "copilot-cli"),
        default="all",
        help="Header style to try for /models and optional generation probes.",
    )
    parser.add_argument(
        "--opencode-user-agent",
        default=os.environ.get("OPENCODE_USER_AGENT", DEFAULT_OPENCODE_USER_AGENT),
        help="User-Agent used for opencode-style probes.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds per request.",
    )
    parser.add_argument(
        "--probe-generation",
        action="store_true",
        help="Also POST tiny generation requests. This may consume Copilot quota.",
    )
    parser.add_argument(
        "--allow-secret-store",
        action="store_true",
        help="Allow Headroom to query OS credential stores. May trigger Keychain/secret-store prompts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of human-readable output.",
    )
    args = parser.parse_args()

    candidates = token_candidates(allow_secret_store=args.allow_secret_store)
    if not candidates:
        print(
            "No Copilot token found in env or opencode auth.json. "
            "Set GITHUB_COPILOT_TOKEN or pass --allow-secret-store.",
            file=sys.stderr,
        )
        return 2

    header_styles = (
        ["minimal", "opencode", "copilot-cli"] if args.header_style == "all" else [args.header_style]
    )

    base_candidates = [
        *args.base_url,
        os.environ.get("GITHUB_COPILOT_API_URL", ""),
        opencode_copilot_base(os.environ.get("GITHUB_COPILOT_ENTERPRISE_URL", "").strip())
        if os.environ.get("GITHUB_COPILOT_ENTERPRISE_URL", "").strip()
        else "",
        opencode_copilot_base(os.environ.get("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "").strip())
        if os.environ.get("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "").strip()
        else "",
        DEFAULT_API_URL,
        DEFAULT_BUSINESS_API_URL,
        *[opencode_copilot_base(candidate.enterprise_url) for candidate in candidates],
    ]
    bases = dedupe_ordered(base_candidates)

    report: dict[str, Any] = {
        "model": args.model,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
        },
        "tokens": [
            {
                "source": candidate.source,
                "fingerprint": candidate.fingerprint,
                "enterprise_url": candidate.enterprise_url,
            }
            for candidate in candidates
        ],
        "probes": [],
    }

    if not args.json:
        print("Headroom Copilot subscription diagnostics")
        print(f"model={args.model}")
        print(f"generation_probes={'enabled' if args.probe_generation else 'skipped'}")
        print("tokens:")
        for idx, candidate in enumerate(candidates, start=1):
            suffix = f" enterprise_url={candidate.enterprise_url}" if candidate.enterprise_url else ""
            print(f"  [{idx}] source={candidate.source} fingerprint={candidate.fingerprint}{suffix}")
        print("base_urls:")
        for base in bases:
            print(f"  {base}")

    advertised_api_urls: list[str] = []
    for candidate in candidates:
        if not args.json:
            print(f"\n== token {candidate.fingerprint} ({candidate.source}) ==")

        user_info = http_request(
            "GET",
            DEFAULT_USER_INFO_URL,
            headers=auth_headers(
                candidate,
                style="minimal",
                opencode_user_agent=args.opencode_user_agent,
            ),
            timeout=args.timeout,
            tokens=candidates,
        )
        advertised = user_info_api_url(user_info.json_body)
        if advertised:
            advertised_api_urls.append(advertised)
        report["probes"].append(
            {
                "kind": "user_info",
                "token": candidate.fingerprint,
                "status": user_info.status,
                "ok": user_info.ok,
                "advertised_api_url": advertised,
                "headers": user_info.headers,
                "error_preview": None if user_info.ok else user_info.text,
            }
        )
        if not args.json:
            print_http_result("user-info", user_info)
            if advertised:
                print(f"  advertised_api_url={advertised}")

        for base_url in dedupe_ordered([*bases, *advertised_api_urls]):
            for style in header_styles:
                models_url = endpoint_url(base_url, "/models")
                models = http_request(
                    "GET",
                    models_url,
                    headers=auth_headers(
                        candidate,
                        style=style,
                        opencode_user_agent=args.opencode_user_agent,
                    ),
                    timeout=args.timeout,
                    tokens=candidates,
                )
                summary = model_summary(models.json_body, args.model)
                catalog = model_catalog_summary(models.json_body)
                report["probes"].append(
                    {
                        "kind": "models",
                        "token": candidate.fingerprint,
                        "base_url": base_url,
                        "header_style": style,
                        "status": models.status,
                        "ok": models.ok,
                        "model": summary,
                        "catalog": catalog,
                        "headers": models.headers,
                        "error_preview": None if models.ok else models.text,
                    }
                )
                if not args.json:
                    print_http_result(f"models header_style={style}", models)
                    if summary is not None:
                        print(f"  model_summary={json.dumps(summary, sort_keys=True)}")
                    if catalog is not None:
                        print(f"  model_catalog={json.dumps(catalog, sort_keys=True)}")

                if not args.probe_generation:
                    continue

                for path in ("/responses", "/v1/responses", "/chat/completions", "/v1/chat/completions"):
                    result = http_request(
                        "POST",
                        endpoint_url(base_url, path),
                        headers=auth_headers(
                            candidate,
                            style=style,
                            opencode_user_agent=args.opencode_user_agent,
                        ),
                        body=generation_body(path, args.model),
                        timeout=args.timeout,
                        tokens=candidates,
                    )
                    report["probes"].append(
                        {
                            "kind": "generation",
                            "token": candidate.fingerprint,
                            "base_url": base_url,
                            "header_style": style,
                            "path": path,
                            "status": result.status,
                            "ok": result.ok,
                            "headers": result.headers,
                            "error_preview": None if result.ok else result.text,
                        }
                    )
                    if not args.json:
                        print_http_result(f"generation {path} header_style={style}", result)

    if args.json:
        print(json.dumps(redact_json(report), indent=2, sort_keys=True))
    elif not args.probe_generation:
        print("\nGeneration probes were skipped. Add --probe-generation to POST tiny requests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
