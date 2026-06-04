"""Tests for `headroom wrap opencode` command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.copilot_auth import DEFAULT_API_URL, CopilotSubscriptionTokenResolution


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _subscription_token_resolution(
    token: str = "gho-existing",
    *,
    api_url: str = DEFAULT_API_URL,
    source: str = "macos-keychain:copilot-cli",
    confidence: str = "high",
) -> CopilotSubscriptionTokenResolution:
    return CopilotSubscriptionTokenResolution(
        token=token,
        source=source,
        confidence=confidence,
        api_url=api_url,
        token_fingerprint="sha256:0123456789ab",
    )


def _model_catalog(*model_ids: str) -> list[dict[str, object]]:
    return [{"id": model_id} for model_id in model_ids]


def test_wrap_opencode_writes_temp_config_and_launches(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}
    config_snapshot: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)
        env = kwargs["env"]
        assert isinstance(env, dict)
        config_dir = Path(env["OPENCODE_CONFIG_DIR"])
        assert config_dir.exists()
        config_snapshot.update(json.loads((config_dir / "opencode.json").read_text()))

    with (
        patch("headroom.cli.wrap._resolve_opencode_binary", return_value="/usr/bin/opencode"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "opencode",
                "--no-rtk",
                "--port",
                "9000",
                "--model",
                "gpt-4o",
                "--",
                "run",
                "explain this repo",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["binary"] == "/usr/bin/opencode"
    assert captured["args"] == ("run", "explain this repo")
    assert captured["tool_label"] == "OPENCODE"
    assert captured["agent_type"] == "opencode"

    env = captured["env"]
    assert isinstance(env, dict)
    assert "OPENCODE_CONFIG_DIR" in env

    provider = config_snapshot["provider"]
    assert isinstance(provider, dict)
    headroom = provider["headroom"]
    assert isinstance(headroom, dict)
    assert config_snapshot["model"] == "headroom/gpt-4o"
    assert config_snapshot["small_model"] == "headroom/gpt-4o"
    assert headroom["npm"] == "@ai-sdk/openai-compatible"
    assert headroom["api"] == "http://127.0.0.1:9000/v1"
    options = headroom["options"]
    assert isinstance(options, dict)
    assert options["baseURL"] == "http://127.0.0.1:9000/v1"
    assert options["apiKey"] == "headroom"
    assert captured["openai_api_url"] is None
    assert captured["copilot_api_token"] is None


def test_wrap_opencode_responses_wire_api_uses_openai_provider(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_snapshot: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        env = kwargs["env"]
        assert isinstance(env, dict)
        config_dir = Path(env["OPENCODE_CONFIG_DIR"])
        config_snapshot.update(json.loads((config_dir / "opencode.json").read_text()))

    with (
        patch("headroom.cli.wrap._resolve_opencode_binary", return_value="/usr/bin/opencode"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "opencode",
                "--no-rtk",
                "--model",
                "gpt-5.4",
                "--wire-api",
                "responses",
            ],
        )

    assert result.exit_code == 0, result.output
    provider = config_snapshot["provider"]
    assert isinstance(provider, dict)
    headroom = provider["headroom"]
    assert isinstance(headroom, dict)
    assert headroom["npm"] == "@ai-sdk/openai"
    assert config_snapshot["model"] == "headroom/gpt-5.4"


def test_wrap_opencode_normalizes_passthrough_model_arg(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}
    config_snapshot: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)
        env = kwargs["env"]
        assert isinstance(env, dict)
        config_dir = Path(env["OPENCODE_CONFIG_DIR"])
        config_snapshot.update(json.loads((config_dir / "opencode.json").read_text()))

    with (
        patch("headroom.cli.wrap._resolve_opencode_binary", return_value="/usr/bin/opencode"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "opencode",
                "--no-rtk",
                "--",
                "run",
                "--model",
                "gpt-5.4",
                "hello",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["args"] == ("run", "--model", "headroom/gpt-5.4", "hello")
    assert config_snapshot["model"] == "headroom/gpt-5.4"
    provider = config_snapshot["provider"]
    assert isinstance(provider, dict)
    headroom = provider["headroom"]
    assert isinstance(headroom, dict)
    assert headroom["npm"] == "@ai-sdk/openai"
    display = captured["env_vars_display"]
    assert isinstance(display, list)
    assert "OPENCODE_MODEL_ARG_NORMALIZED=headroom/gpt-5.4" in display


def test_wrap_opencode_subscription_pins_validated_token_for_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for var in ("GITHUB_COPILOT_API_TOKEN", "GITHUB_COPILOT_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    business_api = "https://api.business.githubcopilot.com"
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap._resolve_opencode_binary", return_value="/usr/bin/opencode"),
        patch(
            "headroom.cli.wrap.resolve_subscription_bearer_token_details",
            return_value=_subscription_token_resolution("gho-validated", api_url=business_api),
        ),
        patch(
            "headroom.cli.wrap.fetch_copilot_model_catalog",
            return_value=_model_catalog("gpt-4o"),
        ),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "opencode",
                "--subscription",
                "--auth-source",
                "headroom",
                "--debug-copilot",
                "--no-rtk",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == business_api
    assert captured["copilot_api_token"] == "gho-validated"
    assert captured["copilot_debug"] is True
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_TARGET_API_URL"] == business_api
    assert env["GITHUB_COPILOT_API_URL"] == business_api
    assert env["HEADROOM_COPILOT_DEBUG"] == "1"
    assert "GITHUB_COPILOT_API_TOKEN" not in env
    assert os.environ.get("GITHUB_COPILOT_API_TOKEN") is None
    assert "gho-validated" not in result.output


def test_wrap_opencode_subscription_can_use_opencode_auth_file(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "github-copilot": {
                    "type": "oauth",
                    "refresh": "opencode-copilot-token",
                    "access": "ignored-access-token",
                    "expires": 0,
                    "enterpriseUrl": "ghe.example.com",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HEADROOM_OPENCODE_AUTH_FILE", str(auth_file))
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap._resolve_opencode_binary", return_value="/usr/bin/opencode"),
        patch(
            "headroom.cli.wrap.fetch_copilot_model_catalog",
            return_value=_model_catalog("gpt-4o"),
        ),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "opencode", "--subscription", "--no-rtk"],
        )

    assert result.exit_code == 0, result.output
    assert captured["copilot_api_token"] == "opencode-copilot-token"
    assert captured["openai_api_url"] == "https://copilot-api.ghe.example.com"
    assert "opencode-copilot-token" not in result.output


def test_wrap_opencode_missing_binary_errors_clearly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "opencode", "--no-rtk"])

    assert result.exit_code == 1
    assert "'opencode' not found in PATH or ~/.opencode/bin" in result.output
