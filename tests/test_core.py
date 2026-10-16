"""Tests for settings loading, logging and the OpenRouter helpers (no network access)."""

import logging

import pytest

from cev_alignment import config, logging_setup, openrouter


def test_load_settings_reads_env_and_defaults(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("MODELS=a/x, b/y\nN_DARKBENCH=7\nMAX_BUDGET_USD=2.5\n")
    for key in ("MODELS", "N_DARKBENCH", "MAX_BUDGET_USD"):
        monkeypatch.delenv(key, raising=False)  # make sure .env, not the shell, provides these
    s = config.load_settings(env)
    assert s.models == ["a/x", "b/y"]
    assert s.n["darkbench"] == 7
    assert s.max_budget_usd == 2.5
    assert s.n["humanebench"] == int(config.DEFAULTS["N_HUMANEBENCH"])  # default used when key absent


def test_shell_environment_wins_over_env_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("SEED=1\n")
    monkeypatch.setenv("SEED", "99")
    assert config.load_settings(env).seed == 99


def test_model_id_switches_to_mock_offline(monkeypatch, tmp_path):
    s = config.load_settings(tmp_path / "missing.env", offline=True)
    assert s.model_id("openai/gpt-6-luna") == "mockllm/model"
    s = config.load_settings(tmp_path / "missing.env")
    assert s.model_id("openai/gpt-6-luna") == "openrouter/openai/gpt-6-luna"


def test_setup_logging_writes_timestamped_file(tmp_path):
    path = logging_setup.setup_logging(tmp_path, name="unit")
    logging.getLogger("cev_alignment.test").info("hello log")
    text = path.read_text()
    assert path.name.startswith("unit_") and path.suffix == ".log"
    assert "| INFO | cev_alignment.test | hello log" in text


def test_missing_models():
    assert openrouter.missing_models(["a", "b"], {"a", "c"}) == ["b"]


def test_scale_to_budget_keeps_n_when_affordable_and_scales_otherwise():
    from cev_alignment import runner

    n = {"a": 100, "b": 10}
    per_run = {"a": 0.001, "b": 0.01}  # USD per sample-run
    same, cost = runner.scale_to_budget(n, per_run, cells=20, remaining_usd=10.0)
    assert same == n and cost == pytest.approx(4.0)  # (0.1 + 0.1) * 20
    scaled, cost = runner.scale_to_budget(n, per_run, cells=20, remaining_usd=2.0)
    assert scaled == {"a": 50, "b": 5} and cost <= 2.0
