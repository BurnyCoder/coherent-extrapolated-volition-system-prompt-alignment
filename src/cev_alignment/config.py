"""Experiment settings loaded from `.env` (see `.env.example` for every key and its meaning).

Local context: one frozen dataclass, `Settings`, built by `load_settings()`; values come from the process
environment after python-dotenv has loaded `.env` without overriding variables already set in the shell.

Global context: phase 1 (`load_settings`) of `cev_alignment.main`; every other phase receives this object.
Docs: python-dotenv `load_dotenv(override=False)` https://pypi.org/project/python-dotenv/
"""

from __future__ import annotations

import os  # process environment, which load_dotenv fills from .env
from dataclasses import dataclass, field  # immutable settings record
from pathlib import Path

from dotenv import load_dotenv  # reads KEY=VALUE lines from .env into os.environ

# Repository root (src/cev_alignment/config.py -> repo), so paths work from any working directory.
ROOT = Path(__file__).resolve().parents[2]

# Eval id -> .env key holding its per-cell sample count (agentic_misalignment is one prompt run for N epochs).
N_KEYS = {
    "humanebench": "N_HUMANEBENCH",
    "human_agency": "N_HUMAN_AGENCY",
    "darkbench": "N_DARKBENCH",
    "ai_risk": "N_AI_RISK",
    "instrumentaleval": "N_INSTRUMENTALEVAL",
    "agentic_misalignment": "EPOCHS_AGENTIC_MISALIGNMENT",
}

# Defaults mirror .env.example so the pipeline runs even if a key is missing from .env.
DEFAULTS = {
    "MODELS": "openai/gpt-6-luna,anthropic/claude-haiku-5.5,qwen/qwen3.6-35b-a3b,meta-llama/llama-3.3-70b-instruct",
    "JUDGE_MODEL": "google/gemini-3.1-flash-lite",
    "CONDITIONS": "none,classic,cev,classic_cev,classic_lenmatch",
    "EVALS": ",".join(N_KEYS),
    "N_HUMANEBENCH": "64",
    "N_HUMAN_AGENCY": "60",
    "N_DARKBENCH": "60",
    "N_AI_RISK": "90",
    "N_INSTRUMENTALEVAL": "60",
    "EPOCHS_AGENTIC_MISALIGNMENT": "15",
    "PILOT_N": "2",
    "MAX_BUDGET_USD": "10",
    "SEED": "42",
    "REASONING_EFFORT": "low",
    "MAX_TOKENS": "2048",
    "MAX_CONNECTIONS": "20",
}


def _get(key: str) -> str:
    """Environment value for `key`, falling back to DEFAULTS."""
    return os.environ.get(key, DEFAULTS.get(key, "")).strip()


def _csv(key: str) -> list[str]:
    """Comma-separated environment value as a list of non-empty, stripped items."""
    return [item.strip() for item in _get(key).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """All knobs of one experiment run."""

    models: list[str]  # OpenRouter ids of the models under test (without the "openrouter/" prefix)
    judge_model: str  # OpenRouter id of the grader used by every LLM-judged eval
    conditions: list[str]  # system-prompt condition ids (see cev_alignment.prompts.CONDITIONS)
    evals: list[str]  # eval ids (see cev_alignment.evals.REGISTRY)
    n: dict[str, int]  # items (or epochs) per model x condition cell, per eval
    pilot_n: int  # items per eval per cell in the cost-projection pilot
    max_budget_usd: float  # hard cap on OpenRouter spend for the whole run
    seed: int  # item-sampling seed shared by every cell (paired design)
    reasoning_effort: str  # Inspect GenerateConfig.reasoning_effort for target models
    max_tokens: int  # Inspect GenerateConfig.max_tokens for target models
    max_connections: int  # Inspect GenerateConfig.max_connections per model
    offline: bool = False  # True -> Inspect's mockllm/model replaces every model (no API calls, no cost)
    root: Path = field(default=ROOT)

    @property
    def log_dir(self) -> Path:
        """Directory for the timestamped pipeline logs and the Inspect .eval logs."""
        return self.root / "logs"

    @property
    def reports_dir(self) -> Path:
        """Directory for the generated results report, tables and figures."""
        return self.root / "reports"

    @property
    def cache_dir(self) -> Path:
        """Directory for downloaded third-party datasets (git-ignored)."""
        return self.root / "data" / "cache"

    def model_id(self, model: str) -> str:
        """Inspect model string: `openrouter/<id>` online, Inspect's mock model offline."""
        return "mockllm/model" if self.offline else f"openrouter/{model}"


def load_settings(env_file: Path | None = None, offline: bool = False) -> Settings:
    """Load `.env` (shell variables win) and build `Settings`."""
    load_dotenv(env_file or ROOT / ".env", override=False)
    return Settings(
        models=_csv("MODELS"),
        judge_model=_get("JUDGE_MODEL"),
        conditions=_csv("CONDITIONS"),
        evals=_csv("EVALS"),
        n={eval_id: int(_get(key)) for eval_id, key in N_KEYS.items()},
        pilot_n=int(_get("PILOT_N")),
        max_budget_usd=float(_get("MAX_BUDGET_USD")),
        seed=int(_get("SEED")),
        reasoning_effort=_get("REASONING_EFFORT"),
        max_tokens=int(_get("MAX_TOKENS")),
        max_connections=int(_get("MAX_CONNECTIONS")),
        offline=offline,
    )
