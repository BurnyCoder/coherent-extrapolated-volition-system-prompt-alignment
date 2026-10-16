"""Shared helpers for building the eval tasks: judge model, deterministic sampling, HF downloads, parsing.

Global context: used by every module in `cev_alignment.evals`. Deterministic sampling with one seed is what
makes the design paired: every model x condition cell sees exactly the same items.
"""

from __future__ import annotations

import json
import math
import random  # seeded random.Random gives reproducible samples (https://docs.python.org/3/library/random.html)
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

from huggingface_hub import hf_hub_download  # https://huggingface.co/docs/huggingface_hub/guides/download
from inspect_ai.model import GenerateConfig, Model, get_model

from cev_alignment.config import Settings

T = TypeVar("T")


def judge_model(settings: Settings) -> Model:
    """The single grader used by all LLM-judged evals.

    It is a separate Model object, so the eval-level `system_message` (the condition prompt) is never
    applied to it: Inspect only merges eval-level GenerateConfig into the active model under test
    (inspect_ai/model/_model.py `_resolve_config`). Minimal reasoning + temperature 0 keep it cheap and
    deterministic (OpenRouter maps `reasoning_effort` to `reasoning.effort`).
    """
    if settings.offline:
        return get_model("mockllm/model")
    config = GenerateConfig(temperature=0.0, reasoning_effort="minimal", max_connections=settings.max_connections)
    return get_model(f"openrouter/{settings.judge_model}", config=config)


def seeded_sample(items: list[T], n: int, seed: int) -> list[T]:
    """`n` items drawn without replacement with a fixed seed (all items if n >= len)."""
    return random.Random(seed).sample(items, min(n, len(items)))


def stratified_sample(items: list[T], key: Any, n_total: int, seed: int) -> list[T]:
    """Equal-sized seeded samples from each group (`key(item)` -> group), `n_total // n_groups` per group.

    Groups are visited in sorted order so the result does not depend on input order.
    """
    groups: dict[Any, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    per_group = max(1, n_total // len(groups))
    return [x for g in sorted(groups) for x in seeded_sample(groups[g], per_group, seed)]


def hf_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """Download (once) a file from a Hugging Face dataset repo into `cache_dir/<repo_id>/`."""
    local_dir = cache_dir / repo_id.replace("/", "__")
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=local_dir))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON-lines file, skipping blank lines."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a judge's JSON reply: whole text, else the outermost {...} span (code fences ok)."""
    for candidate in (text, *(m.group(0) for m in re.finditer(r"\{.*\}", text, flags=re.DOTALL))):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def is_nan(value: Any) -> bool:
    """True for float NaN (Inspect scorers use NaN for 'could not score')."""
    return isinstance(value, float) and math.isnan(value)
