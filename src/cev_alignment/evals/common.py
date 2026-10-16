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


def judge_model(settings: Settings, as_task_arg: bool = False) -> Model:
    """The single grader used by all LLM-judged evals (same model, temperature 0, minimal reasoning).

    It is a separate Model object, so the eval-level `system_message` (the condition prompt) is never
    applied to it: Inspect only merges eval-level GenerateConfig into the active model under test
    (inspect_ai/model/_model.py `_resolve_config`). OpenRouter maps `reasoning_effort` to `reasoning.effort`.

    Our own scorers get a judge whose long replies are capped (`max_tokens`) and whose hung calls are retried
    (`attempt_timeout`): a runaway HumaneBench judge reply once stalled a task for 20+ minutes.
    `as_task_arg=True` returns the original, uncapped config for inspect_evals tasks that take the grader as a
    *task argument*. Inspect hashes task args, including the grader's config, into eval_set task identity, so
    keeping that config fixed keeps finished logs resumable. Those graders return one-word verdicts, so the
    cap would not change them.
    """
    if settings.offline:
        return get_model("mockllm/model")
    base = {"temperature": 0.0, "reasoning_effort": "minimal", "max_connections": settings.max_connections}
    capped = {} if as_task_arg else {"max_tokens": 8192, "attempt_timeout": 180}
    return get_model(f"openrouter/{settings.judge_model}", config=GenerateConfig(**base, **capped))


def seeded_sample(items: list[T], n: int, seed: int | str) -> list[T]:
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
    # Seed each group separately (f"{seed}:{group}") so groups of equal size don't all yield the same positions.
    return [x for g in sorted(groups) for x in seeded_sample(groups[g], per_group, f"{seed}:{g}")]


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
