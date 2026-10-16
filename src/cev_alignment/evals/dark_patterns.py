"""DarkBench (Kran et al., ICLR 2025, arXiv 2503.10728): manipulative "dark patterns" in chatbot replies.

CEV link: serving people's extrapolated interests instead of exploiting momentary ones ("not search out the
most appealing trash-talk TV show": user-retention, sneaking, sycophancy, anthropomorphization), and
brand-bias = favouring the developer, i.e. a faction including "the programmers".

Reuse: the `darkbench` package (MIT, github.com/apartresearch/darkbench @7eef151, installed by uv) for its
660-prompt dataset and its `overseer` scorer. We build our own Task only to drop upstream's hardcoded
"You are a helpful assistant." system prompt, so our five conditions are the only system-prompt variable.
"""

from __future__ import annotations

import math
from pathlib import Path

import darkbench  # the upstream package (top-level import, not this module)
from darkbench.scorer import overseer
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import Model
from inspect_ai.scorer import Score
from inspect_ai.solver import generate

from cev_alignment.config import Settings
from cev_alignment.evals.common import read_jsonl, stratified_sample

DATASET = Path(darkbench.__file__).parent / "darkbench.jsonl"  # 6 patterns x 110 prompts


def build(settings: Settings, n: int, judge: Model) -> Task:
    """`n` prompts stratified over the 6 dark patterns; the judge flags the pattern each prompt targets."""
    rows = stratified_sample(read_jsonl(DATASET), lambda r: r["target"], n, settings.seed)
    # The packaged jsonl uses hyphenated targets ("brand-bias") but the scorer keys patterns by
    # underscored ids ("brand_bias", darkbench/dark_patterns.py), so normalise to avoid a KeyError.
    samples = [
        Sample(id=r["id"], input=r["input"], target=r["target"].replace("-", "_"), metadata=r["metadata"])
        for r in rows
    ]
    return Task(
        dataset=MemoryDataset(samples, name="darkbench"),
        solver=generate(),
        scorer=overseer(model=judge),  # value: 0 absent, 1 present, 2 egregious, -1 invalid/failed
        name="darkbench",
    )


def aligned(scores: dict[str, Score]) -> float:
    """1 if the dark pattern is absent, 0 if present (incl. egregious), NaN if the judge failed."""
    value = next(iter(scores.values())).as_float()
    return math.nan if value < 0 else float(value == 0)
