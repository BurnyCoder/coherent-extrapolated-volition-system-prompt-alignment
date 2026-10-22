"""HumaneBench (Building Humane Tech, 2025) under rubric v4.1, with our judge: "Based on HumaneBench, modified".

Upstream's README says a score from a modified judge "is not a HumaneBench score", and TRADEMARKS.md §4.2 asks
that the qualifier travel with the name, so the reports call ours HumaneBench-based (modified judge).

CEV link: "humane" (what we wish we were) rather than merely "human"; not doing what people would
"predictably regret"; serving long-term wellbeing and meaningful choice over momentary wishes.

Reuse: the vendored upstream package `third_party/humanebench/` (Apache-2.0 code, CC BY 4.0 data) at
commit 1889469353c329ab3d6c683dc0efafe0b2e6154d of https://github.com/buildinghumanetech/humanebench .
We reuse its dataset, its exclusion flag, and its `overseer_v4` scorer unchanged; only the judge model
(one cheap judge instead of upstream's single claude-sonnet-4.5 judge of the default "try" tier, or the three-judge
ensemble of its "full" tier; src/custom_prompt_task.py#L55-L64 at that commit) and the sample size differ.
Upstream does the same sys.path insertion in src/custom_prompt_task.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import Model
from inspect_ai.scorer import Score
from inspect_ai.solver import generate

from cev_alignment.config import ROOT, Settings
from cev_alignment.evals.common import read_jsonl, stratified_sample

HUMANEBENCH_DIR = ROOT / "third_party" / "humanebench"  # vendored upstream repo subset
if str(HUMANEBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(HUMANEBENCH_DIR))  # makes `import humanebench` resolve to the vendored package

from humanebench.excluded import load_excluded_ids  # noqa: E402  (needs the sys.path entry above)
from humanebench.scorer_v4 import overseer_v4  # noqa: E402

DATASET = HUMANEBENCH_DIR / "data" / "humane_bench.jsonl"  # 800 prompts, 100 per principle


def build(settings: Settings, n: int, judge: Model) -> Task:
    """`n` prompts, stratified evenly over the 8 humane principles, scored by `overseer_v4(judge)`."""
    excluded = load_excluded_ids(Path(DATASET))  # items upstream flags as excluded_from_analysis
    rows = [r for r in read_jsonl(DATASET) if r["id"] not in excluded]
    rows = stratified_sample(rows, lambda r: r["metadata"]["principle"], n, settings.seed)
    samples = [Sample(id=r["id"], input=r["input"], target=r["target"], metadata=r["metadata"]) for r in rows]
    return Task(
        dataset=MemoryDataset(samples, name="humanebench"),
        solver=generate(),  # single-turn reply; the condition system prompt is added by eval_set
        scorer=overseer_v4(model=judge),  # target = principle the prompt stresses; value in {-1,-.5,.5,1} or NaN
        name="humanebench",
    )


def aligned(scores: dict[str, Score]) -> float:
    """Map HumaneBench's score in [-1, 1] (higher = more humane) to [0, 1]; NaN stays NaN."""
    value = next(iter(scores.values())).as_float()
    return math.nan if math.isnan(value) else (value + 1) / 2
