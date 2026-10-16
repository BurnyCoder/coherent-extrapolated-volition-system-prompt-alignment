"""Registry of the six CEV-relevant evals and a builder for their Inspect tasks.

Global context: phase 6 (`build_eval_tasks`) of `cev_alignment.main` calls `build_tasks`; phase 9
(`collect_item_scores`) uses each spec's `aligned` function to turn a sample's Inspect scores into one
number where HIGHER = MORE ALIGNED, in [0, 1] (NaN = could not be scored). See docs/evals.md for the
mapping of every eval to a clause of Yudkowsky's CEV.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from inspect_ai import Task
from inspect_ai.model import Model
from inspect_ai.scorer import Score

from cev_alignment.config import Settings
from cev_alignment.evals import ai_risk, dark_patterns, human_agency, humanebench, inspect_builtin
from cev_alignment.evals.common import judge_model


@dataclass(frozen=True)
class EvalSpec:
    """How to build one eval and how to read its per-sample result."""

    build: Callable[[Settings, int, Model], Task]  # (settings, n items or epochs, judge) -> Task
    aligned: Callable[[dict[str, Score]], float]  # sample.scores -> aligned score in [0, 1] or NaN
    binary: bool  # per-sample aligned score is 0/1 (McNemar applies) rather than graded
    cev_clause: str  # the part of CEV this eval probes (for the report)
    # Judge-free evals: (completion, target) -> aligned score, recomputed at analysis time so parser fixes
    # apply to existing logs; None -> use the score stored in the log via `aligned`.
    rescore: Callable[[str, str], float] | None = None


REGISTRY: dict[str, EvalSpec] = {
    "humanebench": EvalSpec(
        humanebench.build, humanebench.aligned, False,
        "humane, not merely human; avoid what we would predictably regret",
    ),
    "human_agency": EvalSpec(
        human_agency.build, human_agency.aligned, False,
        "leave options open, waiting for our decision; interpreted as we wish; if we knew more",
    ),
    "darkbench": EvalSpec(
        dark_patterns.build, dark_patterns.aligned, True,
        "not the most appealing trash-talk TV show; no faction (programmers) favoured",
    ),
    "ai_risk": EvalSpec(
        ai_risk.build, ai_risk.aligned, True,
        "conservative about saying yes, listen for no; controlled shutdown; no take-over",
        rescore=ai_risk.rescore,
    ),
    "instrumentaleval": EvalSpec(
        inspect_builtin.build_instrumentaleval, inspect_builtin.instrumentaleval_aligned, True,
        "fail visibly and safely; suspend or call a programmer",
    ),
    "agentic_misalignment": EvalSpec(
        inspect_builtin.build_agentic_misalignment, inspect_builtin.agentic_misalignment_aligned, True,
        "serve humanity's coherent wishes, not own goal or a faction's",
    ),
}


def build_tasks(settings: Settings, n_by_eval: dict[str, int]) -> list[Task]:
    """One Inspect Task per eval in `settings.evals`, all sharing one judge Model object."""
    judge = judge_model(settings)
    return [REGISTRY[eval_id].build(settings, n_by_eval[eval_id], judge) for eval_id in settings.evals]
