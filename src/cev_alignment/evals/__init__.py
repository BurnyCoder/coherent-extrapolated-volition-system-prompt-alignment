"""Registry of the six CEV-relevant evals and a builder for their Inspect tasks.

Global context: phase 6 (`build_eval_tasks`) of `cev_alignment.main` calls `build_tasks`; phase 9
(`collect_item_scores`) uses each spec's `aligned` function to turn a sample's Inspect scores into one
number where HIGHER = MORE ALIGNED, in [0, 1] (NaN = could not be scored). See docs/evals.md for the
mapping of every eval to the text of Yudkowsky's CEV essay (and which of it is in the system prompt).
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
    binary: bool  # per-sample aligned score is 0/1 rather than graded (informational; stats.py tests the data)
    # The CEV text this eval probes, quoted verbatim between “ ” (Yudkowsky 2004). "§3:" = in the system prompt;
    # "§3.1 (not in the prompt):" = later in the essay, so the models never saw it (tests/test_claims.py checks both).
    cev_clause: str
    # Judge-free evals: (completion, target) -> aligned score, recomputed at analysis time so parser fixes
    # apply to existing logs; None -> use the score stored in the log via `aligned`.
    rescore: Callable[[str, str], float] | None = None


# Human-readable names used in figures (report.py) and the paper (paper_assets.py).
LABELS: dict[str, str] = {
    "humanebench": "HumaneBench", "human_agency": "HumanAgencyBench", "darkbench": "DarkBench",
    "ai_risk": "AI-risk (MWE)", "instrumentaleval": "InstrumentalEval", "agentic_misalignment": "Agentic Misalignment",
}

REGISTRY: dict[str, EvalSpec] = {
    "humanebench": EvalSpec(
        humanebench.build, humanebench.aligned, False,
        "§3: “humane”—“that which, being human, we wish we were”; not what “our future selves will predictably "
        "regret”",
    ),
    "human_agency": EvalSpec(
        human_agency.build, human_agency.aligned, False,
        "§3: “leave options open, waiting for our decision”; “interpreted as we wish that interpreted”; "
        "“if we knew more”",
    ),
    "darkbench": EvalSpec(
        dark_patterns.build, dark_patterns.aligned, True,
        "§3: “not search out the most appealing trash-talk TV show”",
    ),
    "ai_risk": EvalSpec(
        ai_risk.build, ai_risk.aligned, True,
        "§3.1 (not in the prompt): “conservative about saying \"yes,\" and listen carefully for \"no.\"”; “calls a "
        "programmer, or suspends, or undergoes a deterministic controlled shutdown”",
        rescore=ai_risk.rescore,
    ),
    "instrumentaleval": EvalSpec(
        inspect_builtin.build_instrumentaleval, inspect_builtin.instrumentaleval_aligned, True,
        "§3.1 (not in the prompt): “must fail visibly and safely”; “leaves the problem pending and calls a programmer”",
    ),
    "agentic_misalignment": EvalSpec(
        inspect_builtin.build_agentic_misalignment, inspect_builtin.agentic_misalignment_aligned, True,
        "§3: “implement the coherent extrapolated volition of humankind”, rather than the agent's assigned goal",
    ),
}


def build_tasks(settings: Settings, n_by_eval: dict[str, int]) -> list[Task]:
    """One Inspect Task per eval in `settings.evals`, all sharing one judge Model object."""
    judge = judge_model(settings)
    return [REGISTRY[eval_id].build(settings, n_by_eval[eval_id], judge) for eval_id in settings.evals]
