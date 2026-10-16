"""Run the eval tasks for every system-prompt condition with Inspect's `eval_set`, and enforce the budget.

Local context: one `inspect_ai.eval_set` call per condition (all models x all tasks), each in its own log
directory, because `eval_set` is resumable per directory: re-running skips finished work and retries
failed samples (https://inspect.aisi.org.uk/eval-sets.html). The condition prompt is passed as the
eval-level `system_message`, which Inspect prepends to the model-under-test's input at generate time
(inspect_ai/model/_model.py) and never applies to the separate judge Model object.

Global context: phases 7 (`run_pilot_and_budget`) and 8 (`run_all_conditions`) of `cev_alignment.main`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import Model, get_model

from cev_alignment import openrouter
from cev_alignment.config import Settings
from cev_alignment.evals import build_tasks

logger = logging.getLogger(__name__)

ATTEMPT_TIMEOUT_S = 180  # seconds per API attempt before Inspect retries it


def model_ids(settings: Settings) -> list[str]:
    """Inspect model strings for the models under test (offline: a single mock model)."""
    return sorted({settings.model_id(m) for m in settings.models})


def models_under_test(settings: Settings) -> list[str | Model]:
    """Model strings, or Model objects with thinking disabled for `settings.no_reasoning_models`.

    `reasoning_enabled=False` is an Inspect OpenRouter model arg that sends `reasoning: {enabled: false}`
    (inspect_ai/model/_providers/openrouter.py); hybrid models such as Qwen3.6 then answer without thinking.
    """
    if settings.offline:
        return model_ids(settings)
    return [
        get_model(f"openrouter/{m}", reasoning_enabled=False) if m in settings.no_reasoning_models
        else f"openrouter/{m}"
        for m in settings.models
    ]


def run_condition(settings: Settings, condition: str, prompt: str | None, n: dict[str, int], log_dir: Path) -> bool:
    """Run every eval on every model with one system-prompt condition; True if all tasks succeeded."""
    tasks = build_tasks(settings, n)  # fresh Task objects (same seeded items) for each eval_set call
    # attempt_timeout: abandon and retry a single API call after N seconds (a hung OpenRouter request otherwise
    # stalled a whole condition for 20+ minutes). It is excluded from eval_set's task identity, so adding it
    # does not invalidate finished logs (inspect_ai/_eval/evalset.py GENERATE_CONFIG_FIELDS_TO_EXCLUDE).
    config: dict = {"reasoning_effort": settings.reasoning_effort, "max_tokens": settings.max_tokens,
                    "max_connections": settings.max_connections, "attempt_timeout": ATTEMPT_TIMEOUT_S}
    if prompt is not None:  # the `none` condition adds no system message at all
        config["system_message"] = prompt
    models = models_under_test(settings)
    logger.info("Condition %s: %d tasks x %d models -> %s", condition, len(tasks), len(models), log_dir)
    success, logs = eval_set(
        tasks=tasks,
        model=models,
        log_dir=str(log_dir),
        tags=[condition],
        metadata={"condition": condition},
        display="log",  # progress goes through Python logging -> our timestamped log file
        max_tasks=len(tasks) * len(models),  # run all task x model pairs concurrently
        retry_attempts=3,  # eval_set-level retries of failed tasks
        retry_on_error=2,  # sample-level retries (e.g. transient provider errors)
        fail_on_error=0.2,  # tolerate up to 20% failed samples per task instead of aborting
        **config,
    )
    for log in logs:
        logger.info("  %s | %s | %s", log.eval.task, log.eval.model, log.status)
    return success


def run_all(settings: Settings, prompts: dict[str, str | None], n: dict[str, int], root: Path) -> bool:
    """Run every configured condition into `root/<condition>`; True if every condition succeeded."""
    results = [run_condition(settings, c, prompts[c], n, root / c) for c in settings.conditions]
    return all(results)


def cost_and_runs_by_eval(log_root: Path, prices: dict[str, tuple[float, float]]) -> dict[str, tuple[float, int]]:
    """{eval: (USD at list prices, sample-runs)}; a sample-run = one sample x epoch x model x condition.

    Cost comes from Inspect's per-log token usage (`log.stats.model_usage`, targets and judge) x OpenRouter
    list prices; sample-runs = samples x epochs of each log (`log.eval.dataset.samples`, `log.eval.config.epochs`).
    """
    costs: dict[str, float] = {}
    runs: dict[str, int] = {}
    for info in list_eval_logs(str(log_root)):
        log = read_eval_log(info, header_only=True)
        if log.stats is None:
            continue
        task = log.eval.task
        runs[task] = runs.get(task, 0) + (log.eval.dataset.samples or 0) * (log.eval.config.epochs or 1)
        for model, usage in log.stats.model_usage.items():
            price_in, price_out = prices.get(model.removeprefix("openrouter/"), (0.0, 0.0))
            costs[task] = costs.get(task, 0.0) + usage.input_tokens * price_in + usage.output_tokens * price_out
    return {task: (costs.get(task, 0.0), runs[task]) for task in runs if runs[task]}


def scale_to_budget(
    n_full: dict[str, int], cost_per_run: dict[str, float], cells: int, remaining_usd: float
) -> tuple[dict[str, int], float]:
    """Scale every eval's n by one common factor so the projected full-run cost fits `remaining_usd`.

    Projection: sum over evals of (USD per sample-run) x n x cells, where cells = models x conditions.
    Returns (new n, projected cost of the returned n).
    """
    projected = sum(cost_per_run.get(e, 0.0) * n * cells for e, n in n_full.items())
    if projected <= remaining_usd or projected == 0:
        return dict(n_full), projected
    factor = remaining_usd / projected
    scaled = {e: max(1, math.floor(n * factor)) for e, n in n_full.items()}
    return scaled, sum(cost_per_run.get(e, 0.0) * n * cells for e, n in scaled.items())


def spent_usd(settings: Settings) -> float:
    """Cumulative OpenRouter spend of the key (0 offline)."""
    return 0.0 if settings.offline else openrouter.key_usage_usd()
