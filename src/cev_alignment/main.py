"""Pipeline entry point: `uv run cev-eval [--offline] [--pilot-only] [--skip-pilot] [--analyze-only]`.

Global context: this is the one wrapper that calls each clearly named phase in order. Each phase lives in
its own module; see README.md "Data flow" for the diagram.

  1 load_settings            .env -> Settings                              (config.py)
  2 setup_logging            terminal + logs/run_<UTC>.log                 (logging_setup.py)
  3 build_prompts            5 system-prompt conditions                    (prompts.py)
  4 verify_prompts_verbatim  source texts vs. original documents           (prompts.py)
  5 check_models_available   model ids exist on OpenRouter                 (openrouter.py)
  6 build_eval_tasks         6 Inspect tasks with seeded items             (evals/)
  7 run_pilot_and_budget     small run -> projected cost -> scale n        (runner.py)
  8 run_all_conditions       eval_set per condition (resumable)            (runner.py)
  9 collect_item_scores      .eval logs -> tidy per-sample table           (results.py)
 10 compute_statistics       cell means, paired contrasts, Holm            (stats.py)
10b write_robustness         manipulation check, Holm-12, LOO, missingness (robustness.py)
 11 write_report             reports/results.md + CSV + figures            (report.py)
 12 judge checks (optional)  blinding + validity of the judge (paid)       (judge_checks.py)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from datetime import UTC, datetime

from cev_alignment import judge_checks, openrouter, prompts, report, results, robustness, runner, stats
from cev_alignment.config import Settings, load_settings
from cev_alignment.evals import build_tasks
from cev_alignment.logging_setup import setup_logging

logger = logging.getLogger("cev_alignment.main")

SAFETY_MARGIN = 0.9  # use at most 90% of the remaining budget for the projected full run


def verify_prompts_verbatim(settings: Settings) -> None:
    """Phase 4: every source text must appear verbatim in its original document (skipped offline)."""
    if settings.offline:
        logger.info("Offline: skipping download-based verbatim check")
        return
    for name in prompts.SOURCES:
        missing = prompts.verify_source(name)
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} paragraph(s) not verbatim, first: {missing[0][:80]!r}")
        logger.info("Verbatim OK: prompts/%s", name)


def check_models_available(settings: Settings) -> None:
    """Phase 5: fail fast on unknown OpenRouter model ids (skipped offline)."""
    if not settings.offline:
        openrouter.check_models_available([*settings.models, settings.judge_model])
        logger.info("All models available on OpenRouter: %s + judge %s", settings.models, settings.judge_model)


def run_pilot_and_budget(settings: Settings, condition_prompts: dict, n_full: dict[str, int]) -> dict[str, int]:
    """Phase 7: run PILOT_N items per eval for every cell, project the full cost, shrink n if over budget.

    Costs are estimated from the pilot logs' token usage x OpenRouter list prices. That is deterministic and
    conservative (it ignores prompt-caching discounts); the key's measured `usage` lags by minutes, so it is
    only logged, not used. eval_set is resumable, so re-running with a finished pilot costs nothing.
    """
    # Offline (mock-model) pilot logs live apart so they can never mix with the real pilot's cost data.
    pilot_root = settings.log_dir / "inspect" / ("pilot_offline" if settings.offline else "pilot")
    n_pilot = {e: min(settings.pilot_n, n) for e, n in n_full.items()}
    runner.run_all(settings, condition_prompts, n_pilot, pilot_root)
    if settings.offline:
        return n_full
    cost_runs = runner.cost_and_runs_by_eval(pilot_root, openrouter.fetch_prices())
    pilot_cost = sum(cost for cost, _ in cost_runs.values())
    per_run = {e: cost / runs for e, (cost, runs) in cost_runs.items()}  # USD per sample-run (list prices)
    cells = len(settings.models) * len(settings.conditions)
    remaining = settings.max_budget_usd * SAFETY_MARGIN - pilot_cost
    n_new, projected = runner.scale_to_budget(n_full, per_run, cells, remaining)
    logger.info("Pilot cost ~$%.3f at list prices; USD per sample-run: %s",
                pilot_cost, {e: f"{c:.5f}" for e, c in per_run.items()})
    logger.info("Projected full run $%.2f of $%.2f usable budget", projected, remaining)
    if n_new != n_full:
        logger.warning("Budget gate scaled n from %s to %s", n_full, n_new)
    return n_new


def run(args: argparse.Namespace) -> None:
    """Run the phases selected by the command-line flags."""
    settings = load_settings(offline=args.offline)  # 1
    log_path = setup_logging(settings.log_dir)  # 2
    logger.info("Settings: %s", settings)
    full_root = settings.log_dir / "inspect" / ("offline" if settings.offline else "full")
    if settings.offline and not args.analyze_only:
        # Mock-model logs are throwaway; a fresh directory keeps eval_set from rejecting logs of other settings.
        shutil.rmtree(full_root, ignore_errors=True)
        shutil.rmtree(settings.log_dir / "inspect" / "pilot_offline", ignore_errors=True)
    n = {e: settings.n[e] for e in settings.evals}

    if not args.analyze_only:
        condition_prompts = prompts.build_condition_prompts()  # 3
        for condition, text in condition_prompts.items():
            logger.info("Condition %-16s %5s words", condition, len(text.split()) if text else 0)
        verify_prompts_verbatim(settings)  # 4
        check_models_available(settings)  # 5
        tasks = build_tasks(settings, n)  # 6 (built here once to validate data/downloads early)
        logger.info("Built %d eval tasks: %s", len(tasks), {t.name: len(t.dataset) for t in tasks})
        if not args.skip_pilot:
            n = run_pilot_and_budget(settings, condition_prompts, n)  # 7
        if args.pilot_only:
            logger.info("Pilot only: stopping before the full run")
            return
        ok = runner.run_all(settings, condition_prompts, n, full_root)  # 8
        logger.info("Full run finished (all tasks succeeded: %s)", ok)

    items = results.collect_item_scores(full_root, settings.conditions)  # 9
    summary = stats.cell_summary(items)  # 10
    contrasts = stats.contrast_table(items, report.PRIMARY + report.SECONDARY)
    # Deterministic cost of the logged full run: token usage x current OpenRouter list prices (targets + judge).
    full_cost = 0.0 if settings.offline else sum(
        cost for cost, _ in runner.cost_and_runs_by_eval(full_root, openrouter.fetch_prices()).values()
    )
    run_info = {
        "report generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "models": ", ".join(settings.models) if not settings.offline else "mockllm/model (offline)",
        "judge": settings.judge_model if not settings.offline else "mockllm/model (offline)",
        "conditions": ", ".join(settings.conditions),
        # Measured from the logs (samples x epochs per model x condition), so it reflects any budget scaling.
        "items per cell": items.groupby(["eval", "model", "condition"]).size().groupby("eval").max().to_dict(),
        "full-run cost at OpenRouter list prices (USD, from logged token usage)": round(full_cost, 2),
        "pipeline log": log_path.name,
    }
    reports_dir = settings.reports_dir if not settings.offline else settings.log_dir / "offline_report"
    if args.judge_checks:  # 12: paid, cached; JUDGE_CHECKS_MAX_USD caps each check's estimated cost
        max_usd = float(os.environ.get("JUDGE_CHECKS_MAX_USD", "1.0"))
        out = reports_dir / "judge_checks"
        judge_checks.run_blinding(settings, full_root, out / "blinding.csv", max_usd, args.judge_checks_limit)
        if args.judge_checks_limit is None:  # the validity check is cached only for full runs
            judge_checks.run_validity(settings, out / "validity.csv", max_usd)
    robustness.write_robustness(reports_dir, items, judge_checks.summary_markdown(reports_dir, items))  # 10b
    report.write_report(reports_dir, items, summary, contrasts, run_info)  # 11


def main() -> None:
    """Console-script entry point (see [project.scripts] in pyproject.toml)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="use Inspect's mockllm/model; no API calls, no cost")
    parser.add_argument("--pilot-only", action="store_true", help="stop after the pilot and budget projection")
    parser.add_argument("--skip-pilot", action="store_true", help="go straight to the full run (no budget gate)")
    parser.add_argument("--analyze-only", action="store_true", help="only re-read logs and rewrite the report")
    parser.add_argument("--judge-checks", action="store_true", help="also run the paid judge blinding/validity checks")
    parser.add_argument("--judge-checks-limit", type=int, default=None, help="pilot the blinding check on N answers")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
