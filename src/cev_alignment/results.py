"""Turn Inspect `.eval` logs into one tidy table of per-sample aligned scores.

Local context: for each condition directory, keep the newest successful log per (task, model) (eval_set may
leave older failed attempts behind), read its samples, and map each sample's Inspect scores through the
eval's `aligned()` function (HIGHER = MORE ALIGNED, in [0, 1]; NaN = could not be scored).
Docs: https://inspect.aisi.org.uk/eval-logs.html (list_eval_logs, read_eval_log)

Global context: phase 9 (`collect_item_scores`) of `cev_alignment.main`; the DataFrame feeds `stats` and
`report` and is saved as `reports/item_scores.csv`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

from cev_alignment.evals import REGISTRY

logger = logging.getLogger(__name__)

COLUMNS = ["condition", "eval", "model", "item", "epoch", "aligned", "error", "stop_reason", "output_chars"]


def latest_logs(log_dir: Path) -> list[EvalLog]:
    """Newest log per (task, model), preferring status == "success" (headers only, cheap)."""
    chosen: dict[tuple[str, str], EvalLog] = {}
    headers = [read_eval_log(info, header_only=True) for info in list_eval_logs(str(log_dir))]
    for log in sorted(headers, key=lambda h: (h.status == "success", h.eval.created)):  # best last
        chosen[(log.eval.task, log.eval.model)] = log
    return list(chosen.values())


def aligned_score(eval_id: str, scores: dict | None, completion: str, target: str = "") -> float:
    """Aligned score of one sample, NaN if it has no scores, an empty answer, or a malformed score value.

    An empty answer (e.g. the whole token budget went to reasoning) must not count as "aligned" just
    because the judge found nothing harmful in it.
    """
    if not scores or not completion.strip():
        return math.nan
    spec = REGISTRY[eval_id]
    try:
        return spec.rescore(completion, target) if spec.rescore else spec.aligned(scores)
    except (KeyError, TypeError, ValueError, AttributeError):
        return math.nan


def sample_rows(log: EvalLog, condition: str) -> list[dict]:
    """One row per (sample, epoch) of a full log."""
    rows = []
    for sample in log.samples or []:
        has_output = bool(sample.output and sample.output.choices)  # errored samples can have no choices
        completion = sample.output.completion if has_output else ""
        target = sample.target if isinstance(sample.target, str) else "".join(sample.target)
        aligned = aligned_score(log.eval.task, sample.scores, completion, target)
        rows.append({
            "condition": condition,
            "eval": log.eval.task,
            "model": log.eval.model.removeprefix("openrouter/"),
            "item": str(sample.id),
            "epoch": sample.epoch,
            "aligned": aligned,
            "error": sample.error.message[:200] if sample.error else "",
            "stop_reason": sample.output.stop_reason if has_output else "",
            "output_chars": len(completion),
        })
    return rows


def collect_item_scores(log_root: Path, conditions: list[str]) -> pd.DataFrame:
    """All per-sample aligned scores under `log_root/<condition>/` as one DataFrame (see COLUMNS)."""
    rows: list[dict] = []
    for condition in conditions:
        for header in latest_logs(log_root / condition):
            log = read_eval_log(header.location)  # full log including samples
            rows.extend(sample_rows(log, condition))
            n_samples = len(log.samples or [])
            logger.info("Read %s | %s | %s | %d samples", condition, log.eval.task, log.eval.model, n_samples)
    df = pd.DataFrame(rows, columns=COLUMNS)
    logger.info("Collected %d sample rows; %d unscorable (NaN)", len(df), int(df["aligned"].isna().sum()))
    return df
