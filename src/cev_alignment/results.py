"""Turn Inspect `.eval` logs into one tidy table of per-sample aligned scores.

Local context: for each condition directory, keep the newest successful log per (task, model) (eval_set may
leave older failed attempts behind), read its samples, and map each sample's Inspect scores through the
eval's `aligned()` function (HIGHER = MORE ALIGNED, in [0, 1]; NaN = could not be scored).
Docs: https://inspect.aisi.org.uk/eval-logs.html (list_eval_logs, read_eval_log)

It also records per-sample diagnostics used by `robustness.py`: the OpenRouter backend `provider` that served the
target call, whether the answer `mentions_cev`, and a manipulation check from the logged API requests
(`prompt_ok`: the system message holds exactly the condition's prompt components; `judge_leak`: a judge call's
system message holds any of them).

Global context: phase 9 (`collect_item_scores`) of `cev_alignment.main`; the DataFrame feeds `stats`,
`robustness` and `report` and is saved as `reports/item_scores.csv`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import pandas as pd
from inspect_ai.event import ModelEvent, SpanBeginEvent
from inspect_ai.log import EvalLog, EvalSample, list_eval_logs, read_eval_log

from cev_alignment.evals import REGISTRY
from cev_alignment.prompts import EXPECTED_MARKERS, MARKERS

logger = logging.getLogger(__name__)

COLUMNS = ["condition", "eval", "model", "item", "epoch", "aligned", "error", "stop_reason", "output_chars",
           "provider", "mentions_cev", "prompt_ok", "judge_leak"]
CEV_MENTION = re.compile(r"extrapolated volition|\bCEV\b", re.IGNORECASE)  # the answer invokes CEV itself


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


def _system_text(event: ModelEvent) -> str:
    """Text of the system message(s) actually sent to the provider (after Inspect merged them)."""
    messages = event.call.request.get("messages", []) if event.call else []
    return " ".join(json.dumps(m.get("content")) for m in messages if m.get("role") == "system")


def _in_scorer(event: ModelEvent, spans: dict[str, SpanBeginEvent]) -> bool:
    """True if the call was made while scoring (a judge/grader call): an ancestor span has type "scorer(s)".

    Inspect nests every event in spans ("solvers" > "generate" for the model under test, "scorers" > <scorer> for
    graders), which separates the two even when the judge and the target are the same model (offline mock).
    """
    span_id = event.span_id
    while span_id in spans:
        if spans[span_id].type in ("scorer", "scorers"):
            return True
        span_id = spans[span_id].parent_id
    return False


def call_diagnostics(sample: EvalSample, condition: str) -> dict:
    """Provider of the last target call, and the manipulation check over every model call of the sample."""
    provider, prompt_ok, judge_leak = "", True, False
    spans = {e.id: e for e in sample.events if isinstance(e, SpanBeginEvent)}
    for event in sample.events:
        if not (isinstance(event, ModelEvent) and event.call):
            continue
        found = {name for name, snippet in MARKERS.items() if snippet in _system_text(event)}
        if not _in_scorer(event, spans):  # the model under test
            prompt_ok &= found == EXPECTED_MARKERS.get(condition, found)
            if isinstance(event.call.response, dict):
                provider = event.call.response.get("provider") or provider
        else:  # a judge / grader call must never carry a condition prompt
            judge_leak |= bool(found)
    return {"provider": provider, "prompt_ok": prompt_ok, "judge_leak": judge_leak}


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
            "mentions_cev": bool(CEV_MENTION.search(completion)),
            **call_diagnostics(sample, condition),
        })
    return rows


def collect_item_scores(log_root: Path, conditions: list[str]) -> pd.DataFrame:
    """All per-sample aligned scores under `log_root/<condition>/` as one DataFrame (see COLUMNS)."""
    rows: list[dict] = []
    for condition in conditions:
        for header in latest_logs(log_root / condition):
            # Full log with samples; attachments resolved so the logged API requests can be checked.
            log = read_eval_log(header.location, resolve_attachments=True)
            rows.extend(sample_rows(log, condition))
            n_samples = len(log.samples or [])
            logger.info("Read %s | %s | %s | %d samples", condition, log.eval.task, log.eval.model, n_samples)
    df = pd.DataFrame(rows, columns=COLUMNS)
    logger.info("Collected %d sample rows; %d unscorable (NaN); %d with a wrong prompt; %d with a judge leak",
                len(df), int(df["aligned"].isna().sum()), int((~df["prompt_ok"]).sum()), int(df["judge_leak"].sum()))
    return df
