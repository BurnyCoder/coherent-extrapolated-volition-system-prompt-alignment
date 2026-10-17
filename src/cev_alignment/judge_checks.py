"""Paid checks of the LLM judge: is it blind to the condition, and does it agree with the benchmark's own judge?

Local context:
- **Judge-blinding test.** Under the CEV prompts, up to about a fifth of answers explicitly invoke CEV
  (robustness.md §6), and on several evals those answers score much lower. That could be genuine CEV-driven
  behaviour, or the judge penalising the mention itself. Every CEV-mentioning answer is re-judged twice with the
  same scorer and judge: once as written, once with each sentence that mentions CEV removed (answers left empty by
  the removal are skipped). If the redacted copy scores the same, the drop is behavioural; if it scores
  higher, the judge reacts to the word. Redaction can also remove substantive content, so the gap is an upper
  bound on the judge-reaction effect.
- **Judge validity (HumanAgencyBench).** The benchmark's dataset stores its original o3 judge's score for every
  (prompt, response). We re-judge a stratified sample with our judge (the most common rubric template per
  dimension, same scoring) and report agreement.
Both checks call each eval's own Inspect scorer directly on a TaskState rebuilt from the logged sample
(inspect_ai.solver.TaskState), using the same judge Model as the run. Results are cached as CSVs, so a re-run makes
no API calls for rows already judged.

Global context: optional phase 12 of `cev_alignment.main` (`uv run cev-eval --judge-checks`); `summary_markdown`
adds a section to `reports/robustness.md`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import Scorer, Target
from inspect_ai.solver import TaskState
from scipy import stats as st
from statsmodels.stats.inter_rater import cohens_kappa

from cev_alignment import openrouter, stats
from cev_alignment.config import Settings
from cev_alignment.evals import REGISTRY, human_agency
from cev_alignment.evals.common import judge_model
from cev_alignment.results import CEV_MENTION, latest_logs

logger = logging.getLogger(__name__)

JUDGED_EVALS = ["humanebench", "human_agency", "darkbench", "instrumentaleval", "agentic_misalignment"]
CEV_CONDITIONS = ["cev", "classic_cev"]
CONCURRENCY = 16
VALIDITY_PER_DIMENSION = 20
CHARS_PER_TOKEN = 4  # rough token estimate for the budget guard


def redact_cev(text: str) -> str:
    """Remove every sentence (within each line) that mentions CEV / extrapolated volition; keep the rest verbatim."""
    kept_lines = []
    for line in text.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept = [s for s in sentences if not CEV_MENTION.search(s)]
        if kept or not line.strip():
            kept_lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def state_from(sample: EvalSample, model: str, response: str) -> TaskState:
    """TaskState for re-scoring `response` as the answer to the logged `sample` (input, metadata, target kept)."""
    output = ModelOutput.from_content(model=model, content=response)
    messages = [m for m in sample.messages if not isinstance(m, ChatMessageAssistant)] or [
        ChatMessageUser(content=str(sample.input))]
    return TaskState(model=ModelName(model), sample_id=sample.id, epoch=sample.epoch, input=sample.input,
                     messages=[*messages, output.message], target=Target(sample.target), output=output,
                     metadata=sample.metadata or {}, completed=True)


async def _score_all(jobs: list[tuple[Scorer, TaskState]]) -> list:
    """Await each scorer on its state, at most CONCURRENCY at a time; a failure becomes None."""
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(scorer: Scorer, state: TaskState):
        async with gate:
            try:
                return await scorer(state, state.target)
            except Exception as error:  # noqa: BLE001 (a failed judge call is recorded as unscorable)
                logger.warning("Re-scoring %s failed: %s", state.sample_id, error)
                return None

    return await asyncio.gather(*(one(s, st_) for s, st_ in jobs))


def scorers_for(settings: Settings) -> dict[str, Scorer]:
    """Each judged eval's own scorer, built exactly as in the run (same judge Model and settings)."""
    judge = judge_model(settings)
    return {e: REGISTRY[e].build(settings, 1, judge).scorer[0] for e in JUDGED_EVALS}


def estimate_usd(prompt_chars: int, n_calls: int, judge: str, out_tokens: int = 600) -> float:
    """Upper-bound cost of `n_calls` judge calls with prompts totalling `prompt_chars`, at OpenRouter list prices."""
    price_in, price_out = openrouter.fetch_prices().get(judge, (0.0, 0.0))
    return prompt_chars / CHARS_PER_TOKEN * price_in + n_calls * out_tokens * price_out


def blinding_records(log_root: Path) -> list[dict]:
    """Every CEV-mentioning answer of a judged eval under the CEV prompts, with its redacted copy."""
    records = []
    for condition in CEV_CONDITIONS:
        for header in latest_logs(log_root / condition):
            if header.eval.task not in JUDGED_EVALS:
                continue
            for sample in read_eval_log(header.location).samples or []:
                text = sample.output.completion if sample.output and sample.output.choices else ""
                if not CEV_MENTION.search(text):
                    continue
                records.append({"eval": header.eval.task, "condition": condition, "model": header.eval.model,
                                "item": str(sample.id), "epoch": sample.epoch, "sample": sample,
                                "original": text, "redacted": redact_cev(text)})
    return records


def run_blinding(
    settings: Settings, log_root: Path, out_csv: Path, max_usd: float, limit: int | None = None
) -> pd.DataFrame:
    """Re-judge original vs redacted CEV-mentioning answers; cache rows in `out_csv`."""
    records = blinding_records(log_root)[:limit]
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    key = lambda r: f"{r['eval']}|{r['condition']}|{r['model']}|{r['item']}|{r['epoch']}"  # noqa: E731
    seen = set(done.apply(key, axis=1)) if not done.empty else set()
    todo = [r for r in records if key(r) not in seen and r["redacted"]]
    est = estimate_usd(sum(2 * (len(r["original"]) + 9000) for r in todo), 2 * len(todo), settings.judge_model)
    logger.info("Judge blinding: %d answers to re-judge (x2), estimated <= $%.2f", len(todo), est)
    if est > max_usd:
        raise RuntimeError(f"Judge-blinding estimate ${est:.2f} exceeds JUDGE_CHECKS_MAX_USD ${max_usd:.2f}")
    scorers = scorers_for(settings)
    jobs = [(scorers[r["eval"]], state_from(r["sample"], r["model"], r[variant]))
            for r in todo for variant in ("original", "redacted")]  # original at 2i, redacted at 2i+1
    scores = asyncio.run(_score_all(jobs)) if jobs else []
    rows = []
    for i, r in enumerate(todo):
        orig, red = scores[2 * i], scores[2 * i + 1]
        aligned = [REGISTRY[r["eval"]].aligned({"s": s}) if s is not None else math.nan for s in (orig, red)]
        rows.append({k: r[k] for k in ("eval", "condition", "model", "item", "epoch")} |
                    {"aligned_original": aligned[0], "aligned_redacted": aligned[1],
                     "chars_removed": len(r["original"]) - len(r["redacted"])})
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def validity_rows(settings: Settings) -> pd.DataFrame:
    """Stratified sample of HumanAgencyBench rows with the original judge's score (20 per dimension, seeded)."""
    path = settings.cache_dir / human_agency.REPO_ID.replace("/", "__") / human_agency.PARQUET
    cols = ["subject_model", "dimension", "prompt", "misinformation", "subject_response", "subject_system_prompt",
            "score"]
    df = pd.read_parquet(path, columns=cols).dropna(subset=["subject_response", "prompt"])
    # DataFrameGroupBy.sample keeps every column (pandas 3's groupby.apply drops the grouping column).
    return df.groupby("dimension").sample(n=VALIDITY_PER_DIMENSION, random_state=settings.seed).reset_index(drop=True)


def run_validity(settings: Settings, out_csv: Path, max_usd: float, limit: int | None = None) -> pd.DataFrame:
    """Re-judge stored HumanAgencyBench responses with our judge; cache rows in `out_csv`."""
    if out_csv.exists() and limit is None:
        return pd.read_csv(out_csv)
    rows = validity_rows(settings).head(limit) if limit else validity_rows(settings)
    est = estimate_usd(int((rows["subject_response"].str.len() + rows["prompt"].str.len() + 4000).sum()),
                       len(rows), settings.judge_model)
    logger.info("Judge validity: %d HumanAgencyBench rows, estimated <= $%.2f", len(rows), est)
    if est > max_usd:
        raise RuntimeError(f"Judge-validity estimate ${est:.2f} exceeds JUDGE_CHECKS_MAX_USD ${max_usd:.2f}")
    scorer = scorers_for(settings)["human_agency"]
    states = []
    for i, r in rows.iterrows():
        misinformation = r["misinformation"] if isinstance(r["misinformation"], str) else None
        meta = {"dimension": r["dimension"], "misinformation": misinformation,
                "subject_system_prompt": r["subject_system_prompt"] or ""}
        sample = EvalSample(id=i, epoch=1, input=r["prompt"], target=r["dimension"], metadata=meta,
                            messages=[ChatMessageUser(content=r["prompt"])])
        states.append(state_from(sample, f"hab/{r['subject_model']}", r["subject_response"]))
    scores = asyncio.run(_score_all([(scorer, s) for s in states]))
    rows = rows.drop(columns=["prompt", "subject_response", "misinformation", "subject_system_prompt"])
    rows["ours"] = [10 * s.as_float() if s is not None else math.nan for s in scores]
    rows = rows.rename(columns={"score": "original"})
    if limit is None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out_csv, index=False)
    return rows


def summary_markdown(reports_dir: Path, items: pd.DataFrame) -> str:
    """Markdown section for robustness.md from the cached CSVs (empty if the checks were never run)."""
    blind_csv, valid_csv = reports_dir / "judge_checks" / "blinding.csv", reports_dir / "judge_checks" / "validity.csv"
    parts = ["\n## 10. Judge checks (paid; `uv run cev-eval --judge-checks`)\n"]
    if blind_csv.exists():
        b = pd.read_csv(blind_csv)
        b["item"] = b["item"].astype(str)
        rows = []
        for e, g in b.dropna(subset=["aligned_original", "aligned_redacted"]).groupby("eval"):
            r = stats._paired(g["aligned_redacted"].to_numpy(float), g["aligned_original"].to_numpy(float))
            rows.append({"eval": e, "mean_original": g["aligned_original"].mean(),
                         "mean_redacted": g["aligned_redacted"].mean(), **r})
        keys = ["eval", "condition", "model", "item", "epoch"]
        stored = items.assign(model="openrouter/" + items["model"]).merge(b, on=keys)
        retest = stored.dropna(subset=["aligned", "aligned_original"])
        retest_corr = (st.spearmanr(retest["aligned"], retest["aligned_original"]).statistic
                       if len(retest) > 2 else math.nan)
        # Primary contrasts with each CEV-mentioning answer's stored score replaced by its redacted re-judgement.
        sub = b.assign(model=b["model"].str.removeprefix("openrouter/"))[[*keys, "aligned_redacted"]]
        adjusted = items.merge(sub, on=keys, how="left")
        adjusted["aligned"] = adjusted["aligned_redacted"].combine_first(adjusted["aligned"])
        adj = stats.contrast_table(adjusted, [("cev", "classic"), ("classic_cev", "classic_lenmatch")])
        adj = adj[adj["model"] == "pooled"]
        mentioning = items[items["mentions_cev"].astype(bool) & items["eval"].isin(JUDGED_EVALS)
                           & items["condition"].isin(CEV_CONDITIONS)]
        n_both = len(b.dropna(subset=["aligned_original", "aligned_redacted"]))
        parts.append(
            f"### Judge-blinding test\n{len(b)} of the {len(mentioning)} CEV-mentioning answers of the judged evals "
            "under `cev`/`classic_cev` were re-judged as written and with their CEV sentences removed (same scorer, "
            f"same judge); {len(mentioning) - len(b)} were skipped because removing those sentences left no text, and "
            f"{n_both} got a score both times. `diff` = redacted − original "
            "(positive = the judge penalises the mention itself). The redaction can remove substantive content, so "
            "this is an upper bound on the judge-reaction effect.\n\n"
            + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g")
            + f"\n\nJudge test-retest (stored score vs. re-judged original, same answer): Spearman ρ = "
              f"{retest_corr:.3f} over {len(retest)} answers; exact agreement "
              f"{(retest['aligned'] == retest['aligned_original']).mean():.1%}.\n\n"
            "Primary contrasts with redacted re-judgements substituted for the CEV-mentioning answers:\n\n"
            + adj[["contrast", "eval", "diff", "ci_low", "ci_high", "p", "p_holm", "n"]]
            .to_markdown(index=False, floatfmt=".3g")
            + "\n")
    if valid_csv.exists():
        v = pd.read_csv(valid_csv).dropna(subset=["ours", "original"])
        table = pd.crosstab(v["ours"] == 10, v["original"] == 10)
        kappa = cohens_kappa(table.to_numpy()).kappa if table.shape == (2, 2) else math.nan
        per_dim = v.groupby("dimension").apply(
            lambda g: pd.Series({"spearman": st.spearmanr(g["ours"], g["original"]).statistic,
                                 "mae": (g["ours"] - g["original"]).abs().mean(), "n": len(g)}))
        parts.append(
            "### Judge validity on HumanAgencyBench\n"
            f"{len(v)} stored (prompt, response) pairs from {v['subject_model'].nunique()} subject models, re-judged "
            "by our judge with the most common rubric template per dimension (recovered from the dataset; rubric "
            "revisions mean some original judgements used slightly different wording) and the same scoring, compared "
            "with the benchmark's original o3 judge (per the dataset card; 0–10 scale): "
            f"Spearman ρ = {st.spearmanr(v['ours'], v['original']).statistic:.3f}, "
            f"Pearson r = {np.corrcoef(v['ours'], v['original'])[0, 1]:.3f}, MAE = "
            f"{(v['ours'] - v['original']).abs().mean():.2f}, mean ours − original = "
            f"{(v['ours'] - v['original']).mean():+.2f}, Cohen's κ on \"perfect score (10)\" = {kappa:.3f}.\n\n"
            + per_dim.reset_index().to_markdown(index=False, floatfmt=".3g") + "\n")
    return "".join(parts) if len(parts) > 1 else ""
