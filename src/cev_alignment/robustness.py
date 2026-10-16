"""Robustness checks of the main results, written to `reports/robustness.md` (no API calls).

Local context: each check is a small pure function over the per-sample table from `results.collect_item_scores`
and reuses `stats.contrast` / `stats.contrast_table`:
  1. manipulation check: each target call got exactly its condition's prompt, and no judge call got one;
  2. Holm correction across ALL primary tests (2 contrasts x 6 evals), not only within a contrast;
  3. leave-one-model-out: does a pooled effect depend on a single model?
  4. missing-data sensitivity: complete cases only, and imputing unscorable samples at fixed values;
  5. OpenRouter provider routing per condition (open-weight models are served by several backends);
  6. answers that explicitly invoke CEV: how often, and how they score (post-treatment, so descriptive only);
  7. sub-category contrasts (principle / dimension / subset) with CIs and Holm within each eval.
Paid judge checks (judge_checks.py) append their own section when their results exist.

Global context: phase 10b (`write_robustness`) of `cev_alignment.main`, run after the main statistics.
Methodology references: Miller 2024 (arXiv 2411.00640); Holm via statsmodels `multipletests`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from cev_alignment import stats

logger = logging.getLogger(__name__)

PRIMARY = [("cev", "classic"), ("classic_cev", "classic_lenmatch")]
GROUP_SUFFIX = r"-[0-9a-f]{10}$|-\d+$"  # item ids are "<group>-<hash|index>" (see evals/*.py)
CEV_CONDITIONS = ("cev", "classic_cev")


def item_group(items: pd.DataFrame) -> pd.Series:
    """Sub-category of each item: HumaneBench principle, HAB dimension, ai_risk subset, DarkBench pattern."""
    return items["item"].astype(str).str.replace(GROUP_SUFFIX, "", regex=True)


def pooled(table: pd.DataFrame) -> pd.DataFrame:
    """Only the model-pooled rows of a contrast table (empty in, empty out)."""
    return table if table.empty else table[table["model"] == "pooled"].reset_index(drop=True)


def manipulation_summary(items: pd.DataFrame) -> pd.DataFrame:
    """Per condition: samples, samples whose target calls had the wrong prompt, samples with a judge leak."""
    return items.groupby("condition").agg(
        samples=("aligned", "size"), wrong_prompt=("prompt_ok", lambda s: int((~s.astype(bool)).sum())),
        judge_leak=("judge_leak", lambda s: int(s.astype(bool).sum())),
    ).reset_index()


def holm_across_all(items: pd.DataFrame, pairs: list[tuple[str, str]] = PRIMARY) -> pd.DataFrame:
    """Pooled primary contrasts with Holm over all contrast x eval tests together (`p_holm_all`)."""
    table = pooled(stats.contrast_table(items, pairs))
    if table.empty:  # e.g. offline mock runs: nothing scorable
        return table
    table = table.dropna(subset=["p"])
    table["p_holm_all"] = multipletests(table["p"], method="holm")[1] if len(table) else []
    return table


def leave_one_model_out(items: pd.DataFrame, pairs: list[tuple[str, str]] = PRIMARY) -> pd.DataFrame:
    """Pooled primary contrasts recomputed without each model in turn (`dropped` = model left out)."""
    frames = [pooled(stats.contrast_table(items[items["model"] != m], pairs)).assign(dropped=m)
              for m in sorted(items["model"].unique())]
    frames = [f for f in frames if not f.empty]  # dropping the only model leaves nothing to compare
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def missingness_sensitivity(items: pd.DataFrame, eval_id: str, pairs: list[tuple[str, str]],
                            fills: tuple[float, ...] = (0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Contrasts for one eval using complete cases only, and with unscorable samples imputed at each `fills` value.

    Complete case = a (model, item) scored under every condition, so all contrasts use the same units.
    """
    d = items[items["eval"] == eval_id]
    complete = d.groupby(["model", "item"])["aligned"].transform(lambda s: s.notna().all())
    variants = {"complete cases": d[complete]}
    variants |= {f"impute {f}": d.assign(aligned=d["aligned"].fillna(f)) for f in fills}
    rows = [{"method": name, "contrast": f"{a} - {b}", **stats.contrast(v, a, b)}
            for name, v in variants.items() for a, b in pairs]
    return pd.DataFrame(rows)


def provider_table(items: pd.DataFrame) -> pd.DataFrame:
    """Number of samples served by each OpenRouter backend, per model x condition."""
    return items.pivot_table(index=["model", "provider"], columns="condition", values="aligned",
                             aggfunc="size", fill_value=0)


def provider_residuals(items: pd.DataFrame, min_samples: int = 50) -> pd.DataFrame:
    """Mean aligned score of each provider minus its eval x condition x model mean (0 = no provider effect)."""
    d = items.dropna(subset=["aligned"]).copy()
    d["resid"] = d["aligned"] - d.groupby(["eval", "condition", "model"])["aligned"].transform("mean")
    out = d.groupby(["model", "provider"])["resid"].agg(mean_residual="mean", samples="size").reset_index()
    return out[out["samples"] >= min_samples]


def mention_summary(items: pd.DataFrame) -> pd.DataFrame:
    """Under the CEV prompts: share of answers that invoke CEV, and mean aligned score with / without it."""
    d = items[items["condition"].isin(CEV_CONDITIONS)]
    g = d.groupby("eval")
    return pd.DataFrame({
        "share_mentioning": g["mentions_cev"].mean(),
        "aligned_mentioning": d[d["mentions_cev"]].groupby("eval")["aligned"].mean(),
        "aligned_not_mentioning": d[~d["mentions_cev"]].groupby("eval")["aligned"].mean(),
        "n_mentioning": d[d["mentions_cev"]].groupby("eval").size(),
    }).reset_index()


def subgroup_contrasts(items: pd.DataFrame, pairs: list[tuple[str, str]] = PRIMARY) -> pd.DataFrame:
    """Pooled contrasts per sub-category, Holm-adjusted within each eval x contrast (exploratory)."""
    d = items.assign(group=item_group(items))
    rows = [{"eval": e, "group": g, "contrast": f"{a} - {b}", **stats.contrast(dg, a, b)}
            for (e, g), dg in d.groupby(["eval", "group"]) if dg["item"].nunique() > 1 for a, b in pairs]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm_within_eval"] = float("nan")
    for _, idx in out.dropna(subset=["p"]).groupby(["eval", "contrast"]).groups.items():
        out.loc[idx, "p_holm_within_eval"] = multipletests(out.loc[idx, "p"], method="holm")[1]
    return out


def _md(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Markdown table with 3-significant-digit floats."""
    if df.empty:
        return "_no data_"
    return (df[cols] if cols else df).to_markdown(index=False, floatfmt=".3g")


def write_robustness(reports_dir: Path, items: pd.DataFrame, extra_sections: str = "") -> Path:
    """Run every check, save CSVs under `reports/robustness/`, and write `reports/robustness.md`."""
    out_dir = reports_dir / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["contrast", "eval", "diff", "ci_low", "ci_high", "p", "n"]
    checks = {
        "manipulation": manipulation_summary(items),
        "holm_all_primary": holm_across_all(items),
        "leave_one_model_out": leave_one_model_out(items),
        "missingness_humanebench": missingness_sensitivity(items, "humanebench", PRIMARY + [("cev", "none"),
                                                                                           ("classic", "none")]),
        "provider_residuals": provider_residuals(items),
        "cev_mentions": mention_summary(items),
        "subgroups": subgroup_contrasts(items),
    }
    for name, df in checks.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    providers = provider_table(items)
    providers.to_csv(out_dir / "provider_counts.csv")
    markdown = f"""# Robustness checks

_Generated by `uv run cev-eval` from the full-run logs. Each section tests one way the main results could be wrong._

## 1. Manipulation check (from the logged API requests)
Each target call must carry exactly its condition's prompt components (classic / CEV / Huxley); judge calls must
carry none. Wrong-prompt and judge-leak counts should be 0.
{_md(checks["manipulation"])}

## 2. Holm correction across all 12 primary tests
`p_holm_all` corrects over both primary contrasts × 6 evals at once (stricter than the per-contrast Holm in
results.md).
{_md(checks["holm_all_primary"], cols + ["p_holm", "p_holm_all"])}

## 3. Leave-one-model-out (pooled primary contrasts)
Whether an effect depends on a single model. `dropped` is the model left out.
{_md(checks["leave_one_model_out"], ["dropped"] + cols + ["p_holm"])}

## 4. HumaneBench missing-data sensitivity
About 20% of HumaneBench samples are unscorable: the judge marked the targeted principle `not_applicable`, as
rubric v4.1 intends. The pattern is not random, so the contrasts are recomputed using complete cases (a model × item
scored in all five conditions), and with unscorable samples imputed at 0.5, 0.75 and 1.0 (0.75 = "acceptable").
{_md(checks["missingness_humanebench"], ["method", "contrast", "diff", "ci_low", "ci_high", "p", "n"])}

## 5. OpenRouter provider routing
Open-weight models are served by several backends, and the mix varied between conditions (they ran sequentially).
Below: mean aligned score of each provider (with ≥ 50 samples) minus its eval × condition × model mean. Values near 0
mean the provider mix cannot explain the condition effects. Counts per condition: `robustness/provider_counts.csv`.
{_md(checks["provider_residuals"])}

## 6. Answers that explicitly invoke CEV (cev and classic_cev conditions)
Whether an answer mentions CEV is an outcome of the prompt (post-treatment), so this split is descriptive and not
causal. A large gap means the effects are concentrated in answers where the model reasons "as CEV". That fits two
explanations: genuine CEV-driven behaviour, or a judge reacting to the mention. The judge-blinding test below
separates them.
{_md(checks["cev_mentions"])}

## 7. Sub-category contrasts (exploratory)
Pooled over models; `n` = items scored in both conditions. Holm-adjusted within each eval × contrast.
{_md(checks["subgroups"], ["eval", "group", "contrast", "diff", "ci_low", "ci_high", "p", "p_holm_within_eval", "n"])}
{extra_sections}"""
    path = reports_dir / "robustness.md"
    path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote robustness report %s", path)
    return path
