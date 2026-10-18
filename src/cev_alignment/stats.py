"""Statistics: per-cell means with standard errors and paired condition contrasts with CIs and p-values.

Local context (methodology, following Miller 2024, "Adding Error Bars to Evals", arXiv 2411.00640):
- Unit of analysis = item. Epochs (repeated samples of one item) are averaged first, so standard errors are
  not inflated by treating repeats as independent (Miller's "clustered" recommendation).
- Contrasts are PAIRED by item: every condition sees the same items, so we test per-item differences.
  Scores that are 0/1 in both conditions -> exact McNemar test (binomial test on discordant pairs); anything else
  -> Wilcoxon signed-rank. CIs: percentile bootstrap of the mean difference (scipy.stats.bootstrap).
- Pooled over models: per-item differences are averaged over models first (a cluster bootstrap with equal
  cluster sizes), then tested the same way. Those means are fractional, so pooled tests are Wilcoxon even for
  binary evals; McNemar applies to per-model contrasts.
- An eval with a single item run for many epochs (agentic_misalignment) has no pairing; its epochs are
  compared unpaired (Mann-Whitney U, two-sample bootstrap CI). Pooled over models, its binary scores are tested
  with a Cochran-Mantel-Haenszel test stratified by model, so differences between models are not mistaken for an
  effect of the condition (statsmodels StratifiedTable).
- Holm correction across evals within each contrast, on the pooled rows (statsmodels multipletests).
Docs: https://docs.scipy.org/doc/scipy/reference/stats.html , https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html

Global context: phase 10 (`compute_statistics`) of `cev_alignment.main`; output feeds `report`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

SEED = 0  # bootstrap seed (reproducible CIs)
N_RESAMPLES = 2000


def item_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean aligned score per (eval, model, condition, item), averaging epochs and skipping NaN."""
    return df.groupby(["eval", "model", "condition", "item"], as_index=False)["aligned"].mean()


def _epochs_as_items_for_single_item_evals(df: pd.DataFrame) -> pd.DataFrame:
    """For evals with one prompt run for many epochs (agentic_misalignment) the epochs are the units."""
    single = df.groupby("eval")["item"].transform("nunique") == 1
    return df.assign(item=np.where(single, df["item"] + "#" + df["epoch"].astype(str), df["item"]))


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard error (over items) and item count per (eval, model, condition), plus model-pooled rows."""
    im = item_means(_epochs_as_items_for_single_item_evals(df)).dropna(subset=["aligned"])
    pooled = im.groupby(["eval", "condition", "item"], as_index=False)["aligned"].mean().assign(model="pooled")
    both = pd.concat([im, pooled], ignore_index=True)
    out = both.groupby(["eval", "model", "condition"])["aligned"].agg(
        mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan, n_items="count"
    )
    return out.reset_index()


def _bootstrap_ci(*samples: np.ndarray, statistic) -> tuple[float, float]:
    """95% percentile bootstrap CI; degenerate samples give a zero-width interval at the point estimate."""
    point = statistic(*samples)
    if any(len(s) < 2 for s in samples) or all(np.all(s == s[0]) for s in samples):
        return point, point
    res = st.bootstrap(samples, statistic, n_resamples=N_RESAMPLES, method="percentile",
                       random_state=SEED, vectorized=False)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


EMPTY = {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0, "test": "none"}


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired test on per-item scores a (condition A) and b (condition B); diff = mean(a - b)."""
    d = a - b
    # Rank tests need equal differences to be equal floats: 0.3 - 0.1 != 0.8 - 0.6 in floating point, so SciPy would
    # rank such ties as distinct values and misapply its tie correction. Scores lie on grids no finer than 1/40, so 9
    # decimals only removes float noise (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
    d_test = np.round(d, 9)
    binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
    if binary:  # exact McNemar = two-sided binomial test on the discordant pairs
        n10, n01 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
        p = st.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        test = "mcnemar_exact"
    else:
        p = st.wilcoxon(d_test).pvalue if np.any(d_test != 0) else 1.0
        test = "wilcoxon"
    low, high = _bootstrap_ci(d, statistic=np.mean)
    return {"diff": float(d.mean()), "ci_low": low, "ci_high": high, "p": float(p), "n": len(d), "test": test}


def _unpaired(a: np.ndarray, b: np.ndarray) -> dict:
    """Unpaired comparison of repeated samples (single-item evals); diff = mean(a) - mean(b)."""
    if not len(a) or not len(b):
        return dict(EMPTY)
    all_tied = len(np.unique(np.concatenate([a, b]))) <= 1  # identical values everywhere -> no evidence
    p = 1.0 if all_tied or not len(a) or not len(b) else st.mannwhitneyu(a, b).pvalue
    low, high = _bootstrap_ci(a, b, statistic=lambda x, y: np.mean(x) - np.mean(y))
    return {"diff": float(a.mean() - b.mean()), "ci_low": low, "ci_high": high, "p": float(p),
            "n": min(len(a), len(b)), "test": "mannwhitney"}


def _cmh_p(df: pd.DataFrame, cond_a: str, cond_b: str) -> float:
    """Cochran-Mantel-Haenszel p-value of a 0/1 score, A vs B, stratified by model (one 2x2 table per model).

    A model whose epochs all have the same score carries no information and is left out; none left -> 1.0. With
    the continuity correction, because counts are small (on this run's data it matches an exact within-model
    permutation test: 0.27 vs 0.27 and 0.43 vs 0.42 on the primary contrasts).
    Docs: https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.StratifiedTable.html
    """
    tables = [np.array([[((g.condition == c) & (g.aligned == v)).sum() for v in (1.0, 0.0)] for c in (cond_a, cond_b)],
                       dtype=float)
              for _, g in df.groupby("model") if g["aligned"].nunique() > 1]
    return float(StratifiedTable(tables).test_null_odds(correction=True).pvalue) if tables else 1.0


def contrast(df: pd.DataFrame, cond_a: str, cond_b: str) -> dict:
    """Compare condition A vs B on one eval (rows may span several models -> pooled over models)."""
    if df.empty or df["aligned"].isna().all():  # nothing scorable (e.g. an empty complete-case subset)
        return dict(EMPTY)
    if df["item"].nunique() == 1:  # one prompt x many epochs: compare the epochs, unpaired
        d = df[df.condition.isin([cond_a, cond_b])].dropna(subset=["aligned"])
        out = _unpaired(d.loc[d.condition == cond_a, "aligned"].to_numpy(float),
                        d.loc[d.condition == cond_b, "aligned"].to_numpy(float))
        if out["test"] != "none" and d["model"].nunique() > 1 and set(d["aligned"].unique()) <= {0.0, 1.0}:
            out |= {"p": _cmh_p(d, cond_a, cond_b), "test": "cmh"}  # pooled over models: stratify by model
        return out
    im = item_means(df)
    wide = im.pivot_table(index=["model", "item"], columns="condition", values="aligned")
    if not {cond_a, cond_b} <= set(wide.columns):  # a condition with no scorable items at all
        return dict(EMPTY)
    wide = wide.dropna(subset=[cond_a, cond_b])  # keep items scored under both conditions
    if wide.empty:
        return dict(EMPTY)
    per_item = (wide[cond_a] - wide[cond_b]).groupby(level="item").mean()  # average over models
    a = wide[cond_a].groupby(level="item").mean().to_numpy(float)
    b = wide[cond_b].groupby(level="item").mean().to_numpy(float)
    out = _paired(a, b)
    out["diff"] = float(per_item.mean())
    return out


def contrast_table(df: pd.DataFrame, contrasts: list[tuple[str, str]]) -> pd.DataFrame:
    """All contrasts x evals x (each model + pooled), with Holm-adjusted p on pooled rows."""
    rows = []
    for cond_a, cond_b in contrasts:
        for eval_id, de in df.groupby("eval"):
            if not {cond_a, cond_b} <= set(de.condition):
                continue
            for model, dm in [*de.groupby("model"), ("pooled", de)]:
                rows.append({"contrast": f"{cond_a} - {cond_b}", "eval": eval_id, "model": model,
                             **contrast(dm, cond_a, cond_b)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = np.nan
    for _, group in table[(table.model == "pooled") & table.p.notna()].groupby("contrast"):
        table.loc[group.index, "p_holm"] = multipletests(group["p"], method="holm")[1]
    return table
