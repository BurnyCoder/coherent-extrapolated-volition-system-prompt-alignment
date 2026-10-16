"""Statistics: per-cell means with standard errors and paired condition contrasts with CIs and p-values.

Local context (methodology, following Miller 2024, "Adding Error Bars to Evals", arXiv 2411.00640):
- Unit of analysis = item. Epochs (repeated samples of one item) are averaged first, so standard errors are
  not inflated by treating repeats as independent (Miller's "clustered" recommendation).
- Contrasts are PAIRED by item: every condition sees the same items, so we test per-item differences.
  Binary items -> exact McNemar test (binomial test on discordant pairs); graded items -> Wilcoxon
  signed-rank. CIs: percentile bootstrap of the mean difference (scipy.stats.bootstrap).
- Pooled over models: per-item differences are averaged over models first (a cluster bootstrap with equal
  cluster sizes), then tested the same way.
- An eval with a single item run for many epochs (agentic_misalignment) has no pairing; its epochs are
  compared unpaired (Mann-Whitney U, two-sample bootstrap CI).
- Holm correction across evals within each contrast, on the pooled rows (statsmodels multipletests).
Docs: https://docs.scipy.org/doc/scipy/reference/stats.html , https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html

Global context: phase 10 (`compute_statistics`) of `cev_alignment.main`; output feeds `report`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.multitest import multipletests

SEED = 0  # bootstrap seed (reproducible CIs)
N_RESAMPLES = 2000


def item_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean aligned score per (eval, model, condition, item), averaging epochs and skipping NaN."""
    return df.groupby(["eval", "model", "condition", "item"], as_index=False)["aligned"].mean()


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard error (over items) and item count per (eval, model, condition), plus model-pooled rows."""
    im = item_means(df).dropna(subset=["aligned"])
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
    binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
    if binary:  # exact McNemar = two-sided binomial test on the discordant pairs
        n10, n01 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
        p = st.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        test = "mcnemar_exact"
    else:
        p = st.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
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


def contrast(df: pd.DataFrame, cond_a: str, cond_b: str) -> dict:
    """Compare condition A vs B on one eval (rows may span several models -> pooled over models)."""
    if df["item"].nunique() == 1:  # one prompt x many epochs: compare the epochs, unpaired
        a = df.loc[df.condition == cond_a, "aligned"].dropna().to_numpy(float)
        b = df.loc[df.condition == cond_b, "aligned"].dropna().to_numpy(float)
        return _unpaired(a, b)
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
