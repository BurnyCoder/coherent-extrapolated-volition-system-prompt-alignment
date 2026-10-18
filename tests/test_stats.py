"""Tests for cev_alignment.stats on synthetic item scores with known effects."""

import numpy as np
import pandas as pd
import pytest

from cev_alignment import stats


def make_df(eval_id, values_by_condition, models=("m1", "m2"), epochs=1):
    """Long table like results.collect_item_scores: one row per (condition, model, item, epoch)."""
    rows = []
    for condition, values in values_by_condition.items():
        for model in models:
            for item, value in enumerate(values):
                for epoch in range(1, epochs + 1):
                    rows.append({"condition": condition, "eval": eval_id, "model": model, "item": str(item),
                                 "epoch": epoch, "aligned": value})
    return pd.DataFrame(rows)


def test_paired_binary_uses_mcnemar_and_finds_effect():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, 60).astype(float)
    better = np.where(rng.random(60) < 0.5, 1.0, base)  # flips ~half of the zeros to ones
    df = make_df("e", {"cev": better, "classic": base}, models=("m1",))
    out = stats.contrast(df[df.model == "m1"], "cev", "classic")
    assert out["test"] == "mcnemar_exact"
    assert out["diff"] == pytest.approx(better.mean() - base.mean())
    assert out["p"] < 0.01 and out["ci_low"] > 0


def test_paired_continuous_uses_wilcoxon_and_no_effect_gives_large_p():
    values = np.linspace(0, 1, 40)
    df = make_df("e", {"cev": values, "classic": values}, models=("m1",))
    out = stats.contrast(df, "cev", "classic")
    assert out["test"] == "wilcoxon"
    assert out["diff"] == 0 and out["p"] == 1.0


def test_single_item_with_epochs_is_compared_unpaired():
    rows = [{"condition": c, "eval": "am", "model": "m1", "item": "x", "epoch": e, "aligned": float(v)}
            for c, vals in {"cev": [1] * 10, "classic": [0] * 10}.items() for e, v in enumerate(vals)]
    out = stats.contrast(pd.DataFrame(rows), "cev", "classic")
    assert out["test"] == "mannwhitney"
    assert out["diff"] == 1.0 and out["p"] < 0.001


def epochs_df(values_by_model_condition):
    """Single-item eval (one prompt x epochs), like agentic_misalignment."""
    return pd.DataFrame([{"condition": c, "eval": "am", "model": m, "item": "x", "epoch": e, "aligned": float(v)}
                         for (m, c), vals in values_by_model_condition.items() for e, v in enumerate(vals)])


def test_pooled_single_item_eval_is_stratified_by_model():
    df = epochs_df({("m1", "cev"): [1] * 15, ("m1", "classic"): [1] * 15,  # no variation: no information
                    ("m2", "cev"): [1] * 15, ("m2", "classic"): [0] * 12 + [1] * 3})
    out = stats.contrast(df, "cev", "classic")
    assert out["test"] == "cmh" and out["p"] < 0.001 and out["n"] == 30
    assert out["diff"] == pytest.approx(0.4)  # mean over both models' epochs: (30 - 18) / 30
    assert stats.contrast(df[df.model == "m2"], "cev", "classic")["test"] == "mannwhitney"


def test_pooled_single_item_eval_without_within_model_variation_gives_p_1():
    df = epochs_df({("m1", "cev"): [1] * 5, ("m1", "classic"): [1] * 5,
                    ("m2", "cev"): [0] * 5, ("m2", "classic"): [0] * 5})
    assert stats.contrast(df, "cev", "classic")["p"] == 1.0


def test_contrast_table_pools_models_and_applies_holm():
    df = pd.concat([
        make_df("e1", {"cev": [1.0] * 30, "classic": [0.0] * 30}),
        make_df("e2", {"cev": [0.5] * 30, "classic": [0.5] * 30}),
    ])
    table = stats.contrast_table(df, [("cev", "classic")])
    pooled = table[table.model == "pooled"].set_index("eval")
    assert set(table.model) == {"m1", "m2", "pooled"}
    assert pooled.loc["e1", "diff"] == 1.0 and pooled.loc["e1", "p_holm"] < 0.01
    assert pooled.loc["e2", "p_holm"] == 1.0
    assert (pooled["p_holm"] >= pooled["p"]).all()


def test_cell_summary_averages_epochs_before_se():
    df = make_df("e", {"cev": [0.0, 1.0]}, models=("m1",), epochs=3)
    summary = stats.cell_summary(df)
    row = summary.iloc[0]
    assert row["mean"] == 0.5 and row["n_items"] == 2
    assert row["se"] == pytest.approx(0.5)  # std([0,1], ddof=1)/sqrt(2)


def test_nan_items_are_dropped_pairwise():
    df = make_df("e", {"cev": [1.0, np.nan, 1.0], "classic": [0.0, 0.0, np.nan]}, models=("m1",))
    out = stats.contrast(df, "cev", "classic")
    assert out["n"] == 1


def test_condition_without_scorable_items_gives_empty_contrast():
    df = make_df("e", {"cev": [np.nan, np.nan], "classic": [1.0, 0.0]}, models=("m1",))
    out = stats.contrast(df, "cev", "classic")
    assert out["n"] == 0 and np.isnan(out["diff"])


def test_cell_summary_uses_epochs_as_units_for_single_item_evals():
    rows = [{"condition": "cev", "eval": "am", "model": "m1", "item": "x", "epoch": e, "aligned": float(e % 2)}
            for e in range(1, 5)]
    row = stats.cell_summary(pd.DataFrame(rows)).query("model == 'm1'").iloc[0]
    assert row["n_items"] == 4 and row["mean"] == 0.5 and not np.isnan(row["se"])


def test_wilcoxon_treats_differences_equal_in_exact_arithmetic_as_ties():
    # Scores on a 0.1 grid: 0.3 - 0.1 and 0.8 - 0.6 are both 0.2, but not in floating point. Unrounded, SciPy ranks
    # them as distinct values and the tie correction no longer matches the data (found by the second audit).
    a_int = np.array([3, 8, 5, 9, 4, 7, 6, 10, 2, 3, 6, 9, 1, 5, 7, 4, 8, 2, 6, 3])
    b_int = np.array([1, 6, 3, 7, 2, 5, 5, 9, 1, 2, 7, 8, 1, 4, 6, 3, 6, 0, 4, 1])
    exact = stats.st.wilcoxon((a_int - b_int) / 10).pvalue  # integer differences: equal values are equal floats
    assert stats._paired(a_int / 10, b_int / 10)["p"] == pytest.approx(exact)
