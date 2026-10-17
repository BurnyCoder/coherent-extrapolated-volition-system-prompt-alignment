"""Tests for cev_alignment.robustness on synthetic per-sample tables with known structure."""

import numpy as np
import pandas as pd

from cev_alignment import robustness


def table(values_by_model_condition, eval_id="e", group="g"):
    """Rows like results.collect_item_scores, items named "<group>-<index>"."""
    rows = []
    for (model, condition), values in values_by_model_condition.items():
        for i, v in enumerate(values):
            rows.append({"condition": condition, "eval": eval_id, "model": model, "item": f"{group}-{i:03d}",
                         "epoch": 1, "aligned": v, "provider": "P", "mentions_cev": False,
                         "prompt_ok": True, "judge_leak": False, "error": "", "stop_reason": "stop",
                         "output_chars": 9, "score_status": "", "judge_max_output_tokens": 0})
    return pd.DataFrame(rows)


def test_holm_across_all_is_at_least_per_contrast_holm():
    a = table({(m, c): [1.0] * 20 if c == "cev" else [0.0] * 20
               for m in ("m1", "m2") for c in ("cev", "classic", "classic_cev", "classic_lenmatch")})
    out = robustness.holm_across_all(a)
    assert set(out["contrast"]) == {"cev - classic", "classic_cev - classic_lenmatch"}
    assert (out["p_holm_all"] >= out["p_holm"] - 1e-12).all()


def test_leave_one_model_out_detects_single_model_effect():
    data = {("m1", "cev"): [1.0] * 20, ("m1", "classic"): [0.0] * 20,
            ("m2", "cev"): [0.5] * 20, ("m2", "classic"): [0.5] * 20}
    out = robustness.leave_one_model_out(table(data), [("cev", "classic")]).set_index("dropped")
    assert out.loc["m2", "diff"] == 1.0  # only m1 left: full effect
    assert out.loc["m1", "diff"] == 0.0  # only m2 left: no effect


def test_missingness_sensitivity_complete_cases_and_imputation():
    data = {("m1", "cev"): [1.0, np.nan, 1.0], ("m1", "classic"): [0.0, 0.0, 0.0]}
    out = robustness.missingness_sensitivity(table(data, "hb"), "hb", [("cev", "classic")], fills=(0.5,))
    out = out.set_index("method")
    assert out.loc["complete cases", "n"] == 2 and out.loc["complete cases", "diff"] == 1.0
    assert out.loc["impute 0.5", "n"] == 3 and out.loc["impute 0.5", "diff"] == np.mean([1, 0.5, 1])


def test_item_group_strips_index_and_hash_suffixes():
    items = pd.DataFrame({"item": ["survival-instinct-12", "defer_important_decisions-a58d26ba42", "brand-bias-082"]})
    assert list(robustness.item_group(items)) == ["survival-instinct", "defer_important_decisions", "brand-bias"]


def test_mention_summary_splits_scores_by_mention():
    d = table({("m1", "cev"): [0.0, 1.0]})
    d.loc[0, "mentions_cev"] = True
    row = robustness.mention_summary(d).iloc[0]
    assert row["share_mentioning"] == 0.5 and row["aligned_mentioning"] == 0.0 and row["aligned_not_mentioning"] == 1.0


def test_leave_one_model_out_with_a_single_model_is_empty_not_an_error():
    data = {("m1", "cev"): [1.0] * 5, ("m1", "classic"): [0.0] * 5}
    assert robustness.leave_one_model_out(table(data), [("cev", "classic")]).empty


def test_checks_tolerate_tables_with_nothing_scorable(tmp_path):
    data = table({("m1", c): [np.nan] * 3 for c in ("cev", "classic", "classic_cev", "classic_lenmatch", "none")})
    path = robustness.write_robustness(tmp_path, data)
    assert path.exists() and "Robustness checks" in path.read_text()


def test_unscorable_causes_separates_refusals_empty_answers_and_missing_scores():
    d = table({("m1", "cev"): [np.nan, np.nan, np.nan, 1.0]})
    d["error"], d["stop_reason"], d["output_chars"] = ["Error 403", "", "", ""], ["", "", "stop", "stop"], [0, 0, 9, 9]
    d.loc[2, "score_status"] = "not_applicable"
    row = robustness.unscorable_causes(d).iloc[0]
    assert (row["unscorable"], row["provider_refused"], row["empty_answer"], row["no_score"]) == (3, 1, 1, 1)
    assert row["no_score_status"] == "not_applicable: 1"


def test_refusals_lists_provider_errors_and_content_filter_stops():
    d = table({("m1", "cev"): [np.nan, 1.0, 1.0]})
    d["error"], d["stop_reason"], d["output_chars"] = ["Error 403", "", ""], ["", "content_filter", "stop"], [0, 5, 9]
    assert list(robustness.refusals(d)["item"]) == ["g-000", "g-001"]


def test_blackmail_counts_reports_harmful_epochs_as_fractions():
    d = table({("m1", "cev"): [0.0, 1.0, 0.0], ("m1", "classic"): [1.0, 1.0, 1.0]}, eval_id="agentic_misalignment")
    out = robustness.blackmail_counts(d).set_index("model")
    assert out.loc["m1", "cev"] == "2/3" and out.loc["m1", "classic"] == "0/3"


def test_mention_summary_adds_the_same_pairs_score_under_classic():
    d = table({("m1", "cev"): [0.0, 1.0], ("m1", "classic"): [0.4, 0.8]})
    d.loc[0, "mentions_cev"] = True
    row = robustness.mention_summary(d).iloc[0]
    assert row["classic_mentioning"] == 0.4 and row["classic_not_mentioning"] == 0.8
    assert row["share_mentioning_cev_only"] == 0.5
