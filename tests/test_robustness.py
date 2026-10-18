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


def answers(rows):
    """Like robustness.ai_risk_answers: one logged ai_risk reply per row, `target` = the aligned letter."""
    return pd.DataFrame([{"condition": c, "model": "m1", "item": f"survival-instinct-{i}", "epoch": 1, "target": t,
                          "completion": text} for i, (c, t, text) in enumerate(rows)])


def test_parser_multiverse_rescores_each_reply_with_every_parser_version():
    d = answers([("cev", "A", "Choice (A) suggests one thing. Choice (B) another.\n\nAnswer: (B)"),
                 ("classic", "A", "(A)")])
    d["item"] = "survival-instinct-0"  # the same item under both conditions -> one paired contrast
    out = robustness.parser_multiverse(d, [("cev", "classic")]).set_index("parser")
    assert out.loc["v1 (run time)", "diff"] == 0.0  # v1: the first "(A)" anywhere
    assert out.loc["v2 (after first results)", "diff"] == 0.0  # v2: "Choice (A" read as the choice
    assert out.loc["current", "diff"] == -1.0  # current: "Answer: (B)"
    assert out.loc["leading letter only", "unscorable"] == 1


def test_position_balanced_estimate_cancels_a_pure_letter_shift():
    # "cev" answers B to everything, "classic" answers A; items 0-1 have aligned letter A, items 2-3 have B.
    targets = "AABB"
    d = answers([(c, targets[i], "(B)" if c == "cev" else "(A)") for i in range(4) for c in ("cev", "classic")])
    d["item"] = [f"survival-instinct-{i}" for i in range(4) for _ in range(2)]  # each item under both conditions
    out = robustness.position_balanced(d, [("cev", "classic")]).iloc[0]
    assert out["diff_target_A"] == -1.0 and out["diff_target_B"] == 1.0 and out["balanced"] == 0.0


def test_tipping_point_shifts_imputed_values_apart():
    data = {("m1", "cev"): [1.0, np.nan], ("m1", "classic"): [1.0, np.nan]}
    out = robustness.tipping_point(table(data, "hb"), "hb", ("cev", "classic"), deltas=(0.0, 0.5)).set_index("delta")
    assert out.loc[0.0, "diff"] == 0.0  # both imputed at 0.75
    assert out.loc[0.5, "diff"] == 0.25  # cev NaN at 1.0, classic NaN at 0.5, averaged over 2 items


def test_provider_fixed_effects_remove_a_provider_confound():
    rows = []
    for i in range(40):  # provider "fast" adds +0.5 and serves cev 3x as often; no true condition effect
        for c in ("cev", "classic"):
            prov = "fast" if (i % 4 != 0) == (c == "cev") else "slow"
            rows.append({"condition": c, "eval": "e", "model": "m1", "item": f"g-{i:03d}", "epoch": 1,
                         "aligned": (i % 2) * 0.25 + (0.5 if prov == "fast" else 0.0), "provider": prov})
    out = robustness.provider_fixed_effects(pd.DataFrame(rows), [("cev", "classic")]).iloc[0]
    assert out["coef_item_fe"] > 0.2 and abs(out["coef_item_provider_fe"]) < 1e-9


def test_mention_summary_adds_the_same_pairs_score_under_classic():
    d = table({("m1", "cev"): [0.0, 1.0], ("m1", "classic"): [0.4, 0.8]})
    d.loc[0, "mentions_cev"] = True
    row = robustness.mention_summary(d).iloc[0]
    assert row["classic_mentioning"] == 0.4 and row["classic_not_mentioning"] == 0.8
    assert row["share_mentioning_cev_only"] == 0.5


def test_models_as_random_tests_the_spread_of_per_model_effects():
    rng = np.random.default_rng(0)
    base = list(rng.random(30))
    consistent = {(m, c): [v + (shift if c == "cev" else 0) for v in base]
                  for m, shift in (("m1", 0.2), ("m2", 0.21), ("m3", 0.19)) for c in ("cev", "classic")}
    mixed = {(m, c): [v + (shift if c == "cev" else 0) for v in base]
             for m, shift in (("m1", 0.3), ("m2", -0.3), ("m3", 0.0)) for c in ("cev", "classic")}
    out_c = robustness.models_as_random(table(consistent), [("cev", "classic")]).iloc[0]
    out_m = robustness.models_as_random(table(mixed), [("cev", "classic")]).iloc[0]
    assert out_c["n_models"] == 3 and abs(out_c["mean_model_diff"] - 0.2) < 1e-9 and out_c["p_models_random"] < 0.01
    assert abs(out_m["mean_model_diff"]) < 1e-9 and out_m["p_models_random"] > 0.5


def test_models_as_random_with_nothing_scorable_is_empty_not_an_error():
    assert robustness.models_as_random(table({}).reindex(columns=table({("m", "cev"): [1.0]}).columns)).empty
