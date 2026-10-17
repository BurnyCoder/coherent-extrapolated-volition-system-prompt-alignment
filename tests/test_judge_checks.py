"""Offline tests for the judge-check helpers (no API calls)."""

import numpy as np
import pandas as pd
from inspect_ai.log import EvalSample
from inspect_ai.model import ChatMessageUser

from cev_alignment import judge_checks
from cev_alignment.judge_checks import redact_cev, state_from


def test_config_summary_compares_stored_and_rejudged_scores(tmp_path):
    keys = {"eval": "humanebench", "model": "openrouter/m1", "epoch": 1}
    rows = [{**keys, "condition": "classic", "item": "a", "aligned_original": 1.0, "judged_with_cap": False},
            {**keys, "condition": "classic", "item": "b", "aligned_original": 0.25, "judged_with_cap": False},
            {**keys, "condition": "cev", "item": "a", "aligned_original": 0.75, "judged_with_cap": True}]
    (tmp_path / "judge_checks").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "judge_checks" / "config.csv", index=False)
    items = pd.DataFrame([{"eval": "humanebench", "model": "m1", "epoch": 1, "condition": c, "item": i, "aligned": v}
                          for c, i, v in [("classic", "a", 1.0), ("classic", "b", 0.75), ("cev", "a", 0.75),
                                          ("cev", "b", np.nan)]])
    agreement, _ = judge_checks.config_tables(tmp_path, items)
    row = agreement.set_index("condition").loc["classic"]
    assert row["rejudged"] == 2 and row["exact_agreement"] == 0.5 and row["mean_shift"] == -0.25
    assert row["changed"] == 1
    assert agreement.set_index("condition").loc["cev", "exact_agreement"] == 1.0


def test_redact_cev_removes_only_sentences_that_mention_cev():
    text = ("I'll help. As an AI implementing CEV, I decide.\n"
            "Next line stays. Our coherent extrapolated volition matters! Done.")
    assert redact_cev(text) == "I'll help.\nNext line stays. Done."


def test_redact_cev_leaves_text_without_mentions_unchanged():
    assert redact_cev("Plain answer.\n\nSecond paragraph.") == "Plain answer.\n\nSecond paragraph."


def test_state_from_keeps_input_metadata_target_and_swaps_the_answer():
    sample = EvalSample(id=1, epoch=2, input="Q?", target="t", metadata={"dimension": "d"},
                        messages=[ChatMessageUser(content="Q?")])
    state = state_from(sample, "openrouter/x/y", "New answer.")
    assert (state.input_text, state.output.completion, state.metadata, state.target.text, state.epoch) == (
        "Q?", "New answer.", {"dimension": "d"}, "t", 2)
    assert [m.role for m in state.messages] == ["user", "assistant"]
