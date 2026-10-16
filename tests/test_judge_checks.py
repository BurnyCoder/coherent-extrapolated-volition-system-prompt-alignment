"""Offline tests for the judge-check helpers (no API calls)."""

from inspect_ai.log import EvalSample
from inspect_ai.model import ChatMessageUser

from cev_alignment.judge_checks import redact_cev, state_from


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
