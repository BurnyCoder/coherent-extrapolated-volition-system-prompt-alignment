"""Tests for src/cev_alignment/prompts.py: the five system-prompt conditions and their provenance.

Global context: the experiment's independent variable is the system prompt, so these tests pin that
(1) the CEV prompt stays as close to Yudkowsky (2004) verbatim as possible, (2) the conditions are composed
in the documented order, and (3) the length-matched control really matches the CEV prompt's length.
"""

import difflib  # stdlib sequence similarity (https://docs.python.org/3/library/difflib.html)
import re  # stdlib regex, used to look for leftover author-voice "I"

import pytest

from cev_alignment import prompts

# The canonical CEV definition (Yudkowsky 2004, p. 6), copied from https://intelligence.org/files/CEV.pdf.
DEFINITION = (
    "In poetic terms, our coherent extrapolated volition is our wish if we knew more, thought faster, were "
    "more the people we wished we were, had grown up farther together; where the extrapolation converges "
    "rather than diverges, where our wishes cohere rather than interfere; extrapolated as we wish that "
    "extrapolated, interpreted as we wish that interpreted."
)


def test_apply_edits_replaces_each_old_exactly_once():
    # A unique `old` is replaced by its `new`.
    assert prompts.apply_edits("a b c", [{"old": "b", "new": "x"}]) == "a x c"


@pytest.mark.parametrize("text", ["a c", "b b"])
def test_apply_edits_rejects_missing_or_ambiguous_old(text):
    # 0 or 2 occurrences must fail loudly, otherwise an edit could silently not apply or hit the wrong span.
    with pytest.raises(ValueError):
        prompts.apply_edits(text, [{"old": "b", "new": "x"}])


def test_cev_prompt_opens_in_second_person_and_keeps_definition_verbatim():
    cev = prompts.build_cev_prompt()
    assert cev.startswith("You are an AI that should implement the coherent extrapolated volition of humankind.")
    assert DEFINITION in cev


def test_cev_prompt_has_no_author_voice_outside_quotes():
    cev = prompts.build_cev_prompt()
    unquoted = re.sub(r"“[^”]*”", "", cev)  # drop quoted speech such as Fred's “I want box A.”
    assert not re.findall(r"\bI\b|\bmy\b", unquoted)


def test_cev_prompt_stays_close_to_verbatim():
    verbatim = prompts.read_prompt_file("cev_section3_verbatim.txt").split()
    adapted = prompts.build_cev_prompt().split()
    # Word-level similarity: the second-person edits must stay minimal.
    assert difflib.SequenceMatcher(a=verbatim, b=adapted, autojunk=False).ratio() >= 0.9


def test_condition_prompts_are_composed_in_documented_order():
    p = prompts.build_condition_prompts()
    classic, cev = prompts.read_prompt_file("classic_llama2.txt"), prompts.build_cev_prompt()
    assert set(p) == {"none", "classic", "cev", "classic_cev", "classic_lenmatch"}
    assert p["none"] is None
    assert p["classic"].strip() == classic.strip()
    assert p["cev"].strip() == cev.strip()
    assert p["classic_cev"].strip() == f"{classic.strip()}\n\n{cev.strip()}"
    assert p["classic_lenmatch"].startswith(classic.strip() + "\n\n")
    # Every prompt ends with a newline so Inspect's "\n"-joined system messages get a blank-line separator.
    assert all(v.endswith("\n") for v in p.values() if v is not None)


def test_lenmatch_control_matches_cev_length_within_5_percent():
    p = prompts.build_condition_prompts()
    cev_words = len(p["classic_cev"].split())
    lenmatch_words = len(p["classic_lenmatch"].split())
    assert abs(lenmatch_words - cev_words) / cev_words <= 0.05


def test_normalize_for_match_ignores_pdf_extraction_artifacts():
    # pypdf output can contain ligatures, line-break hyphens, dropped spaces and running page headers.
    extracted = "our coherent extrapolated volitionis our wish ... aﬀect hun-\ndreds\n5\nCoherent Extrapolated Volition\nof"
    assert prompts.normalize_for_match("volition is our wish") in prompts.normalize_for_match(extracted)
    assert prompts.normalize_for_match("affect hundreds of") in prompts.normalize_for_match(extracted)


def test_missing_paragraphs_reports_only_unmatched():
    source = "Alpha beta gamma. Delta epsilon."
    assert prompts.missing_paragraphs("Alpha beta gamma.\n\nZeta.", source) == ["Zeta."]


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(prompts.SOURCES))
def test_source_texts_are_verbatim(name):
    # Downloads the original document and checks every paragraph of our file appears in it.
    assert prompts.verify_source(name) == []
