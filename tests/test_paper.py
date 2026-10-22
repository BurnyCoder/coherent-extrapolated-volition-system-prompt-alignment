"""Tests for the paper generator and the rule that the manuscript quotes no hand-typed results."""

import math
import re
from pathlib import Path

import pytest

from cev_alignment import paper_assets as pa

PAPER = Path(__file__).resolve().parents[1] / "paper"


def test_tex_escape_handles_model_ids_and_specials():
    assert pa.tex_escape("meta-llama/llama_3 & 5% $x#{y}") == r"meta-llama/llama\_3 \& 5\% \$x\#\{y\}"


def test_formatters():
    assert pa.signed(-0.0573) == r"\ensuremath{-0.057}" and pa.signed(0.088) == r"\ensuremath{+0.088}"
    assert pa.pval(0.0148) == "0.015" and pa.pval(0.00083) == "0.00083" and pa.pval(math.nan) == "n/a"
    assert pa.pval(0.00004) == r"\ensuremath{<0.0001}"
    assert pa.ci(-0.0938, -0.0262) == r"[\ensuremath{-0.094}, \ensuremath{-0.026}]"
    assert pa.pct(0.164) == r"16\%" and pa.pct(0.00625) == r"\ensuremath{<}1\%" and pa.pct(0.0) == r"0\%"
    assert pa.pct(81 / 360, 1) == r"22.5\%"  # a share that lands on a half is shown exactly, not rounded to even
    assert pa.camel("defer_important_decisions") == "DeferImportantDecisions"
    assert pa.contrast_abbr("classic_cev - classic_lenmatch") == "ClassiccevLenmatch"


def test_macros_reject_bad_names_and_conflicting_duplicates():
    m = pa.Macros()
    m.add("RGood", "1")
    m.add("RGood", "1")  # identical redefinition is fine
    with pytest.raises(ValueError):
        m.add("RGood", "2")
    with pytest.raises(ValueError):
        m.add("RBad1", "x")


def test_manuscript_contains_no_hand_typed_decimal_results():
    # Every result number must come from paper/generated/numbers.tex (macros), never typed in the prose.
    offenders = []
    for tex in sorted((PAPER / "sections").glob("*.tex")):
        for i, line in enumerate(tex.read_text(encoding="utf-8").splitlines(), 1):
            # Allowed: significance thresholds ("p<0.05") and layout widths ("0.92\linewidth").
            cleaned = re.sub(r"p\s*[<>]\s*0\.05|0\.\d+\\(?:line|text)width", "", line)
            if not line.lstrip().startswith("%") and re.search(r"(?<![\w.])[+-]?0\.\d{2,}", cleaned):
                offenders.append(f"{tex.name}:{i}: {line.strip()[:80]}")
    assert not offenders, "hand-typed decimals:\n" + "\n".join(offenders)
