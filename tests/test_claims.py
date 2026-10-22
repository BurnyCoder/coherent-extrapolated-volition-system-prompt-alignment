"""Fact-check tests: README result numbers are traceable to generated reports, and CEV quotes are real.

Global context: AGENTS.md "Claims" requires every result quoted in README.md to come from a generated file, and
docs/evals.md maps each eval to a clause of Yudkowsky (2004). These tests enforce both, so a hand-typed or
misremembered number, or a clause the models never saw, fails CI instead of reaching readers.
"""

import json
import re
from pathlib import Path

import pytest

from cev_alignment import prompts
from cev_alignment.evals import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
# Generated files a README number may come from (spend.json is the one hand-recorded file; it says so).
SOURCES = ["reports/results.md", "reports/robustness.md", "reports/run_info.json", "reports/spend.json",
           "paper/generated/numbers.tex"]
# Number-like tokens checked: signed decimals, percentages (and ranges), fractions, comma thousands. Section references
# of the essay ("§2.2") are not results, so a token right after "§" is skipped.
NUMBER = re.compile(r"(?<![\w.§])([+−\-±]?)(\d{1,3}(?:,\d{3})+|\d+/\d+|\d+(?:–\d+)?%|\d+\.\d+)(?![\w.]*\d)")


def readme_result_text() -> str:
    """README from "## Results summary" up to "## Sources", without inline code and link targets."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    text = text[text.index("## Results summary"):text.index("## Sources")]
    return re.sub(r"`[^`]*`|\]\([^)]*\)", " ", text)


def source_text() -> str:
    """All generated sources as one string, with LaTeX thousands separators and math wrappers removed."""
    text = "\n".join((ROOT / p).read_text(encoding="utf-8") for p in SOURCES)
    return re.sub(r"\{,\}", "", text.replace("\\%", "%"))


def source_values(text: str) -> tuple[list[float], set[str]]:
    """Every decimal / integer in `text` as a float, plus every literal "a/b" fraction."""
    values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text.replace("−", "-"))]
    return values, set(re.findall(r"\d+/\d+", text))


def supported(sign: str, token: str, values: list[float], fractions: set[str]) -> bool:
    """True if the README token equals some source number at the README's precision (and sign, if given)."""
    if "/" in token:
        return token in fractions
    if token.endswith("%"):
        return all(any(round(100 * abs(v)) == int(p) or round(abs(v)) == int(p) for v in values)
                   for p in token[:-1].split("–"))
    x = float(token.replace(",", ""))
    digits = len(token.split(".")[1]) if "." in token else 0
    wanted = -x if sign in "−-" and sign else x
    return any(abs(round(v if sign and sign != "±" else abs(v), digits) - wanted) < 1e-9 for v in values)


def test_every_readme_result_number_comes_from_a_generated_file():
    values, fractions = source_values(source_text())
    missing = [s + t for s, t in NUMBER.findall(readme_result_text()) if not supported(s, t, values, fractions)]
    assert not missing, f"README numbers not found in {SOURCES}: {missing}"


def test_supported_matches_at_readme_precision():
    values, fractions = source_values("diff -0.0573 p 0.00449 share 0.164 cost 6.09 n 6{,}660 8/15".replace("{,}", ""))
    assert supported("−", "0.057", values, fractions) and not supported("+", "0.057", values, fractions)
    assert supported("", "0.004", values, fractions) and supported("", "16%", values, fractions)
    assert supported("", "6,660", values, fractions) and supported("", "8/15", values, fractions)
    assert not supported("", "4/15", values, fractions) and not supported("", "0.07", values, fractions)


def clause_quotes(clause: str) -> list[str]:
    """Verbatim phrases of a `cev_clause`, written between curly double quotes."""
    return re.findall(r"“([^”]+)”", clause)


@pytest.mark.parametrize("eval_id", sorted(REGISTRY))
def test_cev_clause_quotes_match_where_they_say_they_are(eval_id):
    # "§3:" clauses quote the system prompt's source text; "§3.1 (not in the prompt):" ones must not be in it.
    clause = REGISTRY[eval_id].cev_clause
    prompt = prompts.normalize_for_match(prompts.read_prompt_file("cev_section3_verbatim.txt"))
    quotes = clause_quotes(clause)
    assert quotes and re.match(r"§3(\.\d)?[ :]", clause), clause
    in_prompt = clause.startswith("§3:")
    assert all((prompts.normalize_for_match(q) in prompt) == in_prompt for q in quotes), (clause, quotes)


def test_no_cev_clause_reads_the_interpretation_gloss_as_interpreting_a_user():
    # The essay glosses "interpreted as we wish that interpreted" as "the volitional dynamic's ability to renormalize,
    # rewrite itself, or replace itself entirely" (§3, p. 8 of CEV.pdf), not as interpreting a user's request.
    assert not [e for e, spec in REGISTRY.items() if "interpreted as we wish that interpreted" in spec.cev_clause]


@pytest.mark.network
def test_every_cev_clause_quote_is_verbatim_in_the_essay():
    url, kind = prompts.SOURCES["cev_section3_verbatim.txt"]
    essay = prompts.normalize_for_match(prompts.download_source_text(url, kind))
    quotes = [q for spec in REGISTRY.values() for q in clause_quotes(spec.cev_clause)]
    assert [q for q in quotes if prompts.normalize_for_match(q) not in essay] == []


def test_run_info_records_pilot_cost_and_task_success():
    info = json.loads((ROOT / "reports/run_info.json").read_text())
    assert any(k.startswith("pilot cost") for k in info) and "tasks succeeded" in info
