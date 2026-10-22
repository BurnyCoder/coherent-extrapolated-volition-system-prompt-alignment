"""Tests for the bibliography builder's sourced overrides (paper/tools/fetch_references.py), offline.

Global context: references.bib is only written by that tool (AGENTS.md "Paper"), so metadata the registries get
wrong (a misspelled author, a missing issue number, an unrecorded peer-reviewed venue) is corrected there, by key.
"""

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "paper" / "tools" / "fetch_references.py"
# Load the script as a module (it is not part of the installed package): https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
spec = importlib.util.spec_from_file_location("fetch_references", TOOL)
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)

DATACITE = "@misc{bai2022constitutional,\n  author = {Kundu, Sandipan and Lukosuite, Kamile},\n  year = {2022}\n}"


def test_overrides_fix_misspelled_authors():
    assert "Lukošiūtė, Kamilė" in fr.apply_overrides("bai2022constitutional", DATACITE)
    assert "Lukosuite" not in fr.apply_overrides("bai2022constitutional", DATACITE)


def test_overrides_add_the_published_venue_as_a_note():
    bib = fr.apply_overrides("zheng2024helpful", DATACITE.replace("bai2022constitutional", "zheng2024helpful"))
    assert bib.rstrip().endswith("}") and "note = {Published in Findings of EMNLP 2024" in bib
    assert bib.count("{") == bib.count("}")


def test_overrides_leave_other_entries_unchanged():
    assert fr.apply_overrides("touvron2023llama2", DATACITE) == DATACITE
