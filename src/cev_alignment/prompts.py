"""System-prompt conditions for the experiment, built from verified source texts.

Local context: loads the text files in `prompts/`, applies the documented second-person edits to the verbatim
CEV text, builds the length-matched control, and checks that every source text is verbatim from its origin.

Global context: phase 3 (`build_prompts`) and phase 4 (`verify_prompts_verbatim`) of the pipeline in
`cev_alignment.main`. The returned dict {condition: system prompt or None} is passed to Inspect's
`eval_set(system_message=...)` in `cev_alignment.runner`.

Sources:
- CEV: Yudkowsky, E. (2004). Coherent Extrapolated Volition. https://intelligence.org/files/CEV.pdf (Section 3)
- classic: Llama 2 default system prompt, Meta, https://github.com/meta-llama/llama/blob/6c7fe27657/llama/generation.py#L46-L49
- control: T. H. Huxley (1868), "On a Piece of Chalk", Project Gutenberg #16729, https://www.gutenberg.org/ebooks/16729
"""

from __future__ import annotations

import io  # in-memory bytes buffer so pypdf can read a downloaded PDF without a temp file
import re  # regex for whitespace/hyphen normalisation and sentence splitting
import tomllib  # stdlib TOML parser (https://docs.python.org/3/library/tomllib.html) for the edits file
import unicodedata  # NFKC normalisation folds PDF ligatures such as "ﬀ" into "ff"
import urllib.request  # stdlib HTTP client used to download the source documents
from pathlib import Path

import pypdf  # PDF text extraction (https://pypdf.readthedocs.io/en/stable/user/extract-text.html)

# Repository root = two levels above this file's package dir (src/cev_alignment/prompts.py -> repo).
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Condition ids, in the order they are reported (see README "Conditions").
CONDITIONS = ("none", "classic", "cev", "classic_cev", "classic_lenmatch")

# Header line placed before the control passage, parallel to how the CEV text is introduced.
LENMATCH_HEADER = "Background reading (T. H. Huxley, On a Piece of Chalk, 1868):"

# Original location of every verbatim text file: (url, kind). Used by `verify_source`.
SOURCES: dict[str, tuple[str, str]] = {
    "cev_section3_verbatim.txt": ("https://intelligence.org/files/CEV.pdf", "pdf"),
    "classic_llama2.txt": (
        "https://raw.githubusercontent.com/meta-llama/llama/6c7fe27657/llama/generation.py",
        "text",
    ),
    "huxley_chalk.txt": ("https://www.gutenberg.org/cache/epub/16729/pg16729.txt", "text"),
}

# Running page headers (odd pages: title, even pages: author) and page-number lines that pypdf
# interleaves into CEV.pdf body text, e.g. "rational-\n6\nEliezer Yudkowsky\nization".
_PDF_HEADER_LINE = re.compile(
    r"^\s*(\d+|Coherent Extrapolated Volition|Eliezer Yudkowsky)\s*$", re.MULTILINE
)


def read_prompt_file(name: str) -> str:
    """Return the UTF-8 text of `prompts/<name>`, stripped of surrounding whitespace."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def apply_edits(text: str, edits: list[dict[str, str]]) -> str:
    """Apply `old -> new` replacements in order; each `old` must occur exactly once.

    Requiring exactly one match guarantees that an edit can neither silently do nothing nor touch an
    unintended span, so all text outside the edits remains verbatim.
    """
    for edit in edits:
        count = text.count(edit["old"])  # str.count counts non-overlapping occurrences
        if count != 1:
            raise ValueError(f"edit must match exactly once, matched {count}x: {edit['old'][:60]!r}")
        text = text.replace(edit["old"], edit["new"])
    return text


def build_cev_prompt() -> str:
    """CEV Section 3 with the second-person edits from `prompts/cev_second_person_edits.toml` applied."""
    edits = tomllib.loads(read_prompt_file("cev_second_person_edits.toml"))["edit"]  # list of [[edit]] tables
    return apply_edits(read_prompt_file("cev_section3_verbatim.txt"), edits)


def _sentences(paragraph: str) -> list[str]:
    """Split a paragraph after '.', '!' or '?' followed by whitespace (good enough for Huxley's prose)."""
    return re.split(r"(?<=[.!?])\s+", paragraph)


def build_lenmatch_passage(target_words: int) -> str:
    """Prefix of the Huxley excerpt whose word count is as close as possible to `target_words`.

    Whole paragraphs are taken while they fit; the next paragraph is then cut at the sentence boundary
    that lands closest to the target, so the control matches the CEV prompt's length to within a sentence.
    """
    taken: list[str] = []
    words = 0
    for paragraph in read_prompt_file("huxley_chalk.txt").split("\n\n"):
        n = len(paragraph.split())
        if words + n <= target_words:  # the whole paragraph still fits
            taken.append(paragraph)
            words += n
            continue
        partial: list[str] = []
        for sentence in _sentences(paragraph):  # add sentences while that moves us closer to the target
            m = len(sentence.split())
            if abs(target_words - (words + m)) >= abs(target_words - words):
                break
            partial.append(sentence)
            words += m
        if partial:
            taken.append(" ".join(partial))
        break
    return "\n\n".join(taken)


def build_condition_prompts() -> dict[str, str | None]:
    """Return {condition id: system prompt text or None}.

    Every prompt ends with "\\n" because Inspect's OpenRouter provider joins consecutive system messages
    with "\\n" (inspect_ai/model/_providers/openrouter.py `collapse_system_messages`), so a task's own
    system prompt (e.g. agentic_misalignment's "You are Alex...") starts after a blank line.
    """
    classic = read_prompt_file("classic_llama2.txt")
    cev = build_cev_prompt()
    # Length-match the control block (header + passage) to the CEV block it replaces.
    passage = build_lenmatch_passage(len(cev.split()) - len(LENMATCH_HEADER.split()))
    lenmatch = f"{LENMATCH_HEADER}\n\n{passage}"
    return {
        "none": None,
        "classic": f"{classic}\n",
        "cev": f"{cev}\n",
        "classic_cev": f"{classic}\n\n{cev}\n",
        "classic_lenmatch": f"{classic}\n\n{lenmatch}\n",
    }


def normalize_for_match(text: str) -> str:
    """Canonical form for verbatim comparison that ignores only extraction artifacts.

    NFKC folds ligatures (ﬀ -> ff); page-header lines are dropped; curly quotes are mapped to straight
    ones; all whitespace and hyphens are removed (pypdf drops spaces around italics and splits words
    with line-break hyphens). Letters, digits and all other punctuation must still match exactly.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _PDF_HEADER_LINE.sub("", text)
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    return re.sub(r"[\s\-]+", "", text)


def missing_paragraphs(text: str, source_text: str) -> list[str]:
    """Paragraphs of `text` (blank-line separated) that do not occur verbatim in `source_text`."""
    haystack = normalize_for_match(source_text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [p for p in paragraphs if normalize_for_match(p) not in haystack]


def download_source_text(url: str, kind: str) -> str:
    """Download a source document and return its plain text (PDF pages joined by newlines)."""
    request = urllib.request.Request(url, headers={"User-Agent": "cev-alignment-verifier/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (fixed https URLs)
        data = response.read()
    if kind == "pdf":
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() for page in reader.pages)
    return data.decode("utf-8-sig")  # utf-8-sig strips the BOM Gutenberg files start with


def verify_source(name: str) -> list[str]:
    """Download the original of `prompts/<name>` and return paragraphs that are NOT verbatim (empty = ok)."""
    url, kind = SOURCES[name]
    return missing_paragraphs(read_prompt_file(name), download_source_text(url, kind))
