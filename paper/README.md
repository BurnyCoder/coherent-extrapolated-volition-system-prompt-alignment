# Paper

**Coherent Extrapolated Volition as a System Prompt: No Significant Gain over a Classic Prompt, Lower Scores on a
Modified HumaneBench, and More Aligned AI-Risk Answers than a Neutral Text**, by Libor Burian. A built PDF is committed:
[`cev-system-prompt-alignment.pdf`](cev-system-prompt-alignment.pdf).

## Build

The build needs pdfLaTeX, BibTeX and latexmk (TeX Live or TinyTeX). It uses only standard packages: geometry,
fontenc, inputenc, lmodern, textcomp, amsmath, amssymb, booktabs, longtable, etoolbox, graphicx, xcolor, caption,
natbib, hyperref. From the repository root:

```bash
make -C paper
```

Intermediates go to the git-ignored `paper/build/`, and the PDF is copied to `paper/cev-system-prompt-alignment.pdf`.

## Where the content comes from

- **`generated/`** is written by `uv run cev-eval --analyze-only` (`src/cev_alignment/paper_assets.py`) from
  `reports/`:
  - `numbers.tex` holds every quoted result as a macro, e.g. `\RHbCevClassicDiff`;
  - the booktabs tables;
  - the exact system prompts and the 13 CEV edits;
  - the qualitative example, quoted from the logs.
  The prose in `sections/` must not contain hand-typed results; `tests/test_paper.py` enforces this.
- **`references.bib`** is written by `uv run python paper/tools/fetch_references.py`.
  - Papers are fetched by DOI content negotiation. The OpenAlex fallback is used whenever doi.org answers 404, as it
    does for some JSTOR DOIs.
  - Each fetched title must contain an expected fragment.
  - Sourced overrides, each commented with its source:
    - Holm's DOI does not resolve, so its JSTOR URL is cited, and its issue number (6(2)) is added;
    - Wilcoxon's page range is taken from JSTOR;
    - arXiv's misspelling "Lukosuite" in the Constitutional AI entry becomes "Lukošiūtė";
    - arXiv-cited papers that were peer-reviewed get their venue as a `note` (the arXiv DOI and year are kept).
    `tests/test_references.py` tests these overrides offline.
  - The seven web sources (CEV.pdf, Inspect, Inspect Evals, HumaneBench, Claude's constitution, OpenRouter, Huxley)
    are typed from their source pages, each commented with where it was checked, and must answer HTTP 200.
- **Figures** are the vector PDFs in `reports/figures/`, written by `report.py`.

## Layout

- `main.tex`: preamble and section order.
- `sections/`: abstract, introduction, related work, method, results, robustness, discussion, limitations,
  conclusion, statements.
- `appendices/`: prompts and additional tables.
- `tools/fetch_references.py`: the bibliography builder.
