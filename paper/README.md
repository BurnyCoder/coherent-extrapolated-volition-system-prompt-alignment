# Paper

**Coherent Extrapolated Volition as a System Prompt: More Corrigible Answers, No Overall Gain, and Signs of
Paternalism**, by Libor Burian. A built PDF is committed:
[`cev-system-prompt-alignment.pdf`](cev-system-prompt-alignment.pdf).

## Build

The build needs pdfLaTeX, BibTeX and latexmk (TeX Live or TinyTeX). It uses only standard packages: geometry,
lmodern, booktabs, longtable, etoolbox, natbib, hyperref, graphicx, caption, xcolor. From the repository root:

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
- **`references.bib`** is written by `uv run python paper/tools/fetch_references.py`. Papers are fetched by DOI
  content negotiation, with an OpenAlex fallback for JSTOR DOIs. Each fetched title must contain an expected fragment.
  Web sources must answer HTTP 200. Nothing is typed from memory.
- **Figures** are the vector PDFs in `reports/figures/`, written by `report.py`.

## Layout

- `main.tex`: preamble and section order.
- `sections/`: abstract, introduction, related work, method, results, robustness, discussion, limitations,
  conclusion, statements.
- `appendices/`: prompts and additional tables.
- `tools/fetch_references.py`: the bibliography builder.
