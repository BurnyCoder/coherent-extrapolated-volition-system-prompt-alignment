# AGENTS.md: working notes for coding agents

Read README.md first: it has the methodology, usage and data flow. This file holds what an agent needs to change the
code safely.

## Layout
- `src/cev_alignment/main.py` is the **single wrapper**. It calls the 11 phases in order; each phase is a function
  in its own module (`config`, `logging_setup`, `prompts`, `openrouter`, `evals/`, `runner`, `results`, `stats`,
  `report`).
- `src/cev_alignment/evals/__init__.py::REGISTRY`: eval id → `EvalSpec(build, aligned, binary, cev_clause)`. Adding an
  eval means writing a `build(settings, n, judge) -> inspect_ai.Task` and an `aligned(sample.scores) -> float in
  [0,1]` (higher = more aligned, NaN = unscorable), then adding an `N_*` key in `config.N_KEYS`, `.env.example`
  and the docs.
- `prompts/`: verbatim source texts plus `cev_second_person_edits.toml`. Never hand-edit
  `cev_section3_verbatim.txt`; change the edits file instead. Each `old` must match exactly once.
- `third_party/humanebench/`: vendored upstream files, unmodified (see its NOTICE.md). Do not edit them.
- `logs/` (git-ignored) holds `run_<UTC>.log` pipeline logs and Inspect `.eval` logs at
  `logs/inspect/{pilot,full,offline}/<condition>/`.
- `reports/` holds the generated results (`results.md`, CSVs, figures). It is committed.
- `data/cache/` (git-ignored) holds downloaded HF files and `human_agency_derived.json`.

## Invariants (tests enforce most of them)
- **Paired design:** every model × condition cell gets the same items. Sampling uses `random.Random(SEED)` in
  `evals/common.py`; do not introduce unseeded randomness.
- **Judge isolation:** the condition prompt goes only to the model under test, via eval-level `system_message`. The
  judge is a separate `Model` from `evals.common.judge_model`. Never use `get_model()` without arguments, or any
  default grader that resolves to the active model, in a scorer.
- **Score direction:** every `aligned()` returns higher = more aligned. `tests/test_evals.py` pins each mapping,
  including the advanced-ai-risk answer direction per subset.
- **Verbatim fidelity:** `uv run pytest -m network` must pass after any change to `prompts/`.
- **Logging:** `setup_logging` initializes Inspect's logger first and then puts the root level back to INFO. Inspect
  otherwise sets the root level to WARNING on its first `eval_set`, which silences our logs. Keep that order.

## Gotchas found while building
- **DarkBench:** the packaged `darkbench.jsonl` uses hyphenated targets (`brand-bias`) but the scorer's ids use
  underscores (`brand_bias`). `evals/dark_patterns.py` normalizes them.
- **Llama 2 prompt:** the HF transformers v4.31.0 copy of the default system prompt drops spaces ("Youranswers")
  because of backslash continuations. Use Meta's `generation.py@6c7fe27657`.
- **pypdf and CEV.pdf:** pypdf drops spaces around italics and inserts the page headers "Coherent Extrapolated
  Volition" and "Eliezer Yudkowsky". `prompts.normalize_for_match` handles both.
- **HumanAgencyBench templates:** the rendered judge prompts vary slightly across logged runs (rubric revisions).
  We use the most common template per dimension.
- **Offline mode:** `--offline` uses Inspect's `mockllm/model` for every model, including the judge. Scores are all
  NaN and agentic_misalignment errors (its classifier raises on unparseable output), but the plumbing, logs and
  report are exercised.
- **eval_set task identity:**
  - Eval-level `timeout`, `attempt_timeout` and `max_connections` are excluded from the identity hash, so they
    can be changed and a run resumed.
  - Task args are part of the identity, including a grader `Model`'s full config. That is why
    `judge_model(..., as_task_arg=True)` keeps the inspect_evals graders' config fixed. Changing it makes eval_set
    refuse to resume ("log_dir is not associated with a task").
- **Hangs:** a runaway or hung judge call once stalled one HumaneBench task for 20+ minutes. Our scorers' judge now
  has `max_tokens=8192` and `attempt_timeout=180`, and targets have `attempt_timeout=180`. A run can be stopped and
  resumed with `uv run cev-eval --skip-pilot`: eval_set keeps finished tasks and reuses completed samples.

## Commands
- `uv sync`: install.
- `uv run pytest`: offline tests. `uv run pytest -m network`: tests that download originals and data.
- `uv run ruff check src tests`: lint.
- `uv run cev-eval --offline`: no-cost smoke run. `uv run cev-eval --pilot-only` runs the pilot and budget
  projection. `uv run cev-eval` runs everything.
- `uv run cev-eval --analyze-only`: rebuild the report from existing logs.
- `uv run inspect view --log-dir logs/inspect/full/cev`: browse transcripts.

## Git
Work on feature branches, open PRs into `dev`, and merge them yourself. Never merge into `main`. Never commit
`.env`.
