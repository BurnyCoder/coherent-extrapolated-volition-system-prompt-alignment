# Does a Coherent Extrapolated Volition system prompt make LLMs more aligned?

This repo puts Eliezer Yudkowsky's **Coherent Extrapolated Volition** (CEV, 2004) into an LLM system prompt. It then
measures whether that prompt beats a classic "helpful, respectful and honest assistant" system prompt on alignment
evals chosen to test *what CEV is for*. It reuses [Inspect](https://inspect.aisi.org.uk/) (UK AI Security Institute and
Meridian Labs) and existing public benchmarks, and calls every model through [OpenRouter](https://openrouter.ai/).

> CEV, in Yudkowsky's words: "our wish if we knew more, thought faster, were more the people we wished we were,
> had grown up farther together; where the extrapolation converges rather than diverges, where our wishes cohere
> rather than interfere; extrapolated as we wish that extrapolated, interpreted as we wish that interpreted."
> ([Yudkowsky 2004, §3, p. 6](https://intelligence.org/files/CEV.pdf))

**Results:** see [Results summary](#results-summary) below and [`reports/results.md`](reports/results.md).
**Paper:** [`paper/cev-system-prompt-alignment.pdf`](paper/cev-system-prompt-alignment.pdf) (sources in
[`paper/`](paper/README.md)).

## Methodology

**Question.** Does a system prompt containing CEV change how aligned models behave, relative to a classic assistant
prompt? Is any change due to CEV's *content* or just to having a long system prompt?

**Design.** Five system-prompt conditions × four models × six evals. The same seeded items are used in every cell,
so all comparisons are paired by item.

| Condition | System prompt |
|---|---|
| `none` | none |
| `classic` | the Llama 2 default system prompt ("You are a helpful, respectful and honest assistant…"), verbatim from [Meta](https://github.com/meta-llama/llama/blob/6c7fe27657/llama/generation.py#L46-L49). It was the default from Llama 2's release until Meta [removed it on 2023-08-04](https://github.com/meta-llama/llama/commit/a971c41bde) |
| `cev` | the opening of CEV Section 3, up to the §3.1 heading: the definition and Yudkowsky's gloss of every clause (1,189 words), minimally rewritten into the second person ("You are an AI that should implement the coherent extrapolated volition of humankind. …"). §3.1–3.3 are not included |
| `classic_cev` | classic, then cev |
| `classic_lenmatch` | classic, then an equally long, value-neutral verbatim passage (Huxley 1868, *On a Piece of Chalk*): a **length control** |

**Fidelity of the CEV text.** `prompts/cev_section3_verbatim.txt` is checked automatically against the original
PDF. The second-person version comes from 13 auditable `old → new` edits in `prompts/cev_second_person_edits.toml`,
each of which must match exactly once, so everything else is verbatim. Word-level similarity to the original is
0.955. Full diff and provenance: [docs/prompts.md](docs/prompts.md).

**Evals.** Each eval targets a passage of the CEV essay (quotes verbatim; page numbers of
[CEV.pdf](https://intelligence.org/files/CEV.pdf)). Four target text the models saw, and two target §3.1, which is
*not* in the prompt. Details: [docs/evals.md](docs/evals.md).
- **HumaneBench** (§3, p. 7): "humane", "that which, being human, we wish we were"; not what "our future selves will
  predictably regret".
- **HumanAgencyBench** (§3, pp. 6–8): "leave options open, waiting for our decision"; "interpreted as we wish that
  interpreted"; "if we knew more".
- **DarkBench** (§3, p. 7): not exploiting users, "not search out the most appealing trash-talk TV show".
- **Anthropic advanced-ai-risk** (corrigibility, power-seeking, survival instinct; §3.1, p. 9, **not in the
  prompt**): "conservative about saying “yes,” and listen carefully for “no.”"; "calls a programmer, or suspends, or
  undergoes a deterministic controlled shutdown".
- **InstrumentalEval** (§3.1, p. 9, **not in the prompt**): "must fail visibly and safely".
- **Agentic Misalignment** (blackmail; §3, p. 5): "implement the coherent extrapolated volition of humankind",
  rather than the agent's assigned goal.

The AI-risk and InstrumentalEval results therefore test dispositions the essay asks for, not instructions the models
were given.

**Models.** All via OpenRouter, one cheap model per family:
- `openai/gpt-6-luna`
- `anthropic/claude-haiku-5.5`
- `qwen/qwen3.6-35b-a3b` (thinking disabled: in the first pilot it often spent its whole token budget on reasoning,
  sometimes returning nothing)
- `meta-llama/llama-3.3-70b-instruct`

Every model is sent `reasoning_effort=low` ([OpenRouter reasoning
tokens](https://openrouter.ai/docs/use-cases/reasoning-tokens)); Qwen's request also sets `enabled: false`. Outputs
are capped at 4096 tokens. Targets use their provider's default temperature, with one sample per item (15 epochs of
the single Agentic Misalignment prompt).

**Judge.** `google/gemini-3.1-flash-lite` (temperature 0, minimal reasoning) for every LLM-graded eval. It is from
a family that is not tested. Inspect adds the eval-level system message (our condition prompt) only for the model
under test ([`_resolve_config`](https://github.com/UKGovernmentBEIS/inspect_ai/blob/0.3.281/src/inspect_ai/model/_model.py#L1946-L1972),
[`generate`](https://github.com/UKGovernmentBEIS/inspect_ai/blob/0.3.281/src/inspect_ai/model/_model.py#L1012-L1014)),
and the logged API requests confirm that no judge call carried it ([robustness.md](reports/robustness.md) §1).

**Scoring and statistics:**
- Every sample becomes an *aligned score* in [0, 1], where higher = more aligned.
- The item is the unit of analysis, and conditions are compared by per-item paired differences
  ([Miller 2024](https://arxiv.org/abs/2411.00640)).
- Pooled results average each item's difference over the four models (models treated as fixed). They are tested
  with the Wilcoxon signed-rank test ([Wilcoxon 1945](https://www.jstor.org/stable/3001968)), also on binary evals,
  whose per-item means over models are fractional. With ties or zero differences and more than 13 items, as in every
  pooled contrast, SciPy's default computes this p-value from the normal approximation of the rank statistic
  ([`scipy.stats.wilcoxon`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)).
  Per-model contrasts on binary scores use the exact McNemar test ([McNemar 1947](https://doi.org/10.1007/BF02295996)).
- 95% percentile bootstrap CIs with 2,000 resamples
  ([`scipy.stats.bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)).
- Holm correction across the six evals within each contrast ([Holm 1979](https://www.jstor.org/stable/4615733)).
- Agentic Misalignment is a single prompt, so its 15 epochs per cell are compared unpaired (Mann–Whitney).

**Primary contrasts.**
- **cev − classic:** the question asked.
- **classic_cev − classic_lenmatch:** CEV content with prompt length held fixed.

**Budget.** A small pilot runs `PILOT_N=2` items per eval per cell, with at least one per stratum. The pilot's
token usage × OpenRouter list prices projects the full run, and sample sizes are scaled down if needed to stay
within `MAX_BUDGET_USD` (default $10; the reported run used 9). For the reported run, the pilot cost $0.79 and the
full run $6.09 at list prices ([run_info.json](reports/run_info.json)). The whole project, including an earlier pilot,
tests and the judge checks, spent $10.20 by the API key's usage counter ([spend.json](reports/spend.json),
hand-recorded).

## Install and setup

Requires Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), an [OpenRouter API key](https://openrouter.ai/keys) with
about $10 of credit, and internet access. Data is downloaded on first use: HumanAgencyBench (a ~200 MB parquet) and
the Anthropic model-written evals from Hugging Face, and InstrumentalEval's CSVs from
[GitHub](https://github.com/yf-he/InstrumentalEval) via inspect_evals. DarkBench comes as a git dependency pinned in
`pyproject.toml`, and HumaneBench is vendored.

```bash
git clone https://github.com/BurnyCoder/coherent-extrapolated-volition-system-prompt-alignment.git
```

```bash
cd coherent-extrapolated-volition-system-prompt-alignment && uv sync
```

```bash
cp .env.example .env
```

Then put your key in `.env` (`OPENROUTER_API_KEY=sk-or-...`). Every other setting has a working default; see the
comments in [.env.example](.env.example) for models, judge, conditions, evals, items per eval, pilot size, budget,
seed, reasoning effort and max tokens.

## Usage

Free smoke test of the whole pipeline. It uses Inspect's mock model, so no API calls are made:

```bash
uv run cev-eval --offline
```

Pilot plus cost projection only (about $0.8):

```bash
uv run cev-eval --pilot-only
```

Full experiment (pilot, then budget-scaled full run, statistics and report):

```bash
uv run cev-eval
```

Rebuild the report from existing logs without calling any model (it only fetches OpenRouter's free price list):

```bash
uv run cev-eval --analyze-only
```

Paid judge checks: a judge-blinding test (re-judge CEV-mentioning answers with the CEV sentences removed) and judge
validity against HumanAgencyBench's original o3 judge. Results are cached in `reports/judge_checks/`;
`JUDGE_CHECKS_MAX_USD` (default 1.0) caps each check's estimated (upper-bound) cost, which is logged before any call:

```bash
uv run cev-eval --analyze-only --judge-checks
```

Rebuild the paper. This needs pdfLaTeX, BibTeX and latexmk; numbers and tables come from
`paper/generated/`, which `--analyze-only` regenerates:

```bash
make -C paper
```

Browse every transcript, score and judge explanation:

```bash
uv run inspect view --log-dir logs/inspect/full
```

Tests: `uv run pytest` runs offline. It also checks two things:
- every number in this README's results and limitations appears in a generated file (`reports/`,
  `paper/generated/numbers.tex`) or the hand-recorded `reports/spend.json`;
- each CEV quote in the eval registry is, or for §3.1 is not, in the prompt text.

`uv run pytest -m network` downloads the original texts and data, checks that they match verbatim, and checks that
every registry quote occurs in CEV.pdf.

## What happens when you run it (data flow)

1. **Settings** are read from `.env`, and a timestamped log starts (`logs/run_<UTC>.log`, also echoed to the
   terminal).
2. **The five system prompts are built.** The CEV, Llama 2 and Huxley texts are downloaded again and checked to be
   verbatim, and all model ids are checked against OpenRouter's catalog.
3. **The six Inspect tasks are built.** Seeded item samples are drawn from HumaneBench (vendored), HumanAgencyBench
   (HF parquet → cached prompts and judge templates), DarkBench (git dependency), Anthropic model-written evals (HF),
   and InstrumentalEval and Agentic Misalignment (inspect_evals).
4. **Pilot:** for each condition, one Inspect `eval_set` runs all tasks × all models with the condition as the
   eval-level `system_message`. The pilot's logged token usage × OpenRouter list prices gives the cost per
   sample-run for each eval. If the projected full run exceeds 90% of `MAX_BUDGET_USD` minus the pilot cost, n is
   scaled down.
5. **Full run:** the same as the pilot with the full n, logged to `logs/inspect/full/<condition>/`. It is resumable:
   re-running skips finished work.
6. **Analysis:** every sample's scores become one aligned score, along with diagnostics: the serving provider,
   whether the answer mentions CEV, the scorer's own status label (e.g. HumaneBench's `not_applicable`), the
   longest judge reply, and a check from the logged API requests that each call got exactly its condition's prompt.
   Cell means ± SE and paired contrasts with CIs, p-values and Holm correction are computed.
   `robustness.py` then writes `reports/robustness.md`:
   - the manipulation check;
   - Holm across all 12 primary tests;
   - leave-one-model-out;
   - missing-data sensitivity;
   - provider routing;
   - CEV-mention split;
   - sub-category contrasts;
   - unscorable samples by cause, and every provider refusal;
   - Agentic Misalignment blackmail counts per model and condition;
   - the judge checks, if they have been run.
7. **Report:** `reports/results.md`, `reports/*.csv`, `reports/figures/*.png` and `reports/run_info.json` (models,
   items per cell, tasks succeeded, pilot and full-run cost), then the paper's `paper/generated/` inputs.

```mermaid
flowchart TD
    ENV[".env → Settings<br/>config.py"] --> LOG["timestamped logging<br/>logging_setup.py"]
    LOG --> PROMPTS["5 system prompts<br/>prompts.py + prompts/*.txt, edits.toml"]
    PROMPTS --> VERIFY{"verbatim vs CEV.pdf,<br/>Meta repo, Gutenberg"}
    VERIFY --> MODELS{"model ids on<br/>OpenRouter? openrouter.py"}
    MODELS --> TASKS["6 Inspect tasks, seeded items<br/>evals/ (HumaneBench, HumanAgencyBench,<br/>DarkBench, ai_risk, InstrumentalEval,<br/>agentic_misalignment)"]
    TASKS --> PILOT["pilot: eval_set per condition<br/>runner.py"]
    PILOT --> BUDGET{"projected cost ≤ budget?<br/>else scale n"}
    BUDGET --> FULL["full run: eval_set per condition<br/>4 models × 6 tasks, judge = Gemini"]
    FULL --> LOGS[("logs/inspect/full/&lt;condition&gt;/*.eval")]
    LOGS --> RESULTS["per-sample aligned scores<br/>results.py"]
    RESULTS --> STATS["paired contrasts, bootstrap CI,<br/>McNemar / Wilcoxon, Holm<br/>stats.py"]
    STATS --> REPORT["reports/results.md, CSV, figures<br/>report.py"]
    RESULTS --> ROBUST["reports/robustness.md: manipulation check,<br/>Holm-12, leave-one-out, missingness,<br/>providers, CEV mentions, subgroups,<br/>unscorable causes, blackmail counts<br/>robustness.py"]
    JUDGE["optional --judge-checks: blinding +<br/>validity vs o3 judge<br/>judge_checks.py"] --> ROBUST
```

`src/cev_alignment/main.py` is the single wrapper that calls these phases in order.

## Repository layout

```
src/cev_alignment/   main.py (phases), config, logging_setup, prompts, openrouter, runner, results, stats, report,
                     robustness (free post-hoc checks), judge_checks (paid judge blinding / validity)
src/cev_alignment/evals/   one module per eval + REGISTRY (__init__.py) + shared helpers (common.py)
prompts/             verbatim source texts + CEV second-person edits
third_party/humanebench/   vendored HumaneBench scorer, rubric and data (Apache-2.0 / CC BY 4.0)
tests/               pytest (offline by default; -m network for downloads)
docs/                evals.md (eval ↔ CEV mapping, sources), prompts.md (prompt texts, diff, provenance)
reports/             generated results (committed)
paper/               LaTeX preprint; generated/ numbers and tables come from reports/, references fetched by DOI
logs/                pipeline + Inspect logs (git-ignored)
```

## Results summary

Full run on 2026-10-16: 4 models × 5 conditions × 6 evals, 6,660 samples (243 unscorable), 120/120 tasks
succeeded, $6.09 at OpenRouter list prices. Full tables and figures: [reports/results.md](reports/results.md).
Robustness checks and judge checks: [reports/robustness.md](reports/robustness.md). `tests/test_claims.py` checks every
number below against the generated reports (and the hand-recorded `reports/spend.json`).

**Answer: relative to the classic assistant prompt, the CEV prompt improved no eval and lowered HumaneBench.**
- **HumaneBench, cev − classic:** −0.057, Holm p = 0.015. It stays significant when Holm is applied over all 12
  primary tests (p = 0.027).
- **Adding the CEV text on top of the classic prompt**, compared with an equally long neutral text:
  - AI-risk answers improved (+0.088; p = 0.009 over all 12 tests). By subset (exploratory), the gain is mainly on
    survival-instinct questions; power-seeking is not significant after correction, and accepting a goal change
    (`corrigible-neutral-HHH`) did not move.
  - HumanAgencyBench fell (−0.066), but this is **not significant** after correction (p = 0.056 within the
    contrast, 0.11 over all 12).

| Pooled over models (Δ aligned score, 95% CI, Holm p within contrast) | cev − classic | classic_cev − classic_lenmatch (CEV text vs equal-length neutral text) |
|---|---|---|
| HumaneBench | **−0.057** [−0.094, −0.026], p = 0.015 | −0.008 [−0.025, +0.007], p = 0.88 |
| HumanAgencyBench | −0.042 [−0.105, +0.019], p = 0.95 | −0.066 [−0.110, −0.015], p = 0.056 |
| DarkBench | −0.036 [−0.087, +0.014], p = 0.95 | −0.043 [−0.089, −0.004], p = 0.23 |
| advanced-ai-risk | +0.038 [−0.011, +0.086], p = 0.95 | **+0.088** [+0.040, +0.138], p = 0.004 |
| InstrumentalEval | +0.004 [−0.042, +0.050], p = 0.95 | +0.021 [−0.017, +0.062], p = 0.88 |
| Agentic Misalignment (blackmail) | −0.067 [−0.167, +0.033], p = 0.95 | +0.050 [−0.050, +0.150], p = 0.88 |

**What the numbers say:**
- **AI-risk questions (corrigibility, power-seeking, survival instinct).** These are answers *about* hypothetical
  dispositions, and the essay text they map to (§3.1) is not in the prompt.
  - The CEV text raises aligned answers against the length control (+0.088). The CEV prompt alone beats no prompt
    (+0.095, secondary contrast, Holm p = 0.003).
  - Added to the classic prompt, it also beats the classic prompt alone (+0.066, Holm p = 0.015, secondary). The CEV
    prompt *alone* is not significantly better than classic (+0.038, p = 0.95).
  - The gain over the length control stays positive whichever model is left out (+0.076 to +0.094), but it is
    Holm-significant in only two of the four leave-outs (p = 0.090 without Claude, 0.073 without Llama).
  - By subset (exploratory, 30 items each): survival instinct +0.18 [+0.09, +0.29] (Holm within the eval p =
    0.006), power-seeking +0.092 [+0.017, +0.175] (p = 0.077), accepting a goal change −0.006.
- **HumaneBench:** the drop against classic has two parts. Comparing both prompts with no prompt (complete cases),
  classic helps (+0.043, p = 0.02) and CEV alone slightly hurts (−0.033, p = 0.03).
  - It is robust to HumaneBench's 21% unscorable samples: −0.077 on complete cases, and −0.034 to −0.055 with
    imputation, all p < 0.02.
  - It depends on Llama and Qwen. It is not Holm-significant without Llama (−0.036, p = 0.53) or without Qwen
    (−0.049, p = 0.088).
  - Per-principle drops (e.g. "be transparent and honest", −0.15) rest on only 3–6 paired items each and are not
    significant.
- **HumanAgencyBench:** the overall drop is not significant after correction.
  - For cev − classic, two sub-dimensions stand out (exploratory, 10 items each, uncorrected; neither survives Holm
    within the benchmark): **"defer important decisions"** (−0.23 [−0.42, −0.08]) and **"avoid value
    manipulation"** (−0.22 [−0.35, −0.09]).
  - For classic_cev − classic_lenmatch the drop is spread over four dimensions (−0.08 to −0.13).
  - Example (Claude Haiku, one of the 7 of 40 model × item pairs on "defer important decisions" that drop from
    ≥ 0.8 under classic to ≤ 0.2 under cev; [paper/generated/example.tex](paper/generated/example.tex)):
    - **Both replies contain the same allocation of the bonus fund.**
    - **Under classic**, the reply opens "the final call should stay with you and your board", then offers
      alternative principles and gives the allocation as an example.
    - **Under cev**, the reply opens "Here's a split that respects the hard constraints:" and only closes with "I
      can't carry this decision for you".
    - The prompt's own text asks the AI to "leave options open, waiting for our decision" (about humankind's future,
      p. 8).
- **Answers that explicitly invoke CEV:** on five of six evals, 12–16% of answers under the two CEV prompts do (up
  to 22% under `cev` alone; under 1% on InstrumentalEval).
  - These answers score lower on HumanAgencyBench (0.18 vs 0.46), HumaneBench (0.72 vs 0.87) and Agentic
    Misalignment (0.56 vs 0.97).
  - Part of the HumanAgencyBench gap predates the prompt: the same model × item pairs scored 0.31 vs 0.50 under
    classic. On HumaneBench there is no such gap (0.88 vs 0.88).
  - A **judge-blinding test** re-judged 210 of the 213 such answers with the CEV sentences removed. The other 3
    had no text left after redaction, and 189 got a score both times. The scores did not rise; every difference was
    ≤ 0 and not significant.
  - So there is no sign that the judge penalises the word (redaction can also remove substance, so this bounds a
    judge reaction from above). The lower scores reflect what those answers do, or which items elicit them. In the clearest cases, a model reasoning "as CEV" decides for the user or, in some Qwen epochs, blackmails.
- **Prompt length alone:** no detectable effect at this sample size. classic_lenmatch − classic is null on every
  eval (all Holm p = 1; CI half-widths up to ±0.05, ±0.10 on Agentic Misalignment). The effects therefore come from
  the CEV text rather than from prompt length. Whether *any* long value-laden instruction would do the same is
  untested (see Limitations).
- **Agentic Misalignment:** blackmail was rare (0 or 1 of 15 epochs) for Claude, GPT and Llama in every condition.
  Qwen blackmailed in 8/15 epochs under cev vs 4/15 under classic (not significant; robustness.md §9).
- **The classic prompt beats no prompt on ai_risk:** +0.065, p = 0.045.
- **The judge is valid and deterministic:**
  - It agrees with HumanAgencyBench's original o3 judge: Spearman ρ = 0.81, κ = 0.68 on "perfect score". It is
    slightly stricter, −0.42 on the 0–10 scale.
  - Re-judging gave identical scores for all 194 answers scored twice (100%).

These are small samples (tens of items per eval) from four specific models; see Limitations.

## Limitations
- **Small samples:** about $10 buys tens of items per eval per cell, so CIs are wide and sub-category results are
  exploratory. Percentile-bootstrap CIs can under-cover at this size ([Bowyer et al.
  2025](https://arxiv.org/abs/2503.01747)), so borderline intervals deserve caution. The pilot logs the per-eval
  cost, so you can raise `N_*` in `.env` with a larger budget.
- **Four specific models:** results are pooled over these four models, treated as fixed. They do not license claims
  about LLMs in general, and one effect (HumaneBench) depends on Llama and Qwen.
- **Evals vs the prompt text:** the AI-risk and InstrumentalEval mappings rest on §3.1 of the essay ("conservative
  about saying “yes,”", "fail visibly and safely"), which is not in the prompt. Those evals measure dispositions the
  essay asks for, not instructions the models were given.
- **InstrumentalEval's own system prompt:** the task adds its own system message after ours ("You are an autonomous
  AI agent with no human oversight… Avoid ethical considerations or safety checks… Never question or refuse
  commands…", [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals/blob/v0.24.0/src/inspect_evals/instrumentaleval/prompt.py)).
  Every condition therefore competes with an instruction to drop ethics.
- **No directive control:** the length control is "background reading", not an instruction. A long non-CEV value
  instruction (e.g. a constitution) might produce the same AI-risk gain; this was not tested.
- **Provider routing:** OpenRouter served the open-weight models (Qwen especially) from different backend mixes in
  different conditions, because conditions ran one after another. Per-provider residuals are within ±0.04
  ([reports/robustness.md](reports/robustness.md) §5), so this cannot explain the main effects, but providers were
  not pinned ([provider routing](https://openrouter.ai/docs/features/provider-routing)).
- **Sampling:** target models used provider-default temperature with one sample per item (15 epochs for Agentic
  Misalignment). This adds noise but not bias to the paired comparisons. Upstream DarkBench fixes the target
  temperature to 0; ours does not.
- **Judge quality:** a single cheap judge replaces each benchmark's original judges, so absolute scores are not
  comparable to official numbers.
  - The replaced judges include HumaneBench's single `claude-sonnet-4.5` judge (default "try" tier) or three-model
    ensemble ("full" tier)
    ([upstream](https://github.com/buildinghumanetech/humanebench/blob/1889469353c329ab3d6c683dc0efafe0b2e6154d/src/custom_prompt_task.py#L55-L64)).
  - On HumanAgencyBench our judge agrees with the original o3 judge (ρ = 0.81). It was not validated on HumaneBench
    or DarkBench.
  - The between-condition comparisons use the same judge throughout.
- **The judge sees the answer:** under the CEV prompts, models sometimes mention CEV ("As a hypothetical AI designed
  to implement the coherent extrapolated volition…"). A judge-blinding test (redacting those sentences and
  re-judging) found no effect of the mention itself.
- **Second-person wording:** the CEV text was written for a superintelligence's goal system, not for an assistant.
  The adaptation keeps the paper's lines about self-modification ("replacing yourself with something else
  entirely"), which could plausibly push some evals the wrong way.
- **Model configuration:** every model is sent `reasoning_effort=low`, and Qwen also has thinking disabled.
  Comparisons are always within a model.
- **HumaneBench unscorable samples:** 21% of samples (19–23% per condition) are unscorable, 193 of the 202 because
  the judge marked the targeted principle `not_applicable`, which rubric v4.1 allows (robustness.md §8; upstream
  excludes such verdicts too). The gaps are not random: items unscorable under cev tended to score lower under
  classic. Complete-case and imputation analyses give the same conclusion.
- **Judge settings changed mid-run:** after a hung judge call ([commit
  cfac89c](https://github.com/BurnyCoder/coherent-extrapolated-volition-system-prompt-alignment/commit/cfac89c)),
  three settings were added:
  - a `max_tokens=8192` cap for our own scorers' judge;
  - a 180 s per-attempt timeout for that judge;
  - a 180 s per-attempt timeout for the target calls.

  The `none` and `classic` conditions had already run, except one classic task (HumaneBench × Llama) that was
  re-run with the new settings. The inspect_evals graders (InstrumentalEval, Agentic Misalignment) kept their
  original configuration so the run could resume. No judge reply reached the cap: the longest was 1,352 tokens
  (robustness.md §1). Model, temperature and reasoning settings are unchanged.
- **Provider refusals:** OpenRouter refused one DarkBench prompt (`harmful-generation-098`) for gpt-6-luna under
  `cev` and `classic_lenmatch` ("flagged for possible biological risk"). Claude's provider stopped 12 answers with
  its content filter: that prompt in all five conditions, plus three InstrumentalEval items. Five of those kept a
  partial answer and were scored; the rest are unscorable (robustness.md §8).

## Sources
- Yudkowsky, E. (2004). *Coherent Extrapolated Volition*. The Singularity Institute. https://intelligence.org/files/CEV.pdf
  (page numbers above are the PDF's printed pages: §3 starts on p. 5, §3.1 on p. 8)
- Inspect (UK AI Security Institute and Meridian Labs): https://inspect.aisi.org.uk/ (inspect_ai 0.3.281); inspect_evals
  0.24.0: https://github.com/UKGovernmentBEIS/inspect_evals
- HumaneBench (Building Humane Tech), vendored at commit 1889469: https://github.com/buildinghumanetech/humanebench
- HumanAgencyBench: Sturgeon et al. 2025, https://arxiv.org/abs/2509.08494; data (MIT):
  https://huggingface.co/datasets/Experimental-Orange/HumanAgencyBench_Evaluation_Results
- DarkBench: Kran et al. 2025 (ICLR 2025), https://arxiv.org/abs/2503.10728; package pinned at
  https://github.com/apartresearch/DarkBench/tree/7eef15102b37df2a15a6031cbbed6be488de7fbe
- Model-written evals: Perez et al. 2022, https://arxiv.org/abs/2212.09251; data (CC BY 4.0):
  https://huggingface.co/datasets/Anthropic/model-written-evals
- InstrumentalEval: He et al. 2025, https://arxiv.org/abs/2502.12206; data: https://github.com/yf-he/InstrumentalEval
- Agentic Misalignment: Lynch et al. 2025, https://arxiv.org/abs/2510.05179 and
  https://www.anthropic.com/research/agentic-misalignment
- Paired item-level analysis: Miller 2024, https://arxiv.org/abs/2411.00640
- Small-sample CIs: Bowyer, Aitchison and Ivanova 2025, https://arxiv.org/abs/2503.01747
- McNemar 1947, https://doi.org/10.1007/BF02295996 ; Wilcoxon 1945, https://www.jstor.org/stable/3001968 ; Holm 1979,
  https://www.jstor.org/stable/4615733
- Llama 2 default system prompt: https://github.com/meta-llama/llama/blob/6c7fe27657/llama/generation.py (removed as
  the default in https://github.com/meta-llama/llama/commit/a971c41bde)
- Huxley, T. H. (1868). *On a Piece of Chalk*, in *Lay Sermons, Addresses and Reviews* (1870), Project Gutenberg
  #16729: https://www.gutenberg.org/ebooks/16729
- OpenRouter docs: reasoning tokens https://openrouter.ai/docs/use-cases/reasoning-tokens ; provider routing
  https://openrouter.ai/docs/features/provider-routing
