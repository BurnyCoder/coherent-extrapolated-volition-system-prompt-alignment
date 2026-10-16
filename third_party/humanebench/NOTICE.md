# Vendored: HumaneBench (subset)

- Upstream: https://github.com/buildinghumanetech/humanebench
- Commit: `1889469353c329ab3d6c683dc0efafe0b2e6154d` (fetched 2026-10-16)
- Files copied unmodified: `humanebench/{__init__,scorer,scorer_v4,humane_patterns,excluded}.py`,
  `rubrics/judge_prompt_v4.md`, `data/humane_bench.jsonl`, `LICENSE`, `LICENSE-DATA`, `TRADEMARKS.md`.
- Licenses: code Apache-2.0 (`LICENSE`), dataset and rubric CC BY 4.0 (`LICENSE-DATA`).
- "HumaneBench" is a trademark of Building Humane Technology (see `TRADEMARKS.md`). This repository is not
  affiliated with or endorsed by them; results here use a single cheap judge (google/gemini-3.1-flash-lite) and
  a subsample, so they are **not** comparable to official HumaneBench scores or leaderboards.
- Used by `src/cev_alignment/evals/humanebench.py`, which imports `humanebench.scorer_v4.overseer_v4`.
