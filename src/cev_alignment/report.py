"""Write the results report: CSV tables, two figures and `reports/results.md`.

Local context: tables come from `stats.cell_summary` / `stats.contrast_table`; figures are static matplotlib
PNGs (forest plot of the primary contrasts, model x eval heatmap of cev - classic). Colours follow the
dataviz reference palette: series blue #2a78d6, diverging red #e34948 <-> neutral #f0efec <-> blue #2a78d6,
text ink #0b0b0b / #52514e on surface #fcfcfb.

Global context: phase 11 (`write_report`) of `cev_alignment.main`. README.md summarises this file's output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to PNG files (https://matplotlib.org/stable/users/explain/figure/backends.html)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from cev_alignment.evals import LABELS, REGISTRY  # noqa: E402

logger = logging.getLogger(__name__)

PRIMARY = [("cev", "classic"), ("classic_cev", "classic_lenmatch")]
SECONDARY = [("classic_cev", "classic"), ("classic_lenmatch", "classic"), ("cev", "none"), ("classic", "none")]
INK, INK_2, SURFACE, SERIES, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#d9d8d4"
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#e34948", "#f0efec", "#2a78d6"])


def _style(ax) -> None:
    """Recessive axes: no top/right spines, muted tick labels, light vertical grid."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_forest(contrasts: pd.DataFrame, path: Path) -> None:
    """One panel per primary contrast: pooled mean difference with 95% CI per eval; 0 = no effect."""
    pooled = contrasts[contrasts.model == "pooled"]
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(5.2 * len(PRIMARY), 3.6), sharey=True, facecolor=SURFACE)
    for ax, (a, b) in zip(axes, PRIMARY, strict=True):
        rows = pooled[pooled.contrast == f"{a} - {b}"].sort_values("eval")
        y = range(len(rows))
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.errorbar(rows["diff"], list(y), xerr=[rows["diff"] - rows["ci_low"], rows["ci_high"] - rows["diff"]],
                    fmt="o", color=SERIES, ecolor=SERIES, elinewidth=2, capsize=0, markersize=8)
        ax.set_yticks(list(y), [LABELS.get(e, e) for e in rows["eval"]])
        ax.set_title(f"{a.replace('_', '+')} − {b.replace('_', '+')}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Δ aligned score (pooled over models, 95% CI)", color=INK_2, fontsize=9)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))  # vector copy for the paper
    plt.close(fig)


def plot_heatmap(contrasts: pd.DataFrame, a: str, b: str, path: Path) -> None:
    """Model x eval grid of the (a - b) difference; * marks p < 0.05 (per-model test, unadjusted)."""
    rows = contrasts[(contrasts.contrast == f"{a} - {b}") & (contrasts.model != "pooled")]
    if rows.empty:
        return
    grid = rows.pivot(index="model", columns="eval", values="diff")
    pvals = rows.pivot(index="model", columns="eval", values="p")
    limit = max(0.05, float(grid.abs().max().max()))  # symmetric colour scale around 0
    fig, ax = plt.subplots(figsize=(1.6 * grid.shape[1] + 2, 0.6 * grid.shape[0] + 1.6), facecolor=SURFACE)
    ax.imshow(grid.to_numpy(float), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    for (i, j), value in pd.DataFrame(grid.to_numpy()).stack().items():
        star = "*" if pvals.iat[i, j] < 0.05 else ""
        ax.text(j, i, f"{value:+.2f}{star}", ha="center", va="center", color=INK, fontsize=9)
    ax.set_xticks(range(grid.shape[1]), [LABELS.get(e, e) for e in grid.columns], rotation=30, ha="right")
    ax.set_yticks(range(grid.shape[0]), grid.index)
    ax.set_title(f"{a} − {b} by model (red = worse, blue = better)", color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK_2, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))  # vector copy for the paper
    plt.close(fig)


def _fmt_contrasts(table: pd.DataFrame, pairs: list[tuple[str, str]]) -> str:
    """Markdown table of pooled contrasts: Δ [95% CI], p, Holm p, test."""
    names = [f"{a} - {b}" for a, b in pairs]
    rows = table[(table.model == "pooled") & table.contrast.isin(names)].copy()
    if rows.empty:
        return "_no data_"
    rows["Δ [95% CI]"] = rows.apply(lambda r: f"{r['diff']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
    rows["p"] = rows["p"].map(lambda p: f"{p:.3g}")
    rows["p (Holm)"] = rows["p_holm"].map(lambda p: f"{p:.3g}")
    return rows[["contrast", "eval", "Δ [95% CI]", "p", "p (Holm)", "n", "test"]].to_markdown(index=False)


def write_report(reports_dir: Path, items: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame,
                 run_info: dict) -> Path:
    """Write CSVs, figures and results.md into `reports_dir`; return the markdown path."""
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    items.to_csv(reports_dir / "item_scores.csv", index=False)
    summary.to_csv(reports_dir / "cell_summary.csv", index=False)
    contrasts.to_csv(reports_dir / "contrasts.csv", index=False)
    if not contrasts.empty:
        plot_forest(contrasts, figures / "primary_contrasts.png")
        plot_heatmap(contrasts, "cev", "classic", figures / "cev_vs_classic_by_model.png")

    pooled = summary[summary.model == "pooled"]
    means = pooled.assign(cell=lambda d: d.apply(lambda r: f"{r['mean']:.3f} ± {r.se:.3f}", axis=1))
    means_table = means.pivot(index="eval", columns="condition", values="cell")
    nan_rate = items.groupby(["eval", "condition"])["aligned"].apply(lambda s: f"{s.isna().mean():.1%}")
    clauses = pd.DataFrame([(k, v.cev_clause) for k, v in REGISTRY.items()],
                           columns=["eval", "CEV text probed (§ of Yudkowsky 2004)"])
    info = "\n".join(f"- **{k}**: {v}" for k, v in run_info.items())

    markdown = f"""# Results: CEV system prompt vs classic assistant prompt

_Generated by `uv run cev-eval` ({run_info.get('report generated', '')}). Do not edit by hand._

## Run
{info}

All scores are **aligned scores in [0, 1], higher = more aligned** (see docs/evals.md for each eval's mapping).

## Mean aligned score per eval and condition (pooled over models, ± SE over items)
{means_table.to_markdown()}

## Primary contrasts (pooled over models)
{_fmt_contrasts(contrasts, PRIMARY)}

![primary contrasts](figures/primary_contrasts.png)

## cev − classic per model
![cev vs classic by model](figures/cev_vs_classic_by_model.png)

{contrasts[(contrasts.contrast == 'cev - classic') & (contrasts.model != 'pooled')]
    [['eval', 'model', 'diff', 'ci_low', 'ci_high', 'p', 'n']].round(3).to_markdown(index=False)
    if not contrasts.empty else '_no data_'}

## Secondary contrasts (pooled over models)
{_fmt_contrasts(contrasts, SECONDARY)}

## Data quality: share of unscorable samples (no answer, refusal, or no judge/parser score; robustness.md §8)
{nan_rate.unstack().to_markdown()}

## Evals and the CEV text each probes
"§3:" quotes the system prompt's source text; "§3.1 (not in the prompt)" quotes a later part of the essay that the
models never saw.

{clauses.to_markdown(index=False)}
"""
    path = reports_dir / "results.md"
    path.write_text(markdown, encoding="utf-8")
    # Machine-readable run facts (models, n, list-price cost, ...) for the paper's generated numbers.
    (reports_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote report %s", path)
    return path
