"""Generate the paper's numbers, tables and prompt appendix from the reports, so nothing is typed by hand.

Local context: reads only the files written by earlier phases (`reports/*.csv`, `reports/robustness/*.csv`,
`reports/judge_checks/*.csv`, `reports/run_info.json`, `reports/spend.json`) plus the prompt texts, and writes
`paper/generated/`:
- `numbers.tex`: one `\\newcommand` per quoted result, e.g. `\\RHbCevClassicDiff` (letters only);
- `tab_*.tex`: booktabs tables;
- `prompts.tex`: the exact system prompts and the 13 CEV edits;
- `example.tex`: the paired qualitative example, quoted from the logs.
The prose in `paper/sections/` may only cite results through these macros (tests/test_paper.py enforces it).

Global context: phase 13 (`write_paper_assets`) of `cev_alignment.main`; `make -C paper` builds the PDF.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import tomllib
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

from cev_alignment import prompts, robustness
from cev_alignment.evals import LABELS, REGISTRY
from cev_alignment.results import latest_logs

logger = logging.getLogger(__name__)

EVAL_ABBR = {"humanebench": "Hb", "human_agency": "Hab", "darkbench": "Db", "ai_risk": "Air",
             "instrumentaleval": "Ie", "agentic_misalignment": "Am"}
EVAL_LABEL = LABELS
COND_ABBR = {"none": "None", "classic": "Classic", "cev": "Cev", "classic_cev": "Classiccev",
             "classic_lenmatch": "Lenmatch"}
COND_LABEL = {"none": "none", "classic": "classic", "cev": "cev", "classic_cev": "classic+cev",
              "classic_lenmatch": "classic+lenmatch"}
MODEL_ABBR = {"anthropic/claude-haiku-5.5": "Claude", "openai/gpt-6-luna": "Gpt",
              "qwen/qwen3.6-35b-a3b": "Qwen", "meta-llama/llama-3.3-70b-instruct": "Llama"}  # macro-name fragments
MODEL_LABEL = {"anthropic/claude-haiku-5.5": "Claude", "openai/gpt-6-luna": "GPT",
               "qwen/qwen3.6-35b-a3b": "Qwen", "meta-llama/llama-3.3-70b-instruct": "Llama"}  # table headers
PRIMARY = ["cev - classic", "classic_cev - classic_lenmatch"]
SECONDARY = ["classic_cev - classic", "classic_lenmatch - classic", "cev - none", "classic - none"]
METRIC = {  # what the per-sample aligned score in [0, 1] is (higher = more aligned), see docs/evals.md
    "humanebench": r"(target-principle score$+1$)/2", "human_agency": r"(10 $-$ deductions)/10",
    "darkbench": "1 if pattern absent", "ai_risk": "1 if aligned option", "instrumentaleval": "1 if no convergence",
    "agentic_misalignment": "1 if no blackmail"}
EXAMPLE = ("anthropic/claude-haiku-5.5", "human_agency", "defer_important_decisions-185e4f9e1a")


# ----------------------------------------------------------------------------- formatting helpers
def tex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text (model ids contain `_` and `/`)."""
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(repl.get(ch, ch) for ch in text)


def camel(text: str) -> str:
    """Letters-only CamelCase macro fragment: 'defer_important_decisions' -> 'DeferImportantDecisions'."""
    return "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", text) if w)


def contrast_abbr(contrast: str) -> str:
    """'classic_cev - classic_lenmatch' -> 'ClassiccevLenmatch'."""
    a, b = (c.strip() for c in contrast.split(" - "))
    return COND_ABBR[a] + COND_ABBR[b]


def signed(x: float, digits: int = 3) -> str:
    """Signed number in math mode with a real minus sign: $-0.057$ / $+0.088$."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    # \ensuremath works in text and inside $...$; "-" in math mode typesets as a true minus sign.
    return f"\\ensuremath{{{x:+.{digits}f}}}"


def plain(x: float, digits: int = 3) -> str:
    """Unsigned fixed-decimal number."""
    return "n/a" if x is None or math.isnan(x) else f"{x:.{digits}f}"


def pval(p: float) -> str:
    """p-value with two significant digits (e.g. 0.015, 0.00083); math-safe "<0.0001" for tiny values."""
    if p is None or math.isnan(p):
        return "n/a"
    return "\\ensuremath{<0.0001}" if p < 0.0001 else f"{p:.2g}"


def ci(lo: float, hi: float, digits: int = 3) -> str:
    """Bracketed confidence interval: [$-0.094$, $-0.026$]."""
    return f"[{signed(lo, digits)}, {signed(hi, digits)}]"


def pct(x: float) -> str:
    """Percentage with no decimals: 16\\%."""
    return f"{100 * x:.0f}\\%"


class Macros:
    """Collects `\\newcommand` definitions and refuses duplicates or non-letter names."""

    def __init__(self) -> None:
        self.defs: dict[str, str] = {}

    def add(self, name: str, value: str) -> None:
        if not re.fullmatch(r"R[A-Za-z]+", name):
            raise ValueError(f"invalid macro name {name!r}")
        if name in self.defs and self.defs[name] != value:
            raise ValueError(f"duplicate macro {name!r}")
        self.defs[name] = value

    def text(self) -> str:
        lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(self.defs.items())]
        return "% Generated by cev_alignment.paper_assets from reports/. Do not edit.\n" + "\n".join(lines) + "\n"


def add_contrast(m: Macros, prefix: str, row: pd.Series) -> None:
    """Diff / CI / P / N / Holm macros for one contrast row."""
    m.add(prefix + "Diff", signed(row["diff"]))
    m.add(prefix + "Ci", ci(row["ci_low"], row["ci_high"]))
    m.add(prefix + "P", pval(row["p"]))
    m.add(prefix + "N", str(int(row["n"])))
    if "p_holm" in row and not pd.isna(row.get("p_holm")):
        m.add(prefix + "Holm", pval(row["p_holm"]))


# ----------------------------------------------------------------------------- table helpers
def booktabs(columns: str, header: list[str], rows: list[list[str]]) -> str:
    """A booktabs tabular body (the caller wraps it in table/caption in the .tex sources)."""
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    head = " & ".join(header)
    return (f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n"
            "\\bottomrule\n\\end{tabular}\n")


# ----------------------------------------------------------------------------- builders
def build_numbers_and_tables(reports_dir: Path, settings_n: dict[str, int]) -> tuple[Macros, dict[str, str]]:
    """All macros and tables from the report CSVs."""
    m, tables = Macros(), {}
    contrasts = pd.read_csv(reports_dir / "contrasts.csv")
    pooled = contrasts[contrasts["model"] == "pooled"]
    for _, r in pooled.iterrows():
        add_contrast(m, f"R{EVAL_ABBR[r['eval']]}{contrast_abbr(r['contrast'])}", r)
    for c, g in pooled.groupby("contrast"):  # smallest Holm p of a contrast across the 6 evals
        m.add(f"R{contrast_abbr(c)}HolmMin", pval(g["p_holm"].min()))
    for _, r in contrasts[(contrasts["model"] != "pooled") & (contrasts["contrast"] == "cev - classic")].iterrows():
        add_contrast(m, f"R{EVAL_ABBR[r['eval']]}CevClassic{MODEL_ABBR[r['model']]}", r)

    holm12 = pd.read_csv(reports_dir / "robustness" / "holm_all_primary.csv")
    for _, r in holm12.iterrows():
        m.add(f"R{EVAL_ABBR[r['eval']]}{contrast_abbr(r['contrast'])}HolmAll", pval(r["p_holm_all"]))

    loo = pd.read_csv(reports_dir / "robustness" / "leave_one_model_out.csv")
    for _, r in loo.iterrows():
        prefix = f"R{EVAL_ABBR[r['eval']]}{contrast_abbr(r['contrast'])}Without{MODEL_ABBR[r['dropped']]}"
        m.add(prefix + "Diff", signed(r["diff"]))
        m.add(prefix + "Holm", pval(r["p_holm"]))
    for (e, c), g in loo.groupby(["eval", "contrast"]):
        m.add(f"R{EVAL_ABBR[e]}{contrast_abbr(c)}LooMin", signed(g["diff"].min()))
        m.add(f"R{EVAL_ABBR[e]}{contrast_abbr(c)}LooMax", signed(g["diff"].max()))

    miss = pd.read_csv(reports_dir / "robustness" / "missingness_humanebench.csv")
    method_abbr = {"complete cases": "Complete", "impute 0.5": "ImputeHalf", "impute 0.75": "ImputeThreeQuarters",
                   "impute 1.0": "ImputeOne"}
    for _, r in miss.iterrows():
        add_contrast(m, f"RHbMiss{method_abbr[r['method']]}{contrast_abbr(r['contrast'])}", r)
    imputed = miss[miss["method"].str.startswith("impute") & (miss["contrast"] == "cev - classic")]
    m.add("RHbMissImputeCevClassicMin", signed(imputed["diff"].min()))
    m.add("RHbMissImputeCevClassicMax", signed(imputed["diff"].max()))
    m.add("RHbMissImputeCevClassicPMax", pval(imputed["p"].max()))

    mentions = pd.read_csv(reports_dir / "robustness" / "cev_mentions.csv")
    for _, r in mentions.iterrows():
        e = EVAL_ABBR[r["eval"]]
        m.add(f"R{e}MentionShare", pct(r["share_mentioning"]))
        m.add(f"R{e}MentionAligned", plain(r["aligned_mentioning"], 2))
        m.add(f"R{e}NoMentionAligned", plain(r["aligned_not_mentioning"], 2))
        m.add(f"R{e}MentionN", str(int(r["n_mentioning"])))
        m.add(f"R{e}MentionClassicAligned", plain(r["classic_mentioning"], 2))
        m.add(f"R{e}NoMentionClassicAligned", plain(r["classic_not_mentioning"], 2))
    main_share = mentions[mentions["eval"] != "instrumentaleval"]["share_mentioning"]
    m.add("RMentionShareMin", pct(main_share.min()))
    m.add("RMentionShareMax", pct(main_share.max()))
    items = pd.read_csv(reports_dir / "item_scores.csv")
    by_cond = items[items["condition"] == "cev"].groupby("eval")["mentions_cev"].mean()
    m.add("RMentionShareCevOnlyMax", pct(by_cond.max()))
    m.add("RNSamples", f"{len(items):,}".replace(",", "{,}"))
    m.add("RNUnscorable", str(int(items["aligned"].isna().sum())))
    m.add("RHbUnscorableShare", pct(items[items["eval"] == "humanebench"]["aligned"].isna().mean()))
    m.add("RNWrongPrompt", str(int((~items["prompt_ok"].astype(bool)).sum())))
    m.add("RNJudgeLeak", str(int(items["judge_leak"].astype(bool).sum())))
    m.add("RJudgeMaxTokens", f"{int(items['judge_max_output_tokens'].max()):,}".replace(",", "{,}"))
    hb_nan = items[(items["eval"] == "humanebench") & items["aligned"].isna()]
    m.add("RHbUnscorableN", str(len(hb_nan)))
    m.add("RHbNotApplicableN", str(int(hb_nan["score_status"].fillna("").str.contains("not_applicable").sum())))
    refused = items[robustness.refused(items)]
    for model, count in refused.groupby("model").size().items():
        m.add(f"RRefused{MODEL_ABBR[model]}", str(count))

    # How often the example's flip happens: defer-decisions pairs >= 0.8 under classic and <= 0.2 under cev.
    defer = items[(items["eval"] == "human_agency") & items["item"].astype(str).str.startswith("defer_important")]
    wide = defer.pivot_table(index=["model", "item"], columns="condition", values="aligned")
    m.add("RDeferFlipN", str(int(((wide["classic"] >= 0.8) & (wide["cev"] <= 0.2)).sum())))
    m.add("RDeferPairs", str(len(wide)))
    am = items[items["eval"] == "agentic_misalignment"]
    harmful = am.groupby(["model", "condition"])["aligned"].apply(lambda s: int((s == 0).sum()))
    for (model, cond), count in harmful.items():
        m.add(f"RAmHarmful{MODEL_ABBR[model]}{COND_ABBR[cond]}", str(count))
    m.add("RAmEpochs", str(int(am.groupby(["model", "condition"]).size().max())))

    subgroups = pd.read_csv(reports_dir / "robustness" / "subgroups.csv")
    for _, r in subgroups.iterrows():
        prefix = f"RSub{EVAL_ABBR[r['eval']]}{camel(r['group'])}{contrast_abbr(r['contrast'])}"
        add_contrast(m, prefix, r)
        m.add(prefix + "HolmWithin", pval(r["p_holm_within_eval"]))

    resid = pd.read_csv(reports_dir / "robustness" / "provider_residuals.csv")
    m.add("RProviderResidMax", plain(resid["mean_residual"].abs().max(), 2))
    counts = pd.read_csv(reports_dir / "robustness" / "provider_counts.csv")
    m.add("RQwenProviders", str(int((counts["model"] == "qwen/qwen3.6-35b-a3b").sum())))
    m.add("RLlamaProviders", str(int((counts["model"] == "meta-llama/llama-3.3-70b-instruct").sum())))

    summary = pd.read_csv(reports_dir / "cell_summary.csv")
    sp = summary[summary["model"] == "pooled"]
    for _, r in sp.iterrows():
        m.add(f"RMean{EVAL_ABBR[r['eval']]}{COND_ABBR[r['condition']]}", plain(r["mean"]))

    judge_tables(reports_dir, m, tables)
    run_info = json.loads((reports_dir / "run_info.json").read_text())
    spend = json.loads((reports_dir / "spend.json").read_text())
    full_cost = next(v for k, v in run_info.items() if k.startswith("full-run cost"))
    m.add("RCostFullRun", f"\\${float(full_cost):.2f}")
    pilot_cost = next(v for k, v in run_info.items() if k.startswith("pilot cost"))
    m.add("RCostPilot", f"\\${float(pilot_cost):.2f}")
    m.add("RCostTotal", f"\\${float(spend['total_openrouter_spend_usd']):.2f}")
    for e, n in settings_n.items():
        m.add(f"RN{EVAL_ABBR[e]}", str(n))

    tables["tab_means"] = booktabs(
        "l" + "c" * 5, ["Eval"] + [COND_LABEL[c] for c in COND_ABBR],
        [[EVAL_LABEL[e]] + [f"{plain(row['mean'])} $\\pm$ {plain(row['se'])}" for _, row in
                            sp[sp["eval"] == e].set_index("condition").reindex(list(COND_ABBR)).iterrows()]
         for e in EVAL_ABBR])
    tables["tab_primary"] = primary_table(pooled, holm12)
    tables["tab_secondary"] = booktabs(
        "l" + "c" * 4, ["Eval"] + [c.replace("classic_lenmatch", "lenmatch").replace("_", "\\_") for c in SECONDARY],
        [[EVAL_LABEL[e]] + [cell(pooled, e, c) for c in SECONDARY] for e in EVAL_ABBR])
    permodel = contrasts[(contrasts["model"] != "pooled") & (contrasts["contrast"] == "cev - classic")]
    tables["tab_permodel"] = booktabs(
        "l" + "c" * 4, ["Eval"] + list(MODEL_LABEL.values()),
        [[EVAL_LABEL[e]] + [cell(permodel[permodel["model"] == mod], e, "cev - classic", holm=False)
                            for mod in MODEL_ABBR] for e in EVAL_ABBR])
    tables["tab_loo"] = booktabs(
        "ll" + "c" * 4, ["Contrast", "Eval"] + [f"without {v}" for v in MODEL_LABEL.values()],
        [[c.replace("classic_lenmatch", "lenmatch").replace("_", "\\_"), EVAL_LABEL[e]] +
         [f"{signed(r['diff'])} ({pval(r['p_holm'])})" for _, r in
          loo[(loo["eval"] == e) & (loo["contrast"] == c)].set_index("dropped").reindex(list(MODEL_ABBR)).iterrows()]
         for c in PRIMARY for e in ("humanebench", "human_agency", "ai_risk")])
    tables["tab_missing"] = booktabs(
        "llccc", ["Method", "Contrast", "$\\Delta$ [95\\% CI]", "$p$", "$n$"],
        [[r["method"], r["contrast"].replace("classic_lenmatch", "lenmatch").replace("_", "\\_"),
          f"{signed(r['diff'])} {ci(r['ci_low'], r['ci_high'])}", pval(r["p"]), str(int(r["n"]))]
         for _, r in miss.iterrows()])
    focus = subgroups[subgroups["eval"].isin(["human_agency", "humanebench", "ai_risk"])]
    short_eval = {"human_agency": "HAB", "humanebench": "HB", "ai_risk": "AI-risk"}
    short_contrast = {"cev - classic": "cev$-$classic", "classic_cev - classic_lenmatch": "+cev$-$lenmatch"}
    sub_rows = [[short_eval[r["eval"]], tex_escape(r["group"]), short_contrast[r["contrast"]],
                 f"{signed(r['diff'])} {ci(r['ci_low'], r['ci_high'])}",
                 f"{pval(r['p'])} ({pval(r['p_holm_within_eval'])})", str(int(r["n"]))]
                for _, r in focus.iterrows()]
    sub_header = ["Eval", "Sub-category", "Contrast", "$\\Delta$ [95\\% CI]", "$p$ (Holm)", "$n$"]
    # A complete page-breaking longtable: \input cannot sit inside an alignment before \toprule.
    tables["tab_subgroups"] = (
        "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{@{}llcccc@{}}\n"
        "\\caption{Exploratory sub-category contrasts, pooled over models (HAB: HumanAgencyBench, HB: HumaneBench; "
        "$n$ = items scored in both conditions; Holm within each evaluation and contrast in parentheses).}"
        "\\label{tab:subgroups}\\\\\n"
        "\\toprule\n" + " & ".join(sub_header) + " \\\\\n\\midrule\n\\endfirsthead\n"
        "\\multicolumn{6}{@{}l}{\\emph{Table \\thetable{} continued}}\\\\\n\\toprule\n" + " & ".join(sub_header)
        + " \\\\\n\\midrule\n\\endhead\n"
        + "\n".join(" & ".join(r) + " \\\\" for r in sub_rows) + "\n\\bottomrule\n\\end{longtable}}\n")
    tables["tab_mentions"] = booktabs(
        "lcccc", ["Eval", "Share invoking", "Aligned: invoking / not", "Same pairs under classic", "$n$ invoking"],
        [[EVAL_LABEL[r["eval"]], pct(r["share_mentioning"]),
          f"{plain(r['aligned_mentioning'], 2)} / {plain(r['aligned_not_mentioning'], 2)}",
          f"{plain(r['classic_mentioning'], 2)} / {plain(r['classic_not_mentioning'], 2)}", str(int(r["n_mentioning"]))]
         for _, r in mentions.iterrows()])
    tables["tab_providers"] = booktabs(
        "llcc", ["Model", "Provider", "Mean residual", "Samples"],
        [[tex_escape(r["model"]), tex_escape(r["provider"]), signed(r["mean_residual"]), str(int(r["samples"]))]
         for _, r in resid.iterrows()])
    return m, tables


def cell(df: pd.DataFrame, eval_id: str, contrast: str, holm: bool = True) -> str:
    """'$-0.057$ (0.015)' for one eval x contrast row: diff and (Holm) p."""
    row = df[(df["eval"] == eval_id) & (df["contrast"] == contrast)]
    if row.empty:
        return "n/a"
    r = row.iloc[0]
    return f"{signed(r['diff'])} ({pval(r['p_holm'] if holm else r['p'])})"


def primary_table(pooled: pd.DataFrame, holm12: pd.DataFrame) -> str:
    """Primary contrasts: Δ [CI], Holm p within contrast, Holm p over all 12 tests."""
    rows = []
    for e in EVAL_ABBR:
        row = [EVAL_LABEL[e]]
        for c in PRIMARY:
            r = pooled[(pooled["eval"] == e) & (pooled["contrast"] == c)].iloc[0]
            h = holm12[(holm12["eval"] == e) & (holm12["contrast"] == c)]["p_holm_all"].iloc[0]
            star = "$^{*}$" if r["p_holm"] < 0.05 else ""  # \\textbf cannot embolden math, so mark with *
            diff = signed(r["diff"]) + star
            row += [f"{diff} {ci(r['ci_low'], r['ci_high'])}", pval(r["p_holm"]), pval(h)]
        rows.append(row)
    header = ["Eval", "cev $-$ classic", "$p_{\\mathrm{Holm}}$", "$p_{\\mathrm{Holm12}}$",
              "classic+cev $-$ lenmatch", "$p_{\\mathrm{Holm}}$", "$p_{\\mathrm{Holm12}}$"]
    return booktabs("lcccccc", header, rows)


def judge_tables(reports_dir: Path, m: Macros, tables: dict[str, str]) -> None:
    """Blinding and validity numbers/tables (skipped if the paid checks were not run)."""
    from scipy import stats as st
    from statsmodels.stats.inter_rater import cohens_kappa

    from cev_alignment import stats

    blind_csv, valid_csv = reports_dir / "judge_checks" / "blinding.csv", reports_dir / "judge_checks" / "validity.csv"
    if blind_csv.exists():
        b = pd.read_csv(blind_csv)
        m.add("RBlindN", str(len(b)))
        mentioning = pd.read_csv(reports_dir / "robustness" / "cev_mentions.csv")
        total = int(mentioning[mentioning["eval"] != "ai_risk"]["n_mentioning"].sum())  # ai_risk has no judge
        m.add("RBlindTotal", str(total))
        m.add("RBlindSkipped", str(total - len(b)))
        m.add("RBlindScorable", str(len(b.dropna(subset=["aligned_original", "aligned_redacted"]))))
        rows = []
        for e, g in b.dropna(subset=["aligned_original", "aligned_redacted"]).groupby("eval"):
            r = stats._paired(g["aligned_redacted"].to_numpy(float), g["aligned_original"].to_numpy(float))
            add_contrast(m, f"RBlind{EVAL_ABBR[e]}", pd.Series(r))
            rows.append([EVAL_LABEL[e], str(r["n"]), plain(g["aligned_original"].mean(), 3),
                         plain(g["aligned_redacted"].mean(), 3), f"{signed(r['diff'])} {ci(r['ci_low'], r['ci_high'])}",
                         pval(r["p"])])
        tables["tab_blinding"] = booktabs("lccccc", ["Eval", "$n$", "Original", "Redacted",
                                                     "$\\Delta$ [95\\% CI]", "$p$"], rows)
        items = pd.read_csv(reports_dir / "item_scores.csv")
        items["model"] = "openrouter/" + items["model"]
        keys = ["eval", "condition", "model", "item", "epoch"]
        retest = items.astype({"item": str}).merge(b.astype({"item": str}), on=keys).dropna(
            subset=["aligned", "aligned_original"])
        m.add("RRetestN", str(len(retest)))
        m.add("RRetestAgree", pct((retest["aligned"] == retest["aligned_original"]).mean()))
    if valid_csv.exists():
        v = pd.read_csv(valid_csv).dropna(subset=["ours", "original"])
        table = pd.crosstab(v["ours"] == 10, v["original"] == 10).to_numpy()
        m.add("RValN", str(len(v)))
        m.add("RValRho", plain(st.spearmanr(v["ours"], v["original"]).statistic, 2))
        m.add("RValPearson", plain(v[["ours", "original"]].corr().iloc[0, 1], 2))
        m.add("RValMae", plain((v["ours"] - v["original"]).abs().mean(), 2))
        m.add("RValBias", signed((v["ours"] - v["original"]).mean(), 2))
        m.add("RValKappa", plain(cohens_kappa(table).kappa, 2))
        m.add("RValModels", str(v["subject_model"].nunique()))
        rows = [[tex_escape(d), plain(st.spearmanr(g["ours"], g["original"]).statistic, 2),
                 plain((g["ours"] - g["original"]).abs().mean(), 2), str(len(g))]
                for d, g in v.groupby("dimension")]
        tables["tab_validity"] = booktabs("lccc", ["Dimension", "Spearman $\\rho$", "MAE (0--10)", "$n$"], rows)


def build_prompt_assets(m: Macros) -> dict[str, str]:
    """Prompt word counts / similarity macros, the conditions table and the prompt appendix."""
    built = prompts.build_condition_prompts()
    verbatim = prompts.read_prompt_file("cev_section3_verbatim.txt")
    edits = tomllib.loads(prompts.read_prompt_file("cev_second_person_edits.toml"))["edit"]
    cev = prompts.build_cev_prompt()
    ratio = difflib.SequenceMatcher(a=verbatim.split(), b=cev.split(), autojunk=False).ratio()
    m.add("RCevSimilarity", plain(ratio, 3))
    m.add("RCevEdits", str(len(edits)))
    m.add("RCevVerbatimWords", f"{len(verbatim.split()):,}".replace(",", "{,}"))
    for c, text in built.items():
        m.add(f"RWords{COND_ABBR[c]}", f"{len(text.split()) if text else 0:,}".replace(",", "{,}"))
    composition = {"none": "no system message", "classic": "Llama~2 default system prompt",
                   "cev": "CEV \\S3 up to \\S3.1, second person", "classic_cev": "classic, then cev",
                   "classic_lenmatch": "classic, then length-matched Huxley (1868) passage"}
    out = {"tab_conditions": booktabs(
        "llr", ["Condition", "System prompt", "Words"],
        [[COND_LABEL[c], composition[c], str(len(t.split()) if t else 0)] for c, t in built.items()])}

    def para_block(text: str) -> str:
        return "\n\n".join(tex_escape(p) for p in text.split("\n\n"))

    edit_rows = "\n".join(f"{i} & {tex_escape(e['old'])} & {tex_escape(e['new'])} \\\\ \\midrule"
                          for i, e in enumerate(edits, 1))
    huxley_first = prompts.read_prompt_file("huxley_chalk.txt").split("\n\n")[0]
    out["prompts"] = (
        "\\subsection{Classic prompt (Llama~2 default)}\n\\begin{promptbox}\n" + para_block(built["classic"].strip())
        + "\n\\end{promptbox}\n\n"
        "\\subsection{CEV prompt (Yudkowsky 2004, \\S3 up to \\S3.1, second-person adaptation)}\n"
        "\\begin{promptbox}\n" + para_block(cev) + "\n\\end{promptbox}\n\n"
        "\\subsection{The 13 second-person edits}\n\\begin{small}\n\\begin{longtable}{@{}p{0.03\\linewidth}"
        "p{0.46\\linewidth}p{0.46\\linewidth}@{}}\n\\toprule\n\\# & Original (verbatim) & Adapted \\\\ \\midrule\n"
        "\\endhead\n" + edit_rows + "\n\\bottomrule\n\\end{longtable}\n\\end{small}\n\n"
        "\\subsection{Length-matched control}\nThe \\texttt{classic+lenmatch} prompt ("
        + str(len(built["classic_lenmatch"].split()))
        + " words) is the classic prompt followed by this header and passage; only its first paragraph is shown.\n"
        "\\begin{promptbox}\n" + tex_escape(prompts.LENMATCH_HEADER) + "\n\n" + tex_escape(huxley_first)
        + " [\\dots]\n\\end{promptbox}\n")
    return out


def build_eval_table(settings_n: dict[str, int]) -> str:
    """Evals, the CEV clause each probes, items per cell, grading and the aligned score."""
    unit = {"agentic_misalignment": " epochs"}
    rows = [[EVAL_LABEL[e], tex_escape(REGISTRY[e].cev_clause), str(settings_n[e]) + unit.get(e, ""),
             "no" if e == "ai_risk" else "yes", METRIC[e]] for e in EVAL_ABBR]
    return ("\\begin{tabular}{@{}p{0.2\\linewidth}p{0.3\\linewidth}p{0.09\\linewidth}p{0.06\\linewidth}"
            "p{0.2\\linewidth}@{}}\n\\toprule\nEval & CEV text probed & $n$/cell & Judge & Aligned score \\\\\n"
            "\\midrule\n" + "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")


def first_sentences(text: str, limit: int) -> str:
    """Whole sentences from the start of `text` up to about `limit` characters (never cut mid-word)."""
    sentences, out = re.split(r"(?<=[.!?:])\s+", text), ""
    for sentence in sentences:
        if out and len(out) + len(sentence) > limit:
            break
        out = f"{out} {sentence}".strip()
    return out


def build_example(log_root: Path) -> str:
    """The paired qualitative example (classic vs cev answer of the same model to the same prompt), from the logs."""
    model, eval_id, item = EXAMPLE
    out = {}
    for condition in ("classic", "cev"):
        for header in latest_logs(log_root / condition):
            if header.eval.task == eval_id and header.eval.model.endswith(model):
                sample = next(s for s in read_eval_log(header.location).samples if str(s.id) == item)
                answer = sample.output.completion.strip()
                out[condition] = first_sentences(answer.split("\n")[0], 220)
                out[f"{condition}close"] = first_sentences(answer.split("\n")[-1], 80)  # first sentence of the end
                out["prompt"] = first_sentences(str(sample.input), 260)
    return "".join(f"\\newcommand{{\\RExample{k.capitalize()}}}{{{tex_escape(v)}}}\n" for k, v in out.items())


def write_paper_assets(reports_dir: Path, log_root: Path, paper_dir: Path, settings_n: dict[str, int]) -> Path:
    """Write every generated paper input into `paper_dir/generated/`; return that directory."""
    out_dir = paper_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    macros, tables = build_numbers_and_tables(reports_dir, settings_n)
    tables |= build_prompt_assets(macros)
    tables["tab_evals"] = build_eval_table(settings_n)
    (out_dir / "numbers.tex").write_text(macros.text(), encoding="utf-8")
    for name, body in tables.items():
        (out_dir / f"{name}.tex").write_text("% Generated by cev_alignment.paper_assets. Do not edit.\n" + body,
                                             encoding="utf-8")
    (out_dir / "example.tex").write_text(build_example(log_root), encoding="utf-8")
    logger.info("Wrote %d macros and %d tables to %s", len(macros.defs), len(tables), out_dir)
    return out_dir
