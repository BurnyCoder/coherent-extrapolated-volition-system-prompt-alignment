"""Build paper/references.bib from identifiers only, never from memory.

Local context: every paper is fetched by DOI content negotiation (https://doi.org/<doi> with
`Accept: application/x-bibtex`; arXiv papers use the DataCite DOI 10.48550/arXiv.<id>), and its fetched title must
contain an expected fragment, so a wrong identifier fails loudly instead of citing the wrong work. Web sources
(reports, repositories, documentation) get a minimal @misc entry typed from the source page, whose URL must answer
HTTP 200 (their fields are not fetched, so each is commented with where it was checked). Anything that cannot be
verified is listed at the end and must not be cited.

Global context: run once by hand (`uv run python paper/tools/fetch_references.py`); `paper/main.tex` uses the
output with natbib. Docs: https://www.crossref.org/documentation/retrieve-metadata/content-negotiation/ ,
https://info.arxiv.org/help/doi.html
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[1] / "references.bib"

# key -> (DOI, fragment that must occur in the fetched title, case-insensitive)
PAPERS: dict[str, tuple[str, str]] = {
    "touvron2023llama2": ("10.48550/arXiv.2307.09288", "Llama 2"),
    "askell2021general": ("10.48550/arXiv.2112.00861", "General Language Assistant"),
    "bai2022constitutional": ("10.48550/arXiv.2212.08073", "Constitutional AI"),
    "perez2022discovering": ("10.48550/arXiv.2212.09251", "Model-Written Evaluations"),
    "kran2025darkbench": ("10.48550/arXiv.2503.10728", "DarkBench"),
    "sturgeon2025humanagencybench": ("10.48550/arXiv.2509.08494", "HumanAgencyBench"),
    "he2025instrumentaleval": ("10.48550/arXiv.2502.12206", "Instrumental"),
    "lynch2025agentic": ("10.48550/arXiv.2510.05179", "Agentic Misalignment"),
    "ren2025mask": ("10.48550/arXiv.2503.03750", "MASK"),
    "zheng2024safeguarding": ("10.48550/arXiv.2401.18018", "Prompt-Driven Safeguarding"),
    "zheng2024helpful": ("10.48550/arXiv.2311.10054", "Helpful Assistant"),
    "xie2023selfreminder": ("10.1038/s42256-023-00765-8", "self-reminder"),
    "miller2024errorbars": ("10.48550/arXiv.2411.00640", "Error Bars"),
    "bowyer2025position": ("10.48550/arXiv.2503.01747", "CLT in LLM Evals"),
    "mcnemar1947": ("10.1007/BF02295996", "sampling error"),
    "wilcoxon1945": ("10.2307/3001968", "Individual Comparisons"),
    "holm1979": ("10.2307/4615733", "Multiple Test Procedure"),
}

# DOIs that OpenAlex lists but doi.org does not resolve (HTTP 404, checked 2026-10-17): cite the publisher URL.
UNREGISTERED_DOI_URL = {"holm1979": "https://www.jstor.org/stable/4615733"}
# Page ranges missing from both Crossref and OpenAlex, read from the publisher page.
PAGES = {"wilcoxon1945": "80--83"}  # https://www.jstor.org/stable/3001968 (Biometrics Bulletin 1(6), pp. 80-83)

# key -> BibTeX fields for web sources; the URL must be reachable.
WEB: dict[str, dict[str, str]] = {
    "yudkowsky2004cev": {
        "type": "techreport", "author": "Yudkowsky, Eliezer", "title": "Coherent Extrapolated Volition",
        "institution": "The Singularity Institute", "address": "San Francisco, CA", "year": "2004",
        "url": "https://intelligence.org/files/CEV.pdf"},
    # "developed by the UK AI Security Institute and Meridian Labs" (https://inspect.aisi.org.uk/, 2026-10-17)
    "inspect2024": {
        "type": "misc", "author": "{UK AI Security Institute and Meridian Labs}",
        "title": "Inspect: A framework for large language model evaluations",
        "year": "2024", "url": "https://inspect.aisi.org.uk/"},
    "inspectevals2024": {
        "type": "misc", "author": "{UK AI Security Institute and contributors}", "title": "Inspect Evals",
        "year": "2024", "url": "https://github.com/UKGovernmentBEIS/inspect_evals"},
    # Upstream README asks to credit "HumaneBench, Building Humane Tech" (README.md at commit 1889469).
    "humanebench2025": {
        "type": "misc", "author": "{Building Humane Tech}", "title": "HumaneBench", "year": "2025",
        "url": "https://github.com/buildinghumanetech/humanebench"},
    # Page title "Claude's Constitution"; announced 2026-01-22 at anthropic.com/news/claude-new-constitution.
    "anthropic2026constitution": {
        "type": "misc", "author": "{Anthropic}", "title": "Claude's Constitution", "year": "2026",
        "url": "https://www.anthropic.com/constitution"},
    "openrouter": {
        "type": "misc", "author": "{OpenRouter}", "title": "OpenRouter: a unified API for large language models",
        "year": "2026", "url": "https://openrouter.ai/"},
    "huxley1868chalk": {
        "type": "misc", "author": "Huxley, Thomas Henry", "title": "On a Piece of Chalk",
        "howpublished": "In \\emph{Lay Sermons, Addresses and Reviews} (1870); Project Gutenberg eBook \\#16729",
        "year": "1868", "url": "https://www.gutenberg.org/ebooks/16729"},
}

ACCESSED = "Accessed 2026-10-17"


def _check_title(key: str, title: str | None, fragment: str) -> None:
    """Raise if the fetched title does not contain the expected fragment (wrong identifier)."""
    if not title or fragment.lower() not in title.lower():
        raise ValueError(f"{key}: fetched title {title!r} lacks {fragment!r}")


def openalex_bibtex(client: httpx.Client, key: str, doi: str, fragment: str) -> str:
    """@article built from OpenAlex metadata, for any DOI where doi.org answers 404 (e.g. JSTOR 10.2307/...).

    OpenAlex API: https://docs.openalex.org/api-entities/works/get-a-single-work
    """
    work = client.get(f"https://api.openalex.org/works/doi:{doi}").raise_for_status().json()
    _check_title(key, work.get("title"), fragment)
    biblio = work.get("biblio") or {}
    pages = "--".join(p for p in (biblio.get("first_page"), biblio.get("last_page")) if p)
    fields = {
        "author": " and ".join(a["author"]["display_name"] for a in work["authorships"]),
        "title": work["title"],
        "journal": (work.get("primary_location") or {}).get("source", {}).get("display_name", ""),
        "year": str(work["publication_year"]), "volume": biblio.get("volume") or "",
        "number": biblio.get("issue") or "",
        "pages": PAGES.get(key, pages), "doi": doi,
    }
    if key in UNREGISTERED_DOI_URL:  # an unresolvable DOI would be a dead link in the paper
        fields["url"] = UNREGISTERED_DOI_URL[key]
        del fields["doi"]
    fields["title"] = "{" + fields["title"] + "}"  # brace-protect capitalisation
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@article{{{key},\n{body}\n}}"


def fetch_bibtex(client: httpx.Client, key: str, doi: str, fragment: str) -> str:
    """BibTeX for `doi` with its citation key replaced by `key`; raises if the title lacks `fragment`."""
    response = client.get(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"})
    if response.status_code == 404:  # registered DOI without content negotiation -> OpenAlex metadata
        return openalex_bibtex(client, key, doi, fragment)
    response.raise_for_status()
    bib = response.text.strip()
    # DataCite puts one field per line; CrossRef puts all fields on one line ("title={...}, volume=...").
    title = re.search(r"\btitle\s*=\s*\{(.+?)\}\s*,", bib, flags=re.IGNORECASE | re.DOTALL)
    _check_title(key, title.group(1) if title else None, fragment)
    # Brace-protect the title so BibTeX styles keep the fetched capitalisation (LLMs, AI, DarkBench, ...).
    bib = bib.replace(title.group(0), title.group(0).replace("{" + title.group(1) + "}", "{{" + title.group(1) + "}}"))
    bib = complete_pages(client, doi, bib)
    if key in PAGES:
        bib = re.sub(r"pages\s*=\s*\{[^}]*\}", f"pages={{{PAGES[key]}}}", bib)
    return re.sub(r"^@(\w+)\{[^,]*,", lambda m: f"@{m.group(1).lower()}{{{key},", bib, count=1)


def complete_pages(client: httpx.Client, doi: str, bib: str) -> str:
    """CrossRef sometimes gives only a first page (pages={80}); take the full range from OpenAlex when it has one."""
    single = re.search(r"pages\s*=\s*\{(\d+)\}", bib)
    if not single:
        return bib
    biblio = client.get(f"https://api.openalex.org/works/doi:{doi}").raise_for_status().json().get("biblio") or {}
    if biblio.get("first_page") == single.group(1) and biblio.get("last_page") not in (None, single.group(1)):
        return bib.replace(single.group(0), f"pages={{{single.group(1)}--{biblio['last_page']}}}")
    return bib


def web_entry(client: httpx.Client, key: str, fields: dict[str, str]) -> str:
    """Minimal @misc/@techreport entry; raises if its URL is unreachable."""
    client.get(fields["url"], follow_redirects=True).raise_for_status()
    body = ",\n".join(f"  {k} = {{{{{v}}}}}" if k == "title" else f"  {k} = {{{v}}}"
                       for k, v in fields.items() if k != "type")
    return f"@{fields['type']}{{{key},\n{body},\n  note = {{{ACCESSED}}}\n}}"


def main() -> int:
    entries, failures = [], []
    headers = {"User-Agent": "cev-alignment-paper (mailto:burian.lib@gmail.com)"}
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        for key, (doi, fragment) in PAPERS.items():
            try:
                entries.append(fetch_bibtex(client, key, doi, fragment))
                print(f"OK   {key}  {doi}")
            except (httpx.HTTPError, ValueError) as error:
                failures.append(f"{key}: {error}")
                print(f"FAIL {key}  {error}")
        for key, fields in WEB.items():
            try:
                entries.append(web_entry(client, key, fields))
                print(f"OK   {key}  {fields['url']}")
            except httpx.HTTPError as error:
                failures.append(f"{key}: {error}")
                print(f"FAIL {key}  {error}")
    header = "% Generated by paper/tools/fetch_references.py: DOI content negotiation + URL checks. Do not edit.\n\n"
    OUT.write_text(header + "\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"\nWrote {len(entries)} entries to {OUT}")
    if failures:
        print("UNVERIFIED (do not cite):\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
