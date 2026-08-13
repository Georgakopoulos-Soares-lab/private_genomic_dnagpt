#!/usr/bin/env python3
"""Lint the manuscript against the evidence ledger.

Three checks, in order of how badly a reviewer would take them:

1. **Banned terms.** Internal shorthand (``Scheme B``, ``T123``, host names, job numbers,
   fixture and gate names) must not appear in ``manuscript/``. See
   ``context/00_terminology.md``.
2. **Unbacked numbers.** Every numeric literal in the ``.tex`` sources must appear in
   ``evidence/*.yaml``. Structural LaTeX numbers are ignored.
3. **Unverified citations.** ``refs.bib`` entries still marked ``UNVERIFIED``, and ``\\cite``
   keys with no matching entry.

    python paper-docs/scripts/check_numbers.py            # all checks
    python paper-docs/scripts/check_numbers.py --terms    # banned terms only

Exit code 0 clean, 1 on any finding.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper-docs"
EVIDENCE = PAPER / "evidence"
MANUSCRIPT = PAPER / "manuscript"
SOURCE = MANUSCRIPT / "source"

# --- 1. Banned terms --------------------------------------------------------
# Each entry: (regex, what to write instead). Case-insensitive, word-boundary anchored.

BANNED: list[tuple[str, str]] = [
    (r"\bscheme\s+a\b", "non-interactive CKKS"),
    (r"\bscheme\s+b\b", "client-assisted CKKS"),
    (r"\bscheme\s+c\b", "scheme switching"),
    (r"\bT123\b", "the named optimization"),
    (r"\bv[0-3]_(?:affinity|diag)\w*", "the named optimization"),
    (r"\bconfig[123]\b", "the named configuration"),
    (r"\bcpudiagcache\b", "the evaluated implementation"),
    (r"\bsimd_full_t103\w*", "the evaluated implementation"),
    (r"\bbrev\b", "one A100 GPU node"),
    (r"\bTACC\b", "one A100 GPU node"),
    (r"\blonestar\s*6\b|\bls6\b", "one A100 GPU node"),
    (r"\bslurm\b|\bsbatch\b", "(omit — scheduling is not a result)"),
    (r"\bjob\s+\d{6,}\b", "(omit — job identifiers are not results)"),
    (r"\bfixture\b", "the frozen plaintext reference"),
    (
        r"\bgate\s+token\b|\bmanifest\s+row\b|\brun\s+tag\b",
        "(omit — repository bookkeeping)",
    ),
    (r"\bsha-?256\b", "(omit — hashes are not scientific content)"),
    (r"\bround\s+trips?\b", "client boundary crossing"),
    (r"\bFIDESlib\b", "the evaluated GPU backend (cite it, do not name it inline)"),
]

# Terms that are fine in comments but not in body text. We only scan non-comment text.
COMMENT = re.compile(r"(?<!\\)%.*$")

# LaTeX writes thousands as 177{,}734 and 4{,}096; \, and ~ are also used as separators.
# Normalise before matching, or a single number lints as three.
TEX_SEPARATOR = re.compile(r"(?<=\d)(?:\{,\}|\\,|\\;|~)(?=\d)")


def strip_comments(text: str) -> str:
    text = "\n".join(COMMENT.sub("", line) for line in text.splitlines())
    return TEX_SEPARATOR.sub(",", text)


# --- 2. Numbers -------------------------------------------------------------

NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*)(?:\s*\\times\s*10\^\{?(-?\d+)\}?)?")

# LaTeX structure, not claims. These commands and their arguments are removed from a line
# before numbers are counted -- removed, not used to skip the whole line, or a claim sharing a
# line with a \cite would never be checked.
STRUCTURAL = (
    "documentclass|usepackage|geometry|setlength|hbadness|vbadness|hfuzz|vfuzz|"
    "titleformat|setlist|captionsetup|includegraphics|vspace|hspace|fontsize|linespacing|"
    "rowcolor|cmidrule|multirow|multicolumn|label|ref|eqref|cite[tp]?|input|graphicspath|"
    "newcommand|renewcommand|definecolor|colorlet|linewidth|textwidth|columnwidth|"
    "pdfgentounicode|newtheorem|hypersetup|figplaceholder|figplaceholderwide|"
    "bibliographystyle|bibliography|documentstyle|arraybackslash|raggedright|raggedleft"
)
IGNORE_CONTEXTS = re.compile(
    r"\\(?:" + STRUCTURAL + r")\*?(?:\[[^\]]*\])*(?:\{[^{}]*\})*"
)
# Column specifications and lengths carry numbers that are layout, not claims.
LAYOUT_NOISE = re.compile(
    r"\\begin\{tabularx\}\{[^}]*\}|\\begin\{minipage\}\{[^}]*\}|"
    r"\d+(?:\.\d+)?\s*(?:pt|cm|mm|em|ex|in|\\textwidth|\\linewidth|\\columnwidth)"
)

# Small integers used as ordinals, counts of list items, section numbers, etc.
ALWAYS_OK = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100"}


def ledger_numbers() -> set[str]:
    """Every numeric token the ledger authorises, in the renderings prose actually uses.

    The ledger is scanned exhaustively -- scalars, lists, and the prose in ``scope``, ``notes``,
    and ``statement`` fields, because a figure quoted in prose ("approximately 90 percent") is
    authorised by the same sentence that explains it.
    """
    out: set[str] = set()

    def add_scalar(v: float | int) -> None:
        av = abs(v)  # prose writes "20.1 percent lower", not "-20.1"
        for x in {v, av}:
            out.add(str(x))
            try:
                out.add(f"{x:,}")
            except (TypeError, ValueError):
                pass
            if isinstance(x, float):
                out.add(f"{x:g}")
                for places in range(0, 6):
                    out.add(f"{x:.{places}f}")
                    out.add(f"{x:,.{places}f}")
                mantissa, exponent = f"{x:e}".split("e")
                out.add(mantissa.rstrip("0").rstrip("."))
                out.add(str(abs(int(exponent))))
            else:
                # Seconds are routinely quoted as minutes or hours.
                out.add(f"{x / 60:.0f}")
                out.add(f"{x / 60:.1f}")
                out.add(f"{x / 3600:.1f}")
                out.add(f"{x / 3600:.2f}")

    def add(v) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            add_scalar(v)
        elif isinstance(v, str):
            for m in re.finditer(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?", v):
                token = m.group(0)
                out.add(token)
                out.add(token.replace(",", ""))
                try:
                    add_scalar(float(token.replace(",", "")))
                except ValueError:
                    pass
        elif isinstance(v, list):
            for item in v:
                add(item)
        elif isinstance(v, dict):
            walk(v)

    def walk(node) -> None:
        if isinstance(node, dict):
            for val in node.values():
                add(val) if not isinstance(val, dict) else walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item) if isinstance(item, (dict, list)) else add(item)

    for path in sorted(EVIDENCE.glob("*.yaml")):
        walk(yaml.safe_load(path.read_text()))

    out.update(ALWAYS_OK)
    return out


# --- 4. Structure -----------------------------------------------------------

# Files supplied by the TeX distribution rather than by this repository.
TEX_PROVIDED = {"glyphtounicode"}

INPUT = re.compile(r"\\input\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")


def check_structure() -> list[str]:
    """Everything the document points at must exist, and everything it defines should be used."""
    findings = []
    labels, refs, bodies = set(), {}, {}

    for path in tex_files():
        raw = path.read_text()
        bodies[path] = strip_comments(raw)

    for path, body in bodies.items():
        rel = path.relative_to(ROOT)
        for m in INPUT.finditer(body):
            target = m.group(1)
            if target in TEX_PROVIDED:
                continue
            if (
                not (SOURCE / target).exists()
                and not (SOURCE / f"{target}.tex").exists()
            ):
                findings.append(
                    f"{rel}: \\input{{{target}}} does not resolve to a file"
                )
        for m in GRAPHIC.finditer(body):
            name = m.group(1)
            if not any(
                (MANUSCRIPT / "figures" / n).exists()
                for n in (name, f"{name}.pdf", f"{name}.png")
            ):
                findings.append(
                    f"{rel}: figure '{name}' is missing -- run scripts/figures.py"
                )
        labels |= set(LABEL.findall(body))
        for lineno, line in enumerate(body.splitlines(), 1):
            for m in REF.finditer(line):
                refs.setdefault(m.group(1), (rel, lineno))

    for key, (rel, lineno) in refs.items():
        if key not in labels:
            findings.append(f"{rel}:{lineno}: \\ref{{{key}}} has no matching label")

    for fig in sorted((MANUSCRIPT / "figures").glob("*.pdf")):
        if not any(fig.stem in b for b in bodies.values()):
            findings.append(f"figures/{fig.name}: generated but never included")

    return findings


# --- 3. Citations -----------------------------------------------------------

CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}")
BIB_ENTRY = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)


def tex_files() -> list[pathlib.Path]:
    return sorted(SOURCE.rglob("*.tex"))


def prose_files() -> list[pathlib.Path]:
    """Files that carry claims. preamble.tex is pure layout and makes no assertions."""
    return [p for p in tex_files() if p.name != "preamble.tex"]


def check_terms() -> list[str]:
    findings = []
    for path in tex_files():
        body = strip_comments(path.read_text())
        for lineno, line in enumerate(body.splitlines(), 1):
            for pattern, replacement in BANNED:
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    rel = path.relative_to(ROOT)
                    findings.append(
                        f"{rel}:{lineno}: banned term {m.group(0)!r} — use {replacement}"
                    )
    return findings


def check_numbers() -> list[str]:
    allowed = ledger_numbers()
    findings = []
    for path in prose_files():
        body = strip_comments(path.read_text())
        for lineno, raw in enumerate(body.splitlines(), 1):
            line = LAYOUT_NOISE.sub(" ", IGNORE_CONTEXTS.sub(" ", raw))
            for m in NUMBER.finditer(line):
                token = m.group(1)
                if token in allowed or token.replace(",", "") in allowed:
                    continue
                rel = path.relative_to(ROOT)
                findings.append(
                    f"{rel}:{lineno}: {token!r} is not in evidence/*.yaml — "
                    f"add it to the ledger with a source, or remove it"
                )
    return findings


def check_citations() -> list[str]:
    findings = []
    bib = SOURCE / "refs.bib"
    if not bib.exists():
        return [f"{bib.relative_to(ROOT)}: missing"]
    text = bib.read_text()
    keys = set(BIB_ENTRY.findall(text))

    unverified = text.count("%% UNVERIFIED")
    if unverified:
        findings.append(
            f"refs.bib: {unverified} entr{'y' if unverified == 1 else 'ies'} still marked "
            f"UNVERIFIED — run the paper-sources agent before submission"
        )

    for path in tex_files():
        body = strip_comments(path.read_text())
        for lineno, line in enumerate(body.splitlines(), 1):
            for m in CITE.finditer(line):
                for key in (k.strip() for k in m.group(1).split(",")):
                    if key and key not in keys:
                        rel = path.relative_to(ROOT)
                        findings.append(
                            f"{rel}:{lineno}: \\cite{{{key}}} has no bib entry"
                        )
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terms", action="store_true", help="banned-term check only")
    ap.add_argument("--numbers", action="store_true", help="number check only")
    ap.add_argument("--citations", action="store_true", help="citation check only")
    ap.add_argument("--structure", action="store_true", help="structure check only")
    args = ap.parse_args()

    run_all = not (args.terms or args.numbers or args.citations or args.structure)
    blocks: list[tuple[str, list[str]]] = []
    if run_all or args.terms:
        blocks.append(("Banned terms", check_terms()))
    if run_all or args.numbers:
        blocks.append(("Numbers without evidence", check_numbers()))
    if run_all or args.citations:
        blocks.append(("Citations", check_citations()))
    if run_all or args.structure:
        blocks.append(("Structure", check_structure()))

    total = 0
    for title, findings in blocks:
        if findings:
            total += len(findings)
            print(f"\n{title} ({len(findings)}):")
            for f in findings:
                print(f"  {f}")
        else:
            print(f"{title}: clean")

    if total:
        print(f"\n{total} finding(s).")
        return 1
    print("\nAll checks clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
