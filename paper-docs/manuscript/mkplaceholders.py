"""Replace table and algorithm floats with editable-context placeholders.

The Google Docs exchange copy carries prose only. Tables and algorithms stay in
LaTeX because their numbers are checked against paper-docs/evidence/*.yaml, and a
round trip through a word processor is exactly how an unbacked number gets in.
Each float becomes a marked block naming what it holds, so a reader editing the
surrounding text knows what sits there without being able to damage it.
"""

from __future__ import annotations

import pathlib
import re
import sys

# Order matters: LaTeX float counters increment in \input order, not file order.
INPUT_ORDER = [
    "00_frontmatter",
    "01_introduction",
    "02_background",
    "03_scenario_threat_model",
    "04_plaintext_baseline",
    "05_protocol",
    "06_optimization",
    "07_results",
    "09_related_work",
    "08_limitations",
    "10_conclusion",
]


def match_brace(text: str, open_idx: int) -> int:
    """Index just past the group opening at open_idx, honouring nesting."""
    depth = 0
    i = open_idx
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def extract_caption(body: str) -> str:
    m = re.search(r"\\caption\s*\{", body)
    if not m:
        return ""
    end = match_brace(body, m.end() - 1)
    raw = body[m.end() : end - 1]
    raw = re.sub(r"\\label\s*\{[^}]*\}", "", raw)
    raw = re.sub(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", r"\1", raw)
    raw = re.sub(r"\\[a-zA-Z]+", "", raw)
    raw = raw.replace("~", " ").replace("\\", "")
    # Placeholders are plain descriptive text. Leaving math delimiters or other
    # LaTeX specials in them breaks the pandoc parse for no benefit.
    raw = raw.replace("$", "")
    for ch in ("_", "&", "#", "%", "^", "{", "}"):
        raw = raw.replace(ch, " ")
    return re.sub(r"\s+", " ", raw).strip()


def describe_table(body: str) -> str:
    m = re.search(r"\\begin\{tabularx\}\{[^}]*\}\{([^}]*)\}", body)
    if not m:
        return "table body"
    cols = [c for c in m.group(1).split() if c]
    inner = body[m.end() :]
    inner = inner[: inner.find("\\end{tabularx}")]
    rows = len([r for r in inner.split("\\\\") if r.strip()])
    header = ""
    first = inner.split("\\\\")[0]
    if "&" in first:
        cells = [re.sub(r"\\[a-zA-Z]+|[{}]", "", c).strip() for c in first.split("&")]
        header = " | ".join(c for c in cells if c)
    desc = f"{rows} rows x {len(cols)} columns"
    return f"{desc}; columns: {header}" if header else desc


def describe_algorithm(body: str) -> str:
    steps = len(re.findall(r"\\State\b", body))
    return f"{steps} numbered steps of pseudocode"


def placeholder(kind: str, number: int, caption: str, detail: str) -> str:
    return (
        "\n\\begin{quote}\n"
        f"\\textbf{{[[ {kind} {number} — stays in LaTeX, do not edit here ]]}}\n\n"
        f"\\emph{{Caption:}} {caption}\n\n"
        f"\\emph{{Holds:}} {detail}\n"
        "\\end{quote}\n"
    )


def process(text: str, counters: dict[str, int]) -> str:
    pattern = re.compile(r"\\begin\{(table|algorithm)(\*?)\}")
    out = []
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        env = m.group(1) + m.group(2)
        end_tok = f"\\end{{{env}}}"
        end = text.find(end_tok, m.end())
        if end == -1:
            out.append(text[pos:])
            break
        body = text[m.end() : end]
        kind = "TABLE" if m.group(1) == "table" else "ALGORITHM"
        counters[kind] += 1
        detail = describe_table(body) if kind == "TABLE" else describe_algorithm(body)
        out.append(text[pos : m.start()])
        out.append(
            placeholder(kind, counters[kind], extract_caption(body) or "(none)", detail)
        )
        pos = end + len(end_tok)
    return "".join(out)


def main() -> int:
    sections = pathlib.Path(sys.argv[1])
    counters = {"TABLE": 0, "ALGORITHM": 0}
    for name in INPUT_ORDER:
        path = sections / f"{name}.tex"
        if not path.exists():
            continue
        before = dict(counters)
        text = path.read_text(encoding="utf-8")
        path.write_text(process(text, counters), encoding="utf-8")
        made = {k: counters[k] - before[k] for k in counters if counters[k] > before[k]}
        if made:
            print(f"  {name}: {made}")
    print(f"total: {counters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
