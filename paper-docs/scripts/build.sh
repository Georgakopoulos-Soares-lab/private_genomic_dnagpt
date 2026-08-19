#!/usr/bin/env bash
# Build the manuscript end to end: figures, PDF, lint.
#
#   paper-docs/scripts/build.sh            # everything, preprint (no line numbers)
#   paper-docs/scripts/build.sh --review   # everything, with reviewer line numbers
#   paper-docs/scripts/build.sh figures    # figures only
#   paper-docs/scripts/build.sh pdf        # PDF only
#   paper-docs/scripts/build.sh lint       # lint only
#   paper-docs/scripts/build.sh docx       # Google Docs review exchange copy
#
# The committed snapshot at manuscript/dnagpt-fhe-paper.pdf is a preprint build.
# The docx is a review exchange copy, not a source file: it is gitignored and is
# regenerated from source whenever the manuscript changes.
set -euo pipefail

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PAPER}/.venv-paper"
PY="${VENV}/bin/python"
SRC="${PAPER}/manuscript/source"
FIGDIR="${PAPER}/manuscript/figures"
EXCHANGE="${PAPER}/manuscript/exchange"

REVIEW=0
ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --review)   REVIEW=1 ;;
    --preprint) REVIEW=0 ;;
    *)          ARGS+=("${arg}") ;;
  esac
done
TARGET="${ARGS[0]:-all}"

ensure_venv() {
  if [[ ! -x "${PY}" ]]; then
    echo "Creating the paper virtual environment at ${VENV}"
    python3 -m venv "${VENV}"
    "${VENV}/bin/pip" install --quiet --upgrade pip
    "${VENV}/bin/pip" install --quiet -r "${PAPER}/requirements.txt"
  fi
}

build_figures() {
  ensure_venv
  echo "==> figures"
  "${PY}" "${PAPER}/scripts/figures.py"
}

build_pdf() {
  echo "==> pdf"
  # Regenerate the build-mode flag every time so a stale file can never leak
  # reviewer line numbers into a preprint build, or vice versa.
  if [[ "${REVIEW}" -eq 1 ]]; then
    echo '\reviewbuildtrue' > "${SRC}/buildmode.tex"
    echo "    mode: reviewer (line numbers on)"
  else
    echo '\reviewbuildfalse' > "${SRC}/buildmode.tex"
    echo "    mode: preprint (no line numbers)"
  fi
  if command -v latexmk >/dev/null 2>&1; then
    ( cd "${SRC}" && latexmk -pdf main.tex )
    echo "    ${SRC}/main.pdf"
    return 0
  fi
  # Tectonic needs no TeX distribution and fetches the packages it needs on first
  # run. It resolves citations and cross-references in a single invocation.
  if command -v tectonic >/dev/null 2>&1; then
    echo "    latexmk not found; using tectonic"
    ( cd "${SRC}" && tectonic -o "${SRC}" main.tex )
    echo "    ${SRC}/main.pdf"
    return 0
  fi
  echo "No LaTeX engine found. Install either:" >&2
  echo "  brew install tectonic                      (small, self-contained)" >&2
  echo "  brew install --cask mactex-no-gui          (full TeX Live)" >&2
  echo "  apt install texlive-full latexmk           (Debian)" >&2
  return 1
}

# The exchange copy carries prose only. Tables and algorithms become placeholders
# (mkplaceholders.py) because their numbers are checked against evidence/*.yaml and a
# round trip through a word processor is how an unbacked number gets in. Figures are
# rasterised first: pandoc will happily embed a PDF that no word processor renders.
build_docx() {
  for tool in pandoc pdftoppm; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "docx needs ${tool}. brew install pandoc poppler" >&2
      return 1
    fi
  done
  echo "==> docx"
  local stage
  stage="$(mktemp -d)"
  trap 'rm -rf "${stage}"' RETURN

  cp -R "${SRC}" "${stage}/source"
  cp -R "${FIGDIR}" "${stage}/figures"
  rm -f "${stage}/source/main.pdf" "${stage}/source"/*.aux "${stage}/source"/*.bbl

  for f in "${stage}/figures"/*.pdf; do
    pdftoppm -png -r 300 -singlefile "${f}" "${stage}/figures/$(basename "${f}" .pdf)"
  done
  # Point \includegraphics at the rasterised copies, in the staged tree only.
  sed -i.bak -E 's/(includegraphics\[[^]]*\]\{fig_[a-z_]+)\.pdf\}/\1.png}/' \
    "${stage}/source/sections"/*.tex
  rm -f "${stage}/source/sections"/*.bak

  python3 "${PAPER}/manuscript/mkplaceholders.py" "${stage}/source/sections"

  mkdir -p "${EXCHANGE}"
  ( cd "${stage}/source" && pandoc main.tex \
      -o "${EXCHANGE}/dnagpt-fhe-paper.docx" \
      --bibliography=refs.bib --citeproc \
      --resource-path=.:../figures )
  echo "    ${EXCHANGE}/dnagpt-fhe-paper.docx"
}

run_lint() {
  ensure_venv
  echo "==> lint"
  "${PY}" "${PAPER}/scripts/check_numbers.py"
}

case "${TARGET}" in
  figures) build_figures ;;
  pdf)     build_pdf ;;
  lint)    run_lint ;;
  docx)    build_docx ;;
  all)     build_figures; build_pdf; run_lint ;;
  *)       echo "usage: build.sh [all|figures|pdf|lint|docx]" >&2; exit 2 ;;
esac
