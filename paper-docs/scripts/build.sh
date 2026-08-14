#!/usr/bin/env bash
# Build the manuscript end to end: figures, PDF, lint.
#
#   paper-docs/scripts/build.sh            # everything, preprint (no line numbers)
#   paper-docs/scripts/build.sh --review   # everything, with reviewer line numbers
#   paper-docs/scripts/build.sh figures    # figures only
#   paper-docs/scripts/build.sh pdf        # PDF only
#   paper-docs/scripts/build.sh lint       # lint only
#
# The committed snapshot at manuscript/dnagpt-fhe-paper.pdf is a preprint build.
set -euo pipefail

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PAPER}/.venv-paper"
PY="${VENV}/bin/python"
SRC="${PAPER}/manuscript/source"

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

run_lint() {
  ensure_venv
  echo "==> lint"
  "${PY}" "${PAPER}/scripts/check_numbers.py"
}

case "${TARGET}" in
  figures) build_figures ;;
  pdf)     build_pdf ;;
  lint)    run_lint ;;
  all)     build_figures; build_pdf; run_lint ;;
  *)       echo "usage: build.sh [all|figures|pdf|lint]" >&2; exit 2 ;;
esac
