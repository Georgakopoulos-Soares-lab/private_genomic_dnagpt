#!/usr/bin/env bash
# Build the manuscript end to end: figures, PDF, lint.
#
#   paper-docs/scripts/build.sh            # everything
#   paper-docs/scripts/build.sh figures    # figures only
#   paper-docs/scripts/build.sh pdf        # PDF only
#   paper-docs/scripts/build.sh lint       # lint only
set -euo pipefail

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PAPER}/.venv-paper"
PY="${VENV}/bin/python"
SRC="${PAPER}/manuscript/source"
TARGET="${1:-all}"

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
  local mpl_cache="${TMPDIR:-/tmp}/dnagpt-paper-matplotlib"
  mkdir -p "${mpl_cache}"
  MPLCONFIGDIR="${mpl_cache}" "${PY}" "${PAPER}/scripts/figures.py"
}

build_pdf() {
  echo "==> pdf"
  # TACC exposes TeX Live through Lmod rather than on PATH. Load it when
  # available so the same build command works in an interactive shell or job.
  if ! command -v latexmk >/dev/null 2>&1 && ! command -v tectonic >/dev/null 2>&1; then
    if type module >/dev/null 2>&1; then
      module load texlive/2023 >/dev/null 2>&1 || true
    fi
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
