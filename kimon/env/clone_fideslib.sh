#!/usr/bin/env bash
# Clone FIDESlib at the pinned commit (with submodules) for the native build.
# Idempotent: re-running on an existing checkout only verifies the commit.
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly SRC="${FIDESLIB_ROOT:-${SCRIPT_DIR}/fideslib-src}"

if [[ -d "${SRC}/.git" ]]; then
  actual="$(git -C "${SRC}" rev-parse HEAD)"
  if [[ "${actual}" != "${EXPECTED_COMMIT}" ]]; then
    echo "[FATAL] existing checkout at ${SRC} is ${actual}; expected ${EXPECTED_COMMIT}" >&2
    exit 2
  fi
  echo "[ok] FIDESlib already at ${EXPECTED_COMMIT}"
  exit 0
fi

git clone https://github.com/CAPS-UMU/FIDESlib.git "${SRC}"
git -C "${SRC}" checkout "${EXPECTED_COMMIT}"
git -C "${SRC}" submodule update --init --recursive
echo "[ok] HEAD=$(git -C "${SRC}" rev-parse HEAD)"
