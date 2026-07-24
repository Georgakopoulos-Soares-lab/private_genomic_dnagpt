#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BINARY="$(mktemp /tmp/dnagpt-fixture-contract.XXXXXX)"
trap 'rm -f "${BINARY}"' EXIT

"${CXX:-c++}" \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -I"${SCRIPT_DIR}/src" \
  "${SCRIPT_DIR}/src/fixture_contract.cpp" \
  -o "${BINARY}"
"${BINARY}"
