#!/usr/bin/env bash
# Ensure the two encrypted-input fixtures the kimon configs need exist.
#
#   checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0
#       -> configs 1/2/3 "block" runs
#   checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0
#       -> configs 1/3 "e2e" runs (12 blocks + GSR head)
#
# Both are derived, never committed, and reproducible from two frozen inputs:
#   checkpoints/classification.pth  sha256 d63353abdc1a...
#   data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa  sha256 52d046d1fcf0...
# The exporters fail closed if either hash differs, and refuse to overwrite an
# existing fixture directory.
#
# Usage from the repo root:  kimon/env/prepare_fixtures.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
readonly EXPORTS="${REPO}/checkpoints/fhe_exports"
readonly BLOCK_FIX="${EXPORTS}/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
readonly MULTI_FIX="${EXPORTS}/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0"
readonly CKPT="${REPO}/checkpoints/classification.pth"
readonly FASTA="${REPO}/data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa"
readonly PY="${KIMON_PYTHON:-${REPO}/.venv/bin/python}"

have_block=0; have_multi=0
[[ -f "${BLOCK_FIX}/manifest.json" ]] && have_block=1
[[ -f "${MULTI_FIX}/manifest.json" ]] && have_multi=1
if [[ "${have_block}" -eq 1 && "${have_multi}" -eq 1 ]]; then
  echo "[OK] both fixtures present under ${EXPORTS}"
  exit 0
fi

# --- can we regenerate here? ------------------------------------------------
missing=()
[[ -f "${CKPT}" ]]  || missing+=("checkpoints/classification.pth")
[[ -f "${FASTA}" ]] || missing+=("data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa")
[[ -x "${PY}" ]]    || missing+=(".venv (python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt)")

if [[ "${#missing[@]}" -gt 0 ]]; then
  cat >&2 <<EOF
[FATAL] cannot regenerate the fixtures here; missing:
$(printf '  - %s\n' "${missing[@]}")

Either rehydrate those inputs:
  # GSR positives (DeepGSR, Zenodo record 1117159; 255 MB)
  mkdir -p data/gsr && curl -L -o data/gsr/Data.zip \\
    'https://zenodo.org/records/1117159/files/Data.zip?download=1'
  (cd data/gsr && unzip -o Data.zip 'Data/Human/PAS/*.fa')
  # released 0.1b weights (Google Drive folder 10UPPx6V13oQW6knuLV7d8SRIA3D6hYor)
  .venv/bin/gdown --folder 10UPPx6V13oQW6knuLV7d8SRIA3D6hYor -O checkpoints/

...or copy the already-generated fixtures from a machine that has them:
  rsync -av <host>:<repo>/checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0 \\
            <host>:<repo>/checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0 \\
            checkpoints/fhe_exports/
EOF
  exit 2
fi

cd "${REPO}"
if [[ "${have_block}" -eq 0 ]]; then
  echo "=== exporting block-0 T=103 fixture ==="
  PYTHONPATH=. "${PY}" -m fhe.realweights.export_fixture --tokens 103
fi
if [[ "${have_multi}" -eq 0 ]]; then
  echo "=== exporting 12-block + head T=103 fixture ==="
  PYTHONPATH=. "${PY}" -m fhe.multiblock.export_fixture_t103
fi

# --- verify against the frozen byte contracts -------------------------------
# A contract can fail on this platform even when the fixture is numerically
# correct: float64 @-matmul oracle/reference arrays differ in the last bits
# across BLAS/platform (the frozen hashes are Mac/Accelerate-origin). The
# arrays the encrypted C++ actually consumes -- weights + input embeddings --
# are platform-independent and must match. So we classify mismatches instead of
# a blind pass/fail.
echo "=== verifying fixture contracts ==="
_verify() {
  local dir="$1" contract="$2" label="$3"
  local out; out="$(cd "${dir}" && sha256sum --check "${contract}" 2>/dev/null)"
  local ok fail
  ok="$(printf '%s\n' "${out}" | grep -c ': OK$' || true)"
  fail="$(printf '%s\n' "${out}" | grep -c ': FAILED$' || true)"
  if [[ "${fail}" -eq 0 ]]; then
    echo "[OK] ${label}: all ${ok} arrays match the frozen contract"
    return 0
  fi
  echo "[MISMATCH] ${label}: ${ok} OK, ${fail} differ from the frozen contract:"
  printf '%s\n' "${out}" | grep ': FAILED$' | sed 's/^/    /'
  # A mismatch limited to oracle__*/manifest is a benign cross-platform float64
  # matmul difference; a mismatch in weights__*/input__* is a real problem.
  if printf '%s\n' "${out}" | grep ': FAILED$' | grep -qvE 'oracle__|manifest\.json'; then
    echo "    ^ includes a WEIGHT/INPUT array -- this is a real error, not platform float noise." >&2
    return 2
  fi
  echo "    (only oracle/reference arrays differ -> benign platform float64 matmul last-bits;" >&2
  echo "     the fixture is numerically self-consistent but will NOT pass the run scripts'" >&2
  echo "     frozen sha256 gate. Use canonical scp'd fixtures, or a platform-tagged contract.)" >&2
  return 1
}
rc=0
_verify "${BLOCK_FIX}" "${REPO}/fhe/gpu_real_scheme_b/fixture_t103.sha256" "block (config 1/3)" || rc=$?
_verify "${BLOCK_FIX}" "${REPO}/fhe/gpu_real_scheme_b/fixture_t103_qkvshard.sha256" "block qkvshard (config 2)" || rc=$?
_verify "${MULTI_FIX}" "${REPO}/fhe/gpu_real_scheme_b/fixture_all_blocks_head_t103.sha256" "multiblock (e2e)" || rc=$?
if [[ "${rc}" -eq 0 ]]; then
  echo "[OK] fixtures ready and byte-identical to the frozen contracts under ${EXPORTS}"
else
  echo "[WARN] fixtures generated under ${EXPORTS} but do not all match the frozen contracts (rc=${rc}); see above." >&2
fi
exit "${rc}"
