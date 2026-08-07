#!/usr/bin/env bash
# Build + install numpy==2.5.1 into the repo .venv on a TACC node whose
# hypervisor masks the LAHF CPUID bit.
#
# WHY THIS EXISTS
# ---------------
# numpy 2.5.1 pins its x86_64 baseline floor at the x86-64-v2 microarch level
# (meson_cpu/x86/meson.build: every SSE* feature maps to the X86_V2 group). Its
# RUNTIME baseline gate (numpy/_core/src/common/npy_cpu_features.c) computes
#     NPY_CPU_FEATURE_X86_V2 = SSE && SSE2 && SSE3 && SSSE3 && SSE41 && SSE42
#                              && POPCNT && [CX16 &&] LAHF
# The gpu-a100-small nodes here run under a hypervisor that does NOT advertise
# LAHF in CPUID (Fn8000_0001_ECX[0]), even though every other v2 feature and the
# LAHF instruction itself are present on the EPYC 7763 silicon. So the stock
# manylinux wheel AND any from-source build abort at import with
#   "NumPy was built with baseline optimizations (X86_V2) but your machine
#    doesn't support (X86_V2)."
# See memory: tacc-ls6-env-quirks.
#
# THE FIX (numerically identical -> fixtures stay byte-reproducible)
# ------------------------------------------------------------------
# Drop the `&& LAHF` term from the RUNTIME X86_V2 group gate only. LAHF is a
# legacy flags-register instruction (Load AH from Flags); numpy's vectorised
# float64/float32 kernels never emit it. We do NOT touch codegen, the compiled
# baseline flags, or any SSE/AVX kernel -- only the boolean that decides whether
# the (already-correct-for-this-CPU) baseline is "allowed" to run. numpy still
# honestly reports LAHF=0 in __cpu_features__. Result: identical numerics to a
# normal v2 numpy 2.5.1.
#
# Usage (from repo root):  kimon/env/build_numpy_patched.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
readonly VENV="${REPO}/.venv"
# Base scratch on TACC's $SCRATCH (real Lustre scratch), not $(id -u) which is
# the harness container uid (903286 -> unwritable /scratch/903286).
_scratch="${SCRATCH:-$(cd "${REPO}/.." && pwd)}"
readonly WORK="${NUMPY_BUILD_DIR:-${_scratch}/kimon_numpy_build}"
readonly VER=2.5.1

command -v gcc  >/dev/null 2>&1 || { echo "[FATAL] source kimon/env/tacc_env.sh first (needs gcc 13.2 + python 3.12)"; exit 2; }
[[ -x "${VENV}/bin/python" ]] || { echo "[FATAL] no .venv at ${VENV}"; exit 2; }
export CC=gcc CXX=g++
export TMPDIR="${TMPDIR:-${_scratch}/kimon_tmp}"; mkdir -p "${TMPDIR}"

rm -rf "${WORK}"; mkdir -p "${WORK}"; cd "${WORK}"
echo "[1/4] fetching numpy ${VER} sdist"
"${VENV}/bin/pip" download --no-binary numpy --no-deps "numpy==${VER}" -d . >/dev/null
tar xf "numpy-${VER}.tar.gz"
readonly SRC="${WORK}/numpy-${VER}"
readonly CPUF="${SRC}/numpy/_core/src/common/npy_cpu_features.c"

echo "[2/4] patching runtime X86_V2 gate to not require LAHF"
# The gate's final term is exactly this line (see the multi-line expression that
# begins `npy__cpu_have[NPY_CPU_FEATURE_X86_V2] = ...`).
before="npy__cpu_have[NPY_CPU_FEATURE_LAHF];"
after="1 /* LAHF term dropped: TACC hypervisor masks the LAHF CPUID bit; see kimon/env/build_numpy_patched.sh */;"
count="$(grep -c -F "${before}" "${CPUF}" || true)"
if [[ "${count}" -ne 1 ]]; then
  echo "[FATAL] expected exactly 1 occurrence of the X86_V2 LAHF term, found ${count}." >&2
  echo "        numpy layout changed; re-audit npy_cpu_features.c before building." >&2
  exit 3
fi
python3 - "$CPUF" "$before" "$after" <<'PY'
import sys
p, before, after = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
assert s.count(before) == 1, "guard already checked; refusing"
open(p, "w").write(s.replace(before, after))
print("    patched:", p)
PY
# Show the patched gate for the run log.
grep -n -A12 "npy__cpu_have\[NPY_CPU_FEATURE_X86_V2\] =" "${CPUF}" | head -14

echo "[3/4] building + installing numpy ${VER} (baseline X86_V2 minus LAHF)"
"${VENV}/bin/pip" install --force-reinstall --no-deps --no-cache-dir \
  --config-settings=setup-args=-Dcpu-baseline=SSE2 \
  --config-settings=setup-args=-Dcpu-dispatch=max \
  "${SRC}"

echo "[4/4] verifying import + baseline"
"${VENV}/bin/python" - <<PY
import numpy as np
from numpy._core._multiarray_umath import __cpu_baseline__, __cpu_features__
assert np.__version__ == "${VER}", np.__version__
print("numpy", np.__version__)
print("baseline", __cpu_baseline__)
print("LAHF reported:", __cpu_features__.get("LAHF"))
print("norm check:", float(np.linalg.norm(np.arange(9.0).reshape(3,3))))
PY
echo "[ok] numpy ${VER} usable on this node"
