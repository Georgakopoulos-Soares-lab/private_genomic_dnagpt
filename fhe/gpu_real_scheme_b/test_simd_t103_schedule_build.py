"""Compile and execute the FIDES-independent C++ T=103 SIMD contract."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).parent
SOURCE = ROOT / "test_simd_t103_schedule.cpp"
HEADER = ROOT / "src" / "simd_t103_schedule.hpp"


class SimdT103CppScheduleBuildTests(unittest.TestCase):
    def test_cpp_schedule_contract_compiles_and_passes(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("no local C++ compiler")
        with tempfile.TemporaryDirectory(prefix="dnagpt-simd-contract-") as tmp:
            binary = Path(tmp) / "simd_t103_schedule_contract"
            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT),
                    str(SOURCE),
                    "-o",
                    str(binary),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                msg=compile_result.stdout + compile_result.stderr,
            )
            run_result = subprocess.run(
                [str(binary)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                run_result.returncode,
                0,
                msg=run_result.stdout + run_result.stderr,
            )
            self.assertEqual(
                run_result.stdout.strip(),
                "SIMD_T103_CPP_SCHEDULE_CONTRACT_PASS",
            )

    def test_header_and_source_are_additive(self) -> None:
        self.assertTrue(HEADER.is_file())
        self.assertTrue(SOURCE.is_file())
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("simd_t103_schedule_contract", cmake)


if __name__ == "__main__":
    unittest.main()
