from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEME_B = ROOT / "fhe" / "gpu_real_scheme_b"
SOURCE = SCHEME_B / "src"
PARENT = SOURCE / "real_dnagpt_fides_scheme_b.cpp"
DEPTH12 = SOURCE / "real_dnagpt_fides_scheme_b_depth12.cpp"
WRAPPER_PARENT = SOURCE / "real_dnagpt_fides_scheme_b_two_block_refresh.cpp"
WRAPPER = SOURCE / "real_dnagpt_fides_scheme_b_two_block_refresh_depth12.cpp"
RUN = SCHEME_B / "run_scheme_b_two_block_refresh_depth12.sh"
LAUNCH = SCHEME_B / "launch_brev_scheme_b_two_block_refresh_depth12.sh"
WAIT = SCHEME_B / "wait_and_run_scheme_b_two_block_refresh_depth12.sh"
CMAKE = SCHEME_B / "CMakeLists.txt"
BUILD = SCHEME_B / "build_in_fideslib.sh"

PARENT_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
WRAPPER_PARENT_SHA256 = (
    "5cb82f81dc46efe2f4bfd7a1c9e9c1bf90cea3d629432b027b42e5df89093767"
)
DEPTH12_SHA256 = "234e3659eb73685619fede1b95dcf637c72f5b504e54a3631dfb1ab4ce0d5397"
HEADER = (
    "// Depth-12 fork for the Scheme B minimum-context gate.\n"
    "// Parent: real_dnagpt_fides_scheme_b.cpp at SHA-256\n"
    f"// {PARENT_SHA256}.\n"
    "// The contract test requires the parent body to differ only in MULT_DEPTH\n"
    "// and implementation_version.\n\n"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Depth12SourceContractTests(unittest.TestCase):
    def test_frozen_parents_are_byte_identical(self) -> None:
        self.assertEqual(sha256(PARENT), PARENT_SHA256)
        self.assertEqual(sha256(WRAPPER_PARENT), WRAPPER_PARENT_SHA256)

    def test_depth12_single_block_is_exactly_the_declared_two_field_fork(self) -> None:
        expected = HEADER + PARENT.read_text(encoding="utf-8")
        expected = expected.replace(
            "constexpr std::uint32_t MULT_DEPTH = 16;",
            "constexpr std::uint32_t MULT_DEPTH = 12;",
            1,
        )
        expected = expected.replace(
            "t2-scheme-b-v2-batched",
            "t2-scheme-b-v2-batched-depth12",
            1,
        )
        self.assertEqual(DEPTH12.read_text(encoding="utf-8"), expected)
        self.assertEqual(sha256(DEPTH12), DEPTH12_SHA256)

    def test_depth12_is_the_exact_conservative_guard_threshold(self) -> None:
        text = DEPTH12.read_text(encoding="utf-8")
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 12;", text)
        self.assertIn(
            "normalized2[token]->GetLevel(), 4,\n"
            '                "MLP token " + std::to_string(token)',
            text,
        )
        self.assertIn(
            "require_remaining_depth(maximum_level(block_output), 1,\n"
            '                                "final output packing")',
            text,
        )
        # The immutable depth-16 two-block result measured LN2 max=8.
        # This establishes 8+4=12 as the first conservative-guard probe;
        # the real GPU run must still verify actual context generation and accuracy.
        self.assertEqual(8 + 4, 12)

    def test_wrapper_uses_only_the_depth12_evaluator(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('#include "real_dnagpt_fides_scheme_b_depth12.cpp"', text)
        self.assertNotIn('#include "real_dnagpt_fides_scheme_b.cpp"', text)
        self.assertIn(DEPTH12_SHA256, text)
        self.assertIn(PARENT_SHA256, text)
        self.assertIn('"multiplicative_depth\\": " << MULT_DEPTH', text)
        self.assertIn(
            "REAL_DNAGPT_FIDES_SCHEME_B_TWO_BLOCK_REFRESH_DEPTH12_PASS",
            text,
        )
        main_defs = re.findall(r"^int main\(", text, flags=re.MULTILINE)
        self.assertEqual(len(main_defs), 1)

    def test_build_run_and_capacity_gates_are_isolated(self) -> None:
        target = "real_dnagpt_fides_scheme_b_two_block_refresh_depth12"
        self.assertIn(target, CMAKE.read_text(encoding="utf-8"))
        self.assertIn(target, BUILD.read_text(encoding="utf-8"))
        run = RUN.read_text(encoding="utf-8")
        self.assertIn(target, run)
        self.assertIn("EXPECTED_PARENT_SOURCE", run)
        self.assertIn("EXPECTED_DEPTH12_SOURCE", run)
        self.assertIn("refusing to overwrite immutable evidence", run)
        self.assertIn('*"_depth12_"*', run)
        launch = LAUNCH.read_text(encoding="utf-8")
        self.assertIn("gpucap_preflight_confirm_gpu", launch)
        self.assertIn("run_scheme_b_two_block_refresh_depth12.sh", launch)
        wait = WAIT.read_text(encoding="utf-8")
        self.assertIn("gpucap_wait_for_capacity", wait)
        self.assertIn("launch_brev_scheme_b_two_block_refresh_depth12.sh", wait)
        self.assertIn("refusing to reuse orchestrator log", wait)


if __name__ == "__main__":
    unittest.main()
