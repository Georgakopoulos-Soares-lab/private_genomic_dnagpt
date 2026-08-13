// Complete released DNAGPT graph under Scheme B Token-SIMD (B=8) packing at
// T=103, CONFIG-3 "all-optimizations" variant, T123_V3 per-block lever:
// encrypted embeddings -> 12 transformer blocks -> encrypted N-minus-A GSR
// classifier margin, composing the T123_V3 per-block source (CPU-side
// diagonal-vector cache + Tier1 encode-once encoded-Plaintext templates with
// per-use clone + Tier2/3 OMP-parallel batched encode + v3's post-LoadContext
// free of the CPU-side OpenFHE EvalMult/rotation key maps) instead of the
// plain cpudiagcache block. T=103 is the COMPLETE tokenized GSR prompt for
// this record (not a truncated prefix like the T=2 driver), so this is a
// real-task-semantics correctness/composition gate, not only a graph gate.
//
// This file is an ADDITIVE fork of the config-3 "all-optimizations" e2e
// driver real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache.cpp
// (pinned SHA-256 below, design/composition parent, NOT included). Its
// composition logic -- one crypto context/key lineage, 12 blocks, eleven
// full-hidden-state inter-block refreshes, one block-11->head last-token
// refresh, and the GSR N-minus-A head -- is BYTE-FOR-BYTE identical to that
// parent. The ONLY differences are (1) which per-block source it #includes
// (T123_V3 instead of plain cpudiagcache) and (2) one addition in this file's
// own main(): the same post-LoadContext ClearEvalMultKeys()/
// ClearEvalAutomorphismKeys() call T123_V3 makes in its own (unused) main(),
// mirrored here because this driver's main() duplicates its own context
// setup rather than calling the included file's. T123_V3's per-block class
// is a strict superset of the plain cpudiagcache block -- same
// EncryptedEvaluator/Client/Fixture/EvaluationResult/evaluate() symbol
// surface, same op counts and boundary counts -- differing only by an
// internal, class-private encoded-Plaintext template cache (flushed at each
// stage boundary WITHIN evaluate(), and implicitly flushed between blocks
// because this driver -- like its parent -- constructs a brand-new
// EncryptedEvaluator instance per block iteration; see
// "T123 tier composition across blocks" below), so the #include-swap changes
// nothing about the composed encrypted graph or its crypto-op schedule.
//
// T123 tier composition across blocks (why this is safe): T123's Tier-1/2/3
// encoded-Plaintext cache (diagonal_plain_cache_) is a private member of
// EncryptedEvaluator, flushed at the QKV->attention-projection and
// attention-projection->MLP stage boundaries inside evaluate() (see
// flush_diagonal_plains() in the included per-block source). The 12-block
// loop below (copied verbatim from the parent driver) constructs a FRESH
// EncryptedEvaluator instance for every block iteration and lets it go out
// of scope at the end of that iteration, so diagonal_plain_cache_ is
// destroyed (and its memory released) at every block boundary regardless of
// the intra-block flushes. No cache state, and
// no other T123/v3-introduced state, crosses a block or a client-refresh
// boundary; the free-host-keys call in main() runs exactly once, at global
// context setup, before any block executes. This composes safely with the
// declared client refresh protocol.
//
// Fork discipline:
//   - includes the T123_V3 depth-13/digits-3/ring-65536 Token-SIMD B=8 T=103
//     single-block source unchanged (pinned SHA-256 below) and reuses its
//     EncryptedEvaluator/Client/Fixture/free-function per-block logic
//     exactly as-is, 12 times, one instantiation per released block;
//   - adapts the T=2 two-block-refresh source's design (pinned SHA-256
//     below, not included -- its layout is incompatible with Token-SIMD
//     packing) to Token-SIMD's 13-ciphertext-per-hidden-state layout: T=2's
//     refresh decrypts ONE packed ciphertext and re-encrypts T=2 individual
//     per-token ciphertexts; this driver's refresh decrypts ALL 13 token
//     GROUP ciphertexts, recovers the exact T=103 x D=768 hidden state in
//     the clear at the client (the data owner, who already holds the secret
//     key -- this is not a privacy violation, it is the declared hybrid
//     boundary), and re-encrypts 13 fresh level-0 group ciphertexts;
//   - reuses one crypto context and one Client instance across all 12
//     blocks and the head (matching both the T=2 all-blocks-plus-head
//     driver and the T=2 two-block-refresh gate's "same crypto context and
//     key lineage" pattern);
//   - performs eleven declared full-hidden-state refreshes between blocks
//     and one declared block-11-to-head last-token refresh (the head only
//     needs the LAST of the 103 tokens: token index 102, which lives in
//     token group 12 at lane 6 -- the final token group only has 7 active
//     lanes, 0..6, not 8);
//   - a new EncryptedHeadEvaluator (no equivalent exists in the frozen
//     per-block source, which has no classifier-head logic at all) applies
//     the released final LayerNorm, head linear, SiLU (client boundary),
//     head LayerNorm, and the encrypted N-minus-A readout dot product to
//     that single-token group. It reuses the Token-SIMD BSGS/
//     physical_slot/AccumulateSum conventions and the shared Client's
//     public invsqrt_boundary() for its two LayerNorms; SiLU has no
//     equivalent public Client method (depth13's Client only exposes
//     invsqrt_boundary/gelu_boundary/attention-tile methods), so this file
//     adds one standalone SiLU boundary function instead of editing the
//     frozen Client class.
//
// Parent pins (all frozen, none edited):
//   config-3 12blocks_head_cpudiagcache e2e driver SHA-256 (this file's
//   fork/composition parent; composition logic copied verbatim, per-block
//   #include swapped, one main()-local ClearEvalMultKeys/
//   ClearEvalAutomorphismKeys addition):
//     588fde54d38bf17b34b676e683e4434589a0d837f59f63a55effcb9a0ee54b2e
//   T123_V3 per-block source SHA-256 (included here, unchanged):
//     cbba5b6c2b13e0a9fbe9a6ca1db724782bbf0af0b4651fa17f1586fa6b03534b
//   T=2 two-block-refresh source SHA-256 (design adapted from, not
//   included):
//     5cb82f81dc46efe2f4bfd7a1c9e9c1bf90cea3d629432b027b42e5df89093767
//
// Not evidence until a contract-bound results/runs JSON lands. This run may
// take many hours: [stage] progress is logged after every block and every
// refresh/head boundary so a partial failure is diagnosable, not silent.

#define main real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3_main_unused
#include "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3.cpp"
#undef main

// Not pulled in by the frozen depth-13 source's own include list (it uses
// no std::map); needed here for per-stage level traces (BlockSummary/
// HeadEvaluation), matching the T=2 all-blocks-head driver's convention.
#include <map>

namespace {

constexpr std::size_t ALL_BLOCKS = 12;

// Pinned from real-GPU evidence for the single Token-SIMD B=8 T=103 block
// this driver composes 12 times (identical schedule at depth 13 and depth
// 16 -- only MULT_DEPTH differs between those forks, not the protocol):
//   results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_scheme_b_a100_20260731.json
//   protocol.round_trips == 857, protocol.logical_boundary_instances == 129162
constexpr std::size_t PER_BLOCK_ROUND_TRIPS = 857;
constexpr std::size_t PER_BLOCK_LOGICAL_INSTANCES = 129162;

constexpr std::size_t INTER_BLOCK_REFRESHES = ALL_BLOCKS - 1;  // 11
constexpr std::size_t HEAD_INPUT_REFRESHES = 1;
// Of the head's 3 nonlinearity crossings, 2 (final_ln, head_ln) flow
// through the shared Client's own round_trips()/logical_instances()
// counters (its public invsqrt_boundary()); SiLU has no Client method and
// is tracked by this driver's own HeadEvaluation counters instead.
constexpr std::size_t HEAD_CLIENT_TRACKED_ROUND_TRIPS = 2;
constexpr std::size_t HEAD_SILU_ROUND_TRIPS = 1;
constexpr std::size_t HEAD_ROUND_TRIPS =
    HEAD_CLIENT_TRACKED_ROUND_TRIPS + HEAD_SILU_ROUND_TRIPS;

constexpr std::size_t EXPECTED_TOTAL_CROSSINGS =
    ALL_BLOCKS * PER_BLOCK_ROUND_TRIPS + INTER_BLOCK_REFRESHES +
    HEAD_INPUT_REFRESHES + HEAD_ROUND_TRIPS;
constexpr std::size_t EXPECTED_TOTAL_LOGICAL_INSTANCES =
    ALL_BLOCKS * PER_BLOCK_LOGICAL_INSTANCES +
    INTER_BLOCK_REFRESHES * T + HEAD_INPUT_REFRESHES * 1 +
    HEAD_ROUND_TRIPS * 1;

static_assert(EXPECTED_TOTAL_CROSSINGS == 10299);
static_assert(EXPECTED_TOTAL_LOGICAL_INSTANCES == 1551081);

constexpr std::size_t LAST_TOKEN = T - 1;                    // 102
constexpr std::size_t LAST_TOKEN_GROUP = LAST_TOKEN / TOKEN_BATCH;  // 12
constexpr std::size_t LAST_TOKEN_LANE = LAST_TOKEN % TOKEN_BATCH;   // 6
static_assert(LAST_TOKEN_GROUP == TOKEN_GROUPS - 1);
static_assert(LAST_TOKEN_LANE < 7);  // final group only has 7 active lanes

// kimon: optional compile-time override of the fixture-identity pins for a
// platform-tagged (e.g. TACC ls6) build; frozen value is the default. Pass a
// bare-hex -D (no quotes); stringify makes it a literal. See kimon/env/.
#ifndef KIMON_PIN_STRINGIFY
#define KIMON_PIN_STRINGIFY2(x) #x
#define KIMON_PIN_STRINGIFY(x) KIMON_PIN_STRINGIFY2(x)
#endif
// kimon: this pin hashes the whole per-block T123_V3.cpp parent, which the
// platform build edits additively the same way it edits cpudiagcache.cpp
// (see that file's own PINNED_PARENT_* override) -- make it overridable the
// same way so this driver's parent check accepts the in-tree, platform-
// edited T123_V3.cpp.
#ifdef KIMON_PIN_CPUDIAGCACHE_T123_V3_FULL
constexpr std::string_view PINNED_CPUDIAGCACHE_T123_V3_SOURCE_SHA256 = KIMON_PIN_STRINGIFY(KIMON_PIN_CPUDIAGCACHE_T123_V3_FULL);
#else
constexpr std::string_view PINNED_CPUDIAGCACHE_T123_V3_SOURCE_SHA256 =
    "cbba5b6c2b13e0a9fbe9a6ca1db724782bbf0af0b4651fa17f1586fa6b03534b";
#endif
// Composition/design parent (the config-3 all-optimizations e2e driver this
// file forks its block-loop/refresh/head structure from, verbatim, adapted
// only to swap the per-block #include and add the free-host-keys call).
// Frozen, no platform override needed: this pin identifies a file that is
// NOT included and never compiled here, only design-attested.
constexpr std::string_view PINNED_CPUDIAGCACHE_DRIVER_PARENT_SOURCE_SHA256 =
    "588fde54d38bf17b34b676e683e4434589a0d837f59f63a55effcb9a0ee54b2e";
constexpr std::string_view PINNED_TWO_BLOCK_REFRESH_SOURCE_SHA256 =
    "5cb82f81dc46efe2f4bfd7a1c9e9c1bf90cea3d629432b027b42e5df89093767";
#ifdef KIMON_PIN_AB_MANIFEST
constexpr std::string_view PINNED_ALL_BLOCKS_FIXTURE_MANIFEST_SHA256 = KIMON_PIN_STRINGIFY(KIMON_PIN_AB_MANIFEST);
#else
constexpr std::string_view PINNED_ALL_BLOCKS_FIXTURE_MANIFEST_SHA256 =
    "ab55533eaf8779b64a67c5cdd8ac67152419389e33f1c44addd4a092e0ff962d";
#endif
#ifdef KIMON_PIN_AB_CONTRACT
constexpr std::string_view PINNED_ALL_BLOCKS_FIXTURE_CONTRACT_SHA256 = KIMON_PIN_STRINGIFY(KIMON_PIN_AB_CONTRACT);
#else
constexpr std::string_view PINNED_ALL_BLOCKS_FIXTURE_CONTRACT_SHA256 =
    "5256d3f59e215631f9ded7dba68f13e16c12c8fde927c53277bc1752171f7cca";
#endif

struct AllBlocksOptions {
    int gpu = 0;
    std::filesystem::path fixture_dir;
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string cpudiagcache_t123_v3_source_sha256;
    std::string cpudiagcache_driver_parent_source_sha256;
    std::string two_block_refresh_source_sha256;
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256;
    std::string source_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
};

[[noreturn]] void all_blocks_usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: "
        "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3 "
        "--gpu N --fixture-dir PATH --output PATH "
        "--cpudiagcache-t123-v3-source-sha256 SHA "
        "--cpudiagcache-driver-parent-source-sha256 SHA "
        "--two-block-refresh-source-sha256 SHA "
        "--fixture-manifest-sha256 SHA --fixture-contract-sha256 SHA "
        "--source-sha256 SHA [--backend-commit SHA] "
        "[--container-image NAME] [--environment TEXT]");
}

AllBlocksOptions parse_all_blocks_options(int argc, char** argv) {
    AllBlocksOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                all_blocks_usage_error("missing value for " + arg);
            }
            return argv[i];
        };
        if (arg == "--gpu") {
            options.gpu = std::stoi(next());
        } else if (arg == "--fixture-dir") {
            options.fixture_dir = next();
        } else if (arg == "--output") {
            options.output = next();
        } else if (arg == "--backend-commit") {
            options.backend_commit = next();
        } else if (arg == "--cpudiagcache-t123-v3-source-sha256") {
            options.cpudiagcache_t123_v3_source_sha256 = next();
        } else if (arg == "--cpudiagcache-driver-parent-source-sha256") {
            options.cpudiagcache_driver_parent_source_sha256 = next();
        } else if (arg == "--two-block-refresh-source-sha256") {
            options.two_block_refresh_source_sha256 = next();
        } else if (arg == "--fixture-manifest-sha256") {
            options.fixture_manifest_sha256 = next();
        } else if (arg == "--fixture-contract-sha256") {
            options.fixture_contract_sha256 = next();
        } else if (arg == "--source-sha256") {
            options.source_sha256 = next();
        } else if (arg == "--container-image") {
            options.container_image = next();
        } else if (arg == "--environment") {
            options.environment = next();
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: "
                   "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_"
                   "cpudiagcache_t123_v3 "
                   "--gpu N --fixture-dir PATH --output PATH "
                   "--cpudiagcache-t123-v3-source-sha256 SHA "
                   "--cpudiagcache-driver-parent-source-sha256 SHA "
                   "--two-block-refresh-source-sha256 SHA "
                   "--fixture-manifest-sha256 SHA "
                   "--fixture-contract-sha256 SHA --source-sha256 SHA\n"
                   "Complete 12-block DNAGPT backbone plus GSR classifier "
                   "head, Scheme B Token-SIMD B=8, T=103 (full prompt), "
                   "T123_V3 CPU-side diagonal-vector-cache + encode-once "
                   "template cache + free-host-keys per-block lever.\n";
            std::exit(0);
        } else {
            all_blocks_usage_error("unknown argument " + arg);
        }
    }
    if (options.gpu < 0 || options.fixture_dir.empty() ||
        options.output.empty()) {
        all_blocks_usage_error("gpu, fixture-dir, and output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        all_blocks_usage_error("refusing unpinned FIDESlib commit " +
                                options.backend_commit);
    }
    if (options.cpudiagcache_t123_v3_source_sha256 !=
        PINNED_CPUDIAGCACHE_T123_V3_SOURCE_SHA256) {
        all_blocks_usage_error(
            "refusing changed T123_V3 per-block parent source " +
            options.cpudiagcache_t123_v3_source_sha256);
    }
    if (options.cpudiagcache_driver_parent_source_sha256 !=
        PINNED_CPUDIAGCACHE_DRIVER_PARENT_SOURCE_SHA256) {
        all_blocks_usage_error(
            "refusing changed cpudiagcache driver composition-parent source " +
            options.cpudiagcache_driver_parent_source_sha256);
    }
    if (options.two_block_refresh_source_sha256 !=
        PINNED_TWO_BLOCK_REFRESH_SOURCE_SHA256) {
        all_blocks_usage_error(
            "refusing changed two-block-refresh design-parent source " +
            options.two_block_refresh_source_sha256);
    }
    if (options.fixture_manifest_sha256 !=
        PINNED_ALL_BLOCKS_FIXTURE_MANIFEST_SHA256) {
        all_blocks_usage_error("refusing unpinned 12-block fixture manifest " +
                                options.fixture_manifest_sha256);
    }
    if (options.fixture_contract_sha256 !=
        PINNED_ALL_BLOCKS_FIXTURE_CONTRACT_SHA256) {
        all_blocks_usage_error("refusing unpinned 12-block fixture contract " +
                                options.fixture_contract_sha256);
    }
    if (options.source_sha256 == "UNSPECIFIED") {
        all_blocks_usage_error("--source-sha256 is required");
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
                                  options.output.string());
    }
    return options;
}

// ---------------------------------------------------------------------
// Per-block fixture loading (12 released blocks, one weight set each)
// ---------------------------------------------------------------------

Fixture load_block_fixture(const std::filesystem::path& directory,
                            std::size_t block_index) {
    if (block_index >= ALL_BLOCKS) {
        throw std::invalid_argument("released block index must be in [0, 11]");
    }
    Fixture fixture;
    const std::string prefix = "weights__block" + std::to_string(block_index) + "__";
    fixture.ln1_weight = read_f64(directory / (prefix + "ln1.bin"), D);
    fixture.ln2_weight = read_f64(directory / (prefix + "ln2.bin"), D);

    const std::vector<double> qkv =
        read_f64(directory / (prefix + "attn_qkv.bin"), 3 * D * D);
    for (std::size_t part = 0; part < 3; ++part) {
        fixture.qkv[part].assign(
            qkv.begin() + static_cast<std::ptrdiff_t>(part * D * D),
            qkv.begin() + static_cast<std::ptrdiff_t>((part + 1) * D * D));
    }
    fixture.attention_projection =
        read_f64(directory / (prefix + "attn_proj.bin"), D * D);

    const std::vector<double> fc =
        read_f64(directory / (prefix + "mlp_fc.bin"), MLP_DIM * D);
    for (std::size_t part = 0; part < COPIES; ++part) {
        fixture.mlp_fc[part].assign(
            fc.begin() + static_cast<std::ptrdiff_t>(part * D * D),
            fc.begin() + static_cast<std::ptrdiff_t>((part + 1) * D * D));
    }
    const std::vector<double> projection =
        read_f64(directory / (prefix + "mlp_proj.bin"), D * MLP_DIM);
    for (std::size_t part = 0; part < COPIES; ++part) {
        fixture.mlp_projection[part].resize(D * D);
        for (std::size_t row = 0; row < D; ++row) {
            std::copy_n(
                projection.begin() +
                    static_cast<std::ptrdiff_t>(row * MLP_DIM + part * D),
                D,
                fixture.mlp_projection[part].begin() +
                    static_cast<std::ptrdiff_t>(row * D));
        }
    }
    fixture.oracle_block_output = read_f64(
        directory /
            ("oracle__block" + std::to_string(block_index) + "__block_output.bin"),
        T * D);
    return fixture;
}

// ---------------------------------------------------------------------
// Token-SIMD group encode/decode (shared by initial encryption and every
// inter-block refresh)
// ---------------------------------------------------------------------

std::vector<std::size_t> all_active_counts() {
    std::vector<std::size_t> counts(TOKEN_GROUPS);
    for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
        counts[group] = active_tokens(group);
    }
    return counts;
}

std::vector<Ct> encrypt_token_simd_groups(Cc cc, Keys& keys,
                                           const std::vector<double>& flat) {
    if (flat.size() != T * D) {
        throw std::invalid_argument(
            "Token-SIMD group encryption requires exactly T*D values");
    }
    std::vector<Ct> encrypted;
    encrypted.reserve(TOKEN_GROUPS);
    for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
        const std::size_t first = group * TOKEN_BATCH;
        const std::size_t active = active_tokens(group);
        std::vector<double> packed(SLOTS, 0.0);
        for (std::size_t token_lane = 0; token_lane < active; ++token_lane) {
            const std::size_t token = first + token_lane;
            for (std::size_t copy = 0; copy < COPIES; ++copy) {
                for (std::size_t dim = 0; dim < D; ++dim) {
                    packed[physical_slot(copy, dim, token_lane)] =
                        flat[token * D + dim];
                }
            }
        }
        Plaintext plaintext =
            cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
        Ct fresh = cc->Encrypt(keys.publicKey, plaintext);
        if (fresh->GetLevel() != 0) {
            throw std::runtime_error(
                "fresh Token-SIMD group encryption did not return level 0");
        }
        encrypted.push_back(std::move(fresh));
    }
    return encrypted;
}

std::vector<double> decrypt_token_simd_groups(Cc cc, Keys& keys,
                                               const std::vector<Ct>& groups) {
    if (groups.size() != TOKEN_GROUPS) {
        throw std::invalid_argument(
            "Token-SIMD group decryption requires exactly TOKEN_GROUPS "
            "ciphertexts");
    }
    std::vector<double> flat(T * D, 0.0);
    for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
        Ct local = groups[group];
        Plaintext decoded;
        cc->Decrypt(keys.secretKey, local, &decoded);
        decoded->SetLength(SLOTS);
        const std::vector<double> raw = decoded->GetRealPackedValue();
        const std::size_t first = group * TOKEN_BATCH;
        const std::size_t active = active_tokens(group);
        for (std::size_t token_lane = 0; token_lane < active; ++token_lane) {
            for (std::size_t dim = 0; dim < D; ++dim) {
                flat[(first + token_lane) * D + dim] =
                    raw[physical_slot(0, dim, token_lane)];
            }
        }
    }
    return flat;
}

// ---------------------------------------------------------------------
// Declared client refresh boundaries
// ---------------------------------------------------------------------

struct FullHiddenRefresh {
    std::vector<Ct> refreshed;
    Metrics metrics;
    double seconds = 0.0;
    std::size_t input_level = 0;
};

FullHiddenRefresh refresh_all_hidden_token_simd(
    Cc cc, Keys& keys, const std::vector<Ct>& packed_output,
    const std::vector<double>& oracle_block_output) {
    const auto start = Clock::now();
    FullHiddenRefresh result;
    for (const Ct& ct : packed_output) {
        result.input_level = std::max(result.input_level, ct->GetLevel());
    }
    const std::vector<double> flat =
        decrypt_token_simd_groups(cc, keys, packed_output);
    result.metrics = measure_output(flat, oracle_block_output);
    if (!result.metrics.all_finite) {
        throw std::runtime_error(
            "non-finite hidden state at inter-block client refresh");
    }
    result.refreshed = encrypt_token_simd_groups(cc, keys, flat);
    cc->Synchronize();
    result.seconds = elapsed_seconds(start);
    return result;
}

struct LastTokenRefresh {
    Ct refreshed;
    Metrics metrics;
    double seconds = 0.0;
    std::size_t input_level = 0;
};

LastTokenRefresh refresh_last_token_for_head(
    Cc cc, Keys& keys, const std::vector<Ct>& packed_output,
    const std::vector<double>& oracle_block_output) {
    const auto start = Clock::now();
    LastTokenRefresh result;
    for (const Ct& ct : packed_output) {
        result.input_level = std::max(result.input_level, ct->GetLevel());
    }
    const std::vector<double> flat =
        decrypt_token_simd_groups(cc, keys, packed_output);
    result.metrics = measure_output(flat, oracle_block_output);
    if (!result.metrics.all_finite) {
        throw std::runtime_error(
            "non-finite block-11 state at head-input client refresh");
    }

    std::vector<double> packed(SLOTS, 0.0);
    for (std::size_t copy = 0; copy < COPIES; ++copy) {
        for (std::size_t dim = 0; dim < D; ++dim) {
            packed[physical_slot(copy, dim, 0)] = flat[LAST_TOKEN * D + dim];
        }
    }
    Plaintext plaintext = cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
    result.refreshed = cc->Encrypt(keys.publicKey, plaintext);
    cc->Synchronize();
    if (result.refreshed->GetLevel() != 0) {
        throw std::runtime_error("head-input refresh did not return level 0");
    }
    result.seconds = elapsed_seconds(start);
    return result;
}

// ---------------------------------------------------------------------
// Classifier head (new: no equivalent exists in the frozen per-block source)
// ---------------------------------------------------------------------

struct HeadFixture {
    std::vector<double> final_ln;
    std::vector<double> head_linear;  // flat D*D, PyTorch [out,in] order
    std::vector<double> head_ln;
    std::vector<double> margin_n_minus_a;
    std::vector<double> oracle_head_ln;  // T*D, for the direct-dot oracle only
};

HeadFixture load_head_fixture(const std::filesystem::path& directory) {
    HeadFixture fixture;
    fixture.final_ln = read_f64(directory / "weights__final_ln.bin", D);
    fixture.head_linear = read_f64(directory / "weights__head_linear.bin", D * D);
    fixture.head_ln = read_f64(directory / "weights__head_ln.bin", D);
    fixture.margin_n_minus_a = read_f64(
        directory / "weights__head_readout_margin_n_minus_a.bin", D);
    fixture.oracle_head_ln =
        read_f64(directory / "oracle__head__head_ln_output.bin", T * D);
    return fixture;
}

double stable_exact_silu(double value) {
    if (value >= 0.0) {
        return value / (1.0 + std::exp(-value));
    }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

double direct_dot_oracle_margin(const HeadFixture& fixture) {
    double margin = 0.0;
    for (std::size_t dim = 0; dim < D; ++dim) {
        margin += fixture.oracle_head_ln[LAST_TOKEN * D + dim] *
                  fixture.margin_n_minus_a[dim];
    }
    return margin;
}

struct HeadEvaluation {
    Ct encrypted_margin;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels;
    std::size_t matrix_products = 0;
    std::size_t ciphertext_ciphertext_multiplications = 0;
    std::size_t ciphertext_plaintext_multiplications = 0;
    std::size_t silu_round_trips = 0;
    std::size_t silu_logical_instances = 0;
    double silu_seconds = 0.0;
};

class EncryptedHeadEvaluator {
  public:
    EncryptedHeadEvaluator(Cc context, Keys& keys, const HeadFixture& fixture,
                           Client& client)
        : cc_(std::move(context)),
          keys_(keys),
          fixture_(fixture),
          client_(client) {}

    HeadEvaluation evaluate(const Ct& single_token_group) {
        levels_.clear();
        matmul_count_ = 0;
        ct_ct_multiply_count_ = 0;
        ct_plain_multiply_count_ = 0;

        Ct final_ln =
            layernorm(single_token_group, fixture_.final_ln, "head_final_ln");
        record("final_ln", final_ln);

        Ct head_linear = matmul(baby_rotations(final_ln), fixture_.head_linear);
        record("head_linear", head_linear);

        const auto silu_start = Clock::now();
        Ct activated = silu_boundary(head_linear);
        const double silu_seconds = elapsed_seconds(silu_start);
        record("head_silu", activated);

        Ct head_ln = layernorm(activated, fixture_.head_ln, "head_ln");
        record("head_ln", head_ln);

        Ct weighted =
            multiply_plain(head_ln, repeated_plain(fixture_.margin_n_minus_a));
        Ct margin = sum_broadcast(weighted);
        record("margin_n_minus_a", margin);

        HeadEvaluation result;
        result.encrypted_margin = margin;
        result.levels = levels_;
        result.matrix_products = matmul_count_;
        result.ciphertext_ciphertext_multiplications = ct_ct_multiply_count_;
        result.ciphertext_plaintext_multiplications = ct_plain_multiply_count_;
        result.silu_round_trips = 1;
        result.silu_logical_instances = 1;
        result.silu_seconds = silu_seconds;
        return result;
    }

  private:
    struct BabyRotations {
        std::array<Ct, BSGS_N1> values{};
    };

    Plaintext raw_plain(const std::vector<double>& values) {
        if (values.size() != SLOTS) {
            throw std::invalid_argument("head plaintext must have exactly SLOTS values");
        }
        return cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
    }

    Plaintext repeated_plain(const std::vector<double>& values) {
        if (values.size() != D) {
            throw std::invalid_argument("head repeated plaintext must have D values");
        }
        // Broadcasts across every copy AND every token lane, matching the
        // frozen per-block repeated_plain layout exactly. Only lane 0 holds
        // meaningful ciphertext content in the head's single-token group;
        // the other 7 lanes multiply against zero-content slots and are
        // never read back.
        std::vector<double> packed(SLOTS, 0.0);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t dim = 0; dim < D; ++dim) {
                for (std::size_t lane = 0; lane < TOKEN_BATCH; ++lane) {
                    packed[physical_slot(copy, dim, lane)] = values[dim];
                }
            }
        }
        return raw_plain(packed);
    }

    Ct multiply(const Ct& left, const Ct& right) {
        ++ct_ct_multiply_count_;
        return cc_->EvalMult(left, right);
    }

    Ct multiply_plain(const Ct& input, Plaintext plaintext) {
        ++ct_plain_multiply_count_;
        return cc_->EvalMult(input, plaintext);
    }

    Ct sum_broadcast(const Ct& input) {
        return cc_->AccumulateSum(input, static_cast<int>(PACK_WIDTH),
                                   static_cast<int>(TOKEN_BATCH));
    }

    Ct mean_broadcast(const Ct& input) {
        Ct sum = sum_broadcast(input);
        return multiply_plain(
            sum, repeated_plain(std::vector<double>(D, 1.0 / static_cast<double>(D))));
    }

    Ct layernorm(const Ct& input, const std::vector<double>& weight,
                 const std::string& stage) {
        Ct mean = mean_broadcast(input);
        Ct centered = cc_->EvalSub(input, mean);
        Ct variance = mean_broadcast(multiply(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        // Reuses the shared Client's own public boundary method: the head
        // is just one more "logical_instances=1" caller, identical in kind
        // to every per-block LN1/LN2 call.
        Ct inverse = client_.invsqrt_boundary(variance, 1, stage);
        Ct normalized = multiply(centered, inverse);
        return multiply_plain(normalized, repeated_plain(weight));
    }

    BabyRotations baby_rotations(const Ct& input) {
        std::vector<std::int32_t> indices;
        indices.reserve(BSGS_N1 - 1);
        for (std::size_t small = 1; small < BSGS_N1; ++small) {
            indices.push_back(static_cast<std::int32_t>(TOKEN_BATCH * small));
        }
        const auto rotated = cc_->EvalFastRotation(input, indices,
                                                    cc_->GetCyclotomicOrder(), nullptr);
        if (rotated.size() != indices.size()) {
            throw std::runtime_error(
                "head evaluator received incomplete Token-SIMD baby rotations");
        }
        BabyRotations output;
        output.values[0] = input;
        for (std::size_t small = 1; small < BSGS_N1; ++small) {
            output.values[small] = rotated.at(small - 1);
        }
        return output;
    }

    Ct matmul(const BabyRotations& baby, const std::vector<double>& weight) {
        if (weight.size() != D * D) {
            throw std::invalid_argument("head linear matrix must be D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::size_t diagonal = BSGS_N1 * giant + small;
                std::vector<double> packed_values(SLOTS, 0.0);
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH;
                        const std::size_t column = (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            const double coefficient = weight[rolled_row * D + column];
                            for (std::size_t token = 0; token < TOKEN_BATCH; ++token) {
                                packed_values[physical_slot(copy, row, token)] =
                                    coefficient;
                            }
                        }
                    }
                }
                Ct term = multiply_plain(baby.values[small], raw_plain(packed_values));
                inner = small == 0 ? term : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(
                    inner, static_cast<std::int32_t>(TOKEN_BATCH * BSGS_N1 * giant));
            }
            result = giant == 0 ? inner : cc_->EvalAdd(result, inner);
        }
        ++matmul_count_;
        return result;
    }

    // The one head boundary the shared Client class has no method for
    // (depth13's Client only exposes invsqrt_boundary/gelu_boundary/
    // attention-tile methods). Uses keys_ directly, exactly like the
    // frozen gelu_boundary()'s decrypt/transform/re-encrypt pattern, but
    // for SiLU over D features on a single active token lane.
    Ct silu_boundary(const Ct& ciphertext) {
        Plaintext plaintext;
        Ct local = ciphertext;
        cc_->Decrypt(keys_.secretKey, local, &plaintext);
        plaintext->SetLength(SLOTS);
        const std::vector<double> raw = plaintext->GetRealPackedValue();
        std::vector<double> transformed(SLOTS, 0.0);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t dim = 0; dim < D; ++dim) {
                const std::size_t slot = physical_slot(copy, dim, 0);
                transformed[slot] = stable_exact_silu(raw[slot]);
            }
        }
        Plaintext refreshed = cc_->MakeCKKSPackedPlaintext(transformed, 1, 0, nullptr, SLOTS);
        return cc_->Encrypt(keys_.publicKey, refreshed);
    }

    void record(const std::string& name, const Ct& ciphertext) {
        const std::size_t level = ciphertext->GetLevel();
        levels_[name] = {level, level};
        std::cout << "[stage] head " << name << " level=" << level << '\n';
    }

    Cc cc_;
    Keys& keys_;
    const HeadFixture& fixture_;
    Client& client_;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels_;
    std::size_t matmul_count_ = 0;
    std::size_t ct_ct_multiply_count_ = 0;
    std::size_t ct_plain_multiply_count_ = 0;
};

// ---------------------------------------------------------------------
// Evidence assembly
// ---------------------------------------------------------------------

struct BlockSummary {
    std::size_t block = 0;
    std::size_t matrix_products = 0;
    std::size_t ciphertext_ciphertext_multiplications = 0;
    std::size_t ciphertext_plaintext_multiplications = 0;
    std::size_t packed_output_level = 0;
    double evaluation_seconds = 0.0;
    double refresh_seconds = 0.0;
    Metrics refresh_metrics;
};

void append_metrics(std::ostringstream& out, const Metrics& metrics,
                     std::string_view indent) {
    out << indent << "\"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << indent << "\"worst_token_rel_inf\": " << metrics.worst_token_rel_inf
        << ",\n"
        << indent << "\"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << indent << "\"all_finite\": " << (metrics.all_finite ? "true" : "false")
        << ",\n"
        << indent << "\"passed\": " << (metrics.passed ? "true" : "false");
}

std::string make_all_blocks_json(
    const AllBlocksOptions& options, std::uint32_t ring,
    std::size_t rotation_key_count, const std::vector<BlockSummary>& blocks,
    const HeadEvaluation& head, double decrypted_margin, double oracle_margin,
    double fixture_seconds, double setup_seconds, double encryption_seconds,
    double blocks_seconds, double head_seconds, double refresh_seconds,
    double final_decrypt_seconds, std::size_t client_round_trips,
    std::size_t client_logical_instances, double client_nonlinearity_seconds,
    bool passed) {
    std::size_t block_matrix_products = 0;
    std::size_t block_ct_ct = 0;
    std::size_t block_ct_pt = 0;
    for (const BlockSummary& block : blocks) {
        block_matrix_products += block.matrix_products;
        block_ct_ct += block.ciphertext_ciphertext_multiplications;
        block_ct_pt += block.ciphertext_plaintext_multiplications;
    }
    const double margin_abs_error = std::abs(decrypted_margin - oracle_margin);
    const double margin_rel_error =
        margin_abs_error / (std::abs(oracle_margin) + 1e-15);
    const bool label_matches =
        std::signbit(decrypted_margin) == std::signbit(oracle_margin);
    const std::size_t declared_crossings =
        client_round_trips + INTER_BLOCK_REFRESHES + HEAD_INPUT_REFRESHES +
        head.silu_round_trips;
    const std::size_t declared_logical_instances =
        client_logical_instances + INTER_BLOCK_REFRESHES * T +
        HEAD_INPUT_REFRESHES * 1 + head.silu_logical_instances;
    const double client_total =
        client_nonlinearity_seconds + refresh_seconds + head.silu_seconds;
    const double encrypted_total = blocks_seconds + head_seconds + refresh_seconds;

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"Released-weight DNAGPT complete 12-block backbone "
           "plus encrypted GSR N-minus-A head, Scheme B Token-SIMD B=8, "
           "T=103 (full prompt), T123_V3 per-block lever (CPU-side "
           "diagonal-vector cache + Tier1 encode-once encoded-Plaintext "
           "templates with per-use clone + Tier2/3 OMP-parallel batched "
           "encode + post-LoadContext free of CPU-side OpenFHE key maps)\",\n"
        << "  \"implementation_version\": "
           "\"t103-scheme-b-token-simd-all-12-blocks-head-cpudiagcache-t123-v3-"
           "v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS)\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"gate\": \"all_blocks_head_t103_cpudiagcache_t123_v3\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \"" << json_escape(options.backend_commit) << "\",\n"
        << "  \"container_image\": \"" << json_escape(options.container_image)
        << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \"" << json_escape(options.source_sha256) << "\",\n"
        << "  \"cpudiagcache_t123_v3_parent_source_sha256\": \""
        << json_escape(options.cpudiagcache_t123_v3_source_sha256) << "\",\n"
        << "  \"cpudiagcache_driver_parent_source_sha256\": \""
        << json_escape(options.cpudiagcache_driver_parent_source_sha256) << "\",\n"
        << "  \"two_block_refresh_design_parent_source_sha256\": \""
        << json_escape(options.two_block_refresh_source_sha256) << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"fixture_contract_sha256\": \""
        << json_escape(options.fixture_contract_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"model\": \"dna_gpt0.1b_m classification checkpoint, all 12 "
           "released blocks and GSR head\",\n"
        << "    \"blocks\": " << ALL_BLOCKS << ",\n"
        << "    \"D\": " << D << ",\n"
        << "    \"T\": " << T << ",\n"
        << "    \"token_batch\": " << TOKEN_BATCH << ",\n"
        << "    \"token_groups\": " << TOKEN_GROUPS << ",\n"
        << "    \"batch_slots\": " << SLOTS << ",\n"
        << "    \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"rotation_keys\": " << rotation_key_count << ",\n"
        << "    \"bootstraps\": 0,\n"
        << "    \"block_matrix_products\": " << block_matrix_products << ",\n"
        << "    \"head_matrix_products\": " << head.matrix_products << ",\n"
        << "    \"operation_counts\": {\n"
        << "      \"explicit_ciphertext_ciphertext_multiplications\": "
        << (block_ct_ct + head.ciphertext_ciphertext_multiplications) << ",\n"
        << "      \"explicit_ciphertext_plaintext_multiplications\": "
        << (block_ct_pt + head.ciphertext_plaintext_multiplications) << "\n"
        << "    }\n"
        << "  },\n"
        << "  \"composition\": {\n"
        << "    \"same_crypto_context_and_key_lineage\": true,\n"
        << "    \"inter_block_full_hidden_refreshes\": " << INTER_BLOCK_REFRESHES
        << ",\n"
        << "    \"block11_to_head_last_token_refreshes\": " << HEAD_INPUT_REFRESHES
        << ",\n"
        << "    \"last_token_index\": " << LAST_TOKEN << ",\n"
        << "    \"last_token_group\": " << LAST_TOKEN_GROUP << ",\n"
        << "    \"last_token_lane\": " << LAST_TOKEN_LANE << ",\n"
        << "    \"fresh_input_level\": 0\n"
        << "  },\n"
        << "  \"blocks\": [\n";
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const BlockSummary& block = blocks[index];
        out << "    {\n"
            << "      \"block\": " << block.block << ",\n"
            << "      \"packed_output_level\": " << block.packed_output_level
            << ",\n"
            << "      \"matrix_products\": " << block.matrix_products << ",\n"
            << "      \"refresh_metrics\": {\n";
        append_metrics(out, block.refresh_metrics, "        ");
        out << "\n      },\n"
            << "      \"evaluation_seconds\": " << block.evaluation_seconds << ",\n"
            << "      \"refresh_seconds\": " << block.refresh_seconds << "\n"
            << "    }" << (index + 1 == blocks.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"head\": {\n"
        << "    \"encrypted_readout\": \"last token -> final LN -> head "
           "linear -> SiLU -> head LN -> dot(N-A)\",\n"
        << "    \"decrypted_margin_n_minus_a\": " << decrypted_margin << ",\n"
        << "    \"direct_dot_oracle_margin_n_minus_a\": " << oracle_margin << ",\n"
        << "    \"margin_abs_error\": " << margin_abs_error << ",\n"
        << "    \"margin_rel_error\": " << margin_rel_error << ",\n"
        << "    \"label\": \"" << (decrypted_margin >= 0.0 ? "N" : "A") << "\",\n"
        << "    \"label_matches\": " << (label_matches ? "true" : "false") << ",\n"
        << "    \"silu_round_trips\": " << head.silu_round_trips << ",\n"
        << "    \"level_trace\": {\n";
    std::size_t head_level_index = 0;
    for (const auto& [name, range] : head.levels) {
        out << "      \"" << json_escape(name) << "\": {\"min\": " << range.first
            << ", \"max\": " << range.second << "}";
        out << (++head_level_index == head.levels.size() ? "\n" : ",\n");
    }
    out << "    }\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"round_trips\": " << declared_crossings << ",\n"
        << "    \"nonlinearity_round_trips\": " << client_round_trips << ",\n"
        << "    \"full_hidden_state_refresh_round_trips\": " << INTER_BLOCK_REFRESHES
        << ",\n"
        << "    \"last_token_refresh_round_trips\": " << HEAD_INPUT_REFRESHES << ",\n"
        << "    \"head_silu_round_trips\": " << head.silu_round_trips << ",\n"
        << "    \"logical_boundary_instances\": " << declared_logical_instances
        << ",\n"
        << "    \"boundary_seconds_total\": " << client_total << "\n"
        << "  },\n"
        << "  \"timings_seconds\": {\n"
        << "    \"load_verified_fixture\": " << fixture_seconds << ",\n"
        << "    \"context_keygen_load\": " << setup_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds << ",\n"
        << "    \"blocks_encrypted_evaluation\": " << blocks_seconds << ",\n"
        << "    \"head_encrypted_evaluation\": " << head_seconds << ",\n"
        << "    \"refreshes\": " << refresh_seconds << ",\n"
        << "    \"encrypted_evaluation\": " << encrypted_total << ",\n"
        << "    \"client_boundary_seconds_total\": " << client_total << ",\n"
        << "    \"server_linear_algebra_seconds\": "
        << (encrypted_total - client_total) << ",\n"
        << "    \"final_decrypt\": " << final_decrypt_seconds << "\n"
        << "  },\n"
        << "  \"all_finite\": " << (std::isfinite(decrypted_margin) ? "true" : "false")
        << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"input_boundary\": \"token plus position embeddings encrypted by "
           "client\",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"declared_client_boundary_crossings\": " << declared_crossings
        << ",\n"
        << "  \"declared_logical_boundary_instances\": " << declared_logical_instances
        << ",\n"
        << "  \"final_decrypt_calls\": 1,\n"
        << "  \"evaluator_has_private_key\": false\n"
        << "}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;
        const AllBlocksOptions options = parse_all_blocks_options(argc, argv);

        const auto fixture_start = Clock::now();
        const std::vector<double> initial_input =
            read_f64(options.fixture_dir / "input__embeddings.bin", T * D);
        const HeadFixture head_fixture = load_head_fixture(options.fixture_dir);
        double fixture_seconds = elapsed_seconds(fixture_start);

        const std::vector<int> rotation_keys = required_rotation_keys();
        const auto setup_start = Clock::now();
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
        parameters.SetSecretKeyDist(UNIFORM_TERNARY);
        parameters.SetMultiplicativeDepth(MULT_DEPTH);
        parameters.SetScalingModSize(SCALE_BITS);
        parameters.SetFirstModSize(FIRST_MOD_BITS);
        parameters.SetScalingTechnique(FLEXIBLEAUTO);
        parameters.SetKeySwitchTechnique(HYBRID);
        parameters.SetNumLargeDigits(LARGE_DIGITS);
        parameters.SetBatchSize(SLOTS);
        parameters.SetRingDim(RING_DIM);
        parameters.SetDevices({options.gpu});
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);

        Cc cc = GenCryptoContext(parameters);
        std::cout << "[stage] GenCryptoContext done\n";
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        auto keys = cc->KeyGen();
        std::cout << "[stage] KeyGen done\n";
        cc->EvalMultKeyGen(keys.secretKey);
        cc->EvalRotateKeyGen(keys.secretKey, rotation_keys);
        std::cout << "[stage] EvalRotateKeyGen done rotation_keys="
                  << rotation_keys.size() << '\n';
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        // T123_V3: mirrors the included per-block source's own (unused, this
        // driver has its own main() and its own context setup above) free-
        // host-keys addition. LoadContext already copied the relin key + all
        // rotation keys to the GPU; the CPU-side (OpenFHE) copies in the
        // static eval-mult-key/rotation-key maps are dead weight for the
        // rest of this process (one crypto context, no later CPU-side
        // consumer -- see the T123_V3 per-block source's own comment at its
        // call site). Called exactly ONCE here, before any block executes,
        // not once per block.
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalMultKeys();
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalAutomorphismKeys();
        const double setup_seconds = elapsed_seconds(setup_start);
        const std::uint32_t ring = cc->GetRingDimension();
        if (ring < 2 * SLOTS) {
            throw std::runtime_error(
                "ring dimension cannot provide requested Token-SIMD slots");
        }
        std::cout << "[stage] context ready ring=" << ring << " slots=" << SLOTS
                  << " token_batch=" << TOKEN_BATCH << '\n';

        const auto encryption_start = Clock::now();
        std::vector<Ct> encrypted_inputs =
            encrypt_token_simd_groups(cc, keys, initial_input);
        cc->Synchronize();
        const double encryption_seconds = elapsed_seconds(encryption_start);
        std::cout << "[stage] initial encryption done token_groups="
                  << encrypted_inputs.size() << '\n';

        const std::vector<std::size_t> active_counts = all_active_counts();

        Client client(cc, keys);
        std::vector<BlockSummary> summaries;
        summaries.reserve(ALL_BLOCKS);
        double blocks_seconds = 0.0;
        double refresh_seconds = 0.0;
        bool all_blocks_passed = true;
        Ct head_input;

        for (std::size_t block = 0; block < ALL_BLOCKS; ++block) {
            std::cout << "[stage] block=" << block << " begin\n";
            const auto block_fixture_start = Clock::now();
            const Fixture fixture = load_block_fixture(options.fixture_dir, block);
            fixture_seconds += elapsed_seconds(block_fixture_start);

            const std::size_t before_round_trips = client.round_trips();
            const std::size_t before_logical = client.logical_instances();

            const auto block_start = Clock::now();
            EncryptedEvaluator evaluator(cc, fixture, client);
            EvaluationResult evaluation =
                evaluator.evaluate(encrypted_inputs, active_counts);
            cc->Synchronize();
            const double block_seconds = elapsed_seconds(block_start);
            blocks_seconds += block_seconds;
            std::cout << "[stage] block=" << block
                      << " evaluation done evaluation_seconds=" << block_seconds
                      << " packed_output_level=" << evaluation.packed_output_level
                      << '\n';

            if (client.round_trips() - before_round_trips != PER_BLOCK_ROUND_TRIPS ||
                client.logical_instances() - before_logical !=
                    PER_BLOCK_LOGICAL_INSTANCES) {
                throw std::runtime_error("block " + std::to_string(block) +
                                          " crossed an unexpected Scheme B "
                                          "boundary count");
            }

            BlockSummary summary;
            summary.block = block;
            summary.matrix_products = evaluation.matrix_products;
            summary.ciphertext_ciphertext_multiplications =
                evaluation.ct_ct_multiplications;
            summary.ciphertext_plaintext_multiplications =
                evaluation.ct_plain_multiplications;
            summary.packed_output_level = evaluation.packed_output_level;
            summary.evaluation_seconds = block_seconds;

            if (block + 1 < ALL_BLOCKS) {
                std::cout << "[boundary] refresh block=" << block << "->"
                          << (block + 1) << " begin\n";
                FullHiddenRefresh refresh = refresh_all_hidden_token_simd(
                    cc, keys, evaluation.packed_output, fixture.oracle_block_output);
                summary.refresh_metrics = refresh.metrics;
                summary.refresh_seconds = refresh.seconds;
                refresh_seconds += refresh.seconds;
                all_blocks_passed = all_blocks_passed && refresh.metrics.passed;
                std::cout << "[boundary] refresh block=" << block << "->"
                          << (block + 1)
                          << " done global_rel_inf=" << refresh.metrics.global_rel_inf
                          << " passed=" << (refresh.metrics.passed ? "true" : "false")
                          << '\n';
                for (Ct& ct : evaluation.packed_output) {
                    ct.reset();
                }
                for (Ct& ct : encrypted_inputs) {
                    ct.reset();
                }
                encrypted_inputs = std::move(refresh.refreshed);
            } else {
                std::cout << "[boundary] head-input refresh begin\n";
                LastTokenRefresh refresh = refresh_last_token_for_head(
                    cc, keys, evaluation.packed_output, fixture.oracle_block_output);
                summary.refresh_metrics = refresh.metrics;
                summary.refresh_seconds = refresh.seconds;
                refresh_seconds += refresh.seconds;
                all_blocks_passed = all_blocks_passed && refresh.metrics.passed;
                std::cout << "[boundary] head-input refresh done global_rel_inf="
                          << refresh.metrics.global_rel_inf
                          << " passed=" << (refresh.metrics.passed ? "true" : "false")
                          << '\n';
                head_input = std::move(refresh.refreshed);
                for (Ct& ct : evaluation.packed_output) {
                    ct.reset();
                }
                for (Ct& ct : encrypted_inputs) {
                    ct.reset();
                }
            }
            summaries.push_back(std::move(summary));
            std::cout << "[stage] block=" << block << " end\n";
        }

        std::cout << "[stage] head begin\n";
        const std::size_t before_head_round_trips = client.round_trips();
        const std::size_t before_head_logical = client.logical_instances();
        const auto head_start = Clock::now();
        EncryptedHeadEvaluator head_evaluator(cc, keys, head_fixture, client);
        HeadEvaluation head = head_evaluator.evaluate(head_input);
        cc->Synchronize();
        const double head_seconds = elapsed_seconds(head_start);
        std::cout << "[stage] head end evaluation_seconds=" << head_seconds << '\n';

        if (client.round_trips() - before_head_round_trips !=
                HEAD_CLIENT_TRACKED_ROUND_TRIPS ||
            client.logical_instances() - before_head_logical !=
                HEAD_CLIENT_TRACKED_ROUND_TRIPS) {
            throw std::runtime_error(
                "classifier head crossed an unexpected Scheme B boundary count");
        }
        if (head.silu_round_trips != HEAD_SILU_ROUND_TRIPS ||
            head.silu_logical_instances != HEAD_SILU_ROUND_TRIPS) {
            throw std::runtime_error(
                "classifier head SiLU crossed an unexpected boundary count");
        }

        const auto decrypt_start = Clock::now();
        Plaintext decoded_margin;
        cc->Decrypt(keys.secretKey, head.encrypted_margin, &decoded_margin);
        decoded_margin->SetLength(SLOTS);
        const std::vector<double> margin_values = decoded_margin->GetRealPackedValue();
        if (margin_values.size() <= physical_slot(0, 0, 0)) {
            throw std::runtime_error("final margin decrypt returned too few slots");
        }
        const double decrypted_margin = margin_values.at(physical_slot(0, 0, 0));
        const double final_decrypt_seconds = elapsed_seconds(decrypt_start);
        std::cout << "[stage] final decrypt done decrypted_margin_n_minus_a="
                  << decrypted_margin << '\n';

        const double oracle_margin = direct_dot_oracle_margin(head_fixture);
        const double margin_rel_error =
            std::abs(decrypted_margin - oracle_margin) / (std::abs(oracle_margin) + 1e-15);
        const bool label_matches =
            std::signbit(decrypted_margin) == std::signbit(oracle_margin);

        const std::size_t declared_crossings = client.round_trips() +
                                                INTER_BLOCK_REFRESHES +
                                                HEAD_INPUT_REFRESHES +
                                                head.silu_round_trips;
        const std::size_t declared_logical_instances =
            client.logical_instances() + INTER_BLOCK_REFRESHES * T +
            HEAD_INPUT_REFRESHES * 1 + head.silu_logical_instances;
        if (declared_crossings != EXPECTED_TOTAL_CROSSINGS ||
            declared_logical_instances != EXPECTED_TOTAL_LOGICAL_INSTANCES) {
            throw std::runtime_error(
                "complete graph crossed an unexpected total Scheme B boundary "
                "count");
        }

        const bool passed = all_blocks_passed && std::isfinite(decrypted_margin) &&
                             margin_rel_error <= TOL && label_matches;

        const std::string evidence = make_all_blocks_json(
            options, ring, rotation_keys.size(), summaries, head, decrypted_margin,
            oracle_margin, fixture_seconds, setup_seconds, encryption_seconds,
            blocks_seconds, head_seconds, refresh_seconds, final_decrypt_seconds,
            client.round_trips(), client.logical_instances(), client.seconds(),
            passed);
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout << (passed ? "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_T123_V3_PASS"
                             : "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_T123_V3_FAIL")
                  << '\n';
        return passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
