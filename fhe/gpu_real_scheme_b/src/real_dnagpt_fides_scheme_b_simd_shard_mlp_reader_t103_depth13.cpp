// Scheme B (hybrid client-assisted CKKS) 2-GPU process-per-GPU sharding,
// STAGE 2 -- MLP down-projection CHUNK split with an exact ciphertext
// EvalAdd merge. This is the SHARDED MLP READER phase (one of two concurrent
// shard workers).
//
// Scope (docs/hybrid/tasks.md "2-GPU process-per-GPU sharding", Stage 2 --
// the merge-correctness stage that Stage 1 deferred): Stage 1
// (real_dnagpt_fides_scheme_b_simd_shard_{writer,reader}_t103_depth13.cpp)
// split the Q/K/V projection three ways with ZERO ciphertext merge -- Q, K, V
// are independent output ciphertexts, only ever consumed together downstream.
// Stage 2 is the harder case: the MLP down-projection is a genuine
// partial-SUM accumulation. In the frozen full-block source the down
// projection is
//     Ct mlp;
//     for (chunk = 0..COPIES-1)
//         mlp = chunk==0 ? matmul(baby(hidden[chunk]), mlp_projection[chunk])
//                        : EvalAdd(mlp, matmul(...));
//     block_output = EvalAdd(residual1, mlp);
// i.e. four D x D matrix products whose results are summed into the SAME
// output ciphertext. That sum is associative over CKKS EvalAdd (an exact,
// depth-free homomorphic addition), so it can be partitioned: this reader,
// given a disjoint 2-chunk subset via --chunks (worker 0 => "0,1", worker 1
// => "2,3"), computes ONLY its two chunks and EvalAdds them into a per-group
// PARTIAL-sum ciphertext, then serializes those 13 partials to --state-dir.
// A separate merge step
// (real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13.cpp)
// EvalAdds the two workers' partials back into the full down-projection sum
// and adds residual1 -- reproducing the un-sharded block output exactly.
// The 2-and-2 split + EvalAdd-merge is proved a math no-op, to ~1e-9, by
// test_shard_mlp_reader_merge_t103_depth13_contract.py BEFORE this source is
// allowed to run.
//
// Which upstream is (re)computed: like Stage 1's reader (which recomputes
// LN1 on every worker), each Stage-2 worker independently recomputes the
// ENTIRE upstream block-0 lineage (LN1 -> Q/K/V -> attention -> proj ->
// residual1 -> LN2 -> FC -> GELU -> replicate) to obtain hidden[0..3]; the
// only thing that differs between the two workers is which two down
// projection chunks each accumulates. hidden[chunk] is deterministic in the
// plaintext it encrypts (the GELU boundary is exact), so both workers'
// hidden[chunk] agree in plaintext and the cross-worker EvalAdd of their
// partials is exact. The worker that owns chunk 0 (the "primary", --chunks
// starting at 0) additionally serializes residual1 for the merge step.
//
// Fork discipline -- additive; the parents are not edited, only forked:
//   full-block schedule/evaluator parent (this file is a copy of it with a
//   localized down-projection split and a deserialize-then-serialize-partials
//   main; the CPU diagonal-vector cache is inherited unchanged)
//   real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp
//     SHA-256: 686c8b766436fde0fd4422f6a29f80361f8c8a2565768577857e9264b2976fd5
//   Stage-1 sharded-reader parent (deserialize-context/keys-then-LoadContext
//   shape, secret-key-hygiene discipline, --state-dir companion-timing gate)
//   real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp
//     SHA-256: 3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89
//   depth-13 crypto-parameter/schedule source (rotation-key set the Stage-1
//   writer serialized; unchanged)
//   real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp
//     SHA-256: 6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e
//   full-block semantic anchor (unchanged from the full-block parent):
//     70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f
//
// This reader contains ZERO GenCryptoContext/KeyGen/EvalMultKeyGen/
// EvalRotateKeyGen calls -- it deserializes the context/keys the Stage-1
// writer serialized (see the parse/main below), exactly like the Stage-1
// reader. Its EncryptedEvaluator never touches a secret key; only the Client
// class holds the deserialized secret key, for the exact LN1/LN2/GELU
// boundaries. Secret-key hygiene: identical to both parents -- the secret
// key is deserialized to simulate a restarted client process for measurement
// only; its path/contents are never printed. Not a production client/server
// key-custody design (out of scope per CLAUDE.md).
//
// Ciphertext serialization: bringing two GPU processes' partial-sum
// ciphertexts into one merge process requires serializing a Ciphertext to
// disk. FIDESlib's Serialize path is exercised for CryptoContext/keys by the
// Stage-1 writer; whether it round-trips a GPU-resident Ciphertext is an
// UNVERIFIED capability at the time this source was written (no existing
// Scheme B source serializes a ciphertext) -- see the report/[A] note. The
// EvalAdd merge math itself is exact and fixture-proven locally regardless of
// transport.
//
// Physical slot mapping:
//   ((copy * PACK_WIDTH + feature) * TOKEN_BATCH) + token_lane
//
// Therefore every frozen logical rotation k is scaled to TOKEN_BATCH*k and
// cannot mix token lanes.  Public diagonal coefficients are repeated over the
// TOKEN_BATCH innermost lanes.

#include <fideslib.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using namespace fideslib;

namespace {

constexpr std::size_t D = 768;
constexpr std::size_t T = 103;
constexpr std::size_t HEADS = 12;
constexpr std::size_t HEAD_DIM = 64;
constexpr std::size_t MLP_DIM = 3072;
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t HEADS_PER_COPY = HEADS / COPIES;
constexpr std::size_t LOGICAL_SLOTS = COPIES * PACK_WIDTH;
constexpr std::size_t TOKEN_BATCH = 8;
constexpr std::size_t SLOTS = LOGICAL_SLOTS * TOKEN_BATCH;
constexpr std::size_t TOKEN_GROUPS =
    (T + TOKEN_BATCH - 1) / TOKEN_BATCH;
constexpr std::size_t BSGS_N1 = 32;
constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;
constexpr std::uint32_t RING_DIM = 65536;
constexpr std::uint32_t MULT_DEPTH = 13;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 3;
constexpr double EPS = 1e-5;
constexpr double TOL = 4e-2;

static_assert(D == HEADS * HEAD_DIM);
static_assert(MLP_DIM == COPIES * D);
static_assert(HEADS % COPIES == 0);
static_assert(HEADS_PER_COPY * TOKEN_BATCH <= PACK_WIDTH - D);
static_assert(SLOTS == 32768);
static_assert(RING_DIM == 2 * SLOTS);
static_assert(TOKEN_GROUPS == 13);
static_assert(T % TOKEN_BATCH == 7);
static_assert(BSGS_N1 * BSGS_N2 == PACK_WIDTH);

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
// Full-block evaluator parent this Stage-2 reader is a copy-then-localized-
// edit fork of (the CPU-diagonal-cache complete-block gate).
constexpr std::string_view PINNED_FULLBLOCK_PARENT_SHA256 =
    "686c8b766436fde0fd4422f6a29f80361f8c8a2565768577857e9264b2976fd5";
// Stage-1 sharded-reader parent (deserialize/state-dir/secret-key-hygiene).
constexpr std::string_view PINNED_STAGE1_READER_SHA256 =
    "3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89";
constexpr std::string_view PINNED_PARENT_SOURCE_SHA256 =
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e";
constexpr std::string_view PINNED_FROZEN_SOURCE_SHA256 =
    "70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f";
constexpr std::string_view PINNED_SCHEDULE_SHA256 =
    "c6b221f365ba6326f615c5554458d7bd092990d23c4ba0d5106ca7577cb7c3aa";
constexpr std::string_view PINNED_FIXTURE_MANIFEST =
    "d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6";
constexpr std::string_view PINNED_FIXTURE_CONTRACT_SHA256 =
    "061d53bd25bbcaf75d4c12067ec77f032c3ba0e35300a0a6168c2ce15f3672db";

using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;
using Cc = CryptoContext<DCRTPoly>;
using Keys = KeyPair<DCRTPoly>;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double exact_invsqrt(double value) { return 1.0 / std::sqrt(value); }

double exact_gelu(double value) {
    constexpr double c = 0.7978845608028654;
    return 0.5 * value *
           (1.0 + std::tanh(c * (value + 0.044715 * value * value * value)));
}

std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    if (::gmtime_r(&now, &utc) == nullptr) {
        throw std::runtime_error("could not obtain UTC timestamp");
    }
    std::array<char, 32> buffer{};
    if (std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ",
                      &utc) == 0) {
        throw std::runtime_error("could not format UTC timestamp");
    }
    return buffer.data();
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(ch)
                        << std::dec;
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

struct Options {
    int gpu = 0;
    std::filesystem::path state_dir;
    std::filesystem::path fixture_dir;
    std::filesystem::path output;
    // The disjoint 2-chunk subset this worker owns, e.g. {0,1} or {2,3}.
    std::vector<std::size_t> chunks;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string parent_source_sha256;
    std::string frozen_source_sha256;
    std::string schedule_sha256;
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256;
    std::string source_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";

    bool primary() const {
        return !chunks.empty() && chunks.front() == 0;
    }
    std::string chunk_tag() const {
        std::string tag;
        for (const std::size_t chunk : chunks) {
            tag += std::to_string(chunk);
        }
        return tag;
    }
};

std::vector<std::size_t> parse_chunks(const std::string& value) {
    std::vector<std::size_t> chunks;
    std::string current;
    auto flush = [&]() {
        if (current.empty()) {
            return;
        }
        const int parsed = std::stoi(current);
        if (parsed < 0 || parsed >= static_cast<int>(COPIES)) {
            throw std::invalid_argument(
                "MLP chunk index out of range (must be 0.." +
                std::to_string(COPIES - 1) + ")");
        }
        chunks.push_back(static_cast<std::size_t>(parsed));
        current.clear();
    };
    for (const char ch : value) {
        if (ch == ',') {
            flush();
        } else {
            current.push_back(ch);
        }
    }
    flush();
    return chunks;
}

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13 "
        "--gpu N --state-dir PATH --chunks 0,1|2,3 "
        "--fixture-dir PATH --output PATH "
        "--parent-source-sha256 SHA --frozen-source-sha256 SHA "
        "--schedule-sha256 SHA --fixture-manifest-sha256 SHA "
        "[--backend-commit SHA] [--fixture-contract-sha256 SHA] "
        "[--source-sha256 SHA] [--container-image NAME] "
        "[--environment TEXT]");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                usage_error("missing value for " + arg);
            }
            return argv[i];
        };
        if (arg == "--gpu") {
            options.gpu = std::stoi(next());
        } else if (arg == "--state-dir") {
            options.state_dir = next();
        } else if (arg == "--chunks") {
            options.chunks = parse_chunks(next());
        } else if (arg == "--fixture-dir") {
            options.fixture_dir = next();
        } else if (arg == "--output") {
            options.output = next();
        } else if (arg == "--backend-commit") {
            options.backend_commit = next();
        } else if (arg == "--parent-source-sha256") {
            options.parent_source_sha256 = next();
        } else if (arg == "--frozen-source-sha256") {
            options.frozen_source_sha256 = next();
        } else if (arg == "--schedule-sha256") {
            options.schedule_sha256 = next();
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
            << "Token-SIMD T=103 depth-13/digits-3/ring-65536 complete "
               "block-0 gate (CPU diagonal-vector cache)\n";
            std::exit(0);
        } else {
            usage_error("unknown option " + arg);
        }
    }
    if (options.gpu < 0 || options.state_dir.empty() ||
        options.fixture_dir.empty() || options.output.empty()) {
        usage_error(
            "gpu, state-dir, fixture-dir, and output are required");
    }
    // A Stage-2 worker owns exactly one contiguous 2-chunk half of the four
    // MLP down-projection chunks: {0,1} (primary) or {2,3}. Two workers with
    // these disjoint halves reproduce all four chunks with no overlap.
    if (options.chunks.size() != 2 ||
        !((options.chunks[0] == 0 && options.chunks[1] == 1) ||
          (options.chunks[0] == 2 && options.chunks[1] == 3))) {
        usage_error(
            "--chunks must be exactly the disjoint pair \"0,1\" or \"2,3\"");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit");
    }
    if (options.parent_source_sha256 != PINNED_PARENT_SOURCE_SHA256) {
        usage_error("refusing changed frozen parent source");
    }
    if (options.frozen_source_sha256 != PINNED_FROZEN_SOURCE_SHA256) {
        usage_error("refusing changed T=103 full-block semantic anchor");
    }
    if (options.schedule_sha256 != PINNED_SCHEDULE_SHA256) {
        usage_error("refusing changed Token-SIMD schedule contract");
    }
    if (options.fixture_manifest_sha256 != PINNED_FIXTURE_MANIFEST) {
        usage_error("refusing unpinned T=103 fixture manifest");
    }
    if (options.fixture_contract_sha256 !=
        PINNED_FIXTURE_CONTRACT_SHA256) {
        usage_error("refusing unpinned T=103 fixture contract");
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence " +
                                 options.output.string());
    }
    return options;
}

std::vector<double> read_f64(const std::filesystem::path& path,
                             std::size_t expected_values) {
    const std::uintmax_t expected_bytes = expected_values * sizeof(double);
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) != expected_bytes) {
        throw std::runtime_error("fixture array missing or wrong-sized: " +
                                 path.string());
    }
    std::ifstream stream(path, std::ios::binary);
    std::vector<double> values(expected_values);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(expected_bytes));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("short or trailing fixture data: " +
                                 path.string());
    }
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::runtime_error("non-finite fixture data: " + path.string());
    }
    return values;
}

struct Fixture {
    std::vector<double> input;
    std::vector<double> ln1_weight;
    std::vector<double> ln2_weight;
    std::array<std::vector<double>, 3> qkv;
    std::vector<double> attention_projection;
    std::array<std::vector<double>, 4> mlp_fc;
    std::array<std::vector<double>, 4> mlp_projection;
    std::vector<double> oracle_block_output;
};

Fixture load_fixture(const std::filesystem::path& directory) {
    Fixture fixture;
    fixture.input =
        read_f64(directory / "input__embeddings.bin", T * D);
    fixture.ln1_weight =
        read_f64(directory / "weights__ln1.bin", D);
    fixture.ln2_weight =
        read_f64(directory / "weights__ln2.bin", D);
    const std::vector<double> qkv =
        read_f64(directory / "weights__attn_qkv.bin", 3 * D * D);
    for (std::size_t part = 0; part < 3; ++part) {
        fixture.qkv[part].assign(
            qkv.begin() + static_cast<std::ptrdiff_t>(part * D * D),
            qkv.begin() + static_cast<std::ptrdiff_t>((part + 1) * D * D));
    }
    fixture.attention_projection =
        read_f64(directory / "weights__attn_proj.bin", D * D);
    const std::vector<double> fc =
        read_f64(directory / "weights__mlp_fc.bin", MLP_DIM * D);
    for (std::size_t part = 0; part < COPIES; ++part) {
        fixture.mlp_fc[part].assign(
            fc.begin() + static_cast<std::ptrdiff_t>(part * D * D),
            fc.begin() + static_cast<std::ptrdiff_t>((part + 1) * D * D));
    }
    const std::vector<double> projection =
        read_f64(directory / "weights__mlp_proj.bin", D * MLP_DIM);
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
    fixture.oracle_block_output =
        read_f64(directory / "oracle__block_output.bin", T * D);
    return fixture;
}

std::size_t physical_slot(std::size_t copy, std::size_t feature,
                          std::size_t token_lane) {
    if (copy >= COPIES || feature >= PACK_WIDTH ||
        token_lane >= TOKEN_BATCH) {
        throw std::out_of_range("Token-SIMD slot coordinate out of range");
    }
    return (copy * PACK_WIDTH + feature) * TOKEN_BATCH + token_lane;
}

std::size_t active_tokens(std::size_t group) {
    const std::size_t first = group * TOKEN_BATCH;
    return std::min(TOKEN_BATCH, T - first);
}

// NOTE: the parent's canonical_rotation/add_accumulate_keys/
// required_rotation_keys() rotation-key generators are intentionally absent
// here -- this reader deserializes the rotation keys the Stage-1 writer
// already generated (see main), it never regenerates any key, so those
// helpers would be dead code.

std::vector<std::size_t> active_weight_shifts(
    std::size_t query_group, std::size_t key_group) {
    const std::size_t query_start = query_group * TOKEN_BATCH;
    const std::size_t key_start = key_group * TOKEN_BATCH;
    const std::size_t query_count = active_tokens(query_group);
    const std::size_t key_count = active_tokens(key_group);
    std::vector<std::size_t> shifts;
    for (std::size_t delta = 0; delta < TOKEN_BATCH; ++delta) {
        bool active = false;
        for (std::size_t query_lane = 0; query_lane < query_count;
             ++query_lane) {
            const std::size_t key_lane =
                (query_lane + delta) % TOKEN_BATCH;
            if (key_lane < key_count &&
                key_start + key_lane <= query_start + query_lane) {
                active = true;
                break;
            }
        }
        if (active) {
            shifts.push_back(delta);
        }
    }
    return shifts;
}

class Client {
  public:
    Client(Cc context, Keys& keys)
        : cc_(std::move(context)), keys_(keys) {}

    Ct invsqrt_boundary(const Ct& ciphertext,
                        std::size_t logical_instances,
                        std::string_view stage) {
        if (logical_instances == 0 || logical_instances > TOKEN_BATCH) {
            throw std::invalid_argument("invalid Token-SIMD logical count");
        }
        const auto start = Clock::now();
        Plaintext plaintext;
        Ct local = ciphertext;
        cc_->Decrypt(keys_.secretKey, local, &plaintext);
        plaintext->SetLength(SLOTS);
        std::vector<double> values = plaintext->GetRealPackedValue();
        for (double& value : values) {
            value = exact_invsqrt(value);
        }
        Plaintext refreshed =
            cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
        Ct output = cc_->Encrypt(keys_.publicKey, refreshed);
        ++round_trips_;
        logical_instances_ += logical_instances;
        seconds_ += elapsed_seconds(start);
        std::cout << "[boundary] " << stage
                  << " logical=" << logical_instances << '\n';
        return output;
    }

    Ct gelu_boundary(const Ct& ciphertext,
                     std::size_t active) {
        if (active == 0 || active > TOKEN_BATCH) {
            throw std::invalid_argument("invalid GELU active-token count");
        }
        const auto start = Clock::now();
        Plaintext plaintext;
        Ct local = ciphertext;
        cc_->Decrypt(keys_.secretKey, local, &plaintext);
        plaintext->SetLength(SLOTS);
        const std::vector<double> raw = plaintext->GetRealPackedValue();
        std::vector<double> transformed(SLOTS, 0.0);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t feature = 0; feature < D; ++feature) {
                for (std::size_t lane = 0; lane < active; ++lane) {
                    const std::size_t slot =
                        physical_slot(copy, feature, lane);
                    transformed[slot] = exact_gelu(raw[slot]);
                }
            }
        }
        Plaintext refreshed = cc_->MakeCKKSPackedPlaintext(
            transformed, 1, 0, nullptr, SLOTS);
        Ct output = cc_->Encrypt(keys_.publicKey, refreshed);
        ++round_trips_;
        logical_instances_ += active * COPIES;
        seconds_ += elapsed_seconds(start);
        return output;
    }

    void begin_attention() {
        if (attention_active_) {
            throw std::logic_error("attention client state already active");
        }
        scores_.assign(
            T,
            std::vector<std::vector<double>>(
                HEADS, std::vector<double>(T,
                    std::numeric_limits<double>::quiet_NaN())));
        weights_.clear();
        attention_active_ = true;
    }

    void reduce_score_tile(const Ct& ciphertext,
                           std::size_t query_group,
                           std::size_t key_group) {
        if (!attention_active_ || key_group > query_group ||
            scores_.size() != T) {
            throw std::logic_error("score-tile client state mismatch");
        }
        const auto start = Clock::now();
        Plaintext plaintext;
        Ct local = ciphertext;
        cc_->Decrypt(keys_.secretKey, local, &plaintext);
        plaintext->SetLength(SLOTS);
        const std::vector<double> values =
            plaintext->GetRealPackedValue();
        const std::size_t query_start = query_group * TOKEN_BATCH;
        const std::size_t key_start = key_group * TOKEN_BATCH;
        const std::size_t query_count = active_tokens(query_group);
        const std::size_t key_count = active_tokens(key_group);
        std::size_t logical = 0;
        for (std::size_t query_lane = 0; query_lane < query_count;
             ++query_lane) {
            const std::size_t row = query_start + query_lane;
            for (std::size_t key_lane = 0; key_lane < key_count;
                 ++key_lane) {
                const std::size_t col = key_start + key_lane;
                if (col > row) {
                    continue;
                }
                for (std::size_t head = 0; head < HEADS; ++head) {
                    const std::size_t copy = head / HEADS_PER_COPY;
                    const std::size_t local_head =
                        head % HEADS_PER_COPY;
                    const std::size_t feature =
                        local_head * TOKEN_BATCH + key_lane;
                    scores_[row][head][col] =
                        values[physical_slot(copy, feature,
                                             query_lane)];
                    ++logical;
                }
            }
        }
        ++round_trips_;
        logical_instances_ += logical;
        seconds_ += elapsed_seconds(start);
    }

    void finalize_attention() {
        if (!attention_active_ || scores_.size() != T) {
            throw std::logic_error("softmax client state mismatch");
        }
        weights_.assign(
            T,
            std::vector<std::vector<double>>(
                HEADS, std::vector<double>(T, 0.0)));
        for (std::size_t row = 0; row < T; ++row) {
            for (std::size_t head = 0; head < HEADS; ++head) {
                double maximum = -std::numeric_limits<double>::infinity();
                for (std::size_t col = 0; col <= row; ++col) {
                    const double value = scores_[row][head][col];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(
                            "missing/non-finite causal attention score");
                    }
                    maximum = std::max(maximum, value);
                }
                double denominator = 0.0;
                for (std::size_t col = 0; col <= row; ++col) {
                    denominator += std::exp(
                        scores_[row][head][col] - maximum);
                }
                if (!std::isfinite(denominator) || denominator <= 0.0) {
                    throw std::runtime_error(
                        "invalid stable-softmax denominator");
                }
                for (std::size_t col = 0; col <= row; ++col) {
                    weights_[row][head][col] =
                        std::exp(scores_[row][head][col] -
                                 maximum) /
                        denominator;
                }
            }
        }
    }

    Ct emit_weight_tile(std::size_t query_group,
                        std::size_t key_group,
                        std::size_t delta) {
        if (!attention_active_ || weights_.size() != T) {
            throw std::logic_error("weight-tile client state mismatch");
        }
        const auto start = Clock::now();
        const std::size_t query_start = query_group * TOKEN_BATCH;
        const std::size_t key_start = key_group * TOKEN_BATCH;
        const std::size_t query_count = active_tokens(query_group);
        const std::size_t key_count = active_tokens(key_group);
        std::vector<double> packed(SLOTS, 0.0);
        std::size_t logical = 0;
        for (std::size_t query_lane = 0; query_lane < query_count;
             ++query_lane) {
            const std::size_t row = query_start + query_lane;
            const std::size_t key_lane =
                (query_lane + delta) % TOKEN_BATCH;
            if (key_lane >= key_count) {
                continue;
            }
            const std::size_t col = key_start + key_lane;
            if (col > row) {
                continue;
            }
            for (std::size_t head = 0; head < HEADS; ++head) {
                const double weight =
                    weights_[row][head][col];
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    for (std::size_t dim = 0; dim < HEAD_DIM; ++dim) {
                        packed[physical_slot(
                            copy, head * HEAD_DIM + dim,
                            query_lane)] = weight;
                    }
                }
                ++logical;
            }
        }
        if (logical == 0) {
            throw std::runtime_error("emitted empty weight tile");
        }
        Plaintext plaintext = cc_->MakeCKKSPackedPlaintext(
            packed, 1, 0, nullptr, SLOTS);
        Ct output = cc_->Encrypt(keys_.publicKey, plaintext);
        ++round_trips_;
        logical_instances_ += logical;
        seconds_ += elapsed_seconds(start);
        return output;
    }

    void clear_attention() {
        if (!attention_active_) {
            throw std::logic_error("attention client state mismatch");
        }
        scores_.clear();
        weights_.clear();
        attention_active_ = false;
    }

    std::size_t round_trips() const { return round_trips_; }
    std::size_t logical_instances() const { return logical_instances_; }
    double seconds() const { return seconds_; }

  private:
    Cc cc_;
    Keys& keys_;
    std::size_t round_trips_ = 0;
    std::size_t logical_instances_ = 0;
    double seconds_ = 0.0;
    bool attention_active_ = false;
    std::vector<std::vector<std::vector<double>>> scores_;
    std::vector<std::vector<std::vector<double>>> weights_;
};

struct EvaluationResult {
    // Stage-2 sharded output: per-group PARTIAL down-projection sum over the
    // two chunks this worker owns (NOT the full block output). The merge step
    // EvalAdds the two workers' partial_mlp vectors, then adds residual1.
    std::vector<Ct> partial_mlp;
    // residual1 (input_embeddings + attention_projection), needed by the
    // merge step to form block_output = residual1 + (partial_lo + partial_hi).
    // Only serialized by the primary worker; still returned by both so the
    // op-count guard is symmetric.
    std::vector<Ct> residual1;
    std::size_t matrix_products = 0;
    std::size_t down_projection_chunk_matmuls = 0;
    std::size_t ct_ct_multiplications = 0;
    std::size_t ct_plain_multiplications = 0;
    std::size_t explicit_rotations = 0;
    std::size_t accumulate_sum_calls = 0;
    std::size_t score_tile_decryptions = 0;
    std::size_t weight_tile_encryptions = 0;
    std::size_t cached_key_lane_shifts = 0;
    std::size_t cached_value_lane_shifts = 0;
    std::size_t packed_output_level = 0;
    std::size_t diagonal_cache_hits = 0;
    std::size_t diagonal_cache_misses = 0;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture, Client& client)
        : cc_(std::move(context)), fixture_(fixture), client_(client) {}

    EvaluationResult evaluate(
        const std::vector<Ct>& encrypted_inputs,
        const std::vector<std::size_t>& active_tokens,
        const std::vector<std::size_t>& chunks) {
        if (encrypted_inputs.size() != TOKEN_GROUPS ||
            active_tokens.size() != TOKEN_GROUPS) {
            throw std::invalid_argument(
                "Token-SIMD input/active-count mismatch");
        }
        if (chunks.empty()) {
            throw std::invalid_argument(
                "Stage-2 MLP shard reader needs >=1 assigned chunk");
        }
        for (const std::size_t chunk : chunks) {
            if (chunk >= COPIES) {
                throw std::invalid_argument(
                    "assigned MLP chunk index out of range");
            }
        }
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            if (active_tokens[group] != ::active_tokens(group)) {
                throw std::invalid_argument(
                    "Token-SIMD active-count schedule mismatch");
            }
        }

        std::vector<Ct> query(TOKEN_GROUPS);
        std::vector<Ct> key(TOKEN_GROUPS);
        std::vector<Ct> value(TOKEN_GROUPS);
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            Ct normalized = layernorm(
                encrypted_inputs[group], active_tokens[group], "ln1");
            const BabyRotations baby = baby_rotations(normalized);
            query[group] = matmul(baby, fixture_.qkv[0]);
            key[group] = matmul(baby, fixture_.qkv[1]);
            value[group] = matmul(baby, fixture_.qkv[2]);
            std::cout << "[stage] qkv group=" << group
                      << " active=" << active_tokens[group]
                      << " level=" << query[group]->GetLevel() << '\n';
        }

        client_.begin_attention();
        const double score_scale =
            1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
        for (std::size_t key_group = 0; key_group < TOKEN_GROUPS;
             ++key_group) {
            const auto shifted_keys =
                build_group_shifts(key[key_group], key_group, true);
            for (std::size_t query_group = key_group;
                 query_group < TOKEN_GROUPS; ++query_group) {
                Ct packed_score;
                bool first = true;
                for (const std::size_t delta :
                     active_weight_shifts(query_group, key_group)) {
                    Ct products = multiply(
                        query[query_group], shifted_keys[delta]);
                    for (std::size_t head = 0; head < HEADS; ++head) {
                        Ct masked = multiply_plain(
                            products, head_mask_plain(head));
                        Ct score = sum_broadcast(masked);
                        Ct isolated = multiply_plain(
                            score, score_isolation_plain(
                                query_group, key_group, delta, head,
                                score_scale));
                        packed_score =
                            first ? isolated
                                  : cc_->EvalAdd(packed_score, isolated);
                        first = false;
                    }
                }
                if (first) {
                    throw std::runtime_error("empty packed score tile");
                }
                client_.reduce_score_tile(
                    packed_score, query_group, key_group);
                ++score_tile_decryptions_;
                std::cout << "[stage] score tile q=" << query_group
                          << " k=" << key_group << '\n';
            }
        }
        client_.finalize_attention();
        query.clear();
        key.clear();

        std::vector<Ct> context(TOKEN_GROUPS);
        for (std::size_t key_group = 0; key_group < TOKEN_GROUPS;
             ++key_group) {
            const auto shifted_values =
                build_group_shifts(value[key_group], key_group, false);
            for (std::size_t query_group = key_group;
                 query_group < TOKEN_GROUPS; ++query_group) {
                for (const std::size_t delta :
                     active_weight_shifts(query_group, key_group)) {
                    Ct weight = client_.emit_weight_tile(
                        query_group, key_group, delta);
                    ++weight_tile_encryptions_;
                    Ct contribution =
                        multiply(weight, shifted_values[delta]);
                    context[query_group] =
                        context[query_group]
                            ? cc_->EvalAdd(context[query_group], contribution)
                            : contribution;
                }
            }
        }
        client_.clear_attention();
        value.clear();

        std::vector<Ct> attention_projection(TOKEN_GROUPS);
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            attention_projection[group] =
                matmul(baby_rotations(context[group]),
                       fixture_.attention_projection);
        }
        context.clear();

        std::vector<Ct> residual1(TOKEN_GROUPS);
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            residual1[group] = cc_->EvalAdd(
                encrypted_inputs[group], attention_projection[group]);
        }
        attention_projection.clear();

        std::vector<Ct> partial_mlp(TOKEN_GROUPS);
        std::size_t chunk_matmuls = 0;
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            Ct normalized2 = layernorm(
                residual1[group], active_tokens[group], "ln2");
            require_remaining_depth(
                normalized2->GetLevel(), 4,
                "Token-SIMD MLP group " + std::to_string(group));
            const BabyRotations fc_baby =
                baby_rotations(normalized2);
            // FC up-projection + GELU + cross-copy replication are recomputed
            // in full on BOTH workers (hidden[0..3] is deterministic in the
            // plaintext it encrypts), exactly as Stage 1 recomputes LN1 on
            // every worker -- only the down-projection accumulation below is
            // sharded.
            std::array<Ct, COPIES> hidden{};
            for (std::size_t chunk = 0; chunk < COPIES; ++chunk) {
                hidden[chunk] =
                    matmul(fc_baby, fixture_.mlp_fc[chunk]);
            }

            Ct packed_hidden;
            for (std::size_t chunk = 0; chunk < COPIES; ++chunk) {
                Ct selected = multiply_plain(
                    hidden[chunk], copy_active_plain(chunk));
                packed_hidden =
                    chunk == 0
                        ? selected
                        : cc_->EvalAdd(packed_hidden, selected);
            }
            Ct packed_activated = client_.gelu_boundary(
                packed_hidden, active_tokens[group]);
            for (std::size_t chunk = 0; chunk < COPIES; ++chunk) {
                Ct selected = multiply_plain(
                    packed_activated, copy_active_plain(chunk));
                hidden[chunk] = replicate_copies(selected);
            }

            // STAGE-2 SPLIT: accumulate ONLY this worker's assigned chunks
            // into a per-group PARTIAL down-projection sum. The frozen source
            // summed all COPIES chunks here (mlp = c0+c1+c2+c3); this worker
            // computes the disjoint sub-sum over `chunks` (e.g. c0+c1). CKKS
            // EvalAdd is exact and associative, so partial_lo + partial_hi
            // (merged downstream) equals the frozen full sum bit-for-bit up to
            // float reassociation (~1e-13), proved to <=1e-9 by
            // test_shard_mlp_reader_merge_t103_depth13_contract.py.
            Ct partial;
            bool first_chunk = true;
            for (const std::size_t chunk : chunks) {
                Ct contribution =
                    matmul(baby_rotations(hidden[chunk]),
                           fixture_.mlp_projection[chunk]);
                partial = first_chunk
                              ? contribution
                              : cc_->EvalAdd(partial, contribution);
                first_chunk = false;
                ++chunk_matmuls;
            }
            partial_mlp[group] = partial;
            std::cout << "[stage] partial mlp group=" << group
                      << " chunks=" << chunks.size()
                      << " level=" << partial_mlp[group]->GetLevel()
                      << '\n';
        }

        EvaluationResult result;
        result.partial_mlp = std::move(partial_mlp);
        result.residual1 = std::move(residual1);
        result.down_projection_chunk_matmuls = chunk_matmuls;
        result.matrix_products = matmul_count_;
        result.ct_ct_multiplications = ct_ct_multiply_count_;
        result.ct_plain_multiplications = ct_plain_multiply_count_;
        result.explicit_rotations = explicit_rotation_count_;
        result.accumulate_sum_calls = accumulate_sum_count_;
        result.score_tile_decryptions = score_tile_decryptions_;
        result.weight_tile_encryptions = weight_tile_encryptions_;
        result.cached_key_lane_shifts = cached_key_lane_shifts_;
        result.cached_value_lane_shifts = cached_value_lane_shifts_;
        result.diagonal_cache_hits = diagonal_cache_hits_;
        result.diagonal_cache_misses = diagonal_cache_misses_;
        for (const Ct& output : result.partial_mlp) {
            result.packed_output_level =
                std::max(result.packed_output_level,
                         output->GetLevel());
        }
        return result;
    }

  private:
    struct BabyRotations {
        std::array<Ct, BSGS_N1> values{};
    };

    Plaintext raw_plain(const std::vector<double>& values) {
        if (values.size() != SLOTS) {
            throw std::invalid_argument("plaintext slot-count mismatch");
        }
        return cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
    }

    Plaintext repeated_plain(const std::vector<double>& values) {
        if (values.size() != D) {
            throw std::invalid_argument("repeated plaintext must have D values");
        }
        std::vector<double> packed(SLOTS, 0.0);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t dim = 0; dim < D; ++dim) {
                for (std::size_t token = 0; token < TOKEN_BATCH; ++token) {
                    packed[physical_slot(copy, dim, token)] = values[dim];
                }
            }
        }
        return raw_plain(packed);
    }

    Plaintext lane_mask_plain(std::size_t delta, bool no_wrap) {
        if (delta == 0 || delta >= TOKEN_BATCH) {
            throw std::invalid_argument("invalid lane-mask delta");
        }
        std::vector<double> packed(SLOTS, 0.0);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t feature = 0; feature < PACK_WIDTH;
                 ++feature) {
                for (std::size_t lane = 0; lane < TOKEN_BATCH; ++lane) {
                    const bool selected = lane < TOKEN_BATCH - delta;
                    if (selected == no_wrap) {
                        packed[physical_slot(copy, feature, lane)] = 1.0;
                    }
                }
            }
        }
        return raw_plain(packed);
    }

    Plaintext head_mask_plain(std::size_t head) {
        if (head >= HEADS) {
            throw std::invalid_argument("head outside attention shape");
        }
        std::vector<double> mask(D, 0.0);
        std::fill_n(mask.begin() +
                        static_cast<std::ptrdiff_t>(head * HEAD_DIM),
                    HEAD_DIM, 1.0);
        return repeated_plain(mask);
    }

    Plaintext score_isolation_plain(
        std::size_t query_group, std::size_t key_group,
        std::size_t delta, std::size_t head, double scale) {
        const std::size_t query_start = query_group * TOKEN_BATCH;
        const std::size_t key_start = key_group * TOKEN_BATCH;
        const std::size_t query_count = ::active_tokens(query_group);
        const std::size_t key_count = ::active_tokens(key_group);
        const std::size_t copy = head / HEADS_PER_COPY;
        const std::size_t local_head = head % HEADS_PER_COPY;
        std::vector<double> packed(SLOTS, 0.0);
        for (std::size_t query_lane = 0; query_lane < query_count;
             ++query_lane) {
            const std::size_t key_lane =
                (query_lane + delta) % TOKEN_BATCH;
            if (key_lane >= key_count ||
                key_start + key_lane > query_start + query_lane) {
                continue;
            }
            const std::size_t feature =
                local_head * TOKEN_BATCH + key_lane;
            packed[physical_slot(copy, feature, query_lane)] = scale;
        }
        return raw_plain(packed);
    }

    Plaintext copy_active_plain(std::size_t copy) {
        if (copy >= COPIES) {
            throw std::invalid_argument("copy outside Token-SIMD layout");
        }
        std::vector<double> packed(SLOTS, 0.0);
        for (std::size_t feature = 0; feature < D; ++feature) {
            for (std::size_t lane = 0; lane < TOKEN_BATCH; ++lane) {
                packed[physical_slot(copy, feature, lane)] = 1.0;
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
        ++accumulate_sum_count_;
        return cc_->AccumulateSum(
            input, static_cast<int>(PACK_WIDTH),
            static_cast<int>(TOKEN_BATCH));
    }

    Ct mean_broadcast(const Ct& input) {
        Ct sum = sum_broadcast(input);
        Plaintext mean_scale = repeated_plain(std::vector<double>(
            D, 1.0 / static_cast<double>(D)));
        return multiply_plain(sum, mean_scale);
    }

    Ct layernorm(const Ct& input, std::size_t active,
                 std::string_view stage) {
        Ct mean = mean_broadcast(input);
        Ct centered = cc_->EvalSub(input, mean);
        Ct variance = mean_broadcast(multiply(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        Ct inverse = client_.invsqrt_boundary(
            variance, active, stage);
        Ct normalized = multiply(centered, inverse);
        const std::vector<double>& weight =
            stage == "ln1" ? fixture_.ln1_weight
                           : fixture_.ln2_weight;
        return multiply_plain(normalized, repeated_plain(weight));
    }

    BabyRotations baby_rotations(const Ct& input) {
        std::vector<std::int32_t> indices;
        indices.reserve(BSGS_N1 - 1);
        for (std::size_t small = 1; small < BSGS_N1; ++small) {
            indices.push_back(static_cast<std::int32_t>(
                TOKEN_BATCH * small));
        }
        const auto rotated = cc_->EvalFastRotation(
            input, indices, cc_->GetCyclotomicOrder(), nullptr);
        explicit_rotation_count_ += indices.size();
        if (rotated.size() != indices.size()) {
            throw std::runtime_error(
                "FIDESlib returned incomplete Token-SIMD baby rotations");
        }
        BabyRotations output;
        output.values[0] = input;
        for (std::size_t small = 1; small < BSGS_N1; ++small) {
            output.values[small] = rotated.at(small - 1);
        }
        return output;
    }

    // Builds (on the first call for a given weight matrix) or returns
    // (on every later call) the full set of BSGS_N1*BSGS_N2 diagonal
    // std::vector<double> for `weight`, keyed by the weight vector's
    // stable address -- fixture_.qkv[0..2], fixture_.attention_projection,
    // and fixture_.mlp_fc[0..3]/mlp_projection[0..3] each keep one fixed
    // address for the whole evaluate() call, so every one of the 13
    // token-group calls sharing one weight resolves to the same cache
    // entry. Only the CPU-side vector<double> is cached here -- a FRESH
    // Plaintext object is still built from it (via raw_plain()) on every
    // single call in matmul() below, so no GPU-resident Plaintext object
    // is ever reused across multPt calls (the abandoned diagcache
    // attempt's exact crash mechanism; see the file header).
    const std::vector<double>& cached_packed_values(
        const std::vector<double>& weight, std::size_t giant,
        std::size_t small) {
        auto found = diagonal_vector_cache_.find(&weight);
        if (found == diagonal_vector_cache_.end()) {
            std::vector<std::vector<double>> built(BSGS_N1 * BSGS_N2);
            for (std::size_t build_giant = 0; build_giant < BSGS_N2;
                 ++build_giant) {
                for (std::size_t build_small = 0; build_small < BSGS_N1;
                     ++build_small) {
                    const std::size_t diagonal =
                        BSGS_N1 * build_giant + build_small;
                    std::vector<double> packed_values(SLOTS, 0.0);
                    for (std::size_t copy = 0; copy < COPIES; ++copy) {
                        for (std::size_t row = 0; row < PACK_WIDTH;
                             ++row) {
                            const std::size_t rolled_row =
                                (row + PACK_WIDTH -
                                 BSGS_N1 * build_giant) %
                                PACK_WIDTH;
                            const std::size_t column =
                                (rolled_row + diagonal) % PACK_WIDTH;
                            if (rolled_row < D && column < D) {
                                const double coefficient =
                                    weight[rolled_row * D + column];
                                for (std::size_t token = 0;
                                     token < TOKEN_BATCH; ++token) {
                                    packed_values[physical_slot(
                                        copy, row, token)] =
                                        coefficient;
                                }
                            }
                        }
                    }
                    built[diagonal] = std::move(packed_values);
                }
            }
            found = diagonal_vector_cache_
                        .emplace(&weight, std::move(built))
                        .first;
            ++diagonal_cache_misses_;
        } else {
            ++diagonal_cache_hits_;
        }
        return found->second.at(BSGS_N1 * giant + small);
    }

    Ct matmul(const BabyRotations& baby,
              const std::vector<double>& weight) {
        if (weight.size() != D * D) {
            throw std::invalid_argument("query weight must be D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::vector<double>& packed_values =
                    cached_packed_values(weight, giant, small);
                Plaintext diagonal_plain =
                    raw_plain(packed_values);
                Ct term = multiply_plain(
                    baby.values[small], diagonal_plain);
                inner = small == 0 ? term
                                   : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(
                    inner, static_cast<std::int32_t>(
                               TOKEN_BATCH * BSGS_N1 * giant));
                ++explicit_rotation_count_;
            }
            result = giant == 0 ? inner : cc_->EvalAdd(result, inner);
        }
        ++matmul_count_;
        return result;
    }

    Ct lane_shift(const Ct& input, std::size_t delta) {
        if (delta == 0) {
            return input;
        }
        if (delta >= TOKEN_BATCH) {
            throw std::invalid_argument("lane shift outside token batch");
        }
        Ct from_next = cc_->EvalRotate(
            input, static_cast<std::int32_t>(delta));
        Ct from_same = cc_->EvalRotate(
            input, static_cast<std::int32_t>(
                static_cast<std::int64_t>(delta) -
                static_cast<std::int64_t>(TOKEN_BATCH)));
        explicit_rotation_count_ += 2;
        return cc_->EvalAdd(
            multiply_plain(from_next, lane_mask_plain(delta, true)),
            multiply_plain(from_same, lane_mask_plain(delta, false)));
    }

    std::array<Ct, TOKEN_BATCH> build_group_shifts(
        const Ct& input, std::size_t key_group, bool keys) {
        std::array<bool, TOKEN_BATCH> needed{};
        for (std::size_t query_group = key_group;
             query_group < TOKEN_GROUPS; ++query_group) {
            for (const std::size_t delta :
                 active_weight_shifts(query_group, key_group)) {
                needed[delta] = true;
            }
        }
        std::array<Ct, TOKEN_BATCH> shifted{};
        for (std::size_t delta = 0; delta < TOKEN_BATCH; ++delta) {
            if (!needed[delta]) {
                continue;
            }
            shifted[delta] = lane_shift(input, delta);
            if (delta != 0) {
                if (keys) {
                    ++cached_key_lane_shifts_;
                } else {
                    ++cached_value_lane_shifts_;
                }
            }
        }
        return shifted;
    }

    Ct replicate_copies(const Ct& input) {
        const std::vector<std::int32_t> indices = {
            static_cast<std::int32_t>(PACK_WIDTH * TOKEN_BATCH),
            -static_cast<std::int32_t>(PACK_WIDTH * TOKEN_BATCH),
            static_cast<std::int32_t>(
                2 * PACK_WIDTH * TOKEN_BATCH),
        };
        const auto rotated = cc_->EvalFastRotation(
            input, indices, cc_->GetCyclotomicOrder(), nullptr);
        if (rotated.size() != indices.size()) {
            throw std::runtime_error(
                "incomplete Token-SIMD cross-copy rotations");
        }
        explicit_rotation_count_ += indices.size();
        Ct output = input;
        for (const Ct& copy : rotated) {
            output = cc_->EvalAdd(output, copy);
        }
        return output;
    }

    void require_remaining_depth(std::size_t current_level,
                                 std::size_t levels_needed,
                                 const std::string& stage) const {
        if (current_level > MULT_DEPTH ||
            levels_needed > MULT_DEPTH - current_level) {
            throw std::runtime_error(
                "insufficient depth before " + stage);
        }
    }

    Cc cc_;
    const Fixture& fixture_;
    Client& client_;
    std::size_t matmul_count_ = 0;
    std::size_t ct_ct_multiply_count_ = 0;
    std::size_t ct_plain_multiply_count_ = 0;
    std::size_t explicit_rotation_count_ = 0;
    std::size_t accumulate_sum_count_ = 0;
    std::size_t score_tile_decryptions_ = 0;
    std::size_t weight_tile_encryptions_ = 0;
    std::size_t cached_key_lane_shifts_ = 0;
    std::size_t cached_value_lane_shifts_ = 0;
    std::size_t diagonal_cache_hits_ = 0;
    std::size_t diagonal_cache_misses_ = 0;
    std::map<const std::vector<double>*, std::vector<std::vector<double>>>
        diagonal_vector_cache_;
};

// Stage-2 MLP shard READER evidence: this worker computed and serialized its
// per-group PARTIAL down-projection sum. It does NOT measure accuracy -- the
// merge step reconstructs the full block output and measures it. No metrics,
// no final decrypt of a block output here.
std::string make_json(
    const Options& options, std::uint32_t ring,
    const EvaluationResult& evaluation, double fixture_seconds,
    double deserialize_seconds, double load_context_gpu_seconds,
    double encryption_seconds, double evaluation_seconds,
    double serialize_partials_seconds, std::size_t serialized_partials,
    std::size_t serialized_residuals, std::size_t round_trips,
    std::size_t logical_instances, double boundary_seconds) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"Scheme B Token-SIMD T=103 depth-13/digits-3/"
           "ring-65536 2-GPU process-per-GPU sharding, Stage 2 (MLP "
           "down-projection chunk split) -- SHARDED MLP READER phase; "
           "produces a per-group PARTIAL down-projection sum over its two "
           "assigned chunks for a later exact EvalAdd merge\",\n"
        << "  \"implementation_version\": "
           "\"t103-scheme-b-token-simd-shard-mlp-reader-depth13-v1\",\n"
        << "  \"phase\": \"reader\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS)\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \""
        << json_escape(options.backend_commit) << "\",\n"
        << "  \"container_image\": \""
        << json_escape(options.container_image) << "\",\n"
        << "  \"environment\": \""
        << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"chunks_owned\": [";
    for (std::size_t i = 0; i < options.chunks.size(); ++i) {
        out << options.chunks[i]
            << (i + 1 == options.chunks.size() ? "" : ", ");
    }
    out << "],\n"
        << "  \"is_primary_worker\": "
        << (options.primary() ? "true" : "false") << ",\n"
        << "  \"source_sha256\": \""
        << json_escape(options.source_sha256) << "\",\n"
        << "  \"fullblock_parent_sha256\": \""
        << json_escape(std::string(PINNED_FULLBLOCK_PARENT_SHA256))
        << "\",\n"
        << "  \"stage1_reader_parent_sha256\": \""
        << json_escape(std::string(PINNED_STAGE1_READER_SHA256))
        << "\",\n"
        << "  \"schedule_sha256\": \""
        << json_escape(options.schedule_sha256) << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": " << D << ", \"T\": " << T
        << ", \"token_batch\": " << TOKEN_BATCH
        << ", \"token_groups\": "
        << evaluation.partial_mlp.size() << ",\n"
        << "    \"pack_width\": " << PACK_WIDTH
        << ", \"copies\": " << COPIES << ", \"batch_slots\": "
        << SLOTS << ", \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"matrix_products\": " << evaluation.matrix_products << ",\n"
        << "    \"down_projection_chunk_matmuls\": "
        << evaluation.down_projection_chunk_matmuls
        << ", \"expected_down_projection_chunk_matmuls\": "
        << TOKEN_GROUPS * options.chunks.size() << ",\n"
        << "    \"partial_mlp_level\": "
        << evaluation.packed_output_level << ",\n"
        << "    \"diagonal_cache_hits\": "
        << evaluation.diagonal_cache_hits << ",\n"
        << "    \"diagonal_cache_misses\": "
        << evaluation.diagonal_cache_misses << "\n"
        << "  },\n"
        << "  \"serialized_partials\": " << serialized_partials << ",\n"
        << "  \"serialized_residuals\": " << serialized_residuals << ",\n"
        << "  \"cross_process_breakdown\": {\n"
        << "    \"reader_deserialize_seconds\": " << deserialize_seconds
        << ",\n"
        << "    \"reader_load_context_gpu_seconds\": "
        << load_context_gpu_seconds << "\n"
        << "  },\n"
        << "  \"timings_seconds\": {\n"
        << "    \"load_verified_fixture\": " << fixture_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds << ",\n"
        << "    \"encrypted_evaluation\": " << evaluation_seconds << ",\n"
        << "    \"client_boundary_seconds_total\": " << boundary_seconds
        << ",\n"
        << "    \"server_linear_algebra_seconds\": "
        << (evaluation_seconds - boundary_seconds) << ",\n"
        << "    \"serialize_partials_seconds\": "
        << serialize_partials_seconds << "\n"
        << "  },\n"
        << "  \"protocol\": {\"round_trips\": " << round_trips
        << ", \"logical_boundary_instances\": " << logical_instances
        << "},\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"produces_full_block_output\": false,\n"
        << "  \"evaluator_has_private_key\": false\n"
        << "}\n";
    return out.str();
}

void write_exclusive(const std::filesystem::path& path,
                     const std::string& contents) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const int descriptor =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) {
        throw std::runtime_error("could not exclusively create " +
                                 path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t count =
            ::write(descriptor, contents.data() + offset,
                    contents.size() - offset);
        if (count <= 0) {
            ::close(descriptor);
            throw std::runtime_error("failed writing " + path.string());
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error("failed closing " + path.string());
    }
}

// Non-secret key=value companion the Stage-1 writer left in --state-dir --
// just the writer's own timing/config numbers, no keys, no paths (Stage-1
// reader pattern).
std::map<std::string, std::string> read_writer_timing(
    const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(
            "could not open writer timing file: " + path.string());
    }
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(stream, line)) {
        const std::size_t split = line.find('=');
        if (split == std::string::npos) {
            continue;
        }
        values[line.substr(0, split)] = line.substr(split + 1);
    }
    for (const char* required :
         {"multiplicative_depth", "ring_dim"}) {
        if (values.find(required) == values.end()) {
            throw std::runtime_error(
                std::string("writer timing file missing field: ") + required);
        }
    }
    return values;
}

// Serialize one packed ciphertext to --state-dir. THIS is the load-bearing,
// otherwise-unexercised capability the cross-process merge depends on:
// FIDESlib round-tripping a GPU-resident Ciphertext through Serialize. See
// the file header [A] note -- verified only once a real GPU run is allowed.
void serialize_ciphertext(const std::filesystem::path& path, const Ct& ct) {
    if (std::filesystem::exists(path)) {
        throw std::runtime_error(
            "refusing to overwrite existing partial ciphertext: " +
            path.string());
    }
    if (!Serial::SerializeToFile(path.string(), ct, SerType::BINARY)) {
        throw std::runtime_error(
            "could not serialize partial ciphertext to " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        const auto fixture_start = Clock::now();
        const Fixture fixture = load_fixture(options.fixture_dir);
        const double fixture_seconds = elapsed_seconds(fixture_start);

        const std::map<std::string, std::string> writer_timing =
            read_writer_timing(options.state_dir / "writer_timing.txt");
        if (std::stoul(writer_timing.at("multiplicative_depth")) !=
            MULT_DEPTH) {
            throw std::runtime_error(
                "state-dir was not produced by a depth-13 Token-SIMD writer "
                "(multiplicative_depth mismatch)");
        }
        if (std::stoul(writer_timing.at("ring_dim")) != 2 * SLOTS) {
            throw std::runtime_error(
                "state-dir was not produced by a Token-SIMD-parameter-matched "
                "writer (ring_dim mismatch)");
        }

        // Cross-process reload: deserialize everything the Stage-1 shard
        // writer serialized. NO GenCryptoContext / KeyGen / EvalMultKeyGen /
        // EvalRotateKeyGen anywhere in this file -- verified textually by the
        // contract test.
        const auto deserialize_start = Clock::now();
        Cc cc;
        if (!Serial::DeserializeFromFile(
                (options.state_dir / "crypto-context.txt").string(), cc,
                SerType::BINARY)) {
            throw std::runtime_error("could not deserialize CryptoContext");
        }
        Keys keys;
        if (!Serial::DeserializeFromFile(
                (options.state_dir / "public-key.txt").string(),
                keys.publicKey, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize PublicKey");
        }
        // Secret-key hygiene: this reader plays the (restarted) client role
        // for the exact LN1/LN2/GELU boundaries, so it legitimately needs the
        // secret key. Its path/contents are never printed.
        if (!Serial::DeserializeFromFile(
                (options.state_dir / "secret-key.txt").string(),
                keys.secretKey, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize PrivateKey");
        }
        std::ifstream mult_key_stream(
            options.state_dir / "eval-mult-key.bin",
            std::ios::in | std::ios::binary);
        if (!mult_key_stream.is_open() ||
            !cc->DeserializeEvalMultKey(mult_key_stream, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize EvalMultKey");
        }
        mult_key_stream.close();
        std::ifstream rotation_key_stream(
            options.state_dir / "eval-automorphism-key.bin",
            std::ios::in | std::ios::binary);
        if (!rotation_key_stream.is_open() ||
            !cc->DeserializeEvalAutomorphismKey(rotation_key_stream,
                                                SerType::BINARY)) {
            throw std::runtime_error(
                "could not deserialize EvalAutomorphismKey");
        }
        rotation_key_stream.close();
        const double deserialize_seconds =
            elapsed_seconds(deserialize_start);

        const auto load_context_start = Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        const double load_context_gpu_seconds =
            elapsed_seconds(load_context_start);
        const std::uint32_t ring = cc->GetRingDimension();
        if (ring < 2 * SLOTS) {
            throw std::runtime_error(
                "deserialized context cannot provide requested Token-SIMD "
                "slots");
        }
        std::cout << "[context] ring=" << ring << " slots=" << SLOTS
                  << " token_batch=" << TOKEN_BATCH
                  << " chunks=" << options.chunk_tag()
                  << " deserialize_seconds=" << deserialize_seconds
                  << " load_context_gpu_seconds=" << load_context_gpu_seconds
                  << '\n';

        const auto encryption_start = Clock::now();
        std::vector<Ct> encrypted_inputs;
        std::vector<std::size_t> active_counts;
        encrypted_inputs.reserve(TOKEN_GROUPS);
        active_counts.reserve(TOKEN_GROUPS);
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            const std::size_t first = group * TOKEN_BATCH;
            const std::size_t active = active_tokens(group);
            std::vector<double> packed(SLOTS, 0.0);
            for (std::size_t token_lane = 0; token_lane < active;
                 ++token_lane) {
                const std::size_t token = first + token_lane;
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    for (std::size_t dim = 0; dim < D; ++dim) {
                        packed[physical_slot(copy, dim, token_lane)] =
                            fixture.input[token * D + dim];
                    }
                }
            }
            Plaintext plaintext =
                cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
            encrypted_inputs.push_back(
                cc->Encrypt(keys.publicKey, plaintext));
            active_counts.push_back(active);
        }
        cc->Synchronize();
        const double encryption_seconds =
            elapsed_seconds(encryption_start);

        Client client(cc, keys);
        EncryptedEvaluator evaluator(cc, fixture, client);
        const auto evaluation_start = Clock::now();
        EvaluationResult evaluation = evaluator.evaluate(
            encrypted_inputs, active_counts, options.chunks);
        cc->Synchronize();
        const double evaluation_seconds =
            elapsed_seconds(evaluation_start);

        // This worker's declared work: exactly one down-projection matmul per
        // (token group, assigned chunk). Guards against computing a chunk it
        // does not own or skipping one it does.
        if (evaluation.partial_mlp.size() != TOKEN_GROUPS ||
            evaluation.residual1.size() != TOKEN_GROUPS ||
            evaluation.down_projection_chunk_matmuls !=
                TOKEN_GROUPS * options.chunks.size()) {
            throw std::runtime_error(
                "Stage-2 MLP shard-reader declared work count mismatch");
        }

        // Serialize the 13 per-group PARTIAL down-projection sums for the
        // merge step. The primary worker (owns chunk 0) additionally
        // serializes the 13 residual1 ciphertexts the merge needs to form
        // block_output = residual1 + (partial_lo + partial_hi).
        const auto serialize_start = Clock::now();
        const std::string tag = options.chunk_tag();
        std::size_t serialized_partials = 0;
        std::size_t serialized_residuals = 0;
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            serialize_ciphertext(
                options.state_dir /
                    ("partial-mlp-chunks" + tag + "-group" +
                     std::to_string(group) + ".ct"),
                evaluation.partial_mlp[group]);
            ++serialized_partials;
        }
        if (options.primary()) {
            for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
                serialize_ciphertext(
                    options.state_dir /
                        ("residual1-group" + std::to_string(group) + ".ct"),
                    evaluation.residual1[group]);
                ++serialized_residuals;
            }
        }
        const double serialize_partials_seconds =
            elapsed_seconds(serialize_start);

        const std::string evidence = make_json(
            options, ring, evaluation, fixture_seconds, deserialize_seconds,
            load_context_gpu_seconds, encryption_seconds, evaluation_seconds,
            serialize_partials_seconds, serialized_partials,
            serialized_residuals, client.round_trips(),
            client.logical_instances(), client.seconds());
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout
            << "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_MLP_READER_DONE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
