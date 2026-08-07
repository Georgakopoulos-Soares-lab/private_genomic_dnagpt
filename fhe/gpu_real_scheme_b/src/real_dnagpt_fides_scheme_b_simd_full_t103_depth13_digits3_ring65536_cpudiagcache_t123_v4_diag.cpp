// Additive depth-13/digits-3/ring-65536 complete Token-SIMD block-0 gate
// for Scheme B, PLUS a CPU-side diagonal-vector cache.
//
// Scope:
//   input embeddings -> LN1 -> Q/K/V -> exact causal attention ->
//   attention projection -> residual -> LN2 -> GELU/MLP -> block output,
//   for the full T=103 fixture, packing B=8 tokens in each ciphertext.
//
// Its complete schedule is frozen by simd_t103_schedule.hpp and proved by
// test_simd_layout.py before this FIDESlib source is allowed to run.
//
// Fork discipline:
//   direct parent: depth-13/digits-3/ring-65536 (the passing, currently
//   best, byte-for-byte unmodified Token-SIMD source). This fork changes
//   ONLY the CPU-side construction of each BSGS diagonal's
//   std::vector<double> inside matmul(): instead of rebuilding it from
//   scratch on every one of the 156 matmul() calls (13 token groups x 12
//   distinct D x D weight matrices), it is built once per distinct weight
//   matrix (keyed by the weight vector's stable address) and the cached
//   std::vector<double> is reused on the other 12 calls that share that
//   weight. A FRESH Plaintext object is still constructed via
//   raw_plain()/MakeCKKSPackedPlaintext on every single call -- no
//   GPU-resident Plaintext or Ciphertext object is ever reused across
//   multPt calls. MULT_DEPTH/RING_DIM/LARGE_DIGITS/packing/BSGS/TOL are
//   all unchanged from the parent; every existing op-count invariant
//   (matmul/ct-ct/ct-pt/rotation/round-trip counts) is identical to the
//   parent's, because caching the CPU vector changes nothing about which
//   or how many encrypted operations run.
//
//   Reason this is reopened, not a repeat of the abandoned attempt: a
//   2026-07-27/28 "diagonal-plaintext-cache" fix
//   (real_dnagpt_fides_scheme_b_diagcache.cpp, forked from the much older,
//   non-Token-SIMD, T=2 real_dnagpt_fides_scheme_b.cpp) cached the
//   Plaintext OBJECT itself (reusing its GPU-resident identity across two
//   multPt calls) and crashed 6/6 real-GPU attempts, root-caused via
//   addr2line to Ciphertext::multPt -> adjustPlaintextToCiphertext ->
//   RNSPoly::grow -> GPUmalloc -- specifically triggered by reusing a
//   GPU-resident Plaintext across multPt calls (see docs/hybrid/tasks.md,
//   "diagcache" section). This fork structurally cannot hit that mechanism:
//   the cache here holds only a host-side std::vector<double> (never a
//   Plaintext/Ciphertext), and a brand-new Plaintext object (never
//   previously passed to multPt) is constructed via MakeCKKSPackedPlaintext
//   on every call, exactly as the unmodified parent does.
//
//   Two clean-timing repeats of the parent (depth13) landed
//   2026-07-31 2.1-2.9x SLOWER than the original sample despite LIGHTER
//   GPU-memory contamination, with the process pegged at 3900-4700% CPU
//   under host load average 500-1042 -- pointing at host-wide CPU
//   contention from this process's own plaintext-encoding work, not GPU
//   memory or compute, as the bottleneck. This fix targets exactly that
//   CPU-side redundant work (12/13 of the diagonal-vector reconstructions
//   across the 13 token groups are byte-for-byte redundant -- proved
//   against the real T=103 fixture weights in
//   test_diagonal_cache_reuse.py before this source was written).
//
//   parent SHA-256 (the depth-13 source, unmodified, computed fresh -- not
//   trusted from any stale value):
//     6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e
//   full-block semantic anchor: passing T=103 chunked-attention source
//   anchor SHA-256 (unchanged from parent):
//     70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f
//
// Physical slot mapping:
//   ((copy * PACK_WIDTH + feature) * TOKEN_BATCH) + token_lane
//
// Therefore every frozen logical rotation k is scaled to TOKEN_BATCH*k and
// cannot mix token lanes.  Public diagonal coefficients are repeated over the
// TOKEN_BATCH innermost lanes.

#include <fideslib.hpp>
// v3: raw OpenFHE header, used ONLY to call ClearEvalMultKeys()/
// ClearEvalAutomorphismKeys() right after LoadContext (see main()). FIDESlib's
// public wrapper hides lbcrypto:: types entirely (this->cpu is a std::any),
// so freeing OpenFHE's internal CPU-side key-map cache needs the bare OpenFHE
// static API directly -- not a FIDESlib patch. Link-time symbols are already
// satisfied transitively via fideslib::fideslib (FIDESlib's own LoadContext()
// calls the very same key-map accessors internally). Source-verified
// (kimon/logs/optimization_campaign/PLAN.md, v3 entry): LoadContext is the
// ONLY call site in FIDESlib's non-vendored source that reads these maps;
// every GPU-path EvalMult/EvalRotate/Decrypt/Encrypt afterward operates
// purely on the already-loaded GPU-resident context. Validated empirically
// by the micro-gate's TEST_I (job 3345399: post-clear ct*ct/ct*pt/EvalRotate/
// fresh-roundtrip all correct, maxdiff ~1e-9-1e-10) BEFORE being wired in
// here.
#include <openfhe.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
// kimon: the parent anchor is the depth13 source, which the platform build
// edits additively; make its pin compile-time-overridable (bare-hex -D,
// stringified) so a platform build accepts the edited depth13 sha.
#ifndef KIMON_PIN_STRINGIFY
#define KIMON_PIN_STRINGIFY2(x) #x
#define KIMON_PIN_STRINGIFY(x) KIMON_PIN_STRINGIFY2(x)
#endif
#ifdef KIMON_PIN_CPUDIAG_PARENT
constexpr std::string_view PINNED_PARENT_SOURCE_SHA256 =
    KIMON_PIN_STRINGIFY(KIMON_PIN_CPUDIAG_PARENT);
#else
constexpr std::string_view PINNED_PARENT_SOURCE_SHA256 =
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e";
#endif
constexpr std::string_view PINNED_FROZEN_SOURCE_SHA256 =
    "70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f";
constexpr std::string_view PINNED_SCHEDULE_SHA256 =
    "c6b221f365ba6326f615c5554458d7bd092990d23c4ba0d5106ca7577cb7c3aa";
// kimon: optional compile-time override of the fixture-identity pins for a
// platform-tagged (e.g. TACC ls6) build; frozen value is the default. Pass a
// bare-hex -D (no quotes); stringify makes it a literal. See kimon/env/.
#ifndef KIMON_PIN_STRINGIFY
#define KIMON_PIN_STRINGIFY2(x) #x
#define KIMON_PIN_STRINGIFY(x) KIMON_PIN_STRINGIFY2(x)
#endif
#ifdef KIMON_PIN_MANIFEST
constexpr std::string_view PINNED_FIXTURE_MANIFEST = KIMON_PIN_STRINGIFY(KIMON_PIN_MANIFEST);
#else
constexpr std::string_view PINNED_FIXTURE_MANIFEST =
    "d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6";
#endif
#ifdef KIMON_PIN_CONTRACT
constexpr std::string_view PINNED_FIXTURE_CONTRACT_SHA256 = KIMON_PIN_STRINGIFY(KIMON_PIN_CONTRACT);
#else
constexpr std::string_view PINNED_FIXTURE_CONTRACT_SHA256 =
    "061d53bd25bbcaf75d4c12067ec77f032c3ba0e35300a0a6168c2ce15f3672db";
#endif

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
    std::filesystem::path fixture_dir;
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string parent_source_sha256;
    std::string frozen_source_sha256;
    std::string schedule_sha256;
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256;
    std::string source_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_simd_full_t103 "
        "--gpu N "
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
    if (options.gpu < 0 || options.fixture_dir.empty() ||
        options.output.empty()) {
        usage_error("gpu, fixture-dir, and output are required");
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

int canonical_rotation(std::int64_t index) {
    std::int64_t result = index % static_cast<std::int64_t>(SLOTS);
    if (result < 0) {
        result += static_cast<std::int64_t>(SLOTS);
    }
    if (result > static_cast<std::int64_t>(SLOTS / 2)) {
        result -= static_cast<std::int64_t>(SLOTS);
    }
    return static_cast<int>(result);
}

void add_accumulate_keys(std::set<int>& keys, int count, int stride) {
    constexpr int base_step = 4;
    for (std::int64_t step = 1; step < count; step *= base_step) {
        for (int multiple = 1;
             multiple < base_step && step * multiple < count; ++multiple) {
            keys.insert(canonical_rotation(
                static_cast<std::int64_t>(stride) * step * multiple));
        }
    }
}

std::vector<int> required_rotation_keys() {
    std::set<int> indices;
    for (std::size_t baby = 1; baby < BSGS_N1; ++baby) {
        indices.insert(canonical_rotation(
            static_cast<std::int64_t>(TOKEN_BATCH * baby)));
    }
    for (std::size_t giant = 1; giant < BSGS_N2; ++giant) {
        indices.insert(canonical_rotation(static_cast<std::int64_t>(
            TOKEN_BATCH * BSGS_N1 * giant)));
    }
    add_accumulate_keys(indices, static_cast<int>(PACK_WIDTH),
                        static_cast<int>(TOKEN_BATCH));
    for (std::size_t delta = 1; delta < TOKEN_BATCH; ++delta) {
        indices.insert(canonical_rotation(static_cast<std::int64_t>(delta)));
        indices.insert(canonical_rotation(
            static_cast<std::int64_t>(delta) -
            static_cast<std::int64_t>(TOKEN_BATCH)));
    }
    indices.insert(canonical_rotation(
        static_cast<std::int64_t>(PACK_WIDTH * TOKEN_BATCH)));
    indices.insert(canonical_rotation(
        -static_cast<std::int64_t>(PACK_WIDTH * TOKEN_BATCH)));
    indices.insert(canonical_rotation(
        2 * static_cast<std::int64_t>(PACK_WIDTH * TOKEN_BATCH)));
    indices.erase(0);
    return {indices.begin(), indices.end()};
}

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
    std::vector<Ct> packed_output;
    std::size_t matrix_products = 0;
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
        const std::vector<std::size_t>& active_tokens) {
        if (encrypted_inputs.size() != TOKEN_GROUPS ||
            active_tokens.size() != TOKEN_GROUPS) {
            throw std::invalid_argument(
                "Token-SIMD input/active-count mismatch");
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
        // T123 stage flush: the QKV weights (qkv[0..2]) are not used again;
        // free their encoded templates before the MLP stage builds its 8.
        flush_diagonal_plains();

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
        // T123 stage flush: the attention-projection weight is done; free its
        // templates so the MLP stage (mlp_fc[0..3]+mlp_projection[0..3]) is the
        // only weight set resident during its group loop.
        flush_diagonal_plains();

        std::vector<Ct> block_output(TOKEN_GROUPS);
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            Ct normalized2 = layernorm(
                residual1[group], active_tokens[group], "ln2");
            require_remaining_depth(
                normalized2->GetLevel(), 4,
                "Token-SIMD MLP group " + std::to_string(group));
            const BabyRotations fc_baby =
                baby_rotations(normalized2);
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

            Ct mlp;
            for (std::size_t chunk = 0; chunk < COPIES; ++chunk) {
                Ct contribution =
                    matmul(baby_rotations(hidden[chunk]),
                           fixture_.mlp_projection[chunk]);
                mlp = chunk == 0
                          ? contribution
                          : cc_->EvalAdd(mlp, contribution);
            }
            block_output[group] =
                cc_->EvalAdd(residual1[group], mlp);
            std::cout << "[stage] block output group=" << group
                      << " level=" << block_output[group]->GetLevel()
                      << '\n';
        }

        EvaluationResult result;
        result.packed_output = std::move(block_output);
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
        for (const Ct& output : result.packed_output) {
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

    // ---- T123 Tier 1: encode-once + per-use clone -------------------------
    // The retained cpudiagcache reuses the diagonal vector<double> but still
    // CKKS-encodes a fresh Plaintext (the ~41 ms iDFT) on EVERY multiply
    // (159,744 matmul PC-mults, ~90% redundant). Here each distinct diagonal
    // is encoded ONCE into a pristine, UNLOADED template; each multiply gets a
    // lightweight clone that shares the cached CPU encoding (a std::any /
    // shared_ptr copy, no re-encode) with fresh device state. This is SAFE
    // where the abandoned object-reuse path crashed: FIDESlib's
    // LoadPlaintext() returns early when pt->loaded, so reusing a once-loaded
    // Plaintext across ciphertext levels used stale GPU limbs; a fresh
    // unloaded clone is (re)loaded at the correct level, and ~PlaintextImpl
    // only evicts when (loaded && gpu) -> identical per-op lifecycle, no leak.
    // (Validated by the Stage-2 micro-gate kimon/configs/microgate.sbatch.)
    Plaintext clone_encoded(const Plaintext& tmpl) {
        Plaintext c = std::make_shared<PlaintextImpl>();
        c->cpu = tmpl->cpu;                       // share encoded data (no encode)
        c->parent_context = tmpl->parent_context; // needed for dtor eviction
        c->loaded = false;                        // fresh device load per use
        c->gpu = 0;
        return c;
    }

    // Tier 1 + Tier 2/3: return the full set of encoded diagonal templates for
    // a weight, building + CKKS-encoding all diagonals ONCE on first use. The
    // encode loop is parallelised across cores (OMP_NUM_THREADS); each index is
    // written independently (no data race in this code -- library encode
    // thread-safety is checked by the micro-gate's TEST_D).
    // Evict all cached encoded templates. Called at STAGE boundaries in
    // evaluate() so the resident set is bounded to one stage's weights: the
    // weights interleave group-major within a stage (so all of a stage's
    // weights must stay resident across its 13 token groups to get reuse), but
    // no weight recurs across stages -> flushing between stages bounds peak
    // memory to the largest stage (MLP, 8 weights) instead of all 12 at once.
    void flush_diagonal_plains() { diagonal_plain_cache_.clear(); }

    // Tier 1 + Tier 2/3, memory-bounded. Templates are cached per (weight,
    // level) and ENCODED AT THE USE-LEVEL (fewer RNS limbs than level 0 ->
    // ~2x smaller), so the largest stage's template set fits this node's
    // ~65 GiB. Encoding a plaintext at the level it is multiplied is exact
    // (OpenFHE would otherwise drop a level-0 plaintext to this level anyway),
    // and is verified by the block oracle. One lookup per matmul() call ->
    // 12 misses (one per distinct weight, each used at a single level within
    // its stage) and 156-12 = 144 hits across the 13 token groups.
    const std::vector<Plaintext>& cached_diagonal_plains(
        const std::vector<double>& weight, std::uint32_t level) {
        const auto key = std::make_pair(&weight, level);
        auto found = diagonal_plain_cache_.find(key);
        if (found != diagonal_plain_cache_.end()) {
            ++diagonal_cache_hits_;
            return found->second;
        }
        ++diagonal_cache_misses_;
        // THROWAWAY DIAGNOSTIC (v4 feasibility check only, not part of any
        // lasting optimization): misses happen in a fixed, deterministic
        // order -- 1=qkv[0] 2=qkv[1] 3=qkv[2] 4=attn_proj 5-8=mlp_fc[0..3]
        // 9-12=mlp_projection[0..3] -- so (miss_number, level) unambiguously
        // identifies each of the 12 weights' actual encode level.
        std::cerr << "[diag] miss=" << diagonal_cache_misses_
                  << " level=" << level << " ndiag=" << (BSGS_N1 * BSGS_N2)
                  << "\n" << std::flush;
        const std::size_t ndiag = BSGS_N1 * BSGS_N2;
        // 1) build every packed diagonal vector (same math as the retained
        //    cached_packed_values, just materialised for the whole weight).
        std::vector<std::vector<double>> vecs(ndiag);
        for (std::size_t build_giant = 0; build_giant < BSGS_N2;
             ++build_giant) {
            for (std::size_t build_small = 0; build_small < BSGS_N1;
                 ++build_small) {
                const std::size_t diagonal =
                    BSGS_N1 * build_giant + build_small;
                std::vector<double> packed_values(SLOTS, 0.0);
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH - BSGS_N1 * build_giant) %
                            PACK_WIDTH;
                        const std::size_t column =
                            (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            const double coefficient =
                                weight[rolled_row * D + column];
                            for (std::size_t token = 0; token < TOKEN_BATCH;
                                 ++token) {
                                packed_values[physical_slot(copy, row,
                                                            token)] =
                                    coefficient;
                            }
                        }
                    }
                }
                vecs[diagonal] = std::move(packed_values);
            }
        }
        // 2) encode all templates ONCE, at the use-level, in parallel.
        std::vector<Plaintext> plains(ndiag);
        const auto ndiag_signed = static_cast<std::ptrdiff_t>(ndiag);
#pragma omp parallel for schedule(dynamic)
        for (std::ptrdiff_t idx = 0; idx < ndiag_signed; ++idx) {
            plains[static_cast<std::size_t>(idx)] =
                cc_->MakeCKKSPackedPlaintext(
                    vecs[static_cast<std::size_t>(idx)], 1, level, nullptr,
                    SLOTS);
        }
        found = diagonal_plain_cache_.emplace(key, std::move(plains)).first;
        return found->second;
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
        // T123: encode-once (all diagonals of this weight, parallel, at the
        // multiply's level) then clone per multiply -- no per-op re-encode.
        const std::uint32_t use_level = baby.values.at(0)->GetLevel();
        const std::vector<Plaintext>& diagonal_plains =
            cached_diagonal_plains(weight, use_level);
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                Plaintext diagonal_plain = clone_encoded(
                    diagonal_plains[BSGS_N1 * giant + small]);
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
    // T123 Tier 1: cache of ENCODED diagonal templates, keyed by (weight ptr,
    // use-level), flushed between stages (flush_diagonal_plains).
    std::map<std::pair<const std::vector<double>*, std::uint32_t>,
             std::vector<Plaintext>>
        diagonal_plain_cache_;
};

struct Metrics {
    double global_rel_inf = 0.0;
    double worst_token_rel_inf = 0.0;
    double max_abs_error = 0.0;
    bool all_finite = true;
    bool passed = false;
};

Metrics measure_output(const std::vector<double>& got,
                       const std::vector<double>& oracle) {
    if (got.size() != T * D || oracle.size() != T * D) {
        throw std::invalid_argument("block-output metric size mismatch");
    }
    Metrics metrics;
    double global_denominator = 0.0;
    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t dim = 0; dim < D; ++dim) {
            const std::size_t index = token * D + dim;
            const double actual = got[index];
            const double expected = oracle[index];
            metrics.all_finite =
                metrics.all_finite && std::isfinite(actual);
            const double error = std::abs(actual - expected);
            metrics.max_abs_error =
                std::max(metrics.max_abs_error, error);
            token_error = std::max(token_error, error);
            token_denominator =
                std::max(token_denominator, std::abs(expected));
            global_denominator =
                std::max(global_denominator, std::abs(expected));
        }
        metrics.worst_token_rel_inf =
            std::max(metrics.worst_token_rel_inf,
                     token_error / (token_denominator + 1e-15));
    }
    metrics.global_rel_inf =
        metrics.max_abs_error / (global_denominator + 1e-15);
    metrics.passed =
        metrics.all_finite && metrics.global_rel_inf <= TOL &&
        metrics.worst_token_rel_inf <= TOL;
    return metrics;
}

std::string make_json(
    const Options& options, std::uint32_t ring,
    std::size_t rotation_keys, const EvaluationResult& evaluation,
    const Metrics& metrics, double fixture_seconds, double setup_seconds,
    double encryption_seconds, double evaluation_seconds,
    double boundary_seconds, double decrypt_seconds,
    std::size_t round_trips, std::size_t logical_instances) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"Scheme B Token-SIMD T=103 "
           "depth-13/digits-3/ring-65536 complete "
           "released-weight DNAGPT block-0 gate, T123 = CPU-side "
           "diagonal-vector cache + Tier1 encode-once encoded-Plaintext "
           "templates with per-use clone + Tier2/3 multi-core "
           "(OMP) parallel batched encode; v3 = free the CPU-side "
           "(OpenFHE) EvalMult/rotation key maps right after LoadContext, "
           "same crypto-op schedule/oracle\",\n"
        << "  \"implementation_version\": "
           "\"t103-scheme-b-token-simd-full-depth13-digits3-ring65536-"
           "cpudiagcache-t123-v3\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"gate\": \"full\",\n"
        << "  \"mode\": \"packed\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \""
        << json_escape(options.backend_commit) << "\",\n"
        << "  \"container_image\": \""
        << json_escape(options.container_image) << "\",\n"
        << "  \"environment\": \""
        << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \""
        << json_escape(options.source_sha256) << "\",\n"
        << "  \"parent_source_sha256\": \""
        << json_escape(options.parent_source_sha256) << "\",\n"
        << "  \"frozen_source_sha256\": \""
        << json_escape(options.frozen_source_sha256) << "\",\n"
        << "  \"schedule_sha256\": \""
        << json_escape(options.schedule_sha256) << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"fixture_contract_sha256\": \""
        << json_escape(options.fixture_contract_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": " << D << ", \"T\": " << T
        << ", \"token_batch\": " << TOKEN_BATCH
        << ", \"token_groups\": "
        << evaluation.packed_output.size() << ",\n"
        << "    \"pack_width\": " << PACK_WIDTH
        << ", \"copies\": " << COPIES << ", \"batch_slots\": "
        << SLOTS << ", \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH
        << ", \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"matrix_products\": " << evaluation.matrix_products
        << ", \"frozen_equivalent_matrix_products\": " << 12 * T << ",\n"
        << "    \"dense_call_reduction\": "
        << static_cast<double>(12 * T) /
               static_cast<double>(evaluation.matrix_products)
        << ",\n"
        << "    \"packed_output_level\": "
        << evaluation.packed_output_level << ",\n"
        << "    \"explicit_ciphertext_ciphertext_multiplications\": "
        << evaluation.ct_ct_multiplications << ",\n"
        << "    \"explicit_ciphertext_plaintext_multiplications\": "
        << evaluation.ct_plain_multiplications << ",\n"
        << "    \"explicit_rotations\": "
        << evaluation.explicit_rotations << ",\n"
        << "    \"accumulate_sum_calls\": "
        << evaluation.accumulate_sum_calls << ",\n"
        << "    \"score_tile_decryptions\": "
        << evaluation.score_tile_decryptions << ",\n"
        << "    \"weight_tile_encryptions\": "
        << evaluation.weight_tile_encryptions << ",\n"
        << "    \"cached_key_lane_shifts\": "
        << evaluation.cached_key_lane_shifts << ",\n"
        << "    \"cached_value_lane_shifts\": "
        << evaluation.cached_value_lane_shifts << ",\n"
        << "    \"diagonal_cache_hits\": "
        << evaluation.diagonal_cache_hits << ",\n"
        << "    \"diagonal_cache_misses\": "
        << evaluation.diagonal_cache_misses << "\n"
        << "  },\n"
        << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << "  \"worst_token_rel_inf\": "
        << metrics.worst_token_rel_inf << ",\n"
        << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << "  \"all_finite\": "
        << (metrics.all_finite ? "true" : "false") << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": "
        << (metrics.passed ? "true" : "false") << ",\n"
        << "  \"protocol\": {\"round_trips\": " << round_trips
        << ", \"logical_boundary_instances\": " << logical_instances
        << ", \"boundary_seconds_total\": " << boundary_seconds
        << "},\n"
        << "  \"timings_seconds\": {\n"
        << "    \"load_verified_fixture\": " << fixture_seconds << ",\n"
        << "    \"context_keygen_load\": " << setup_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds
        << ",\n"
        << "    \"encrypted_evaluation\": " << evaluation_seconds << ",\n"
        << "    \"client_boundary_seconds_total\": "
        << boundary_seconds << ",\n"
        << "    \"server_linear_algebra_seconds\": "
        << (evaluation_seconds - boundary_seconds) << ",\n"
        << "    \"final_decrypt\": " << decrypt_seconds << "\n"
        << "  },\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"declared_client_boundary_crossings\": "
        << round_trips << ",\n"
        << "  \"declared_logical_boundary_instances\": "
        << logical_instances << ",\n"
        << "  \"final_decrypt_calls\": "
        << evaluation.packed_output.size() << ",\n"
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        const auto fixture_start = Clock::now();
        const Fixture fixture = load_fixture(options.fixture_dir);
        const double fixture_seconds = elapsed_seconds(fixture_start);

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
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        cc->EvalRotateKeyGen(keys.secretKey, rotation_keys);
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        // v3: LoadContext already copied the relin key + all rotation keys
        // to the GPU (AddEvalKey/AddRotationKey). The CPU-side (OpenFHE)
        // copies in the static eval-mult-key/rotation-key maps are dead
        // weight for the rest of the run -- nothing in the GPU op path
        // (EvalMult/EvalRotate/Decrypt/Encrypt) reads them again. This
        // process holds exactly one crypto context, so the global
        // no-argument overload is equivalent to a scoped clear. "How", not
        // "what": the crypto-op schedule and oracle are unaffected; only
        // host RAM is freed earlier than the process-exit destructor would.
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalMultKeys();
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalAutomorphismKeys();
        const double setup_seconds = elapsed_seconds(setup_start);
        const std::uint32_t ring = cc->GetRingDimension();
        if (ring < 2 * SLOTS) {
            throw std::runtime_error(
                "ring dimension cannot provide requested Token-SIMD slots");
        }
        std::cout << "[context] ring=" << ring << " slots=" << SLOTS
                  << " token_batch=" << TOKEN_BATCH
                  << " rotation_keys=" << rotation_keys.size() << '\n';

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
        EvaluationResult evaluation =
            evaluator.evaluate(encrypted_inputs, active_counts);
        cc->Synchronize();
        const double evaluation_seconds =
            elapsed_seconds(evaluation_start);

        const auto decrypt_start = Clock::now();
        std::vector<double> output(T * D, 0.0);
        for (std::size_t group = 0;
             group < evaluation.packed_output.size(); ++group) {
            Plaintext decoded;
            cc->Decrypt(keys.secretKey, evaluation.packed_output[group],
                        &decoded);
            decoded->SetLength(SLOTS);
            const std::vector<double> raw =
                decoded->GetRealPackedValue();
            const std::size_t first = group * TOKEN_BATCH;
            const std::size_t active = active_tokens(group);
            for (std::size_t token_lane = 0; token_lane < active;
                 ++token_lane) {
                for (std::size_t dim = 0; dim < D; ++dim) {
                    output[(first + token_lane) * D + dim] =
                        raw[physical_slot(0, dim, token_lane)];
                }
            }
        }
        const double decrypt_seconds =
            elapsed_seconds(decrypt_start);

        constexpr std::size_t EXPECTED_MATRIX_PRODUCTS = 156;
        constexpr std::size_t EXPECTED_CT_CT = 1506;
        constexpr std::size_t EXPECTED_CT_PLAIN = 177734;
        constexpr std::size_t EXPECTED_ROTATIONS = 8173;
        constexpr std::size_t EXPECTED_ACCUMULATE_SUM = 8776;
        constexpr std::size_t EXPECTED_SCORE_TILES = 91;
        constexpr std::size_t EXPECTED_WEIGHT_TILES = 727;
        constexpr std::size_t EXPECTED_CACHED_SHIFTS = 90;
        constexpr std::size_t EXPECTED_ROUND_TRIPS = 857;
        constexpr std::size_t EXPECTED_LOGICAL_INSTANCES = 129162;
        // The cache is keyed by weight-matrix address; there are exactly
        // 12 distinct weight matrices touched across the whole evaluate()
        // call (fixture_.qkv[0..2], attention_projection, mlp_fc[0..3],
        // mlp_projection[0..3]), so exactly 12 misses (one per weight, on
        // its first-ever cached_packed_values() call) and
        // 156 * BSGS_N1 * BSGS_N2 - 12 hits (every other lookup, across
        // all 13 token groups for every weight). A mismatch here would
        // mean the cache is not actually being reused as designed.
        // T123 accounting: the encoded-template cache is looked up ONCE per
        // matmul() call (not once per diagonal as the retained cpudiagcache
        // did). 12 distinct (weight, use-level) keys are built exactly once
        // each -> 12 misses; the other EXPECTED_MATRIX_PRODUCTS - 12 matmul
        // calls reuse a cached weight across the 13 token groups -> 144 hits.
        // The crypto-op invariants below (ct_plain, rotations, etc.) are
        // unchanged from the retained schedule -- only where the plaintext
        // comes from changed, not how many operations run.
        constexpr std::size_t EXPECTED_DIAGONAL_CACHE_MISSES = 12;
        constexpr std::size_t EXPECTED_DIAGONAL_CACHE_HITS =
            EXPECTED_MATRIX_PRODUCTS - EXPECTED_DIAGONAL_CACHE_MISSES;
        if (evaluation.packed_output.size() != TOKEN_GROUPS ||
            evaluation.matrix_products != EXPECTED_MATRIX_PRODUCTS ||
            evaluation.ct_ct_multiplications != EXPECTED_CT_CT ||
            evaluation.ct_plain_multiplications != EXPECTED_CT_PLAIN ||
            evaluation.explicit_rotations != EXPECTED_ROTATIONS ||
            evaluation.accumulate_sum_calls != EXPECTED_ACCUMULATE_SUM ||
            evaluation.score_tile_decryptions != EXPECTED_SCORE_TILES ||
            evaluation.weight_tile_encryptions != EXPECTED_WEIGHT_TILES ||
            evaluation.cached_key_lane_shifts != EXPECTED_CACHED_SHIFTS ||
            evaluation.cached_value_lane_shifts != EXPECTED_CACHED_SHIFTS ||
            evaluation.diagonal_cache_misses !=
                EXPECTED_DIAGONAL_CACHE_MISSES ||
            evaluation.diagonal_cache_hits != EXPECTED_DIAGONAL_CACHE_HITS ||
            client.round_trips() != EXPECTED_ROUND_TRIPS ||
            client.logical_instances() != EXPECTED_LOGICAL_INSTANCES) {
            throw std::runtime_error(
                "Token-SIMD complete-block schedule count mismatch");
        }
        const Metrics metrics =
            measure_output(output, fixture.oracle_block_output);
        const std::string evidence = make_json(
            options, ring, rotation_keys.size(), evaluation, metrics,
            fixture_seconds, setup_seconds, encryption_seconds,
            evaluation_seconds, client.seconds(), decrypt_seconds,
            client.round_trips(), client.logical_instances());
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout << (metrics.passed
                          ? "REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_CPUDIAGCACHE_T123_V3_GATE_PASS\n"
                          : "REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_CPUDIAGCACHE_GATE_FAIL\n");
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
