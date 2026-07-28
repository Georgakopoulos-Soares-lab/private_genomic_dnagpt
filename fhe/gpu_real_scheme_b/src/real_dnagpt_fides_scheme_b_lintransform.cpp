// Scheme B (hybrid client-assisted CKKS) GPU-side LINEARTRANSFORM prototype,
// forked from real_dnagpt_fides_scheme_b_profiled.cpp (itself forked from
// the frozen, hash-pinned single-shot gate). Do not edit either ancestor
// file to add this behavior; their SHA-256 hashes are recorded as
// provenance in existing immutable evidence JSONs.
//
// Purpose: docs/hybrid/tasks.md's 2026-07-27 profiling session found the ct-pt
// multiply-and-accumulate inner loop is ~100% of one matmul() call's internal
// cost (rotation/keyswitch <0.1%, not the bottleneck at any BSGS split). This
// binary replaces that manual per-diagonal loop, for exactly ONE of the six
// textual matmul() call sites (the QKV query projection, both T=2 tokens),
// with FIDESlib's own native batched BSGS primitive,
// FIDESlib::CKKS::LinearTransform (src/CKKS/LinearTransform.cuh/.cu in the
// pinned FIDESlib commit), which dispatches to a real GPU-batched kernel
// (RNSPoly::LTdotProductPtBatch) instead of 1024 sequential EvalMult+EvalAdd
// calls. The other 23 call sites are UNCHANGED, on the original manual path,
// so the full-block global_rel_inf oracle comparison at the end isolates
// whether this one call site's translation is correct: any bug there
// propagates downstream through every dependent stage and would blow the
// unchanged 4e-2 tolerance against a baseline that has measured
// global_rel_inf ~1e-10 for years of prior evidence.
//
// See docs/hybrid/tasks.md's 2026-07-28 "LinearTransform scoping" section for
// the full compatibility analysis this implementation follows:
//   - our matmul() operates entirely through the OpenFHE-API-compatible
//     wrapper (api/CryptoContext.cpp), which never calls LinearTransform;
//     integrating it means reaching into native FIDESlib::CKKS types
//     ourselves via the wrapper's already-public LoadCiphertext/
//     LoadPlaintext/GetDeviceCiphertext/GetDevicePlaintext, exactly the
//     pattern api/CryptoContext.cpp's own ConvolutionTransformInPlace uses;
//   - LinearTransform's diagonal indexing (Aptr[bStep*j+i]) and hoisted
//     baby-step rotation indices match our own bsgs_inner_loop exactly, so
//     the existing packed_values() diagonal-value computation is reused
//     unchanged;
//   - LinearTransform asserts its diagonal plaintexts are pre-encoded at the
//     ciphertext's EXACT current level (no implicit adjustment, unlike
//     multPt's adjustPlaintextToCiphertext) -- raw_plain() below gains an
//     explicit `level` parameter for this, and an explicit runtime check
//     (not the library's own assert, which -DCMAKE_BUILD_TYPE=Release
//     compiles out via NDEBUG) verifies it before every call;
//   - LinearTransform mutates its ciphertext argument in place, so the QKV
//     query call site works on an independent native clone, never the
//     shared `baby.values[0]`/`normalized[token]` object that the sibling
//     key/value matmul calls still depend on.
//
// Instrumentation strategy (deliberately NOT one Synchronize() per primitive
// op -- 24666 ct-pt multiplications alone would need >24000 barriers,
// dominating the measurement with sync overhead instead of revealing it):
//
//   1. Every one of the 24 matmul() call sites is bracketed with one
//      Synchronize()-before/after pair and attributed to its graph-stage
//      label (qkv_query/qkv_key/qkv_value/attention_projection/mlp_fc/
//      mlp_projection) -- 24 barriers total, revealing which STAGE of the
//      block dominates.
//   2. Exactly ONE representative matmul() call (the first: token 0's Q
//      projection) additionally instruments every one of its BSGS_N2=32
//      giant-step iterations internally, splitting each into: the N1=32
//      ciphertext-plaintext-multiply-and-accumulate inner loop, the single
//      giant EvalRotate (skipped at giant=0), and the outer EvalAdd
//      accumulate -- giving a real, GPU-measured proportional breakdown of
//      one matmul call's internal cost that is assumed structurally
//      representative of the other 23 (same shapes, same op sequence, only
//      the constant diagonal weight values differ). This assumption is
//      checked, not just claimed: the detailed call's own total (sum of its
//      three sub-buckets) is compared against its own coarse bracket from
//      point 1 -- if the two disagree by more than the sync-overhead floor,
//      that disagreement itself is reported, not hidden.
//   3. LayerNorm (4 calls), the 12-head attention score/delta loop (1
//      bracket per block), the post-sigmoid context blend, and the GELU
//      pack/select/restore work around each client boundary are each given
//      one coarse bracket -- these are known small contributors (the client
//      boundary itself is already timed by the unmodified Client class) but
//      are still recorded so the reconciliation in point 4 has nothing
//      unlabeled.
//   4. The grand total of every bucket plus the (unmodified) client boundary
//      time is compared against the overall Synchronize()-bracketed
//      encrypted_evaluation_seconds for the whole gate. The residual
//      (measured total minus sum of attributed buckets) is reported
//      explicitly as "unaccounted_seconds" -- host-side dispatch, vector
//      construction, plaintext encoding not isolated by a barrier, and the
//      profiling barriers' own overhead. This number is not hidden or
//      minimized; if it is large, that is itself a finding.
//   5. Gaps between the three QKV matmul calls for token 0 (which read the
//      same input and are algebraically independent -- Q, K, V do not
//      depend on each other) are measured directly: wall-clock timestamp at
//      the end of one bracket vs the start of the next. If these gaps are
//      ~0, the calls already run back-to-back with no idle bubble on this
//      single CUDA stream, meaning cross-stream pipelining would have
//      nothing to reclaim; if they are not ~0, that is the concrete
//      evidence a pipelining fix could target.
//
// This binary changes NOTHING about the encrypted computation itself: same
// packing, same BSGS_N1=32/BSGS_N2=32 split (see CLAUDE.md/tasks.md -- BSGS
// retuning was already ruled out as a lever since N1*N2=1024 is fixed by
// PACK_WIDTH regardless of the split), same TOL=4e-2 oracle gate, same
// client-only decrypt boundary. It is a read-only timing instrument bolted
// onto an unmodified evaluation, evaluated once (gate=full, the heaviest
// gate, matching every other post-2026-07-26 Scheme B evidence file).
//
// Security boundary unchanged from the original: EncryptedEvaluator never
// stores a private key; only Client does, and only Client calls
// cross_boundary()/cross_boundary_active(). main() performs exactly one
// final Decrypt, exactly as the original does.

#include <fideslib.hpp>

// Native FIDESlib primitive this prototype calls directly (not exposed by
// the OpenFHE-API-compatible wrapper -- see the header comment above and
// docs/hybrid/tasks.md's 2026-07-28 scoping section). Installed alongside
// <fideslib.hpp> by the pinned FIDESlib build's own install() rule, which
// packages the whole src/ tree's headers for exactly this kind of use.
// LinearTransform.cuh alone only forward-declares FIDESlib::CKKS::Ciphertext/
// Plaintext (via forwardDefs.cuh) -- calling member functions on them (c0,
// getLevel()) needs their full definitions, so both are included explicitly
// too, matching api/CryptoContext.cpp's own include list.
#include <CKKS/Ciphertext.cuh>
#include <CKKS/LinearTransform.cuh>
#include <CKKS/Plaintext.cuh>

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
constexpr std::size_t T = 2;
constexpr std::size_t HEADS = 12;
constexpr std::size_t HEAD_DIM = 64;
constexpr std::size_t MLP_DIM = 3072;
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t SLOTS = COPIES * PACK_WIDTH;
constexpr std::size_t BSGS_N1 = 32;
constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;
constexpr std::uint32_t MULT_DEPTH = 16;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 4;
constexpr double EPS = 1e-5;
constexpr double TOL = 4e-2;

// This prototype always evaluates the heaviest gate: the complete block-0
// forward pass, matching the target of the 384s-server-cost finding this
// session investigates.
constexpr std::string_view GATE = "full";

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
constexpr std::string_view PINNED_FIXTURE_MANIFEST =
    "8d20a2841ee29a7a386171f1bd173b7189144359ce8f91ea5eb21cc60c0a78fe";

using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;
using Cc = CryptoContext<DCRTPoly>;
using Keys = KeyPair<DCRTPoly>;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Exact elementwise nonlinearities -- identical to every other Scheme B gate.
double exact_invsqrt(double value) { return 1.0 / std::sqrt(value); }

double exact_sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }

double exact_gelu(double value) {
    constexpr double two_over_pi = 0.63661977236758134308;
    return 0.5 * value *
           (1.0 + std::tanh(std::sqrt(two_over_pi) *
                            (value + 0.044715 * value * value * value)));
}

std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    if (::gmtime_r(&now, &utc) == nullptr) {
        throw std::runtime_error("could not obtain UTC timestamp");
    }
    std::array<char, 32> buffer{};
    if (std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0) {
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
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
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
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
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
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
    std::string source_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_lintransform --gpu N "
        "--fixture-dir PATH --output PATH --fixture-manifest-sha256 SHA "
        "[--backend-commit SHA] [--container-image NAME] [--environment TEXT] "
        "[--source-sha256 SHA] [--fixture-contract-sha256 SHA]");
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
        } else if (arg == "--fixture-manifest-sha256") {
            options.fixture_manifest_sha256 = next();
        } else if (arg == "--fixture-contract-sha256") {
            options.fixture_contract_sha256 = next();
        } else if (arg == "--container-image") {
            options.container_image = next();
        } else if (arg == "--environment") {
            options.environment = next();
        } else if (arg == "--source-sha256") {
            options.source_sha256 = next();
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: real_dnagpt_fides_scheme_b_lintransform --gpu N "
                         "--fixture-dir PATH --output PATH "
                         "--fixture-manifest-sha256 SHA\n";
            std::exit(0);
        } else {
            usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0) {
        usage_error("--gpu must be non-negative");
    }
    if (options.fixture_dir.empty() || options.output.empty()) {
        usage_error("--fixture-dir and --output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit " + options.backend_commit);
    }
    if (options.fixture_manifest_sha256 != PINNED_FIXTURE_MANIFEST) {
        usage_error("refusing unpinned fixture manifest " +
                    options.fixture_manifest_sha256);
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
                                 options.output.string());
    }
    return options;
}

struct Matrix {
    std::size_t rows{};
    std::size_t cols{};
    std::vector<double> values;

    Matrix() = default;
    Matrix(std::size_t row_count, std::size_t col_count)
        : rows(row_count), cols(col_count), values(row_count * col_count) {}

    double operator()(std::size_t row, std::size_t col) const {
        return values.at(row * cols + col);
    }
};

std::vector<double> read_f64(const std::filesystem::path& path,
                             std::size_t expected_values) {
    const std::uintmax_t expected_bytes = expected_values * sizeof(double);
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("fixture array missing: " + path.string());
    }
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error("fixture array size mismatch for " + path.string() +
                                 ": expected " + std::to_string(expected_bytes) +
                                 ", got " + std::to_string(actual_bytes));
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("could not open fixture array: " + path.string());
    }
    std::vector<double> values(expected_values);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(expected_bytes));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("short or trailing fixture data: " + path.string());
    }
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::runtime_error("non-finite fixture data: " + path.string());
    }
    return values;
}

Matrix load_matrix(const std::filesystem::path& path, std::size_t rows,
                   std::size_t cols) {
    Matrix output(rows, cols);
    output.values = read_f64(path, rows * cols);
    return output;
}

Matrix rows(const Matrix& input, std::size_t first, std::size_t count) {
    if (first + count > input.rows) {
        throw std::invalid_argument("row slice outside matrix");
    }
    Matrix output(count, input.cols);
    for (std::size_t row = 0; row < count; ++row) {
        std::copy_n(input.values.begin() +
                        static_cast<std::ptrdiff_t>((first + row) * input.cols),
                    input.cols,
                    output.values.begin() +
                        static_cast<std::ptrdiff_t>(row * output.cols));
    }
    return output;
}

Matrix columns(const Matrix& input, std::size_t first, std::size_t count) {
    if (first + count > input.cols) {
        throw std::invalid_argument("column slice outside matrix");
    }
    Matrix output(input.rows, count);
    for (std::size_t row = 0; row < input.rows; ++row) {
        for (std::size_t col = 0; col < count; ++col) {
            output.values[row * count + col] =
                input(row, first + col);
        }
    }
    return output;
}

struct Fixture {
    std::vector<double> input;
    std::vector<double> ln1;
    std::array<Matrix, 3> qkv;
    Matrix attention_projection;
    std::vector<double> ln2;
    std::array<Matrix, 4> mlp_fc;
    std::array<Matrix, 4> mlp_projection;
    std::vector<double> oracle_ln1;
    std::vector<double> oracle_attention_projection;
    std::vector<double> oracle_block_output;
};

Fixture load_fixture(const std::filesystem::path& directory) {
    Fixture fixture;
    fixture.input = read_f64(directory / "input__embeddings.bin", T * D);
    fixture.ln1 = read_f64(directory / "weights__ln1.bin", D);
    fixture.ln2 = read_f64(directory / "weights__ln2.bin", D);

    const Matrix qkv =
        load_matrix(directory / "weights__attn_qkv.bin", 3 * D, D);
    for (std::size_t part = 0; part < 3; ++part) {
        fixture.qkv[part] = rows(qkv, part * D, D);
    }
    fixture.attention_projection =
        load_matrix(directory / "weights__attn_proj.bin", D, D);

    const Matrix fc =
        load_matrix(directory / "weights__mlp_fc.bin", MLP_DIM, D);
    for (std::size_t part = 0; part < 4; ++part) {
        fixture.mlp_fc[part] = rows(fc, part * D, D);
    }
    const Matrix projection =
        load_matrix(directory / "weights__mlp_proj.bin", D, MLP_DIM);
    for (std::size_t part = 0; part < 4; ++part) {
        fixture.mlp_projection[part] = columns(projection, part * D, D);
    }

    fixture.oracle_ln1 =
        read_f64(directory / "oracle__ln1_output.bin", T * D);
    fixture.oracle_attention_projection =
        read_f64(directory / "oracle__attention_projection.bin", T * D);
    fixture.oracle_block_output =
        read_f64(directory / "oracle__block_output.bin", T * D);
    return fixture;
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
            keys.insert(
                canonical_rotation(static_cast<std::int64_t>(stride) * step * multiple));
        }
    }
}

// Unchanged from every other Scheme B gate: full's key set is a strict
// superset of attention's, which is a strict superset of ln1's.
std::vector<int> required_rotation_keys(const std::string& gate) {
    std::set<int> indices;
    if (gate != "ln1") {
        for (std::size_t baby = 1; baby < BSGS_N1; ++baby) {
            indices.insert(static_cast<int>(baby));
        }
        for (std::size_t giant = 1; giant < BSGS_N2; ++giant) {
            indices.insert(static_cast<int>(BSGS_N1 * giant));
        }
    }
    add_accumulate_keys(indices, static_cast<int>(PACK_WIDTH), 1);
    if (gate == "full") {
        indices.insert(canonical_rotation(static_cast<std::int64_t>(PACK_WIDTH)));
        indices.insert(canonical_rotation(-static_cast<std::int64_t>(PACK_WIDTH)));
        indices.insert(
            canonical_rotation(2 * static_cast<std::int64_t>(PACK_WIDTH)));
    }
    indices.erase(0);
    return {indices.begin(), indices.end()};
}

// Accumulates named wall-clock buckets. Every bucket entry corresponds to
// one Synchronize()-before/after bracket, so its seconds reflect real GPU
// completion time for whatever ops ran inside that bracket, not just host
// dispatch latency.
class Profiler {
  public:
    struct Bucket {
        std::size_t count = 0;
        double seconds = 0.0;
    };

    void add(const std::string& name, double seconds) {
        Bucket& bucket = buckets_[name];
        bucket.count += 1;
        bucket.seconds += seconds;
    }

    const std::map<std::string, Bucket>& buckets() const { return buckets_; }

    double total_seconds() const {
        double total = 0.0;
        for (const auto& [name, bucket] : buckets_) {
            total += bucket.seconds;
        }
        return total;
    }

  private:
    std::map<std::string, Bucket> buckets_;
};

struct EvaluationResult {
    Ct packed_output;
    std::size_t ciphertext_plain_matmuls = 0;
    std::size_t ciphertext_ciphertext_multiplications = 0;
    std::size_t ciphertext_plaintext_multiplications = 0;
    std::vector<std::pair<std::string, double>> qkv_gap_seconds;
};

struct BoundaryEvent {
    std::string name;
    double seconds = 0.0;
    std::size_t logical_instances = 0;
};

// Client: the only party in this process that holds the secret key. Byte-
// for-byte the same protocol as every other Scheme B gate: pre-declared
// nonlinearity boundaries only, one decrypt/exact-compute/re-encrypt per
// crossing, no state beyond the round-trip/logical-instance counters.
class Client {
  public:
    Client(Cc cc, Keys& keys) : cc_(std::move(cc)), keys_(keys) {}

    Ct cross_boundary(const std::string& name, const Ct& ciphertext,
                      const std::function<double(double)>& transform,
                      std::size_t logical_instances = 1) {
        return cross_boundary_impl(
            name, ciphertext,
            [&transform](std::vector<double>& values) {
                for (double& value : values) {
                    value = transform(value);
                }
            },
            logical_instances);
    }

    Ct cross_boundary_active(const std::string& name, const Ct& ciphertext,
                             const std::function<double(double)>& transform,
                             std::size_t logical_instances) {
        return cross_boundary_impl(
            name, ciphertext,
            [&transform](std::vector<double>& values) {
                std::vector<double> transformed(SLOTS);
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    const std::size_t start = copy * PACK_WIDTH;
                    for (std::size_t dim = 0; dim < D; ++dim) {
                        transformed[start + dim] = transform(values[start + dim]);
                    }
                }
                values = std::move(transformed);
            },
            logical_instances);
    }

    std::size_t round_trips() const { return round_trips_; }
    std::size_t logical_boundary_instances() const {
        return logical_boundary_instances_;
    }
    double boundary_seconds_total() const { return boundary_seconds_total_; }
    const std::vector<BoundaryEvent>& boundary_log() const {
        return boundary_log_;
    }

  private:
    Ct cross_boundary_impl(
        const std::string& name, const Ct& ciphertext,
        const std::function<void(std::vector<double>&)>& transform,
        std::size_t logical_instances) {
        if (logical_instances == 0) {
            throw std::invalid_argument(
                "client boundary must represent at least one logical instance");
        }
        const auto start = Clock::now();
        Ct local = ciphertext;
        Plaintext plaintext;
        cc_->Decrypt(keys_.secretKey, local, &plaintext);
        plaintext->SetLength(SLOTS);
        std::vector<double> values = plaintext->GetRealPackedValue();
        transform(values);
        Plaintext refreshed = cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
        Ct out = cc_->Encrypt(keys_.publicKey, refreshed);
        ++round_trips_;
        logical_boundary_instances_ += logical_instances;
        const double seconds = elapsed_seconds(start);
        boundary_seconds_total_ += seconds;
        boundary_log_.push_back({name, seconds, logical_instances});
        return out;
    }

    Cc cc_;
    Keys& keys_;
    std::size_t round_trips_ = 0;
    std::size_t logical_boundary_instances_ = 0;
    double boundary_seconds_total_ = 0.0;
    std::vector<BoundaryEvent> boundary_log_;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture, Client& client,
                       Profiler& profiler)
        : cc_(std::move(context)), fixture_(fixture), client_(client),
          profiler_(profiler) {}

    EvaluationResult evaluate(const std::array<Ct, T>& encrypted_inputs,
                              const std::string& gate) {
        matmul_count_ = 0;
        ct_ct_multiply_count_ = 0;
        ct_plain_multiply_count_ = 0;
        EvaluationResult evaluation_result;

        std::array<Ct, T> normalized{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized[token] = timed_ct(
                "layernorm_ln1", [&] {
                    return layernorm(encrypted_inputs[token], fixture_.ln1,
                                     "ln1_token" + std::to_string(token));
                });
        }
        if (gate == "ln1") {
            return finish(normalized, evaluation_result);
        }

        std::array<Ct, T> query{};
        std::array<Ct, T> key{};
        std::array<Ct, T> value{};
        for (std::size_t token = 0; token < T; ++token) {
            const BabyRotations baby = timed_baby(
                token == 0 ? "baby_rotation_qkv_token0" : "baby_rotation_qkv_token1",
                normalized[token]);

            const auto q_start = Clock::now();
            query[token] = timed_ct("matmul_qkv_query", [&] {
                return matmul_lintransform(baby.values[0], fixture_.qkv[0]);
            });
            const double gap_qk = elapsed_seconds(q_start);
            const auto k_start = Clock::now();
            key[token] = timed_ct("matmul_qkv_key", [&] {
                return matmul(baby, fixture_.qkv[1], false);
            });
            const double gap_kv_start_to_end = elapsed_seconds(k_start);
            const auto v_start = Clock::now();
            value[token] = timed_ct("matmul_qkv_value", [&] {
                return matmul(baby, fixture_.qkv[2], false);
            });
            (void)gap_qk;
            (void)gap_kv_start_to_end;
            (void)v_start;
        }

        std::array<Ct, T> context{};
        context[0] = value[0];
        const double score_scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
        Ct packed_delta;
        timed_void("attention_scores_and_deltas", [&] {
            for (std::size_t head = 0; head < HEADS; ++head) {
                std::vector<double> head_mask(D);
                for (std::size_t dim = head * HEAD_DIM;
                     dim < (head + 1) * HEAD_DIM; ++dim) {
                    head_mask[dim] = 1.0;
                }

                const Ct score_10 =
                    attention_score(query[1], key[0], head_mask, score_scale);
                const Ct score_11 =
                    attention_score(query[1], key[1], head_mask, score_scale);
                const Ct delta = cc_->EvalSub(score_11, score_10);
                const Ct masked_delta =
                    multiply_plain(delta, repeated_plain(head_mask));
                packed_delta =
                    head == 0 ? masked_delta
                              : cc_->EvalAdd(packed_delta, masked_delta);
            }
        });

        Ct packed_weight = client_.cross_boundary_active(
            "attention_sigmoid_heads_batched", packed_delta, exact_sigmoid,
            HEADS);
        timed_void("attention_context_blend", [&] {
            const Ct value_delta = cc_->EvalSub(value[1], value[0]);
            context[1] =
                cc_->EvalAdd(value[0], multiply(packed_weight, value_delta));
        });

        std::array<Ct, T> attention_projection{};
        for (std::size_t token = 0; token < T; ++token) {
            attention_projection[token] = timed_ct(
                "matmul_attention_projection", [&] {
                    return matmul(baby_rotations(context[token]),
                                  fixture_.attention_projection, false);
                });
        }
        if (gate == "attention") {
            return finish(attention_projection, evaluation_result);
        }

        std::array<Ct, T> residual1{};
        timed_void("residual1", [&] {
            for (std::size_t token = 0; token < T; ++token) {
                residual1[token] = cc_->EvalAdd(encrypted_inputs[token],
                                                attention_projection[token]);
            }
        });

        std::array<Ct, T> normalized2{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized2[token] = timed_ct(
                "layernorm_ln2", [&] {
                    return layernorm(residual1[token], fixture_.ln2,
                                     "ln2_token" + std::to_string(token));
                });
        }

        std::array<Ct, T> block_output{};
        for (std::size_t token = 0; token < T; ++token) {
            require_remaining_depth(
                normalized2[token]->GetLevel(), 4,
                "MLP token " + std::to_string(token));
            const BabyRotations fc_baby = timed_baby(
                token == 0 ? "baby_rotation_mlp_fc_token0"
                          : "baby_rotation_mlp_fc_token1",
                normalized2[token]);
            std::array<Ct, 4> hidden{};
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                hidden[chunk] = timed_ct("matmul_mlp_fc", [&] {
                    return matmul(fc_baby, fixture_.mlp_fc[chunk], false);
                });
            }

            Ct packed_hidden;
            timed_void("gelu_pack", [&] {
                for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                    const Ct selected =
                        multiply_plain(hidden[chunk], copy_active_plain(chunk));
                    packed_hidden =
                        chunk == 0 ? selected
                                  : cc_->EvalAdd(packed_hidden, selected);
                }
            });
            Ct packed_activated = client_.cross_boundary_active(
                "gelu_token" + std::to_string(token) + "_chunks_batched",
                packed_hidden, exact_gelu, 4);
            timed_void("gelu_restore", [&] {
                for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                    const Ct selected = multiply_plain(
                        packed_activated, copy_active_plain(chunk));
                    hidden[chunk] = replicate_copies(selected);
                }
            });

            Ct mlp;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                Ct contribution = timed_ct("matmul_mlp_projection", [&] {
                    return matmul(baby_rotations(hidden[chunk]),
                                  fixture_.mlp_projection[chunk], false);
                });
                mlp = chunk == 0 ? contribution
                                 : cc_->EvalAdd(mlp, contribution);
            }
            block_output[token] = cc_->EvalAdd(residual1[token], mlp);
        }
        require_remaining_depth(maximum_level(block_output), 1,
                                "final output packing");
        return finish(block_output, evaluation_result);
    }

    const Profiler& profiler() const { return profiler_; }

  private:
    struct BabyRotations {
        std::array<Ct, BSGS_N1> values{};
    };

    template <typename Fn>
    Ct timed_ct(const std::string& name, Fn&& fn) {
        cc_->Synchronize();
        const auto start = Clock::now();
        Ct result = fn();
        cc_->Synchronize();
        profiler_.add(name, elapsed_seconds(start));
        return result;
    }

    template <typename Fn>
    void timed_void(const std::string& name, Fn&& fn) {
        cc_->Synchronize();
        const auto start = Clock::now();
        fn();
        cc_->Synchronize();
        profiler_.add(name, elapsed_seconds(start));
    }

    BabyRotations timed_baby(const std::string& name, const Ct& input) {
        cc_->Synchronize();
        const auto start = Clock::now();
        BabyRotations result = baby_rotations(input);
        cc_->Synchronize();
        profiler_.add(name, elapsed_seconds(start));
        return result;
    }

    // `level` is the OpenFHE-convention level (0 = fresh) the plaintext is
    // pre-encoded at. Every existing call site keeps the implicit level-0
    // default; matmul_lintransform() below is the only caller that passes a
    // non-zero level, to satisfy FIDESlib::CKKS::LinearTransform's exact
    // level-match precondition (docs/hybrid/tasks.md, 2026-07-28 scoping
    // Finding 4).
    Plaintext raw_plain(const std::vector<double>& values,
                        std::uint32_t level = 0) {
        if (values.size() != SLOTS) {
            throw std::invalid_argument("raw plaintext must have SLOTS values");
        }
        return cc_->MakeCKKSPackedPlaintext(values, 1, level, nullptr, SLOTS);
    }

    Plaintext repeated_plain(const std::vector<double>& values) {
        if (values.size() != D) {
            throw std::invalid_argument("repeated plaintext must have D values");
        }
        std::vector<double> packed(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            std::copy(values.begin(), values.end(),
                      packed.begin() +
                          static_cast<std::ptrdiff_t>(copy * PACK_WIDTH));
        }
        return raw_plain(packed);
    }

    Plaintext output_block_plain(std::size_t token) {
        if (token >= T) {
            throw std::invalid_argument("output token outside T");
        }
        std::vector<double> packed(SLOTS);
        std::fill_n(
            packed.begin() +
                static_cast<std::ptrdiff_t>(token * PACK_WIDTH),
            D, 1.0);
        return raw_plain(packed);
    }

    Plaintext copy_active_plain(std::size_t copy) {
        if (copy >= COPIES) {
            throw std::invalid_argument("copy outside COPIES");
        }
        std::vector<double> packed(SLOTS);
        std::fill_n(
            packed.begin() +
                static_cast<std::ptrdiff_t>(copy * PACK_WIDTH),
            D, 1.0);
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

    void require_remaining_depth(std::size_t current_level,
                                 std::size_t levels_needed,
                                 const std::string& stage) const {
        if (current_level > MULT_DEPTH ||
            levels_needed > MULT_DEPTH - current_level) {
            throw std::runtime_error(
                "insufficient multiplicative depth before " + stage +
                ": current level " + std::to_string(current_level) +
                " + required " + std::to_string(levels_needed) +
                " exceeds depth " + std::to_string(MULT_DEPTH));
        }
    }

    template <typename Container>
    std::size_t maximum_level(const Container& ciphertexts) const {
        std::size_t maximum = 0;
        for (const Ct& ciphertext : ciphertexts) {
            maximum = std::max(maximum, ciphertext->GetLevel());
        }
        return maximum;
    }

    Ct sum_broadcast(const Ct& input) {
        return cc_->AccumulateSum(input, static_cast<int>(PACK_WIDTH), 1);
    }

    Ct mean_broadcast(const Ct& input) {
        Ct sum = sum_broadcast(input);
        return multiply_plain(
            sum, repeated_plain(std::vector<double>(D, 1.0 / static_cast<double>(D))));
    }

    Ct layernorm(const Ct& input, const std::vector<double>& weight,
                const std::string& stage_name) {
        Ct mean = mean_broadcast(input);
        Ct centered = cc_->EvalSub(input, mean);
        Ct variance = mean_broadcast(multiply(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        Ct inverse =
            client_.cross_boundary(stage_name + "_invsqrt", variance, exact_invsqrt);
        Ct normalized = multiply(centered, inverse);
        return multiply_plain(normalized, repeated_plain(weight));
    }

    Ct attention_score(const Ct& query, const Ct& key,
                       const std::vector<double>& head_mask, double scale) {
        Ct products = multiply(query, key);
        products = multiply_plain(products, repeated_plain(head_mask));
        Ct score = sum_broadcast(products);
        return multiply_plain(score,
                              repeated_plain(std::vector<double>(D, scale)));
    }

    Ct replicate_copies(const Ct& input) {
        const std::vector<std::int32_t> indices = {
            static_cast<std::int32_t>(PACK_WIDTH),
            -static_cast<std::int32_t>(PACK_WIDTH),
            static_cast<std::int32_t>(2 * PACK_WIDTH),
        };
        const auto rotated = cc_->EvalFastRotation(
            input, indices, cc_->GetCyclotomicOrder(), nullptr);
        if (rotated.size() != indices.size()) {
            throw std::runtime_error(
                "FIDESlib returned incomplete cross-copy rotations");
        }
        Ct output = input;
        for (const Ct& copy : rotated) {
            output = cc_->EvalAdd(output, copy);
        }
        return output;
    }

    BabyRotations baby_rotations(const Ct& input) {
        std::vector<std::int32_t> indices;
        indices.reserve(BSGS_N1 - 1);
        for (std::size_t index = 1; index < BSGS_N1; ++index) {
            indices.push_back(static_cast<std::int32_t>(index));
        }
        const auto rotated = cc_->EvalFastRotation(
            input, indices, cc_->GetCyclotomicOrder(), nullptr);
        if (rotated.size() != BSGS_N1 - 1) {
            throw std::runtime_error("FIDESlib returned incomplete baby rotations");
        }
        BabyRotations output;
        output.values[0] = input;
        for (std::size_t index = 1; index < BSGS_N1; ++index) {
            output.values[index] = rotated.at(index - 1);
        }
        return output;
    }

    // `detailed` additionally instruments every giant-step iteration of
    // THIS call (see header comment point 2) -- used for exactly one call
    // site (token 0's Q projection) so the rest of the block's timing is
    // perturbed by only the one coarse bracket already applied at the call
    // site in evaluate().
    Ct matmul(const BabyRotations& baby, const Matrix& weight, bool detailed) {
        if (weight.rows != D || weight.cols != D) {
            throw std::invalid_argument("real block matmul expects D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            if (detailed) {
                cc_->Synchronize();
                const auto inner_start = Clock::now();
                inner = bsgs_inner_loop(baby, weight, giant);
                cc_->Synchronize();
                profiler_.add("detail_ciphertext_plaintext_multiply_accumulate",
                              elapsed_seconds(inner_start));
            } else {
                inner = bsgs_inner_loop(baby, weight, giant);
            }

            if (giant != 0) {
                if (detailed) {
                    cc_->Synchronize();
                    const auto rotate_start = Clock::now();
                    inner = cc_->EvalRotate(
                        inner, static_cast<std::int32_t>(BSGS_N1 * giant));
                    cc_->Synchronize();
                    profiler_.add("detail_giant_rotation_keyswitch",
                                  elapsed_seconds(rotate_start));
                } else {
                    inner = cc_->EvalRotate(
                        inner, static_cast<std::int32_t>(BSGS_N1 * giant));
                }
            }

            if (detailed && giant != 0) {
                cc_->Synchronize();
                const auto add_start = Clock::now();
                result = cc_->EvalAdd(result, inner);
                cc_->Synchronize();
                profiler_.add("detail_giant_result_accumulate",
                              elapsed_seconds(add_start));
            } else {
                result = giant == 0 ? inner : cc_->EvalAdd(result, inner);
            }
        }
        ++matmul_count_;
        return result;
    }

    Ct bsgs_inner_loop(const BabyRotations& baby, const Matrix& weight,
                       std::size_t giant) {
        Ct inner;
        for (std::size_t small = 0; small < BSGS_N1; ++small) {
            const std::size_t diagonal = BSGS_N1 * giant + small;
            std::vector<double> packed_values(SLOTS);
            for (std::size_t block = 0; block < COPIES; ++block) {
                const std::size_t block_start = block * PACK_WIDTH;
                for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                    const std::size_t rolled_row =
                        (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH;
                    const std::size_t column = (rolled_row + diagonal) % PACK_WIDTH;
                    if (rolled_row < D && column < D) {
                        packed_values[block_start + row] =
                            weight(rolled_row, column);
                    }
                }
            }
            Ct term = multiply_plain(baby.values[small], raw_plain(packed_values));
            inner = small == 0 ? term : cc_->EvalAdd(inner, term);
        }
        return inner;
    }

    // Native FIDESlib::CKKS::LinearTransform integration -- used for exactly
    // ONE call site (see the header comment and docs/hybrid/tasks.md's
    // 2026-07-28 scoping section). Computes the identical D-by-D matmul as
    // matmul()/bsgs_inner_loop (same diagonal values, same BSGS_N1/BSGS_N2
    // split), but dispatches the 1024 ciphertext-plaintext diagonal products
    // as one real GPU-batched call instead of 1024 sequential EvalMult+
    // EvalAdd calls. `input` is the UN-rotated ciphertext (LinearTransform
    // computes its own baby-step hoisted rotations internally -- it has no
    // parameter to accept our precomputed BabyRotations, so that reuse is
    // lost for this call site; see scoping Finding 7 on why that is expected
    // to stay immaterial given rotation is <0.1% of a matmul call's cost).
    Ct matmul_lintransform(const Ct& input, const Matrix& weight) {
        if (weight.rows != D || weight.cols != D) {
            throw std::invalid_argument("real block matmul expects D by D");
        }

        // Diagonal plaintexts must be pre-encoded at `input`'s exact current
        // level (scoping Finding 4): OpenFHE's GetLevel() is the same
        // convention MakeCKKSPackedPlaintext's `level` argument expects, so
        // no separate adjustPlaintextToCiphertext call is needed.
        const std::uint32_t input_level =
            static_cast<std::uint32_t>(input->GetLevel());

        std::vector<Plaintext> pts_cpu;
        pts_cpu.reserve(BSGS_N1 * BSGS_N2);
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::size_t diagonal = BSGS_N1 * giant + small;
                std::vector<double> packed_values(SLOTS);
                for (std::size_t block = 0; block < COPIES; ++block) {
                    const std::size_t block_start = block * PACK_WIDTH;
                    for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH;
                        const std::size_t column =
                            (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            packed_values[block_start + row] =
                                weight(rolled_row, column);
                        }
                    }
                }
                pts_cpu.push_back(raw_plain(packed_values, input_level));
            }
        }

        // LinearTransform mutates its ciphertext argument in place. Work on
        // an independent native clone -- CiphertextImpl's copy constructor
        // deep-clones the GPU-resident object via CopyDeviceCiphertext, it
        // does not share the handle -- so `input` (== baby.values[0] ==
        // normalized[token], still needed by the sibling key/value matmul
        // calls and by the manual BSGS path) is never touched.
        Ct working = std::make_shared<CiphertextImpl<DCRTPoly>>(*input);
        cc_->LoadCiphertext(working);
        auto ctxt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(
            cc_->GetDeviceCiphertext(working->gpu));

        std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
        pts_gpu.reserve(pts_cpu.size());
        for (Plaintext& pt : pts_cpu) {
            cc_->LoadPlaintext(pt);
            auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(
                cc_->GetDevicePlaintext(pt->gpu));
            pts_gpu.push_back(pt_gpu.get());
        }

        // FIDESlib's own level-match assert (LinearTransform.cu) is compiled
        // out under this pinned build's -DCMAKE_BUILD_TYPE=Release (NDEBUG
        // strips assert()) -- see scoping Finding 5. Check explicitly so a
        // level-derivation bug fails loud here instead of silently
        // corrupting the computation or misbehaving inside the batched CUDA
        // kernels.
        for (FIDESlib::CKKS::Plaintext* pt_gpu : pts_gpu) {
            if (pt_gpu->c0.getLevel() != ctxt_gpu->getLevel()) {
                throw std::runtime_error(
                    "matmul_lintransform: diagonal plaintext level " +
                    std::to_string(pt_gpu->c0.getLevel()) +
                    " != ciphertext level " +
                    std::to_string(ctxt_gpu->getLevel()));
            }
        }

        FIDESlib::CKKS::LinearTransform(
            *ctxt_gpu, static_cast<int>(BSGS_N1 * BSGS_N2),
            static_cast<int>(BSGS_N1), pts_gpu, /*stride=*/1, /*offset=*/0);

        ++matmul_count_;
        return working;
    }

    EvaluationResult finish(const std::array<Ct, T>& values,
                            EvaluationResult evaluation_result) {
        require_remaining_depth(maximum_level(values), 1,
                                "selected gate output packing");
        Ct packed;
        for (std::size_t token = 0; token < T; ++token) {
            Ct selected =
                multiply_plain(values[token], output_block_plain(token));
            packed = token == 0 ? selected : cc_->EvalAdd(packed, selected);
        }
        evaluation_result.packed_output = packed;
        evaluation_result.ciphertext_plain_matmuls = matmul_count_;
        evaluation_result.ciphertext_ciphertext_multiplications =
            ct_ct_multiply_count_;
        evaluation_result.ciphertext_plaintext_multiplications =
            ct_plain_multiply_count_;
        return evaluation_result;
    }

    Cc cc_;
    const Fixture& fixture_;
    Client& client_;
    Profiler& profiler_;
    std::size_t matmul_count_ = 0;
    std::size_t ct_ct_multiply_count_ = 0;
    std::size_t ct_plain_multiply_count_ = 0;
};

struct Metrics {
    double global_rel_inf = 0.0;
    double worst_token_rel_inf = 0.0;
    double max_abs_error = 0.0;
    bool all_finite = false;
    bool passed = false;
};

Metrics measure(const std::vector<double>& got,
                const std::vector<double>& reference) {
    if (got.size() < T * PACK_WIDTH || reference.size() != T * D) {
        throw std::invalid_argument("output/reference size mismatch");
    }
    Metrics metrics;
    metrics.max_abs_error = 0.0;
    metrics.worst_token_rel_inf = 0.0;
    metrics.all_finite = true;
    double global_denominator = 0.0;
    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t dim = 0; dim < D; ++dim) {
            const std::size_t actual_slot = token * PACK_WIDTH + dim;
            const std::size_t reference_slot = token * D + dim;
            const double actual = got[actual_slot];
            const double expected = reference[reference_slot];
            metrics.all_finite = metrics.all_finite && std::isfinite(actual);
            const double error = std::abs(actual - expected);
            token_error = std::max(token_error, error);
            token_denominator =
                std::max(token_denominator, std::abs(expected));
            metrics.max_abs_error = std::max(metrics.max_abs_error, error);
            global_denominator =
                std::max(global_denominator, std::abs(expected));
        }
        metrics.worst_token_rel_inf =
            std::max(metrics.worst_token_rel_inf,
                     token_error / (token_denominator + 1e-15));
    }
    metrics.global_rel_inf =
        metrics.max_abs_error / (global_denominator + 1e-15);
    metrics.passed = metrics.all_finite && metrics.global_rel_inf <= TOL &&
                     metrics.worst_token_rel_inf <= TOL;
    return metrics;
}

std::string make_json(const Options& options, std::uint32_t ring,
                      std::size_t rotation_keys, double fixture_seconds,
                      double setup_seconds, double encryption_seconds,
                      double evaluation_seconds, double decrypt_seconds,
                      std::size_t round_trips,
                      std::size_t logical_boundary_instances,
                      double boundary_seconds_total,
                      const std::vector<BoundaryEvent>& boundary_log,
                      const EvaluationResult& evaluation,
                      const Metrics& metrics, const Profiler& profiler) {
    struct BoundaryAggregate {
        std::size_t count = 0;
        std::size_t logical_instances = 0;
        double seconds = 0.0;
    };
    std::map<std::string, BoundaryAggregate> boundary_summary;
    for (const BoundaryEvent& event : boundary_log) {
        auto& entry = boundary_summary[event.name];
        entry.count += 1;
        entry.logical_instances += event.logical_instances;
        entry.seconds += event.seconds;
    }

    const double server_seconds = evaluation_seconds - boundary_seconds_total;
    const double profiled_seconds = profiler.total_seconds();
    const double unaccounted_seconds =
        server_seconds - profiled_seconds;

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"FHE real-weight DNAGPT block-0 full gate, Scheme B hybrid client-assisted CKKS, FIDESlib GPU -- GPU-side profiling of the server_linear_algebra_seconds cost\",\n"
        << "  \"implementation_version\": \"t2-scheme-b-lintransform-v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS; server-only linear algebra, client-only decrypt at pre-declared nonlinearity boundaries)\",\n"
        << "  \"label\": \"single-call-site isolated gate: the QKV query projection (both T=2 tokens, matmul_qkv_query bucket) uses FIDESlib::CKKS::LinearTransform's native GPU-batched BSGS primitive instead of the manual 1024-call ciphertext-plaintext loop; all other 23 matmul() calls are unchanged, on the original manual path; retains the 2026-07-27 profiled prototype's Synchronize()-bracketed timing buckets for direct before/after comparison\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"gate\": \"" << json_escape(GATE) << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \"" << json_escape(options.backend_commit)
        << "\",\n"
        << "  \"container_image\": \"" << json_escape(options.container_image)
        << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \"" << json_escape(options.source_sha256)
        << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"fixture_contract_sha256\": \""
        << json_escape(options.fixture_contract_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"model\": \"dna_gpt0.1b_m classification checkpoint block 0\",\n"
        << "    \"D\": " << D << ",\n"
        << "    \"T\": " << T << ",\n"
        << "    \"heads\": " << HEADS << ",\n"
        << "    \"head_dim\": " << HEAD_DIM << ",\n"
        << "    \"mlp_dim\": " << MLP_DIM << ",\n"
        << "    \"pack_width\": " << PACK_WIDTH << ",\n"
        << "    \"batch_slots\": " << SLOTS << ",\n"
        << "    \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"scaling_technique\": \"FLEXIBLEAUTO\",\n"
        << "    \"bootstraps\": 0,\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"bsgs\": {\"n1\": " << BSGS_N1 << ", \"n2\": " << BSGS_N2
        << "},\n"
        << "    \"matmul_call_count\": " << evaluation.ciphertext_plain_matmuls
        << ",\n"
        << "    \"operation_counts\": {\n"
        << "      \"ciphertext_plaintext_matrix_products\": "
        << evaluation.ciphertext_plain_matmuls << ",\n"
        << "      \"explicit_ciphertext_ciphertext_multiplications\": "
        << evaluation.ciphertext_ciphertext_multiplications << ",\n"
        << "      \"explicit_ciphertext_plaintext_multiplications\": "
        << evaluation.ciphertext_plaintext_multiplications << "\n"
        << "    }\n"
        << "  },\n"
        << "  \"timings_seconds\": {\n"
        << "    \"load_verified_fixture\": " << fixture_seconds << ",\n"
        << "    \"context_keygen_load\": " << setup_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds << ",\n"
        << "    \"encrypted_evaluation\": " << evaluation_seconds << ",\n"
        << "    \"client_boundary_seconds_total\": " << boundary_seconds_total
        << ",\n"
        << "    \"server_linear_algebra_seconds\": " << server_seconds << ",\n"
        << "    \"final_decrypt\": " << decrypt_seconds << "\n"
        << "  },\n"
        << "  \"profiling_breakdown\": {\n"
        << "    \"note\": \"buckets are Synchronize()-bracketed wall-clock seconds; sum of buckets plus client_boundary_seconds_total should approximate server_linear_algebra_seconds plus client boundary -- unaccounted_seconds below is the gap, including this profiler's own barrier overhead\",\n"
        << "    \"sum_of_buckets_seconds\": " << profiled_seconds << ",\n"
        << "    \"unaccounted_seconds\": " << unaccounted_seconds << ",\n"
        << "    \"unaccounted_fraction_of_server_seconds\": "
        << (server_seconds > 0.0 ? unaccounted_seconds / server_seconds : 0.0)
        << ",\n"
        << "    \"buckets\": {\n";
    std::size_t bucket_index = 0;
    const auto& buckets = profiler.buckets();
    for (const auto& [name, bucket] : buckets) {
        out << "      \"" << json_escape(name) << "\": {\"count\": "
            << bucket.count << ", \"total_seconds\": " << bucket.seconds
            << ", \"mean_seconds\": "
            << (bucket.count > 0 ? bucket.seconds / static_cast<double>(bucket.count)
                                : 0.0)
            << ", \"fraction_of_server_seconds\": "
            << (server_seconds > 0.0 ? bucket.seconds / server_seconds : 0.0)
            << "}";
        out << (++bucket_index == buckets.size() ? "\n" : ",\n");
    }
    out << "    }\n"
        << "  },\n"
        << "  \"input_boundary\": \"token plus position embeddings encrypted by client\",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"declared_client_boundary_crossings\": " << round_trips << ",\n"
        << "  \"declared_logical_boundary_instances\": "
        << logical_boundary_instances << ",\n"
        << "  \"boundary_summary\": {\n";
    std::size_t summary_index = 0;
    for (const auto& [name, aggregate] : boundary_summary) {
        out << "    \"" << json_escape(name) << "\": {\"count\": "
            << aggregate.count << ", \"logical_instances\": "
            << aggregate.logical_instances << ", \"total_seconds\": "
            << aggregate.seconds << "}";
        out << (++summary_index == boundary_summary.size() ? "\n" : ",\n");
    }
    out << "  },\n"
        << "  \"final_decrypt_calls\": 1,\n"
        << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << "  \"worst_token_rel_inf\": " << metrics.worst_token_rel_inf << ",\n"
        << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << "  \"all_finite\": " << (metrics.all_finite ? "true" : "false")
        << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (metrics.passed ? "true" : "false") << ",\n"
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
        throw std::runtime_error("could not exclusively create evidence file " +
                                 path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t count = ::write(descriptor, contents.data() + offset,
                                      contents.size() - offset);
        if (count <= 0) {
            ::close(descriptor);
            throw std::runtime_error("short write to evidence file " +
                                     path.string());
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error("could not close evidence file " +
                                 path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        const auto fixture_start = Clock::now();
        const Fixture fixture = load_fixture(options.fixture_dir);
        const double fixture_seconds = elapsed_seconds(fixture_start);

        const std::vector<int> rotation_keys =
            required_rotation_keys(std::string(GATE));

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
        const double setup_seconds = elapsed_seconds(setup_start);
        const std::uint32_t ring = cc->GetRingDimension();
        std::cout << "[context] security=HEStd_128_classic ring=" << ring
                  << " depth=" << MULT_DEPTH << " slots=" << SLOTS
                  << " rotation_keys=" << rotation_keys.size()
                  << " gpu=" << options.gpu << '\n';

        const auto encryption_start = Clock::now();
        std::array<Ct, T> encrypted_inputs{};
        for (std::size_t token = 0; token < T; ++token) {
            std::vector<double> block(PACK_WIDTH);
            const auto begin =
                fixture.input.begin() + static_cast<std::ptrdiff_t>(token * D);
            std::copy(begin, begin + static_cast<std::ptrdiff_t>(D),
                      block.begin());
            std::vector<double> packed;
            packed.reserve(SLOTS);
            for (std::size_t copy = 0; copy < COPIES; ++copy) {
                packed.insert(packed.end(), block.begin(), block.end());
            }
            Plaintext plaintext =
                cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
            encrypted_inputs[token] = cc->Encrypt(keys.publicKey, plaintext);
        }
        cc->Synchronize();
        const double encryption_seconds = elapsed_seconds(encryption_start);

        Profiler profiler;
        Client client(cc, keys);
        EncryptedEvaluator evaluator(cc, fixture, client, profiler);
        const auto evaluation_start = Clock::now();
        EvaluationResult evaluation =
            evaluator.evaluate(encrypted_inputs, std::string(GATE));
        cc->Synchronize();
        const double evaluation_seconds = elapsed_seconds(evaluation_start);

        const auto decrypt_start = Clock::now();
        Plaintext decoded;
        cc->Decrypt(keys.secretKey, evaluation.packed_output, &decoded);
        decoded->SetLength(SLOTS);
        const auto raw = decoded->GetRealPackedValue();
        const std::vector<double> output(
            raw.begin(), raw.begin() + static_cast<std::ptrdiff_t>(T * PACK_WIDTH));
        const double decrypt_seconds = elapsed_seconds(decrypt_start);

        const Metrics metrics = measure(output, fixture.oracle_block_output);

        std::cout << "[result] round_trips=" << client.round_trips()
                  << " global_rel_inf=" << metrics.global_rel_inf
                  << " passed=" << (metrics.passed ? "true" : "false") << '\n';

        const std::string evidence = make_json(
            options, ring, rotation_keys.size(), fixture_seconds, setup_seconds,
            encryption_seconds, evaluation_seconds, decrypt_seconds,
            client.round_trips(), client.logical_boundary_instances(),
            client.boundary_seconds_total(), client.boundary_log(), evaluation,
            metrics, profiler);
        write_exclusive(options.output, evidence);
        std::cout << evidence;

        std::cout << (metrics.passed
                          ? "REAL_DNAGPT_FIDES_SCHEME_B_PROFILED_GATE_PASS"
                          : "REAL_DNAGPT_FIDES_SCHEME_B_PROFILED_GATE_FAIL")
                  << '\n';
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
