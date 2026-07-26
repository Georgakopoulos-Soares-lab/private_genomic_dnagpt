// Scheme B (hybrid client-assisted CKKS) in-process context/key CACHING
// prototype, forked from real_dnagpt_fides_scheme_b.cpp (the frozen,
// hash-pinned single-shot gate). Do not edit that file to add this behavior;
// its SHA-256 is recorded as provenance in existing immutable evidence JSONs.
//
// Every existing Scheme B gate re-runs GenCryptoContext/KeyGen/
// EvalMultKeyGen/EvalRotateKeyGen/LoadContext on every process invocation.
// The repo's 12-block extrapolation (docs/hybrid/tasks.md) assumes this setup
// is a one-time cost amortized across many block evaluations, but that has
// never actually been measured -- it is an [A] assumption. This binary proves
// (or disproves) the in-process half of that assumption: one context/key
// setup, then the SAME released-weight block-0 "full" gate evaluated
// `--repeats` times against fresh per-iteration Client/EncryptedEvaluator
// instances (so each iteration's round-trip/logical-boundary counters start
// at zero -- the checkable "no state leakage" property), each measured
// separately against the unchanged 4e-2 oracle gate.
//
// Honesty boundaries (see the "label" field emitted in the evidence JSON):
//   - This is a cache-LIFECYCLE proof, not a claim that two repetitions equal
//     two distinct DNAGPT blocks. Every repeat evaluates the identical
//     block-0 fixture (same weights, same input).
//   - This targets in-process reuse only. Whether a *serialized* context/key
//     lineage can be reloaded by a *separate* process invocation is scoped in
//     docs/hybrid/tasks.md but not implemented or measured here.
//   - Per FIDESlib's own CryptoContext.cpp (pinned commit
//     786c7600fb2f16b724e0acf73df367b27b8afed6), EvalMultKeyGen/
//     EvalRotateKeyGen must be called before the single LoadContext call (it
//     throws otherwise), so this binary always requests the rotation-key set
//     for the heaviest gate ("full") once, up front -- that set is a strict
//     superset of "attention"'s and "ln1"'s (see required_rotation_keys,
//     defined below), so it covers any gate mix, not just repeated "full"
//     calls.
//
// Security boundary unchanged from the original: EncryptedEvaluator never
// stores a private key; only Client does, and only Client calls
// cross_boundary()/cross_boundary_active(). main() performs exactly one final
// Decrypt per iteration, exactly as the original does once per process.

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
// forward pass, so the measured per-evaluation cost is the one the 12-block
// extrapolation actually cares about.
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

// Exact elementwise nonlinearities -- identical to the single-shot gate.
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
    std::size_t repeats = 2;
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
        "\nusage: real_dnagpt_fides_scheme_b_cached --gpu N --repeats N (>=2) "
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
        } else if (arg == "--repeats") {
            options.repeats = static_cast<std::size_t>(std::stoul(next()));
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
            std::cout << "usage: real_dnagpt_fides_scheme_b_cached --gpu N "
                         "--repeats N (>=2) --fixture-dir PATH --output PATH "
                         "--fixture-manifest-sha256 SHA\n";
            std::exit(0);
        } else {
            usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0) {
        usage_error("--gpu must be non-negative");
    }
    if (options.repeats < 2) {
        usage_error(
            "--repeats must be at least 2 -- this binary exists to prove "
            "setup is reused across more than one evaluation");
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

// Unchanged from the single-shot gate: full's key set is a strict superset
// of attention's, which is a strict superset of ln1's (BSGS + accumulate
// keys, plus 3 extra full-only rotations). Requesting "full"'s set once
// therefore covers any gate mix -- see the header comment above.
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

struct EvaluationResult {
    Ct packed_output;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels;
    std::size_t packed_level = 0;
    std::size_t ciphertext_plain_matmuls = 0;
    std::size_t ciphertext_ciphertext_multiplications = 0;
    std::size_t ciphertext_plaintext_multiplications = 0;
};

struct BoundaryEvent {
    std::string name;
    double seconds = 0.0;
    std::size_t logical_instances = 0;
};

// Client: the only party in this process that holds the secret key. One new
// Client is constructed per repeat iteration in main() below (see
// "no state leakage" in the header comment) -- its counters therefore start
// at zero for every iteration, by construction, not by an explicit reset.
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
    EncryptedEvaluator(Cc context, const Fixture& fixture, Client& client)
        : cc_(std::move(context)), fixture_(fixture), client_(client) {}

    EvaluationResult evaluate(const std::array<Ct, T>& encrypted_inputs,
                              const std::string& gate) {
        levels_.clear();
        matmul_count_ = 0;
        ct_ct_multiply_count_ = 0;
        ct_plain_multiply_count_ = 0;

        std::array<Ct, T> normalized{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized[token] =
                layernorm(encrypted_inputs[token], fixture_.ln1, "ln1_token" +
                                                                       std::to_string(token));
        }
        record("ln1", normalized);
        if (gate == "ln1") {
            return finish(normalized);
        }

        std::array<Ct, T> query{};
        std::array<Ct, T> key{};
        std::array<Ct, T> value{};
        for (std::size_t token = 0; token < T; ++token) {
            const BabyRotations baby = baby_rotations(normalized[token]);
            query[token] = matmul(baby, fixture_.qkv[0]);
            key[token] = matmul(baby, fixture_.qkv[1]);
            value[token] = matmul(baby, fixture_.qkv[2]);
        }
        record("qkv", query);

        std::array<Ct, T> context{};
        context[0] = value[0];
        const double score_scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
        Ct packed_delta;
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

        Ct packed_weight = client_.cross_boundary_active(
            "attention_sigmoid_heads_batched", packed_delta, exact_sigmoid,
            HEADS);
        const Ct value_delta = cc_->EvalSub(value[1], value[0]);
        context[1] =
            cc_->EvalAdd(value[0], multiply(packed_weight, value_delta));
        record("attention_context", context);

        std::array<Ct, T> attention_projection{};
        for (std::size_t token = 0; token < T; ++token) {
            attention_projection[token] =
                matmul(baby_rotations(context[token]),
                       fixture_.attention_projection);
        }
        record("attention_projection", attention_projection);
        if (gate == "attention") {
            return finish(attention_projection);
        }

        std::array<Ct, T> residual1{};
        for (std::size_t token = 0; token < T; ++token) {
            residual1[token] =
                cc_->EvalAdd(encrypted_inputs[token], attention_projection[token]);
        }
        record("residual1", residual1);

        std::array<Ct, T> normalized2{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized2[token] =
                layernorm(residual1[token], fixture_.ln2,
                          "ln2_token" + std::to_string(token));
        }
        record("ln2", normalized2);

        std::array<Ct, T> block_output{};
        for (std::size_t token = 0; token < T; ++token) {
            require_remaining_depth(
                normalized2[token]->GetLevel(), 4,
                "MLP token " + std::to_string(token));
            const BabyRotations fc_baby = baby_rotations(normalized2[token]);
            std::array<Ct, 4> hidden{};
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                hidden[chunk] = matmul(fc_baby, fixture_.mlp_fc[chunk]);
            }

            Ct packed_hidden;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                const Ct selected =
                    multiply_plain(hidden[chunk], copy_active_plain(chunk));
                packed_hidden =
                    chunk == 0 ? selected
                               : cc_->EvalAdd(packed_hidden, selected);
            }
            Ct packed_activated = client_.cross_boundary_active(
                "gelu_token" + std::to_string(token) + "_chunks_batched",
                packed_hidden, exact_gelu, 4);
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                const Ct selected = multiply_plain(
                    packed_activated, copy_active_plain(chunk));
                hidden[chunk] = replicate_copies(selected);
            }

            Ct mlp;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                Ct contribution =
                    matmul(baby_rotations(hidden[chunk]),
                           fixture_.mlp_projection[chunk]);
                mlp = chunk == 0 ? contribution
                                 : cc_->EvalAdd(mlp, contribution);
            }
            block_output[token] = cc_->EvalAdd(residual1[token], mlp);
        }
        record("block_output", block_output);
        require_remaining_depth(maximum_level(block_output), 1,
                                "final output packing");
        return finish(block_output);
    }

  private:
    struct BabyRotations {
        std::array<Ct, BSGS_N1> values{};
    };

    Plaintext raw_plain(const std::vector<double>& values) {
        if (values.size() != SLOTS) {
            throw std::invalid_argument("raw plaintext must have SLOTS values");
        }
        return cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
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

    Ct matmul(const BabyRotations& baby, const Matrix& weight) {
        if (weight.rows != D || weight.cols != D) {
            throw std::invalid_argument("real block matmul expects D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::size_t diagonal = BSGS_N1 * giant + small;
                std::vector<double> packed_values(SLOTS);
                for (std::size_t block = 0; block < COPIES; ++block) {
                    const std::size_t block_start = block * PACK_WIDTH;
                    for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH - BSGS_N1 * giant) %
                            PACK_WIDTH;
                        const std::size_t column =
                            (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            packed_values[block_start + row] =
                                weight(rolled_row, column);
                        }
                    }
                }
                Ct term =
                    multiply_plain(baby.values[small], raw_plain(packed_values));
                inner =
                    small == 0 ? term : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(
                    inner, static_cast<std::int32_t>(BSGS_N1 * giant));
            }
            result =
                giant == 0 ? inner : cc_->EvalAdd(result, inner);
        }
        ++matmul_count_;
        return result;
    }

    EvaluationResult finish(const std::array<Ct, T>& values) {
        require_remaining_depth(maximum_level(values), 1,
                                "selected gate output packing");
        Ct packed;
        for (std::size_t token = 0; token < T; ++token) {
            Ct selected =
                multiply_plain(values[token], output_block_plain(token));
            packed = token == 0 ? selected : cc_->EvalAdd(packed, selected);
        }
        return {.packed_output = packed,
                .levels = levels_,
                .packed_level = packed->GetLevel(),
                .ciphertext_plain_matmuls = matmul_count_,
                .ciphertext_ciphertext_multiplications =
                    ct_ct_multiply_count_,
                .ciphertext_plaintext_multiplications =
                    ct_plain_multiply_count_};
    }

    template <typename Container>
    void record(const std::string& name, const Container& ciphertexts) {
        std::size_t minimum = std::numeric_limits<std::size_t>::max();
        std::size_t maximum = 0;
        for (const Ct& ciphertext : ciphertexts) {
            minimum = std::min(minimum, ciphertext->GetLevel());
            maximum = std::max(maximum, ciphertext->GetLevel());
        }
        levels_[name] = {minimum, maximum};
    }

    Cc cc_;
    const Fixture& fixture_;
    Client& client_;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels_;
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

struct IterationRecord {
    std::size_t index = 0;
    double encrypt_embedded_inputs_seconds = 0.0;
    double encrypted_evaluation_seconds = 0.0;
    double final_decrypt_seconds = 0.0;
    std::size_t round_trips = 0;
    std::size_t logical_boundary_instances = 0;
    double boundary_seconds_total = 0.0;
    Metrics metrics;
    std::vector<BoundaryEvent> boundary_log;
};

std::string make_json_cached(const Options& options, std::uint32_t ring,
                             std::size_t rotation_keys,
                             double fixture_seconds, double setup_seconds,
                             const std::vector<IterationRecord>& iterations) {
    bool all_passed = true;
    for (const IterationRecord& iteration : iterations) {
        all_passed = all_passed && iteration.metrics.passed;
    }

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"FHE real-weight DNAGPT block-0 full gate, Scheme B hybrid client-assisted CKKS, FIDESlib GPU -- in-process context/key caching prototype\",\n"
        << "  \"implementation_version\": \"t2-scheme-b-cached-v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS; server-only linear algebra, client-only decrypt at pre-declared nonlinearity boundaries)\",\n"
        << "  \"label\": \"in-process cache-lifecycle proof: same block-0 full gate evaluated "
        << iterations.size()
        << " times sharing one context/key setup; not evidence of distinct multi-block correctness; not evidence of cross-process (serialized) caching\",\n"
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
        << "    \"rotation_key_gate_requested\": \"full (covers attention and ln1; see header comment)\",\n"
        << "    \"bsgs\": {\"n1\": " << BSGS_N1 << ", \"n2\": "
        << BSGS_N2 << "}\n"
        << "  },\n"
        << "  \"repeats\": " << iterations.size() << ",\n"
        << "  \"setup\": {\n"
        << "    \"note\": \"one GenCryptoContext + KeyGen + EvalMultKeyGen + EvalRotateKeyGen + LoadContext + Synchronize, shared by every iteration below\",\n"
        << "    \"load_verified_fixture_seconds\": " << fixture_seconds << ",\n"
        << "    \"context_keygen_load_seconds\": " << setup_seconds << "\n"
        << "  },\n"
        << "  \"iterations\": [\n";
    for (std::size_t i = 0; i < iterations.size(); ++i) {
        const IterationRecord& it = iterations[i];
        std::map<std::string, std::pair<std::size_t, std::size_t>> boundary_summary_count;
        struct BoundaryAggregate {
            std::size_t count = 0;
            std::size_t logical_instances = 0;
            double seconds = 0.0;
        };
        std::map<std::string, BoundaryAggregate> boundary_summary;
        for (const BoundaryEvent& event : it.boundary_log) {
            auto& entry = boundary_summary[event.name];
            entry.count += 1;
            entry.logical_instances += event.logical_instances;
            entry.seconds += event.seconds;
        }
        out << "    {\n"
            << "      \"index\": " << it.index << ",\n"
            << "      \"encrypt_embedded_inputs_seconds\": "
            << it.encrypt_embedded_inputs_seconds << ",\n"
            << "      \"encrypted_evaluation_seconds\": "
            << it.encrypted_evaluation_seconds << ",\n"
            << "      \"client_boundary_seconds_total\": "
            << it.boundary_seconds_total << ",\n"
            << "      \"server_linear_algebra_seconds\": "
            << (it.encrypted_evaluation_seconds - it.boundary_seconds_total)
            << ",\n"
            << "      \"final_decrypt_seconds\": " << it.final_decrypt_seconds
            << ",\n"
            << "      \"round_trips\": " << it.round_trips << ",\n"
            << "      \"logical_boundary_instances\": "
            << it.logical_boundary_instances << ",\n"
            << "      \"boundary_summary\": {\n";
        std::size_t summary_index = 0;
        for (const auto& [name, aggregate] : boundary_summary) {
            out << "        \"" << json_escape(name) << "\": {\"count\": "
                << aggregate.count << ", \"logical_instances\": "
                << aggregate.logical_instances << ", \"total_seconds\": "
                << aggregate.seconds << "}";
            out << (++summary_index == boundary_summary.size() ? "\n" : ",\n");
        }
        out << "      },\n"
            << "      \"global_rel_inf\": " << it.metrics.global_rel_inf << ",\n"
            << "      \"worst_token_rel_inf\": " << it.metrics.worst_token_rel_inf
            << ",\n"
            << "      \"max_abs_error\": " << it.metrics.max_abs_error << ",\n"
            << "      \"all_finite\": "
            << (it.metrics.all_finite ? "true" : "false") << ",\n"
            << "      \"tol\": " << TOL << ",\n"
            << "      \"passed\": " << (it.metrics.passed ? "true" : "false")
            << ",\n"
            << "      \"intermediate_decrypt_attempts\": 0,\n"
            << "      \"final_decrypt_calls\": 1\n"
            << "    }";
        out << (i + 1 == iterations.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"all_iterations_passed\": " << (all_passed ? "true" : "false")
        << ",\n"
        << "  \"input_boundary\": \"token plus position embeddings encrypted by client, per iteration, from the same fixture\",\n"
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

        // One rotation-key set, requested for "full" (the covering gate),
        // generated once, before the single LoadContext call below -- see
        // the header comment: FIDESlib throws if EvalRotateKeyGen is called
        // after LoadContext, so every key any iteration will ever need must
        // be requested here.
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
                  << " gpu=" << options.gpu
                  << " repeats=" << options.repeats << '\n';

        std::vector<IterationRecord> iterations;
        iterations.reserve(options.repeats);

        for (std::size_t iteration = 0; iteration < options.repeats;
             ++iteration) {
            const auto encryption_start = Clock::now();
            std::array<Ct, T> encrypted_inputs{};
            for (std::size_t token = 0; token < T; ++token) {
                std::vector<double> block(PACK_WIDTH);
                const auto begin = fixture.input.begin() +
                                   static_cast<std::ptrdiff_t>(token * D);
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

            // Fresh Client/EncryptedEvaluator per iteration: their counters
            // (round trips, logical boundary instances, boundary log) start
            // at zero for this iteration by construction -- the concrete,
            // checkable "no state leakage between repeated evaluations"
            // property. Both still share the same `cc`/`keys` set up once
            // above.
            Client client(cc, keys);
            EncryptedEvaluator evaluator(cc, fixture, client);
            const auto evaluation_start = Clock::now();
            EvaluationResult evaluation =
                evaluator.evaluate(encrypted_inputs, std::string(GATE));
            cc->Synchronize();
            const double evaluation_seconds =
                elapsed_seconds(evaluation_start);

            const auto decrypt_start = Clock::now();
            Plaintext decoded;
            cc->Decrypt(keys.secretKey, evaluation.packed_output, &decoded);
            decoded->SetLength(SLOTS);
            const auto raw = decoded->GetRealPackedValue();
            const std::vector<double> output(
                raw.begin(),
                raw.begin() + static_cast<std::ptrdiff_t>(T * PACK_WIDTH));
            const double decrypt_seconds = elapsed_seconds(decrypt_start);

            const Metrics metrics =
                measure(output, fixture.oracle_block_output);

            IterationRecord record;
            record.index = iteration;
            record.encrypt_embedded_inputs_seconds = encryption_seconds;
            record.encrypted_evaluation_seconds = evaluation_seconds;
            record.final_decrypt_seconds = decrypt_seconds;
            record.round_trips = client.round_trips();
            record.logical_boundary_instances =
                client.logical_boundary_instances();
            record.boundary_seconds_total = client.boundary_seconds_total();
            record.metrics = metrics;
            record.boundary_log = client.boundary_log();
            iterations.push_back(std::move(record));

            std::cout << "[iteration] index=" << iteration
                      << " round_trips=" << client.round_trips()
                      << " global_rel_inf=" << metrics.global_rel_inf
                      << " passed=" << (metrics.passed ? "true" : "false")
                      << '\n';
        }

        const std::string evidence = make_json_cached(
            options, ring, rotation_keys.size(), fixture_seconds,
            setup_seconds, iterations);
        write_exclusive(options.output, evidence);
        std::cout << evidence;

        bool all_passed = true;
        for (const IterationRecord& iteration : iterations) {
            all_passed = all_passed && iteration.metrics.passed;
        }
        std::cout << (all_passed
                          ? "REAL_DNAGPT_FIDES_SCHEME_B_CACHED_GATE_PASS"
                          : "REAL_DNAGPT_FIDES_SCHEME_B_CACHED_GATE_FAIL")
                  << '\n';
        return all_passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
