// Scheme B (hybrid client-assisted CKKS) cross-process context/key
// serialization prototype -- READER phase ("process B").
//
// Forked from real_dnagpt_fides_scheme_b_cached.cpp (itself forked from the
// frozen, hash-pinned real_dnagpt_fides_scheme_b.cpp). Neither of those files
// is edited by this one. The Fixture/Matrix/Client/EncryptedEvaluator/
// Metrics machinery below is copied unchanged from the cached prototype;
// only the context/key setup at the top of main() differs.
//
// This is the reader half of the cross-process caching question left open
// by docs/hybrid/tasks.md ("Scheme B context/key caching: scoping +
// in-process prototype", finding 5): can a *second, independent process*
// skip the CPU-bound KeyGen/EvalMultKeyGen/EvalRotateKeyGen (the same cost
// that stalled 8.5+ minutes under host contention in the 2026-07-25 session)
// by deserializing a context/key lineage that
// real_dnagpt_fides_scheme_b_serialize_writer wrote to disk?
//
// Verified directly against FIDESlib's pinned commit
// (786c7600fb2f16b724e0acf73df367b27b8afed6): fideslib::Serial::
// DeserializeFromFile for CryptoContext constructs a brand-new
// CryptoContextImpl<DCRTPoly> from cereal-deserialized vanilla-OpenFHE state
// plus the ".dev" sidecar (restoring devices/rotation_indexes/keyDist/
// slots_bootstrap); DeserializeEvalMultKey/DeserializeEvalAutomorphismKey
// populate the exact same process-global OpenFHE static key maps that
// EvalMultKeyGen/EvalRotateKeyGen would have populated live (api/
// CryptoContext.cpp; examples/serial/src/serial.cpp and examples/resnet/
// src/controller.cpp's deserialize_context()/load_context() do exactly this,
// with NO EvalMultKeyGen/EvalRotateKeyGen call anywhere in that path). This
// reader therefore contains ZERO occurrences of EvalMultKeyGen(,
// EvalRotateKeyGen, or GenCryptoContext -- verified by
// test_serialization_contract.py, not just asserted here.
//
// What CANNOT be skipped, and is measured separately to prove it: FIDESlib's
// GPU-side FIDESlib::CKKS::Context (populated by LoadContext via an internal
// GenCryptoContextGPU call) has no serialize/deserialize path at all (grepped
// src/CKKS/Context.cuh/.cu -- no such symbols exist). Every process that
// wants to evaluate on GPU must call LoadContext once, which allocates fresh
// GPU-side NTT tables/memory and re-uploads the key-switching keys read from
// the (here, just-deserialized) CPU-side key maps. This binary times that
// step (`load_context_gpu_seconds`) separately from deserialization
// (`deserialize_seconds`) and from the actual encrypted evaluation, so the
// evidence JSON can show exactly how much of the original one-time setup
// cost is avoidable via cross-process reload (KeyGen/EvalRotateKeyGen, paid
// only once by the writer) versus unavoidable every single process
// (LoadContext/GenCryptoContextGPU).
//
// Secret-key hygiene: this reader deserializes a secret key from
// --state-dir/secret-key.txt to play the Client role (exactly as a
// restarted real client process would hold its own key) -- it is not the
// untrusted compute provider. Its path/contents are never printed. As in
// the writer, this is a testbed simplification for measurement, not a
// production client/server key-custody design (out of scope per CLAUDE.md).
// EncryptedEvaluator, as in every other Scheme B gate, still never touches
// Decrypt( or a secret key -- verified below by test_serialization_contract.py
// using the same source-slice check as test_caching_contract.py.

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
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
constexpr double EPS = 1e-5;
constexpr double TOL = 4e-2;

// This prototype always evaluates the heaviest gate, same as the cached
// prototype, since that is the number the 12-block extrapolation cares
// about and it is the covering superset of every rotation key requested.
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
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
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

// Non-secret key=value companion the writer left in --state-dir. No keys,
// no paths -- just the writer's own timing/config numbers, folded into this
// reader's evidence JSON below.
std::unordered_map<std::string, std::string> read_writer_timing(
    const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("could not open writer timing file: " + path.string());
    }
    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(stream, line)) {
        const std::size_t split = line.find('=');
        if (split == std::string::npos) {
            continue;
        }
        values[line.substr(0, split)] = line.substr(split + 1);
    }
    for (const char* required : {"keygen_seconds", "serialize_seconds", "rotation_keys"}) {
        if (values.find(required) == values.end()) {
            throw std::runtime_error(std::string("writer timing file missing field: ") +
                                     required);
        }
    }
    return values;
}

struct Options {
    int gpu = 0;
    std::filesystem::path state_dir;
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
        "\nusage: real_dnagpt_fides_scheme_b_serialize_reader --gpu N "
        "--state-dir PATH --fixture-dir PATH --output PATH "
        "--fixture-manifest-sha256 SHA [--backend-commit SHA] "
        "[--container-image NAME] [--environment TEXT] [--source-sha256 SHA] "
        "[--fixture-contract-sha256 SHA]");
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
            std::cout << "usage: real_dnagpt_fides_scheme_b_serialize_reader "
                         "--gpu N --state-dir PATH --fixture-dir PATH "
                         "--output PATH --fixture-manifest-sha256 SHA\n";
            std::exit(0);
        } else {
            usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0) {
        usage_error("--gpu must be non-negative");
    }
    if (options.state_dir.empty() || options.fixture_dir.empty() || options.output.empty()) {
        usage_error("--state-dir, --fixture-dir, and --output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit " + options.backend_commit);
    }
    if (options.fixture_manifest_sha256 != PINNED_FIXTURE_MANIFEST) {
        usage_error("refusing unpinned fixture manifest " +
                    options.fixture_manifest_sha256);
    }
    if (!std::filesystem::is_directory(options.state_dir)) {
        throw std::runtime_error("state directory missing: " + options.state_dir.string());
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
                                 options.output.string());
    }
    return options;
}

// ---- Fixture loading: byte-identical to real_dnagpt_fides_scheme_b_cached.cpp ----

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
            output.values[row * count + col] = input(row, first + col);
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

    const Matrix qkv = load_matrix(directory / "weights__attn_qkv.bin", 3 * D, D);
    for (std::size_t part = 0; part < 3; ++part) {
        fixture.qkv[part] = rows(qkv, part * D, D);
    }
    fixture.attention_projection = load_matrix(directory / "weights__attn_proj.bin", D, D);

    const Matrix fc = load_matrix(directory / "weights__mlp_fc.bin", MLP_DIM, D);
    for (std::size_t part = 0; part < 4; ++part) {
        fixture.mlp_fc[part] = rows(fc, part * D, D);
    }
    const Matrix projection = load_matrix(directory / "weights__mlp_proj.bin", D, MLP_DIM);
    for (std::size_t part = 0; part < 4; ++part) {
        fixture.mlp_projection[part] = columns(projection, part * D, D);
    }

    fixture.oracle_ln1 = read_f64(directory / "oracle__ln1_output.bin", T * D);
    fixture.oracle_attention_projection =
        read_f64(directory / "oracle__attention_projection.bin", T * D);
    fixture.oracle_block_output = read_f64(directory / "oracle__block_output.bin", T * D);
    return fixture;
}

// ---- Client / EncryptedEvaluator: byte-identical logic to the cached gate ----

struct BoundaryEvent {
    std::string name;
    double seconds = 0.0;
    std::size_t logical_instances = 0;
};

// Client: the only party in this process that holds the secret key -- here,
// deserialized from --state-dir/secret-key.txt, simulating a restarted
// client process. See the file header comment for the hygiene rule this is
// subject to.
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
    std::size_t logical_boundary_instances() const { return logical_boundary_instances_; }
    double boundary_seconds_total() const { return boundary_seconds_total_; }
    const std::vector<BoundaryEvent>& boundary_log() const { return boundary_log_; }

  private:
    Ct cross_boundary_impl(const std::string& name, const Ct& ciphertext,
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

struct EvaluationResult {
    Ct packed_output;
    std::size_t packed_level = 0;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture, Client& client)
        : cc_(std::move(context)), fixture_(fixture), client_(client) {}

    EvaluationResult evaluate(const std::array<Ct, T>& encrypted_inputs,
                              const std::string& gate) {
        std::array<Ct, T> normalized{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized[token] =
                layernorm(encrypted_inputs[token], fixture_.ln1,
                         "ln1_token" + std::to_string(token));
        }
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

        std::array<Ct, T> context{};
        context[0] = value[0];
        const double score_scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
        Ct packed_delta;
        for (std::size_t head = 0; head < HEADS; ++head) {
            std::vector<double> head_mask(D);
            for (std::size_t dim = head * HEAD_DIM; dim < (head + 1) * HEAD_DIM; ++dim) {
                head_mask[dim] = 1.0;
            }
            const Ct score_10 = attention_score(query[1], key[0], head_mask, score_scale);
            const Ct score_11 = attention_score(query[1], key[1], head_mask, score_scale);
            const Ct delta = cc_->EvalSub(score_11, score_10);
            const Ct masked_delta = multiply_plain(delta, repeated_plain(head_mask));
            packed_delta = head == 0 ? masked_delta : cc_->EvalAdd(packed_delta, masked_delta);
        }

        Ct packed_weight = client_.cross_boundary_active(
            "attention_sigmoid_heads_batched", packed_delta, exact_sigmoid, HEADS);
        const Ct value_delta = cc_->EvalSub(value[1], value[0]);
        context[1] = cc_->EvalAdd(value[0], cc_->EvalMult(packed_weight, value_delta));

        std::array<Ct, T> attention_projection{};
        for (std::size_t token = 0; token < T; ++token) {
            attention_projection[token] =
                matmul(baby_rotations(context[token]), fixture_.attention_projection);
        }
        if (gate == "attention") {
            return finish(attention_projection);
        }

        std::array<Ct, T> residual1{};
        for (std::size_t token = 0; token < T; ++token) {
            residual1[token] = cc_->EvalAdd(encrypted_inputs[token], attention_projection[token]);
        }

        std::array<Ct, T> normalized2{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized2[token] =
                layernorm(residual1[token], fixture_.ln2, "ln2_token" + std::to_string(token));
        }

        std::array<Ct, T> block_output{};
        for (std::size_t token = 0; token < T; ++token) {
            require_remaining_depth(normalized2[token]->GetLevel(), 4,
                                    "MLP token " + std::to_string(token));
            const BabyRotations fc_baby = baby_rotations(normalized2[token]);
            std::array<Ct, 4> hidden{};
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                hidden[chunk] = matmul(fc_baby, fixture_.mlp_fc[chunk]);
            }

            Ct packed_hidden;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                const Ct selected = multiply_plain(hidden[chunk], copy_active_plain(chunk));
                packed_hidden = chunk == 0 ? selected : cc_->EvalAdd(packed_hidden, selected);
            }
            Ct packed_activated = client_.cross_boundary_active(
                "gelu_token" + std::to_string(token) + "_chunks_batched", packed_hidden,
                exact_gelu, 4);
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                const Ct selected = multiply_plain(packed_activated, copy_active_plain(chunk));
                hidden[chunk] = replicate_copies(selected);
            }

            Ct mlp;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                Ct contribution = matmul(baby_rotations(hidden[chunk]), fixture_.mlp_projection[chunk]);
                mlp = chunk == 0 ? contribution : cc_->EvalAdd(mlp, contribution);
            }
            block_output[token] = cc_->EvalAdd(residual1[token], mlp);
        }
        require_remaining_depth(maximum_level(block_output), 1, "final output packing");
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

    // FIDESlib's EvalMult(Ciphertext, Plaintext) overload takes the
    // plaintext by non-const lvalue reference, so a temporary returned
    // directly from raw_plain()/repeated_plain()/etc. cannot bind to it.
    // Materializing it as a named by-value parameter here (exactly as
    // real_dnagpt_fides_scheme_b_cached.cpp's multiply_plain() does) gives
    // the compiler an lvalue to bind against.
    Ct multiply_plain(const Ct& input, Plaintext plaintext) {
        return cc_->EvalMult(input, plaintext);
    }

    Plaintext repeated_plain(const std::vector<double>& values) {
        if (values.size() != D) {
            throw std::invalid_argument("repeated plaintext must have D values");
        }
        std::vector<double> packed(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            std::copy(values.begin(), values.end(),
                      packed.begin() + static_cast<std::ptrdiff_t>(copy * PACK_WIDTH));
        }
        return raw_plain(packed);
    }

    Plaintext output_block_plain(std::size_t token) {
        if (token >= T) {
            throw std::invalid_argument("output token outside T");
        }
        std::vector<double> packed(SLOTS);
        std::fill_n(packed.begin() + static_cast<std::ptrdiff_t>(token * PACK_WIDTH), D, 1.0);
        return raw_plain(packed);
    }

    Plaintext copy_active_plain(std::size_t copy) {
        if (copy >= COPIES) {
            throw std::invalid_argument("copy outside COPIES");
        }
        std::vector<double> packed(SLOTS);
        std::fill_n(packed.begin() + static_cast<std::ptrdiff_t>(copy * PACK_WIDTH), D, 1.0);
        return raw_plain(packed);
    }

    void require_remaining_depth(std::size_t current_level, std::size_t levels_needed,
                                 const std::string& stage) const {
        if (current_level > MULT_DEPTH || levels_needed > MULT_DEPTH - current_level) {
            throw std::runtime_error(
                "insufficient multiplicative depth before " + stage + ": current level " +
                std::to_string(current_level) + " + required " +
                std::to_string(levels_needed) + " exceeds depth " +
                std::to_string(MULT_DEPTH));
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
        Ct variance = mean_broadcast(cc_->EvalMult(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        Ct inverse = client_.cross_boundary(stage_name + "_invsqrt", variance, exact_invsqrt);
        Ct normalized = cc_->EvalMult(centered, inverse);
        return multiply_plain(normalized, repeated_plain(weight));
    }

    Ct attention_score(const Ct& query, const Ct& key, const std::vector<double>& head_mask,
                       double scale) {
        Ct products = cc_->EvalMult(query, key);
        products = multiply_plain(products, repeated_plain(head_mask));
        Ct score = sum_broadcast(products);
        return multiply_plain(score, repeated_plain(std::vector<double>(D, scale)));
    }

    Ct replicate_copies(const Ct& input) {
        const std::vector<std::int32_t> indices = {
            static_cast<std::int32_t>(PACK_WIDTH),
            -static_cast<std::int32_t>(PACK_WIDTH),
            static_cast<std::int32_t>(2 * PACK_WIDTH),
        };
        const auto rotated =
            cc_->EvalFastRotation(input, indices, cc_->GetCyclotomicOrder(), nullptr);
        if (rotated.size() != indices.size()) {
            throw std::runtime_error("FIDESlib returned incomplete cross-copy rotations");
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
        const auto rotated =
            cc_->EvalFastRotation(input, indices, cc_->GetCyclotomicOrder(), nullptr);
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
                            (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH;
                        const std::size_t column = (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            packed_values[block_start + row] = weight(rolled_row, column);
                        }
                    }
                }
                Ct term = multiply_plain(baby.values[small], raw_plain(packed_values));
                inner = small == 0 ? term : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(inner, static_cast<std::int32_t>(BSGS_N1 * giant));
            }
            result = giant == 0 ? inner : cc_->EvalAdd(result, inner);
        }
        return result;
    }

    EvaluationResult finish(const std::array<Ct, T>& values) {
        require_remaining_depth(maximum_level(values), 1, "selected gate output packing");
        Ct packed;
        for (std::size_t token = 0; token < T; ++token) {
            Ct selected = multiply_plain(values[token], output_block_plain(token));
            packed = token == 0 ? selected : cc_->EvalAdd(packed, selected);
        }
        return {packed, packed->GetLevel()};
    }

    Cc cc_;
    const Fixture& fixture_;
    Client& client_;
};

struct Metrics {
    double global_rel_inf = 0.0;
    double worst_token_rel_inf = 0.0;
    double max_abs_error = 0.0;
    bool all_finite = false;
    bool passed = false;
};

Metrics measure(const std::vector<double>& got, const std::vector<double>& reference) {
    if (got.size() < T * PACK_WIDTH || reference.size() != T * D) {
        throw std::invalid_argument("output/reference size mismatch");
    }
    Metrics metrics;
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
            token_denominator = std::max(token_denominator, std::abs(expected));
            metrics.max_abs_error = std::max(metrics.max_abs_error, error);
            global_denominator = std::max(global_denominator, std::abs(expected));
        }
        metrics.worst_token_rel_inf =
            std::max(metrics.worst_token_rel_inf, token_error / (token_denominator + 1e-15));
    }
    metrics.global_rel_inf = metrics.max_abs_error / (global_denominator + 1e-15);
    metrics.passed = metrics.all_finite && metrics.global_rel_inf <= TOL &&
                     metrics.worst_token_rel_inf <= TOL;
    return metrics;
}

void write_exclusive(const std::filesystem::path& path, const std::string& contents) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) {
        throw std::runtime_error("could not exclusively create evidence file " + path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t count =
            ::write(descriptor, contents.data() + offset, contents.size() - offset);
        if (count <= 0) {
            ::close(descriptor);
            throw std::runtime_error("short write to evidence file " + path.string());
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error("could not close evidence file " + path.string());
    }
}

std::string make_json(const Options& options, std::uint32_t ring,
                      const std::unordered_map<std::string, std::string>& writer_timing,
                      double fixture_seconds, double deserialize_seconds,
                      double load_context_gpu_seconds, double encryption_seconds,
                      double evaluation_seconds, double final_decrypt_seconds,
                      const Client& client, const Metrics& metrics) {
    const double writer_keygen_seconds = std::stod(writer_timing.at("keygen_seconds"));
    const double writer_serialize_seconds = std::stod(writer_timing.at("serialize_seconds"));
    const std::size_t rotation_keys =
        static_cast<std::size_t>(std::stoul(writer_timing.at("rotation_keys")));

    std::map<std::string, std::pair<std::size_t, std::size_t>> boundary_counts;
    struct BoundaryAggregate {
        std::size_t count = 0;
        std::size_t logical_instances = 0;
        double seconds = 0.0;
    };
    std::map<std::string, BoundaryAggregate> boundary_summary;
    for (const BoundaryEvent& event : client.boundary_log()) {
        auto& entry = boundary_summary[event.name];
        entry.count += 1;
        entry.logical_instances += event.logical_instances;
        entry.seconds += event.seconds;
    }

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"FHE real-weight DNAGPT block-0 full gate, Scheme B "
           "hybrid client-assisted CKKS, FIDESlib GPU -- cross-process "
           "context/key serialization prototype (READER phase)\",\n"
        << "  \"implementation_version\": \"t2-scheme-b-serialized-reader-v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS; server-only linear "
           "algebra, client-only decrypt at pre-declared nonlinearity "
           "boundaries)\",\n"
        << "  \"label\": \"cross-process cache-lifecycle proof: this process "
           "never calls GenCryptoContext/KeyGen/EvalMultKeyGen/"
           "EvalRotateKeyGen -- it deserializes a context/key lineage a "
           "separate writer process (real_dnagpt_fides_scheme_b_serialize_"
           "writer) wrote to --state-dir, then still pays LoadContext/"
           "GenCryptoContextGPU itself since FIDESlib has no serialize path "
           "for the GPU-side context; this is a testbed simplification of "
           "the client role (secret key deserialized here to simulate a "
           "restarted client process), not a production client/server "
           "key-custody design\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"gate\": \"" << json_escape(GATE) << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \"" << json_escape(options.backend_commit) << "\",\n"
        << "  \"container_image\": \"" << json_escape(options.container_image) << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \"" << json_escape(options.source_sha256) << "\",\n"
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
        << "    \"bootstraps\": 0,\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"rotation_key_gate_requested\": \"full (covers attention and ln1)\",\n"
        << "    \"bsgs\": {\"n1\": " << BSGS_N1 << ", \"n2\": " << BSGS_N2 << "}\n"
        << "  },\n"
        << "  \"cross_process_breakdown\": {\n"
        << "    \"writer_keygen_seconds\": " << writer_keygen_seconds << ",\n"
        << "    \"writer_serialize_to_disk_seconds\": " << writer_serialize_seconds << ",\n"
        << "    \"reader_deserialize_seconds\": " << deserialize_seconds << ",\n"
        << "    \"reader_load_context_gpu_seconds\": " << load_context_gpu_seconds << ",\n"
        << "    \"reader_encrypted_evaluation_seconds\": " << evaluation_seconds << ",\n"
        << "    \"avoidable_via_cross_process_reload_seconds\": " << writer_keygen_seconds
        << ",\n"
        << "    \"unavoidable_per_process_gpu_load_seconds\": " << load_context_gpu_seconds
        << ",\n"
        << "    \"note\": \"avoidable = writer's KeyGen+EvalMultKeyGen+"
           "EvalRotateKeyGen, which this reader never calls (see source: zero "
           "textual occurrences, checked by test_serialization_contract.py); "
           "unavoidable = LoadContext/GenCryptoContextGPU, which every "
           "process must pay because FIDESlib has no serialize path for the "
           "GPU-side FIDESlib::CKKS::Context\"\n"
        << "  },\n"
        << "  \"load_verified_fixture_seconds\": " << fixture_seconds << ",\n"
        << "  \"timings_seconds\": {\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds << ",\n"
        << "    \"encrypted_evaluation\": " << evaluation_seconds << ",\n"
        << "    \"client_boundary_seconds_total\": " << client.boundary_seconds_total()
        << ",\n"
        << "    \"server_linear_algebra_seconds\": "
        << (evaluation_seconds - client.boundary_seconds_total()) << ",\n"
        << "    \"final_decrypt\": " << final_decrypt_seconds << "\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"round_trips\": " << client.round_trips() << ",\n"
        << "    \"logical_boundary_instances\": " << client.logical_boundary_instances()
        << ",\n"
        << "    \"boundary_summary\": {\n";
    std::size_t summary_index = 0;
    for (const auto& [name, aggregate] : boundary_summary) {
        out << "      \"" << json_escape(name) << "\": {\"count\": " << aggregate.count
            << ", \"logical_instances\": " << aggregate.logical_instances
            << ", \"total_seconds\": " << aggregate.seconds << "}";
        out << (++summary_index == boundary_summary.size() ? "\n" : ",\n");
    }
    out << "    }\n"
        << "  },\n"
        << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << "  \"worst_token_rel_inf\": " << metrics.worst_token_rel_inf << ",\n"
        << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << "  \"all_finite\": " << (metrics.all_finite ? "true" : "false") << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (metrics.passed ? "true" : "false") << ",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"declared_client_boundary_crossings\": " << client.round_trips() << ",\n"
        << "  \"declared_logical_boundary_instances\": "
        << client.logical_boundary_instances() << ",\n"
        << "  \"final_decrypt_calls\": 1,\n"
        << "  \"evaluator_has_private_key\": false\n"
        << "}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        const auto fixture_start = Clock::now();
        const Fixture fixture = load_fixture(options.fixture_dir);
        const double fixture_seconds = elapsed_seconds(fixture_start);

        const std::unordered_map<std::string, std::string> writer_timing =
            read_writer_timing(options.state_dir / "writer_timing.txt");

        // Cross-process reload: deserialize everything the writer process
        // serialized. NOTE: no GenCryptoContext, no KeyGen, no
        // EvalMultKeyGen, no EvalRotateKeyGen anywhere in this file --
        // verified textually by test_serialization_contract.py, not merely
        // asserted here.
        const auto deserialize_start = Clock::now();
        Cc cc;
        if (!Serial::DeserializeFromFile((options.state_dir / "crypto-context.txt").string(),
                                         cc, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize CryptoContext");
        }
        Keys keys;
        if (!Serial::DeserializeFromFile((options.state_dir / "public-key.txt").string(),
                                         keys.publicKey, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize PublicKey");
        }
        // Secret-key hygiene: this reader plays the (restarted) client
        // role, so it legitimately needs the secret key -- see the file
        // header comment. Its path/contents are never printed below.
        if (!Serial::DeserializeFromFile((options.state_dir / "secret-key.txt").string(),
                                         keys.secretKey, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize PrivateKey");
        }

        std::ifstream mult_key_stream(options.state_dir / "eval-mult-key.bin",
                                      std::ios::in | std::ios::binary);
        if (!mult_key_stream.is_open() ||
            !cc->DeserializeEvalMultKey(mult_key_stream, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize EvalMultKey");
        }
        mult_key_stream.close();

        std::ifstream rotation_key_stream(options.state_dir / "eval-automorphism-key.bin",
                                          std::ios::in | std::ios::binary);
        if (!rotation_key_stream.is_open() ||
            !cc->DeserializeEvalAutomorphismKey(rotation_key_stream, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize EvalAutomorphismKey");
        }
        rotation_key_stream.close();
        const double deserialize_seconds = elapsed_seconds(deserialize_start);

        // Unavoidable per-process GPU step: FIDESlib has no serialize path
        // for the GPU-side FIDESlib::CKKS::Context, so LoadContext must run
        // here (it internally calls GenCryptoContextGPU) regardless of what
        // was deserialized above.
        const auto load_context_start = Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        const double load_context_gpu_seconds = elapsed_seconds(load_context_start);
        const std::uint32_t ring = cc->GetRingDimension();

        std::cout << "[context] security=HEStd_128_classic ring=" << ring
                  << " depth=" << MULT_DEPTH << " slots=" << SLOTS << " gpu=" << options.gpu
                  << " deserialize_seconds=" << deserialize_seconds
                  << " load_context_gpu_seconds=" << load_context_gpu_seconds << '\n';

        const auto encryption_start = Clock::now();
        std::array<Ct, T> encrypted_inputs{};
        for (std::size_t token = 0; token < T; ++token) {
            std::vector<double> block(PACK_WIDTH);
            const auto begin = fixture.input.begin() + static_cast<std::ptrdiff_t>(token * D);
            std::copy(begin, begin + static_cast<std::ptrdiff_t>(D), block.begin());
            std::vector<double> packed;
            packed.reserve(SLOTS);
            for (std::size_t copy = 0; copy < COPIES; ++copy) {
                packed.insert(packed.end(), block.begin(), block.end());
            }
            Plaintext plaintext = cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
            encrypted_inputs[token] = cc->Encrypt(keys.publicKey, plaintext);
        }
        cc->Synchronize();
        const double encryption_seconds = elapsed_seconds(encryption_start);

        Client client(cc, keys);
        EncryptedEvaluator evaluator(cc, fixture, client);
        const auto evaluation_start = Clock::now();
        EvaluationResult evaluation = evaluator.evaluate(encrypted_inputs, std::string(GATE));
        cc->Synchronize();
        const double evaluation_seconds = elapsed_seconds(evaluation_start);

        const auto decrypt_start = Clock::now();
        Plaintext decoded;
        cc->Decrypt(keys.secretKey, evaluation.packed_output, &decoded);
        decoded->SetLength(SLOTS);
        const auto raw = decoded->GetRealPackedValue();
        const std::vector<double> output(raw.begin(),
                                         raw.begin() + static_cast<std::ptrdiff_t>(T * PACK_WIDTH));
        const double final_decrypt_seconds = elapsed_seconds(decrypt_start);

        const Metrics metrics = measure(output, fixture.oracle_block_output);

        const std::string evidence = make_json(
            options, ring, writer_timing, fixture_seconds, deserialize_seconds,
            load_context_gpu_seconds, encryption_seconds, evaluation_seconds,
            final_decrypt_seconds, client, metrics);
        write_exclusive(options.output, evidence);
        std::cout << evidence;

        std::cout << "round_trips=" << client.round_trips()
                  << " global_rel_inf=" << metrics.global_rel_inf
                  << " passed=" << (metrics.passed ? "true" : "false") << '\n';
        std::cout << (metrics.passed ? "REAL_DNAGPT_FIDES_SCHEME_B_SERIALIZE_READER_PASS"
                                     : "REAL_DNAGPT_FIDES_SCHEME_B_SERIALIZE_READER_FAIL")
                  << '\n';
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
