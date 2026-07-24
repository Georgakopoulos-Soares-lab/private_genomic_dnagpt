// Released-weight DNAGPT blocks 0 and 1 in one FIDESlib CKKS/CUDA lineage.
//
// Both blocks specialize T=2 causal attention to
//   w1 = sigmoid(s11 - s10); context = v0 + w1 * (v1 - v0)
// A public /20 condition, native single-context bootstrap, and public *20
// restoration separate the blocks. There is no client interaction or
// intermediate decryption at the refresh boundary.
//
// Security boundary: EncryptedEvaluator receives the crypto context, public
// model parameters, and ciphertexts. It never receives a private key. main()
// performs exactly one final Decrypt after both blocks are complete.

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
constexpr std::uint32_t MULT_DEPTH = 64;
constexpr std::uint32_t SCALE_BITS = 59;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 4;
constexpr double EPS = 1e-5;
constexpr double TOL = 4e-2;
constexpr std::array<std::uint32_t, 2> BOOTSTRAP_LEVEL_BUDGET = {4, 4};
constexpr double BLOCK0_REFRESH_BOUND = 20.0;
constexpr double ATTENTION_ZERO_GUARD = 0.25;
constexpr std::size_t PREDICTED_BLOCK0_MAX_LEVEL = 34;
constexpr std::size_t PREDICTED_REFRESH_CONDITIONED_MAX_LEVEL = 35;
constexpr std::size_t PREDICTED_REFRESH_OUTPUT_MAX_LEVEL = 17;
constexpr std::size_t PREDICTED_REFRESH_RESTORED_MAX_LEVEL = 18;
constexpr std::size_t PREDICTED_BLOCK1_MAX_LEVEL = 52;
constexpr std::size_t PREDICTED_PACKED_MAX_LEVEL = 53;

struct ApproximationContract {
    double ln1_lo;
    double ln1_hi;
    double attention_lo;
    double attention_hi;
    double ln2_lo;
    double ln2_hi;
    double gelu_lo;
    double gelu_hi;
    std::size_t degree_ln1;
    std::size_t degree_sigmoid;
    std::size_t degree_ln2;
    std::size_t degree_gelu;
    double ln1_public_scale;
    double ln2_public_scale;
};

// [A] Fixed public contracts from the immutable multiblock fixture and
// range_control_public_fixture_optimized_v2_PASS.json. No bound is selected
// from a private value during encrypted evaluation.
constexpr std::array<ApproximationContract, 2> BLOCK_CONTRACTS = {{
    {0.0030030699438575933, 0.010106134790696561,
     -6.069866743337161, ATTENTION_ZERO_GUARD,
     0.504245970267154, 1.289585701152818,
     -3.46661639687445, 3.46661639687445,
     7, 9, 7, 15, 0.00390625, 1.0},
    {0.7231809701265808, 1.5063129576985312,
     -8.981407641021296, ATTENTION_ZERO_GUARD,
     0.7487770832251114, 1.3440084965164667,
     -9.905612243716092, 9.905612243716092,
     7, 11, 7, 39, 1.0, 1.0},
}};

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
constexpr std::string_view PINNED_FIXTURE_MANIFEST =
    "3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c";
constexpr std::string_view PINNED_RANGE_CONTROL =
    "b148e30b42c430405a0a2c41401ab295701e0e1c3655ef00f06984ff9f8faba2";
constexpr std::string_view PINNED_CONTAINER_IMAGE =
    "dnagpt-fideslib:786c-asymfix2@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4";

using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;
using Cc = CryptoContext<DCRTPoly>;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
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
    std::string range_control_sha256;
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
    std::string source_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: dnagpt_two_block_fides --gpu N "
        "--fixture-dir PATH --output PATH --fixture-manifest-sha256 SHA "
        "--range-control-sha256 SHA "
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
        } else if (arg == "--range-control-sha256") {
            options.range_control_sha256 = next();
        } else if (arg == "--container-image") {
            options.container_image = next();
        } else if (arg == "--environment") {
            options.environment = next();
        } else if (arg == "--source-sha256") {
            options.source_sha256 = next();
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: dnagpt_two_block_fides --gpu N "
                         "--fixture-dir PATH --output PATH "
                         "--fixture-manifest-sha256 SHA "
                         "--range-control-sha256 SHA\n";
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
    if (options.range_control_sha256 != PINNED_RANGE_CONTROL) {
        usage_error("refusing unpinned range-control contract " +
                    options.range_control_sha256);
    }
    if (options.source_sha256 == "UNSPECIFIED" ||
        options.fixture_contract_sha256 == "UNSPECIFIED") {
        usage_error("source and fixture contract identities are required");
    }
    if (options.container_image != PINNED_CONTAINER_IMAGE) {
        usage_error("refusing unpinned container image " +
                    options.container_image);
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

struct BlockWeights {
    std::vector<double> ln1;
    std::array<Matrix, 3> qkv;
    Matrix attention_projection;
    std::vector<double> ln2;
    std::array<Matrix, 4> mlp_fc;
    std::array<Matrix, 4> mlp_projection;
};

struct Fixture {
    std::vector<double> input;
    std::array<BlockWeights, 2> blocks;
    std::vector<double> oracle_block1_output;
};

Fixture load_fixture(const std::filesystem::path& directory) {
    Fixture fixture;
    fixture.input = read_f64(directory / "input__embeddings.bin", T * D);
    for (std::size_t block = 0; block < fixture.blocks.size(); ++block) {
        BlockWeights& weights = fixture.blocks[block];
        const std::string prefix =
            "weights__block" + std::to_string(block) + "__";
        weights.ln1 = read_f64(directory / (prefix + "ln1.bin"), D);
        weights.ln2 = read_f64(directory / (prefix + "ln2.bin"), D);

        const Matrix qkv =
            load_matrix(directory / (prefix + "attn_qkv.bin"), 3 * D, D);
        for (std::size_t part = 0; part < 3; ++part) {
            weights.qkv[part] = rows(qkv, part * D, D);
        }
        weights.attention_projection =
            load_matrix(directory / (prefix + "attn_proj.bin"), D, D);

        const Matrix fc =
            load_matrix(directory / (prefix + "mlp_fc.bin"), MLP_DIM, D);
        for (std::size_t part = 0; part < 4; ++part) {
            weights.mlp_fc[part] = rows(fc, part * D, D);
        }
        const Matrix projection =
            load_matrix(directory / (prefix + "mlp_proj.bin"), D, MLP_DIM);
        for (std::size_t part = 0; part < 4; ++part) {
            weights.mlp_projection[part] =
                columns(projection, part * D, D);
        }
    }
    fixture.oracle_block1_output =
        read_f64(directory / "oracle__block1__block_output.bin", T * D);
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

std::vector<int> required_rotation_keys() {
    std::set<int> indices;
    for (std::size_t baby = 1; baby < BSGS_N1; ++baby) {
        indices.insert(static_cast<int>(baby));
    }
    for (std::size_t giant = 1; giant < BSGS_N2; ++giant) {
        indices.insert(static_cast<int>(BSGS_N1 * giant));
    }
    add_accumulate_keys(indices, static_cast<int>(PACK_WIDTH), 1);
    indices.erase(0);
    return {indices.begin(), indices.end()};
}

struct EvaluationResult {
    Ct packed_output;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels;
    std::size_t packed_level = 0;
    std::size_t refresh_input_max_level = 0;
    std::size_t refresh_conditioned_max_level = 0;
    std::size_t refresh_output_max_level = 0;
    std::size_t refresh_restored_max_level = 0;
    std::size_t bootstrap_boundaries = 1;
    std::size_t bootstrap_primitive_calls = T;
    std::size_t ciphertext_plain_matmuls = 0;
    std::size_t ciphertext_ciphertext_multiplications = 0;
    std::size_t ciphertext_plaintext_multiplications = 0;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture)
        : cc_(std::move(context)), fixture_(fixture) {
        std::function<double(double)> invsqrt =
            [](double value) { return 1.0 / std::sqrt(value); };
        std::function<double(double)> sigmoid = [](double value) {
            return 1.0 / (1.0 + std::exp(-value));
        };
        std::function<double(double)> gelu = [](double value) {
            constexpr double two_over_pi = 0.63661977236758134308;
            return 0.5 * value *
                   (1.0 + std::tanh(std::sqrt(two_over_pi) *
                                    (value + 0.044715 * value * value * value)));
        };
        for (std::size_t block = 0; block < BLOCK_CONTRACTS.size(); ++block) {
            const ApproximationContract& contract = BLOCK_CONTRACTS[block];
            coefficients_[block].ln1 = cc_->GetChebyshevCoefficients(
                invsqrt, contract.ln1_lo / contract.ln1_public_scale,
                contract.ln1_hi / contract.ln1_public_scale,
                contract.degree_ln1);
            coefficients_[block].ln2 = cc_->GetChebyshevCoefficients(
                invsqrt, contract.ln2_lo / contract.ln2_public_scale,
                contract.ln2_hi / contract.ln2_public_scale,
                contract.degree_ln2);
            coefficients_[block].sigmoid = cc_->GetChebyshevCoefficients(
                sigmoid, contract.attention_lo, contract.attention_hi,
                contract.degree_sigmoid);
            coefficients_[block].gelu = cc_->GetChebyshevCoefficients(
                gelu, contract.gelu_lo, contract.gelu_hi,
                contract.degree_gelu);
        }
    }

    EvaluationResult evaluate(const std::array<Ct, T>& encrypted_inputs) {
        levels_.clear();
        matmul_count_ = 0;
        ct_ct_multiply_count_ = 0;
        ct_plain_multiply_count_ = 0;

        const std::array<Ct, T> block0 = block(encrypted_inputs, 0);
        const std::size_t refresh_input = maximum_level(block0);
        std::array<Ct, T> conditioned{};
        std::array<Ct, T> refreshed{};
        std::array<Ct, T> restored{};
        const Plaintext condition = repeated_plain(
            std::vector<double>(D, 1.0 / BLOCK0_REFRESH_BOUND));
        const Plaintext restore =
            repeated_plain(std::vector<double>(D, BLOCK0_REFRESH_BOUND));
        for (std::size_t token = 0; token < T; ++token) {
            conditioned[token] = multiply_plain(block0[token], condition);
            refreshed[token] =
                cc_->EvalBootstrap(conditioned[token], 1, 0, false);
            restored[token] = multiply_plain(refreshed[token], restore);
        }
        cc_->Synchronize();
        record("refresh.conditioned", conditioned);
        record("refresh.refreshed", refreshed);
        record("refresh.restored", restored);
        const std::size_t refresh_output = maximum_level(refreshed);
        const std::size_t refresh_conditioned = maximum_level(conditioned);
        if (refresh_output >= refresh_conditioned) {
            throw std::runtime_error(
                "bootstrap did not restore multiplicative depth");
        }
        require_remaining_depth(maximum_level(restored), 35,
                                "released block1 graph plus final packing");

        const std::array<Ct, T> block1 = block(restored, 1);
        require_remaining_depth(maximum_level(block1), 1,
                                "two-block final output packing");
        return finish(block1, refresh_input, refresh_conditioned,
                      refresh_output, maximum_level(restored));
    }

  private:
    struct PolynomialCoefficients {
        std::vector<double> ln1;
        std::vector<double> sigmoid;
        std::vector<double> ln2;
        std::vector<double> gelu;
    };

    struct BabyRotations {
        std::array<Ct, BSGS_N1> values{};
    };

    std::array<Ct, T> block(const std::array<Ct, T>& encrypted_inputs,
                            std::size_t block_index) {
        if (block_index >= fixture_.blocks.size()) {
            throw std::invalid_argument("block index outside two-block fixture");
        }
        const BlockWeights& weights = fixture_.blocks[block_index];
        const ApproximationContract& contract =
            BLOCK_CONTRACTS[block_index];
        PolynomialCoefficients& coefficients = coefficients_[block_index];
        const std::string prefix = "block" + std::to_string(block_index) + ".";

        std::array<Ct, T> normalized{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized[token] =
                layernorm(encrypted_inputs[token], weights.ln1,
                          coefficients.ln1, contract.ln1_lo,
                          contract.ln1_hi, contract.ln1_public_scale);
        }
        record(prefix + "ln1", normalized);

        std::array<Ct, T> query{};
        std::array<Ct, T> key{};
        std::array<Ct, T> value{};
        for (std::size_t token = 0; token < T; ++token) {
            const BabyRotations baby = baby_rotations(normalized[token]);
            query[token] = matmul(baby, weights.qkv[0]);
            key[token] = matmul(baby, weights.qkv[1]);
            value[token] = matmul(baby, weights.qkv[2]);
        }
        record(prefix + "qkv", query);

        std::array<std::array<Ct, T>, HEADS> value_heads{};
        for (std::size_t head = 0; head < HEADS; ++head) {
            std::vector<double> head_mask(D);
            for (std::size_t dim = head * HEAD_DIM;
                 dim < (head + 1) * HEAD_DIM; ++dim) {
                head_mask[dim] = 1.0;
            }
            const Plaintext mask = repeated_plain(head_mask);
            for (std::size_t token = 0; token < T; ++token) {
                value_heads[head][token] = multiply_plain(value[token], mask);
            }
        }

        std::array<Ct, T> context{};
        const double score_scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
        for (std::size_t head = 0; head < HEADS; ++head) {
            std::vector<double> head_mask(D);
            for (std::size_t dim = head * HEAD_DIM;
                 dim < (head + 1) * HEAD_DIM; ++dim) {
                head_mask[dim] = 1.0;
            }

            // The first causal row has one source, so softmax is exactly one.
            context[0] =
                head == 0 ? value_heads[head][0]
                          : cc_->EvalAdd(context[0], value_heads[head][0]);

            const Ct score_10 =
                attention_score(query[1], key[0], head_mask, score_scale);
            const Ct score_11 =
                attention_score(query[1], key[1], head_mask, score_scale);
            const Ct delta = cc_->EvalSub(score_11, score_10);
            Ct weight_1 = cc_->EvalChebyshevSeries(
                delta, coefficients.sigmoid, contract.attention_lo,
                contract.attention_hi);

            // Exact T=2 identity:
            //   w1 = sigmoid(s11 - s10)
            //   context = v0 + w1 * (v1 - v0)
            // The deeper sigmoid operand is deliberately first in EvalMult.
            const Ct value_delta =
                cc_->EvalSub(value_heads[head][1], value_heads[head][0]);
            const Ct weighted_delta = multiply(weight_1, value_delta);
            const Ct head_context =
                cc_->EvalAdd(value_heads[head][0], weighted_delta);
            context[1] =
                head == 0 ? head_context : cc_->EvalAdd(context[1], head_context);
        }
        record(prefix + "attention_context", context);

        std::array<Ct, T> attention_projection{};
        for (std::size_t token = 0; token < T; ++token) {
            attention_projection[token] =
                matmul(baby_rotations(context[token]),
                       weights.attention_projection);
        }
        record(prefix + "attention_projection", attention_projection);

        std::array<Ct, T> residual1{};
        for (std::size_t token = 0; token < T; ++token) {
            residual1[token] =
                cc_->EvalAdd(encrypted_inputs[token], attention_projection[token]);
        }
        record(prefix + "residual1", residual1);

        std::array<Ct, T> normalized2{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized2[token] =
                layernorm(residual1[token], weights.ln2, coefficients.ln2,
                          contract.ln2_lo, contract.ln2_hi,
                          contract.ln2_public_scale);
        }
        record(prefix + "ln2", normalized2);

        std::array<Ct, T> block_output{};
        for (std::size_t token = 0; token < T; ++token) {
            require_remaining_depth(
                normalized2[token]->GetLevel(),
                block_index == 0 ? 8 : 10,
                prefix + "MLP token " + std::to_string(token));
            const BabyRotations fc_baby = baby_rotations(normalized2[token]);
            std::array<Ct, 4> hidden{};
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                hidden[chunk] = matmul(fc_baby, weights.mlp_fc[chunk]);
                hidden[chunk] = cc_->EvalChebyshevSeries(
                    hidden[chunk], coefficients.gelu, contract.gelu_lo,
                    contract.gelu_hi);
            }

            Ct mlp;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                Ct contribution =
                    matmul(baby_rotations(hidden[chunk]),
                           weights.mlp_projection[chunk]);
                mlp = chunk == 0 ? contribution
                                 : cc_->EvalAdd(mlp, contribution);
            }
            block_output[token] = cc_->EvalAdd(residual1[token], mlp);
            std::cout << "[stage] " << prefix << "block_output token=" << token
                      << " level=" << block_output[token]->GetLevel() << '\n';
        }
        record(prefix + "block_output", block_output);
        return block_output;
    }

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
                 std::vector<double>& coefficients, double domain_lo,
                 double domain_hi, double public_scale) {
        Ct mean = mean_broadcast(input);
        Ct centered = cc_->EvalSub(input, mean);
        Ct variance = mean_broadcast(multiply(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        Ct normalized_variance = variance;
        if (public_scale != 1.0) {
            normalized_variance = multiply_plain(
                variance,
                repeated_plain(std::vector<double>(D, 1.0 / public_scale)));
        }
        Ct inverse = cc_->EvalChebyshevSeries(
            normalized_variance, coefficients, domain_lo / public_scale,
            domain_hi / public_scale);
        if (public_scale != 1.0) {
            inverse = multiply_plain(
                inverse,
                repeated_plain(
                    std::vector<double>(D, 1.0 / std::sqrt(public_scale))));
        }
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

    EvaluationResult finish(const std::array<Ct, T>& values,
                            std::size_t refresh_input,
                            std::size_t refresh_conditioned,
                            std::size_t refresh_output,
                            std::size_t refresh_restored) {
        require_remaining_depth(maximum_level(values), 1,
                                "two-block output packing");
        Ct packed;
        for (std::size_t token = 0; token < T; ++token) {
            Ct selected =
                multiply_plain(values[token], output_block_plain(token));
            packed = token == 0 ? selected : cc_->EvalAdd(packed, selected);
        }
        return {.packed_output = packed,
                .levels = levels_,
                .packed_level = packed->GetLevel(),
                .refresh_input_max_level = refresh_input,
                .refresh_conditioned_max_level = refresh_conditioned,
                .refresh_output_max_level = refresh_output,
                .refresh_restored_max_level = refresh_restored,
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
        std::cout << "[stage] " << name << " level_min=" << minimum
                  << " level_max=" << maximum << '\n';
    }

    Cc cc_;
    const Fixture& fixture_;
    std::array<PolynomialCoefficients, 2> coefficients_;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels_;
    std::size_t matmul_count_ = 0;
    std::size_t ct_ct_multiply_count_ = 0;
    std::size_t ct_plain_multiply_count_ = 0;
};

struct Metrics {
    double global_rel_inf = std::numeric_limits<double>::infinity();
    double worst_token_rel_inf = std::numeric_limits<double>::infinity();
    double max_abs_error = std::numeric_limits<double>::infinity();
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
                      std::size_t rotation_keys,
                      const EvaluationResult& evaluation, const Metrics& metrics,
                      double fixture_seconds, double setup_seconds,
                      double encryption_seconds, double evaluation_seconds,
                      double decrypt_seconds) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"FHE released-weight DNAGPT blocks 0 and 1, one encrypted lineage with block-boundary refresh\",\n"
        << "  \"implementation_version\": \"d768-t2-blocks0-1-sigmoid-refresh-v1\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
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
        << "  \"range_control_sha256\": \""
        << json_escape(options.range_control_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"model\": \"dna_gpt0.1b_m classification checkpoint blocks 0 and 1\",\n"
        << "    \"released_weight_blocks\": [0, 1],\n"
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
        << "    \"bootstrap_boundaries\": "
        << evaluation.bootstrap_boundaries << ",\n"
        << "    \"bootstrap_primitive_calls\": "
        << evaluation.bootstrap_primitive_calls << ",\n"
        << "    \"bootstrap_level_budget\": ["
        << BOOTSTRAP_LEVEL_BUDGET[0] << ", "
        << BOOTSTRAP_LEVEL_BUDGET[1] << "],\n"
        << "    \"block0_refresh_public_bound\": "
        << BLOCK0_REFRESH_BOUND << ",\n"
        << "    \"refresh_schedule\": \"public /20; native same-context bootstrap per token ciphertext; public *20\",\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"bsgs\": {\"n1\": " << BSGS_N1 << ", \"n2\": "
        << BSGS_N2 << "},\n"
        << "    \"matrix_products\": " << evaluation.ciphertext_plain_matmuls
        << ",\n"
        << "    \"operation_counts\": {\n"
        << "      \"ciphertext_plaintext_matrix_products\": "
        << evaluation.ciphertext_plain_matmuls << ",\n"
        << "      \"explicit_ciphertext_ciphertext_multiplications\": "
        << evaluation.ciphertext_ciphertext_multiplications << ",\n"
        << "      \"explicit_ciphertext_plaintext_multiplications\": "
        << evaluation.ciphertext_plaintext_multiplications << "\n"
        << "    },\n"
        << "    \"attention_schedule\": \"both blocks use exact T=2 identity: w1=sigmoid(s11-s10); context=v0+w1*(v1-v0); deeper sigmoid operand first\",\n"
        << "    \"public_domain_contract\": {\n"
        << "      \"status\": \"[A] immutable public multiblock fixture/range-control assumptions; no private-query adaptation\",\n"
        << "      \"block0\": {\"ln1\": ["
        << BLOCK_CONTRACTS[0].ln1_lo << ", " << BLOCK_CONTRACTS[0].ln1_hi
        << "], \"attention_delta\": ["
        << BLOCK_CONTRACTS[0].attention_lo << ", "
        << BLOCK_CONTRACTS[0].attention_hi << "], \"ln2\": ["
        << BLOCK_CONTRACTS[0].ln2_lo << ", " << BLOCK_CONTRACTS[0].ln2_hi
        << "], \"gelu\": [" << BLOCK_CONTRACTS[0].gelu_lo << ", "
        << BLOCK_CONTRACTS[0].gelu_hi
        << "], \"ln1_public_power_of_two_scale\": "
        << BLOCK_CONTRACTS[0].ln1_public_scale
        << ", \"ln2_public_power_of_two_scale\": "
        << BLOCK_CONTRACTS[0].ln2_public_scale << "},\n"
        << "      \"block1\": {\"ln1\": ["
        << BLOCK_CONTRACTS[1].ln1_lo << ", " << BLOCK_CONTRACTS[1].ln1_hi
        << "], \"attention_delta\": ["
        << BLOCK_CONTRACTS[1].attention_lo << ", "
        << BLOCK_CONTRACTS[1].attention_hi << "], \"ln2\": ["
        << BLOCK_CONTRACTS[1].ln2_lo << ", " << BLOCK_CONTRACTS[1].ln2_hi
        << "], \"gelu\": [" << BLOCK_CONTRACTS[1].gelu_lo << ", "
        << BLOCK_CONTRACTS[1].gelu_hi
        << "], \"ln1_public_power_of_two_scale\": "
        << BLOCK_CONTRACTS[1].ln1_public_scale
        << ", \"ln2_public_power_of_two_scale\": "
        << BLOCK_CONTRACTS[1].ln2_public_scale << "}\n"
        << "    },\n"
        << "    \"chebyshev_degrees\": {\n"
        << "      \"block0\": {\"ln1_invsqrt\": "
        << BLOCK_CONTRACTS[0].degree_ln1
        << ", \"attention_sigmoid\": "
        << BLOCK_CONTRACTS[0].degree_sigmoid
        << ", \"ln2_invsqrt\": " << BLOCK_CONTRACTS[0].degree_ln2
        << ", \"gelu\": " << BLOCK_CONTRACTS[0].degree_gelu << "},\n"
        << "      \"block1\": {\"ln1_invsqrt\": "
        << BLOCK_CONTRACTS[1].degree_ln1
        << ", \"attention_sigmoid\": "
        << BLOCK_CONTRACTS[1].degree_sigmoid
        << ", \"ln2_invsqrt\": " << BLOCK_CONTRACTS[1].degree_ln2
        << ", \"gelu\": " << BLOCK_CONTRACTS[1].degree_gelu << "}\n"
        << "    },\n"
        << "    \"depth_guards\": {\"block0_mlp_required_levels\": 8, "
           "\"block1_full_graph_and_pack_required_levels\": 35, "
           "\"block1_mlp_required_levels\": 10, "
           "\"final_pack_required_levels\": 1, "
           "\"bootstrap_must_restore_depth\": true, \"fail_closed\": true},\n"
        << "    \"predicted_level_maxima\": {\"block0_output\": "
        << PREDICTED_BLOCK0_MAX_LEVEL
        << ", \"refresh_conditioned\": "
        << PREDICTED_REFRESH_CONDITIONED_MAX_LEVEL
        << ", \"refresh_output\": "
        << PREDICTED_REFRESH_OUTPUT_MAX_LEVEL
        << ", \"refresh_restored\": "
        << PREDICTED_REFRESH_RESTORED_MAX_LEVEL
        << ", \"block1_output\": " << PREDICTED_BLOCK1_MAX_LEVEL
        << ", \"packed_output\": " << PREDICTED_PACKED_MAX_LEVEL
        << ", \"status\": \"[A] static schedule prediction; actual trace is authoritative\"},\n"
        << "    \"packed_output_level\": " << evaluation.packed_level << ",\n"
        << "    \"refresh_levels\": {\"input_max\": "
        << evaluation.refresh_input_max_level
        << ", \"conditioned_max\": "
        << evaluation.refresh_conditioned_max_level
        << ", \"after_bootstrap_max\": "
        << evaluation.refresh_output_max_level
        << ", \"restored_max\": "
        << evaluation.refresh_restored_max_level << "},\n"
        << "    \"level_trace\": {\n";
    std::size_t stage = 0;
    for (const auto& [name, range] : evaluation.levels) {
        out << "      \"" << json_escape(name) << "\": {\"min\": "
            << range.first << ", \"max\": " << range.second << "}";
        out << (++stage == evaluation.levels.size() ? "\n" : ",\n");
    }
    out << "    }\n"
        << "  },\n"
        << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << "  \"worst_token_rel_inf\": " << metrics.worst_token_rel_inf
        << ",\n"
        << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << "  \"all_finite\": " << (metrics.all_finite ? "true" : "false")
        << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (metrics.passed ? "true" : "false") << ",\n"
        << "  \"timings_seconds\": {\n"
        << "    \"load_verified_fixture\": " << fixture_seconds << ",\n"
        << "    \"context_keygen_load\": " << setup_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds << ",\n"
        << "    \"encrypted_evaluation\": " << evaluation_seconds << ",\n"
        << "    \"final_decrypt\": " << decrypt_seconds << "\n"
        << "  },\n"
        << "  \"input_boundary\": \"token plus position embeddings encrypted by client\",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"final_decrypt_calls\": 1,\n"
        << "  \"evaluator_has_private_key\": false,\n"
        << "  \"same_crypto_context_and_key_lineage\": true,\n"
        << "  \"client_round_trips\": 0\n"
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
        parameters.SetDevices({options.gpu});
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);

        Cc cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
        cc->EvalBootstrapSetup(
            {BOOTSTRAP_LEVEL_BUDGET[0], BOOTSTRAP_LEVEL_BUDGET[1]},
            {0, 0}, SLOTS, 0);

        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        cc->EvalRotateKeyGen(keys.secretKey, rotation_keys);
        cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS);
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

        // Deliberately construct the evaluator without keys. This is the
        // server-side no-private-key/no-decrypt boundary.
        EncryptedEvaluator evaluator(cc, fixture);
        const auto evaluation_start = Clock::now();
        EvaluationResult evaluation = evaluator.evaluate(encrypted_inputs);
        cc->Synchronize();
        const double evaluation_seconds = elapsed_seconds(evaluation_start);

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
            measure(output, fixture.oracle_block1_output);
        const std::string evidence =
            make_json(options, ring, rotation_keys.size(), evaluation, metrics,
                      fixture_seconds, setup_seconds, encryption_seconds,
                      evaluation_seconds, decrypt_seconds);
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout << (metrics.passed
                          ? "DNAGPT_TWO_BLOCK_FIDES_GATE_PASS"
                          : "DNAGPT_TWO_BLOCK_FIDES_GATE_FAIL")
                  << '\n';
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
