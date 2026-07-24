// End-to-end encrypted toy DNAGPT block on the public FIDESlib 2.1.3 API.
//
// Security boundary: EncryptedEvaluator receives a CryptoContext and public model
// parameters only. It never receives a private key. The only private-key operation
// is the final packed-output readout in main.

#include "toy_fixture.hpp"

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

using dnagpt::toy::Fixture;
using dnagpt::toy::Matrix;
using dnagpt::toy::Tokens;
using dnagpt::toy::Vec;

constexpr std::size_t D = dnagpt::toy::kD;
constexpr std::size_t T = dnagpt::toy::kT;
constexpr std::size_t HEADS = dnagpt::toy::kHeads;
constexpr std::size_t HEAD_DIM = dnagpt::toy::kHeadDim;
constexpr std::size_t MLP_DIM = dnagpt::toy::kMlp;
constexpr std::size_t SLOTS = D * T;
constexpr std::uint32_t MULT_DEPTH = 49;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 4;
constexpr double TOL = 4e-2;
constexpr double INVSQRT_LO = dnagpt::toy::kInvsqrtLow;
constexpr double INVSQRT_HI = dnagpt::toy::kInvsqrtHigh;
constexpr double EXP_LO = dnagpt::toy::kExpLow;
constexpr double EXP_HI = dnagpt::toy::kExpHigh;
constexpr double RECIP_LO = dnagpt::toy::kReciprocalLow;
constexpr double RECIP_HI = dnagpt::toy::kReciprocalHigh;
constexpr double GELU_LO = dnagpt::toy::kGeluLow;
constexpr double GELU_HI = dnagpt::toy::kGeluHigh;
constexpr std::size_t DEG_INVSQRT = 27;
constexpr std::size_t DEG_EXP = 13;
constexpr std::size_t DEG_RECIP = 27;
constexpr std::size_t DEG_GELU = 7;
constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";

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
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
    std::string source_sha256 = "UNSPECIFIED";
    std::string fixture_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: toy_dnagpt_fides --gpu N --output PATH "
        "[--backend-commit SHA] [--container-image NAME] [--environment TEXT] "
        "[--source-sha256 SHA] [--fixture-sha256 SHA]");
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
        } else if (arg == "--output") {
            options.output = next();
        } else if (arg == "--backend-commit") {
            options.backend_commit = next();
        } else if (arg == "--container-image") {
            options.container_image = next();
        } else if (arg == "--environment") {
            options.environment = next();
        } else if (arg == "--source-sha256") {
            options.source_sha256 = next();
        } else if (arg == "--fixture-sha256") {
            options.fixture_sha256 = next();
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: toy_dnagpt_fides --gpu N --output PATH "
                   "[--backend-commit SHA] [--container-image NAME] [--environment TEXT] "
                   "[--source-sha256 SHA] [--fixture-sha256 SHA]\n";
            std::exit(0);
        } else {
            usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0) {
        usage_error("--gpu must be non-negative");
    }
    if (options.output.empty()) {
        usage_error("--output is required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit " + options.backend_commit);
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
                                 options.output.string());
    }
    return options;
}

Matrix rows(const Matrix& input, std::size_t first, std::size_t count) {
    if (first + count > input.rows) {
        throw std::invalid_argument("row slice outside matrix");
    }
    Matrix output(count, input.cols);
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t col = 0; col < input.cols; ++col) {
            output(row, col) = input(first + row, col);
        }
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
            output(row, col) = input(row, first + col);
        }
    }
    return output;
}

std::vector<double> slice(const std::vector<double>& input, std::size_t first,
                          std::size_t count) {
    if (first + count > input.size()) {
        throw std::invalid_argument("vector slice outside input");
    }
    return {input.begin() + static_cast<std::ptrdiff_t>(first),
            input.begin() + static_cast<std::ptrdiff_t>(first + count)};
}

std::vector<double> zeros(std::size_t count) {
    return std::vector<double>(count, 0.0);
}

struct EvaluationResult {
    Ct packed_output;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels;
    std::size_t packed_level = 0;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture)
        : cc_(std::move(context)), fixture_(fixture) {
        std::function<double(double)> invsqrt =
            [](double value) { return 1.0 / std::sqrt(value); };
        std::function<double(double)> exponential =
            [](double value) { return std::exp(value); };
        std::function<double(double)> reciprocal =
            [](double value) { return 1.0 / value; };
        std::function<double(double)> gelu =
            [](double value) { return dnagpt::toy::gelu_tanh(value); };
        invsqrt_coefficients_ =
            cc_->GetChebyshevCoefficients(invsqrt, INVSQRT_LO, INVSQRT_HI, DEG_INVSQRT);
        exp_coefficients_ =
            cc_->GetChebyshevCoefficients(exponential, EXP_LO, EXP_HI, DEG_EXP);
        reciprocal_coefficients_ =
            cc_->GetChebyshevCoefficients(reciprocal, RECIP_LO, RECIP_HI, DEG_RECIP);
        gelu_coefficients_ =
            cc_->GetChebyshevCoefficients(gelu, GELU_LO, GELU_HI, DEG_GELU);
    }

    EvaluationResult evaluate(const std::array<Ct, T>& encrypted_inputs) {
        levels_.clear();
        std::array<Ct, T> normalized{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized[token] =
                layernorm(encrypted_inputs[token], fixture_.ln1_weight, fixture_.ln1_bias);
        }
        record("ln1", normalized);

        const Matrix wq = rows(fixture_.attention_weight, 0, D);
        const Matrix wk = rows(fixture_.attention_weight, D, D);
        const Matrix wv = rows(fixture_.attention_weight, 2 * D, D);
        const auto bq = slice(fixture_.attention_bias, 0, D);
        const auto bk = slice(fixture_.attention_bias, D, D);
        const auto bv = slice(fixture_.attention_bias, 2 * D, D);

        std::array<Ct, T> query{};
        std::array<Ct, T> key{};
        std::array<Ct, T> value{};
        for (std::size_t token = 0; token < T; ++token) {
            query[token] = matmul(normalized[token], wq, bq);
            key[token] = matmul(normalized[token], wk, bk);
            value[token] = matmul(normalized[token], wv, bv);
        }
        record("qkv", query);

        std::array<std::array<Ct, T>, HEADS> value_heads{};
        for (std::size_t head = 0; head < HEADS; ++head) {
            std::vector<double> mask(D);
            for (std::size_t dim = 0; dim < D; ++dim) {
                mask[dim] =
                    (dim >= head * HEAD_DIM && dim < (head + 1) * HEAD_DIM) ? 1.0 : 0.0;
            }
            for (std::size_t token = 0; token < T; ++token) {
                value_heads[head][token] = multiply_plain(value[token], repeated_plain(mask));
            }
        }

        std::array<Ct, T> attention{};
        const double score_scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
        for (std::size_t token = 0; token < T; ++token) {
            Ct joined_heads;
            for (std::size_t head = 0; head < HEADS; ++head) {
                std::vector<double> head_mask(D);
                for (std::size_t dim = 0; dim < D; ++dim) {
                    head_mask[dim] =
                        (dim >= head * HEAD_DIM && dim < (head + 1) * HEAD_DIM) ? 1.0 : 0.0;
                }

                Ct score_row;
                for (std::size_t source = 0; source <= token; ++source) {
                    auto products = multiply(query[token], key[source]);
                    products = multiply_plain(products, repeated_plain(head_mask));
                    auto score = sum_broadcast(products);
                    std::vector<double> score_slot(D);
                    score_slot[source] = score_scale;
                    score = multiply_plain(score, repeated_plain(score_slot));
                    score_row = source == 0 ? score : cc_->EvalAdd(score_row, score);
                }

                auto exponentials =
                    cc_->EvalChebyshevSeries(score_row, exp_coefficients_, EXP_LO, EXP_HI);
                std::vector<double> causal_mask(D);
                for (std::size_t source = 0; source <= token; ++source) {
                    causal_mask[source] = 1.0;
                }
                exponentials =
                    multiply_plain(exponentials, repeated_plain(causal_mask));
                auto denominator = sum_broadcast(exponentials);
                auto inverse = cc_->EvalChebyshevSeries(
                    denominator, reciprocal_coefficients_, RECIP_LO, RECIP_HI);

                // Numerator-first attention: avoid materializing encrypted softmax
                // weights. This is algebraically identical and saves one level.
                Ct numerator;
                for (std::size_t source = 0; source <= token; ++source) {
                    auto scalar = select_broadcast(exponentials, source);
                    auto term = multiply(scalar, value_heads[head][source]);
                    numerator = source == 0 ? term : cc_->EvalAdd(numerator, term);
                }
                auto head_output = multiply(numerator, inverse);
                joined_heads =
                    head == 0 ? head_output : cc_->EvalAdd(joined_heads, head_output);
            }
            attention[token] =
                matmul(joined_heads, fixture_.projection_weight,
                       dnagpt::toy::to_vector(fixture_.projection_bias));
            std::cout << "[stage] attention token=" << token
                      << " level=" << attention[token]->GetLevel() << '\n';
        }
        record("attention_projection", attention);

        std::array<Ct, T> residual1{};
        for (std::size_t token = 0; token < T; ++token) {
            residual1[token] = cc_->EvalAdd(encrypted_inputs[token], attention[token]);
        }
        record("residual1", residual1);

        std::array<Ct, T> normalized2{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized2[token] =
                layernorm(residual1[token], fixture_.ln2_weight, fixture_.ln2_bias);
        }
        record("ln2", normalized2);

        std::array<Ct, T> output_blocks{};
        for (std::size_t token = 0; token < T; ++token) {
            std::array<Ct, 4> hidden_chunks{};
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                const auto weight = rows(fixture_.fc_weight, chunk * D, D);
                const auto bias = slice(fixture_.fc_bias, chunk * D, D);
                auto hidden = matmul(normalized2[token], weight, bias);
                hidden =
                    cc_->EvalChebyshevSeries(hidden, gelu_coefficients_, GELU_LO, GELU_HI);
                hidden_chunks[chunk] = hidden;
            }

            Ct mlp;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                const auto weight = columns(fixture_.fcp_weight, chunk * D, D);
                const auto bias =
                    chunk == 0 ? dnagpt::toy::to_vector(fixture_.fcp_bias) : zeros(D);
                auto contribution =
                    matmul_to_output_block(hidden_chunks[chunk], weight, bias, token);
                mlp = chunk == 0 ? contribution : cc_->EvalAdd(mlp, contribution);
            }
            auto residual =
                multiply_plain(residual1[token], output_block_plain(std::vector<double>(D, 1.0),
                                                                    token));
            output_blocks[token] = cc_->EvalAdd(residual, mlp);
            std::cout << "[stage] mlp token=" << token
                      << " level=" << output_blocks[token]->GetLevel() << '\n';
        }
        record("block_output", output_blocks);

        Ct packed = output_blocks[0];
        for (std::size_t token = 1; token < T; ++token) {
            packed = cc_->EvalAdd(packed, output_blocks[token]);
        }
        return {.packed_output = packed,
                .levels = levels_,
                .packed_level = packed->GetLevel()};
    }

  private:
    Plaintext raw_plain(const std::vector<double>& values) {
        if (values.size() != SLOTS) {
            throw std::invalid_argument("raw plaintext must have T*D values");
        }
        return cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
    }

    Plaintext repeated_plain(const std::vector<double>& values) {
        if (values.size() != D) {
            throw std::invalid_argument("repeated plaintext must have D values");
        }
        std::vector<double> packed;
        packed.reserve(SLOTS);
        for (std::size_t token = 0; token < T; ++token) {
            packed.insert(packed.end(), values.begin(), values.end());
        }
        return raw_plain(packed);
    }

    Plaintext repeated_plain(const Vec& values) {
        return repeated_plain(dnagpt::toy::to_vector(values));
    }

    Plaintext output_block_plain(const std::vector<double>& values, std::size_t token) {
        if (values.size() != D || token >= T) {
            throw std::invalid_argument("invalid output-block plaintext");
        }
        std::vector<double> packed(SLOTS);
        std::copy(values.begin(), values.end(),
                  packed.begin() + static_cast<std::ptrdiff_t>(token * D));
        return raw_plain(packed);
    }

    Ct multiply(const Ct& left, const Ct& right) {
        return cc_->EvalMult(left, right);
    }

    Ct multiply_plain(const Ct& input, Plaintext plaintext) {
        return cc_->EvalMult(input, plaintext);
    }

    Ct sum_broadcast(const Ct& input) {
        return cc_->AccumulateSum(input, static_cast<int>(D), 1);
    }

    Ct mean_broadcast(const Ct& input) {
        auto sum = sum_broadcast(input);
        return multiply_plain(sum,
                              repeated_plain(std::vector<double>(D, 1.0 / static_cast<double>(D))));
    }

    Ct select_broadcast(const Ct& input, std::size_t slot) {
        std::vector<double> one_hot(D);
        one_hot.at(slot) = 1.0;
        return sum_broadcast(multiply_plain(input, repeated_plain(one_hot)));
    }

    Ct layernorm(const Ct& input, const Vec& weight, const Vec& bias) {
        auto mean = mean_broadcast(input);
        auto centered = cc_->EvalSub(input, mean);
        auto variance = mean_broadcast(multiply(centered, centered));
        variance = cc_->EvalAdd(variance, dnagpt::toy::kEps);
        auto inverse = cc_->EvalChebyshevSeries(
            variance, invsqrt_coefficients_, INVSQRT_LO, INVSQRT_HI);
        auto normalized = multiply(centered, inverse);
        normalized = multiply_plain(normalized, repeated_plain(weight));
        auto bias_plain = repeated_plain(bias);
        return cc_->EvalAdd(normalized, bias_plain);
    }

    Ct matmul(const Ct& input, const Matrix& weight, const std::vector<double>& bias) {
        if (weight.rows != D || weight.cols != D || bias.size() != D) {
            throw std::invalid_argument("toy matmul expects D by D");
        }
        constexpr std::size_t n1 = 2;
        constexpr std::size_t n2 = D / n1;
        const std::vector<std::int32_t> baby_indices = {1};
        // On the pinned GPU API the precomputation handle is intentionally null;
        // the vector overload performs native hoisted rotations.
        auto baby_rotations = cc_->EvalFastRotation(
            input, baby_indices, cc_->GetCyclotomicOrder(), nullptr);
        std::array<Ct, n1> baby = {input, baby_rotations.at(0)};

        Ct result;
        for (std::size_t giant = 0; giant < n2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < n1; ++small) {
                const std::size_t diagonal = n1 * giant + small;
                std::vector<double> values(D);
                for (std::size_t row = 0; row < D; ++row) {
                    const std::size_t rolled_row = (row + D - n1 * giant) % D;
                    values[row] =
                        weight(rolled_row, (rolled_row + diagonal) % D);
                }
                auto term = multiply_plain(baby[small], repeated_plain(values));
                inner = small == 0 ? term : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(inner, static_cast<std::int32_t>(n1 * giant));
            }
            result = giant == 0 ? inner : cc_->EvalAdd(result, inner);
        }
        auto bias_plain = repeated_plain(bias);
        return cc_->EvalAdd(result, bias_plain);
    }

    Ct matmul_to_output_block(const Ct& input, const Matrix& weight,
                              const std::vector<double>& bias, std::size_t token) {
        if (weight.rows != D || weight.cols != D || bias.size() != D) {
            throw std::invalid_argument("toy packed matmul expects D by D");
        }
        const std::vector<std::int32_t> indices = {1, 2, 3, 4, 5, 6, 7};
        auto rotations =
            cc_->EvalFastRotation(input, indices, cc_->GetCyclotomicOrder(), nullptr);
        Ct result;
        for (std::size_t diagonal = 0; diagonal < D; ++diagonal) {
            std::vector<double> values(D);
            for (std::size_t row = 0; row < D; ++row) {
                values[row] = weight(row, (row + diagonal) % D);
            }
            const Ct& rotated = diagonal == 0 ? input : rotations.at(diagonal - 1);
            auto term =
                multiply_plain(rotated, output_block_plain(values, token));
            result = diagonal == 0 ? term : cc_->EvalAdd(result, term);
        }
        auto bias_plain = output_block_plain(bias, token);
        return cc_->EvalAdd(result, bias_plain);
    }

    template <typename Container>
    void record(const std::string& name, const Container& ciphertexts) {
        std::size_t minimum = std::numeric_limits<std::size_t>::max();
        std::size_t maximum = 0;
        for (const auto& ciphertext : ciphertexts) {
            minimum = std::min(minimum, ciphertext->GetLevel());
            maximum = std::max(maximum, ciphertext->GetLevel());
        }
        levels_[name] = {minimum, maximum};
        std::cout << "[stage] " << name << " level_min=" << minimum
                  << " level_max=" << maximum << '\n';
    }

    Cc cc_;
    const Fixture& fixture_;
    std::vector<double> invsqrt_coefficients_;
    std::vector<double> exp_coefficients_;
    std::vector<double> reciprocal_coefficients_;
    std::vector<double> gelu_coefficients_;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels_;
};

struct Metrics {
    double global_rel_inf = std::numeric_limits<double>::infinity();
    double worst_token_rel_inf = std::numeric_limits<double>::infinity();
    double max_abs_error = std::numeric_limits<double>::infinity();
    bool all_finite = false;
    bool passed = false;
};

Metrics measure(const std::vector<double>& got, const Tokens& reference) {
    if (got.size() < SLOTS) {
        throw std::invalid_argument("decrypted vector is shorter than packed output");
    }
    Metrics metrics;
    metrics.max_abs_error = 0.0;
    double global_denominator = 0.0;
    metrics.worst_token_rel_inf = 0.0;
    metrics.all_finite = true;
    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t dim = 0; dim < D; ++dim) {
            const double actual = got[token * D + dim];
            const double expected = reference[token][dim];
            metrics.all_finite = metrics.all_finite && std::isfinite(actual);
            const double error = std::abs(actual - expected);
            token_error = std::max(token_error, error);
            token_denominator = std::max(token_denominator, std::abs(expected));
            metrics.max_abs_error = std::max(metrics.max_abs_error, error);
            global_denominator = std::max(global_denominator, std::abs(expected));
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
                      const EvaluationResult& evaluation, const Metrics& metrics,
                      const dnagpt::toy::DomainAudit& domains,
                      double setup_seconds, double encryption_seconds,
                      double evaluation_seconds, double decrypt_seconds) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"FHE toy DNAGPT block, FIDESlib GPU, final-only decrypt\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \"" << json_escape(options.backend_commit) << "\",\n"
        << "  \"container_image\": \"" << json_escape(options.container_image) << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \"" << json_escape(options.source_sha256) << "\",\n"
        << "  \"fixture_sha256\": \"" << json_escape(options.fixture_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": " << D << ",\n"
        << "    \"T\": " << T << ",\n"
        << "    \"heads\": " << HEADS << ",\n"
        << "    \"head_dim\": " << HEAD_DIM << ",\n"
        << "    \"mlp_dim\": " << MLP_DIM << ",\n"
        << "    \"batch_slots\": " << SLOTS << ",\n"
        << "    \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"scaling_technique\": \"FLEXIBLEAUTO\",\n"
        << "    \"bootstraps\": 0,\n"
        << "    \"attention_schedule\": \"numerator-first\",\n"
        << "    \"output_packing\": \"final MLP diagonal masks plus additive pack\",\n"
        << "    \"chebyshev_degrees\": {\"invsqrt\": " << DEG_INVSQRT
        << ", \"exp\": " << DEG_EXP << ", \"reciprocal\": " << DEG_RECIP
        << ", \"gelu\": " << DEG_GELU << "},\n"
        << "    \"public_domains\": {\"invsqrt\": [" << INVSQRT_LO << ", "
        << INVSQRT_HI << "], \"exp\": [" << EXP_LO << ", " << EXP_HI
        << "], \"reciprocal\": [" << RECIP_LO << ", " << RECIP_HI
        << "], \"gelu\": [" << GELU_LO << ", " << GELU_HI << "]},\n"
        << "    \"oracle_fixture_ranges\": {\"variance\": ["
        << domains.variance_min << ", " << domains.variance_max
        << "], \"score\": [" << domains.score_min << ", " << domains.score_max
        << "], \"unshifted_denominator\": [" << domains.denominator_min << ", "
        << domains.denominator_max << "], \"gelu_input\": ["
        << domains.gelu_input_min << ", " << domains.gelu_input_max
        << "], \"inside_public_domains\": "
        << (domains.inside_contract() ? "true" : "false") << "},\n"
        << "    \"packed_output_level\": " << evaluation.packed_level << ",\n"
        << "    \"level_trace\": {\n";
    std::size_t stage = 0;
    for (const auto& [name, range] : evaluation.levels) {
        out << "      \"" << json_escape(name) << "\": {\"min\": " << range.first
            << ", \"max\": " << range.second << "}";
        out << (++stage == evaluation.levels.size() ? "\n" : ",\n");
    }
    out << "    }\n"
        << "  },\n"
        << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << "  \"worst_token_rel_inf\": " << metrics.worst_token_rel_inf << ",\n"
        << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << "  \"all_finite\": " << (metrics.all_finite ? "true" : "false") << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (metrics.passed ? "true" : "false") << ",\n"
        << "  \"timings_seconds\": {\n"
        << "    \"context_keygen_load\": " << setup_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds << ",\n"
        << "    \"encrypted_evaluation\": " << evaluation_seconds << ",\n"
        << "    \"final_decrypt\": " << decrypt_seconds << "\n"
        << "  },\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"final_decrypt_calls\": 1,\n"
        << "  \"evaluator_has_private_key\": false\n"
        << "}\n";
    return out.str();
}

void write_exclusive(const std::filesystem::path& path, const std::string& contents) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        throw std::runtime_error("could not exclusively create evidence file " +
                                 path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t count =
            ::write(fd, contents.data() + offset, contents.size() - offset);
        if (count <= 0) {
            ::close(fd);
            throw std::runtime_error("short write to evidence file " + path.string());
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("could not close evidence file " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const Fixture fixture = dnagpt::toy::make_fixture();
        dnagpt::toy::DomainAudit domain_audit;
        const Tokens oracle = dnagpt::toy::block_oracle(fixture, &domain_audit);
        if (!domain_audit.inside_contract()) {
            throw std::runtime_error("fixture is outside a public approximation domain");
        }

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
        cc->EvalRotateKeyGen(keys.secretKey, {1, 2, 3, 4, 5, 6, 7});
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        const double setup_seconds = elapsed_seconds(setup_start);
        const std::uint32_t ring = cc->GetRingDimension();
        std::cout << "[context] security=HEStd_128_classic ring=" << ring
                  << " depth=" << MULT_DEPTH << " slots=" << SLOTS
                  << " gpu=" << options.gpu << '\n';

        const auto encryption_start = Clock::now();
        std::array<Ct, T> encrypted_inputs{};
        for (std::size_t token = 0; token < T; ++token) {
            std::vector<double> packed;
            packed.reserve(SLOTS);
            const auto values = dnagpt::toy::to_vector(fixture.embedded_inputs[token]);
            for (std::size_t block = 0; block < T; ++block) {
                packed.insert(packed.end(), values.begin(), values.end());
            }
            auto plaintext =
                cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
            encrypted_inputs[token] = cc->Encrypt(keys.publicKey, plaintext);
        }
        cc->Synchronize();
        const double encryption_seconds = elapsed_seconds(encryption_start);

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
        const std::vector<double> actual(raw.begin(), raw.begin() + SLOTS);
        const double decrypt_seconds = elapsed_seconds(decrypt_start);

        const Metrics metrics = measure(actual, oracle);
        const std::string evidence =
            make_json(options, ring, evaluation, metrics, domain_audit, setup_seconds,
                      encryption_seconds, evaluation_seconds, decrypt_seconds);
        write_exclusive(options.output, evidence);

        std::cout << "[metric] global_rel_inf=" << std::scientific
                  << metrics.global_rel_inf
                  << " worst_token_rel_inf=" << metrics.worst_token_rel_inf
                  << " tol=" << TOL << " finite=" << metrics.all_finite << '\n';
        std::cout << "[time] setup=" << std::fixed << std::setprecision(3)
                  << setup_seconds << "s encrypt=" << encryption_seconds
                  << "s evaluate=" << evaluation_seconds
                  << "s final_decrypt=" << decrypt_seconds << "s\n";
        std::cout << "[evidence] " << options.output << '\n';
        std::cout << (metrics.passed ? "FIDES_TOY_DNAGPT_PASS"
                                    : "FIDES_TOY_DNAGPT_FAIL")
                  << '\n';
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
