#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dnagpt::toy {

inline constexpr std::size_t kD = 8;
inline constexpr std::size_t kT = 4;
inline constexpr std::size_t kHeads = 2;
inline constexpr std::size_t kHeadDim = kD / kHeads;
inline constexpr std::size_t kMlp = 4 * kD;
inline constexpr double kEps = 1e-5;
inline constexpr double kInvsqrtLow = 0.01;
inline constexpr double kInvsqrtHigh = 0.90;
inline constexpr double kExpLow = -2.0;
inline constexpr double kExpHigh = 2.0;
inline constexpr double kReciprocalLow = 0.50;
inline constexpr double kReciprocalHigh = 4.50;
inline constexpr double kGeluLow = -2.0;
inline constexpr double kGeluHigh = 2.0;

using Vec = std::array<double, kD>;
using Tokens = std::array<Vec, kT>;

struct Matrix {
    std::size_t rows{};
    std::size_t cols{};
    std::vector<double> values;

    Matrix() = default;
    Matrix(std::size_t row_count, std::size_t col_count)
        : rows(row_count), cols(col_count), values(row_count * col_count) {}

    double& operator()(std::size_t row, std::size_t col) {
        return values.at(row * cols + col);
    }

    double operator()(std::size_t row, std::size_t col) const {
        return values.at(row * cols + col);
    }
};

struct Fixture {
    Tokens embedded_inputs{};
    Vec ln1_weight{};
    Vec ln1_bias{};
    Matrix attention_weight{3 * kD, kD};
    std::vector<double> attention_bias = std::vector<double>(3 * kD);
    Matrix projection_weight{kD, kD};
    Vec projection_bias{};
    Vec ln2_weight{};
    Vec ln2_bias{};
    Matrix fc_weight{kMlp, kD};
    std::vector<double> fc_bias = std::vector<double>(kMlp);
    Matrix fcp_weight{kD, kMlp};
    Vec fcp_bias{};
};

struct DomainAudit {
    double variance_min = std::numeric_limits<double>::infinity();
    double variance_max = -std::numeric_limits<double>::infinity();
    double score_min = std::numeric_limits<double>::infinity();
    double score_max = -std::numeric_limits<double>::infinity();
    double denominator_min = std::numeric_limits<double>::infinity();
    double denominator_max = -std::numeric_limits<double>::infinity();
    double gelu_input_min = std::numeric_limits<double>::infinity();
    double gelu_input_max = -std::numeric_limits<double>::infinity();

    bool inside_contract() const {
        return variance_min >= kInvsqrtLow && variance_max <= kInvsqrtHigh &&
               score_min >= kExpLow && score_max <= kExpHigh &&
               denominator_min >= kReciprocalLow &&
               denominator_max <= kReciprocalHigh &&
               gelu_input_min >= kGeluLow && gelu_input_max <= kGeluHigh;
    }
};

inline double deterministic_weight(std::size_t row, std::size_t col, double phase) {
    const double scale = 0.5 / std::sqrt(static_cast<double>(kD));
    return scale * std::sin((row + 1.0) * 0.613 + (col + 1.0) * 0.377 + phase);
}

inline Fixture make_fixture() {
    Fixture f;
    for (std::size_t token = 0; token < kT; ++token) {
        for (std::size_t dim = 0; dim < kD; ++dim) {
            f.embedded_inputs[token][dim] =
                0.36 * std::sin((token + 1.0) * (dim + 1.0) * 0.37) +
                0.09 * std::cos((token + 2.0) * (dim + 1.0) * 0.19);
        }
    }
    for (std::size_t dim = 0; dim < kD; ++dim) {
        f.ln1_weight[dim] = 0.96 + 0.035 * std::sin((dim + 1.0) * 0.41);
        f.ln1_bias[dim] = 0.012 * std::cos((dim + 1.0) * 0.29);
        f.ln2_weight[dim] = 1.01 + 0.03 * std::cos((dim + 1.0) * 0.33);
        f.ln2_bias[dim] = 0.01 * std::sin((dim + 1.0) * 0.47);
        f.projection_bias[dim] = 0.008 * std::sin((dim + 1.0) * 0.23);
        f.fcp_bias[dim] = 0.006 * std::cos((dim + 1.0) * 0.31);
    }
    for (std::size_t row = 0; row < 3 * kD; ++row) {
        for (std::size_t col = 0; col < kD; ++col) {
            f.attention_weight(row, col) = deterministic_weight(row, col, 0.11);
        }
        f.attention_bias[row] = 0.007 * std::sin((row + 1.0) * 0.17);
    }
    for (std::size_t row = 0; row < kD; ++row) {
        for (std::size_t col = 0; col < kD; ++col) {
            f.projection_weight(row, col) = deterministic_weight(row, col, 0.73);
        }
    }
    for (std::size_t row = 0; row < kMlp; ++row) {
        for (std::size_t col = 0; col < kD; ++col) {
            f.fc_weight(row, col) = deterministic_weight(row, col, 1.17);
        }
        f.fc_bias[row] = 0.006 * std::cos((row + 1.0) * 0.13);
    }
    for (std::size_t row = 0; row < kD; ++row) {
        for (std::size_t col = 0; col < kMlp; ++col) {
            f.fcp_weight(row, col) =
                (0.5 / std::sqrt(static_cast<double>(kMlp))) *
                std::sin((row + 1.0) * 0.571 + (col + 1.0) * 0.211 + 1.83);
        }
    }
    return f;
}

inline std::vector<double> matvec(const Matrix& matrix, const std::vector<double>& input,
                                  const std::vector<double>& bias) {
    if (matrix.cols != input.size() || matrix.rows != bias.size()) {
        throw std::invalid_argument("matvec shape mismatch");
    }
    std::vector<double> output(matrix.rows);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        output[row] = bias[row];
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            output[row] += matrix(row, col) * input[col];
        }
    }
    return output;
}

inline std::vector<double> to_vector(const Vec& input) {
    return {input.begin(), input.end()};
}

inline Vec to_vec(const std::vector<double>& input) {
    if (input.size() != kD) {
        throw std::invalid_argument("expected D values");
    }
    Vec output{};
    std::copy(input.begin(), input.end(), output.begin());
    return output;
}

inline Vec layernorm(const Vec& input, const Vec& weight, const Vec& bias) {
    double mean = 0.0;
    for (double value : input) {
        mean += value;
    }
    mean /= static_cast<double>(kD);
    double variance = 0.0;
    for (double value : input) {
        variance += (value - mean) * (value - mean);
    }
    variance /= static_cast<double>(kD);
    const double inverse_std = 1.0 / std::sqrt(variance + kEps);
    Vec output{};
    for (std::size_t dim = 0; dim < kD; ++dim) {
        output[dim] = (input[dim] - mean) * inverse_std * weight[dim] + bias[dim];
    }
    return output;
}

inline double variance_plus_eps(const Vec& input) {
    double mean = 0.0;
    for (double value : input) {
        mean += value;
    }
    mean /= static_cast<double>(kD);
    double variance = 0.0;
    for (double value : input) {
        variance += (value - mean) * (value - mean);
    }
    return variance / static_cast<double>(kD) + kEps;
}

inline double gelu_tanh(double value) {
    constexpr double two_over_pi = 0.63661977236758134308;
    return 0.5 * value *
           (1.0 + std::tanh(std::sqrt(two_over_pi) *
                            (value + 0.044715 * value * value * value)));
}

inline Tokens block_oracle(const Fixture& fixture, DomainAudit* audit = nullptr) {
    std::array<Vec, kT> normalized{};
    std::array<Vec, kT> query{};
    std::array<Vec, kT> key{};
    std::array<Vec, kT> value{};
    for (std::size_t token = 0; token < kT; ++token) {
        if (audit != nullptr) {
            const double variance = variance_plus_eps(fixture.embedded_inputs[token]);
            audit->variance_min = std::min(audit->variance_min, variance);
            audit->variance_max = std::max(audit->variance_max, variance);
        }
        normalized[token] =
            layernorm(fixture.embedded_inputs[token], fixture.ln1_weight, fixture.ln1_bias);
        const auto qkv =
            matvec(fixture.attention_weight, to_vector(normalized[token]), fixture.attention_bias);
        for (std::size_t dim = 0; dim < kD; ++dim) {
            query[token][dim] = qkv[dim];
            key[token][dim] = qkv[kD + dim];
            value[token][dim] = qkv[2 * kD + dim];
        }
    }

    Tokens attention{};
    const double score_scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    for (std::size_t token = 0; token < kT; ++token) {
        for (std::size_t head = 0; head < kHeads; ++head) {
            std::vector<double> logits(token + 1);
            for (std::size_t source = 0; source <= token; ++source) {
                double score = 0.0;
                for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
                    const std::size_t channel = head * kHeadDim + dim;
                    score += query[token][channel] * key[source][channel];
                }
                logits[source] = score * score_scale;
                if (audit != nullptr) {
                    audit->score_min = std::min(audit->score_min, logits[source]);
                    audit->score_max = std::max(audit->score_max, logits[source]);
                }
            }
            const double public_shift = *std::max_element(logits.begin(), logits.end());
            std::vector<double> exponentials(token + 1);
            double denominator = 0.0;
            double unshifted_denominator = 0.0;
            for (std::size_t source = 0; source <= token; ++source) {
                exponentials[source] = std::exp(logits[source] - public_shift);
                denominator += exponentials[source];
                unshifted_denominator += std::exp(logits[source]);
            }
            if (audit != nullptr) {
                audit->denominator_min =
                    std::min(audit->denominator_min, unshifted_denominator);
                audit->denominator_max =
                    std::max(audit->denominator_max, unshifted_denominator);
            }
            for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
                const std::size_t channel = head * kHeadDim + dim;
                for (std::size_t source = 0; source <= token; ++source) {
                    attention[token][channel] +=
                        exponentials[source] * value[source][channel] / denominator;
                }
            }
        }
    }

    Tokens residual1{};
    for (std::size_t token = 0; token < kT; ++token) {
        const auto projected = matvec(fixture.projection_weight, to_vector(attention[token]),
                                      to_vector(fixture.projection_bias));
        for (std::size_t dim = 0; dim < kD; ++dim) {
            residual1[token][dim] = fixture.embedded_inputs[token][dim] + projected[dim];
        }
    }

    Tokens output{};
    for (std::size_t token = 0; token < kT; ++token) {
        if (audit != nullptr) {
            const double variance = variance_plus_eps(residual1[token]);
            audit->variance_min = std::min(audit->variance_min, variance);
            audit->variance_max = std::max(audit->variance_max, variance);
        }
        const auto norm = layernorm(residual1[token], fixture.ln2_weight, fixture.ln2_bias);
        auto hidden = matvec(fixture.fc_weight, to_vector(norm), fixture.fc_bias);
        for (double& value : hidden) {
            if (audit != nullptr) {
                audit->gelu_input_min = std::min(audit->gelu_input_min, value);
                audit->gelu_input_max = std::max(audit->gelu_input_max, value);
            }
            value = gelu_tanh(value);
        }
        const auto projected =
            matvec(fixture.fcp_weight, hidden, to_vector(fixture.fcp_bias));
        for (std::size_t dim = 0; dim < kD; ++dim) {
            output[token][dim] = residual1[token][dim] + projected[dim];
        }
    }
    return output;
}

}  // namespace dnagpt::toy
