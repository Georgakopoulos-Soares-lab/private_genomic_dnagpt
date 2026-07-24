// Interleaved D=768/T=8 DNAGPT block-0 attention-through-projection gate.
//
// One ciphertext carries every token:
//   slot(copy, channel, token) = copy*(1024*8) + channel*8 + token
// Four identical copies preserve the 32x32 Halevi-Shoup/BSGS wrap contract.
// Causal attention is vectorized by token offset: eight ciphertexts represent
// all 36 legal query/source pairs. The evaluator has no private key and never
// decrypts. main() decrypts exactly once after the projection is complete.

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
constexpr std::size_t T = 8;
constexpr std::size_t P = T;
constexpr std::size_t HEADS = 12;
constexpr std::size_t HEAD_DIM = 64;
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t SLOTS_PER_COPY = PACK_WIDTH * P;
constexpr std::size_t SLOTS = COPIES * SLOTS_PER_COPY;
constexpr std::size_t BSGS_N1 = 32;
constexpr std::size_t BSGS_N2 = 32;
constexpr std::uint32_t MULT_DEPTH = 29;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 4;
constexpr double EPS = 1e-5;
constexpr double TOL = 4e-2;
constexpr double LN_VARIANCE_SCALE = 0.00390625;
constexpr double LN_SCALED_LO = 0.768768;
constexpr double LN_SCALED_HI = 2.5881600000000002;
constexpr double EXP_LO = -8.9;
constexpr double EXP_HI = 0.0;
constexpr double RECIPROCAL_LO = 0.62;
constexpr double RECIPROCAL_HI = 4.0;
constexpr std::size_t DEG_LN_INVSQRT = 7;
constexpr std::size_t DEG_EXP = 15;
constexpr std::size_t DEG_RECIPROCAL = 15;
constexpr std::size_t EXPECTED_LN1_MAX_LEVEL = 10;
constexpr std::size_t EXPECTED_QKV_MAX_LEVEL = 11;
constexpr std::size_t EXPECTED_NUMERATOR_MAX_LEVEL = 18;
constexpr std::size_t EXPECTED_RECIPROCAL_MAX_LEVEL = 22;
constexpr std::size_t EXPECTED_CONTEXT_MAX_LEVEL = 24;
constexpr std::size_t EXPECTED_PROJECTION_MAX_LEVEL = 25;

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
constexpr std::string_view PINNED_FIXTURE_MANIFEST =
    "75ce745757a8141df42054612d54a0837b5f0c2e7a771cafd57fc5055e054162";
constexpr std::string_view PINNED_PUBLIC_CONTRACT =
    "a913d3b8e3365525279b313a3dfb8ae863a6f7c0bf1c6d222e8f6886a853a0f4";

// [A] Public fixture-calibrated shifts: max causal score + 0.25 guard for
// every public query/head. They are never derived from a private ciphertext.
constexpr std::array<std::array<double, T>, HEADS> PUBLIC_SHIFTS = {{
    {{-4.58206609188219, 1.7619360697397495, 2.108265088880268,
      2.8009299613717062, 3.543305680700544, 3.292793141534338,
      3.5845862428562163, 3.5935972739066315}},
    {{0.47145594077201636, 2.1054318638932603, 2.333358956296627,
      2.4886434695338786, 2.2451884897569645, 2.1005917363062876,
      1.7680607503169816, 2.095058722877658}},
    {{-3.8563326374573705, 3.560666555443948, 4.470765378521653,
      4.676570358940506, 5.022332177978248, 5.011652324135089,
      4.885782567017004, 5.358021733076418}},
    {{-6.53436158479851, 1.5265805399592114, 1.7073275639282623,
      1.945376949975351, 1.980440246694852, 1.9461953191318009,
      1.9178586117913405, 1.8905782333474033}},
    {{-2.3970399790460997, 3.401184005461483, 3.3465683555580155,
      3.6135939317419674, 3.7371749106992542, 3.5511662913346655,
      2.9568438686557874, 3.548167172840476}},
    {{-6.941941723488476, 0.8936423783308813, 1.1962836220232527,
      1.6968840775654717, 2.313416136778719, 2.234281456604882,
      2.2771888969233345, 2.5857555813554196}},
    {{-2.7902277649529412, 2.694021967852604, 3.1882573848035816,
      3.6022038749255936, 3.309108104088758, 3.481343852250617,
      3.018294522982821, 3.0933996082624287}},
    {{-0.9465747038811574, 3.079570955296625, 3.0758937733092546,
      3.8158857060600453, 3.3746783406862906, 3.396385426491433,
      3.6005237451372354, 3.245119940532217}},
    {{-1.1661983152591806, 2.5572107283432897, 2.571579018070847,
      2.7018451510093486, 2.390518515531783, 2.239672821245957,
      2.3618595538283436, 1.8426398127290877}},
    {{-3.451531847820469, 2.1336417161269448, 2.8129696587325324,
      3.1213073194268124, 3.417995324189999, 3.3028167681828005,
      3.4356536269491444, 3.214700769068288}},
    {{-0.32157480907932934, 0.5492848252223939, 0.7253415235842002,
      0.7930999126526739, 0.8665964014172177, 0.8555305053990504,
      0.7979810823306414, 0.8076153382682861}},
    {{3.233303088164625, 2.016682829266845, 1.2393185357813614,
      1.5567903971667396, 1.03925846712713, 0.7618644464883253,
      1.0448064386472389, 0.5547960075045857}},
}};

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
    if (std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ",
                      &utc) == 0) {
        throw std::runtime_error("could not format UTC timestamp");
    }
    return buffer.data();
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            out << '\\' << ch;
        } else if (ch == '\n') {
            out << "\\n";
        } else if (ch >= 0x20) {
            out << ch;
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
    std::string public_contract_sha256;
    std::string source_sha256;
    std::string container_image;
    std::string environment = "Brev GPU";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: dnagpt_packed_t8_fides --gpu N --fixture-dir PATH "
        "--output PATH --fixture-manifest-sha256 SHA "
        "--public-contract-sha256 SHA --source-sha256 SHA "
        "--container-image IMAGE [--environment TEXT] [--backend-commit SHA]");
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
        } else if (arg == "--public-contract-sha256") {
            options.public_contract_sha256 = next();
        } else if (arg == "--source-sha256") {
            options.source_sha256 = next();
        } else if (arg == "--container-image") {
            options.container_image = next();
        } else if (arg == "--environment") {
            options.environment = next();
        } else {
            usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0 || options.fixture_dir.empty() || options.output.empty()) {
        usage_error("gpu must be non-negative; fixture and output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT ||
        options.fixture_manifest_sha256 != PINNED_FIXTURE_MANIFEST ||
        options.public_contract_sha256 != PINNED_PUBLIC_CONTRACT) {
        usage_error("refusing an unpinned backend, fixture, or public contract");
    }
    if (options.source_sha256.empty() || options.container_image.empty()) {
        usage_error("source and immutable container identities are required");
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable output: " +
                                 options.output.string());
    }
    return options;
}

struct Matrix {
    std::size_t rows{};
    std::size_t cols{};
    std::vector<double> values;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), values(r * c) {}
    double operator()(std::size_t row, std::size_t col) const {
        return values.at(row * cols + col);
    }
};

std::vector<double> read_f64(const std::filesystem::path& path,
                             std::size_t elements) {
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) != elements * sizeof(double)) {
        throw std::runtime_error("fixture file size contract failed: " +
                                 path.string());
    }
    std::vector<double> values(elements);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof() ||
        !std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::runtime_error("invalid fixture array: " + path.string());
    }
    return values;
}

Matrix matrix(const std::filesystem::path& path, std::size_t rows,
              std::size_t cols) {
    Matrix result(rows, cols);
    result.values = read_f64(path, rows * cols);
    return result;
}

Matrix row_slice(const Matrix& input, std::size_t first) {
    Matrix output(D, D);
    std::copy_n(input.values.begin() +
                    static_cast<std::ptrdiff_t>(first * D),
                D * D, output.values.begin());
    return output;
}

struct Fixture {
    std::vector<double> input;
    std::vector<double> ln1;
    std::array<Matrix, 3> qkv;
    Matrix projection;
    std::vector<double> oracle_projection;
};

Fixture load_fixture(const std::filesystem::path& directory) {
    Fixture fixture;
    fixture.input = read_f64(directory / "input__embeddings.bin", T * D);
    fixture.ln1 = read_f64(directory / "weights__ln1.bin", D);
    const Matrix qkv = matrix(directory / "weights__attn_qkv.bin", 3 * D, D);
    for (std::size_t part = 0; part < 3; ++part) {
        fixture.qkv[part] = row_slice(qkv, part * D);
    }
    fixture.projection =
        matrix(directory / "weights__attn_proj.bin", D, D);
    fixture.oracle_projection =
        read_f64(directory / "oracle__attention_projection.bin", T * D);
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
            keys.insert(canonical_rotation(
                static_cast<std::int64_t>(stride) * step * multiple));
        }
    }
}

std::vector<int> required_rotation_keys() {
    std::set<int> keys;
    for (std::size_t small = 1; small < BSGS_N1; ++small) {
        keys.insert(canonical_rotation(static_cast<std::int64_t>(small * P)));
    }
    for (std::size_t giant = 1; giant < BSGS_N2; ++giant) {
        keys.insert(canonical_rotation(
            static_cast<std::int64_t>(giant * BSGS_N1 * P)));
    }
    add_accumulate_keys(keys, static_cast<int>(PACK_WIDTH), static_cast<int>(P));
    add_accumulate_keys(keys, static_cast<int>(HEAD_DIM), static_cast<int>(P));
    add_accumulate_keys(keys, static_cast<int>(HEAD_DIM), -static_cast<int>(P));
    for (int offset = 1; offset < static_cast<int>(T); ++offset) {
        keys.insert(-offset);
    }
    keys.erase(0);
    return {keys.begin(), keys.end()};
}

struct EvaluationResult {
    Ct output;
    std::map<std::string, std::size_t> levels;
    std::size_t matrix_products = 0;
    std::size_t explicit_ct_ct_multiplications = 0;
    std::size_t explicit_ct_plain_multiplications = 0;
    std::size_t accumulate_sum_calls = 0;
    std::size_t direct_logical_rotations = 0;
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
        invsqrt_coefficients_ = cc_->GetChebyshevCoefficients(
            invsqrt, LN_SCALED_LO, LN_SCALED_HI, DEG_LN_INVSQRT);
        exp_coefficients_ = cc_->GetChebyshevCoefficients(
            exponential, EXP_LO, EXP_HI, DEG_EXP);
        reciprocal_coefficients_ = cc_->GetChebyshevCoefficients(
            reciprocal, RECIPROCAL_LO, RECIPROCAL_HI, DEG_RECIPROCAL);
    }

    EvaluationResult evaluate(const Ct& encrypted_input) {
        Ct normalized = layernorm(encrypted_input);
        record("ln1", normalized);
        require_remaining_depth(normalized->GetLevel(), 15,
                                "QKV through packed attention projection");

        const BabyRotations normalized_baby = baby_rotations(normalized);
        const Ct query = matmul(normalized_baby, fixture_.qkv[0]);
        const Ct key = matmul(normalized_baby, fixture_.qkv[1]);
        const Ct value = matmul(normalized_baby, fixture_.qkv[2]);
        record("qkv", query);
        require_remaining_depth(query->GetLevel(), 14,
                                "packed attention through projection");

        std::array<Ct, T> numerators{};
        for (std::size_t offset = 0; offset < T; ++offset) {
            const Ct aligned_key =
                offset == 0 ? key : rotate(key, -static_cast<int>(offset));
            Ct products = multiply(query, aligned_key);
            Ct anchors = accumulate(products, HEAD_DIM, static_cast<int>(P));
            anchors = multiply_plain(anchors, anchor_scale_mask(offset));
            anchors = cc_->EvalSub(anchors, active_shift_plain(offset));
            require_remaining_depth(
                anchors->GetLevel(), 4,
                "attention exponential offset " + std::to_string(offset));
            numerators[offset] = cc_->EvalChebyshevSeries(
                anchors, exp_coefficients_, EXP_LO, EXP_HI);
            numerators[offset] =
                multiply_plain(numerators[offset], anchor_mask(offset));
        }
        record("attention_numerators", numerators[0]);

        Ct denominator = numerators[0];
        for (std::size_t offset = 1; offset < T; ++offset) {
            denominator = cc_->EvalAdd(denominator, numerators[offset]);
        }
        denominator = cc_->EvalAdd(denominator, reciprocal_guard_plain());
        require_remaining_depth(denominator->GetLevel(), 4,
                                "attention reciprocal");
        Ct reciprocal = cc_->EvalChebyshevSeries(
            denominator, reciprocal_coefficients_, RECIPROCAL_LO,
            RECIPROCAL_HI);
        record("attention_reciprocal", reciprocal);
        require_remaining_depth(reciprocal->GetLevel(), 3,
                                "attention weighting and projection");

        Ct context;
        for (std::size_t offset = 0; offset < T; ++offset) {
            // The deeper reciprocal must be the first operand for the pinned
            // FIDES/OpenFHE asymmetric-level multiplication contract.
            Ct weights = multiply(reciprocal, numerators[offset]);
            weights = accumulate(weights, HEAD_DIM, -static_cast<int>(P));
            const Ct aligned_value =
                offset == 0 ? value : rotate(value, -static_cast<int>(offset));
            Ct contribution = multiply(weights, aligned_value);
            context =
                offset == 0 ? contribution : cc_->EvalAdd(context, contribution);
        }
        record("attention_context", context);

        require_remaining_depth(context->GetLevel(), 1,
                                "attention output projection");
        Ct projection = matmul(baby_rotations(context), fixture_.projection);
        record("attention_projection", projection);
        return {.output = projection,
                .levels = levels_,
                .matrix_products = matrix_products_,
                .explicit_ct_ct_multiplications = ct_ct_count_,
                .explicit_ct_plain_multiplications = ct_plain_count_,
                .accumulate_sum_calls = accumulate_count_,
                .direct_logical_rotations = rotation_count_};
    }

  private:
    struct BabyRotations {
        std::array<Ct, BSGS_N1> values{};
    };

    std::size_t slot(std::size_t copy, std::size_t channel,
                     std::size_t token) const {
        return copy * SLOTS_PER_COPY + channel * P + token;
    }

    Plaintext raw_plain(const std::vector<double>& values) {
        if (values.size() != SLOTS) {
            throw std::invalid_argument("plaintext slot count mismatch");
        }
        return cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
    }

    Plaintext active_channel_mask() {
        std::vector<double> values(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t channel = 0; channel < D; ++channel) {
                for (std::size_t token = 0; token < T; ++token) {
                    values[slot(copy, channel, token)] = 1.0;
                }
            }
        }
        return raw_plain(values);
    }

    Plaintext inactive_channel_guard_plain() {
        std::vector<double> values(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t channel = D; channel < PACK_WIDTH; ++channel) {
                for (std::size_t token = 0; token < T; ++token) {
                    values[slot(copy, channel, token)] = 1.0;
                }
            }
        }
        return raw_plain(values);
    }

    Plaintext repeated_channel_plain(const std::vector<double>& channels) {
        if (channels.size() != D) {
            throw std::invalid_argument("channel plaintext requires D values");
        }
        std::vector<double> values(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t channel = 0; channel < D; ++channel) {
                for (std::size_t token = 0; token < T; ++token) {
                    values[slot(copy, channel, token)] = channels[channel];
                }
            }
        }
        return raw_plain(values);
    }

    Plaintext anchor_mask(std::size_t minimum_query) {
        std::vector<double> values(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t head = 0; head < HEADS; ++head) {
                const std::size_t channel = head * HEAD_DIM;
                for (std::size_t query = minimum_query; query < T; ++query) {
                    values[slot(copy, channel, query)] = 1.0;
                }
            }
        }
        return raw_plain(values);
    }

    Plaintext anchor_scale_mask(std::size_t minimum_query) {
        std::vector<double> values(SLOTS);
        const double scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t head = 0; head < HEADS; ++head) {
                const std::size_t channel = head * HEAD_DIM;
                for (std::size_t query = minimum_query; query < T; ++query) {
                    values[slot(copy, channel, query)] = scale;
                }
            }
        }
        return raw_plain(values);
    }

    Plaintext active_shift_plain(std::size_t minimum_query) {
        std::vector<double> values(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t head = 0; head < HEADS; ++head) {
                const std::size_t channel = head * HEAD_DIM;
                for (std::size_t query = minimum_query; query < T; ++query) {
                    values[slot(copy, channel, query)] =
                        PUBLIC_SHIFTS[head][query];
                }
            }
        }
        return raw_plain(values);
    }

    Plaintext reciprocal_guard_plain() {
        std::vector<double> values(SLOTS, 1.0);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t head = 0; head < HEADS; ++head) {
                const std::size_t channel = head * HEAD_DIM;
                for (std::size_t query = 0; query < T; ++query) {
                    values[slot(copy, channel, query)] = 0.0;
                }
            }
        }
        return raw_plain(values);
    }

    Ct multiply(const Ct& left, const Ct& right) {
        ++ct_ct_count_;
        return cc_->EvalMult(left, right);
    }

    Ct multiply_plain(const Ct& input, Plaintext plain) {
        ++ct_plain_count_;
        return cc_->EvalMult(input, plain);
    }

    Ct rotate(const Ct& input, int index) {
        ++rotation_count_;
        return cc_->EvalRotate(input, index);
    }

    Ct accumulate(const Ct& input, std::size_t count, int stride) {
        ++accumulate_count_;
        return cc_->AccumulateSum(input, static_cast<int>(count), stride);
    }

    Ct sum_channels(const Ct& input) {
        return accumulate(input, PACK_WIDTH, static_cast<int>(P));
    }

    Ct layernorm(const Ct& input) {
        Ct mean = multiply_plain(
            sum_channels(input),
            repeated_channel_plain(std::vector<double>(D, 1.0 / D)));
        Ct centered = cc_->EvalSub(input, mean);
        centered = multiply_plain(centered, active_channel_mask());
        Ct variance = sum_channels(multiply(centered, centered));
        variance = multiply_plain(
            variance,
            repeated_channel_plain(std::vector<double>(D, 1.0 / D)));
        variance = cc_->EvalAdd(variance, EPS);
        Ct scaled = multiply_plain(
            variance,
            repeated_channel_plain(
                std::vector<double>(D, 1.0 / LN_VARIANCE_SCALE)));
        // Padding channels are public and carry no model data. Put them at
        // 1.0 before inverse-sqrt so Chebyshev is never evaluated outside its
        // positive public domain, then annihilate them with centered==0.
        scaled = cc_->EvalAdd(scaled, inactive_channel_guard_plain());
        Ct inverse = cc_->EvalChebyshevSeries(
            scaled, invsqrt_coefficients_, LN_SCALED_LO, LN_SCALED_HI);
        Ct normalized = multiply(inverse, centered);
        std::vector<double> scaled_weight(D);
        for (std::size_t channel = 0; channel < D; ++channel) {
            scaled_weight[channel] =
                fixture_.ln1[channel] / std::sqrt(LN_VARIANCE_SCALE);
        }
        return multiply_plain(normalized,
                              repeated_channel_plain(scaled_weight));
    }

    BabyRotations baby_rotations(const Ct& input) {
        std::vector<std::int32_t> indices;
        for (std::size_t small = 1; small < BSGS_N1; ++small) {
            indices.push_back(static_cast<std::int32_t>(small * P));
        }
        const auto rotations = cc_->EvalFastRotation(
            input, indices, cc_->GetCyclotomicOrder(), nullptr);
        if (rotations.size() != BSGS_N1 - 1) {
            throw std::runtime_error("incomplete packed baby rotations");
        }
        rotation_count_ += BSGS_N1 - 1;
        BabyRotations result;
        result.values[0] = input;
        for (std::size_t small = 1; small < BSGS_N1; ++small) {
            result.values[small] = rotations.at(small - 1);
        }
        return result;
    }

    Ct matmul(const BabyRotations& baby, const Matrix& weight) {
        if (weight.rows != D || weight.cols != D) {
            throw std::invalid_argument("packed matmul requires D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::size_t diagonal = BSGS_N1 * giant + small;
                std::vector<double> plain(SLOTS);
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH;
                        const std::size_t column =
                            (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            for (std::size_t token = 0; token < T; ++token) {
                                plain[slot(copy, row, token)] =
                                    weight(rolled_row, column);
                            }
                        }
                    }
                }
                Ct term = multiply_plain(baby.values[small], raw_plain(plain));
                inner = small == 0 ? term : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = rotate(
                    inner, static_cast<int>(giant * BSGS_N1 * P));
            }
            result = giant == 0 ? inner : cc_->EvalAdd(result, inner);
        }
        ++matrix_products_;
        return result;
    }

    void require_remaining_depth(std::size_t level, std::size_t required,
                                 const std::string& stage) const {
        if (level > MULT_DEPTH || required > MULT_DEPTH - level) {
            throw std::runtime_error(
                "insufficient depth before " + stage + ": level " +
                std::to_string(level) + " + " + std::to_string(required) +
                " exceeds " + std::to_string(MULT_DEPTH));
        }
    }

    void record(const std::string& stage, const Ct& value) {
        levels_[stage] = value->GetLevel();
        std::cout << "[stage] " << stage << " level=" << value->GetLevel()
                  << '\n';
    }

    Cc cc_;
    const Fixture& fixture_;
    std::vector<double> invsqrt_coefficients_;
    std::vector<double> exp_coefficients_;
    std::vector<double> reciprocal_coefficients_;
    std::map<std::string, std::size_t> levels_;
    std::size_t matrix_products_ = 0;
    std::size_t ct_ct_count_ = 0;
    std::size_t ct_plain_count_ = 0;
    std::size_t accumulate_count_ = 0;
    std::size_t rotation_count_ = 0;
};

struct Metrics {
    double global_rel_inf = std::numeric_limits<double>::infinity();
    double worst_token_rel_inf = std::numeric_limits<double>::infinity();
    double max_abs_error = std::numeric_limits<double>::infinity();
    bool all_finite = false;
    bool passed = false;
};

Metrics measure(const std::vector<double>& slots,
                const std::vector<double>& reference) {
    if (slots.size() < SLOTS_PER_COPY || reference.size() != T * D) {
        throw std::invalid_argument("decoded output size mismatch");
    }
    Metrics metrics;
    metrics.max_abs_error = 0.0;
    metrics.worst_token_rel_inf = 0.0;
    metrics.all_finite = true;
    double global_denominator = 0.0;
    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t channel = 0; channel < D; ++channel) {
            const double actual = slots[channel * P + token];
            const double expected = reference[token * D + channel];
            metrics.all_finite = metrics.all_finite && std::isfinite(actual);
            const double error = std::abs(actual - expected);
            metrics.max_abs_error = std::max(metrics.max_abs_error, error);
            token_error = std::max(token_error, error);
            token_denominator =
                std::max(token_denominator, std::abs(expected));
            global_denominator =
                std::max(global_denominator, std::abs(expected));
        }
        metrics.worst_token_rel_inf = std::max(
            metrics.worst_token_rel_inf,
            token_error / std::max(token_denominator, 1e-15));
    }
    metrics.global_rel_inf =
        metrics.max_abs_error / std::max(global_denominator, 1e-15);
    metrics.passed = metrics.all_finite && metrics.global_rel_inf <= TOL &&
                     metrics.worst_token_rel_inf <= TOL;
    return metrics;
}

std::string make_json(const Options& options, std::uint32_t ring,
                      std::size_t rotation_keys,
                      const EvaluationResult& evaluation,
                      const Metrics& metrics, double load_seconds,
                      double setup_seconds, double encrypt_seconds,
                      double evaluate_seconds, double decrypt_seconds) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"D=768/T=8 packed DNAGPT attention-through-projection, final-only decrypt\",\n"
        << "  \"implementation_version\": \"packed-t8-offset-v1\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \"" << options.backend_commit << "\",\n"
        << "  \"container_image\": \"" << json_escape(options.container_image)
        << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"source_sha256\": \"" << options.source_sha256 << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << options.fixture_manifest_sha256 << "\",\n"
        << "  \"public_contract_sha256\": \""
        << options.public_contract_sha256 << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": 768, \"T\": 8, \"heads\": 12, \"head_dim\": 64,\n"
        << "    \"layout\": \"copy*(1024*P)+channel*P+token\",\n"
        << "    \"P\": 8, \"pack_width\": 1024, \"copies\": 4,\n"
        << "    \"input_ciphertexts\": 1, \"output_ciphertexts\": 1,\n"
        << "    \"batch_slots\": " << SLOTS << ", \"ring_dim\": " << ring
        << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH
        << ", \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"bsgs\": {\"n1\": 32, \"n2\": 32},\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"attention_schedule\": \"8 causal token offsets; numerator-first exp then reciprocal\",\n"
        << "    \"public_domains\": {\"ln_scaled_variance\": ["
        << LN_SCALED_LO << ", " << LN_SCALED_HI
        << "], \"shifted_score\": [" << EXP_LO << ", " << EXP_HI
        << "], \"denominator\": [" << RECIPROCAL_LO << ", "
        << RECIPROCAL_HI << "]},\n"
        << "    \"chebyshev_degrees\": {\"ln1_invsqrt\": "
        << DEG_LN_INVSQRT << ", \"exp\": " << DEG_EXP
        << ", \"reciprocal\": " << DEG_RECIPROCAL << "},\n"
        << "    \"predicted_level_maxima\": {\"ln1\": "
        << EXPECTED_LN1_MAX_LEVEL << ", \"qkv\": "
        << EXPECTED_QKV_MAX_LEVEL << ", \"numerator\": "
        << EXPECTED_NUMERATOR_MAX_LEVEL << ", \"reciprocal\": "
        << EXPECTED_RECIPROCAL_MAX_LEVEL << ", \"context\": "
        << EXPECTED_CONTEXT_MAX_LEVEL << ", \"projection\": "
        << EXPECTED_PROJECTION_MAX_LEVEL << "},\n"
        << "    \"actual_level_trace\": {";
    std::size_t level_index = 0;
    for (const auto& [name, level] : evaluation.levels) {
        out << (level_index++ == 0 ? "\n" : ",\n")
            << "      \"" << name << "\": " << level;
    }
    out << "\n    },\n"
        << "    \"operation_counts\": {\n"
        << "      \"ciphertext_plaintext_matrix_products\": "
        << evaluation.matrix_products << ",\n"
        << "      \"explicit_ciphertext_ciphertext_multiplications\": "
        << evaluation.explicit_ct_ct_multiplications << ",\n"
        << "      \"explicit_ciphertext_plaintext_multiplications\": "
        << evaluation.explicit_ct_plain_multiplications << ",\n"
        << "      \"accumulate_sum_calls\": "
        << evaluation.accumulate_sum_calls << ",\n"
        << "      \"direct_logical_rotations\": "
        << evaluation.direct_logical_rotations << "\n"
        << "    },\n"
        << "    \"depth_guard\": {\"projection_required_levels\": 1, \"fail_closed\": true}\n"
        << "  },\n"
        << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << "  \"worst_token_rel_inf\": " << metrics.worst_token_rel_inf
        << ",\n"
        << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << "  \"all_finite\": " << (metrics.all_finite ? "true" : "false")
        << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (metrics.passed ? "true" : "false") << ",\n"
        << "  \"timings_seconds\": {\"load_verified_fixture\": "
        << load_seconds << ", \"context_keygen_load\": " << setup_seconds
        << ", \"encrypt_embedded_input\": " << encrypt_seconds
        << ", \"encrypted_evaluation\": " << evaluate_seconds
        << ", \"final_decrypt\": " << decrypt_seconds << "},\n"
        << "  \"input_boundary\": \"all 8 token plus position embeddings in one ciphertext\",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"final_decrypt_calls\": 1,\n"
        << "  \"evaluator_has_private_key\": false,\n"
        << "  \"public_calibration_only\": true,\n"
        << "  \"private_query_domain_adaptation\": false\n"
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
        throw std::runtime_error("could not exclusively create " + path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t count =
            ::write(descriptor, contents.data() + offset,
                    contents.size() - offset);
        if (count <= 0) {
            ::close(descriptor);
            throw std::runtime_error("short evidence write");
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error("could not close evidence output");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto load_start = Clock::now();
        const Fixture fixture = load_fixture(options.fixture_dir);
        const double load_seconds = elapsed_seconds(load_start);

        const std::vector<int> rotations = required_rotation_keys();
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
        cc->EvalRotateKeyGen(keys.secretKey, rotations);
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        const double setup_seconds = elapsed_seconds(setup_start);

        std::cout << "[context] ring=" << cc->GetRingDimension()
                  << " slots=" << SLOTS << " depth=" << MULT_DEPTH
                  << " rotation_keys=" << rotations.size() << '\n';

        const auto encrypt_start = Clock::now();
        std::vector<double> packed(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            for (std::size_t channel = 0; channel < D; ++channel) {
                for (std::size_t token = 0; token < T; ++token) {
                    packed[copy * SLOTS_PER_COPY + channel * P + token] =
                        fixture.input[token * D + channel];
                }
            }
        }
        Plaintext plaintext =
            cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
        const Ct encrypted = cc->Encrypt(keys.publicKey, plaintext);
        cc->Synchronize();
        const double encrypt_seconds = elapsed_seconds(encrypt_start);

        // Server boundary: evaluator gets context, public model, ciphertext;
        // it is structurally unable to decrypt.
        EncryptedEvaluator evaluator(cc, fixture);
        const auto evaluate_start = Clock::now();
        const EvaluationResult evaluation = evaluator.evaluate(encrypted);
        cc->Synchronize();
        const double evaluate_seconds = elapsed_seconds(evaluate_start);

        const auto decrypt_start = Clock::now();
        Plaintext decoded;
        cc->Decrypt(keys.secretKey, evaluation.output, &decoded);
        decoded->SetLength(SLOTS);
        const std::vector<double> slots = decoded->GetRealPackedValue();
        const double decrypt_seconds = elapsed_seconds(decrypt_start);

        const Metrics metrics = measure(slots, fixture.oracle_projection);
        const std::string evidence = make_json(
            options, cc->GetRingDimension(), rotations.size(), evaluation,
            metrics, load_seconds, setup_seconds, encrypt_seconds,
            evaluate_seconds, decrypt_seconds);
        write_exclusive(options.output, evidence);
        std::cout << evidence
                  << (metrics.passed ? "DNAGPT_PACKED_T8_PASS\n"
                                     : "DNAGPT_PACKED_T8_FAIL\n");
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
