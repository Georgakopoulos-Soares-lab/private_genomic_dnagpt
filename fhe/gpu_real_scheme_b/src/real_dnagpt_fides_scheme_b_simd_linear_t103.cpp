// Additive Token-SIMD dense-linear micro-gate for Scheme B.
//
// Scope:
//   input embeddings -> exact client-assisted LN1 -> encrypted Q projection
//   for the full T=103 fixture, packing B=8 tokens in each ciphertext.
//
// Its slot algebra is contract-tested locally by test_simd_layout.py and
// test_simd_linear_source_contract.py.  Build/run wiring is additive and the
// Brev launcher is capacity-gated so this micro-gate cannot displace an
// existing GPU job.
//
// Fork discipline:
//   parent: frozen/evidenced T=32 general-attention source
//   parent SHA-256:
//     607e9c429bdea107149733e10f19765578175a5301b350556bd8604a1d384252
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
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t LOGICAL_SLOTS = COPIES * PACK_WIDTH;
constexpr std::size_t TOKEN_BATCH = 8;
constexpr std::size_t SLOTS = LOGICAL_SLOTS * TOKEN_BATCH;
constexpr std::size_t TOKEN_GROUPS =
    (T + TOKEN_BATCH - 1) / TOKEN_BATCH;
constexpr std::size_t BSGS_N1 = 32;
constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;
constexpr std::uint32_t MULT_DEPTH = 16;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 4;
constexpr double EPS = 1e-5;
constexpr double TOL = 4e-2;

static_assert(D == HEADS * HEAD_DIM);
static_assert(SLOTS == 32768);
static_assert(TOKEN_GROUPS == 13);
static_assert(T % TOKEN_BATCH == 7);
static_assert(BSGS_N1 * BSGS_N2 == PACK_WIDTH);

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
constexpr std::string_view PINNED_PARENT_SOURCE_SHA256 =
    "607e9c429bdea107149733e10f19765578175a5301b350556bd8604a1d384252";
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
    std::string mode;
    std::filesystem::path fixture_dir;
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string parent_source_sha256;
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256;
    std::string source_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_simd_linear_t103 "
        "--gpu N --mode packed|serial_control "
        "--fixture-dir PATH --output PATH "
        "--parent-source-sha256 SHA --fixture-manifest-sha256 SHA "
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
        } else if (arg == "--mode") {
            options.mode = next();
        } else if (arg == "--fixture-dir") {
            options.fixture_dir = next();
        } else if (arg == "--output") {
            options.output = next();
        } else if (arg == "--backend-commit") {
            options.backend_commit = next();
        } else if (arg == "--parent-source-sha256") {
            options.parent_source_sha256 = next();
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
            std::cout << "Token-SIMD T=103 dense-linear micro-gate\n";
            std::exit(0);
        } else {
            usage_error("unknown option " + arg);
        }
    }
    if (options.gpu < 0 || options.fixture_dir.empty() ||
        options.output.empty()) {
        usage_error("gpu, fixture-dir, and output are required");
    }
    if (options.mode != "packed" &&
        options.mode != "serial_control") {
        usage_error("--mode must be packed or serial_control");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit");
    }
    if (options.parent_source_sha256 != PINNED_PARENT_SOURCE_SHA256) {
        usage_error("refusing changed frozen parent source");
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
    std::vector<double> query_weight;
    std::vector<double> oracle_query;
};

Fixture load_fixture(const std::filesystem::path& directory) {
    Fixture fixture;
    fixture.input =
        read_f64(directory / "input__embeddings.bin", T * D);
    fixture.ln1_weight =
        read_f64(directory / "weights__ln1.bin", D);
    const std::vector<double> qkv =
        read_f64(directory / "weights__attn_qkv.bin", 3 * D * D);
    fixture.query_weight.assign(qkv.begin(),
                                qkv.begin() +
                                    static_cast<std::ptrdiff_t>(D * D));
    fixture.oracle_query =
        read_f64(directory / "oracle__query.bin", HEADS * T * HEAD_DIM);
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

std::size_t mode_group_count(const std::string& mode) {
    return mode == "packed" ? TOKEN_GROUPS : T;
}

std::size_t mode_first_token(const std::string& mode,
                             std::size_t group) {
    return mode == "packed" ? group * TOKEN_BATCH : group;
}

std::size_t mode_active_tokens(const std::string& mode,
                               std::size_t group) {
    if (mode == "serial_control") {
        return 1;
    }
    const std::size_t first = mode_first_token(mode, group);
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
    indices.erase(0);
    return {indices.begin(), indices.end()};
}

class Client {
  public:
    Client(Cc context, Keys& keys)
        : cc_(std::move(context)), keys_(keys) {}

    Ct invsqrt_boundary(const Ct& ciphertext,
                        std::size_t logical_instances) {
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
        return output;
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
};

struct EvaluationResult {
    std::vector<Ct> packed_query;
    std::size_t matrix_products = 0;
    std::size_t packed_level = 0;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture, Client& client)
        : cc_(std::move(context)), fixture_(fixture), client_(client) {}

    EvaluationResult evaluate(
        const std::vector<Ct>& encrypted_inputs,
        const std::vector<std::size_t>& active_tokens) {
        if (encrypted_inputs.size() != active_tokens.size() ||
            encrypted_inputs.empty()) {
            throw std::invalid_argument(
                "Token-SIMD input/active-count mismatch");
        }
        EvaluationResult result;
        result.packed_query.reserve(encrypted_inputs.size());
        for (std::size_t group = 0; group < encrypted_inputs.size();
             ++group) {
            Ct normalized =
                layernorm(encrypted_inputs[group], active_tokens[group]);
            Ct query =
                matmul(baby_rotations(normalized), fixture_.query_weight);
            result.packed_level =
                std::max(result.packed_level,
                         query->GetLevel());
            result.packed_query.push_back(std::move(query));
            std::cout << "[stage] token_group=" << group
                      << " active=" << active_tokens[group]
                      << " query_level="
                      << result.packed_query[group]->GetLevel() << '\n';
        }
        result.matrix_products = matmul_count_;
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

    Ct mean_broadcast(const Ct& input) {
        Ct sum = cc_->AccumulateSum(input, static_cast<int>(PACK_WIDTH),
                                    static_cast<int>(TOKEN_BATCH));
        Plaintext mean_scale = repeated_plain(std::vector<double>(
            D, 1.0 / static_cast<double>(D)));
        return cc_->EvalMult(sum, mean_scale);
    }

    Ct layernorm(const Ct& input, std::size_t active_tokens) {
        Ct mean = mean_broadcast(input);
        Ct centered = cc_->EvalSub(input, mean);
        Ct variance = mean_broadcast(cc_->EvalMult(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        Ct inverse =
            client_.invsqrt_boundary(variance, active_tokens);
        Ct normalized = cc_->EvalMult(centered, inverse);
        Plaintext ln1_weight = repeated_plain(fixture_.ln1_weight);
        return cc_->EvalMult(normalized, ln1_weight);
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

    Ct matmul(const BabyRotations& baby,
              const std::vector<double>& weight) {
        if (weight.size() != D * D) {
            throw std::invalid_argument("query weight must be D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::size_t diagonal =
                    BSGS_N1 * giant + small;
                std::vector<double> packed_values(SLOTS, 0.0);
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH - BSGS_N1 * giant) %
                            PACK_WIDTH;
                        const std::size_t column =
                            (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            const double coefficient =
                                weight[rolled_row * D + column];
                            for (std::size_t token = 0;
                                 token < TOKEN_BATCH; ++token) {
                                packed_values[physical_slot(
                                    copy, row, token)] = coefficient;
                            }
                        }
                    }
                }
                Plaintext diagonal_plain =
                    raw_plain(packed_values);
                Ct term = cc_->EvalMult(
                    baby.values[small], diagonal_plain);
                inner = small == 0 ? term
                                   : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(
                    inner, static_cast<std::int32_t>(
                               TOKEN_BATCH * BSGS_N1 * giant));
            }
            result = giant == 0 ? inner : cc_->EvalAdd(result, inner);
        }
        ++matmul_count_;
        return result;
    }

    Cc cc_;
    const Fixture& fixture_;
    Client& client_;
    std::size_t matmul_count_ = 0;
};

struct Metrics {
    double global_rel_inf = 0.0;
    double worst_token_rel_inf = 0.0;
    double max_abs_error = 0.0;
    bool all_finite = true;
    bool passed = false;
};

Metrics measure_query(const std::vector<double>& got,
                      const std::vector<double>& oracle_head_major) {
    if (got.size() != T * D ||
        oracle_head_major.size() != HEADS * T * HEAD_DIM) {
        throw std::invalid_argument("query metric size mismatch");
    }
    Metrics metrics;
    double global_denominator = 0.0;
    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t head = 0; head < HEADS; ++head) {
            for (std::size_t dim = 0; dim < HEAD_DIM; ++dim) {
                const std::size_t got_index =
                    token * D + head * HEAD_DIM + dim;
                const std::size_t expected_index =
                    (head * T + token) * HEAD_DIM + dim;
                const double actual = got[got_index];
                const double expected =
                    oracle_head_major[expected_index];
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
        << "  \"task\": \"Scheme B Token-SIMD T=103 local-design GPU "
           "micro-gate: packed LN1 plus encrypted query projection\",\n"
        << "  \"implementation_version\": "
           "\"t103-scheme-b-token-simd-linear-v1\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"gate\": \"token_simd_ln1_query\",\n"
        << "  \"mode\": \"" << json_escape(options.mode) << "\",\n"
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
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"fixture_contract_sha256\": \""
        << json_escape(options.fixture_contract_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": " << D << ", \"T\": " << T
        << ", \"token_batch\": " << TOKEN_BATCH
        << ", \"token_groups\": "
        << evaluation.packed_query.size() << ",\n"
        << "    \"pack_width\": " << PACK_WIDTH
        << ", \"copies\": " << COPIES << ", \"batch_slots\": "
        << SLOTS << ", \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH
        << ", \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"matrix_products\": " << evaluation.matrix_products
        << ", \"frozen_equivalent_matrix_products\": " << T << ",\n"
        << "    \"dense_call_reduction\": "
        << static_cast<double>(T) /
               static_cast<double>(evaluation.matrix_products)
        << ",\n"
        << "    \"packed_output_level\": "
        << evaluation.packed_level << "\n"
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
        << evaluation.packed_query.size() << ",\n"
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
        if (ring < 2 * SLOTS) {
            throw std::runtime_error(
                "ring dimension cannot provide requested Token-SIMD slots");
        }
        std::cout << "[context] ring=" << ring << " slots=" << SLOTS
                  << " token_batch=" << TOKEN_BATCH
                  << " rotation_keys=" << rotation_keys.size() << '\n';

        const auto encryption_start = Clock::now();
        const std::size_t group_count =
            mode_group_count(options.mode);
        std::vector<Ct> encrypted_inputs;
        std::vector<std::size_t> active_tokens;
        encrypted_inputs.reserve(group_count);
        active_tokens.reserve(group_count);
        for (std::size_t group = 0; group < group_count; ++group) {
            const std::size_t first =
                mode_first_token(options.mode, group);
            const std::size_t active =
                mode_active_tokens(options.mode, group);
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
            active_tokens.push_back(active);
        }
        cc->Synchronize();
        const double encryption_seconds =
            elapsed_seconds(encryption_start);

        Client client(cc, keys);
        EncryptedEvaluator evaluator(cc, fixture, client);
        const auto evaluation_start = Clock::now();
        EvaluationResult evaluation =
            evaluator.evaluate(encrypted_inputs, active_tokens);
        cc->Synchronize();
        const double evaluation_seconds =
            elapsed_seconds(evaluation_start);

        const auto decrypt_start = Clock::now();
        std::vector<double> output(T * D, 0.0);
        for (std::size_t group = 0;
             group < evaluation.packed_query.size(); ++group) {
            Plaintext decoded;
            cc->Decrypt(keys.secretKey, evaluation.packed_query[group],
                        &decoded);
            decoded->SetLength(SLOTS);
            const std::vector<double> raw =
                decoded->GetRealPackedValue();
            const std::size_t first =
                mode_first_token(options.mode, group);
            const std::size_t active =
                mode_active_tokens(options.mode, group);
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

        if (evaluation.matrix_products != group_count ||
            client.round_trips() != group_count ||
            client.logical_instances() != T) {
            throw std::runtime_error(
                "Token-SIMD declared work count mismatch");
        }
        const Metrics metrics =
            measure_query(output, fixture.oracle_query);
        const std::string evidence = make_json(
            options, ring, rotation_keys.size(), evaluation, metrics,
            fixture_seconds, setup_seconds, encryption_seconds,
            evaluation_seconds, client.seconds(), decrypt_seconds,
            client.round_trips(), client.logical_instances());
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout << (metrics.passed
                          ? "REAL_DNAGPT_TOKEN_SIMD_LINEAR_GATE_PASS\n"
                          : "REAL_DNAGPT_TOKEN_SIMD_LINEAR_GATE_FAIL\n");
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
