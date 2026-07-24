// Real-width encrypted refresh + continuation gate for DNAGPT.
//
// Security boundary: EncryptedRefreshEvaluator receives evaluation capability and a
// ciphertext, never a private key. main performs the sole final readout.

#include <fideslib.hpp>

#include <CKKS/Ciphertext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <openfhe.h>

#include <algorithm>
#include <any>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
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

namespace F = FIDESlib::CKKS;

constexpr std::size_t D = 768;
constexpr std::size_t T = 2;
constexpr std::size_t ACTIVE_VALUES = D * T;
// OpenFHE/FIDESlib CKKS bootstrapping requires a power-of-two slot transform.
// The 1,536 active D768/T2 values occupy the prefix; the remainder is public padding.
constexpr std::size_t SLOTS = 2048;
constexpr std::uint32_t DEPTH = 43;
constexpr std::uint32_t SCALE_BITS = 59;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 4;
constexpr std::array<std::uint32_t, 2> LEVEL_BUDGET = {4, 4};
// Frozen before execution and query-independent. It includes ~27% headroom over
// the fixture's 15.7544 absolute maximum, keeping the message away from ±1.
constexpr double PUBLIC_BOUND = 20.0;
constexpr double LAYERNORM_EPSILON = 1e-5;
constexpr double TOL = 4e-2;
constexpr std::uint32_t CPU_ITERATIONS = 2;
constexpr std::uint32_t CPU_PRECISION = 6;
constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
constexpr std::string_view PINNED_OPENFHE_COMMIT =
    "dd8d3740604473c1fb0adec3ca1a0fa5f0cf9964";
constexpr std::string_view PINNED_ACTIVATION_SHA256 =
    "829b2b4b62b6315f8d93a33562cb4a4d82935c363b4ebfa9a56a8dd29edfe63f";
constexpr std::string_view PINNED_BACKEND_PATCH_SHA256 =
    "81b6f6d8f466c67bc4e764c3f43d38ad2f6caec44b807e8fb916fc8b1063e295";

using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;
using Cc = CryptoContext<DCRTPoly>;
using OfheCt = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using OfheCc = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;

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

enum class RefreshMode {
    NativeGpu,
    CpuIterativeInterop,
};

std::string_view mode_name(RefreshMode mode) {
    return mode == RefreshMode::NativeGpu ? "native-gpu" : "cpu-iterative-interop";
}

struct Options {
    int gpu = 0;
    RefreshMode mode = RefreshMode::NativeGpu;
    std::filesystem::path activation;
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
    std::string source_sha256 = "UNSPECIFIED";
    std::string activation_sha256 = "UNSPECIFIED";
    std::string backend_patch_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: dnagpt_fides_refresh --gpu N "
        "--mode native-gpu|cpu-iterative-interop --activation PATH --output PATH "
        "--backend-commit SHA --container-image IMAGE --environment TEXT "
        "--source-sha256 SHA --activation-sha256 SHA --backend-patch-sha256 SHA");
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
            const std::string mode = next();
            if (mode == "native-gpu") {
                options.mode = RefreshMode::NativeGpu;
            } else if (mode == "cpu-iterative-interop") {
                options.mode = RefreshMode::CpuIterativeInterop;
            } else {
                usage_error("unsupported mode " + mode);
            }
        } else if (arg == "--activation") {
            options.activation = next();
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
        } else if (arg == "--activation-sha256") {
            options.activation_sha256 = next();
        } else if (arg == "--backend-patch-sha256") {
            options.backend_patch_sha256 = next();
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: dnagpt_fides_refresh --gpu N "
                   "--mode native-gpu|cpu-iterative-interop --activation PATH "
                   "--output PATH --backend-commit SHA --container-image IMAGE "
                   "--environment TEXT --source-sha256 SHA --activation-sha256 SHA "
                   "--backend-patch-sha256 SHA\n";
            std::exit(0);
        } else {
            usage_error("unknown option " + arg);
        }
    }

    if (options.gpu < 0) {
        usage_error("--gpu must be non-negative");
    }
    if (options.activation.empty() || options.output.empty()) {
        usage_error("--activation and --output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit " + options.backend_commit);
    }
    if (options.activation_sha256 != PINNED_ACTIVATION_SHA256) {
        usage_error("refusing activation outside the frozen real block-output fixture");
    }
    if (options.backend_patch_sha256 != PINNED_BACKEND_PATCH_SHA256) {
        usage_error("refusing an unpinned FIDESlib backend patch");
    }
    if (options.source_sha256 == "UNSPECIFIED" ||
        options.activation_sha256 == "UNSPECIFIED" ||
        options.backend_patch_sha256 == "UNSPECIFIED" ||
        options.container_image == "UNSPECIFIED") {
        usage_error("source, activation, and container identity metadata are required");
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence " +
                                 options.output.string());
    }
    return options;
}

std::vector<double> load_activation(const std::filesystem::path& path) {
    static_assert(std::endian::native == std::endian::little,
                  "fixture contract is little-endian float64");
    constexpr std::uintmax_t EXPECTED_BYTES = ACTIVE_VALUES * sizeof(double);
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("activation fixture missing " + path.string());
    }
    if (std::filesystem::file_size(path) != EXPECTED_BYTES) {
        throw std::runtime_error("activation fixture size mismatch");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open activation fixture");
    }
    std::vector<double> values(ACTIVE_VALUES);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(EXPECTED_BYTES));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("short or trailing activation fixture data");
    }
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::runtime_error("activation fixture contains a non-finite value");
    }
    return values;
}

struct Timings {
    double conditioning_seconds = 0.0;
    double download_seconds = 0.0;
    double bootstrap_seconds = 0.0;
    double upload_seconds = 0.0;
    double tail_seconds = 0.0;
};

struct EvaluationResult {
    Ct tail;
    std::size_t input_level = 0;
    std::size_t conditioned_level = 0;
    std::size_t refreshed_level = 0;
    std::size_t tail_level = 0;
    std::size_t refreshed_noise_scale_degree = 0;
    Timings timings;
};

class EncryptedRefreshEvaluator {
  public:
    EncryptedRefreshEvaluator(Cc context, RefreshMode mode)
        : cc_(std::move(context)), mode_(mode) {}

    EvaluationResult evaluate(Ct encrypted_activation) {
        EvaluationResult result;
        result.input_level = encrypted_activation->GetLevel();

        const auto conditioning_start = Clock::now();
        std::vector<double> inverse_bound(SLOTS, 1.0 / PUBLIC_BOUND);
        auto scale = cc_->MakeCKKSPackedPlaintext(
            inverse_bound, 1, static_cast<std::uint32_t>(result.input_level),
            nullptr, SLOTS);
        auto conditioned = cc_->EvalMult(encrypted_activation, scale);
        cc_->RescaleInPlace(conditioned);
        cc_->Synchronize();
        result.timings.conditioning_seconds =
            elapsed_seconds(conditioning_start);
        result.conditioned_level = conditioned->GetLevel();

        Ct refreshed;
        if (mode_ == RefreshMode::NativeGpu) {
            const auto bootstrap_start = Clock::now();
            refreshed = cc_->EvalBootstrap(conditioned, 1, 0, false);
            cc_->Synchronize();
            result.timings.bootstrap_seconds =
                elapsed_seconds(bootstrap_start);
        } else {
            refreshed = cpu_iterative_refresh(conditioned, result.timings);
        }
        result.refreshed_level = refreshed->GetLevel();
        result.refreshed_noise_scale_degree = refreshed->GetNoiseScaleDeg();

        const auto tail_start = Clock::now();
        auto square = cc_->EvalSquare(refreshed);
        cc_->RescaleInPlace(square);
        auto tail = cc_->EvalAdd(square, LAYERNORM_EPSILON);
        cc_->Synchronize();
        result.timings.tail_seconds = elapsed_seconds(tail_start);
        result.tail_level = tail->GetLevel();
        result.tail = std::move(tail);
        return result;
    }

  private:
    OfheCt download_without_readout(const Ct& input, double& seconds) {
        const auto started = Clock::now();
        cc_->Synchronize();

        const auto& cpu_template =
            std::any_cast<const OfheCt&>(input->cpu);
        OfheCt cpu_copy = cpu_template->Clone();
        if (input->loaded) {
            auto gpu_ciphertext = std::static_pointer_cast<F::Ciphertext>(
                cc_->GetDeviceCiphertext(input->gpu));
            F::RawCipherText raw;
            gpu_ciphertext->store(raw);
            const std::size_t capacity =
                cpu_copy->GetElements()[0].GetAllElements().size();
            if (capacity < raw.numRes) {
                throw std::runtime_error(
                    "CPU ciphertext template lacks interop limb capacity");
            }
            F::GetOpenFHECipherText(cpu_copy, raw);
        }
        seconds = elapsed_seconds(started);
        return cpu_copy;
    }

    Ct upload_without_readout(OfheCt cpu_ciphertext, double& seconds) {
        const auto started = Clock::now();
        Ct result =
            std::make_shared<CiphertextImpl<DCRTPoly>>(Cc(cc_));
        result->cpu = std::make_any<OfheCt>(std::move(cpu_ciphertext));
        result->loaded = false;
        result->gpu = 0;
        result->parent_context = cc_;
        cc_->LoadCiphertext(result);
        cc_->Synchronize();
        seconds = elapsed_seconds(started);
        return result;
    }

    Ct cpu_iterative_refresh(const Ct& input, Timings& timings) {
        OfheCt cpu_input =
            download_without_readout(input, timings.download_seconds);
        auto& cpu_context = std::any_cast<OfheCc&>(cc_->cpu);

        const auto bootstrap_start = Clock::now();
        OfheCt refreshed =
            cpu_context->EvalBootstrap(cpu_input, CPU_ITERATIONS, CPU_PRECISION);
        while (refreshed->GetNoiseScaleDeg() > 1) {
            cpu_context->ModReduceInPlace(refreshed);
        }
        timings.bootstrap_seconds = elapsed_seconds(bootstrap_start);
        return upload_without_readout(std::move(refreshed),
                                      timings.upload_seconds);
    }

    Cc cc_;
    RefreshMode mode_;
};

struct Metrics {
    double global_rel_inf = std::numeric_limits<double>::infinity();
    double worst_token_rel_inf = std::numeric_limits<double>::infinity();
    double max_abs_error = std::numeric_limits<double>::infinity();
    double raw_min = std::numeric_limits<double>::infinity();
    double raw_max = -std::numeric_limits<double>::infinity();
    double raw_abs_max = 0.0;
    bool all_finite = false;
    bool domain_covered = false;
    bool depth_restored = false;
    bool tail_consumed_level = false;
    bool passed = false;
};

Metrics measure(const std::vector<double>& actual,
                const std::vector<double>& raw_reference,
                const EvaluationResult& evaluation) {
    if (actual.size() < SLOTS || raw_reference.size() != ACTIVE_VALUES) {
        throw std::invalid_argument("metric vectors violate the D768/T2 contract");
    }
    Metrics metrics;
    metrics.max_abs_error = 0.0;
    metrics.worst_token_rel_inf = 0.0;
    metrics.all_finite =
        std::all_of(actual.begin(), actual.begin() + SLOTS,
                    [](double value) { return std::isfinite(value); });
    double global_denominator = 0.0;

    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t dim = 0; dim < D; ++dim) {
            const std::size_t index = token * D + dim;
            const double scaled = raw_reference[index] / PUBLIC_BOUND;
            const double expected =
                scaled * scaled + LAYERNORM_EPSILON;
            const double observed = actual[index];
            metrics.raw_min = std::min(metrics.raw_min, raw_reference[index]);
            metrics.raw_max = std::max(metrics.raw_max, raw_reference[index]);
            metrics.raw_abs_max =
                std::max(metrics.raw_abs_max, std::abs(raw_reference[index]));
            const double error = std::abs(observed - expected);
            token_error = std::max(token_error, error);
            token_denominator =
                std::max(token_denominator, std::abs(expected));
            metrics.max_abs_error =
                std::max(metrics.max_abs_error, error);
            global_denominator =
                std::max(global_denominator, std::abs(expected));
        }
        metrics.worst_token_rel_inf =
            std::max(metrics.worst_token_rel_inf,
                     token_error / (token_denominator + 1e-15));
    }

    metrics.global_rel_inf =
        metrics.max_abs_error / (global_denominator + 1e-15);
    metrics.domain_covered = metrics.raw_abs_max <= PUBLIC_BOUND;
    metrics.depth_restored =
        evaluation.refreshed_level < evaluation.conditioned_level;
    metrics.tail_consumed_level =
        evaluation.tail_level > evaluation.refreshed_level;
    metrics.passed =
        metrics.all_finite && metrics.domain_covered &&
        metrics.depth_restored && metrics.tail_consumed_level &&
        metrics.global_rel_inf <= TOL &&
        metrics.worst_token_rel_inf <= TOL;
    return metrics;
}

std::string make_json(const Options& options, std::uint32_t ring,
                      const EvaluationResult& evaluation,
                      const Metrics& metrics, double setup_seconds,
                      double encryption_seconds, double decrypt_seconds) {
    const bool native = options.mode == RefreshMode::NativeGpu;
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"DNAGPT real-width encrypted refresh and post-refresh nonlinear tail\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA with patched OpenFHE 1.5.1\",\n"
        << "  \"backend_commit\": \"" << json_escape(options.backend_commit) << "\",\n"
        << "  \"patched_openfhe_commit\": \"" << PINNED_OPENFHE_COMMIT << "\",\n"
        << "  \"backend_patch_sha256\": \""
        << json_escape(options.backend_patch_sha256) << "\",\n"
        << "  \"container_image\": \"" << json_escape(options.container_image) << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \"" << json_escape(options.source_sha256) << "\",\n"
        << "  \"activation_sha256\": \"" << json_escape(options.activation_sha256) << "\",\n"
        << "  \"activation_file\": \"" << json_escape(options.activation.filename().string()) << "\",\n"
        << "  \"mode\": \"" << mode_name(options.mode) << "\",\n"
        << "  \"bootstrap_location\": \""
        << (native ? "GPU/CUDA" : "CPU/OpenFHE same-lineage interop") << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": " << D << ",\n"
        << "    \"T\": " << T << ",\n"
        << "    \"active_values\": " << ACTIVE_VALUES << ",\n"
        << "    \"ckks_slots_power_of_two\": " << SLOTS << ",\n"
        << "    \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << DEPTH << ",\n"
        << "    \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"first_mod_bits\": " << FIRST_MOD_BITS << ",\n"
        << "    \"scaling_technique\": \"FIXEDMANUAL\",\n"
        << "    \"level_budget\": [" << LEVEL_BUDGET[0] << ", "
        << LEVEL_BUDGET[1] << "],\n"
        << "    \"public_bootstrap_bound\": " << PUBLIC_BOUND << ",\n"
        << "    \"raw_activation_range\": [" << metrics.raw_min << ", "
        << metrics.raw_max << "],\n"
        << "    \"raw_activation_abs_max\": " << metrics.raw_abs_max << ",\n"
        << "    \"bootstrap_iterations_requested\": "
        << (native ? 1 : CPU_ITERATIONS) << ",\n"
        << "    \"bootstrap_precision_requested\": "
        << (native ? 0 : CPU_PRECISION) << ",\n"
        << "    \"native_gpu_ignores_iteration_and_precision_arguments\": "
        << (native ? "true" : "false") << ",\n"
        << "    \"post_refresh_tail\": \"square(scaled activation) + layernorm epsilon\",\n"
        << "    \"layernorm_epsilon\": " << LAYERNORM_EPSILON << "\n"
        << "  },\n"
        << "  \"levels\": {\n"
        << "    \"encrypted_input\": " << evaluation.input_level << ",\n"
        << "    \"conditioned_before_bootstrap\": " << evaluation.conditioned_level << ",\n"
        << "    \"after_bootstrap\": " << evaluation.refreshed_level << ",\n"
        << "    \"after_nonlinear_tail\": " << evaluation.tail_level << ",\n"
        << "    \"restored\": "
        << (metrics.depth_restored ? "true" : "false") << ",\n"
        << "    \"tail_consumed_level\": "
        << (metrics.tail_consumed_level ? "true" : "false") << ",\n"
        << "    \"refreshed_noise_scale_degree\": "
        << evaluation.refreshed_noise_scale_degree << "\n"
        << "  },\n"
        << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << "  \"worst_token_rel_inf\": " << metrics.worst_token_rel_inf << ",\n"
        << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << "  \"all_finite\": " << (metrics.all_finite ? "true" : "false") << ",\n"
        << "  \"public_domain_covered\": "
        << (metrics.domain_covered ? "true" : "false") << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (metrics.passed ? "true" : "false") << ",\n"
        << "  \"timings_seconds\": {\n"
        << "    \"context_bootstrap_setup_keygen_load\": " << setup_seconds << ",\n"
        << "    \"encrypt_real_block_activation\": " << encryption_seconds << ",\n"
        << "    \"encrypted_public_conditioning\": "
        << evaluation.timings.conditioning_seconds << ",\n"
        << "    \"ciphertext_download_no_decrypt\": "
        << evaluation.timings.download_seconds << ",\n"
        << "    \"bootstrap\": " << evaluation.timings.bootstrap_seconds << ",\n"
        << "    \"ciphertext_upload_no_decrypt\": "
        << evaluation.timings.upload_seconds << ",\n"
        << "    \"interop_transfer_total\": "
        << evaluation.timings.download_seconds + evaluation.timings.upload_seconds << ",\n"
        << "    \"post_refresh_gpu_tail\": "
        << evaluation.timings.tail_seconds << ",\n"
        << "    \"final_decrypt\": " << decrypt_seconds << "\n"
        << "  },\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"final_decrypt_calls\": 1,\n"
        << "  \"evaluator_has_private_key\": false,\n"
        << "  \"same_crypto_context_and_key_lineage\": true\n"
        << "}\n";
    return out.str();
}

void write_exclusive(const std::filesystem::path& path,
                     const std::string& contents) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        throw std::runtime_error("could not exclusively create evidence " +
                                 path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t written =
            ::write(fd, contents.data() + offset, contents.size() - offset);
        if (written <= 0) {
            ::close(fd);
            throw std::runtime_error("short write to evidence " + path.string());
        }
        offset += static_cast<std::size_t>(written);
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("could not close evidence " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<double> activation =
            load_activation(options.activation);

        const auto setup_start = Clock::now();
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
        parameters.SetSecretKeyDist(UNIFORM_TERNARY);
        parameters.SetMultiplicativeDepth(DEPTH);
        parameters.SetScalingModSize(SCALE_BITS);
        parameters.SetFirstModSize(FIRST_MOD_BITS);
        parameters.SetScalingTechnique(FIXEDMANUAL);
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
            {LEVEL_BUDGET[0], LEVEL_BUDGET[1]}, {0, 0}, SLOTS, 0);

        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS);
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        const double setup_seconds = elapsed_seconds(setup_start);

        const std::uint32_t ring = cc->GetRingDimension();
        if (ring != 131072) {
            throw std::runtime_error("secure refresh profile requires ring dimension 131072");
        }
        std::cout << "[context] security=HEStd_128_classic ring=" << ring
                  << " depth=" << DEPTH << " slots=" << SLOTS
                  << " mode=" << mode_name(options.mode) << '\n';

        const auto encryption_start = Clock::now();
        std::vector<double> packed_activation(SLOTS, 0.0);
        std::copy(activation.begin(), activation.end(),
                  packed_activation.begin());
        auto plaintext = cc->MakeCKKSPackedPlaintext(
            packed_activation, 1, DEPTH - 2, nullptr, SLOTS);
        Ct encrypted = cc->Encrypt(keys.publicKey, plaintext);
        cc->Synchronize();
        const double encryption_seconds =
            elapsed_seconds(encryption_start);

        EncryptedRefreshEvaluator evaluator(cc, options.mode);
        EvaluationResult evaluation = evaluator.evaluate(encrypted);

        const auto decrypt_start = Clock::now();
        Plaintext decoded;
        cc->Decrypt(keys.secretKey, evaluation.tail, &decoded);
        decoded->SetLength(SLOTS);
        const auto raw = decoded->GetRealPackedValue();
        const std::vector<double> actual(raw.begin(), raw.begin() + SLOTS);
        const double decrypt_seconds = elapsed_seconds(decrypt_start);

        const Metrics metrics = measure(actual, activation, evaluation);
        write_exclusive(
            options.output,
            make_json(options, ring, evaluation, metrics, setup_seconds,
                      encryption_seconds, decrypt_seconds));

        std::cout << "[levels] input=" << evaluation.input_level
                  << " conditioned=" << evaluation.conditioned_level
                  << " refreshed=" << evaluation.refreshed_level
                  << " tail=" << evaluation.tail_level << '\n';
        std::cout << "[metric] global_rel_inf=" << std::scientific
                  << metrics.global_rel_inf
                  << " worst_token_rel_inf=" << metrics.worst_token_rel_inf
                  << " tol=" << TOL
                  << " finite=" << metrics.all_finite << '\n';
        std::cout << "[time] bootstrap=" << std::fixed << std::setprecision(3)
                  << evaluation.timings.bootstrap_seconds
                  << "s download=" << evaluation.timings.download_seconds
                  << "s upload=" << evaluation.timings.upload_seconds
                  << "s tail=" << evaluation.timings.tail_seconds << "s\n";
        std::cout << "[evidence] " << options.output << '\n';
        std::cout << (metrics.passed ? "FIDES_REFRESH_GATE_PASS"
                                    : "FIDES_REFRESH_GATE_FAIL")
                  << '\n';
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
