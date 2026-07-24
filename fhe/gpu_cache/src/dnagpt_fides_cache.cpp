// Public evaluation-key cache provision/reload parity gate for DNAGPT/FIDESlib.
//
// The cache contains no private key. Provision writes the oracle/client private
// key only to an explicit path outside the cache. Reload imports the context,
// public key, eval-mult key, and all full-real-gate rotation keys before calling
// LoadContext. The evaluator receives no private key and cannot decrypt.

#include <fideslib.hpp>

#include "generated_rotation_contract.hpp"

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
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace fideslib;

namespace {

constexpr std::uint32_t MULT_DEPTH = 43;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 4;
constexpr std::uint32_t SLOTS = 4096;
constexpr double TOL = 4e-2;
constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";

constexpr std::string_view CONTEXT_FILE = "crypto-context.bin";
constexpr std::string_view PUBLIC_KEY_FILE = "public-key.bin";
constexpr std::string_view EVAL_MULT_FILE = "eval-mult.bin";
constexpr std::string_view EVAL_AUTOMORPHISM_FILE = "eval-automorphism.bin";

using Clock = std::chrono::steady_clock;
using Cc = CryptoContext<DCRTPoly>;
using Ct = Ciphertext<DCRTPoly>;

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
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(ch)
                           << std::dec;
                } else {
                    output << ch;
                }
        }
    }
    return output.str();
}

void write_exclusive(const std::filesystem::path& path,
                     const std::string& contents, mode_t mode = 0644) {
    if (path.has_parent_path() &&
        !std::filesystem::is_directory(path.parent_path())) {
        throw std::runtime_error("output parent directory missing: " +
                                 path.parent_path().string());
    }
    const int descriptor =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, mode);
    if (descriptor < 0) {
        throw std::runtime_error("refusing to overwrite output: " +
                                 path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t count =
            ::write(descriptor, contents.data() + offset,
                    contents.size() - offset);
        if (count <= 0) {
            ::close(descriptor);
            throw std::runtime_error("short write: " + path.string());
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0 || ::close(descriptor) != 0) {
        throw std::runtime_error("could not commit output: " + path.string());
    }
}

bool path_is_within(const std::filesystem::path& child,
                    const std::filesystem::path& parent) {
    const auto normalize = [](const std::filesystem::path& path) {
        const auto absolute = std::filesystem::absolute(path);
        const auto canonical_parent =
            std::filesystem::weakly_canonical(absolute.parent_path());
        return (canonical_parent / absolute.filename()).lexically_normal();
    };
    const auto normalized_child = normalize(child);
    const auto normalized_parent = normalize(parent);
    auto child_part = normalized_child.begin();
    for (auto parent_part = normalized_parent.begin();
         parent_part != normalized_parent.end(); ++parent_part, ++child_part) {
        if (child_part == normalized_child.end() ||
            *child_part != *parent_part) {
            return false;
        }
    }
    return true;
}

std::vector<std::int32_t> full_rotations() {
    return {dnagpt_cache_contract::kFullRotations.begin(),
            dnagpt_cache_contract::kFullRotations.end()};
}

Cc make_context(int gpu) {
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
    parameters.SetDevices({gpu});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    Cc context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    return context;
}

struct Options {
    std::string command;
    int gpu = 0;
    std::filesystem::path cache_dir;
    std::filesystem::path oracle_key;
    std::filesystem::path provision_metadata;
    std::filesystem::path output;
    std::string manifest_sha256;
    std::string container_image = "UNSPECIFIED";
    std::string source_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nprovision: dnagpt_fides_cache provision --gpu N --cache-dir PATH "
        "--oracle-key-output PATH --metadata-output PATH\n"
        "reload: dnagpt_fides_cache reload --gpu N --cache-dir PATH "
        "--oracle-key PATH --output PATH --manifest-sha256 SHA "
        "--container-image IMAGE@sha256:DIGEST --source-sha256 SHA");
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        usage_error("missing provision|reload command");
    }
    Options options;
    options.command = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) {
                usage_error("missing value for " + argument);
            }
            return argv[index];
        };
        if (argument == "--gpu") {
            options.gpu = std::stoi(next());
        } else if (argument == "--cache-dir") {
            options.cache_dir = next();
        } else if (argument == "--oracle-key-output" ||
                   argument == "--oracle-key") {
            options.oracle_key = next();
        } else if (argument == "--metadata-output") {
            options.provision_metadata = next();
        } else if (argument == "--output") {
            options.output = next();
        } else if (argument == "--manifest-sha256") {
            options.manifest_sha256 = next();
        } else if (argument == "--container-image") {
            options.container_image = next();
        } else if (argument == "--source-sha256") {
            options.source_sha256 = next();
        } else {
            usage_error("unknown option: " + argument);
        }
    }
    if (options.command != "provision" && options.command != "reload") {
        usage_error("command must be provision or reload");
    }
    if (options.gpu < 0 || options.cache_dir.empty() ||
        options.oracle_key.empty()) {
        usage_error("--gpu, --cache-dir, and oracle-key path are required");
    }
    if (path_is_within(options.oracle_key, options.cache_dir)) {
        usage_error("oracle/client private key path must be outside cache");
    }
    if (options.command == "provision" &&
        options.provision_metadata.empty()) {
        usage_error("provision requires --metadata-output");
    }
    if (options.command == "reload" &&
        (options.output.empty() || options.manifest_sha256.size() != 64 ||
         options.container_image == "UNSPECIFIED" ||
         options.source_sha256.size() != 64)) {
        usage_error(
            "reload requires output, manifest/source SHA-256, and exact image");
    }
    return options;
}

void require_serialization(bool result, const std::string& what) {
    if (!result) {
        throw std::runtime_error("serialization failed: " + what);
    }
}

void provision(const Options& options) {
    if (std::filesystem::exists(options.cache_dir)) {
        throw std::runtime_error("refusing existing cache directory: " +
                                 options.cache_dir.string());
    }
    if (std::filesystem::exists(options.oracle_key) ||
        std::filesystem::exists(options.provision_metadata)) {
        throw std::runtime_error(
            "refusing existing oracle key or metadata output");
    }
    if (!std::filesystem::create_directory(options.cache_dir)) {
        throw std::runtime_error("could not create cache directory");
    }

    const auto context_start = Clock::now();
    Cc context = make_context(options.gpu);
    const double context_seconds = elapsed_seconds(context_start);

    const auto keygen_start = Clock::now();
    auto keys = context->KeyGen();
    const double keygen_seconds = elapsed_seconds(keygen_start);

    const auto eval_mult_start = Clock::now();
    context->EvalMultKeyGen(keys.secretKey);
    const double eval_mult_seconds = elapsed_seconds(eval_mult_start);

    const auto rotations_start = Clock::now();
    const std::vector<std::int32_t> rotations = full_rotations();
    context->EvalRotateKeyGen(keys.secretKey, rotations);
    const double rotations_seconds = elapsed_seconds(rotations_start);

    const auto serialization_start = Clock::now();
    require_serialization(
        Serial::SerializeToFile(
            (options.cache_dir / CONTEXT_FILE).string(), context,
            SerType::BINARY),
        "crypto context");
    require_serialization(
        Serial::SerializeToFile(
            (options.cache_dir / PUBLIC_KEY_FILE).string(), keys.publicKey,
            SerType::BINARY),
        "public key");
    {
        std::ofstream stream(options.cache_dir / EVAL_MULT_FILE,
                             std::ios::out | std::ios::binary);
        if (!stream ||
            !context->SerializeEvalMultKey(stream, SerType::BINARY)) {
            throw std::runtime_error("serialization failed: eval-mult key");
        }
    }
    {
        std::ofstream stream(options.cache_dir / EVAL_AUTOMORPHISM_FILE,
                             std::ios::out | std::ios::binary);
        if (!stream ||
            !context->SerializeEvalAutomorphismKey(stream, SerType::BINARY)) {
            throw std::runtime_error(
                "serialization failed: eval-automorphism keys");
        }
    }
    const double public_serialization_seconds =
        elapsed_seconds(serialization_start);

    // Client-only oracle material. This path is required to be outside cache.
    const auto oracle_start = Clock::now();
    require_serialization(
        Serial::SerializeToFile(options.oracle_key.string(), keys.secretKey,
                                SerType::BINARY),
        "client oracle key");
    if (::chmod(options.oracle_key.c_str(), 0600) != 0) {
        throw std::runtime_error("could not restrict client oracle key mode");
    }
    const double oracle_serialization_seconds = elapsed_seconds(oracle_start);

    std::ostringstream metadata;
    metadata << std::setprecision(17)
             << "{\n"
             << "  \"rotation_count\": " << rotations.size() << ",\n"
             << "  \"private_key_cached\": false,\n"
             << "  \"timings_seconds\": {\n"
             << "    \"context_generation\": " << context_seconds << ",\n"
             << "    \"keygen\": " << keygen_seconds << ",\n"
             << "    \"eval_mult_keygen\": " << eval_mult_seconds << ",\n"
             << "    \"eval_automorphism_keygen\": " << rotations_seconds
             << ",\n"
             << "    \"public_cache_serialization\": "
             << public_serialization_seconds << ",\n"
             << "    \"client_oracle_key_serialization\": "
             << oracle_serialization_seconds << "\n"
             << "  }\n"
             << "}\n";
    write_exclusive(options.provision_metadata, metadata.str(), 0600);
    std::cout << "FIDES_PUBLIC_CACHE_PROVISIONED rotation_keys="
              << rotations.size() << " private_key_cached=false\n";
}

struct EvaluationResult {
    Ct packed;
    std::size_t output_level = 0;
    std::size_t rotations_exercised = 0;
    std::size_t ciphertext_multiplications = 0;
};

class ReloadEvaluator {
  public:
    ReloadEvaluator(Cc context, std::vector<std::int32_t> rotations)
        : context_(std::move(context)), rotations_(std::move(rotations)) {}

    EvaluationResult evaluate(const Ct& encrypted_input) {
        // Slot 63 validates the cached eval-mult/relinearization key.
        Ct product = context_->EvalMult(encrypted_input, encrypted_input);
        product = select_slot(product, rotations_.size());
        const std::size_t target_level = product->GetLevel();
        Ct packed = product;

        // Slots 0..62 independently validate every exact full-gate rotation.
        Plaintext ones = plaintext_with_value(1.0);
        for (std::size_t index = 0; index < rotations_.size(); ++index) {
            Ct rotated = context_->EvalRotate(encrypted_input, rotations_[index]);
            Ct selected = select_slot(rotated, index);
            while (selected->GetLevel() < target_level) {
                selected = context_->EvalMult(selected, ones);
            }
            if (selected->GetLevel() != target_level) {
                throw std::runtime_error(
                    "could not align packed parity ciphertext levels");
            }
            packed = context_->EvalAdd(packed, selected);
        }
        return {
            .packed = packed,
            .output_level = packed->GetLevel(),
            .rotations_exercised = rotations_.size(),
            .ciphertext_multiplications = 1,
        };
    }

  private:
    Plaintext plaintext_with_value(double value) {
        return context_->MakeCKKSPackedPlaintext(
            std::vector<double>(SLOTS, value), 1, 0, nullptr, SLOTS);
    }

    Ct select_slot(const Ct& input, std::size_t slot) {
        if (slot >= SLOTS) {
            throw std::invalid_argument("parity output slot outside batch");
        }
        std::vector<double> mask(SLOTS);
        mask[slot] = 1.0;
        Plaintext selector = context_->MakeCKKSPackedPlaintext(
            mask, 1, 0, nullptr, SLOTS);
        return context_->EvalMult(input, selector);
    }

    Cc context_;
    std::vector<std::int32_t> rotations_;
};

struct Metrics {
    double global_rel_inf = std::numeric_limits<double>::infinity();
    double worst_rotation_rel_error =
        std::numeric_limits<double>::infinity();
    double max_rotation_abs_error = std::numeric_limits<double>::infinity();
    double multiplication_rel_error =
        std::numeric_limits<double>::infinity();
    double multiplication_abs_error =
        std::numeric_limits<double>::infinity();
    double max_unselected_abs = std::numeric_limits<double>::infinity();
    bool all_finite = false;
    bool passed = false;
};

double probe_input(std::size_t slot) {
    return 0.25 +
           0.5 * static_cast<double>(slot) / static_cast<double>(SLOTS - 1);
}

Metrics measure(const std::vector<double>& actual,
                const std::vector<std::int32_t>& rotations) {
    if (actual.size() < SLOTS || rotations.size() >= SLOTS) {
        throw std::invalid_argument("parity output shape mismatch");
    }
    Metrics metrics;
    metrics.all_finite =
        std::all_of(actual.begin(), actual.begin() + SLOTS,
                    [](double value) { return std::isfinite(value); });
    metrics.worst_rotation_rel_error = 0.0;
    metrics.max_rotation_abs_error = 0.0;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    for (std::size_t index = 0; index < rotations.size(); ++index) {
        const std::size_t source =
            (index + static_cast<std::size_t>(rotations[index])) % SLOTS;
        const double reference = probe_input(source);
        const double error = std::abs(actual[index] - reference);
        metrics.max_rotation_abs_error =
            std::max(metrics.max_rotation_abs_error, error);
        metrics.worst_rotation_rel_error =
            std::max(metrics.worst_rotation_rel_error,
                     error / std::max(std::abs(reference), 1e-15));
        maximum_error = std::max(maximum_error, error);
        maximum_reference =
            std::max(maximum_reference, std::abs(reference));
    }

    const std::size_t multiplication_slot = rotations.size();
    const double multiplication_reference =
        probe_input(multiplication_slot) * probe_input(multiplication_slot);
    metrics.multiplication_abs_error =
        std::abs(actual[multiplication_slot] - multiplication_reference);
    metrics.multiplication_rel_error =
        metrics.multiplication_abs_error /
        std::max(std::abs(multiplication_reference), 1e-15);
    maximum_error =
        std::max(maximum_error, metrics.multiplication_abs_error);
    maximum_reference =
        std::max(maximum_reference, std::abs(multiplication_reference));

    metrics.max_unselected_abs = 0.0;
    for (std::size_t slot = multiplication_slot + 1; slot < SLOTS; ++slot) {
        metrics.max_unselected_abs =
            std::max(metrics.max_unselected_abs, std::abs(actual[slot]));
    }
    maximum_error = std::max(maximum_error, metrics.max_unselected_abs);
    metrics.global_rel_inf =
        maximum_error / std::max(maximum_reference, 1e-15);
    metrics.passed =
        metrics.all_finite && metrics.global_rel_inf <= TOL &&
        metrics.worst_rotation_rel_error <= TOL &&
        metrics.multiplication_rel_error <= TOL &&
        metrics.max_unselected_abs <= TOL;
    return metrics;
}

std::string evidence_json(
    const Options& options, std::uint32_t ring,
    const std::vector<std::int32_t>& rotations,
    const EvaluationResult& evaluation, const Metrics& metrics,
    double context_seconds, double public_key_seconds,
    double eval_mult_seconds, double eval_automorphism_seconds,
    double load_context_seconds, double encryption_seconds,
    double evaluation_seconds, double oracle_key_seconds,
    double decrypt_seconds) {
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"task\": \"FIDESlib public cache clean-process reload parity\",\n"
           << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
           << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
           << "  \"backend_commit\": \"" << PINNED_FIDES_COMMIT << "\",\n"
           << "  \"container_image\": \""
           << json_escape(options.container_image) << "\",\n"
           << "  \"source_sha256\": \"" << json_escape(options.source_sha256)
           << "\",\n"
           << "  \"manifest_sha256\": \""
           << json_escape(options.manifest_sha256) << "\",\n"
           << "  \"real_source_sha256\": \""
           << dnagpt_cache_contract::kRealSourceSha256 << "\",\n"
           << "  \"gpu\": " << options.gpu << ",\n"
           << "  \"config\": {\n"
           << "    \"security\": \"HEStd_128_classic\",\n"
           << "    \"ring_dim\": " << ring << ",\n"
           << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
           << "    \"scale_bits\": " << SCALE_BITS << ",\n"
           << "    \"first_mod_bits\": " << FIRST_MOD_BITS << ",\n"
           << "    \"large_digits\": " << LARGE_DIGITS << ",\n"
           << "    \"batch_slots\": " << SLOTS << ",\n"
           << "    \"rotation_count\": " << rotations.size() << ",\n"
           << "    \"rotation_steps\": [";
    for (std::size_t index = 0; index < rotations.size(); ++index) {
        output << rotations[index]
               << (index + 1 == rotations.size() ? "" : ", ");
    }
    output << "],\n"
           << "    \"output_level\": " << evaluation.output_level << ",\n"
           << "    \"ciphertext_multiplications\": "
           << evaluation.ciphertext_multiplications << "\n"
           << "  },\n"
           << "  \"rotations_exercised\": "
           << evaluation.rotations_exercised << ",\n"
           << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
           << "  \"worst_rotation_rel_error\": "
           << metrics.worst_rotation_rel_error << ",\n"
           << "  \"max_rotation_abs_error\": "
           << metrics.max_rotation_abs_error << ",\n"
           << "  \"multiplication_rel_error\": "
           << metrics.multiplication_rel_error << ",\n"
           << "  \"multiplication_abs_error\": "
           << metrics.multiplication_abs_error << ",\n"
           << "  \"max_unselected_abs\": "
           << metrics.max_unselected_abs << ",\n"
           << "  \"all_finite\": "
           << (metrics.all_finite ? "true" : "false") << ",\n"
           << "  \"tol\": " << TOL << ",\n"
           << "  \"passed\": " << (metrics.passed ? "true" : "false")
           << ",\n"
           << "  \"timings_seconds\": {\n"
           << "    \"deserialize_context\": " << context_seconds << ",\n"
           << "    \"deserialize_public_key\": " << public_key_seconds
           << ",\n"
           << "    \"deserialize_eval_mult\": " << eval_mult_seconds << ",\n"
           << "    \"deserialize_eval_automorphism\": "
           << eval_automorphism_seconds << ",\n"
           << "    \"gpu_load_context\": " << load_context_seconds << ",\n"
           << "    \"encrypt_probe\": " << encryption_seconds << ",\n"
           << "    \"encrypted_parity_evaluation\": " << evaluation_seconds
           << ",\n"
           << "    \"deserialize_client_oracle_key_after_evaluation\": "
           << oracle_key_seconds << ",\n"
           << "    \"final_oracle_decrypt\": " << decrypt_seconds << "\n"
           << "  },\n"
           << "  \"clean_process_reload\": true,\n"
           << "  \"cache_private_key_files\": 0,\n"
           << "  \"intermediate_decrypt_attempts\": 0,\n"
           << "  \"final_decrypt_calls\": 1,\n"
           << "  \"evaluator_has_private_key\": false,\n"
           << "  \"load_order\": \"context -> public -> eval-mult -> "
              "eval-automorphism -> LoadContext\"\n"
           << "}\n";
    return output.str();
}

bool reload(const Options& options) {
    if (!std::filesystem::is_directory(options.cache_dir) ||
        !std::filesystem::is_regular_file(options.oracle_key) ||
        std::filesystem::exists(options.output)) {
        throw std::runtime_error(
            "reload inputs missing or immutable output already exists");
    }
    const std::vector<std::int32_t> rotations = full_rotations();

    // The order below is load-bearing: the process-global OpenFHE eval-key
    // maps must be populated before FIDESlib copies keys to GPU.
    const auto context_start = Clock::now();
    Cc context;
    require_serialization(
        Serial::DeserializeFromFile(
            (options.cache_dir / CONTEXT_FILE).string(), context,
            SerType::BINARY),
        "crypto context deserialize");
    context->SetDevices({options.gpu});
    const double context_seconds = elapsed_seconds(context_start);

    const auto public_key_start = Clock::now();
    PublicKey<DCRTPoly> public_key;
    require_serialization(
        Serial::DeserializeFromFile(
            (options.cache_dir / PUBLIC_KEY_FILE).string(), public_key,
            SerType::BINARY),
        "public key deserialize");
    const double public_key_seconds = elapsed_seconds(public_key_start);

    const auto eval_mult_start = Clock::now();
    {
        std::ifstream stream(options.cache_dir / EVAL_MULT_FILE,
                             std::ios::in | std::ios::binary);
        if (!stream ||
            !context->DeserializeEvalMultKey(stream, SerType::BINARY)) {
            throw std::runtime_error("eval-mult deserialize failed");
        }
    }
    const double eval_mult_seconds = elapsed_seconds(eval_mult_start);

    const auto eval_automorphism_start = Clock::now();
    {
        std::ifstream stream(options.cache_dir / EVAL_AUTOMORPHISM_FILE,
                             std::ios::in | std::ios::binary);
        if (!stream ||
            !context->DeserializeEvalAutomorphismKey(stream,
                                                     SerType::BINARY)) {
            throw std::runtime_error(
                "eval-automorphism deserialize failed");
        }
    }
    const double eval_automorphism_seconds =
        elapsed_seconds(eval_automorphism_start);

    const auto load_context_start = Clock::now();
    context->LoadContext(public_key);
    context->Synchronize();
    const double load_context_seconds = elapsed_seconds(load_context_start);
    const std::uint32_t ring = context->GetRingDimension();

    const auto encryption_start = Clock::now();
    std::vector<double> input(SLOTS);
    for (std::size_t slot = 0; slot < SLOTS; ++slot) {
        input[slot] = probe_input(slot);
    }
    Plaintext plaintext = context->MakeCKKSPackedPlaintext(
        input, 1, 0, nullptr, SLOTS);
    Ct encrypted_input = context->Encrypt(public_key, plaintext);
    context->Synchronize();
    const double encryption_seconds = elapsed_seconds(encryption_start);

    // The evaluator is deliberately constructed without any private key.
    ReloadEvaluator evaluator(context, rotations);
    const auto evaluation_start = Clock::now();
    EvaluationResult evaluation = evaluator.evaluate(encrypted_input);
    context->Synchronize();
    const double evaluation_seconds = elapsed_seconds(evaluation_start);

    // Client/oracle material is loaded only after encrypted evaluation ends.
    const auto oracle_key_start = Clock::now();
    PrivateKey<DCRTPoly> oracle_key;
    require_serialization(
        Serial::DeserializeFromFile(options.oracle_key.string(), oracle_key,
                                    SerType::BINARY),
        "client oracle key deserialize");
    const double oracle_key_seconds = elapsed_seconds(oracle_key_start);

    const auto decrypt_start = Clock::now();
    Plaintext decoded;
    context->Decrypt(oracle_key, evaluation.packed, &decoded);
    decoded->SetLength(SLOTS);
    const auto packed_values = decoded->GetRealPackedValue();
    const std::vector<double> actual(
        packed_values.begin(),
        packed_values.begin() + static_cast<std::ptrdiff_t>(SLOTS));
    const double decrypt_seconds = elapsed_seconds(decrypt_start);

    const Metrics metrics = measure(actual, rotations);
    const std::string evidence = evidence_json(
        options, ring, rotations, evaluation, metrics, context_seconds,
        public_key_seconds, eval_mult_seconds, eval_automorphism_seconds,
        load_context_seconds, encryption_seconds, evaluation_seconds,
        oracle_key_seconds, decrypt_seconds);
    write_exclusive(options.output, evidence);
    std::cout << evidence
              << (metrics.passed ? "FIDES_CACHE_RELOAD_PARITY_PASS\n"
                                 : "FIDES_CACHE_RELOAD_PARITY_FAIL\n");
    return metrics.passed;
}

}  // namespace

int main(int argc, char** argv) {
    ::umask(0077);
    try {
        const Options options = parse_options(argc, argv);
        if (options.command == "provision") {
            provision(options);
            return 0;
        } else {
            return reload(options) ? 0 : 5;
        }
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
