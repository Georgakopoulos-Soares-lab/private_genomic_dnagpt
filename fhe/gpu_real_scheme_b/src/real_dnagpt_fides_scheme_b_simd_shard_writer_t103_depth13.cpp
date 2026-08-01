// Scheme B (hybrid client-assisted CKKS) cross-process context/key
// serialization prototype, TOKEN-SIMD-PARAMETER-MATCHED FORK -- WRITER phase
// ("process A").
//
// Scope (see docs/hybrid/tasks.md's 2026-07-31 "2-GPU process-per-GPU
// sharding" entry, Stage 1): the existing
// real_dnagpt_fides_scheme_b_serialize_writer/reader pair proves cross-process
// CryptoContext/key serialization works, but only for the original
// single-token-per-ciphertext T=2 layout (SLOTS=4096,
// required_rotation_keys_for_full() sized for that layout). The Token-SIMD
// B=8 T=103 depth-13/digits-3/ring-65536 gate
// (real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp)
// uses SLOTS=32768 and its own TOKEN_BATCH-scaled rotation-key set, so a
// worker deserializing the OLD writer's state cannot evaluate the Token-SIMD
// schedule: rotations it needs (e.g. TOKEN_BATCH*baby, lane-shift deltas)
// were never generated, and the plaintext/ciphertext batch size itself
// mismatches. This is a discovered, scoped prerequisite for any 2-GPU
// Token-SIMD shard worker -- not a small addendum to the existing reader.
//
// This file is a structural fork of real_dnagpt_fides_scheme_b_serialize_writer.cpp
// (same GenCryptoContext -> Enable(s) -> KeyGen -> EvalMultKeyGen ->
// EvalRotateKeyGen -> serialize-and-exit shape; still never calls
// LoadContext/GenCryptoContextGPU; still defines no Client/EncryptedEvaluator
// class) combined with the CCParams and required_rotation_keys() from
// real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp
// (SLOTS=32768, MULT_DEPTH=13, SCALE_BITS=50, LARGE_DIGITS=3,
// RING_DIM=65536, TOKEN_BATCH-scaled required_rotation_keys()). Neither
// parent file is edited by this one.
//
// Parent hashes (computed directly against this repo's working tree, not
// trusted from any stale prior value -- see docs/hybrid/tasks.md's 2026-07-31
// entry, finding 2, "this new writer fork ... is scoped but not yet
// written"):
//   structural parent (real_dnagpt_fides_scheme_b_serialize_writer.cpp)
//     SHA-256: ea720f1b6c49524f1b49853b67622307fd55c9fb751317fee4b30c6750f69403
//   Token-SIMD parameter source
//     (real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp)
//     SHA-256: 6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e
//
// Verified directly against FIDESlib's pinned commit
// (786c7600fb2f16b724e0acf73df367b27b8afed6, api/Serialize.hpp/.cpp,
// api/CryptoContext.hpp/.cpp) -- same properties the structural parent's file
// header already documented (EvalMultKeyGen/EvalRotateKeyGen only throw once
// `this->loaded` is true; SerializeToFile/SerializeEvalMultKey/
// SerializeEvalAutomorphismKey do not require LoadContext first). Unchanged
// by the Token-SIMD parameter swap.
//
// This writer therefore performs ONLY: GenCryptoContext -> Enable(s) ->
// KeyGen -> EvalMultKeyGen -> EvalRotateKeyGen for the Token-SIMD depth-13
// schedule's full rotation-key set -> serialize CryptoContext + PublicKey +
// PrivateKey + EvalMultKey + EvalAutomorphismKey to --state-dir, then exits.
// It never calls LoadContext/GenCryptoContextGPU (exclusively the sharded
// reader's job, see real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp)
// and never evaluates anything -- no Fixture, no Client, no
// EncryptedEvaluator exist in this translation unit at all.
//
// Secret-key hygiene (~/.agents/policies/secrets.md): identical discipline to
// the structural parent -- this writer breaks the "never persist the secret
// key" rule ONLY to simulate a restarted client process for measurement,
// explicitly a testbed simplification, not a production client/server
// key-custody design (out of scope per CLAUDE.md). The secret-key file is
// created with restrictive permissions (chmod 0600) immediately after being
// written, and its path/contents are never printed to stdout/stderr or
// recorded in the evidence JSON.

#include <fideslib.hpp>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace fideslib;

namespace {

// ---- Token-SIMD depth-13/digits-3/ring-65536 parameters, copied from
// real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp
// (pinned parent SHA-256 above). Only the constants required to reproduce
// its exact required_rotation_keys() output and CCParams are needed here --
// this writer never evaluates T, HEADS, MLP_DIM, etc.
constexpr std::size_t D = 768;
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t LOGICAL_SLOTS = COPIES * PACK_WIDTH;
constexpr std::size_t TOKEN_BATCH = 8;
constexpr std::size_t SLOTS = LOGICAL_SLOTS * TOKEN_BATCH;
constexpr std::size_t BSGS_N1 = 32;
constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;
constexpr std::uint32_t RING_DIM = 65536;
constexpr std::uint32_t MULT_DEPTH = 13;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 3;

static_assert(SLOTS == 32768);
static_assert(RING_DIM == 2 * SLOTS);
static_assert(BSGS_N1 * BSGS_N2 == PACK_WIDTH);

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
constexpr std::string_view PINNED_STRUCTURAL_PARENT_SHA256 =
    "ea720f1b6c49524f1b49853b67622307fd55c9fb751317fee4b30c6750f69403";
constexpr std::string_view PINNED_TOKEN_SIMD_PARENT_SHA256 =
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e";

using Clock = std::chrono::steady_clock;
using Cc = CryptoContext<DCRTPoly>;
using Keys = KeyPair<DCRTPoly>;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    if (::gmtime_r(&now, &utc) == nullptr) {
        throw std::runtime_error("could not obtain UTC timestamp");
    }
    char buffer[32] = {};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0) {
        throw std::runtime_error("could not format UTC timestamp");
    }
    return buffer;
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
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
    std::filesystem::path state_dir;
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string structural_parent_sha256;
    std::string token_simd_parent_sha256;
    std::string source_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13 "
        "--gpu N --state-dir PATH --output PATH "
        "--structural-parent-sha256 SHA --token-simd-parent-sha256 SHA "
        "[--backend-commit SHA] [--source-sha256 SHA]");
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
        } else if (arg == "--output") {
            options.output = next();
        } else if (arg == "--backend-commit") {
            options.backend_commit = next();
        } else if (arg == "--structural-parent-sha256") {
            options.structural_parent_sha256 = next();
        } else if (arg == "--token-simd-parent-sha256") {
            options.token_simd_parent_sha256 = next();
        } else if (arg == "--source-sha256") {
            options.source_sha256 = next();
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13 "
                         "--gpu N --state-dir PATH --output PATH\n";
            std::exit(0);
        } else {
            usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0) {
        usage_error("--gpu must be non-negative");
    }
    if (options.state_dir.empty() || options.output.empty()) {
        usage_error("--state-dir and --output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit " + options.backend_commit);
    }
    if (options.structural_parent_sha256 != PINNED_STRUCTURAL_PARENT_SHA256) {
        usage_error("refusing changed structural parent (serialize_writer.cpp)");
    }
    if (options.token_simd_parent_sha256 != PINNED_TOKEN_SIMD_PARENT_SHA256) {
        usage_error("refusing changed Token-SIMD depth-13 parameter source");
    }
    if (std::filesystem::exists(options.state_dir)) {
        throw std::runtime_error(
            "refusing to reuse an existing state directory (must be fresh "
            "per writer/reader pair): " +
            options.state_dir.string());
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
                                 options.output.string());
    }
    return options;
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

// Copied verbatim (modulo namespace) from
// real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp's
// own required_rotation_keys(), per the pinned Token-SIMD parent hash above --
// this is the exact rotation-key set the sharded reader needs, including the
// TOKEN_BATCH*baby/giant scaling and the per-lane-shift deltas used by
// build_group_shifts()/lane_shift() for Q/K/V shifted-key/value tiles.
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
    for (std::size_t delta = 1; delta < TOKEN_BATCH; ++delta) {
        indices.insert(canonical_rotation(static_cast<std::int64_t>(delta)));
        indices.insert(canonical_rotation(
            static_cast<std::int64_t>(delta) -
            static_cast<std::int64_t>(TOKEN_BATCH)));
    }
    indices.insert(canonical_rotation(
        static_cast<std::int64_t>(PACK_WIDTH * TOKEN_BATCH)));
    indices.insert(canonical_rotation(
        -static_cast<std::int64_t>(PACK_WIDTH * TOKEN_BATCH)));
    indices.insert(canonical_rotation(
        2 * static_cast<std::int64_t>(PACK_WIDTH * TOKEN_BATCH)));
    indices.erase(0);
    return {indices.begin(), indices.end()};
}

void write_exclusive(const std::filesystem::path& path, const std::string& contents) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) {
        throw std::runtime_error("could not exclusively create file " + path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t count =
            ::write(descriptor, contents.data() + offset, contents.size() - offset);
        if (count <= 0) {
            ::close(descriptor);
            throw std::runtime_error("short write to " + path.string());
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error("could not close " + path.string());
    }
}

std::string make_writer_json(const Options& options, std::uint32_t ring,
                             std::size_t rotation_keys, double keygen_seconds,
                             double serialize_seconds) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"Scheme B (hybrid) cross-process context/key "
           "serialization prototype, Token-SIMD B=8 T=103 depth-13/digits-3/"
           "ring-65536-parameter-matched fork -- WRITER phase (2-GPU "
           "process-per-GPU sharding, Stage 1: Q/K/V split prerequisite)\",\n"
        << "  \"implementation_version\": "
           "\"t103-scheme-b-token-simd-shard-writer-depth13-v1\",\n"
        << "  \"phase\": \"writer\",\n"
        << "  \"label\": \"process-A: one-time CPU-bound "
           "GenCryptoContext/KeyGen/EvalMultKeyGen/EvalRotateKeyGen using the "
           "Token-SIMD depth-13/digits-3/ring-65536 CCParams and rotation-key "
           "schedule, then serialize CryptoContext+PublicKey+PrivateKey+"
           "EvalMultKey+EvalAutomorphismKey to --state-dir; never calls "
           "LoadContext/GenCryptoContextGPU (sharded-reader-only) and never "
           "evaluates anything\",\n"
        << "  \"secret_key_handling\": \"secret-key file written to "
           "--state-dir with chmod 0600; its path and contents are never "
           "printed to stdout/stderr or recorded in this JSON; this "
           "cross-process split is a testbed simplification of the client "
           "role for measurement only, not a production client/server "
           "key-custody design (out of scope per CLAUDE.md)\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \"" << json_escape(options.backend_commit) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \"" << json_escape(options.source_sha256) << "\",\n"
        << "  \"structural_parent_sha256\": \""
        << json_escape(options.structural_parent_sha256) << "\",\n"
        << "  \"token_simd_parent_sha256\": \""
        << json_escape(options.token_simd_parent_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": " << D << ",\n"
        << "    \"pack_width\": " << PACK_WIDTH << ",\n"
        << "    \"copies\": " << COPIES << ",\n"
        << "    \"token_batch\": " << TOKEN_BATCH << ",\n"
        << "    \"batch_slots\": " << SLOTS << ",\n"
        << "    \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"large_digits\": " << LARGE_DIGITS << ",\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"rotation_key_source\": \"Token-SIMD depth-13/digits-3/"
           "ring-65536 complete-block required_rotation_keys()\"\n"
        << "  },\n"
        << "  \"timings_seconds\": {\n"
        << "    \"keygen_seconds\": " << keygen_seconds << ",\n"
        << "    \"serialize_to_disk_seconds\": " << serialize_seconds << "\n"
        << "  },\n"
        << "  \"serialized_artifacts\": [\n"
        << "    \"crypto-context.txt\", \"crypto-context.txt.dev\",\n"
        << "    \"public-key.txt\",\n"
        << "    \"eval-mult-key.bin\", \"eval-automorphism-key.bin\",\n"
        << "    \"writer_timing.txt\"\n"
        << "  ],\n"
        << "  \"serialized_artifacts_note\": \"one additional file (the secret "
           "key) was also written to --state-dir with restricted permissions; "
           "its filename is intentionally omitted from this list per the "
           "secret_key_handling hygiene rule above\"\n"
        << "}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::filesystem::create_directories(options.state_dir);

        const std::vector<int> rotation_keys = required_rotation_keys();

        const auto keygen_start = Clock::now();
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
        parameters.SetRingDim(RING_DIM);
        parameters.SetDevices({options.gpu});
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);

        Cc cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);

        Keys keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        cc->EvalRotateKeyGen(keys.secretKey, rotation_keys);
        // Deliberately not calling LoadContext here -- that GPU step
        // is exclusively the sharded reader's job (see file header).
        const double keygen_seconds = elapsed_seconds(keygen_start);
        const std::uint32_t ring = cc->GetRingDimension();
        if (ring < 2 * SLOTS) {
            throw std::runtime_error(
                "ring dimension cannot provide requested Token-SIMD slots");
        }

        const auto serialize_start = Clock::now();
        const std::filesystem::path context_path = options.state_dir / "crypto-context.txt";
        const std::filesystem::path public_key_path = options.state_dir / "public-key.txt";
        const std::filesystem::path secret_key_path = options.state_dir / "secret-key.txt";
        const std::filesystem::path mult_key_path = options.state_dir / "eval-mult-key.bin";
        const std::filesystem::path rotation_key_path =
            options.state_dir / "eval-automorphism-key.bin";

        if (!Serial::SerializeToFile(context_path.string(), cc, SerType::BINARY)) {
            throw std::runtime_error("could not serialize CryptoContext");
        }
        if (!Serial::SerializeToFile(public_key_path.string(), keys.publicKey,
                                     SerType::BINARY)) {
            throw std::runtime_error("could not serialize PublicKey");
        }
        // Secret-key hygiene: written last, permissions restricted
        // immediately, path/contents never logged. See file header comment.
        if (!Serial::SerializeToFile(secret_key_path.string(), keys.secretKey,
                                     SerType::BINARY)) {
            throw std::runtime_error("could not serialize PrivateKey");
        }
        if (::chmod(secret_key_path.c_str(), S_IRUSR | S_IWUSR) != 0) {
            throw std::runtime_error(
                "could not restrict permissions on the secret-key file");
        }

        std::ofstream mult_key_stream(mult_key_path, std::ios::out | std::ios::binary);
        if (!cc->SerializeEvalMultKey(mult_key_stream, SerType::BINARY)) {
            throw std::runtime_error("could not serialize EvalMultKey");
        }
        mult_key_stream.close();

        std::ofstream rotation_key_stream(rotation_key_path,
                                          std::ios::out | std::ios::binary);
        if (!cc->SerializeEvalAutomorphismKey(rotation_key_stream, SerType::BINARY)) {
            throw std::runtime_error("could not serialize EvalAutomorphismKey");
        }
        rotation_key_stream.close();
        const double serialize_seconds = elapsed_seconds(serialize_start);

        // Non-secret, plain key=value companion file the reader parses --
        // no keys, no paths, just timing/config numbers.
        std::ostringstream timing;
        timing << std::setprecision(17);
        timing << "keygen_seconds=" << keygen_seconds << '\n';
        timing << "serialize_seconds=" << serialize_seconds << '\n';
        timing << "rotation_keys=" << rotation_keys.size() << '\n';
        timing << "ring_dim=" << ring << '\n';
        timing << "multiplicative_depth=" << MULT_DEPTH << '\n';
        write_exclusive(options.state_dir / "writer_timing.txt", timing.str());

        const std::string evidence =
            make_writer_json(options, ring, rotation_keys.size(), keygen_seconds,
                             serialize_seconds);
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout << "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_WRITER_DONE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
