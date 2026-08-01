// Scheme B (hybrid client-assisted CKKS) 2-GPU process-per-GPU sharding,
// STAGE 2 -- MLP down-projection CHUNK split, the exact ciphertext EvalAdd
// MERGE step. This is the third process in the Stage-2 pipeline: two shard
// readers each computed a disjoint 2-chunk PARTIAL down-projection sum and
// serialized 13 per-group partial ciphertexts; this step deserializes both
// workers' partials and the primary's residual1, EvalAdds the two partials
// back into the full down-projection sum, adds residual1, and measures the
// reconstructed block output against the oracle.
//
// The whole Stage-2 novelty lives in one line:
//     mlp[group] = EvalAdd(partial_lo[group], partial_hi[group]);
// CKKS EvalAdd is an exact, depth-free homomorphic addition, and the frozen
// full-block down-projection accumulated the four chunk matmuls with exactly
// this operator (mlp = c0+c1+c2+c3). Splitting that sum as (c0+c1)+(c2+c3)
// and re-adding across processes reproduces the un-sharded sum bit-for-bit up
// to floating-point reassociation (~1e-13). Proved a math no-op to <=1e-9 by
// test_shard_mlp_reader_merge_t103_depth13_contract.py BEFORE this source is
// allowed to run.
//
// Fork discipline -- additive; parents pinned and computed fresh, not edited:
//   Stage-1 sharded-reader parent (deserialize-context/keys-then-LoadContext
//   shape, --state-dir companion-timing gate, secret-key hygiene, oracle
//   measurement helper shape)
//   real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp
//     SHA-256: 3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89
//   Stage-2 MLP shard-reader sibling (produces the partial-sum ciphertexts
//   this step merges; shares the physical_slot/active_tokens layout and the
//   --chunks tag convention)
//   real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13.cpp
//     (referenced by name; its hash is pinned by the contract test, not here,
//     because the two siblings are developed together)
//   depth-13 crypto-parameter/schedule source (rotation-key set the Stage-1
//   writer serialized; unchanged)
//   real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp
//     SHA-256: 6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e
//
// This step contains ZERO GenCryptoContext/KeyGen/EvalMultKeyGen/
// EvalRotateKeyGen calls -- it deserializes the context/keys the Stage-1
// writer serialized. It never evaluates any dense transform, no matmul, no
// attention, no LayerNorm/GELU boundary: its ONLY encrypted operations are
// the 13 EvalAdd merges and 13 EvalAdd residual additions. It holds the
// deserialized secret key solely to decrypt the final reconstructed block
// output for measurement (client role), exactly as the Stage-1 reader does
// for its final decrypt. Secret-key hygiene: path/contents never printed.
//
// Ciphertext serialization: deserializing the partial-sum ciphertexts the
// readers wrote requires FIDESlib round-tripping a GPU-resident Ciphertext
// through its Serialize path -- an UNVERIFIED capability at the time this
// source was written (no prior Scheme B source serializes/deserializes a
// ciphertext). See the report/[A] note. The EvalAdd merge math is exact and
// fixture-proved locally regardless of transport.

#include <fideslib.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using namespace fideslib;

namespace {

constexpr std::size_t D = 768;
constexpr std::size_t T = 103;
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t LOGICAL_SLOTS = COPIES * PACK_WIDTH;
constexpr std::size_t TOKEN_BATCH = 8;
constexpr std::size_t SLOTS = LOGICAL_SLOTS * TOKEN_BATCH;
constexpr std::size_t TOKEN_GROUPS = (T + TOKEN_BATCH - 1) / TOKEN_BATCH;
constexpr std::uint32_t MULT_DEPTH = 13;
constexpr double TOL = 4e-2;

static_assert(SLOTS == 32768);
static_assert(TOKEN_GROUPS == 13);
static_assert(T % TOKEN_BATCH == 7);

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
constexpr std::string_view PINNED_STAGE1_READER_SHA256 =
    "3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89";
constexpr std::string_view PINNED_SCHEDULE_PARENT_SHA256 =
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e";
constexpr std::string_view PINNED_FIXTURE_MANIFEST =
    "d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6";

// The two disjoint chunk-subset tags the two Stage-2 readers used to name
// their serialized partials (must match the readers' options.chunk_tag()).
const std::array<std::string, 2> PARTIAL_TAGS = {"01", "23"};

using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;
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
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
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
    std::filesystem::path state_dir;
    std::filesystem::path fixture_dir;
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string stage1_reader_sha256;
    std::string schedule_parent_sha256;
    std::string fixture_manifest_sha256;
    std::string source_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13 "
        "--gpu N --state-dir PATH --fixture-dir PATH --output PATH "
        "--stage1-reader-sha256 SHA --schedule-parent-sha256 SHA "
        "--fixture-manifest-sha256 SHA [--backend-commit SHA] "
        "[--container-image NAME] [--environment TEXT] [--source-sha256 SHA]");
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
        } else if (arg == "--stage1-reader-sha256") {
            options.stage1_reader_sha256 = next();
        } else if (arg == "--schedule-parent-sha256") {
            options.schedule_parent_sha256 = next();
        } else if (arg == "--fixture-manifest-sha256") {
            options.fixture_manifest_sha256 = next();
        } else if (arg == "--container-image") {
            options.container_image = next();
        } else if (arg == "--environment") {
            options.environment = next();
        } else if (arg == "--source-sha256") {
            options.source_sha256 = next();
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Stage-2 MLP down-projection EvalAdd merge step\n";
            std::exit(0);
        } else {
            usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0 || options.state_dir.empty() ||
        options.fixture_dir.empty() || options.output.empty()) {
        usage_error(
            "gpu, state-dir, fixture-dir, and output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit");
    }
    if (options.stage1_reader_sha256 != PINNED_STAGE1_READER_SHA256) {
        usage_error("refusing changed Stage-1 sharded-reader parent");
    }
    if (options.schedule_parent_sha256 != PINNED_SCHEDULE_PARENT_SHA256) {
        usage_error("refusing changed depth-13 schedule parent");
    }
    if (options.fixture_manifest_sha256 != PINNED_FIXTURE_MANIFEST) {
        usage_error("refusing unpinned fixture manifest");
    }
    if (!std::filesystem::is_directory(options.state_dir)) {
        throw std::runtime_error("state directory missing: " +
                                 options.state_dir.string());
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
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

std::size_t physical_slot(std::size_t copy, std::size_t feature,
                          std::size_t token_lane) {
    if (copy >= COPIES || feature >= PACK_WIDTH ||
        token_lane >= TOKEN_BATCH) {
        throw std::out_of_range("Token-SIMD slot coordinate out of range");
    }
    return (copy * PACK_WIDTH + feature) * TOKEN_BATCH + token_lane;
}

std::size_t active_tokens(std::size_t group) {
    const std::size_t first = group * TOKEN_BATCH;
    return std::min(TOKEN_BATCH, T - first);
}

std::map<std::string, std::string> read_writer_timing(
    const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(
            "could not open writer timing file: " + path.string());
    }
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(stream, line)) {
        const std::size_t split = line.find('=');
        if (split == std::string::npos) {
            continue;
        }
        values[line.substr(0, split)] = line.substr(split + 1);
    }
    for (const char* required : {"multiplicative_depth", "ring_dim"}) {
        if (values.find(required) == values.end()) {
            throw std::runtime_error(
                std::string("writer timing file missing field: ") + required);
        }
    }
    return values;
}

Ct deserialize_ciphertext(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("missing partial/residual ciphertext: " +
                                 path.string());
    }
    Ct ct;
    if (!Serial::DeserializeFromFile(path.string(), ct, SerType::BINARY)) {
        throw std::runtime_error("could not deserialize ciphertext: " +
                                 path.string());
    }
    return ct;
}

struct Metrics {
    double global_rel_inf = 0.0;
    double worst_token_rel_inf = 0.0;
    double max_abs_error = 0.0;
    bool all_finite = true;
    bool passed = false;
};

Metrics measure_output(const std::vector<double>& got,
                       const std::vector<double>& oracle) {
    if (got.size() != T * D || oracle.size() != T * D) {
        throw std::invalid_argument("block-output metric size mismatch");
    }
    Metrics metrics;
    double global_denominator = 0.0;
    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t dim = 0; dim < D; ++dim) {
            const std::size_t index = token * D + dim;
            const double actual = got[index];
            const double expected = oracle[index];
            metrics.all_finite =
                metrics.all_finite && std::isfinite(actual);
            const double error = std::abs(actual - expected);
            metrics.max_abs_error = std::max(metrics.max_abs_error, error);
            token_error = std::max(token_error, error);
            token_denominator =
                std::max(token_denominator, std::abs(expected));
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
        const ssize_t count = ::write(descriptor, contents.data() + offset,
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

std::string make_json(const Options& options, std::uint32_t ring,
                      const Metrics& metrics, std::size_t merge_evaladds,
                      std::size_t residual_evaladds, double fixture_seconds,
                      double deserialize_context_seconds,
                      double load_context_gpu_seconds,
                      double deserialize_partials_seconds,
                      double merge_seconds, double decrypt_seconds) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"Scheme B Token-SIMD T=103 depth-13/digits-3/"
           "ring-65536 2-GPU process-per-GPU sharding, Stage 2 (MLP "
           "down-projection chunk split) -- EvalAdd MERGE step; EvalAdds the "
           "two shard workers' partial down-projection sums, adds residual1, "
           "and measures the reconstructed block output against the oracle\",\n"
        << "  \"implementation_version\": "
           "\"t103-scheme-b-token-simd-shard-mlp-merge-depth13-v1\",\n"
        << "  \"phase\": \"merge\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS)\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
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
        << "  \"stage1_reader_parent_sha256\": \""
        << json_escape(options.stage1_reader_sha256) << "\",\n"
        << "  \"schedule_parent_sha256\": \""
        << json_escape(options.schedule_parent_sha256) << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": " << D << ", \"T\": " << T
        << ", \"token_batch\": " << TOKEN_BATCH
        << ", \"token_groups\": " << TOKEN_GROUPS << ",\n"
        << "    \"pack_width\": " << PACK_WIDTH << ", \"copies\": " << COPIES
        << ", \"batch_slots\": " << SLOTS << ", \"ring_dim\": " << ring
        << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"partial_tags_merged\": [\"" << PARTIAL_TAGS[0] << "\", \""
        << PARTIAL_TAGS[1] << "\"],\n"
        << "    \"merge_evaladds\": " << merge_evaladds
        << ", \"expected_merge_evaladds\": " << TOKEN_GROUPS << ",\n"
        << "    \"residual_evaladds\": " << residual_evaladds
        << ", \"expected_residual_evaladds\": " << TOKEN_GROUPS << ",\n"
        << "    \"dense_matrix_products\": 0, \"attention_ops\": 0, "
           "\"boundary_crossings\": 0\n"
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
        << "    \"deserialize_context_keys\": " << deserialize_context_seconds
        << ",\n"
        << "    \"load_context_gpu\": " << load_context_gpu_seconds << ",\n"
        << "    \"deserialize_partials\": " << deserialize_partials_seconds
        << ",\n"
        << "    \"evaladd_merge\": " << merge_seconds << ",\n"
        << "    \"final_decrypt\": " << decrypt_seconds << "\n"
        << "  },\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"merger_regenerates_keys\": false,\n"
        << "  \"merger_evaluates_dense_transforms\": false\n"
        << "}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        const auto fixture_start = Clock::now();
        const std::vector<double> oracle_block_output =
            read_f64(options.fixture_dir / "oracle__block_output.bin", T * D);
        const double fixture_seconds = elapsed_seconds(fixture_start);

        const std::map<std::string, std::string> writer_timing =
            read_writer_timing(options.state_dir / "writer_timing.txt");
        if (std::stoul(writer_timing.at("multiplicative_depth")) !=
            MULT_DEPTH) {
            throw std::runtime_error(
                "state-dir was not produced by a depth-13 Token-SIMD writer");
        }
        if (std::stoul(writer_timing.at("ring_dim")) != 2 * SLOTS) {
            throw std::runtime_error(
                "state-dir ring_dim mismatch for Token-SIMD parameters");
        }

        // Deserialize context + keys (NO key regeneration anywhere here).
        const auto deserialize_context_start = Clock::now();
        Cc cc;
        if (!Serial::DeserializeFromFile(
                (options.state_dir / "crypto-context.txt").string(), cc,
                SerType::BINARY)) {
            throw std::runtime_error("could not deserialize CryptoContext");
        }
        Keys keys;
        if (!Serial::DeserializeFromFile(
                (options.state_dir / "public-key.txt").string(),
                keys.publicKey, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize PublicKey");
        }
        // Secret key: needed only to decrypt the final reconstructed block
        // output for measurement (client role). Path/contents never printed.
        if (!Serial::DeserializeFromFile(
                (options.state_dir / "secret-key.txt").string(),
                keys.secretKey, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize PrivateKey");
        }
        std::ifstream mult_key_stream(
            options.state_dir / "eval-mult-key.bin",
            std::ios::in | std::ios::binary);
        if (!mult_key_stream.is_open() ||
            !cc->DeserializeEvalMultKey(mult_key_stream, SerType::BINARY)) {
            throw std::runtime_error("could not deserialize EvalMultKey");
        }
        mult_key_stream.close();
        std::ifstream rotation_key_stream(
            options.state_dir / "eval-automorphism-key.bin",
            std::ios::in | std::ios::binary);
        if (!rotation_key_stream.is_open() ||
            !cc->DeserializeEvalAutomorphismKey(rotation_key_stream,
                                                SerType::BINARY)) {
            throw std::runtime_error(
                "could not deserialize EvalAutomorphismKey");
        }
        rotation_key_stream.close();
        const double deserialize_context_seconds =
            elapsed_seconds(deserialize_context_start);

        const auto load_context_start = Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        const double load_context_gpu_seconds =
            elapsed_seconds(load_context_start);
        const std::uint32_t ring = cc->GetRingDimension();
        if (ring < 2 * SLOTS) {
            throw std::runtime_error(
                "deserialized context cannot provide requested slots");
        }

        // Deserialize both workers' 13 partial-sum ciphertexts and the
        // primary's 13 residual1 ciphertexts.
        const auto deserialize_partials_start = Clock::now();
        std::array<std::vector<Ct>, 2> partials;
        for (std::size_t worker = 0; worker < PARTIAL_TAGS.size(); ++worker) {
            partials[worker].reserve(TOKEN_GROUPS);
            for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
                partials[worker].push_back(deserialize_ciphertext(
                    options.state_dir /
                    ("partial-mlp-chunks" + PARTIAL_TAGS[worker] + "-group" +
                     std::to_string(group) + ".ct")));
            }
        }
        std::vector<Ct> residual1;
        residual1.reserve(TOKEN_GROUPS);
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            residual1.push_back(deserialize_ciphertext(
                options.state_dir /
                ("residual1-group" + std::to_string(group) + ".ct")));
        }
        const double deserialize_partials_seconds =
            elapsed_seconds(deserialize_partials_start);

        // THE Stage-2 merge: EvalAdd the two workers' partial down-projection
        // sums into the full sum, then add residual1 -> block output.
        const auto merge_start = Clock::now();
        std::vector<Ct> block_output(TOKEN_GROUPS);
        std::size_t merge_evaladds = 0;
        std::size_t residual_evaladds = 0;
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            Ct mlp = cc->EvalAdd(partials[0][group], partials[1][group]);
            ++merge_evaladds;
            block_output[group] = cc->EvalAdd(residual1[group], mlp);
            ++residual_evaladds;
        }
        cc->Synchronize();
        const double merge_seconds = elapsed_seconds(merge_start);

        const auto decrypt_start = Clock::now();
        std::vector<double> output(T * D, 0.0);
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            Plaintext decoded;
            cc->Decrypt(keys.secretKey, block_output[group], &decoded);
            decoded->SetLength(SLOTS);
            const std::vector<double> raw = decoded->GetRealPackedValue();
            const std::size_t first = group * TOKEN_BATCH;
            const std::size_t active = active_tokens(group);
            for (std::size_t token_lane = 0; token_lane < active;
                 ++token_lane) {
                for (std::size_t dim = 0; dim < D; ++dim) {
                    output[(first + token_lane) * D + dim] =
                        raw[physical_slot(0, dim, token_lane)];
                }
            }
        }
        const double decrypt_seconds = elapsed_seconds(decrypt_start);

        if (merge_evaladds != TOKEN_GROUPS ||
            residual_evaladds != TOKEN_GROUPS) {
            throw std::runtime_error(
                "Stage-2 merge EvalAdd count mismatch");
        }
        const Metrics metrics = measure_output(output, oracle_block_output);
        const std::string evidence = make_json(
            options, ring, metrics, merge_evaladds, residual_evaladds,
            fixture_seconds, deserialize_context_seconds,
            load_context_gpu_seconds, deserialize_partials_seconds,
            merge_seconds, decrypt_seconds);
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout << (metrics.passed
                          ? "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_MLP_MERGE_PASS\n"
                          : "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_MLP_MERGE_FAIL\n");
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
