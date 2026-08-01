// Scheme B (hybrid client-assisted CKKS) cross-process context/key
// serialization prototype, TOKEN-SIMD-PARAMETER-MATCHED FORK -- SHARDED
// READER phase ("process B", one of two concurrent shard workers).
//
// Scope (docs/hybrid/tasks.md's 2026-07-31 "2-GPU process-per-GPU sharding"
// entry, Stage 1 -- Q/K/V split, zero-merge infra proof): this binary
// deserializes the context/keys written by
// real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13 and, for a
// caller-selected subset of {query, key, value} (the "--part" flag),
// independently computes the Token-SIMD B=8 T=103 LN1 + dense Q/K/V
// projection for all 13 token groups. Two instances of this binary, each
// launched with a disjoint --part list, in two containers each pinned to a
// different physical GPU via Docker's own "--gpus device=N" (see
// fhe/gpu_real_scheme_b/shard_layout.py's qkv_shard_assignment(2) --
// {query,key} on worker 0, {value} on worker 1), reading the SAME
// deserialized state-dir, is the zero-merge infra proof: Q, K, V are three
// independent dense transforms of the same LN1 output and are only ever
// consumed together downstream (attention), never combined via any
// ciphertext operation, so there is no cross-process ciphertext-merge
// question at this stage (that is deferred to the MLP-chunk stage, see
// docs/hybrid/tasks.md Stage 2).
//
// Fork discipline -- two parents, both pinned and verified computed fresh
// against this repo's working tree (not trusted from any stale prior value):
//   structural parent (deserialize-then-LoadContext shape, Client class,
//   never regenerates keys -- real_dnagpt_fides_scheme_b_serialize_reader.cpp)
//     SHA-256: 4dbb003fe8407046c852bf10f0c45b23b579cd1bbc7390099966999b6d6ab5d6
//   matmul/schedule parent (Token-SIMD B=8 physical_slot mapping,
//   baby_rotations/matmul BSGS transform, layernorm, required_rotation_keys()
//   -- real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp)
//     SHA-256: 6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e
//
// This reader contains ZERO occurrences of GenCryptoContext/EvalMultKeyGen/
// EvalRotateKeyGen/KeyGen calls -- verified by
// test_shard_writer_reader_t103_depth13_contract.py, not just asserted here.
// Its EncryptedEvaluator never calls Decrypt( or touches a secret key (same
// invariant as every other Scheme B gate); only the Client class (playing
// the restarted-client-process role, exactly as the structural parent's
// Client does) holds the deserialized secret key, for the LN1 invsqrt
// boundary only -- there is no attention/GELU/MLP stage in this reader, so
// there is no score-tile or GELU boundary here at all.
//
// Secret-key hygiene: identical discipline to both parents -- the secret key
// is deserialized here to simulate a restarted client process for
// measurement only; its path/contents are never printed. Not a production
// client/server key-custody design (out of scope per CLAUDE.md).
//
// --part flag: comma-separated and/or repeatable, drawn from
// {query,key,value}; at least one is required; unknown tokens are rejected.
// For every token group (all 13, regardless of which parts are requested),
// baby_rotations(normalized) is computed exactly once and shared across
// every *requested* part's matmul() call for that group -- the same sharing
// the frozen full-block source already does across all 3 of Q/K/V
// unconditionally; here it is simply gated by which parts were requested.
// All 3 QKV weight slices and all 3 corresponding oracle arrays
// (oracle.query/oracle.key/oracle.value) are loaded regardless of which
// parts are requested (see checkpoints/fhe_exports/.../manifest.json for the
// exact array/file names), but only the requested parts are ever matmul()'d
// or measured -- accuracy is reported per requested part only.

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
constexpr std::size_t T = 103;
constexpr std::size_t HEADS = 12;
constexpr std::size_t HEAD_DIM = 64;
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t LOGICAL_SLOTS = COPIES * PACK_WIDTH;
constexpr std::size_t TOKEN_BATCH = 8;
constexpr std::size_t SLOTS = LOGICAL_SLOTS * TOKEN_BATCH;
constexpr std::size_t TOKEN_GROUPS = (T + TOKEN_BATCH - 1) / TOKEN_BATCH;
constexpr std::size_t BSGS_N1 = 32;
constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;
constexpr std::uint32_t MULT_DEPTH = 13;
constexpr double EPS = 1e-5;
constexpr double TOL = 4e-2;

static_assert(D == HEADS * HEAD_DIM);
static_assert(SLOTS == 32768);
static_assert(TOKEN_GROUPS == 13);
static_assert(T % TOKEN_BATCH == 7);
static_assert(BSGS_N1 * BSGS_N2 == PACK_WIDTH);

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
constexpr std::string_view PINNED_STRUCTURAL_PARENT_SHA256 =
    "4dbb003fe8407046c852bf10f0c45b23b579cd1bbc7390099966999b6d6ab5d6";
constexpr std::string_view PINNED_SCHEDULE_PARENT_SHA256 =
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e";
constexpr std::string_view PINNED_FIXTURE_MANIFEST =
    "d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6";

const std::array<std::string, 3> PART_NAMES = {"query", "key", "value"};

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

int part_index(const std::string& name) {
    for (std::size_t i = 0; i < PART_NAMES.size(); ++i) {
        if (PART_NAMES[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<std::string> split_comma(const std::string& value) {
    std::vector<std::string> pieces;
    std::string current;
    for (const char ch : value) {
        if (ch == ',') {
            pieces.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    pieces.push_back(current);
    return pieces;
}

// Non-secret key=value companion the writer left in --state-dir. No keys,
// no paths -- just the writer's own timing/config numbers.
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
    for (const char* required :
         {"keygen_seconds", "serialize_seconds", "rotation_keys", "ring_dim",
          "multiplicative_depth"}) {
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
    std::vector<std::string> parts;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string structural_parent_sha256;
    std::string schedule_parent_sha256;
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
    std::string source_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13 "
        "--gpu N --state-dir PATH --fixture-dir PATH --output PATH "
        "--part query,key,value (repeatable/comma-separated) "
        "--structural-parent-sha256 SHA --schedule-parent-sha256 SHA "
        "--fixture-manifest-sha256 SHA [--backend-commit SHA] "
        "[--container-image NAME] [--environment TEXT] [--source-sha256 SHA] "
        "[--fixture-contract-sha256 SHA]");
}

Options parse_options(int argc, char** argv) {
    Options options;
    std::vector<std::string> raw_parts;
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
        } else if (arg == "--part") {
            for (const std::string& piece : split_comma(next())) {
                raw_parts.push_back(piece);
            }
        } else if (arg == "--backend-commit") {
            options.backend_commit = next();
        } else if (arg == "--structural-parent-sha256") {
            options.structural_parent_sha256 = next();
        } else if (arg == "--schedule-parent-sha256") {
            options.schedule_parent_sha256 = next();
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
            std::cout << "usage: real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13 "
                         "--gpu N --state-dir PATH --fixture-dir PATH "
                         "--output PATH --part query,key,value ...\n";
            std::exit(0);
        } else {
            usage_error("unknown option: " + arg);
        }
    }

    if (raw_parts.empty()) {
        usage_error("at least one --part is required (query, key, and/or value)");
    }
    std::vector<int> seen(PART_NAMES.size(), 0);
    for (const std::string& part : raw_parts) {
        const int index = part_index(part);
        if (index < 0) {
            usage_error("unknown --part value: " + part +
                        " (must be one of query, key, value)");
        }
        if (seen[static_cast<std::size_t>(index)] == 0) {
            seen[static_cast<std::size_t>(index)] = 1;
            options.parts.push_back(part);
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
    if (options.structural_parent_sha256 != PINNED_STRUCTURAL_PARENT_SHA256) {
        usage_error("refusing changed structural parent (serialize_reader.cpp)");
    }
    if (options.schedule_parent_sha256 != PINNED_SCHEDULE_PARENT_SHA256) {
        usage_error("refusing changed Token-SIMD depth-13 schedule parent");
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

// ---- Fixture loading -- byte-identical conventions to
// real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp and
// real_dnagpt_fides_scheme_b_simd_linear_t103.cpp (schedule parent). ----

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
        throw std::runtime_error("short or trailing fixture data: " + path.string());
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
    // All 3 QKV weight slices, loaded regardless of which parts are
    // requested (see file header comment).
    std::array<std::vector<double>, 3> qkv;
    // All 3 corresponding oracle arrays (head-major [HEADS, T, HEAD_DIM]),
    // loaded regardless of which parts are requested.
    std::array<std::vector<double>, 3> oracle;
};

Fixture load_fixture(const std::filesystem::path& directory) {
    Fixture fixture;
    fixture.input = read_f64(directory / "input__embeddings.bin", T * D);
    fixture.ln1_weight = read_f64(directory / "weights__ln1.bin", D);
    const std::vector<double> qkv =
        read_f64(directory / "weights__attn_qkv.bin", 3 * D * D);
    for (std::size_t part = 0; part < 3; ++part) {
        fixture.qkv[part].assign(
            qkv.begin() + static_cast<std::ptrdiff_t>(part * D * D),
            qkv.begin() + static_cast<std::ptrdiff_t>((part + 1) * D * D));
    }
    fixture.oracle[0] =
        read_f64(directory / "oracle__query.bin", HEADS * T * HEAD_DIM);
    fixture.oracle[1] =
        read_f64(directory / "oracle__key.bin", HEADS * T * HEAD_DIM);
    fixture.oracle[2] =
        read_f64(directory / "oracle__value.bin", HEADS * T * HEAD_DIM);
    return fixture;
}

std::size_t physical_slot(std::size_t copy, std::size_t feature,
                          std::size_t token_lane) {
    if (copy >= COPIES || feature >= PACK_WIDTH || token_lane >= TOKEN_BATCH) {
        throw std::out_of_range("Token-SIMD slot coordinate out of range");
    }
    return (copy * PACK_WIDTH + feature) * TOKEN_BATCH + token_lane;
}

std::size_t active_tokens(std::size_t group) {
    const std::size_t first = group * TOKEN_BATCH;
    return std::min(TOKEN_BATCH, T - first);
}

// ---- Client: the only party in this process that holds the secret key --
// here, deserialized from --state-dir/secret-key.txt, simulating a
// restarted client process. This stage only needs the LN1 invsqrt boundary
// (no attention/GELU/MLP in this reader), so the Client class is smaller
// than the full-block source's. ----
class Client {
  public:
    Client(Cc context, Keys& keys) : cc_(std::move(context)), keys_(keys) {}

    Ct invsqrt_boundary(const Ct& ciphertext, std::size_t logical_instances) {
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
        Plaintext refreshed = cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
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
    // Keyed by part index (0=query, 1=key, 2=value); only requested parts
    // are populated.
    std::array<std::vector<Ct>, 3> packed;
    std::size_t matrix_products = 0;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture, Client& client)
        : cc_(std::move(context)), fixture_(fixture), client_(client) {}

    // requested: which of {0=query,1=key,2=value} to compute. For every
    // group (all TOKEN_GROOUPS, unconditionally), baby_rotations(normalized)
    // is computed exactly once and shared across every requested part's
    // matmul() -- the same sharing the frozen full-block source does across
    // all 3 of Q/K/V unconditionally; here it is gated by `requested`.
    EvaluationResult evaluate(const std::vector<Ct>& encrypted_inputs,
                              const std::vector<std::size_t>& active,
                              const std::vector<int>& requested) {
        if (encrypted_inputs.size() != TOKEN_GROUPS || active.size() != TOKEN_GROUPS) {
            throw std::invalid_argument("Token-SIMD input/active-count mismatch");
        }
        EvaluationResult result;
        for (const int part : requested) {
            result.packed[static_cast<std::size_t>(part)].resize(TOKEN_GROUPS);
        }
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            if (active[group] != ::active_tokens(group)) {
                throw std::invalid_argument("Token-SIMD active-count schedule mismatch");
            }
            Ct normalized = layernorm(encrypted_inputs[group], active[group]);
            const BabyRotations baby = baby_rotations(normalized);
            for (const int part : requested) {
                Ct projected = matmul(baby, fixture_.qkv[static_cast<std::size_t>(part)]);
                result.packed[static_cast<std::size_t>(part)][group] = std::move(projected);
            }
            std::cout << "[stage] token_group=" << group << " active=" << active[group]
                      << " parts_computed=" << requested.size() << '\n';
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
        Plaintext mean_scale =
            repeated_plain(std::vector<double>(D, 1.0 / static_cast<double>(D)));
        return cc_->EvalMult(sum, mean_scale);
    }

    Ct layernorm(const Ct& input, std::size_t active) {
        Ct mean = mean_broadcast(input);
        Ct centered = cc_->EvalSub(input, mean);
        Ct variance = mean_broadcast(cc_->EvalMult(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        Ct inverse = client_.invsqrt_boundary(variance, active);
        Ct normalized = cc_->EvalMult(centered, inverse);
        Plaintext ln1_weight = repeated_plain(fixture_.ln1_weight);
        return cc_->EvalMult(normalized, ln1_weight);
    }

    BabyRotations baby_rotations(const Ct& input) {
        std::vector<std::int32_t> indices;
        indices.reserve(BSGS_N1 - 1);
        for (std::size_t small = 1; small < BSGS_N1; ++small) {
            indices.push_back(static_cast<std::int32_t>(TOKEN_BATCH * small));
        }
        const auto rotated =
            cc_->EvalFastRotation(input, indices, cc_->GetCyclotomicOrder(), nullptr);
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

    Ct matmul(const BabyRotations& baby, const std::vector<double>& weight) {
        if (weight.size() != D * D) {
            throw std::invalid_argument("QKV weight must be D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::size_t diagonal = BSGS_N1 * giant + small;
                std::vector<double> packed_values(SLOTS, 0.0);
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH;
                        const std::size_t column = (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            const double coefficient = weight[rolled_row * D + column];
                            for (std::size_t token = 0; token < TOKEN_BATCH; ++token) {
                                packed_values[physical_slot(copy, row, token)] = coefficient;
                            }
                        }
                    }
                }
                Plaintext diagonal_plain = raw_plain(packed_values);
                Ct term = cc_->EvalMult(baby.values[small], diagonal_plain);
                inner = small == 0 ? term : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(
                    inner, static_cast<std::int32_t>(TOKEN_BATCH * BSGS_N1 * giant));
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

Metrics measure_part(const std::vector<double>& got,
                     const std::vector<double>& oracle_head_major) {
    if (got.size() != T * D || oracle_head_major.size() != HEADS * T * HEAD_DIM) {
        throw std::invalid_argument("part metric size mismatch");
    }
    Metrics metrics;
    double global_denominator = 0.0;
    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t head = 0; head < HEADS; ++head) {
            for (std::size_t dim = 0; dim < HEAD_DIM; ++dim) {
                const std::size_t got_index = token * D + head * HEAD_DIM + dim;
                const std::size_t expected_index = (head * T + token) * HEAD_DIM + dim;
                const double actual = got[got_index];
                const double expected = oracle_head_major[expected_index];
                metrics.all_finite = metrics.all_finite && std::isfinite(actual);
                const double error = std::abs(actual - expected);
                metrics.max_abs_error = std::max(metrics.max_abs_error, error);
                token_error = std::max(token_error, error);
                token_denominator = std::max(token_denominator, std::abs(expected));
                global_denominator = std::max(global_denominator, std::abs(expected));
            }
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
        throw std::runtime_error("could not exclusively create evidence file " +
                                 path.string());
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
                      double evaluation_seconds, const Client& client,
                      const EvaluationResult& evaluation,
                      const std::map<std::string, Metrics>& metrics_by_part) {
    const double writer_keygen_seconds = std::stod(writer_timing.at("keygen_seconds"));
    const double writer_serialize_seconds = std::stod(writer_timing.at("serialize_seconds"));
    const std::size_t rotation_keys =
        static_cast<std::size_t>(std::stoul(writer_timing.at("rotation_keys")));

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"Scheme B Token-SIMD T=103 depth-13/digits-3/"
           "ring-65536 2-GPU process-per-GPU sharding, Stage 1 (Q/K/V split) "
           "-- SHARDED READER phase\",\n"
        << "  \"implementation_version\": "
           "\"t103-scheme-b-token-simd-shard-reader-depth13-v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS; server-only "
           "linear algebra, client-only decrypt at pre-declared "
           "nonlinearity boundaries)\",\n"
        << "  \"label\": \"one of two concurrent shard workers computing a "
           "disjoint subset of {query,key,value} from a context/key lineage "
           "deserialized from a separate writer process's --state-dir; zero "
           "ciphertext merge with the other worker at this stage -- Q/K/V "
           "are only ever consumed together downstream (attention), never "
           "combined via any encrypted operation\",\n"
        << "  \"parts_requested\": [";
    for (std::size_t i = 0; i < options.parts.size(); ++i) {
        out << "\"" << json_escape(options.parts[i]) << "\""
            << (i + 1 == options.parts.size() ? "" : ", ");
    }
    out << "],\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \"" << json_escape(options.backend_commit) << "\",\n"
        << "  \"container_image\": \"" << json_escape(options.container_image) << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \"" << json_escape(options.source_sha256) << "\",\n"
        << "  \"structural_parent_sha256\": \""
        << json_escape(options.structural_parent_sha256) << "\",\n"
        << "  \"schedule_parent_sha256\": \""
        << json_escape(options.schedule_parent_sha256) << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"fixture_contract_sha256\": \""
        << json_escape(options.fixture_contract_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"D\": " << D << ", \"T\": " << T
        << ", \"token_batch\": " << TOKEN_BATCH << ", \"token_groups\": " << TOKEN_GROUPS
        << ",\n"
        << "    \"pack_width\": " << PACK_WIDTH << ", \"copies\": " << COPIES
        << ", \"batch_slots\": " << SLOTS << ", \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"matrix_products\": " << evaluation.matrix_products
        << ", \"expected_matrix_products\": "
        << TOKEN_GROUPS * options.parts.size() << "\n"
        << "  },\n"
        << "  \"cross_process_breakdown\": {\n"
        << "    \"writer_keygen_seconds\": " << writer_keygen_seconds << ",\n"
        << "    \"writer_serialize_to_disk_seconds\": " << writer_serialize_seconds << ",\n"
        << "    \"reader_deserialize_seconds\": " << deserialize_seconds << ",\n"
        << "    \"reader_load_context_gpu_seconds\": " << load_context_gpu_seconds << "\n"
        << "  },\n"
        << "  \"load_verified_fixture_seconds\": " << fixture_seconds << ",\n"
        << "  \"timings_seconds\": {\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds << ",\n"
        << "    \"encrypted_evaluation\": " << evaluation_seconds << ",\n"
        << "    \"client_boundary_seconds_total\": " << client.seconds() << ",\n"
        << "    \"server_linear_algebra_seconds\": "
        << (evaluation_seconds - client.seconds()) << "\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"round_trips\": " << client.round_trips() << ",\n"
        << "    \"logical_boundary_instances\": " << client.logical_instances() << ",\n"
        << "    \"note\": \"LN1 client round trips are shared across every "
           "requested part -- one boundary crossing per token group covers "
           "all parts this worker computes, since LN1 happens once per "
           "group before any Q/K/V matmul\"\n"
        << "  },\n"
        << "  \"metrics_by_part\": {\n";
    for (std::size_t i = 0; i < options.parts.size(); ++i) {
        const std::string& part = options.parts[i];
        const Metrics& metrics = metrics_by_part.at(part);
        out << "    \"" << json_escape(part) << "\": {\n"
            << "      \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
            << "      \"worst_token_rel_inf\": " << metrics.worst_token_rel_inf << ",\n"
            << "      \"max_abs_error\": " << metrics.max_abs_error << ",\n"
            << "      \"all_finite\": " << (metrics.all_finite ? "true" : "false") << ",\n"
            << "      \"passed\": " << (metrics.passed ? "true" : "false") << "\n"
            << "    }" << (i + 1 == options.parts.size() ? "\n" : ",\n");
    }
    out << "  },\n";
    bool all_passed = true;
    for (const auto& [name, metrics] : metrics_by_part) {
        (void)name;
        all_passed = all_passed && metrics.passed;
    }
    out << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (all_passed ? "true" : "false") << ",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"evaluator_has_private_key\": false\n"
        << "}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        std::vector<int> requested;
        requested.reserve(options.parts.size());
        for (const std::string& part : options.parts) {
            requested.push_back(part_index(part));
        }

        const auto fixture_start = Clock::now();
        const Fixture fixture = load_fixture(options.fixture_dir);
        const double fixture_seconds = elapsed_seconds(fixture_start);

        const std::unordered_map<std::string, std::string> writer_timing =
            read_writer_timing(options.state_dir / "writer_timing.txt");
        if (std::stoul(writer_timing.at("multiplicative_depth")) != MULT_DEPTH) {
            throw std::runtime_error(
                "state-dir was not produced by a depth-13 Token-SIMD writer "
                "(multiplicative_depth mismatch)");
        }
        if (std::stoul(writer_timing.at("ring_dim")) != 2 * SLOTS) {
            throw std::runtime_error(
                "state-dir was not produced by a Token-SIMD-parameter-matched "
                "writer (ring_dim mismatch)");
        }

        // Cross-process reload: deserialize everything the shard writer
        // process serialized. NOTE: no GenCryptoContext, no KeyGen, no
        // EvalMultKeyGen, no EvalRotateKeyGen anywhere in this file --
        // verified textually by
        // test_shard_writer_reader_t103_depth13_contract.py.
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
        // here regardless of what was deserialized above. Physical-GPU
        // selection is entirely a Docker "--gpus device=N" concern (see
        // file header comment and docs/hybrid/tasks.md's 2026-07-31 finding
        // 1) -- this process's own --gpu value addresses device index 0
        // inside its own container regardless of which physical GPU Docker
        // mapped in.
        const auto load_context_start = Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        const double load_context_gpu_seconds = elapsed_seconds(load_context_start);
        const std::uint32_t ring = cc->GetRingDimension();
        if (ring < 2 * SLOTS) {
            throw std::runtime_error(
                "deserialized context cannot provide requested Token-SIMD slots");
        }

        std::cout << "[context] security=HEStd_128_classic ring=" << ring
                  << " depth=" << MULT_DEPTH << " slots=" << SLOTS << " gpu=" << options.gpu
                  << " parts=";
        for (std::size_t i = 0; i < options.parts.size(); ++i) {
            std::cout << options.parts[i] << (i + 1 == options.parts.size() ? "" : "+");
        }
        std::cout << " deserialize_seconds=" << deserialize_seconds
                  << " load_context_gpu_seconds=" << load_context_gpu_seconds << '\n';

        const auto encryption_start = Clock::now();
        std::vector<Ct> encrypted_inputs;
        std::vector<std::size_t> active_counts;
        encrypted_inputs.reserve(TOKEN_GROUPS);
        active_counts.reserve(TOKEN_GROUPS);
        for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
            const std::size_t first = group * TOKEN_BATCH;
            const std::size_t active = active_tokens(group);
            std::vector<double> packed(SLOTS, 0.0);
            for (std::size_t token_lane = 0; token_lane < active; ++token_lane) {
                const std::size_t token = first + token_lane;
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    for (std::size_t dim = 0; dim < D; ++dim) {
                        packed[physical_slot(copy, dim, token_lane)] =
                            fixture.input[token * D + dim];
                    }
                }
            }
            Plaintext plaintext = cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
            encrypted_inputs.push_back(cc->Encrypt(keys.publicKey, plaintext));
            active_counts.push_back(active);
        }
        cc->Synchronize();
        const double encryption_seconds = elapsed_seconds(encryption_start);

        Client client(cc, keys);
        EncryptedEvaluator evaluator(cc, fixture, client);
        const auto evaluation_start = Clock::now();
        EvaluationResult evaluation =
            evaluator.evaluate(encrypted_inputs, active_counts, requested);
        cc->Synchronize();
        const double evaluation_seconds = elapsed_seconds(evaluation_start);

        std::map<std::string, Metrics> metrics_by_part;
        for (const int part : requested) {
            const std::string& name = PART_NAMES[static_cast<std::size_t>(part)];
            std::vector<double> output(T * D, 0.0);
            for (std::size_t group = 0; group < TOKEN_GROUPS; ++group) {
                Plaintext decoded;
                cc->Decrypt(keys.secretKey,
                           evaluation.packed[static_cast<std::size_t>(part)][group],
                           &decoded);
                decoded->SetLength(SLOTS);
                const std::vector<double> raw = decoded->GetRealPackedValue();
                const std::size_t first = group * TOKEN_BATCH;
                const std::size_t active = active_tokens(group);
                for (std::size_t token_lane = 0; token_lane < active; ++token_lane) {
                    for (std::size_t dim = 0; dim < D; ++dim) {
                        output[(first + token_lane) * D + dim] =
                            raw[physical_slot(0, dim, token_lane)];
                    }
                }
            }
            metrics_by_part[name] =
                measure_part(output, fixture.oracle[static_cast<std::size_t>(part)]);
        }

        if (evaluation.matrix_products != TOKEN_GROUPS * requested.size()) {
            throw std::runtime_error(
                "Token-SIMD shard-reader declared work count mismatch -- computed a "
                "part that was not requested, or skipped a requested part");
        }

        const std::string evidence = make_json(
            options, ring, writer_timing, fixture_seconds, deserialize_seconds,
            load_context_gpu_seconds, encryption_seconds, evaluation_seconds, client,
            evaluation, metrics_by_part);
        write_exclusive(options.output, evidence);
        std::cout << evidence;

        bool all_passed = true;
        for (const auto& [name, metrics] : metrics_by_part) {
            std::cout << "part=" << name << " global_rel_inf=" << metrics.global_rel_inf
                      << " passed=" << (metrics.passed ? "true" : "false") << '\n';
            all_passed = all_passed && metrics.passed;
        }
        std::cout << (all_passed ? "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_READER_PASS"
                                 : "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_READER_FAIL")
                  << '\n';
        return all_passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
