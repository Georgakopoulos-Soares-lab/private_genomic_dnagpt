// Scheme B (hybrid client-assisted CKKS) -- COMBINED-lever correctness
// micro-gate: 2-GPU process-per-GPU Q/K/V sharding (Stage 1) PLUS the
// CPU-side diagonal-vector cache, on top of the same Token-SIMD B=8 T=103
// depth-13/digits-3/ring-65536 base as both parents. Scope: prove the two
// previously-independently-passing levers still pass correctness when
// combined in one process. NOT a speed run, NOT the 12-block end-to-end
// run -- see docs/hybrid/tasks.md's dated entry that reopened this gate.
//
// Fork discipline -- THREE parents, all pinned and verified computed fresh
// against this repo's working tree (not trusted from any stale prior
// value). Neither parent is edited by this file.
//   shard-reader parent (the passing 2-GPU Q/K/V sharded reader this file
//   forks almost verbatim --
//   real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp)
//     SHA-256: 3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89
//   structural parent (deserialize-then-LoadContext shape, Client class,
//   never regenerates keys -- real_dnagpt_fides_scheme_b_serialize_reader.cpp,
//   unchanged transitively via the shard-reader parent)
//     SHA-256: 4dbb003fe8407046c852bf10f0c45b23b579cd1bbc7390099966999b6d6ab5d6
//   schedule parent (Token-SIMD B=8 physical_slot mapping, baby_rotations/
//   matmul BSGS transform, layernorm, required_rotation_keys() --
//   real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp,
//   unchanged transitively via the shard-reader parent)
//     SHA-256: 6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e
//   cpudiagcache parent (the passing CPU-side diagonal-vector cache this
//   file ports verbatim into matmul() --
//   real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp)
//     SHA-256: 686c8b766436fde0fd4422f6a29f80361f8c8a2565768577857e9264b2976fd5
//
// What actually changed vs. the shard-reader parent: ONLY
// EncryptedEvaluator's matmul() and its private cache-lookup helper
// (cached_packed_values(), ported verbatim from the cpudiagcache parent's
// own method of the same name/body). Everything else -- option parsing,
// --part handling, Client class, fixture loading, cross-process
// deserialize/LoadContext sequencing, evidence-JSON shape -- is byte-for-
// byte unchanged from the shard-reader parent except for the additive
// diagonal_cache_hits/diagonal_cache_misses fields and the new
// --cpudiagcache-parent-sha256 pin.
//
// Per-shard cache keying still makes sense unmodified: the cache is keyed
// by the weight vector's stable host address (cached_packed_values() below,
// verbatim from the cpudiagcache parent), and each of the two shard workers
// only ever calls matmul() with the 1-2 weight addresses drawn from its own
// --part selection (fixture_.qkv[0]/[1]/[2] for query/key/value
// respectively) -- there is no cross-part or cross-process cache sharing
// question here, because each worker is its own process with its own
// std::map, and it never touches a weight address outside its own
// `requested` set. Per weight, cached_packed_values() is called
// BSGS_N1*BSGS_N2=1024 times per matmul() call (once per BSGS diagonal, all
// built and cached together on the first call for that weight) -- so a
// worker computing 1 part (e.g. "value") sees exactly 1 cache miss (its
// very first cached_packed_values() call) and
// 13*1024 - 1 = 13311 hits (every other lookup, across all 13 token
// groups' matmul() calls for that one weight); a worker computing 2 parts
// (e.g. "query,key") sees exactly 2 misses and 2*13311 = 26622 hits --
// asserted as an explicit invariant in main() below, the same discipline
// the cpudiagcache parent's own EXPECTED_DIAGONAL_CACHE_MISSES/_HITS
// constants use (there: 12 misses across 156 matmul calls x 1024
// diagonals).
//
// Same structural invariant as both parents: this reader contains ZERO
// occurrences of GenCryptoContext/EvalMultKeyGen/EvalRotateKeyGen/KeyGen --
// verified by test_shard_reader_cpudiagcache_contract.py, not just asserted
// here. The cache holds only a host-side std::vector<double> (never a
// Plaintext/Ciphertext), and a brand-new Plaintext object is still
// constructed via raw_plain()/MakeCKKSPackedPlaintext on every single
// matmul() call, so no GPU-resident Plaintext/Ciphertext object is ever
// reused across multPt calls (the abandoned 2026-07-27/28 diagcache
// attempt's exact crash mechanism; see the cpudiagcache parent's own file
// header for the full root-cause account). This fork structurally cannot
// hit that mechanism for the same reason the cpudiagcache parent doesn't.
//
// Secret-key hygiene: identical discipline to both parents -- the secret
// key is deserialized here to simulate a restarted client process for
// measurement only; its path/contents are never printed. Not a production
// client/server key-custody design (out of scope per CLAUDE.md).
//
// --part flag: identical semantics to the shard-reader parent --
// comma-separated and/or repeatable, drawn from {query,key,value}; at least
// one is required; unknown tokens are rejected.

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
constexpr std::string_view PINNED_SHARD_READER_PARENT_SHA256 =
    "3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89";
constexpr std::string_view PINNED_STRUCTURAL_PARENT_SHA256 =
    "4dbb003fe8407046c852bf10f0c45b23b579cd1bbc7390099966999b6d6ab5d6";
constexpr std::string_view PINNED_SCHEDULE_PARENT_SHA256 =
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e";
constexpr std::string_view PINNED_CPUDIAGCACHE_PARENT_SHA256 =
    "686c8b766436fde0fd4422f6a29f80361f8c8a2565768577857e9264b2976fd5";
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
    std::string shard_reader_parent_sha256;
    std::string structural_parent_sha256;
    std::string schedule_parent_sha256;
    std::string cpudiagcache_parent_sha256;
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
    std::string source_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13 "
        "--gpu N --state-dir PATH --fixture-dir PATH --output PATH "
        "--part query,key,value (repeatable/comma-separated) "
        "--shard-reader-parent-sha256 SHA --structural-parent-sha256 SHA "
        "--schedule-parent-sha256 SHA --cpudiagcache-parent-sha256 SHA "
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
        } else if (arg == "--shard-reader-parent-sha256") {
            options.shard_reader_parent_sha256 = next();
        } else if (arg == "--structural-parent-sha256") {
            options.structural_parent_sha256 = next();
        } else if (arg == "--schedule-parent-sha256") {
            options.schedule_parent_sha256 = next();
        } else if (arg == "--cpudiagcache-parent-sha256") {
            options.cpudiagcache_parent_sha256 = next();
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
            std::cout << "usage: real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13 "
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
    if (options.shard_reader_parent_sha256 != PINNED_SHARD_READER_PARENT_SHA256) {
        usage_error("refusing changed shard-reader parent (simd_shard_reader_t103_depth13.cpp)");
    }
    if (options.structural_parent_sha256 != PINNED_STRUCTURAL_PARENT_SHA256) {
        usage_error("refusing changed structural parent (serialize_reader.cpp)");
    }
    if (options.schedule_parent_sha256 != PINNED_SCHEDULE_PARENT_SHA256) {
        usage_error("refusing changed Token-SIMD depth-13 schedule parent");
    }
    if (options.cpudiagcache_parent_sha256 != PINNED_CPUDIAGCACHE_PARENT_SHA256) {
        usage_error("refusing changed CPU-diagonal-cache parent");
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

// ---- Fixture loading -- byte-identical conventions to both parents. ----

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
    fixture.oracle[0] = read_f64(directory / "oracle__query.bin", HEADS * T * HEAD_DIM);
    fixture.oracle[1] = read_f64(directory / "oracle__key.bin", HEADS * T * HEAD_DIM);
    fixture.oracle[2] = read_f64(directory / "oracle__value.bin", HEADS * T * HEAD_DIM);
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
    std::size_t diagonal_cache_hits = 0;
    std::size_t diagonal_cache_misses = 0;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture, Client& client)
        : cc_(std::move(context)), fixture_(fixture), client_(client) {}

    // requested: which of {0=query,1=key,2=value} to compute. For every
    // group (all TOKEN_GROUPS, unconditionally), baby_rotations(normalized)
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
        result.diagonal_cache_hits = diagonal_cache_hits_;
        result.diagonal_cache_misses = diagonal_cache_misses_;
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

    // Builds (on the first call for a given weight matrix) or returns (on
    // every later call) the full set of BSGS_N1*BSGS_N2 diagonal
    // std::vector<double> for `weight`, keyed by the weight vector's stable
    // address -- ported verbatim from the cpudiagcache parent's method of
    // the same name/body (see this file's header comment). In THIS process,
    // `weight` is always one of fixture_.qkv[0..2], and only the addresses
    // in this worker's own `requested` set are ever passed in (see
    // evaluate() above), so the cache never grows past `requested.size()`
    // entries -- there is no cross-part or cross-process sharing here, only
    // reuse across this one worker's own 13 token-group calls per requested
    // part. Only the CPU-side vector<double> is cached -- a FRESH Plaintext
    // object is still built from it (via raw_plain()) on every single call
    // in matmul() below, so no GPU-resident Plaintext object is ever reused
    // across multPt calls.
    const std::vector<double>& cached_packed_values(
        const std::vector<double>& weight, std::size_t giant, std::size_t small) {
        auto found = diagonal_vector_cache_.find(&weight);
        if (found == diagonal_vector_cache_.end()) {
            std::vector<std::vector<double>> built(BSGS_N1 * BSGS_N2);
            for (std::size_t build_giant = 0; build_giant < BSGS_N2; ++build_giant) {
                for (std::size_t build_small = 0; build_small < BSGS_N1; ++build_small) {
                    const std::size_t diagonal = BSGS_N1 * build_giant + build_small;
                    std::vector<double> packed_values(SLOTS, 0.0);
                    for (std::size_t copy = 0; copy < COPIES; ++copy) {
                        for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                            const std::size_t rolled_row =
                                (row + PACK_WIDTH - BSGS_N1 * build_giant) % PACK_WIDTH;
                            const std::size_t column = (rolled_row + diagonal) % PACK_WIDTH;
                            if (rolled_row < D && column < D) {
                                const double coefficient = weight[rolled_row * D + column];
                                for (std::size_t token = 0; token < TOKEN_BATCH; ++token) {
                                    packed_values[physical_slot(copy, row, token)] =
                                        coefficient;
                                }
                            }
                        }
                    }
                    built[diagonal] = std::move(packed_values);
                }
            }
            found = diagonal_vector_cache_.emplace(&weight, std::move(built)).first;
            ++diagonal_cache_misses_;
        } else {
            ++diagonal_cache_hits_;
        }
        return found->second.at(BSGS_N1 * giant + small);
    }

    Ct matmul(const BabyRotations& baby, const std::vector<double>& weight) {
        if (weight.size() != D * D) {
            throw std::invalid_argument("QKV weight must be D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::vector<double>& packed_values =
                    cached_packed_values(weight, giant, small);
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
    std::size_t diagonal_cache_hits_ = 0;
    std::size_t diagonal_cache_misses_ = 0;
    std::map<const std::vector<double>*, std::vector<std::vector<double>>>
        diagonal_vector_cache_;
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
           "ring-65536 2-GPU process-per-GPU sharding (Stage 1, Q/K/V "
           "split) COMBINED with the CPU-side diagonal-vector cache -- "
           "correctness-only micro-gate, one of two concurrent shard "
           "workers\",\n"
        << "  \"implementation_version\": "
           "\"t103-scheme-b-token-simd-shard-reader-cpudiagcache-depth13-v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS; server-only "
           "linear algebra, client-only decrypt at pre-declared "
           "nonlinearity boundaries)\",\n"
        << "  \"label\": \"combines two previously-independently-passing "
           "levers: (1) this worker computes a disjoint subset of "
           "{query,key,value} from a context/key lineage deserialized from "
           "a separate writer process's --state-dir, concurrent with a "
           "second worker on a different physical GPU (zero ciphertext "
           "merge between them); (2) this worker's own BSGS diagonal "
           "std::vector<double> construction is cached per distinct weight "
           "address, independently of the other worker's cache (no "
           "cross-process cache sharing)\",\n"
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
        << "  \"shard_reader_parent_sha256\": \""
        << json_escape(options.shard_reader_parent_sha256) << "\",\n"
        << "  \"structural_parent_sha256\": \""
        << json_escape(options.structural_parent_sha256) << "\",\n"
        << "  \"schedule_parent_sha256\": \""
        << json_escape(options.schedule_parent_sha256) << "\",\n"
        << "  \"cpudiagcache_parent_sha256\": \""
        << json_escape(options.cpudiagcache_parent_sha256) << "\",\n"
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
        << TOKEN_GROUPS * options.parts.size() << ",\n"
        << "    \"diagonal_cache_hits\": " << evaluation.diagonal_cache_hits << ",\n"
        << "    \"diagonal_cache_misses\": " << evaluation.diagonal_cache_misses << ",\n"
        << "    \"expected_diagonal_cache_misses\": " << options.parts.size() << ",\n"
        << "    \"expected_diagonal_cache_hits\": "
        << (TOKEN_GROUPS * BSGS_N1 * BSGS_N2 - 1) * options.parts.size() << "\n"
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
        << "  \"timings_note\": \"informational only -- this host has shown "
           "severe, documented CPU contention (load average 500-1042, see "
           "docs/hybrid/tasks.md's 2026-07-31 depth-13 repeat-run "
           "retraction) that invalidates any single wall-clock sample as a "
           "speedup claim; no speedup is claimed from this run\",\n"
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
        // verified textually by test_shard_reader_cpudiagcache_contract.py.
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
        // selection is entirely a Docker "--gpus device=N" concern.
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
        // Per-shard cache invariant (see file header comment): this worker's
        // cache is keyed by weight address and only ever sees the 1-2
        // addresses in its own `requested` set. cached_packed_values() is
        // called BSGS_N1*BSGS_N2 times per matmul() call (once per BSGS
        // diagonal), and only the very first call for a given weight is a
        // miss (it builds and caches all diagonals at once) -- so it must
        // record exactly `requested.size()` misses (one per distinct
        // weight) and
        // `(TOKEN_GROUPS * BSGS_N1 * BSGS_N2 - 1) * requested.size()` hits
        // (every other lookup, across all 13 token groups' matmul() calls
        // for each requested weight). A mismatch here would mean the cache
        // is not actually being reused as designed, or is leaking across
        // weights it should not touch.
        const std::size_t expected_misses = requested.size();
        const std::size_t expected_hits =
            (TOKEN_GROUPS * BSGS_N1 * BSGS_N2 - 1) * requested.size();
        if (evaluation.diagonal_cache_misses != expected_misses ||
            evaluation.diagonal_cache_hits != expected_hits) {
            throw std::runtime_error(
                "Token-SIMD shard-reader diagonal-cache hit/miss count mismatch");
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
        std::cout << "diagonal_cache_hits=" << evaluation.diagonal_cache_hits
                  << " diagonal_cache_misses=" << evaluation.diagonal_cache_misses << '\n';
        std::cout << (all_passed
                          ? "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_READER_CPUDIAGCACHE_PASS"
                          : "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_READER_CPUDIAGCACHE_FAIL")
                  << '\n';
        return all_passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
