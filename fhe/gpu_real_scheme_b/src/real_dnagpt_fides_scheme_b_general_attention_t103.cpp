// Real-weight DNAGPT block-0 gate, Scheme B (hybrid client-assisted CKKS),
// GENERAL causal-attention prototype at T=103 -- the full task-representative
// GSR prompt length (docs/roadmap.md), using the exact tokenized prompt's
// entire 103-token length, not a truncated prefix.
//
// Forked from real_dnagpt_fides_scheme_b_general_attention_t32.cpp (itself
// frozen -- never edit that file, nor T=8/T=3/T=2 before it). Carries the
// T=8 file's output-packing fix forward unchanged (OUTPUT_GROUPS=
// ceil(T/COPIES)) and REPLACES the T=32 file's causal-score packing with a
// fundamentally different mechanism: T=32's spread-across-copies fix raised
// the packing ceiling from T<=21 to T<=85 by changing WHERE heads are
// packed, but any *fixed-size* per-row reservation still has a hard ceiling
// (HEADS_PER_COPY*T<=PACK_WIDTH-D), and T=103 exceeds it (T<=85 < 103). This
// file removes the ceiling entirely by never requiring a whole causal row's
// raw scores to fit in one ciphertext.
//
// Design (scoped and numpy-proven in docs/hybrid/tasks.md's 2026-07-29
// "chunked/streaming causal-score packing" entry BEFORE this file was
// written -- see that entry for the full derivation):
//
//   Fix CHUNK_WIDTH = (PACK_WIDTH-D)/HEADS_PER_COPY (=85 at current
//   constants) -- a CONSTANT, independent of T. Any causal row, however
//   long, is split into ceil(row_length/CHUNK_WIDTH) chunks of at most
//   CHUNK_WIDTH columns each.
//
//   Phase A (reduction, one crossing per chunk): for each chunk, the server
//   packs that chunk's raw scores exactly like the T=32 file's
//   spread-across-copies scheme (one-hot isolated per assigned copy/local
//   slot), but indexed by the column's position WITHIN the chunk, not its
//   global column index. The client (Client::cross_boundary_reduce_chunk)
//   decrypts and appends the recovered per-head values into its own
//   in-process cache for that row (`row_score_cache_`) -- an IDENTITY
//   transform, so the re-encrypted ciphertext this call returns is
//   unchanged and never used by the rest of the server-side graph; the
//   call exists only so the client learns the chunk's plaintext values,
//   going through the same cross_boundary_impl decrypt/re-encrypt mechanics
//   as every other boundary crossing (unchanged) for consistent
//   bookkeeping. Once every chunk for a row has been reduced,
//   Client::finalize_row_softmax computes that row's max and softmax
//   denominator directly over the FULL reassembled row -- no online/
//   incremental-max rescaling trick is needed (the numerical-stability
//   concern hardware-streaming flash-attention kernels solve): the client
//   is a single process holding the whole row in float64 memory once the
//   last chunk lands, so computing max/sum directly is exact. This is a
//   pure client-side computation, not a boundary crossing.
//
//   Phase B (selection, exactly `row` crossings per row -- SAME COUNT as
//   every prior general-attention file): for causal column j, the client
//   already has s_ij and the row's max/sum cached from phase A, so no new
//   decrypt is needed at all -- Client::cross_boundary_emit_weight is
//   ENCRYPT-ONLY (no Decrypt call), computing
//   weight_ij=exp(s_ij-max)/sum from its own cache and broadcasting it
//   into a fresh ciphertext across all COPIES copies exactly as the T=32
//   file's write side did (that half is genuinely unchanged).
//
// Numpy-proven before this file was written (docs/hybrid/tasks.md): chunk
// partitioning is a bijection over every row length tested (including
// T=103's own row lengths); the two-phase design reproduces the current
// (T=32) single-shot softmax bit-for-bit at row lengths inside the old
// T<=85 ceiling; it matches a plain reference softmax at row lengths up to
// 1000 (impossible for the T=32 packing scheme past T=85); the honest
// round-trip growth formula is confirmed against this file's own real T=103
// oracle scores by the accompanying contract test.
//
// Cost, honestly disclosed, not tuned away: attention round trips grow from
// T=32's `T*(T-1)/2 = 496` to `sum_{row=1}^{T-1}(ceil((row+1)/85)+row) =
// 5373` at T=103 (verified by the contract test's numpy formula against
// this file's real oracle) -- still O(T^2), the same asymptotic growth as
// every prior general-attention file, with a modest constant-factor
// overhead from the reduction-phase crossings (one extra crossing per
// 85-column chunk, not per column).
//
// Everything else -- BSGS matmul, encrypted-input boundary, LayerNorm/GELU
// client boundaries, the 4e-2 dual accuracy gate, the T=8 output-packing
// fix -- is untouched from the frozen T=32 file.

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
constexpr std::size_t T = 103;
constexpr std::size_t HEADS = 12;
constexpr std::size_t HEAD_DIM = 64;
constexpr std::size_t MLP_DIM = 3072;
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t SLOTS = COPIES * PACK_WIDTH;
constexpr std::size_t BSGS_N1 = 32;
constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;
// Heads assigned per physical copy for the causal-score packing (unchanged
// from the T=32 file -- this part of that fix is still correct, just no
// longer the whole story once a row exceeds one chunk).
constexpr std::size_t HEADS_PER_COPY = HEADS / COPIES;
static_assert(HEADS % COPIES == 0,
             "HEADS must divide evenly across COPIES for the score-packing "
             "spread assignment");
// The actual fix in this file: a FIXED chunk width, independent of T. Any
// causal row, however long, is split into ceil(row_length/CHUNK_WIDTH)
// chunks of at most this many columns -- removing the T=32 file's
// SCORE_SLOT_STRIDE=T bound (which forced HEADS_PER_COPY*T<=PACK_WIDTH-D,
// i.e. T<=85) entirely.
constexpr std::size_t CHUNK_WIDTH = (PACK_WIDTH - D) / HEADS_PER_COPY;
static_assert(HEADS_PER_COPY * CHUNK_WIDTH <= PACK_WIDTH - D,
             "one chunk's packed causal-score reservation must not collide "
             "with the D real dimensions any head span also occupies");
static_assert(CHUNK_WIDTH > 0, "chunk width must be positive");
// See MULT_DEPTH comment in the frozen T=2/T=8/T=32 files: same starting
// budget, carried forward unchanged. Phase A's one-hot isolation is still
// exactly one multiply_plain per causal score (same as T=32's), and Phase B
// consumes no decrypted-value depth at all (it is an encrypt-only fresh
// ciphertext at level 0) -- so this file's depth pressure is unchanged from
// T=32's, confirmed empirically via this run's own level_trace field.
constexpr std::uint32_t MULT_DEPTH = 16;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 4;
constexpr double EPS = 1e-5;
constexpr double TOL = 4e-2;

constexpr std::string_view PINNED_FIDES_COMMIT =
    "786c7600fb2f16b724e0acf73df367b27b8afed6";
// T=103 fixture (gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0) -- the
// FULL tokenized GSR prompt length (not a truncated prefix), same
// checkpoint and GSR FASTA as the frozen T=2/T=3/T=8/T=32 fixtures.
constexpr std::string_view PINNED_FIXTURE_MANIFEST =
    "d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6";

using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;
using Cc = CryptoContext<DCRTPoly>;
using Keys = KeyPair<DCRTPoly>;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Exact elementwise nonlinearities. Scheme A approximates these with fixed
// public-domain Chebyshev series; Scheme B computes them exactly in the
// client's plaintext domain, so no calibrated input range is required.
double exact_invsqrt(double value) { return 1.0 / std::sqrt(value); }

double exact_gelu(double value) {
    constexpr double two_over_pi = 0.63661977236758134308;
    return 0.5 * value *
           (1.0 + std::tanh(std::sqrt(two_over_pi) *
                            (value + 0.044715 * value * value * value)));
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
    std::string gate;
    std::filesystem::path fixture_dir;
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
    std::string source_sha256 = "UNSPECIFIED";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_general_attention_t103 --gpu N "
        "--gate ln1|attention|full "
        "--fixture-dir PATH --output PATH --fixture-manifest-sha256 SHA "
        "[--backend-commit SHA] [--container-image NAME] [--environment TEXT] "
        "[--source-sha256 SHA] [--fixture-contract-sha256 SHA]");
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
        } else if (arg == "--gate") {
            options.gate = next();
        } else if (arg == "--fixture-dir") {
            options.fixture_dir = next();
        } else if (arg == "--output") {
            options.output = next();
        } else if (arg == "--backend-commit") {
            options.backend_commit = next();
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
            std::cout << "usage: real_dnagpt_fides_scheme_b_general_attention_t103 "
                         "--gpu N --gate ln1|attention|full --fixture-dir PATH "
                         "--output PATH --fixture-manifest-sha256 SHA\n";
            std::exit(0);
        } else {
            usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0) {
        usage_error("--gpu must be non-negative");
    }
    if (options.gate != "ln1" && options.gate != "attention" &&
        options.gate != "full") {
        usage_error("--gate must be ln1, attention, or full");
    }
    if (options.fixture_dir.empty() || options.output.empty()) {
        usage_error("--fixture-dir and --output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        usage_error("refusing unpinned FIDESlib commit " + options.backend_commit);
    }
    if (options.fixture_manifest_sha256 != PINNED_FIXTURE_MANIFEST) {
        usage_error("refusing unpinned fixture manifest " +
                    options.fixture_manifest_sha256);
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
                                 options.output.string());
    }
    return options;
}

struct Matrix {
    std::size_t rows{};
    std::size_t cols{};
    std::vector<double> values;

    Matrix() = default;
    Matrix(std::size_t row_count, std::size_t col_count)
        : rows(row_count), cols(col_count), values(row_count * col_count) {}

    double operator()(std::size_t row, std::size_t col) const {
        return values.at(row * cols + col);
    }
};

std::vector<double> read_f64(const std::filesystem::path& path,
                             std::size_t expected_values) {
    const std::uintmax_t expected_bytes = expected_values * sizeof(double);
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("fixture array missing: " + path.string());
    }
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error("fixture array size mismatch for " + path.string() +
                                 ": expected " + std::to_string(expected_bytes) +
                                 ", got " + std::to_string(actual_bytes));
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("could not open fixture array: " + path.string());
    }
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

Matrix load_matrix(const std::filesystem::path& path, std::size_t rows,
                   std::size_t cols) {
    Matrix output(rows, cols);
    output.values = read_f64(path, rows * cols);
    return output;
}

Matrix rows(const Matrix& input, std::size_t first, std::size_t count) {
    if (first + count > input.rows) {
        throw std::invalid_argument("row slice outside matrix");
    }
    Matrix output(count, input.cols);
    for (std::size_t row = 0; row < count; ++row) {
        std::copy_n(input.values.begin() +
                        static_cast<std::ptrdiff_t>((first + row) * input.cols),
                    input.cols,
                    output.values.begin() +
                        static_cast<std::ptrdiff_t>(row * output.cols));
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
            output.values[row * count + col] =
                input(row, first + col);
        }
    }
    return output;
}

struct Fixture {
    std::vector<double> input;
    std::vector<double> ln1;
    std::array<Matrix, 3> qkv;
    Matrix attention_projection;
    std::vector<double> ln2;
    std::array<Matrix, 4> mlp_fc;
    std::array<Matrix, 4> mlp_projection;
    std::vector<double> oracle_ln1;
    std::vector<double> oracle_attention_projection;
    std::vector<double> oracle_block_output;
};

Fixture load_fixture(const std::filesystem::path& directory) {
    Fixture fixture;
    fixture.input = read_f64(directory / "input__embeddings.bin", T * D);
    fixture.ln1 = read_f64(directory / "weights__ln1.bin", D);
    fixture.ln2 = read_f64(directory / "weights__ln2.bin", D);

    const Matrix qkv =
        load_matrix(directory / "weights__attn_qkv.bin", 3 * D, D);
    for (std::size_t part = 0; part < 3; ++part) {
        fixture.qkv[part] = rows(qkv, part * D, D);
    }
    fixture.attention_projection =
        load_matrix(directory / "weights__attn_proj.bin", D, D);

    const Matrix fc =
        load_matrix(directory / "weights__mlp_fc.bin", MLP_DIM, D);
    for (std::size_t part = 0; part < 4; ++part) {
        fixture.mlp_fc[part] = rows(fc, part * D, D);
    }
    const Matrix projection =
        load_matrix(directory / "weights__mlp_proj.bin", D, MLP_DIM);
    for (std::size_t part = 0; part < 4; ++part) {
        fixture.mlp_projection[part] = columns(projection, part * D, D);
    }

    fixture.oracle_ln1 =
        read_f64(directory / "oracle__ln1_output.bin", T * D);
    fixture.oracle_attention_projection =
        read_f64(directory / "oracle__attention_projection.bin", T * D);
    fixture.oracle_block_output =
        read_f64(directory / "oracle__block_output.bin", T * D);
    return fixture;
}

const std::vector<double>& oracle_for_gate(const Fixture& fixture,
                                           const std::string& gate) {
    if (gate == "ln1") {
        return fixture.oracle_ln1;
    }
    if (gate == "attention") {
        return fixture.oracle_attention_projection;
    }
    return fixture.oracle_block_output;
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

std::vector<int> required_rotation_keys(const std::string& gate) {
    std::set<int> indices;
    if (gate != "ln1") {
        for (std::size_t baby = 1; baby < BSGS_N1; ++baby) {
            indices.insert(static_cast<int>(baby));
        }
        for (std::size_t giant = 1; giant < BSGS_N2; ++giant) {
            indices.insert(static_cast<int>(BSGS_N1 * giant));
        }
    }
    add_accumulate_keys(indices, static_cast<int>(PACK_WIDTH), 1);
    if (gate == "full") {
        indices.insert(canonical_rotation(static_cast<std::int64_t>(PACK_WIDTH)));
        indices.insert(canonical_rotation(-static_cast<std::int64_t>(PACK_WIDTH)));
        indices.insert(
            canonical_rotation(2 * static_cast<std::int64_t>(PACK_WIDTH)));
    }
    indices.erase(0);
    return {indices.begin(), indices.end()};
}

// Output-packing group size, unchanged from the T=8/T=32 files' fix: groups
// tokens into ceil(T/COPIES) output ciphertexts, addressed by each token's
// position WITHIN its group rather than its raw index. Independent of the
// causal-score packing fix this file makes; already generalizes to any T.
constexpr std::size_t OUTPUT_GROUPS = (T + COPIES - 1) / COPIES;

// One chunk of a causal row: global columns [start, start+count).
struct ChunkRange {
    std::size_t start;
    std::size_t count;
};

// Partition columns 0..row_length-1 into chunks of at most CHUNK_WIDTH
// columns each -- the mechanism that removes the T=32 file's T<=85 ceiling.
// A chunk_width parameter (rather than always CHUNK_WIDTH) lets the numpy/
// unit-test side exercise other widths; production code always calls this
// with CHUNK_WIDTH.
std::vector<ChunkRange> chunk_row(std::size_t row_length,
                                  std::size_t chunk_width) {
    if (chunk_width == 0) {
        throw std::invalid_argument("chunk width must be positive");
    }
    std::vector<ChunkRange> chunks;
    std::size_t start = 0;
    while (start < row_length) {
        const std::size_t count = std::min(chunk_width, row_length - start);
        chunks.push_back({start, count});
        start += count;
    }
    return chunks;
}

struct EvaluationResult {
    std::vector<Ct> packed_output;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels;
    std::size_t packed_level = 0;
    std::size_t ciphertext_plain_matmuls = 0;
    std::size_t ciphertext_ciphertext_multiplications = 0;
    std::size_t ciphertext_plaintext_multiplications = 0;
};

struct BoundaryEvent {
    std::string name;
    double seconds = 0.0;
    std::size_t logical_instances = 0;
};

// Client: the only party in this process that holds the secret key.
// Compared to the T=32 file's Client, this adds THREE new public methods
// (cross_boundary_reduce_chunk, finalize_row_softmax,
// cross_boundary_emit_weight) implementing the two-phase chunked softmax;
// cross_boundary/cross_boundary_active and the private cross_boundary_impl
// mechanics are copied byte-for-byte unchanged.
class Client {
  public:
    Client(Cc cc, Keys& keys) : cc_(std::move(cc)), keys_(keys) {}

    Ct cross_boundary(const std::string& name, const Ct& ciphertext,
                      const std::function<double(double)>& transform,
                      std::size_t logical_instances = 1) {
        return cross_boundary_impl(
            name, ciphertext,
            [&transform](std::vector<double>& values) {
                for (double& value : values) {
                    value = transform(value);
                }
            },
            logical_instances);
    }

    Ct cross_boundary_active(const std::string& name, const Ct& ciphertext,
                             const std::function<double(double)>& transform,
                             std::size_t logical_instances) {
        return cross_boundary_impl(
            name, ciphertext,
            [&transform](std::vector<double>& values) {
                std::vector<double> transformed(SLOTS);
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    const std::size_t start = copy * PACK_WIDTH;
                    for (std::size_t dim = 0; dim < D; ++dim) {
                        transformed[start + dim] = transform(values[start + dim]);
                    }
                }
                values = std::move(transformed);
            },
            logical_instances);
    }

    // Phase A (reduction). Decrypts one chunk of one causal row's raw
    // scores (already isolated per assigned copy/local slot by the server,
    // exactly as the T=32 file's write side did, but indexed by the
    // column's position WITHIN this chunk) and appends the recovered
    // per-head values into row_score_cache_ -- an IDENTITY transform, so
    // the ciphertext returned to the server is unchanged and never used
    // downstream; this call exists only so the client learns the chunk's
    // plaintext values via the same decrypt/re-encrypt bookkeeping every
    // other boundary crossing uses.
    Ct cross_boundary_reduce_chunk(const std::string& name,
                                   const Ct& ciphertext, std::size_t row,
                                   std::size_t heads,
                                   std::size_t heads_per_copy,
                                   std::size_t chunk_count,
                                   std::size_t chunk_width,
                                   std::size_t logical_instances) {
        return cross_boundary_impl(
            name, ciphertext,
            [this, row, heads, heads_per_copy, chunk_count,
             chunk_width](std::vector<double>& values) {
                auto& cache = row_score_cache_[row];
                if (cache.empty()) {
                    cache.resize(heads);
                }
                for (std::size_t head = 0; head < heads; ++head) {
                    const std::size_t copy = head / heads_per_copy;
                    const std::size_t local_head = head % heads_per_copy;
                    const std::size_t read_start =
                        copy * PACK_WIDTH + local_head * chunk_width;
                    for (std::size_t local_col = 0; local_col < chunk_count;
                         ++local_col) {
                        cache[head].push_back(values[read_start + local_col]);
                    }
                }
                // `values` intentionally left unchanged -- see the class
                // comment above.
            },
            logical_instances);
    }

    // Pure client-side computation, NOT a boundary crossing (no ciphertext
    // moves): once every chunk for a row has been reduced, computes that
    // row's max and softmax denominator directly over the full
    // reassembled row. No online/incremental-max rescaling is needed since
    // the client holds the whole row in float64 memory already.
    void finalize_row_softmax(std::size_t row, std::size_t heads) {
        const auto& cache = row_score_cache_.at(row);
        auto& max_vec = row_max_[row];
        auto& sum_vec = row_sum_[row];
        max_vec.resize(heads);
        sum_vec.resize(heads);
        for (std::size_t head = 0; head < heads; ++head) {
            const auto& scores = cache.at(head);
            if (scores.empty()) {
                throw std::runtime_error(
                    "finalize_row_softmax called with no cached scores for "
                    "row " +
                    std::to_string(row));
            }
            double max_value = scores.front();
            for (const double value : scores) {
                max_value = std::max(max_value, value);
            }
            double sum_value = 0.0;
            for (const double value : scores) {
                sum_value += std::exp(value - max_value);
            }
            max_vec[head] = max_value;
            sum_vec[head] = sum_value;
        }
    }

    // Phase B (selection). ENCRYPT-ONLY -- no Decrypt call at all, unlike
    // every other boundary crossing in this file family: the client
    // already has column `col`'s raw score (cached from phase A) and the
    // row's max/sum (from finalize_row_softmax), so it computes
    // weight_ij=exp(s_ij-max)/sum purely from its own memory and encrypts
    // a fresh broadcast, replicated across all COPIES copies exactly as
    // the T=32 file's write side did (that half is genuinely unchanged --
    // the value ciphertexts this multiplies against are still fully
    // replicated). Still counted as a round trip/boundary crossing for
    // honest evidence bookkeeping, since a fresh ciphertext leaves the
    // client at this point.
    Ct cross_boundary_emit_weight(const std::string& name, std::size_t row,
                                  std::size_t col, std::size_t heads,
                                  std::size_t write_span,
                                  std::size_t logical_instances) {
        if (logical_instances == 0) {
            throw std::invalid_argument(
                "client boundary must represent at least one logical instance");
        }
        const auto start = Clock::now();
        const auto& cache = row_score_cache_.at(row);
        const auto& max_vec = row_max_.at(row);
        const auto& sum_vec = row_sum_.at(row);
        std::vector<double> transformed(SLOTS, 0.0);
        for (std::size_t head = 0; head < heads; ++head) {
            const double weight =
                std::exp(cache.at(head).at(col) - max_vec.at(head)) /
                sum_vec.at(head);
            for (std::size_t out_copy = 0; out_copy < COPIES; ++out_copy) {
                const std::size_t write_start =
                    out_copy * PACK_WIDTH + head * write_span;
                for (std::size_t dim = 0; dim < write_span; ++dim) {
                    transformed[write_start + dim] = weight;
                }
            }
        }
        Plaintext refreshed =
            cc_->MakeCKKSPackedPlaintext(transformed, 1, 0, nullptr, SLOTS);
        Ct out = cc_->Encrypt(keys_.publicKey, refreshed);
        ++round_trips_;
        logical_boundary_instances_ += logical_instances;
        const double seconds = elapsed_seconds(start);
        boundary_seconds_total_ += seconds;
        boundary_log_.push_back({name, seconds, logical_instances});
        return out;
    }

    // Frees a row's cached scores/statistics once its context accumulation
    // is complete -- bounds memory to O(max row length), not O(T^2 rows).
    void clear_row_cache(std::size_t row) {
        row_score_cache_.erase(row);
        row_max_.erase(row);
        row_sum_.erase(row);
    }

    std::size_t round_trips() const { return round_trips_; }
    std::size_t logical_boundary_instances() const {
        return logical_boundary_instances_;
    }
    double boundary_seconds_total() const { return boundary_seconds_total_; }
    const std::vector<BoundaryEvent>& boundary_log() const {
        return boundary_log_;
    }

  private:
    Ct cross_boundary_impl(
        const std::string& name, const Ct& ciphertext,
        const std::function<void(std::vector<double>&)>& transform,
        std::size_t logical_instances) {
        if (logical_instances == 0) {
            throw std::invalid_argument(
                "client boundary must represent at least one logical instance");
        }
        const auto start = Clock::now();
        Ct local = ciphertext;
        Plaintext plaintext;
        cc_->Decrypt(keys_.secretKey, local, &plaintext);
        plaintext->SetLength(SLOTS);
        std::vector<double> values = plaintext->GetRealPackedValue();
        transform(values);
        Plaintext refreshed = cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
        Ct out = cc_->Encrypt(keys_.publicKey, refreshed);
        ++round_trips_;
        logical_boundary_instances_ += logical_instances;
        const double seconds = elapsed_seconds(start);
        boundary_seconds_total_ += seconds;
        boundary_log_.push_back({name, seconds, logical_instances});
        return out;
    }

    Cc cc_;
    Keys& keys_;
    std::size_t round_trips_ = 0;
    std::size_t logical_boundary_instances_ = 0;
    double boundary_seconds_total_ = 0.0;
    std::vector<BoundaryEvent> boundary_log_;
    // Phase A/B state: row -> per-head raw score arrays (ordered by global
    // column index, built by concatenating chunks in order), and row ->
    // per-head (max, softmax-denominator) once finalized.
    std::map<std::size_t, std::vector<std::vector<double>>> row_score_cache_;
    std::map<std::size_t, std::vector<double>> row_max_;
    std::map<std::size_t, std::vector<double>> row_sum_;
};

class EncryptedEvaluator {
  public:
    EncryptedEvaluator(Cc context, const Fixture& fixture, Client& client)
        : cc_(std::move(context)), fixture_(fixture), client_(client) {}

    EvaluationResult evaluate(const std::array<Ct, T>& encrypted_inputs,
                              const std::string& gate) {
        levels_.clear();
        matmul_count_ = 0;
        ct_ct_multiply_count_ = 0;
        ct_plain_multiply_count_ = 0;

        std::array<Ct, T> normalized{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized[token] =
                layernorm(encrypted_inputs[token], fixture_.ln1, "ln1_token" +
                                                                       std::to_string(token));
        }
        record("ln1", normalized);
        if (gate == "ln1") {
            return finish(normalized);
        }

        std::array<Ct, T> query{};
        std::array<Ct, T> key{};
        std::array<Ct, T> value{};
        for (std::size_t token = 0; token < T; ++token) {
            const BabyRotations baby = baby_rotations(normalized[token]);
            query[token] = matmul(baby, fixture_.qkv[0]);
            key[token] = matmul(baby, fixture_.qkv[1]);
            value[token] = matmul(baby, fixture_.qkv[2]);
        }
        record("qkv", query);

        std::array<Ct, T> context{};
        context[0] = value[0];
        const double score_scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));

        for (std::size_t row = 1; row < T; ++row) {
            const std::size_t row_length = row + 1;
            const std::vector<ChunkRange> chunks =
                chunk_row(row_length, CHUNK_WIDTH);

            // Phase A: one client reduction crossing per chunk, spreading
            // heads across copies exactly like the T=32 file, but indexed
            // by position WITHIN the chunk rather than the global column.
            for (std::size_t chunk_index = 0; chunk_index < chunks.size();
                 ++chunk_index) {
                const ChunkRange& chunk = chunks[chunk_index];
                Ct packed_chunk;
                bool first_term = true;
                for (std::size_t head = 0; head < HEADS; ++head) {
                    std::vector<double> head_mask(D);
                    for (std::size_t dim = head * HEAD_DIM;
                         dim < (head + 1) * HEAD_DIM; ++dim) {
                        head_mask[dim] = 1.0;
                    }
                    const std::size_t copy = head / HEADS_PER_COPY;
                    const std::size_t local_head = head % HEADS_PER_COPY;
                    for (std::size_t local_col = 0; local_col < chunk.count;
                         ++local_col) {
                        const std::size_t col = chunk.start + local_col;
                        const Ct raw_score = attention_score(
                            query[row], key[col], head_mask, score_scale);
                        const Ct isolated = multiply_plain(
                            raw_score,
                            one_hot_single_copy_plain(
                                copy, local_head * CHUNK_WIDTH + local_col));
                        packed_chunk = first_term
                                           ? isolated
                                           : cc_->EvalAdd(packed_chunk, isolated);
                        first_term = false;
                    }
                }
                client_.cross_boundary_reduce_chunk(
                    "attention_reduce_row" + std::to_string(row) + "_chunk" +
                        std::to_string(chunk_index) + "_heads_batched",
                    packed_chunk, row, HEADS, HEADS_PER_COPY, chunk.count,
                    CHUNK_WIDTH, HEADS * chunk.count);
                std::cout << "[stage] attention reduce row=" << row
                          << " chunk=" << chunk_index
                          << " count=" << chunk.count << std::endl;
            }
            client_.finalize_row_softmax(row, HEADS);

            // Phase B: exactly `row` crossings, same count and shape as
            // every prior general-attention file -- only the client's own
            // computation inside each crossing changed (no decrypt now).
            Ct accumulated_context = value[0];
            for (std::size_t col = 1; col <= row; ++col) {
                Ct weight_broadcast = client_.cross_boundary_emit_weight(
                    "attention_select_row" + std::to_string(row) + "_col" +
                        std::to_string(col) + "_heads_batched",
                    row, col, HEADS, HEAD_DIM, HEADS);
                const Ct value_delta = cc_->EvalSub(value[col], value[0]);
                accumulated_context = cc_->EvalAdd(
                    accumulated_context, multiply(weight_broadcast, value_delta));
                std::cout << "[stage] attention row=" << row << " col=" << col
                           << " done, level=" << accumulated_context->GetLevel()
                           << std::endl;
            }
            context[row] = accumulated_context;
            client_.clear_row_cache(row);
        }
        record("attention_context", context);

        std::array<Ct, T> attention_projection{};
        for (std::size_t token = 0; token < T; ++token) {
            attention_projection[token] =
                matmul(baby_rotations(context[token]),
                       fixture_.attention_projection);
        }
        record("attention_projection", attention_projection);
        if (gate == "attention") {
            return finish(attention_projection);
        }

        std::array<Ct, T> residual1{};
        for (std::size_t token = 0; token < T; ++token) {
            residual1[token] =
                cc_->EvalAdd(encrypted_inputs[token], attention_projection[token]);
        }
        record("residual1", residual1);

        std::array<Ct, T> normalized2{};
        for (std::size_t token = 0; token < T; ++token) {
            normalized2[token] =
                layernorm(residual1[token], fixture_.ln2,
                          "ln2_token" + std::to_string(token));
        }
        record("ln2", normalized2);

        std::array<Ct, T> block_output{};
        for (std::size_t token = 0; token < T; ++token) {
            require_remaining_depth(
                normalized2[token]->GetLevel(), 4,
                "MLP token " + std::to_string(token));
            const BabyRotations fc_baby = baby_rotations(normalized2[token]);
            std::array<Ct, 4> hidden{};
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                hidden[chunk] = matmul(fc_baby, fixture_.mlp_fc[chunk]);
            }

            Ct packed_hidden;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                const Ct selected =
                    multiply_plain(hidden[chunk], copy_active_plain(chunk));
                packed_hidden =
                    chunk == 0 ? selected
                               : cc_->EvalAdd(packed_hidden, selected);
            }
            Ct packed_activated = client_.cross_boundary_active(
                "gelu_token" + std::to_string(token) + "_chunks_batched",
                packed_hidden, exact_gelu, 4);
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                const Ct selected = multiply_plain(
                    packed_activated, copy_active_plain(chunk));
                hidden[chunk] = replicate_copies(selected);
            }

            Ct mlp;
            for (std::size_t chunk = 0; chunk < 4; ++chunk) {
                Ct contribution =
                    matmul(baby_rotations(hidden[chunk]),
                           fixture_.mlp_projection[chunk]);
                mlp = chunk == 0 ? contribution
                                 : cc_->EvalAdd(mlp, contribution);
            }
            block_output[token] = cc_->EvalAdd(residual1[token], mlp);
            std::cout << "[stage] block_output token=" << token
                      << " level=" << block_output[token]->GetLevel() << std::endl;
        }
        record("block_output", block_output);
        require_remaining_depth(maximum_level(block_output), 1,
                                "final output packing");
        return finish(block_output);
    }

  private:
    struct BabyRotations {
        std::array<Ct, BSGS_N1> values{};
    };

    Plaintext raw_plain(const std::vector<double>& values) {
        if (values.size() != SLOTS) {
            throw std::invalid_argument("raw plaintext must have SLOTS values");
        }
        return cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
    }

    Plaintext repeated_plain(const std::vector<double>& values) {
        if (values.size() != D) {
            throw std::invalid_argument("repeated plaintext must have D values");
        }
        std::vector<double> packed(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            std::copy(values.begin(), values.end(),
                      packed.begin() +
                          static_cast<std::ptrdiff_t>(copy * PACK_WIDTH));
        }
        return raw_plain(packed);
    }

    // One-hot plaintext isolating a single reserved score slot within ONE
    // specific physical copy, at a slot local to whatever unit of work
    // (previously a whole row, now one chunk) is being packed. Unchanged
    // from the T=32 file.
    Plaintext one_hot_single_copy_plain(std::size_t copy,
                                        std::size_t slot_within_copy) {
        if (copy >= COPIES) {
            throw std::invalid_argument("one-hot copy outside COPIES");
        }
        if (slot_within_copy >= D) {
            throw std::invalid_argument(
                "packed score slot must stay inside the D..PACK_WIDTH slack");
        }
        std::vector<double> packed(SLOTS, 0.0);
        packed[copy * PACK_WIDTH + slot_within_copy] = 1.0;
        return raw_plain(packed);
    }

    // `local_slot` is the token's position WITHIN its output group (see
    // OUTPUT_GROUPS), not its raw token index. Unchanged from the T=8/T=32
    // fix.
    Plaintext output_block_plain(std::size_t local_slot) {
        if (local_slot >= COPIES) {
            throw std::invalid_argument("output local_slot outside COPIES");
        }
        std::vector<double> packed(SLOTS);
        std::fill_n(
            packed.begin() +
                static_cast<std::ptrdiff_t>(local_slot * PACK_WIDTH),
            D, 1.0);
        return raw_plain(packed);
    }

    Plaintext copy_active_plain(std::size_t copy) {
        if (copy >= COPIES) {
            throw std::invalid_argument("copy outside COPIES");
        }
        std::vector<double> packed(SLOTS);
        std::fill_n(
            packed.begin() +
                static_cast<std::ptrdiff_t>(copy * PACK_WIDTH),
            D, 1.0);
        return raw_plain(packed);
    }

    Ct multiply(const Ct& left, const Ct& right) {
        ++ct_ct_multiply_count_;
        return cc_->EvalMult(left, right);
    }

    Ct multiply_plain(const Ct& input, Plaintext plaintext) {
        ++ct_plain_multiply_count_;
        return cc_->EvalMult(input, plaintext);
    }

    void require_remaining_depth(std::size_t current_level,
                                 std::size_t levels_needed,
                                 const std::string& stage) const {
        if (current_level > MULT_DEPTH ||
            levels_needed > MULT_DEPTH - current_level) {
            throw std::runtime_error(
                "insufficient multiplicative depth before " + stage +
                ": current level " + std::to_string(current_level) +
                " + required " + std::to_string(levels_needed) +
                " exceeds depth " + std::to_string(MULT_DEPTH));
        }
    }

    template <typename Container>
    std::size_t maximum_level(const Container& ciphertexts) const {
        std::size_t maximum = 0;
        for (const Ct& ciphertext : ciphertexts) {
            maximum = std::max(maximum, ciphertext->GetLevel());
        }
        return maximum;
    }

    Ct sum_broadcast(const Ct& input) {
        return cc_->AccumulateSum(input, static_cast<int>(PACK_WIDTH), 1);
    }

    Ct mean_broadcast(const Ct& input) {
        Ct sum = sum_broadcast(input);
        return multiply_plain(
            sum, repeated_plain(std::vector<double>(D, 1.0 / static_cast<double>(D))));
    }

    Ct layernorm(const Ct& input, const std::vector<double>& weight,
                const std::string& stage_name) {
        Ct mean = mean_broadcast(input);
        Ct centered = cc_->EvalSub(input, mean);
        Ct variance = mean_broadcast(multiply(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        Ct inverse =
            client_.cross_boundary(stage_name + "_invsqrt", variance, exact_invsqrt);
        Ct normalized = multiply(centered, inverse);
        return multiply_plain(normalized, repeated_plain(weight));
    }

    Ct attention_score(const Ct& query, const Ct& key,
                       const std::vector<double>& head_mask, double scale) {
        Ct products = multiply(query, key);
        products = multiply_plain(products, repeated_plain(head_mask));
        Ct score = sum_broadcast(products);
        return multiply_plain(score,
                              repeated_plain(std::vector<double>(D, scale)));
    }

    Ct replicate_copies(const Ct& input) {
        const std::vector<std::int32_t> indices = {
            static_cast<std::int32_t>(PACK_WIDTH),
            -static_cast<std::int32_t>(PACK_WIDTH),
            static_cast<std::int32_t>(2 * PACK_WIDTH),
        };
        const auto rotated = cc_->EvalFastRotation(
            input, indices, cc_->GetCyclotomicOrder(), nullptr);
        if (rotated.size() != indices.size()) {
            throw std::runtime_error(
                "FIDESlib returned incomplete cross-copy rotations");
        }
        Ct output = input;
        for (const Ct& copy : rotated) {
            output = cc_->EvalAdd(output, copy);
        }
        return output;
    }

    BabyRotations baby_rotations(const Ct& input) {
        std::vector<std::int32_t> indices;
        indices.reserve(BSGS_N1 - 1);
        for (std::size_t index = 1; index < BSGS_N1; ++index) {
            indices.push_back(static_cast<std::int32_t>(index));
        }
        const auto rotated = cc_->EvalFastRotation(
            input, indices, cc_->GetCyclotomicOrder(), nullptr);
        if (rotated.size() != BSGS_N1 - 1) {
            throw std::runtime_error("FIDESlib returned incomplete baby rotations");
        }
        BabyRotations output;
        output.values[0] = input;
        for (std::size_t index = 1; index < BSGS_N1; ++index) {
            output.values[index] = rotated.at(index - 1);
        }
        return output;
    }

    Ct matmul(const BabyRotations& baby, const Matrix& weight) {
        if (weight.rows != D || weight.cols != D) {
            throw std::invalid_argument("real block matmul expects D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::size_t diagonal = BSGS_N1 * giant + small;
                std::vector<double> packed_values(SLOTS);
                for (std::size_t block = 0; block < COPIES; ++block) {
                    const std::size_t block_start = block * PACK_WIDTH;
                    for (std::size_t row = 0; row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH - BSGS_N1 * giant) %
                            PACK_WIDTH;
                        const std::size_t column =
                            (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            packed_values[block_start + row] =
                                weight(rolled_row, column);
                        }
                    }
                }
                Ct term =
                    multiply_plain(baby.values[small], raw_plain(packed_values));
                inner =
                    small == 0 ? term : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(
                    inner, static_cast<std::int32_t>(BSGS_N1 * giant));
            }
            result =
                giant == 0 ? inner : cc_->EvalAdd(result, inner);
        }
        ++matmul_count_;
        return result;
    }

    EvaluationResult finish(const std::array<Ct, T>& values) {
        require_remaining_depth(maximum_level(values), 1,
                                "selected gate output packing");
        std::vector<Ct> packed_groups;
        packed_groups.reserve(OUTPUT_GROUPS);
        std::size_t packed_level = 0;
        for (std::size_t group = 0; group < OUTPUT_GROUPS; ++group) {
            const std::size_t first_token = group * COPIES;
            const std::size_t last_token = std::min(T, first_token + COPIES);
            Ct packed;
            bool first = true;
            for (std::size_t token = first_token; token < last_token; ++token) {
                const std::size_t local_slot = token - first_token;
                Ct selected = multiply_plain(values[token],
                                             output_block_plain(local_slot));
                packed = first ? selected : cc_->EvalAdd(packed, selected);
                first = false;
            }
            packed_level = std::max(packed_level, packed->GetLevel());
            packed_groups.push_back(std::move(packed));
        }
        return {.packed_output = std::move(packed_groups),
                .levels = levels_,
                .packed_level = packed_level,
                .ciphertext_plain_matmuls = matmul_count_,
                .ciphertext_ciphertext_multiplications =
                    ct_ct_multiply_count_,
                .ciphertext_plaintext_multiplications =
                    ct_plain_multiply_count_};
    }

    template <typename Container>
    void record(const std::string& name, const Container& ciphertexts) {
        std::size_t minimum = std::numeric_limits<std::size_t>::max();
        std::size_t maximum = 0;
        for (const Ct& ciphertext : ciphertexts) {
            minimum = std::min(minimum, ciphertext->GetLevel());
            maximum = std::max(maximum, ciphertext->GetLevel());
        }
        levels_[name] = {minimum, maximum};
        std::cout << "[stage] " << name << " level_min=" << minimum
                  << " level_max=" << maximum << std::endl;
    }

    Cc cc_;
    const Fixture& fixture_;
    Client& client_;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels_;
    std::size_t matmul_count_ = 0;
    std::size_t ct_ct_multiply_count_ = 0;
    std::size_t ct_plain_multiply_count_ = 0;
};

struct Metrics {
    double global_rel_inf = 0.0;
    double worst_token_rel_inf = 0.0;
    double max_abs_error = 0.0;
    bool all_finite = false;
    bool passed = false;
};

// `got` is a flat T*D vector, already reassembled by main() from however
// many output groups finish() produced (see OUTPUT_GROUPS).
Metrics measure(const std::vector<double>& got,
                const std::vector<double>& reference) {
    if (got.size() != T * D || reference.size() != T * D) {
        throw std::invalid_argument("output/reference size mismatch");
    }
    Metrics metrics;
    metrics.max_abs_error = 0.0;
    metrics.worst_token_rel_inf = 0.0;
    metrics.all_finite = true;
    double global_denominator = 0.0;
    for (std::size_t token = 0; token < T; ++token) {
        double token_error = 0.0;
        double token_denominator = 0.0;
        for (std::size_t dim = 0; dim < D; ++dim) {
            const std::size_t actual_slot = token * D + dim;
            const std::size_t reference_slot = token * D + dim;
            const double actual = got[actual_slot];
            const double expected = reference[reference_slot];
            metrics.all_finite = metrics.all_finite && std::isfinite(actual);
            const double error = std::abs(actual - expected);
            token_error = std::max(token_error, error);
            token_denominator =
                std::max(token_denominator, std::abs(expected));
            metrics.max_abs_error = std::max(metrics.max_abs_error, error);
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

std::string make_json(const Options& options, std::uint32_t ring,
                      std::size_t rotation_keys,
                      const EvaluationResult& evaluation, const Metrics& metrics,
                      double fixture_seconds, double setup_seconds,
                      double encryption_seconds, double evaluation_seconds,
                      double decrypt_seconds, std::size_t round_trips,
                      std::size_t logical_boundary_instances,
                      double boundary_seconds_total,
                      const std::vector<BoundaryEvent>& boundary_log) {
    // Unlike the T=2/T=3/T=8/T=32 files, this file does NOT build a
    // per-name boundary_summary: at T=103 there are thousands of distinct
    // (row,col)/(row,chunk) crossing names, and a full breakdown would
    // bloat the evidence JSON to little benefit. The aggregate fields below
    // (round_trips, logical_boundary_instances, boundary_seconds_total)
    // remain fully honest; boundary_log itself still exists in-process if
    // deeper inspection is ever needed.
    (void)boundary_log;
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"FHE real-weight DNAGPT block-0 gate, Scheme B hybrid client-assisted CKKS, FIDESlib GPU, GENERAL causal attention (T>2), T=103 (full task-representative GSR prompt length) using a two-phase chunked/streaming softmax with a chunk width independent of T, removing the T=32 file's T<=85 packing ceiling\",\n"
        << "  \"implementation_version\": \"t103-scheme-b-chunked-softmax-v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS; server-only linear algebra, client-only decrypt at pre-declared nonlinearity boundaries)\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"gate\": \"" << json_escape(options.gate) << "\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \"" << json_escape(options.backend_commit)
        << "\",\n"
        << "  \"container_image\": \"" << json_escape(options.container_image)
        << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment) << "\",\n"
        << "  \"gpu\": " << options.gpu << ",\n"
        << "  \"security\": \"HEStd_128_classic\",\n"
        << "  \"source_sha256\": \"" << json_escape(options.source_sha256)
        << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"fixture_contract_sha256\": \""
        << json_escape(options.fixture_contract_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"model\": \"dna_gpt0.1b_m classification checkpoint block 0\",\n"
        << "    \"D\": " << D << ",\n"
        << "    \"T\": " << T << ",\n"
        << "    \"heads\": " << HEADS << ",\n"
        << "    \"head_dim\": " << HEAD_DIM << ",\n"
        << "    \"mlp_dim\": " << MLP_DIM << ",\n"
        << "    \"pack_width\": " << PACK_WIDTH << ",\n"
        << "    \"batch_slots\": " << SLOTS << ",\n"
        << "    \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"scaling_technique\": \"FLEXIBLEAUTO\",\n"
        << "    \"bootstraps\": 0,\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"bsgs\": {\"n1\": " << BSGS_N1 << ", \"n2\": "
        << BSGS_N2 << "},\n"
        << "    \"matrix_products\": " << evaluation.ciphertext_plain_matmuls
        << ",\n"
        << "    \"operation_counts\": {\n"
        << "      \"ciphertext_plaintext_matrix_products\": "
        << evaluation.ciphertext_plain_matmuls << ",\n"
        << "      \"explicit_ciphertext_ciphertext_multiplications\": "
        << evaluation.ciphertext_ciphertext_multiplications << ",\n"
        << "      \"explicit_ciphertext_plaintext_multiplications\": "
        << evaluation.ciphertext_plaintext_multiplications << "\n"
        << "    },\n"
        << "    \"attention_schedule\": \"two-phase chunked causal softmax: for row i>=1, row_length=i+1 raw scores per head are split into ceil(row_length/CHUNK_WIDTH) fixed-width chunks; phase A reduces each chunk to the client's cache (one crossing per chunk), phase B emits one encrypt-only weight broadcast per causal column (same count as the single-shot design); context_i = v0 + sum_{j=1}^{i} weight_ij*(v_j-v0)\",\n"
        << "    \"chunk_width\": " << CHUNK_WIDTH << ",\n"
        << "    \"heads_per_copy\": " << HEADS_PER_COPY << ",\n"
        << "    \"client_boundary_batching\": \"attention phase A: 12 heads in one ciphertext per (row,chunk) reduction crossing; attention phase B: 12 heads in one encrypt-only crossing per (row,column); GELU: four 768-value chunks in one ciphertext per token; LayerNorm remains per token\",\n"
        << "    \"public_domain_contract\": \"not applicable under Scheme B: nonlinearities are computed exactly at the client plaintext boundary, so no calibrated public input range is required\",\n"
        << "    \"depth_guards\": {\"mlp_required_levels\": 4, "
           "\"final_pack_required_levels\": 1, \"fail_closed\": true},\n"
        << "    \"packed_output_level\": " << evaluation.packed_level << ",\n"
        << "    \"level_trace\": {\n";
    std::size_t stage = 0;
    for (const auto& [name, range] : evaluation.levels) {
        out << "      \"" << json_escape(name) << "\": {\"min\": "
            << range.first << ", \"max\": " << range.second << "}";
        out << (++stage == evaluation.levels.size() ? "\n" : ",\n");
    }
    out << "    }\n"
        << "  },\n"
        << "  \"global_rel_inf\": " << metrics.global_rel_inf << ",\n"
        << "  \"worst_token_rel_inf\": " << metrics.worst_token_rel_inf
        << ",\n"
        << "  \"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << "  \"all_finite\": " << (metrics.all_finite ? "true" : "false")
        << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (metrics.passed ? "true" : "false") << ",\n"
        << "  \"protocol\": {\n"
        << "    \"round_trips\": " << round_trips << ",\n"
        << "    \"logical_boundary_instances\": "
        << logical_boundary_instances << ",\n"
        << "    \"boundary_seconds_total\": " << boundary_seconds_total
        << ",\n"
        << "    \"boundary_summary_omitted\": \"per-(row,col/chunk) breakdown suppressed from this file to keep evidence JSON size reasonable at T=103's round-trip count; see timings_seconds and protocol.round_trips/logical_boundary_instances for the honest aggregate\"\n"
        << "  },\n"
        << "  \"timings_seconds\": {\n"
        << "    \"load_verified_fixture\": " << fixture_seconds << ",\n"
        << "    \"context_keygen_load\": " << setup_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds << ",\n"
        << "    \"encrypted_evaluation\": " << evaluation_seconds << ",\n"
        << "    \"client_boundary_seconds_total\": " << boundary_seconds_total
        << ",\n"
        << "    \"server_linear_algebra_seconds\": "
        << (evaluation_seconds - boundary_seconds_total) << ",\n"
        << "    \"final_decrypt\": " << decrypt_seconds << "\n"
        << "  },\n"
        << "  \"input_boundary\": \"token plus position embeddings encrypted by client\",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"declared_client_boundary_crossings\": " << round_trips << ",\n"
        << "  \"declared_logical_boundary_instances\": "
        << logical_boundary_instances << ",\n"
        << "  \"final_decrypt_calls\": " << OUTPUT_GROUPS << ",\n"
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
        throw std::runtime_error("could not exclusively create evidence file " +
                                 path.string());
    }
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t count = ::write(descriptor, contents.data() + offset,
                                      contents.size() - offset);
        if (count <= 0) {
            ::close(descriptor);
            throw std::runtime_error("short write to evidence file " +
                                     path.string());
        }
        offset += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error("could not close evidence file " +
                                 path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        const auto fixture_start = Clock::now();
        const Fixture fixture = load_fixture(options.fixture_dir);
        const double fixture_seconds = elapsed_seconds(fixture_start);

        const std::vector<int> rotation_keys =
            required_rotation_keys(options.gate);
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
        std::cout << "[context] security=HEStd_128_classic ring=" << ring
                  << " depth=" << MULT_DEPTH << " slots=" << SLOTS
                  << " rotation_keys=" << rotation_keys.size()
                  << " gpu=" << options.gpu << std::endl;

        const auto encryption_start = Clock::now();
        std::array<Ct, T> encrypted_inputs{};
        for (std::size_t token = 0; token < T; ++token) {
            std::vector<double> block(PACK_WIDTH);
            const auto begin =
                fixture.input.begin() + static_cast<std::ptrdiff_t>(token * D);
            std::copy(begin, begin + static_cast<std::ptrdiff_t>(D),
                      block.begin());
            std::vector<double> packed;
            packed.reserve(SLOTS);
            for (std::size_t copy = 0; copy < COPIES; ++copy) {
                packed.insert(packed.end(), block.begin(), block.end());
            }
            Plaintext plaintext =
                cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
            encrypted_inputs[token] = cc->Encrypt(keys.publicKey, plaintext);
        }
        cc->Synchronize();
        const double encryption_seconds = elapsed_seconds(encryption_start);

        // Scheme B boundary: Client is the only holder of the secret key. It
        // performs the encrypted-input encryption above, the pre-declared
        // cross_boundary()/cross_boundary_reduce_chunk()/
        // cross_boundary_emit_weight() calls inside evaluate(), and the one
        // final Decrypt below.
        Client client(cc, keys);
        EncryptedEvaluator evaluator(cc, fixture, client);
        const auto evaluation_start = Clock::now();
        EvaluationResult evaluation =
            evaluator.evaluate(encrypted_inputs, options.gate);
        cc->Synchronize();
        const double evaluation_seconds = elapsed_seconds(evaluation_start);

        // One decrypt per output group (see OUTPUT_GROUPS): T=103 needs
        // ceil(103/4)=26 final ciphertexts.
        const auto decrypt_start = Clock::now();
        std::vector<double> output(T * D);
        for (std::size_t group = 0; group < evaluation.packed_output.size();
             ++group) {
            Plaintext decoded;
            cc->Decrypt(keys.secretKey, evaluation.packed_output[group],
                        &decoded);
            decoded->SetLength(SLOTS);
            const auto raw = decoded->GetRealPackedValue();
            const std::size_t first_token = group * COPIES;
            const std::size_t last_token = std::min(T, first_token + COPIES);
            for (std::size_t token = first_token; token < last_token;
                 ++token) {
                const std::size_t local_slot = token - first_token;
                std::copy_n(
                    raw.begin() +
                        static_cast<std::ptrdiff_t>(local_slot * PACK_WIDTH),
                    D, output.begin() + static_cast<std::ptrdiff_t>(token * D));
            }
        }
        const double decrypt_seconds = elapsed_seconds(decrypt_start);

        const Metrics metrics =
            measure(output, oracle_for_gate(fixture, options.gate));
        const std::string evidence = make_json(
            options, ring, rotation_keys.size(), evaluation, metrics,
            fixture_seconds, setup_seconds, encryption_seconds,
            evaluation_seconds, decrypt_seconds, client.round_trips(),
            client.logical_boundary_instances(),
            client.boundary_seconds_total(), client.boundary_log());
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout << (metrics.passed
                          ? "REAL_DNAGPT_FIDES_SCHEME_B_GATE_PASS"
                          : "REAL_DNAGPT_FIDES_SCHEME_B_GATE_FAIL")
                  << '\n';
        return metrics.passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
