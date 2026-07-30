// Complete released DNAGPT graph under Scheme B at T=2:
// encrypted embeddings -> all 12 transformer blocks -> encrypted N-minus-A
// classifier margin. This is a correctness/composition gate, not a speed claim.
//
// Fork discipline:
//   - includes the frozen single-block evaluator at SHA-256
//     d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df;
//   - reuses one crypto context and key lineage for every block and the head;
//   - performs eleven declared full-hidden-state client refreshes between blocks;
//   - performs one declared block-11-to-head last-token refresh;
//   - keeps the final-LN/head-linear/SiLU/head-LN/readout graph encrypted except
//     for exact nonlinearities at declared client boundaries.
//
// Not evidence until a contract-bound results/runs JSON lands.

#define main real_dnagpt_fides_scheme_b_single_block_main_unused
#include "real_dnagpt_fides_scheme_b.cpp"
#undef main

namespace {

constexpr std::size_t ALL_BLOCKS = 12;
constexpr std::size_t EXPECTED_BLOCK_ROUND_TRIPS = 7;
constexpr std::size_t EXPECTED_BLOCK_LOGICAL_INSTANCES = 24;
constexpr std::size_t EXPECTED_INTER_BLOCK_REFRESHES = ALL_BLOCKS - 1;
constexpr std::size_t EXPECTED_HEAD_INPUT_REFRESHES = 1;
constexpr std::size_t EXPECTED_HEAD_ROUND_TRIPS = 3;
constexpr std::size_t EXPECTED_TOTAL_CROSSINGS =
    ALL_BLOCKS * EXPECTED_BLOCK_ROUND_TRIPS +
    EXPECTED_INTER_BLOCK_REFRESHES + EXPECTED_HEAD_INPUT_REFRESHES +
    EXPECTED_HEAD_ROUND_TRIPS;
constexpr std::size_t EXPECTED_TOTAL_LOGICAL_INSTANCES =
    ALL_BLOCKS * EXPECTED_BLOCK_LOGICAL_INSTANCES +
    EXPECTED_INTER_BLOCK_REFRESHES * T + EXPECTED_HEAD_INPUT_REFRESHES +
    EXPECTED_HEAD_ROUND_TRIPS;
constexpr std::string_view ALL_BLOCKS_FIXTURE_MANIFEST =
    "3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c";
constexpr std::string_view FROZEN_SCHEME_B_SOURCE_SHA256 =
    "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df";

static_assert(EXPECTED_TOTAL_CROSSINGS == 99);
static_assert(EXPECTED_TOTAL_LOGICAL_INSTANCES == 314);

struct AllBlocksOptions {
    int gpu = 0;
    std::filesystem::path fixture_dir;
    std::filesystem::path output;
    std::string backend_commit = std::string(PINNED_FIDES_COMMIT);
    std::string fixture_manifest_sha256;
    std::string fixture_contract_sha256 = "UNSPECIFIED";
    std::string container_image = "UNSPECIFIED";
    std::string environment = "Brev GPU";
    std::string source_sha256 = "UNSPECIFIED";
    std::string frozen_source_sha256 = "UNSPECIFIED";
};

[[noreturn]] void all_blocks_usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_all_blocks_head_t2 --gpu N "
        "--fixture-dir PATH --output PATH --fixture-manifest-sha256 SHA "
        "--fixture-contract-sha256 SHA --source-sha256 SHA "
        "--frozen-source-sha256 SHA [--backend-commit SHA] "
        "[--container-image NAME] [--environment TEXT]");
}

AllBlocksOptions parse_all_blocks_options(int argc, char** argv) {
    AllBlocksOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                all_blocks_usage_error("missing value for " + arg);
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
        } else if (arg == "--fixture-contract-sha256") {
            options.fixture_contract_sha256 = next();
        } else if (arg == "--container-image") {
            options.container_image = next();
        } else if (arg == "--environment") {
            options.environment = next();
        } else if (arg == "--source-sha256") {
            options.source_sha256 = next();
        } else if (arg == "--frozen-source-sha256") {
            options.frozen_source_sha256 = next();
        } else {
            all_blocks_usage_error("unknown argument " + arg);
        }
    }

    if (options.gpu < 0 || options.fixture_dir.empty() ||
        options.output.empty()) {
        all_blocks_usage_error("gpu, fixture-dir, and output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        all_blocks_usage_error("refusing unpinned FIDESlib commit " +
                               options.backend_commit);
    }
    if (options.fixture_manifest_sha256 != ALL_BLOCKS_FIXTURE_MANIFEST) {
        all_blocks_usage_error("refusing unpinned all-block fixture manifest " +
                               options.fixture_manifest_sha256);
    }
    if (options.frozen_source_sha256 != FROZEN_SCHEME_B_SOURCE_SHA256) {
        all_blocks_usage_error("refusing changed frozen Scheme B source " +
                               options.frozen_source_sha256);
    }
    if (options.source_sha256 == "UNSPECIFIED" ||
        options.fixture_contract_sha256 == "UNSPECIFIED") {
        all_blocks_usage_error(
            "source and fixture contract identities are required");
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
                                 options.output.string());
    }
    return options;
}

Fixture load_all_blocks_fixture(const std::filesystem::path& directory,
                                std::size_t block_index) {
    if (block_index >= ALL_BLOCKS) {
        throw std::invalid_argument("released block index must be in [0, 11]");
    }

    Fixture fixture;
    fixture.input = read_f64(directory / "input__embeddings.bin", T * D);
    const std::string prefix =
        "weights__block" + std::to_string(block_index) + "__";
    fixture.ln1 = read_f64(directory / (prefix + "ln1.bin"), D);
    fixture.ln2 = read_f64(directory / (prefix + "ln2.bin"), D);

    const Matrix qkv =
        load_matrix(directory / (prefix + "attn_qkv.bin"), 3 * D, D);
    for (std::size_t part = 0; part < 3; ++part) {
        fixture.qkv[part] = rows(qkv, part * D, D);
    }
    fixture.attention_projection =
        load_matrix(directory / (prefix + "attn_proj.bin"), D, D);

    const Matrix fc =
        load_matrix(directory / (prefix + "mlp_fc.bin"), MLP_DIM, D);
    for (std::size_t part = 0; part < 4; ++part) {
        fixture.mlp_fc[part] = rows(fc, part * D, D);
    }
    const Matrix projection =
        load_matrix(directory / (prefix + "mlp_proj.bin"), D, MLP_DIM);
    for (std::size_t part = 0; part < 4; ++part) {
        fixture.mlp_projection[part] =
            columns(projection, part * D, D);
    }

    fixture.oracle_block_output = read_f64(
        directory / ("oracle__block" + std::to_string(block_index) +
                     "__block_output.bin"),
        T * D);
    return fixture;
}

std::array<Ct, T> encrypt_all_token_states(
    Cc cc, Keys& keys, const std::vector<double>& flat) {
    if (flat.size() != T * D) {
        throw std::invalid_argument(
            "token-state encryption requires exactly T*D values");
    }
    std::array<Ct, T> encrypted{};
    for (std::size_t token = 0; token < T; ++token) {
        std::vector<double> block(PACK_WIDTH, 0.0);
        const auto begin =
            flat.begin() + static_cast<std::ptrdiff_t>(token * D);
        std::copy(begin, begin + static_cast<std::ptrdiff_t>(D),
                  block.begin());
        std::vector<double> packed;
        packed.reserve(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            packed.insert(packed.end(), block.begin(), block.end());
        }
        Plaintext plaintext =
            cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
        encrypted[token] = cc->Encrypt(keys.publicKey, plaintext);
        if (encrypted[token]->GetLevel() != 0) {
            throw std::runtime_error(
                "fresh token-state encryption did not return level 0");
        }
    }
    return encrypted;
}

struct FullHiddenRefresh {
    std::array<Ct, T> refreshed{};
    Metrics metrics;
    double seconds = 0.0;
    std::size_t input_level = 0;
};

std::vector<double> decrypt_packed_hidden(Cc cc, Keys& keys,
                                          const Ct& packed_output) {
    Ct local = packed_output;
    Plaintext decoded;
    cc->Decrypt(keys.secretKey, local, &decoded);
    decoded->SetLength(SLOTS);
    const std::vector<double> raw = decoded->GetRealPackedValue();
    if (raw.size() < T * PACK_WIDTH) {
        throw std::runtime_error(
            "hidden-state boundary returned too few packed slots");
    }
    return raw;
}

FullHiddenRefresh refresh_all_hidden(
    Cc cc, Keys& keys, const Ct& packed_output,
    const std::vector<double>& oracle) {
    const auto start = Clock::now();
    FullHiddenRefresh result;
    result.input_level = packed_output->GetLevel();
    const std::vector<double> raw =
        decrypt_packed_hidden(cc, keys, packed_output);
    const std::vector<double> packed_measurement(
        raw.begin(),
        raw.begin() + static_cast<std::ptrdiff_t>(T * PACK_WIDTH));
    result.metrics = measure(packed_measurement, oracle);
    if (!result.metrics.all_finite) {
        throw std::runtime_error(
            "non-finite hidden state at inter-block client refresh");
    }

    std::vector<double> flat(T * D);
    for (std::size_t token = 0; token < T; ++token) {
        for (std::size_t dim = 0; dim < D; ++dim) {
            flat[token * D + dim] =
                raw[token * PACK_WIDTH + dim];
        }
    }
    result.refreshed = encrypt_all_token_states(cc, keys, flat);
    cc->Synchronize();
    result.seconds = elapsed_seconds(start);
    return result;
}

struct LastTokenRefresh {
    Ct refreshed;
    Metrics metrics;
    double seconds = 0.0;
    std::size_t input_level = 0;
};

LastTokenRefresh refresh_last_token_for_head(
    Cc cc, Keys& keys, const Ct& packed_output,
    const std::vector<double>& oracle) {
    const auto start = Clock::now();
    LastTokenRefresh result;
    result.input_level = packed_output->GetLevel();
    const std::vector<double> raw =
        decrypt_packed_hidden(cc, keys, packed_output);
    const std::vector<double> packed_measurement(
        raw.begin(),
        raw.begin() + static_cast<std::ptrdiff_t>(T * PACK_WIDTH));
    result.metrics = measure(packed_measurement, oracle);
    if (!result.metrics.all_finite) {
        throw std::runtime_error(
            "non-finite block-11 state at head-input client refresh");
    }

    std::vector<double> flat(D);
    for (std::size_t dim = 0; dim < D; ++dim) {
        flat[dim] = raw[(T - 1) * PACK_WIDTH + dim];
    }
    std::vector<double> block(PACK_WIDTH, 0.0);
    std::copy(flat.begin(), flat.end(), block.begin());
    std::vector<double> packed;
    packed.reserve(SLOTS);
    for (std::size_t copy = 0; copy < COPIES; ++copy) {
        packed.insert(packed.end(), block.begin(), block.end());
    }
    Plaintext plaintext =
        cc->MakeCKKSPackedPlaintext(packed, 1, 0, nullptr, SLOTS);
    result.refreshed = cc->Encrypt(keys.publicKey, plaintext);
    cc->Synchronize();
    if (result.refreshed->GetLevel() != 0) {
        throw std::runtime_error(
            "head-input refresh did not return level 0");
    }
    result.seconds = elapsed_seconds(start);
    return result;
}

struct HeadFixture {
    std::vector<double> final_ln;
    Matrix head_linear;
    std::vector<double> head_ln;
    std::vector<double> margin_n_minus_a;
    std::vector<double> oracle_head_ln;
};

HeadFixture load_head_fixture(const std::filesystem::path& directory) {
    HeadFixture fixture;
    fixture.final_ln =
        read_f64(directory / "weights__final_ln.bin", D);
    fixture.head_linear =
        load_matrix(directory / "weights__head_linear.bin", D, D);
    fixture.head_ln =
        read_f64(directory / "weights__head_ln.bin", D);
    fixture.margin_n_minus_a = read_f64(
        directory / "weights__head_readout_margin_n_minus_a.bin", D);
    fixture.oracle_head_ln = read_f64(
        directory / "oracle__head__head_ln_output.bin", T * D);
    return fixture;
}

double stable_exact_silu(double value) {
    if (value >= 0.0) {
        return value / (1.0 + std::exp(-value));
    }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

struct HeadEvaluation {
    Ct encrypted_margin;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels;
    std::size_t matrix_products = 0;
    std::size_t ciphertext_ciphertext_multiplications = 0;
    std::size_t ciphertext_plaintext_multiplications = 0;
};

class EncryptedHeadEvaluator {
  public:
    EncryptedHeadEvaluator(Cc context, const HeadFixture& fixture,
                           Client& client)
        : cc_(std::move(context)), fixture_(fixture), client_(client) {}

    HeadEvaluation evaluate(const Ct& last_token) {
        levels_.clear();
        matmul_count_ = 0;
        ct_ct_multiply_count_ = 0;
        ct_plain_multiply_count_ = 0;

        Ct final_ln =
            layernorm(last_token, fixture_.final_ln, "head_final_ln");
        record("final_ln", final_ln);

        Ct head_linear =
            matmul(baby_rotations(final_ln), fixture_.head_linear);
        record("head_linear", head_linear);

        Ct activated = client_.cross_boundary_active(
            "head_silu", head_linear, stable_exact_silu, 1);
        record("head_silu", activated);

        Ct head_ln =
            layernorm(activated, fixture_.head_ln, "head_ln");
        record("head_ln", head_ln);

        Ct weighted = multiply_plain(
            head_ln, repeated_plain(fixture_.margin_n_minus_a));
        Ct margin = sum_broadcast(weighted);
        record("margin_n_minus_a", margin);
        return {
            .encrypted_margin = margin,
            .levels = levels_,
            .matrix_products = matmul_count_,
            .ciphertext_ciphertext_multiplications =
                ct_ct_multiply_count_,
            .ciphertext_plaintext_multiplications =
                ct_plain_multiply_count_,
        };
    }

  private:
    struct BabyRotations {
        std::array<Ct, BSGS_N1> values{};
    };

    Plaintext raw_plain(const std::vector<double>& values) {
        if (values.size() != SLOTS) {
            throw std::invalid_argument(
                "head plaintext must have exactly SLOTS values");
        }
        return cc_->MakeCKKSPackedPlaintext(
            values, 1, 0, nullptr, SLOTS);
    }

    Plaintext repeated_plain(const std::vector<double>& values) {
        if (values.size() != D) {
            throw std::invalid_argument(
                "head repeated plaintext must have D values");
        }
        std::vector<double> packed(SLOTS);
        for (std::size_t copy = 0; copy < COPIES; ++copy) {
            std::copy(values.begin(), values.end(),
                      packed.begin() +
                          static_cast<std::ptrdiff_t>(copy * PACK_WIDTH));
        }
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

    Ct sum_broadcast(const Ct& input) {
        return cc_->AccumulateSum(
            input, static_cast<int>(PACK_WIDTH), 1);
    }

    Ct mean_broadcast(const Ct& input) {
        Ct sum = sum_broadcast(input);
        return multiply_plain(
            sum, repeated_plain(std::vector<double>(
                     D, 1.0 / static_cast<double>(D))));
    }

    Ct layernorm(const Ct& input, const std::vector<double>& weight,
                 const std::string& name) {
        Ct mean = mean_broadcast(input);
        Ct centered = cc_->EvalSub(input, mean);
        Ct variance = mean_broadcast(multiply(centered, centered));
        variance = cc_->EvalAdd(variance, EPS);
        Ct inverse = client_.cross_boundary(
            name + "_invsqrt", variance, exact_invsqrt, 1);
        Ct normalized = multiply(centered, inverse);
        return multiply_plain(normalized, repeated_plain(weight));
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
            throw std::runtime_error(
                "head evaluator received incomplete baby rotations");
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
            throw std::invalid_argument(
                "head linear matrix must be D by D");
        }
        Ct result;
        for (std::size_t giant = 0; giant < BSGS_N2; ++giant) {
            Ct inner;
            for (std::size_t small = 0; small < BSGS_N1; ++small) {
                const std::size_t diagonal =
                    BSGS_N1 * giant + small;
                std::vector<double> packed_values(SLOTS);
                for (std::size_t copy = 0; copy < COPIES; ++copy) {
                    const std::size_t copy_start =
                        copy * PACK_WIDTH;
                    for (std::size_t row = 0;
                         row < PACK_WIDTH; ++row) {
                        const std::size_t rolled_row =
                            (row + PACK_WIDTH -
                             BSGS_N1 * giant) %
                            PACK_WIDTH;
                        const std::size_t column =
                            (rolled_row + diagonal) % PACK_WIDTH;
                        if (rolled_row < D && column < D) {
                            packed_values[copy_start + row] =
                                weight(rolled_row, column);
                        }
                    }
                }
                Ct term = multiply_plain(
                    baby.values[small],
                    raw_plain(packed_values));
                inner = small == 0 ? term
                                   : cc_->EvalAdd(inner, term);
            }
            if (giant != 0) {
                inner = cc_->EvalRotate(
                    inner,
                    static_cast<std::int32_t>(
                        BSGS_N1 * giant));
            }
            result = giant == 0
                         ? inner
                         : cc_->EvalAdd(result, inner);
        }
        ++matmul_count_;
        return result;
    }

    void record(const std::string& name, const Ct& ciphertext) {
        const std::size_t level = ciphertext->GetLevel();
        levels_[name] = {level, level};
        std::cout << "[head] " << name << " level=" << level << '\n';
    }

    Cc cc_;
    const HeadFixture& fixture_;
    Client& client_;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels_;
    std::size_t matmul_count_ = 0;
    std::size_t ct_ct_multiply_count_ = 0;
    std::size_t ct_plain_multiply_count_ = 0;
};

struct BlockSummary {
    std::size_t block = 0;
    Metrics metrics;
    std::map<std::string, std::pair<std::size_t, std::size_t>> levels;
    std::size_t packed_level = 0;
    std::size_t matrix_products = 0;
    std::size_t ciphertext_ciphertext_multiplications = 0;
    std::size_t ciphertext_plaintext_multiplications = 0;
    double evaluation_seconds = 0.0;
    double refresh_seconds = 0.0;
};

void append_metrics(std::ostringstream& out, const Metrics& metrics,
                    std::string_view indent) {
    out << indent << "\"global_rel_inf\": "
        << metrics.global_rel_inf << ",\n"
        << indent << "\"worst_token_rel_inf\": "
        << metrics.worst_token_rel_inf << ",\n"
        << indent << "\"max_abs_error\": "
        << metrics.max_abs_error << ",\n"
        << indent << "\"all_finite\": "
        << (metrics.all_finite ? "true" : "false") << ",\n"
        << indent << "\"passed\": "
        << (metrics.passed ? "true" : "false");
}

double direct_dot_oracle_margin(const HeadFixture& fixture) {
    double margin = 0.0;
    const std::size_t last_token = T - 1;
    for (std::size_t dim = 0; dim < D; ++dim) {
        margin +=
            fixture.oracle_head_ln[last_token * D + dim] *
            fixture.margin_n_minus_a[dim];
    }
    return margin;
}

std::string make_all_blocks_json(
    const AllBlocksOptions& options, std::uint32_t ring,
    std::size_t rotation_keys,
    const std::vector<BlockSummary>& blocks,
    const HeadEvaluation& head, double decrypted_margin,
    double oracle_margin, double fixture_seconds,
    double setup_seconds, double encryption_seconds,
    double blocks_seconds, double head_seconds,
    double refresh_seconds, double final_decrypt_seconds,
    std::size_t client_round_trips,
    std::size_t client_logical_instances,
    double client_nonlinearity_seconds, bool passed) {
    std::size_t block_matrix_products = 0;
    std::size_t block_ct_ct = 0;
    std::size_t block_ct_pt = 0;
    for (const BlockSummary& block : blocks) {
        block_matrix_products += block.matrix_products;
        block_ct_ct +=
            block.ciphertext_ciphertext_multiplications;
        block_ct_pt +=
            block.ciphertext_plaintext_multiplications;
    }
    const double margin_abs_error =
        std::abs(decrypted_margin - oracle_margin);
    const double margin_rel_error =
        margin_abs_error / (std::abs(oracle_margin) + 1e-15);
    const bool label_matches =
        std::signbit(decrypted_margin) ==
        std::signbit(oracle_margin);
    const std::size_t declared_crossings =
        client_round_trips + EXPECTED_INTER_BLOCK_REFRESHES +
        EXPECTED_HEAD_INPUT_REFRESHES;
    const std::size_t declared_logical_instances =
        client_logical_instances +
        EXPECTED_INTER_BLOCK_REFRESHES * T +
        EXPECTED_HEAD_INPUT_REFRESHES;
    const double client_total =
        client_nonlinearity_seconds + refresh_seconds;
    const double encrypted_total =
        blocks_seconds + head_seconds + refresh_seconds;

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"Released-weight DNAGPT complete 12-block "
           "backbone plus encrypted GSR N-minus-A head, Scheme B T=2\",\n"
        << "  \"implementation_version\": "
           "\"t2-scheme-b-all-12-blocks-head-v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS)\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"gate\": \"all_blocks_head_t2\",\n"
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
        << "  \"frozen_single_block_source_sha256\": \""
        << json_escape(options.frozen_source_sha256) << "\",\n"
        << "  \"fixture_manifest_sha256\": \""
        << json_escape(options.fixture_manifest_sha256) << "\",\n"
        << "  \"fixture_contract_sha256\": \""
        << json_escape(options.fixture_contract_sha256) << "\",\n"
        << "  \"config\": {\n"
        << "    \"model\": \"dna_gpt0.1b_m classification "
           "checkpoint, all 12 released blocks and GSR head\",\n"
        << "    \"blocks\": 12,\n"
        << "    \"D\": " << D << ",\n"
        << "    \"T\": " << T << ",\n"
        << "    \"batch_slots\": " << SLOTS << ",\n"
        << "    \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": "
        << MULT_DEPTH << ",\n"
        << "    \"rotation_keys\": " << rotation_keys << ",\n"
        << "    \"bootstraps\": 0,\n"
        << "    \"block_matrix_products\": "
        << block_matrix_products << ",\n"
        << "    \"head_matrix_products\": "
        << head.matrix_products << ",\n"
        << "    \"operation_counts\": {\n"
        << "      \"explicit_ciphertext_ciphertext_multiplications\": "
        << (block_ct_ct +
            head.ciphertext_ciphertext_multiplications)
        << ",\n"
        << "      \"explicit_ciphertext_plaintext_multiplications\": "
        << (block_ct_pt +
            head.ciphertext_plaintext_multiplications)
        << "\n"
        << "    }\n"
        << "  },\n"
        << "  \"composition\": {\n"
        << "    \"same_crypto_context_and_key_lineage\": true,\n"
        << "    \"inter_block_full_hidden_refreshes\": "
        << EXPECTED_INTER_BLOCK_REFRESHES << ",\n"
        << "    \"block11_to_head_last_token_refreshes\": "
        << EXPECTED_HEAD_INPUT_REFRESHES << ",\n"
        << "    \"fresh_input_level\": 0\n"
        << "  },\n"
        << "  \"blocks\": [\n";
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const BlockSummary& block = blocks[index];
        out << "    {\n"
            << "      \"block\": " << block.block << ",\n"
            << "      \"packed_output_level\": "
            << block.packed_level << ",\n"
            << "      \"metrics\": {\n";
        append_metrics(out, block.metrics, "        ");
        out << "\n      },\n"
            << "      \"level_trace\": {\n";
        std::size_t level_index = 0;
        for (const auto& [name, range] : block.levels) {
            out << "        \"" << json_escape(name)
                << "\": {\"min\": " << range.first
                << ", \"max\": " << range.second << "}";
            out << (++level_index == block.levels.size()
                        ? "\n"
                        : ",\n");
        }
        out << "      },\n"
            << "      \"evaluation_seconds\": "
            << block.evaluation_seconds << ",\n"
            << "      \"refresh_seconds\": "
            << block.refresh_seconds << "\n"
            << "    }"
            << (index + 1 == blocks.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"head\": {\n"
        << "    \"encrypted_readout\": \"last token -> final LN -> "
           "head linear -> SiLU -> head LN -> dot(N-A)\",\n"
        << "    \"decrypted_margin_n_minus_a\": "
        << decrypted_margin << ",\n"
        << "    \"direct_dot_oracle_margin_n_minus_a\": "
        << oracle_margin << ",\n"
        << "    \"margin_abs_error\": " << margin_abs_error
        << ",\n"
        << "    \"margin_rel_error\": " << margin_rel_error
        << ",\n"
        << "    \"label\": \""
        << (decrypted_margin >= 0.0 ? "N" : "A")
        << "\",\n"
        << "    \"label_matches\": "
        << (label_matches ? "true" : "false") << ",\n"
        << "    \"level_trace\": {\n";
    std::size_t head_level_index = 0;
    for (const auto& [name, range] : head.levels) {
        out << "      \"" << json_escape(name)
            << "\": {\"min\": " << range.first
            << ", \"max\": " << range.second << "}";
        out << (++head_level_index == head.levels.size()
                    ? "\n"
                    : ",\n");
    }
    out << "    }\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"round_trips\": " << declared_crossings
        << ",\n"
        << "    \"nonlinearity_round_trips\": "
        << client_round_trips << ",\n"
        << "    \"full_hidden_state_refresh_round_trips\": "
        << EXPECTED_INTER_BLOCK_REFRESHES << ",\n"
        << "    \"last_token_refresh_round_trips\": "
        << EXPECTED_HEAD_INPUT_REFRESHES << ",\n"
        << "    \"logical_boundary_instances\": "
        << declared_logical_instances << ",\n"
        << "    \"boundary_seconds_total\": "
        << client_total << "\n"
        << "  },\n"
        << "  \"timings_seconds\": {\n"
        << "    \"load_verified_fixture\": "
        << fixture_seconds << ",\n"
        << "    \"context_keygen_load\": "
        << setup_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": "
        << encryption_seconds << ",\n"
        << "    \"blocks_encrypted_evaluation\": "
        << blocks_seconds << ",\n"
        << "    \"head_encrypted_evaluation\": "
        << head_seconds << ",\n"
        << "    \"refreshes\": " << refresh_seconds
        << ",\n"
        << "    \"encrypted_evaluation\": "
        << encrypted_total << ",\n"
        << "    \"client_boundary_seconds_total\": "
        << client_total << ",\n"
        << "    \"server_linear_algebra_seconds\": "
        << (encrypted_total - client_total) << ",\n"
        << "    \"final_decrypt\": "
        << final_decrypt_seconds << "\n"
        << "  },\n"
        << "  \"all_finite\": "
        << (std::isfinite(decrypted_margin) ? "true" : "false")
        << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false")
        << ",\n"
        << "  \"input_boundary\": \"token plus position "
           "embeddings encrypted by client\",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"declared_client_boundary_crossings\": "
        << declared_crossings << ",\n"
        << "  \"declared_logical_boundary_instances\": "
        << declared_logical_instances << ",\n"
        << "  \"final_decrypt_calls\": 1,\n"
        << "  \"evaluator_has_private_key\": false\n"
        << "}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        std::cerr << std::unitbuf;
        const AllBlocksOptions options =
            parse_all_blocks_options(argc, argv);

        const auto fixture_start = Clock::now();
        const std::vector<double> initial_input =
            read_f64(options.fixture_dir / "input__embeddings.bin",
                     T * D);
        const HeadFixture head_fixture =
            load_head_fixture(options.fixture_dir);
        double fixture_seconds =
            elapsed_seconds(fixture_start);

        const std::vector<int> rotation_keys =
            required_rotation_keys("full");
        const auto setup_start = Clock::now();
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecurityLevel(
            SecurityLevel::HEStd_128_classic);
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
        const double setup_seconds =
            elapsed_seconds(setup_start);
        const std::uint32_t ring =
            cc->GetRingDimension();

        const auto encryption_start = Clock::now();
        std::array<Ct, T> encrypted_inputs =
            encrypt_all_token_states(cc, keys, initial_input);
        cc->Synchronize();
        const double encryption_seconds =
            elapsed_seconds(encryption_start);

        Client client(cc, keys);
        std::vector<BlockSummary> summaries;
        summaries.reserve(ALL_BLOCKS);
        double blocks_seconds = 0.0;
        double refresh_seconds = 0.0;
        bool all_blocks_passed = true;
        Ct head_input;

        for (std::size_t block = 0; block < ALL_BLOCKS; ++block) {
            const auto block_fixture_start = Clock::now();
            const Fixture fixture =
                load_all_blocks_fixture(
                    options.fixture_dir, block);
            fixture_seconds +=
                elapsed_seconds(block_fixture_start);
            const std::size_t before_round_trips =
                client.round_trips();
            const std::size_t before_logical =
                client.logical_boundary_instances();

            const auto block_start = Clock::now();
            EncryptedEvaluator evaluator(cc, fixture, client);
            EvaluationResult evaluation =
                evaluator.evaluate(encrypted_inputs, "full");
            cc->Synchronize();
            const double block_seconds =
                elapsed_seconds(block_start);
            blocks_seconds += block_seconds;

            if (client.round_trips() - before_round_trips !=
                    EXPECTED_BLOCK_ROUND_TRIPS ||
                client.logical_boundary_instances() -
                        before_logical !=
                    EXPECTED_BLOCK_LOGICAL_INSTANCES) {
                throw std::runtime_error(
                    "block " + std::to_string(block) +
                    " crossed an unexpected Scheme B boundary count");
            }

            BlockSummary summary;
            summary.block = block;
            summary.levels = evaluation.levels;
            summary.packed_level = evaluation.packed_level;
            summary.matrix_products =
                evaluation.ciphertext_plain_matmuls;
            summary.ciphertext_ciphertext_multiplications =
                evaluation.ciphertext_ciphertext_multiplications;
            summary.ciphertext_plaintext_multiplications =
                evaluation.ciphertext_plaintext_multiplications;
            summary.evaluation_seconds = block_seconds;

            if (block + 1 < ALL_BLOCKS) {
                FullHiddenRefresh refresh =
                    refresh_all_hidden(
                        cc, keys, evaluation.packed_output,
                        fixture.oracle_block_output);
                summary.metrics = refresh.metrics;
                summary.refresh_seconds = refresh.seconds;
                refresh_seconds += refresh.seconds;
                all_blocks_passed =
                    all_blocks_passed &&
                    refresh.metrics.passed;
                evaluation.packed_output.reset();
                for (Ct& input : encrypted_inputs) {
                    input.reset();
                }
                encrypted_inputs =
                    std::move(refresh.refreshed);
            } else {
                LastTokenRefresh refresh =
                    refresh_last_token_for_head(
                        cc, keys, evaluation.packed_output,
                        fixture.oracle_block_output);
                summary.metrics = refresh.metrics;
                summary.refresh_seconds = refresh.seconds;
                refresh_seconds += refresh.seconds;
                all_blocks_passed =
                    all_blocks_passed &&
                    refresh.metrics.passed;
                head_input = std::move(refresh.refreshed);
                evaluation.packed_output.reset();
                for (Ct& input : encrypted_inputs) {
                    input.reset();
                }
            }
            summaries.push_back(std::move(summary));
        }

        const std::size_t before_head_round_trips =
            client.round_trips();
        const std::size_t before_head_logical =
            client.logical_boundary_instances();
        const auto head_start = Clock::now();
        EncryptedHeadEvaluator head_evaluator(
            cc, head_fixture, client);
        HeadEvaluation head =
            head_evaluator.evaluate(head_input);
        cc->Synchronize();
        const double head_seconds =
            elapsed_seconds(head_start);
        if (client.round_trips() -
                    before_head_round_trips !=
                EXPECTED_HEAD_ROUND_TRIPS ||
            client.logical_boundary_instances() -
                    before_head_logical !=
                EXPECTED_HEAD_ROUND_TRIPS) {
            throw std::runtime_error(
                "classifier head crossed an unexpected Scheme B "
                "boundary count");
        }

        const auto decrypt_start = Clock::now();
        Plaintext decoded_margin;
        cc->Decrypt(
            keys.secretKey, head.encrypted_margin,
            &decoded_margin);
        decoded_margin->SetLength(SLOTS);
        const std::vector<double> margin_values =
            decoded_margin->GetRealPackedValue();
        if (margin_values.empty()) {
            throw std::runtime_error(
                "final margin decrypt returned no slots");
        }
        const double decrypted_margin =
            margin_values.front();
        const double final_decrypt_seconds =
            elapsed_seconds(decrypt_start);

        const double oracle_margin =
            direct_dot_oracle_margin(head_fixture);
        const double margin_rel_error =
            std::abs(decrypted_margin - oracle_margin) /
            (std::abs(oracle_margin) + 1e-15);
        const bool label_matches =
            std::signbit(decrypted_margin) ==
            std::signbit(oracle_margin);
        const std::size_t declared_crossings =
            client.round_trips() +
            EXPECTED_INTER_BLOCK_REFRESHES +
            EXPECTED_HEAD_INPUT_REFRESHES;
        const std::size_t declared_logical_instances =
            client.logical_boundary_instances() +
            EXPECTED_INTER_BLOCK_REFRESHES * T +
            EXPECTED_HEAD_INPUT_REFRESHES;
        if (declared_crossings != EXPECTED_TOTAL_CROSSINGS ||
            declared_logical_instances !=
                EXPECTED_TOTAL_LOGICAL_INSTANCES) {
            throw std::runtime_error(
                "complete graph crossed an unexpected total "
                "Scheme B boundary count");
        }
        const bool passed =
            all_blocks_passed &&
            std::isfinite(decrypted_margin) &&
            margin_rel_error <= TOL && label_matches;

        const std::string evidence =
            make_all_blocks_json(
                options, ring, rotation_keys.size(), summaries,
                head, decrypted_margin, oracle_margin,
                fixture_seconds, setup_seconds,
                encryption_seconds, blocks_seconds,
                head_seconds, refresh_seconds,
                final_decrypt_seconds, client.round_trips(),
                client.logical_boundary_instances(),
                client.boundary_seconds_total(), passed);
        write_exclusive(options.output, evidence);
        std::cout << evidence;
        std::cout
            << (passed
                    ? "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T2_PASS"
                    : "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T2_FAIL")
            << '\n';
        return passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
