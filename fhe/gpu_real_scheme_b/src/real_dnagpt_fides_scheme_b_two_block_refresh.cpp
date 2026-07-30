// Released-weight DNAGPT blocks 0 -> 1 under Scheme B (hybrid
// client-assisted CKKS), T=2 correctness gate.
//
// Fork discipline:
//   - the frozen batched single-block implementation below remains untouched;
//   - this translation unit includes that implementation at its pinned SHA-256
//     and renames only its standalone main();
//   - both released blocks run through the exact same EncryptedEvaluator;
//   - the only new protocol operation is one pre-declared client boundary
//     between blocks. The client decrypts block 0's packed output, measures the
//     already-authorized plaintext against the block-0 oracle, unpacks the two
//     token states, and re-encrypts each token as a fresh level-0 ciphertext.
//
// This is deliberately a two-block protocol/correctness prototype, not a
// speed optimization and not evidence until a results/runs JSON lands.

#define main real_dnagpt_fides_scheme_b_single_block_main_unused
#include "real_dnagpt_fides_scheme_b.cpp"
#undef main

namespace {

constexpr std::string_view TWO_BLOCK_FIXTURE_MANIFEST =
    "3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c";
constexpr std::string_view FROZEN_SCHEME_B_SOURCE_SHA256 =
    "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df";
constexpr std::size_t EXPECTED_ROUND_TRIPS_PER_BLOCK = 7;
constexpr std::size_t EXPECTED_LOGICAL_BOUNDARIES_PER_BLOCK = 24;

struct TwoBlockOptions {
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

[[noreturn]] void two_block_usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: real_dnagpt_fides_scheme_b_two_block_refresh --gpu N "
        "--fixture-dir PATH --output PATH --fixture-manifest-sha256 SHA "
        "--fixture-contract-sha256 SHA --source-sha256 SHA "
        "--frozen-source-sha256 SHA [--backend-commit SHA] "
        "[--container-image NAME] [--environment TEXT]");
}

TwoBlockOptions parse_two_block_options(int argc, char** argv) {
    TwoBlockOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                two_block_usage_error("missing value for " + arg);
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: real_dnagpt_fides_scheme_b_two_block_refresh "
                   "--gpu N --fixture-dir PATH --output PATH "
                   "--fixture-manifest-sha256 SHA "
                   "--fixture-contract-sha256 SHA --source-sha256 SHA "
                   "--frozen-source-sha256 SHA\n";
            std::exit(0);
        } else {
            two_block_usage_error("unknown option: " + arg);
        }
    }
    if (options.gpu < 0) {
        two_block_usage_error("--gpu must be non-negative");
    }
    if (options.fixture_dir.empty() || options.output.empty()) {
        two_block_usage_error("--fixture-dir and --output are required");
    }
    if (options.backend_commit != PINNED_FIDES_COMMIT) {
        two_block_usage_error("refusing unpinned FIDESlib commit " +
                              options.backend_commit);
    }
    if (options.fixture_manifest_sha256 != TWO_BLOCK_FIXTURE_MANIFEST) {
        two_block_usage_error("refusing unpinned two-block fixture manifest " +
                              options.fixture_manifest_sha256);
    }
    if (options.frozen_source_sha256 != FROZEN_SCHEME_B_SOURCE_SHA256) {
        two_block_usage_error("refusing changed frozen Scheme B source " +
                              options.frozen_source_sha256);
    }
    if (options.source_sha256 == "UNSPECIFIED" ||
        options.fixture_contract_sha256 == "UNSPECIFIED") {
        two_block_usage_error(
            "source and fixture contract identities are required");
    }
    if (std::filesystem::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite immutable evidence: " +
                                 options.output.string());
    }
    return options;
}

Fixture load_multiblock_fixture(const std::filesystem::path& directory,
                                std::size_t block_index) {
    if (block_index > 1) {
        throw std::invalid_argument(
            "two-block Scheme B fixture index must be 0 or 1");
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

std::array<Ct, T> encrypt_token_states(Cc cc, Keys& keys,
                                       const std::vector<double>& flat) {
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

struct HiddenRefreshResult {
    std::array<Ct, T> refreshed{};
    Metrics plaintext_metrics;
    double seconds = 0.0;
    std::size_t input_level = 0;
};

HiddenRefreshResult refresh_full_hidden_state(
    Cc cc, Keys& keys, const Ct& packed_block_output,
    const std::vector<double>& block0_oracle) {
    const auto start = Clock::now();
    HiddenRefreshResult result;
    result.input_level = packed_block_output->GetLevel();

    Ct local = packed_block_output;
    Plaintext decoded;
    cc->Decrypt(keys.secretKey, local, &decoded);
    decoded->SetLength(SLOTS);
    const std::vector<double> raw = decoded->GetRealPackedValue();
    if (raw.size() < T * PACK_WIDTH) {
        throw std::runtime_error(
            "block-boundary decrypt returned too few packed slots");
    }

    const std::vector<double> packed_measurement(
        raw.begin(),
        raw.begin() + static_cast<std::ptrdiff_t>(T * PACK_WIDTH));
    result.plaintext_metrics =
        measure(packed_measurement, block0_oracle);
    if (!result.plaintext_metrics.all_finite) {
        throw std::runtime_error(
            "non-finite block-0 hidden state at client refresh boundary");
    }

    std::vector<double> flat(T * D);
    for (std::size_t token = 0; token < T; ++token) {
        for (std::size_t dim = 0; dim < D; ++dim) {
            flat[token * D + dim] = raw[token * PACK_WIDTH + dim];
        }
    }
    result.refreshed = encrypt_token_states(cc, keys, flat);
    cc->Synchronize();
    result.seconds = elapsed_seconds(start);
    return result;
}

void append_metrics_json(std::ostringstream& out, const Metrics& metrics,
                         std::string_view indent) {
    out << indent << "\"global_rel_inf\": " << metrics.global_rel_inf
        << ",\n"
        << indent << "\"worst_token_rel_inf\": "
        << metrics.worst_token_rel_inf << ",\n"
        << indent << "\"max_abs_error\": " << metrics.max_abs_error << ",\n"
        << indent << "\"all_finite\": "
        << (metrics.all_finite ? "true" : "false") << ",\n"
        << indent << "\"passed\": "
        << (metrics.passed ? "true" : "false");
}

void append_level_trace_json(std::ostringstream& out,
                             const EvaluationResult& evaluation,
                             std::string_view indent) {
    std::size_t index = 0;
    for (const auto& [name, range] : evaluation.levels) {
        out << indent << "\"" << json_escape(name)
            << "\": {\"min\": " << range.first
            << ", \"max\": " << range.second << "}";
        out << (++index == evaluation.levels.size() ? "\n" : ",\n");
    }
}

std::string make_two_block_json(
    const TwoBlockOptions& options, std::uint32_t ring,
    std::size_t rotation_key_count, const EvaluationResult& block0,
    const EvaluationResult& block1, const Metrics& block0_metrics,
    const Metrics& block1_metrics, double fixture_seconds,
    double setup_seconds, double encryption_seconds,
    double block0_evaluation_seconds, double refresh_seconds,
    double block1_evaluation_seconds, double final_decrypt_seconds,
    std::size_t nonlinear_round_trips,
    std::size_t nonlinear_logical_instances,
    double nonlinear_boundary_seconds, std::size_t block0_round_trips,
    std::size_t block0_logical_instances,
    std::size_t refresh_input_level) {
    const bool passed = block0_metrics.passed && block1_metrics.passed;
    const std::size_t total_round_trips = nonlinear_round_trips + 1;
    const std::size_t total_logical_instances =
        nonlinear_logical_instances + T;
    const double client_seconds =
        nonlinear_boundary_seconds + refresh_seconds;
    const double server_seconds =
        block0_evaluation_seconds + block1_evaluation_seconds -
        nonlinear_boundary_seconds;
    const double encrypted_evaluation_seconds =
        block0_evaluation_seconds + refresh_seconds +
        block1_evaluation_seconds;

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"task\": \"Released-weight DNAGPT blocks 0->1, Scheme B "
           "full-hidden-state client refresh correctness gate\",\n"
        << "  \"implementation_version\": "
           "\"t2-scheme-b-two-block-full-hidden-refresh-v1\",\n"
        << "  \"scheme\": \"B (hybrid client-assisted CKKS)\",\n"
        << "  \"measured_at_utc\": \"" << utc_now() << "\",\n"
        << "  \"gate\": \"two_block_full\",\n"
        << "  \"backend\": \"FIDESlib CKKS/CUDA\",\n"
        << "  \"backend_commit\": \""
        << json_escape(options.backend_commit) << "\",\n"
        << "  \"container_image\": \""
        << json_escape(options.container_image) << "\",\n"
        << "  \"environment\": \"" << json_escape(options.environment)
        << "\",\n"
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
        << "    \"model\": \"dna_gpt0.1b_m classification checkpoint "
           "blocks 0 and 1\",\n"
        << "    \"blocks\": [0, 1],\n"
        << "    \"D\": " << D << ",\n"
        << "    \"T\": " << T << ",\n"
        << "    \"heads\": " << HEADS << ",\n"
        << "    \"batch_slots\": " << SLOTS << ",\n"
        << "    \"ring_dim\": " << ring << ",\n"
        << "    \"multiplicative_depth\": " << MULT_DEPTH << ",\n"
        << "    \"scale_bits\": " << SCALE_BITS << ",\n"
        << "    \"bootstraps\": 0,\n"
        << "    \"rotation_keys\": " << rotation_key_count << ",\n"
        << "    \"matrix_products\": "
        << block0.ciphertext_plain_matmuls +
               block1.ciphertext_plain_matmuls
        << ",\n"
        << "    \"operation_counts\": {\n"
        << "      \"ciphertext_plaintext_matrix_products\": "
        << block0.ciphertext_plain_matmuls +
               block1.ciphertext_plain_matmuls
        << ",\n"
        << "      \"explicit_ciphertext_ciphertext_multiplications\": "
        << block0.ciphertext_ciphertext_multiplications +
               block1.ciphertext_ciphertext_multiplications
        << ",\n"
        << "      \"explicit_ciphertext_plaintext_multiplications\": "
        << block0.ciphertext_plaintext_multiplications +
               block1.ciphertext_plaintext_multiplications
        << "\n"
        << "    }\n"
        << "  },\n"
        << "  \"composition\": {\n"
        << "    \"same_crypto_context_and_key_lineage\": true,\n"
        << "    \"block_boundary\": \"client decrypts packed block-0 "
           "hidden state, unpacks T token states, re-encrypts each as a "
           "fresh level-0 replicated ciphertext\",\n"
        << "    \"block_boundary_count\": 1,\n"
        << "    \"block_boundary_decrypt_calls\": 1,\n"
        << "    \"block_boundary_encrypt_calls\": " << T << ",\n"
        << "    \"block_boundary_logical_instances\": " << T << ",\n"
        << "    \"block0_packed_output_level\": " << refresh_input_level
        << ",\n"
        << "    \"block1_input_level\": 0\n"
        << "  },\n"
        << "  \"blocks\": {\n"
        << "    \"block0\": {\n"
        << "      \"packed_output_level\": " << block0.packed_level
        << ",\n"
        << "      \"round_trips\": " << block0_round_trips << ",\n"
        << "      \"logical_boundary_instances\": "
        << block0_logical_instances << ",\n"
        << "      \"metrics_at_refresh_boundary\": {\n";
    append_metrics_json(out, block0_metrics, "        ");
    out << "\n"
        << "      },\n"
        << "      \"level_trace\": {\n";
    append_level_trace_json(out, block0, "        ");
    out << "      }\n"
        << "    },\n"
        << "    \"block1\": {\n"
        << "      \"packed_output_level\": " << block1.packed_level
        << ",\n"
        << "      \"round_trips\": "
        << nonlinear_round_trips - block0_round_trips << ",\n"
        << "      \"logical_boundary_instances\": "
        << nonlinear_logical_instances - block0_logical_instances
        << ",\n"
        << "      \"metrics\": {\n";
    append_metrics_json(out, block1_metrics, "        ");
    out << "\n"
        << "      },\n"
        << "      \"level_trace\": {\n";
    append_level_trace_json(out, block1, "        ");
    out << "      }\n"
        << "    }\n"
        << "  },\n"
        << "  \"global_rel_inf\": " << block1_metrics.global_rel_inf
        << ",\n"
        << "  \"worst_token_rel_inf\": "
        << block1_metrics.worst_token_rel_inf << ",\n"
        << "  \"max_abs_error\": " << block1_metrics.max_abs_error
        << ",\n"
        << "  \"all_finite\": "
        << (block0_metrics.all_finite && block1_metrics.all_finite
                ? "true"
                : "false")
        << ",\n"
        << "  \"tol\": " << TOL << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
        << "  \"protocol\": {\n"
        << "    \"round_trips\": " << total_round_trips << ",\n"
        << "    \"nonlinearity_round_trips\": " << nonlinear_round_trips
        << ",\n"
        << "    \"full_hidden_state_refresh_round_trips\": 1,\n"
        << "    \"logical_boundary_instances\": "
        << total_logical_instances << ",\n"
        << "    \"boundary_seconds_total\": " << client_seconds << "\n"
        << "  },\n"
        << "  \"timings_seconds\": {\n"
        << "    \"load_verified_fixture\": " << fixture_seconds << ",\n"
        << "    \"context_keygen_load\": " << setup_seconds << ",\n"
        << "    \"encrypt_embedded_inputs\": " << encryption_seconds
        << ",\n"
        << "    \"block0_encrypted_evaluation\": "
        << block0_evaluation_seconds << ",\n"
        << "    \"full_hidden_state_refresh\": " << refresh_seconds
        << ",\n"
        << "    \"block1_encrypted_evaluation\": "
        << block1_evaluation_seconds << ",\n"
        << "    \"encrypted_evaluation\": "
        << encrypted_evaluation_seconds << ",\n"
        << "    \"client_boundary_seconds_total\": " << client_seconds
        << ",\n"
        << "    \"server_linear_algebra_seconds\": " << server_seconds
        << ",\n"
        << "    \"final_decrypt\": " << final_decrypt_seconds << "\n"
        << "  },\n"
        << "  \"input_boundary\": \"token plus position embeddings "
           "encrypted by client\",\n"
        << "  \"intermediate_decrypt_attempts\": 0,\n"
        << "  \"declared_client_boundary_crossings\": "
        << total_round_trips << ",\n"
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
        const TwoBlockOptions options = parse_two_block_options(argc, argv);

        const auto fixture_start = Clock::now();
        const Fixture block0_fixture =
            load_multiblock_fixture(options.fixture_dir, 0);
        const Fixture block1_fixture =
            load_multiblock_fixture(options.fixture_dir, 1);
        const double fixture_seconds = elapsed_seconds(fixture_start);

        const std::vector<int> rotation_keys =
            required_rotation_keys("full");
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
        std::cout << "[setup] GenCryptoContext done\n";
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        std::cout
            << "[setup] Enable(PKE|KEYSWITCH|LEVELEDSHE|ADVANCEDSHE) done\n";
        auto keys = cc->KeyGen();
        std::cout << "[setup] KeyGen done\n";
        cc->EvalMultKeyGen(keys.secretKey);
        std::cout << "[setup] EvalMultKeyGen done\n";
        cc->EvalRotateKeyGen(keys.secretKey, rotation_keys);
        std::cout << "[setup] EvalRotateKeyGen done rotation_keys="
                  << rotation_keys.size() << "\n";
        cc->LoadContext(keys.publicKey);
        std::cout << "[setup] LoadContext done\n";
        cc->Synchronize();
        std::cout << "[setup] Synchronize done\n";
        const double setup_seconds = elapsed_seconds(setup_start);
        const std::uint32_t ring = cc->GetRingDimension();

        const auto encryption_start = Clock::now();
        std::array<Ct, T> encrypted_inputs =
            encrypt_token_states(cc, keys, block0_fixture.input);
        cc->Synchronize();
        const double encryption_seconds =
            elapsed_seconds(encryption_start);

        Client client(cc, keys);

        const auto block0_start = Clock::now();
        EncryptedEvaluator block0_evaluator(cc, block0_fixture, client);
        EvaluationResult block0 =
            block0_evaluator.evaluate(encrypted_inputs, "full");
        cc->Synchronize();
        const double block0_seconds = elapsed_seconds(block0_start);
        const std::size_t block0_round_trips = client.round_trips();
        const std::size_t block0_logical_instances =
            client.logical_boundary_instances();
        if (block0_round_trips != EXPECTED_ROUND_TRIPS_PER_BLOCK ||
            block0_logical_instances !=
                EXPECTED_LOGICAL_BOUNDARIES_PER_BLOCK) {
            throw std::runtime_error(
                "block 0 crossed an unexpected Scheme B boundary count");
        }

        HiddenRefreshResult refresh = refresh_full_hidden_state(
            cc, keys, block0.packed_output,
            block0_fixture.oracle_block_output);
        block0.packed_output.reset();
        for (Ct& encrypted_input : encrypted_inputs) {
            encrypted_input.reset();
        }

        const auto block1_start = Clock::now();
        EncryptedEvaluator block1_evaluator(cc, block1_fixture, client);
        EvaluationResult block1 =
            block1_evaluator.evaluate(refresh.refreshed, "full");
        cc->Synchronize();
        const double block1_seconds = elapsed_seconds(block1_start);
        if (client.round_trips() - block0_round_trips !=
                EXPECTED_ROUND_TRIPS_PER_BLOCK ||
            client.logical_boundary_instances() -
                    block0_logical_instances !=
                EXPECTED_LOGICAL_BOUNDARIES_PER_BLOCK) {
            throw std::runtime_error(
                "block 1 crossed an unexpected Scheme B boundary count");
        }

        const auto decrypt_start = Clock::now();
        Plaintext decoded;
        cc->Decrypt(keys.secretKey, block1.packed_output, &decoded);
        decoded->SetLength(SLOTS);
        const std::vector<double> raw = decoded->GetRealPackedValue();
        if (raw.size() < T * PACK_WIDTH) {
            throw std::runtime_error(
                "final decrypt returned too few packed slots");
        }
        const std::vector<double> output(
            raw.begin(),
            raw.begin() + static_cast<std::ptrdiff_t>(T * PACK_WIDTH));
        const double decrypt_seconds = elapsed_seconds(decrypt_start);
        const Metrics block1_metrics =
            measure(output, block1_fixture.oracle_block_output);

        const std::string evidence = make_two_block_json(
            options, ring, rotation_keys.size(), block0, block1,
            refresh.plaintext_metrics, block1_metrics, fixture_seconds,
            setup_seconds, encryption_seconds, block0_seconds,
            refresh.seconds, block1_seconds, decrypt_seconds,
            client.round_trips(), client.logical_boundary_instances(),
            client.boundary_seconds_total(), block0_round_trips,
            block0_logical_instances, refresh.input_level);
        write_exclusive(options.output, evidence);
        std::cout << evidence;

        const bool passed =
            refresh.plaintext_metrics.passed && block1_metrics.passed;
        std::cout
            << (passed
                    ? "REAL_DNAGPT_FIDES_SCHEME_B_TWO_BLOCK_REFRESH_PASS"
                    : "REAL_DNAGPT_FIDES_SCHEME_B_TWO_BLOCK_REFRESH_FAIL")
            << '\n';
        return passed ? 0 : 5;
    } catch (const std::exception& error) {
        std::cerr << "[FATAL] " << error.what() << '\n';
        return 2;
    }
}
