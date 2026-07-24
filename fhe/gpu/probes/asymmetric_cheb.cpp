#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <fideslib.hpp>

using namespace fideslib;

int main() {
    constexpr uint32_t depth = 14;
    constexpr uint32_t slots = 32;
    constexpr double lo = 0.03;
    constexpr double hi = 0.9;
    constexpr size_t degree = 27;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(HEStd_128_classic);
    params.SetMultiplicativeDepth(depth);
    params.SetScalingModSize(50);
    params.SetBatchSize(slots);
    params.SetDevices({0});
    params.SetPlaintextAutoload(false);
    params.SetCiphertextAutoload(true);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->LoadContext(keys.publicKey);

    std::vector<double> x(slots);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = lo + (hi - lo) * static_cast<double>(i) / (x.size() - 1);
    }
    auto plaintext = cc->MakeCKKSPackedPlaintext(x);
    auto ciphertext = cc->Encrypt(keys.publicKey, plaintext);

    std::function<double(double)> invsqrt =
        [](double value) { return 1.0 / std::sqrt(value); };
    auto coefficients =
        cc->GetChebyshevCoefficients(invsqrt, lo, hi, degree);

    const auto started = std::chrono::steady_clock::now();
    auto result =
        cc->EvalChebyshevSeries(ciphertext, coefficients, lo, hi);
    Plaintext decrypted;
    cc->Decrypt(keys.secretKey, result, &decrypted);
    const auto finished = std::chrono::steady_clock::now();
    decrypted->SetLength(slots);

    const auto got = decrypted->GetRealPackedValue();
    double max_absolute_error = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double expected = invsqrt(x[i]);
        max_absolute_error =
            std::max(max_absolute_error, std::abs(got[i] - expected));
        denominator = std::max(denominator, std::abs(expected));
    }
    const double rel_inf = max_absolute_error / denominator;
    const double seconds =
        std::chrono::duration<double>(finished - started).count();
    const bool passed = std::isfinite(rel_inf) && rel_inf <= 4e-2;

    std::cout << std::setprecision(17)
              << "{\"ring_dim\":" << cc->GetRingDimension()
              << ",\"security\":\"HEStd_128_classic\""
              << ",\"domain\":[" << lo << "," << hi << "]"
              << ",\"degree\":" << degree
              << ",\"output_level\":" << result->GetLevel()
              << ",\"max_abs_err\":" << max_absolute_error
              << ",\"rel_inf\":" << rel_inf
              << ",\"seconds\":" << seconds
              << ",\"passed\":" << (passed ? "true" : "false")
              << "}\n";
    return passed ? 0 : 1;
}
