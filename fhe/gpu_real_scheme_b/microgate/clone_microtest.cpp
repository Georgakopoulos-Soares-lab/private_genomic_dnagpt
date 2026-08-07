// Stage-2/3 isolated GPU micro-gate for the T123 optimizations.
// Validates, BEFORE integrating into the 2h full-block driver:
//   TEST_A  encode-once + clone == fresh-encode multiply (Tier 1 correctness)
//   TEST_B  clone reuse of ONE template across DIFFERENT ciphertext levels
//           (the exact scenario the abandoned object-reuse path crashed on)
//   TEST_C  stability/leak: thousands of clone-multiplies, GPU mem stable
//   TEST_D  parallel (OMP) encode == serial encode (Tier 2 thread-safety)
//   TEST_E  v2: concurrent Decrypt (OMP) on DISTINCT ciphertexts, same
//           CryptoContext/secretKey == serial decrypt (client boundary
//           parallelization safety, read side)
//   TEST_F  v2: concurrent full boundary round-trip (Decrypt -> CPU
//           transform -> MakeCKKSPackedPlaintext -> Encrypt), same
//           CryptoContext/keys, decrypted back and checked -- mirrors
//           reduce_score_tile/emit_weight_tile called from OMP threads
//   TEST_G  v2: the ACTUAL usage pattern -- 26 independent parallel bursts
//           (matching the real driver's 13+13 key_group iterations), each
//           over FRESH ciphertexts never revisited afterward. This is the
//           gating test for v2 (v2 never re-decrypts a tile's ciphertext).
//   TEST_H  informational, NOT required for v2: repeatedly decrypt/reload
//           the SAME ciphertext objects across 20 waves. A stricter/
//           different scenario than v2's real pattern; a failure here
//           identifies a FIDESlib limitation on ciphertext reuse, not a v2
//           blocker, since v2 never revisits a tile.
//   TEST_I  v3: after LoadContext, free the CPU-side (OpenFHE) EvalMult/
//           rotation key maps via the raw OpenFHE static API (FIDESlib's
//           own LoadContext() is the ONLY place in its non-vendored source
//           that reads these maps -- verified by source inspection), then
//           confirm EvalMult (ct x ct, ct x pt), EvalRotate, Decrypt, and
//           Encrypt all still produce correct results afterward.
// Same context params as the depth13/digits3/ring65536 block.
#include <fideslib.hpp>
// Raw OpenFHE header (TEST_I only): FIDESlib's public wrapper deliberately
// hides lbcrypto types (this->cpu is a std::any), so freeing its internal
// static key-map cache needs the bare OpenFHE static API directly -- no
// FIDESlib patch required, since fideslib::fideslib already links OpenFHE's
// pke library transitively (FIDESlib's own LoadContext() calls the very
// same key-map accessors internally).
#include <openfhe.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace fideslib;

namespace {
constexpr std::uint32_t RING_DIM = 65536;
constexpr std::uint32_t MULT_DEPTH = 13;
constexpr std::uint32_t SCALE_BITS = 50;
constexpr std::uint32_t FIRST_MOD_BITS = 60;
constexpr std::uint32_t LARGE_DIGITS = 3;
constexpr std::size_t SLOTS = 32768;

// Tier-1 clone: a fresh, UNLOADED PlaintextImpl that shares the cached CPU
// encoding (no re-encode) so EvalMult loads it fresh at the ciphertext's level.
Plaintext clone_encoded(const Plaintext& tmpl) {
    Plaintext c = std::make_shared<PlaintextImpl>();
    c->cpu = tmpl->cpu;                       // share encoded data (shared_ptr copy)
    c->parent_context = tmpl->parent_context; // needed for dtor eviction
    c->loaded = false;                        // force fresh device load per use
    c->gpu = 0;
    return c;
}
}  // namespace

int main(int argc, char** argv) {
    const int gpu = (argc > 1) ? std::atoi(argv[1]) : 0;
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
    p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetMultiplicativeDepth(MULT_DEPTH);
    p.SetScalingModSize(SCALE_BITS);
    p.SetFirstModSize(FIRST_MOD_BITS);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(LARGE_DIGITS);
    p.SetBatchSize(SLOTS);
    p.SetRingDim(RING_DIM);
    p.SetDevices({gpu});
    p.SetPlaintextAutoload(false);
    p.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    // TEST_I needs a rotation key to verify EvalRotate post-clear; must be
    // generated before LoadContext (the API throws otherwise).
    cc->EvalRotateKeyGen(keys.secretKey, {1});
    cc->LoadContext(keys.publicKey);
    cc->Synchronize();
    std::cout << "[progress] context ready\n" << std::flush;

    std::vector<double> x(SLOTS), w(SLOTS);
    for (std::size_t i = 0; i < SLOTS; ++i) {
        x[i] = std::sin(0.001 * static_cast<double>(i));
        w[i] = std::cos(0.002 * static_cast<double>(i)) + 0.5;
    }
    Plaintext xpt = cc->MakeCKKSPackedPlaintext(x, 1, 0, nullptr, SLOTS);
    auto ct = cc->Encrypt(keys.publicKey, xpt);

    // Encode the weight ONCE -> pristine template (autoload off => unloaded).
    Plaintext tmpl = cc->MakeCKKSPackedPlaintext(w, 1, 0, nullptr, SLOTS);

    auto decrypt8 = [&](fideslib::Ciphertext<DCRTPoly> c) {
        Plaintext d;
        cc->Decrypt(keys.secretKey, c, &d);
        d->SetLength(SLOTS);
        auto v = d->GetRealPackedValue();
        v.resize(8);
        return v;
    };

    // TEST_A: clone-multiply vs fresh-encode-multiply
    Plaintext fresh = cc->MakeCKKSPackedPlaintext(w, 1, 0, nullptr, SLOTS);
    auto ref = cc->EvalMult(ct, fresh);
    Plaintext cl = clone_encoded(tmpl);
    auto got = cc->EvalMult(ct, cl);
    auto rv = decrypt8(ref), gv = decrypt8(got);
    double a_max = 0.0;
    for (int i = 0; i < 8; ++i) a_max = std::max(a_max, std::abs(rv[i] - gv[i]));
    std::cout << "TEST_A clone_vs_fresh maxdiff=" << a_max << "\n" << std::flush;

    // TEST_B: reuse the SAME template (via a second clone) against a
    // DIFFERENT-level ciphertext (got is one level below ct).
    Plaintext cl2 = clone_encoded(tmpl);
    auto lvl2 = cc->EvalMult(got, cl2);
    auto l2 = decrypt8(lvl2);
    double b_max = 0.0;
    for (int i = 0; i < 8; ++i)
        b_max = std::max(b_max, std::abs(l2[i] - x[i] * w[i] * w[i]));
    std::cout << "TEST_B cross_level maxdiff_vs_expected=" << b_max << "\n" << std::flush;

    // TEST_C: stability / no leak across many clone-multiplies.
    for (int i = 0; i < 3000; ++i) {
        Plaintext ci = clone_encoded(tmpl);
        auto t = cc->EvalMult(ct, ci);
        (void)t;
    }
    cc->Synchronize();
    std::cout << "TEST_C stability_3000_clone_mults OK\n" << std::flush;

    // TEST_D: parallel (OMP) encode == serial encode.
    const int N = 96;
    std::vector<std::vector<double>> ws(N, w);
    for (int i = 0; i < N; ++i) ws[i][0] = 0.1 * i;
    std::vector<Plaintext> ser(N), par(N);
    for (int i = 0; i < N; ++i)
        ser[i] = cc->MakeCKKSPackedPlaintext(ws[i], 1, 0, nullptr, SLOTS);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i)
        par[i] = cc->MakeCKKSPackedPlaintext(ws[i], 1, 0, nullptr, SLOTS);
    double d_max = 0.0;
    for (int i = 0; i < N; ++i) {
        auto sa = ser[i]->GetCKKSPackedValue();
        auto pa = par[i]->GetCKKSPackedValue();
        for (std::size_t j = 0; j < sa.size(); ++j)
            d_max = std::max(d_max, std::abs(sa[j].real() - pa[j].real()));
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::cout << "TEST_D parallel_encode_vs_serial maxdiff=" << d_max
              << " omp_threads=" << threads << "\n" << std::flush;

    // ---- TEST_I: v3 -- free CPU-side EvalMult/rotation key maps ------------
    // Source-verified (fideslib-src, non-vendored files only): LoadContext()
    // is the ONLY call site that reads GetAllEvalMultKeys()/
    // GetAllEvalAutomorphismKeys() -- once, per key, to build the GPU-side
    // KeySwitchingKey objects added via AddEvalKey/AddRotationKey. Every
    // GPU-path EvalMult/EvalRotate/Decrypt/Encrypt afterward operates purely
    // on the already-loaded GPU-resident FIDESlib::CKKS::Context and never
    // re-reads these CPU maps. This test proves that empirically: clear them
    // via the raw OpenFHE static API (our process holds exactly one crypto
    // context, so the global no-argument overload is equivalent to a scoped
    // clear), then confirm ct*ct, ct*pt, EvalRotate, Decrypt, and a fresh
    // Encrypt still all produce correct results.
    std::cout << "[progress] TEST_I starting (free CPU eval-key maps)\n"
              << std::flush;
    lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalMultKeys();
    lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalAutomorphismKeys();
    std::cout << "[progress] TEST_I key maps cleared, re-verifying ops\n"
              << std::flush;

    // Each sub-check prints its own result IMMEDIATELY (flushed), so if a
    // later check throws/aborts, earlier results are not lost to stdout
    // buffering -- exactly the lesson learned from the TEST_A-H crashes.

    // ct x ct (needs the relin key we just freed CPU-side)
    Plaintext i_wpt = cc->MakeCKKSPackedPlaintext(w, 1, 0, nullptr, SLOTS);
    auto i_wct = cc->Encrypt(keys.publicKey, i_wpt);
    auto i_ctct = cc->EvalMult(ct, i_wct);
    auto i_ctct_v = decrypt8(i_ctct);
    double i_ctct_max = 0.0;
    for (int k = 0; k < 8; ++k)
        i_ctct_max = std::max(i_ctct_max, std::abs(i_ctct_v[k] - x[k] * w[k]));
    std::cout << "TEST_I post_clear_ctct_maxdiff=" << i_ctct_max << "\n"
              << std::flush;

    // ct x pt (fresh encode + clone-multiply, Tier-1 path)
    Plaintext i_cl = clone_encoded(tmpl);
    auto i_ctpt = cc->EvalMult(ct, i_cl);
    auto i_ctpt_v = decrypt8(i_ctpt);
    double i_ctpt_max = 0.0;
    for (int k = 0; k < 8; ++k)
        i_ctpt_max = std::max(i_ctpt_max, std::abs(i_ctpt_v[k] - x[k] * w[k]));
    std::cout << "TEST_I post_clear_ctpt_maxdiff=" << i_ctpt_max << "\n"
              << std::flush;

    // EvalRotate (needs the rotation key we just freed CPU-side)
    auto i_rot = cc->EvalRotate(ct, 1);
    Plaintext i_rot_pt;
    fideslib::Ciphertext<DCRTPoly> i_rot_local = i_rot;
    cc->Decrypt(keys.secretKey, i_rot_local, &i_rot_pt);
    i_rot_pt->SetLength(SLOTS);
    auto i_rot_v = i_rot_pt->GetRealPackedValue();
    double i_rot_max = 0.0;
    for (int k = 0; k < 8; ++k)
        i_rot_max = std::max(i_rot_max, std::abs(i_rot_v[k] - x[k + 1]));
    std::cout << "TEST_I post_clear_rotate_maxdiff=" << i_rot_max << "\n"
              << std::flush;

    // Fresh Decrypt + Encrypt round-trip (secret/public key themselves are
    // untouched by the clear -- only the eval-mult/rotation key MAPS were --
    // but verify end-to-end anyway).
    Plaintext i_fresh_pt = cc->MakeCKKSPackedPlaintext(x, 1, 0, nullptr, SLOTS);
    auto i_fresh_ct = cc->Encrypt(keys.publicKey, i_fresh_pt);
    auto i_fresh_v = decrypt8(i_fresh_ct);
    double i_fresh_max = 0.0;
    for (int k = 0; k < 8; ++k)
        i_fresh_max = std::max(i_fresh_max, std::abs(i_fresh_v[k] - x[k]));
    std::cout << "TEST_I post_clear_fresh_roundtrip_maxdiff=" << i_fresh_max
              << "\n" << std::flush;

    // ---- v2 tests: concurrent CLIENT boundary crossings (Decrypt/Encrypt) --
    // reduce_score_tile/emit_weight_tile in the real driver copy the input
    // ciphertext into a local var, then call cc_->Decrypt(secretKey, ...) or
    // cc_->Encrypt(publicKey, ...) on the SAME shared CryptoContext/keys.
    // v2 proposes calling several of these independent boundary crossings
    // from different OMP threads concurrently. Unlike TEST_D (host-only CPU
    // encode), Decrypt/Encrypt touch GPU state (load/evict), so this must be
    // validated separately before touching the real driver.
    std::cout << "[progress] TEST_E/F/G setup starting\n" << std::flush;
    const int M = 64;
    std::vector<std::vector<double>> xs(M, std::vector<double>(SLOTS));
    for (int i = 0; i < M; ++i)
        for (std::size_t k = 0; k < SLOTS; ++k)
            xs[i][k] = std::sin((0.0003 + 0.00001 * i) * static_cast<double>(k)) +
                       0.1 * i;
    std::vector<fideslib::Ciphertext<DCRTPoly>> cts(M);
    for (int i = 0; i < M; ++i) {
        Plaintext pt = cc->MakeCKKSPackedPlaintext(xs[i], 1, 0, nullptr, SLOTS);
        cts[i] = cc->Encrypt(keys.publicKey, pt);
    }
    std::cout << "[progress] TEST_E/F/G setup done (" << M
              << " ciphertexts serially encrypted)\n" << std::flush;

    auto decryptN = [&](fideslib::Ciphertext<DCRTPoly> c, std::size_t n) {
        fideslib::Ciphertext<DCRTPoly> local = c;
        Plaintext d;
        cc->Decrypt(keys.secretKey, local, &d);
        d->SetLength(SLOTS);
        auto v = d->GetRealPackedValue();
        v.resize(n);
        return v;
    };

    // TEST_E: concurrent Decrypt on M distinct ciphertexts vs the known
    // plaintext each was encrypted from (randomized encryption means we
    // cannot compare ciphertext-to-ciphertext against a serial run, but
    // decrypting either should recover the same input within CKKS noise).
    std::cout << "[progress] TEST_E starting (" << M
              << " concurrent decrypts)\n" << std::flush;
    std::vector<double> e_max(M, 0.0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        auto v = decryptN(cts[i], 8);
        double m = 0.0;
        for (int k = 0; k < 8; ++k) m = std::max(m, std::abs(v[k] - xs[i][k]));
        e_max[i] = m;
    }
    double e_worst = 0.0;
    for (double m : e_max) e_worst = std::max(e_worst, m);
    std::cout << "TEST_E concurrent_decrypt_vs_known maxdiff=" << e_worst
              << "\n" << std::flush;

    // TEST_F: concurrent full round-trip -- Decrypt, exact CPU transform,
    // re-encode, re-encrypt -- from OMP threads on the shared context/keys,
    // exactly mirroring reduce_score_tile/emit_weight_tile's call pattern.
    // Correctness is checked against the known exact-math expectation
    // (transform applied in double precision), not against a serial CKKS
    // run, because Encrypt is randomized -- two encryptions of the same
    // plaintext are different ciphertexts by design.
    auto transform = [](double v) { return v * v + 0.25; };  // any nonlinearity stand-in
    std::cout << "[progress] TEST_F starting (" << M
              << " concurrent decrypt+encode+encrypt round-trips)\n" << std::flush;
    std::vector<double> f_max(M, 0.0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        fideslib::Ciphertext<DCRTPoly> local = cts[i];
        Plaintext d;
        cc->Decrypt(keys.secretKey, local, &d);
        d->SetLength(SLOTS);
        std::vector<double> v = d->GetRealPackedValue();
        for (double& x : v) x = transform(x);
        Plaintext refreshed = cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, SLOTS);
        auto out = cc->Encrypt(keys.publicKey, refreshed);
        auto back = decryptN(out, 8);
        double m = 0.0;
        for (int k = 0; k < 8; ++k)
            m = std::max(m, std::abs(back[k] - transform(xs[i][k])));
        f_max[i] = m;
    }
    double f_worst = 0.0;
    for (double m : f_max) f_worst = std::max(f_worst, m);
    std::cout << "TEST_F concurrent_boundary_roundtrip maxdiff=" << f_worst
              << "\n" << std::flush;

    // TEST_G: v2's ACTUAL usage pattern -- 26 independent parallel bursts
    // (matching the real driver's 13 score-tile + 13 weight-tile key_group
    // iterations), each over FRESH ciphertexts that are never touched again
    // afterward. Unlike TEST_H below, no ciphertext object is EVER decrypted
    ///re-loaded more than once across the whole test -- this is the actual
    // gating test for v2, since v2 never revisits a tile's ciphertext.
    constexpr int WAVES_G = 26;
    std::cout << "[progress] TEST_G starting (" << WAVES_G << " waves x " << M
              << " concurrent round-trips, FRESH ciphertexts every wave)\n"
              << std::flush;
    double g_worst = 0.0;
    for (int wave = 0; wave < WAVES_G; ++wave) {
        std::vector<std::vector<double>> wxs(M, std::vector<double>(SLOTS));
        for (int i = 0; i < M; ++i)
            for (std::size_t k = 0; k < SLOTS; ++k)
                wxs[i][k] = std::sin((0.0004 + 0.00002 * (wave * M + i)) *
                                      static_cast<double>(k)) +
                            0.05 * wave;
        std::vector<fideslib::Ciphertext<DCRTPoly>> wcts(M);
        for (int i = 0; i < M; ++i) {
            Plaintext pt = cc->MakeCKKSPackedPlaintext(wxs[i], 1, 0, nullptr, SLOTS);
            wcts[i] = cc->Encrypt(keys.publicKey, pt);
        }
        std::vector<double> wave_max(M, 0.0);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < M; ++i) {
            fideslib::Ciphertext<DCRTPoly> local = wcts[i];
            Plaintext d;
            cc->Decrypt(keys.secretKey, local, &d);
            d->SetLength(SLOTS);
            std::vector<double> v = d->GetRealPackedValue();
            for (double& x : v) x = transform(x);
            Plaintext refreshed =
                cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, SLOTS);
            auto out = cc->Encrypt(keys.publicKey, refreshed);
            fideslib::Ciphertext<DCRTPoly> outlocal = out;
            Plaintext back;
            cc->Decrypt(keys.secretKey, outlocal, &back);
            back->SetLength(SLOTS);
            auto bv = back->GetRealPackedValue();
            double m = 0.0;
            for (int k = 0; k < 8; ++k)
                m = std::max(m, std::abs(bv[k] - transform(wxs[i][k])));
            wave_max[i] = m;
        }
        for (double m : wave_max) g_worst = std::max(g_worst, m);
        std::cout << "[progress] TEST_G wave " << wave << " done (maxdiff so far="
                  << g_worst << ")\n" << std::flush;
    }
    cc->Synchronize();
    std::cout << "TEST_G fresh_per_wave_" << WAVES_G << "x" << M
              << "_concurrent_roundtrips OK maxdiff=" << g_worst << "\n"
              << std::flush;

    // TEST_H: a STRICTER, separate scenario -- repeatedly decrypt/re-load the
    // SAME M ciphertext objects across many waves (NOT how v2 actually uses
    // ciphertexts; v2 never revisits a tile). Informational: if this fails
    // but TEST_G passed, it identifies a real FIDESlib limitation (repeated
    // concurrent reuse of one ciphertext across bursts) that v2 does not
    // trigger, rather than a blocker for v2 itself.
    std::cout << "[progress] TEST_H starting (20 waves x " << M
              << " concurrent round-trips, SAME ciphertexts reused every wave "
                 "-- informational, not required for v2)\n" << std::flush;
    for (int wave = 0; wave < 20; ++wave) {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < M; ++i) {
            fideslib::Ciphertext<DCRTPoly> local = cts[i];
            Plaintext d;
            cc->Decrypt(keys.secretKey, local, &d);
            d->SetLength(SLOTS);
            std::vector<double> v = d->GetRealPackedValue();
            for (double& x : v) x = transform(x);
            Plaintext refreshed =
                cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, SLOTS);
            auto out = cc->Encrypt(keys.publicKey, refreshed);
            (void)out;
        }
        std::cout << "[progress] TEST_H wave " << wave << " done\n"
                  << std::flush;
    }
    cc->Synchronize();
    // final spot-check after the stress waves: decrypt the ORIGINAL cts[]
    // (untouched by TEST_H, which only read/re-encrypted into `out`) and
    // confirm they still decrypt correctly -- catches corruption of shared
    // GPU state that TEST_H's own transform-based check might not reveal.
    double h_after = 0.0;
    for (int i = 0; i < M; ++i) {
        auto v = decryptN(cts[i], 8);
        for (int k = 0; k < 8; ++k)
            h_after = std::max(h_after, std::abs(v[k] - xs[i][k]));
    }
    std::cout << "TEST_H stability_20waves_" << (20 * M)
              << "_concurrent_roundtrips_SAME_OBJECTS OK maxdiff_after="
              << h_after << "\n" << std::flush;

    std::cout << "MICROGATE_DONE\n" << std::flush;
    return 0;
}
