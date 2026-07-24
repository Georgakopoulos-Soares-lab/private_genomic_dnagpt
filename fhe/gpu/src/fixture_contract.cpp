#include "toy_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

int main() {
    constexpr std::array<double, dnagpt::toy::kT * dnagpt::toy::kD> golden = {
        -0.16741677289706122,  -0.14388345454115914, 0.004426402723787172,
        0.23574232299239492,   0.46672964553269169,  0.60278232242602003,
        0.57395049590278757,   0.3632135240745013,   -0.15775069270175968,
        -0.10033578114944187,  -0.059047090408349504, -0.039843345828210038,
        -0.016801053717575742, 0.040754895144396203, 0.13628940524383606,
        0.23188121535712786,   0.03276130048679228,  -0.03598888069047327,
        -0.30398811770948814,  -0.39727909901757008, -0.067679150270082811,
        0.48211673664826571,   0.76200671905535444,  0.4885193464473862,
        0.10876980263758262,   -0.2402561956112178,  -0.58018942890480829,
        -0.16103552417717176,  0.53928026584147093,  0.57373452465073616,
        0.094697934567531075,  -0.035451416626039657,
    };

    const auto fixture = dnagpt::toy::make_fixture();
    dnagpt::toy::DomainAudit domains;
    const auto output = dnagpt::toy::block_oracle(fixture, &domains);
    double max_error = 0.0;
    bool finite = true;
    std::size_t index = 0;
    for (const auto& token : output) {
        for (const double value : token) {
            finite = finite && std::isfinite(value);
            max_error = std::max(max_error, std::abs(value - golden[index++]));
        }
    }
    const bool pass = finite && max_error <= 1e-13 && domains.inside_contract();
    std::cout.precision(17);
    std::cout << "fixture=D8_T4_H2_MLP32_formula_v1 max_abs_error=" << max_error
              << " finite=" << finite << '\n';
    std::cout << "domains variance=[" << domains.variance_min << ","
              << domains.variance_max << "] score=[" << domains.score_min << ","
              << domains.score_max << "] denominator=[" << domains.denominator_min << ","
              << domains.denominator_max << "] gelu_input=["
              << domains.gelu_input_min << "," << domains.gelu_input_max
              << "] inside_contract=" << domains.inside_contract() << '\n';
    std::cout << (pass ? "FIXTURE_CONTRACT_PASS" : "FIXTURE_CONTRACT_FAIL") << '\n';
    return pass ? 0 : 1;
}
