#include "src/simd_t103_schedule.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace simd = dnagpt::scheme_b::simd;

void test_slot_bijection() {
    std::set<std::size_t> seen;
    for (std::size_t copy = 0; copy < simd::COPIES; ++copy) {
        for (std::size_t feature = 0; feature < simd::PACK_WIDTH;
             ++feature) {
            for (std::size_t lane = 0; lane < simd::BATCH; ++lane) {
                const std::size_t slot =
                    simd::physical_slot(copy, feature, lane);
                assert(seen.insert(slot).second);
                assert(slot % simd::BATCH == lane);
            }
        }
    }
    assert(seen.size() == simd::PHYSICAL_SLOTS);
    assert(*seen.begin() == 0);
    assert(*seen.rbegin() == simd::PHYSICAL_SLOTS - 1);
}

void test_modes_and_partial_tail() {
    assert(simd::group_count("packed") == 13);
    assert(simd::group_count("serial_control") == 103);
    assert(simd::first_token("packed", 12) == 96);
    assert(simd::active_tokens("packed", 12) == 7);
    assert(simd::first_token("serial_control", 102) == 102);
    assert(simd::active_tokens("serial_control", 102) == 1);
    bool rejected = false;
    try {
        static_cast<void>(simd::group_count("invalid"));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

void test_lane_shift_and_scaled_rotation() {
    constexpr std::size_t logical = 19;
    std::vector<double> values(logical * simd::BATCH);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<double>(index);
    }
    for (std::int64_t delta = -16; delta <= 16; ++delta) {
        const auto got = simd::lane_shift(values, delta);
        std::int64_t canonical =
            delta % static_cast<std::int64_t>(simd::BATCH);
        if (canonical < 0) {
            canonical += static_cast<std::int64_t>(simd::BATCH);
        }
        for (std::size_t position = 0; position < logical; ++position) {
            for (std::size_t lane = 0; lane < simd::BATCH; ++lane) {
                const std::size_t source_lane =
                    (lane + static_cast<std::size_t>(canonical)) %
                    simd::BATCH;
                assert(got[position * simd::BATCH + lane] ==
                       values[position * simd::BATCH + source_lane]);
            }
        }
    }

    const auto got = simd::logical_rotate(values, 3);
    for (std::size_t position = 0; position < logical; ++position) {
        const std::size_t source_position =
            (position + 3) % logical;
        for (std::size_t lane = 0; lane < simd::BATCH; ++lane) {
            assert(got[position * simd::BATCH + lane] ==
                   values[source_position * simd::BATCH + lane]);
        }
    }
}

void test_every_causal_pair_once() {
    std::set<std::pair<std::size_t, std::size_t>> covered;
    std::size_t visits = 0;
    for (std::size_t query_group = 0; query_group < simd::GROUPS;
         ++query_group) {
        const std::size_t query_start = query_group * simd::BATCH;
        const std::size_t query_count =
            std::min(simd::BATCH, simd::T - query_start);
        for (std::size_t key_group = 0; key_group <= query_group;
             ++key_group) {
            const std::size_t key_start = key_group * simd::BATCH;
            const std::size_t key_count =
                std::min(simd::BATCH, simd::T - key_start);
            for (const std::size_t delta : simd::active_weight_shifts(
                     query_start, query_count, key_start, key_count)) {
                for (std::size_t query_lane = 0;
                     query_lane < query_count; ++query_lane) {
                    const std::size_t key_lane =
                        (query_lane + delta) % simd::BATCH;
                    if (key_lane >= key_count) {
                        continue;
                    }
                    const std::size_t row =
                        query_start + query_lane;
                    const std::size_t col = key_start + key_lane;
                    if (col <= row) {
                        ++visits;
                        assert(covered.insert({row, col}).second);
                    }
                }
            }
        }
    }
    assert(visits == simd::T * (simd::T + 1) / 2);
    assert(covered.size() == 5356);
    for (std::size_t row = 0; row < simd::T; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            assert(covered.contains({row, col}));
        }
    }
}

void test_protocol_and_server_counts() {
    const auto protocol = simd::protocol_counts();
    assert(protocol.token_groups == 13);
    assert(protocol.score_tile_decryptions == 91);
    assert(protocol.weight_tile_encryptions == 727);
    assert(protocol.attention_gate_crossings() == 831);
    assert(protocol.full_block_crossings() == 857);
    assert(simd::unique_cached_lane_shifts() == 90);

    const auto packed = simd::simd_server_counts();
    assert(packed.matrix_products == 156);
    assert(packed.ct_ct == 1506);
    assert(packed.ct_plain == 177734);
    assert(packed.explicit_rotations == 8173);
    assert(packed.accumulate_sum_calls == 8776);

    const auto frozen = simd::frozen_server_counts();
    assert(frozen.matrix_products == 1236);
    assert(frozen.ct_ct == 69925);
    assert(frozen.ct_plain == 1459989);
    assert(std::abs(
               static_cast<double>(frozen.matrix_products) /
                   static_cast<double>(packed.matrix_products) -
               7.923076923076923) <
           1e-12);
}

int main() {
    test_slot_bijection();
    test_modes_and_partial_tail();
    test_lane_shift_and_scaled_rotation();
    test_every_causal_pair_once();
    test_protocol_and_server_counts();
    std::cout << "SIMD_T103_CPP_SCHEDULE_CONTRACT_PASS\n";
    return 0;
}
