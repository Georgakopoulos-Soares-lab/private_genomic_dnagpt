#pragma once

// FIDES-independent C++ contract for the Scheme B T=103 SIMD schedule.
// This header contains no cryptographic implementation. It freezes the slot,
// causal-tile, boundary, and source-operation arithmetic that the future full
// FIDESlib source must use.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dnagpt::scheme_b::simd {

constexpr std::size_t D = 768;
constexpr std::size_t T = 103;
constexpr std::size_t HEADS = 12;
constexpr std::size_t HEAD_DIM = 64;
constexpr std::size_t PACK_WIDTH = 1024;
constexpr std::size_t COPIES = 4;
constexpr std::size_t LOGICAL_SLOTS = COPIES * PACK_WIDTH;
constexpr std::size_t BATCH = 8;
constexpr std::size_t PHYSICAL_SLOTS = LOGICAL_SLOTS * BATCH;
constexpr std::size_t GROUPS = (T + BATCH - 1) / BATCH;
constexpr std::size_t BSGS_N1 = 32;
constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;

static_assert(D == HEADS * HEAD_DIM);
static_assert(PHYSICAL_SLOTS == 32768);
static_assert(GROUPS == 13);
static_assert(T % BATCH == 7);
static_assert(BSGS_N1 * BSGS_N2 == PACK_WIDTH);

constexpr std::size_t physical_slot(std::size_t copy,
                                    std::size_t feature,
                                    std::size_t lane) {
    if (copy >= COPIES || feature >= PACK_WIDTH || lane >= BATCH) {
        throw std::out_of_range("SIMD slot coordinate outside layout");
    }
    return (copy * PACK_WIDTH + feature) * BATCH + lane;
}

constexpr std::size_t group_count(std::string_view mode) {
    if (mode == "packed") {
        return GROUPS;
    }
    if (mode == "serial_control") {
        return T;
    }
    throw std::invalid_argument("unknown SIMD execution mode");
}

constexpr std::size_t first_token(std::string_view mode,
                                  std::size_t group) {
    if (mode == "packed") {
        return group * BATCH;
    }
    if (mode == "serial_control") {
        return group;
    }
    throw std::invalid_argument("unknown SIMD execution mode");
}

constexpr std::size_t active_tokens(std::string_view mode,
                                    std::size_t group) {
    if (mode == "serial_control") {
        return 1;
    }
    if (mode != "packed") {
        throw std::invalid_argument("unknown SIMD execution mode");
    }
    const std::size_t first = first_token(mode, group);
    return std::min(BATCH, T - first);
}

inline std::vector<double> rotate_left(const std::vector<double>& values,
                                       std::int64_t amount) {
    if (values.empty()) {
        return {};
    }
    const std::int64_t size = static_cast<std::int64_t>(values.size());
    std::int64_t canonical = amount % size;
    if (canonical < 0) {
        canonical += size;
    }
    std::vector<double> output(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::size_t source =
            (index + static_cast<std::size_t>(canonical)) % values.size();
        output[index] = values[source];
    }
    return output;
}

inline std::vector<double> logical_rotate(
    const std::vector<double>& values, std::int64_t logical_amount) {
    return rotate_left(values,
                       logical_amount * static_cast<std::int64_t>(BATCH));
}

inline std::vector<double> lane_shift(const std::vector<double>& values,
                                      std::int64_t token_amount) {
    if (values.size() % BATCH != 0) {
        throw std::invalid_argument(
            "lane-shift input must divide by SIMD batch");
    }
    std::int64_t amount =
        token_amount % static_cast<std::int64_t>(BATCH);
    if (amount < 0) {
        amount += static_cast<std::int64_t>(BATCH);
    }
    if (amount == 0) {
        return values;
    }
    const auto from_next = rotate_left(values, amount);
    const auto from_same =
        rotate_left(values, amount - static_cast<std::int64_t>(BATCH));
    std::vector<double> output(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::size_t lane = index % BATCH;
        output[index] =
            lane < BATCH - static_cast<std::size_t>(amount)
                ? from_next[index]
                : from_same[index];
    }
    return output;
}

inline std::vector<std::size_t> active_weight_shifts(
    std::size_t query_start, std::size_t query_count,
    std::size_t key_start, std::size_t key_count) {
    std::vector<std::size_t> active;
    for (std::size_t delta = 0; delta < BATCH; ++delta) {
        bool found = false;
        for (std::size_t query_lane = 0; query_lane < query_count;
             ++query_lane) {
            const std::size_t key_lane =
                (query_lane + delta) % BATCH;
            if (key_lane < key_count &&
                key_start + key_lane <= query_start + query_lane) {
                found = true;
                break;
            }
        }
        if (found) {
            active.push_back(delta);
        }
    }
    return active;
}

struct ProtocolCounts {
    std::size_t token_groups = 0;
    std::size_t score_tile_decryptions = 0;
    std::size_t weight_tile_encryptions = 0;
    std::size_t ln1_crossings = 0;
    std::size_t ln2_crossings = 0;
    std::size_t gelu_crossings = 0;

    constexpr std::size_t attention_gate_crossings() const {
        return score_tile_decryptions + weight_tile_encryptions +
               ln1_crossings;
    }

    constexpr std::size_t full_block_crossings() const {
        return attention_gate_crossings() + ln2_crossings +
               gelu_crossings;
    }
};

inline ProtocolCounts protocol_counts() {
    ProtocolCounts counts;
    counts.token_groups = GROUPS;
    counts.ln1_crossings = GROUPS;
    counts.ln2_crossings = GROUPS;
    counts.gelu_crossings = GROUPS;
    for (std::size_t query_group = 0; query_group < GROUPS;
         ++query_group) {
        const std::size_t query_start = query_group * BATCH;
        const std::size_t query_count =
            std::min(BATCH, T - query_start);
        for (std::size_t key_group = 0; key_group <= query_group;
             ++key_group) {
            const std::size_t key_start = key_group * BATCH;
            const std::size_t key_count =
                std::min(BATCH, T - key_start);
            ++counts.score_tile_decryptions;
            counts.weight_tile_encryptions +=
                active_weight_shifts(query_start, query_count,
                                     key_start, key_count)
                    .size();
        }
    }
    return counts;
}

inline std::size_t unique_cached_lane_shifts() {
    std::vector<std::set<std::size_t>> by_key_group(GROUPS);
    for (std::size_t query_group = 0; query_group < GROUPS;
         ++query_group) {
        const std::size_t query_start = query_group * BATCH;
        const std::size_t query_count =
            std::min(BATCH, T - query_start);
        for (std::size_t key_group = 0; key_group <= query_group;
             ++key_group) {
            const std::size_t key_start = key_group * BATCH;
            const std::size_t key_count =
                std::min(BATCH, T - key_start);
            for (const std::size_t delta : active_weight_shifts(
                     query_start, query_count, key_start, key_count)) {
                if (delta != 0) {
                    by_key_group[key_group].insert(delta);
                }
            }
        }
    }
    std::size_t total = 0;
    for (const auto& shifts : by_key_group) {
        total += shifts.size();
    }
    return total;
}

struct ServerCounts {
    std::size_t matrix_products = 0;
    std::size_t ct_ct = 0;
    std::size_t ct_plain = 0;
    std::size_t explicit_rotations = 0;
    std::size_t accumulate_sum_calls = 0;
};

inline ServerCounts simd_server_counts() {
    const ProtocolCounts protocol = protocol_counts();
    const std::size_t alignments =
        protocol.weight_tile_encryptions;
    const std::size_t matrix_products = 12 * GROUPS;
    const std::size_t cached_shifts = unique_cached_lane_shifts();

    const std::size_t dense_baby =
        GROUPS * 7 * (BSGS_N1 - 1);
    const std::size_t dense_giant =
        matrix_products * (BSGS_N2 - 1);
    const std::size_t lane_rotations = 4 * cached_shifts;
    const std::size_t mlp_copy_rotations = 12 * GROUPS;

    const std::size_t dense_plain =
        matrix_products * PACK_WIDTH;
    const std::size_t score_plain =
        2 * HEADS * alignments;
    const std::size_t lane_plain = 4 * cached_shifts;
    const std::size_t mlp_copy_plain = 8 * GROUPS;
    const std::size_t layernorm_plain = 6 * GROUPS;

    return {
        .matrix_products = matrix_products,
        .ct_ct = 2 * alignments + 4 * GROUPS,
        .ct_plain = dense_plain + score_plain + lane_plain +
                    mlp_copy_plain + layernorm_plain,
        .explicit_rotations = dense_baby + dense_giant +
                              lane_rotations + mlp_copy_rotations,
        .accumulate_sum_calls = HEADS * alignments + 4 * GROUPS,
    };
}

inline ServerCounts frozen_server_counts() {
    const std::size_t matrix_products = 12 * T;
    const std::size_t raw_scores =
        HEADS * (T * (T + 1) / 2 - 1);
    const std::size_t context_products = T * (T - 1) / 2;
    return {
        .matrix_products = matrix_products,
        .ct_ct = raw_scores + context_products + 4 * T,
        .ct_plain =
            matrix_products * PACK_WIDTH + 3 * raw_scores + 15 * T,
        .explicit_rotations = 0,
        .accumulate_sum_calls = 0,
    };
}

}  // namespace dnagpt::scheme_b::simd
