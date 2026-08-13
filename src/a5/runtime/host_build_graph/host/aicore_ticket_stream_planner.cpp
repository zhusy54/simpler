/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "aicore_ticket_stream_planner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

int32_t fanin_count(const AicoreReadonlyGraphV0 &graph, uint32_t task_id) {
    const uint8_t *payload = aicore_graph_payload_v0(graph, static_cast<int64_t>(task_id));
    return *reinterpret_cast<const int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
}

bool stream_is_ordered(const std::vector<uint32_t> &stream, const std::vector<uint32_t> &bottom_level) {
    return std::is_sorted(stream.begin(), stream.end(), [&](uint32_t lhs, uint32_t rhs) {
        if (bottom_level[lhs] != bottom_level[rhs]) return bottom_level[lhs] > bottom_level[rhs];
        return lhs < rhs;
    });
}

}  // namespace

bool build_aicore_ticket_streams(
    const AicoreReadonlyGraphV0 &graph, const std::vector<uint8_t> &inline_completed, AicoreTicketStreams *result
) {
    if (result == nullptr || graph.task_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        inline_completed.size() != graph.task_count) {
        return false;
    }

    AicoreTicketStreams planned;
    planned.bottom_level.assign(static_cast<size_t>(graph.task_count), 0);
    if (graph.task_count == 0) {
        *result = std::move(planned);
        return true;
    }
    if (graph.descriptors_address == 0 || graph.payloads_address == 0) return false;

    std::vector<AicoreRootCoreTypeV0> core_types(static_cast<size_t>(graph.task_count), AicoreRootCoreTypeV0::NONE);
    std::vector<uint8_t> covered(static_cast<size_t>(graph.task_count), 0);
    planned.aic.reserve(static_cast<size_t>(graph.task_count));
    planned.aiv.reserve(static_cast<size_t>(graph.task_count));

    for (uint32_t task_id = 0; task_id < graph.task_count; ++task_id) {
        AicoreTaskInfoV0 task{};
        const AicoreRootStatusV0 status = aicore_classify_task_v0(graph, task_id, &task);
        if (inline_completed[task_id] != 0) {
            if (status != AicoreRootStatusV0::UNSUPPORTED_SHAPE) return false;
            covered[task_id] = 1;
            continue;
        }
        if (status != AicoreRootStatusV0::OK) return false;
        core_types[task_id] = task.core_type;
        (task.core_type == AicoreRootCoreTypeV0::AIC ? planned.aic : planned.aiv).push_back(task_id);
        covered[task_id] = 1;
    }

    for (uint32_t consumer = static_cast<uint32_t>(graph.task_count); consumer-- > 0;) {
        const int32_t count = fanin_count(graph, consumer);
        for (int32_t index = 0; index < count; ++index) {
            const int32_t producer = aicore_graph_fanin_id_v0(graph, consumer, index);
            const uint32_t candidate = planned.bottom_level[consumer] + 1;
            planned.bottom_level[static_cast<uint32_t>(producer)] =
                std::max(planned.bottom_level[static_cast<uint32_t>(producer)], candidate);
        }
    }

    auto priority_before = [&](uint32_t lhs, uint32_t rhs) {
        if (planned.bottom_level[lhs] != planned.bottom_level[rhs]) {
            return planned.bottom_level[lhs] > planned.bottom_level[rhs];
        }
        return lhs < rhs;
    };
    std::sort(planned.aic.begin(), planned.aic.end(), priority_before);
    std::sort(planned.aiv.begin(), planned.aiv.end(), priority_before);

    if (!std::all_of(
            covered.begin(), covered.end(),
            [](uint8_t value) {
                return value == 1;
            }
        ) ||
        !stream_is_ordered(planned.aic, planned.bottom_level) ||
        !stream_is_ordered(planned.aiv, planned.bottom_level)) {
        return false;
    }

    std::vector<uint8_t> streamed(static_cast<size_t>(graph.task_count), 0);
    auto validate_stream = [&](const std::vector<uint32_t> &stream, AicoreRootCoreTypeV0 expected_type) {
        for (uint32_t task_id : stream) {
            if (task_id >= graph.task_count || streamed[task_id] != 0 || inline_completed[task_id] != 0 ||
                core_types[task_id] != expected_type) {
                return false;
            }
            streamed[task_id] = 1;
        }
        return true;
    };
    if (!validate_stream(planned.aic, AicoreRootCoreTypeV0::AIC) ||
        !validate_stream(planned.aiv, AicoreRootCoreTypeV0::AIV)) {
        return false;
    }

    for (uint32_t consumer = 0; consumer < graph.task_count; ++consumer) {
        if (inline_completed[consumer] == 0 && streamed[consumer] == 0) return false;
        const int32_t count = fanin_count(graph, consumer);
        for (int32_t index = 0; index < count; ++index) {
            const uint32_t producer = static_cast<uint32_t>(aicore_graph_fanin_id_v0(graph, consumer, index));
            if (planned.bottom_level[producer] <= planned.bottom_level[consumer]) return false;
        }
    }

    *result = std::move(planned);
    return true;
}
