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

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "pto2_dispatch_payload.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __host__
#define __host__
#endif

inline constexpr uint32_t AICORE_GRAPH_TASK_DESCRIPTOR_STRIDE_V0 = 40;
inline constexpr uint32_t AICORE_GRAPH_TASK_PAYLOAD_STRIDE_V0 = 4864;
inline constexpr uint32_t AICORE_GRAPH_TASK_ID_OFFSET_V0 = 0;
inline constexpr uint32_t AICORE_GRAPH_KERNEL_IDS_OFFSET_V0 = 8;
inline constexpr uint32_t AICORE_GRAPH_FANIN_COUNT_OFFSET_V0 = 8;
inline constexpr uint32_t AICORE_GRAPH_FANIN_IDS_OFFSET_V0 = 12;
inline constexpr int32_t AICORE_GRAPH_MAX_FANIN_V0 = 128;
inline constexpr uint32_t AICORE_CORE_CALLABLE_RESOLVED_ADDR_OFFSET_V0 = 136;
inline constexpr int32_t AICORE_GRAPH_INVALID_KERNEL_ID_V0 = -1;

enum class AicoreRootCoreTypeV0 : uint64_t {
    AIC = 0,
    AIV = 1,
    NONE = UINT64_MAX,
};

enum class AicoreRootStatusV0 : uint64_t {
    OK = 0,
    EMPTY = 1,
    INVALID_TASK_COUNT = 2,
    INVALID_TASK_ID = 3,
    HAS_FANIN = 4,
    UNSUPPORTED_SHAPE = 5,
    INVALID_ARGUMENTS = 6,
    INVALID_CALLABLE = 7,
    INVALID_FANIN_COUNT = 8,
    INVALID_FANIN_ID = 9,
    MIXED_CORE_TYPE = 10,
};

struct AicoreReadonlyGraphV0 {
    uint64_t descriptors_address;
    uint64_t payloads_address;
    uint64_t task_count;
    uint64_t task_window_mask;
};

struct AicoreRootInfoV0 {
    int64_t task_id;
    int32_t kernel_id;
    int32_t subtask_slot;
    AicoreRootCoreTypeV0 core_type;
};

using AicoreTaskInfoV0 = AicoreRootInfoV0;

struct AicoreTaskShapeV1 {
    int64_t task_id;
    int32_t kernel_ids[3];
    uint8_t active_mask;
    uint8_t reserved[3];
};

static_assert(sizeof(AicoreReadonlyGraphV0) == 32, "read-only graph view layout changed");
static_assert(offsetof(AicoreReadonlyGraphV0, descriptors_address) == 0, "descriptor address offset changed");
static_assert(offsetof(AicoreReadonlyGraphV0, payloads_address) == 8, "payload address offset changed");
static_assert(sizeof(AicoreRootInfoV0) == 24, "root classification result layout changed");
static_assert(sizeof(AicoreTaskShapeV1) == 24, "task shape result layout changed");

inline __host__ __aicore__ __gm__ uint8_t *
aicore_graph_descriptor_v0(const AicoreReadonlyGraphV0 &graph, int64_t task_id) {
    uint64_t slot = static_cast<uint64_t>(task_id) & graph.task_window_mask;
    return reinterpret_cast<__gm__ uint8_t *>(graph.descriptors_address) +
           slot * AICORE_GRAPH_TASK_DESCRIPTOR_STRIDE_V0;
}
inline __host__ __aicore__ __gm__ uint8_t *
aicore_graph_payload_v0(const AicoreReadonlyGraphV0 &graph, int64_t task_id) {
    uint64_t slot = static_cast<uint64_t>(task_id) & graph.task_window_mask;
    return reinterpret_cast<__gm__ uint8_t *>(graph.payloads_address) + slot * AICORE_GRAPH_TASK_PAYLOAD_STRIDE_V0;
}

inline __host__ __aicore__ int32_t
aicore_graph_fanin_id_v0(const AicoreReadonlyGraphV0 &graph, int64_t task_id, int32_t fanin_index) {
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    return *reinterpret_cast<__gm__ int32_t *>(
        payload + AICORE_GRAPH_FANIN_IDS_OFFSET_V0 + static_cast<uint64_t>(fanin_index) * sizeof(int32_t)
    );
}

inline __host__ __aicore__ AicoreRootStatusV0
aicore_classify_task_shape_v1(const AicoreReadonlyGraphV0 &graph, int64_t task_id, AicoreTaskShapeV1 *shape) {
    if (shape == nullptr) return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
    *shape = {-1,
              {AICORE_GRAPH_INVALID_KERNEL_ID_V0, AICORE_GRAPH_INVALID_KERNEL_ID_V0,
               AICORE_GRAPH_INVALID_KERNEL_ID_V0},
              0,
              {0, 0, 0}};
    if (graph.task_count == 0) return AicoreRootStatusV0::EMPTY;
    if (graph.descriptors_address == 0 || graph.payloads_address == 0 || task_id < 0 ||
        static_cast<uint64_t>(task_id) >= graph.task_count) {
        return AicoreRootStatusV0::INVALID_TASK_COUNT;
    }

    __gm__ uint8_t *descriptor = aicore_graph_descriptor_v0(graph, task_id);
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    uint64_t stored_task_id = *reinterpret_cast<__gm__ uint64_t *>(descriptor + AICORE_GRAPH_TASK_ID_OFFSET_V0);
    if (stored_task_id != static_cast<uint64_t>(task_id)) return AicoreRootStatusV0::INVALID_TASK_ID;

    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
    if (fanin_count < 0 || fanin_count > AICORE_GRAPH_MAX_FANIN_V0) {
        return AicoreRootStatusV0::INVALID_FANIN_COUNT;
    }
    for (int32_t i = 0; i < fanin_count; ++i) {
        int32_t producer = aicore_graph_fanin_id_v0(graph, task_id, i);
        if (producer < 0 || producer >= task_id) return AicoreRootStatusV0::INVALID_FANIN_ID;
        for (int32_t prior = 0; prior < i; ++prior) {
            if (aicore_graph_fanin_id_v0(graph, task_id, prior) == producer) {
                return AicoreRootStatusV0::INVALID_FANIN_ID;
            }
        }
    }

    __gm__ int32_t *kernel_ids = reinterpret_cast<__gm__ int32_t *>(descriptor + AICORE_GRAPH_KERNEL_IDS_OFFSET_V0);
    for (int32_t slot = 0; slot < 3; ++slot) {
        if (kernel_ids[slot] == AICORE_GRAPH_INVALID_KERNEL_ID_V0) continue;
        if (kernel_ids[slot] < 0) return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
        shape->kernel_ids[slot] = kernel_ids[slot];
        shape->active_mask |= static_cast<uint8_t>(1U << slot);
    }
    if (shape->active_mask == 0) return AicoreRootStatusV0::UNSUPPORTED_SHAPE;

    shape->task_id = task_id;
    return AicoreRootStatusV0::OK;
}

inline __host__ __aicore__ AicoreRootStatusV0
aicore_classify_task_v0(const AicoreReadonlyGraphV0 &graph, int64_t task_id, AicoreTaskInfoV0 *task) {
    if (task == nullptr) return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
    *task = {-1, AICORE_GRAPH_INVALID_KERNEL_ID_V0, -1, AicoreRootCoreTypeV0::NONE};
    AicoreTaskShapeV1 shape{};
    AicoreRootStatusV0 status = aicore_classify_task_shape_v1(graph, task_id, &shape);
    if (status != AicoreRootStatusV0::OK) return status;
    if ((shape.active_mask & static_cast<uint8_t>(shape.active_mask - 1)) != 0) {
        return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
    }
    int32_t active_slot = (shape.active_mask & 1U) != 0 ? 0 : ((shape.active_mask & 2U) != 0 ? 1 : 2);

    task->task_id = task_id;
    task->kernel_id = shape.kernel_ids[active_slot];
    task->subtask_slot = active_slot;
    task->core_type = active_slot == 0 ? AicoreRootCoreTypeV0::AIC : AicoreRootCoreTypeV0::AIV;
    return AicoreRootStatusV0::OK;
}

inline __host__ __aicore__ AicoreRootStatusV0
aicore_classify_single_root_v0(const AicoreReadonlyGraphV0 &graph, AicoreRootInfoV0 *root) {
    if (root == nullptr) return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
    *root = {-1, AICORE_GRAPH_INVALID_KERNEL_ID_V0, -1, AicoreRootCoreTypeV0::NONE};
    if (graph.task_count == 0) return AicoreRootStatusV0::EMPTY;
    if (graph.task_count != 1 || graph.descriptors_address == 0 || graph.payloads_address == 0) {
        return AicoreRootStatusV0::INVALID_TASK_COUNT;
    }

    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, 0);
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
    if (fanin_count != 0) return AicoreRootStatusV0::HAS_FANIN;
    return aicore_classify_task_v0(graph, 0, root);
}

inline __host__ __aicore__ AicoreRootStatusV0 aicore_materialize_task_payload_resolved_v0(
    const AicoreReadonlyGraphV0 &graph, const AicoreTaskInfoV0 &task, uint64_t function_bin_address,
    __gm__ PTO2DispatchPayload *dispatch_payload, int32_t block_idx = 0, int32_t block_num = 1
) {
    if (dispatch_payload == nullptr || function_bin_address == 0 || task.task_id < 0 || block_idx < 0 ||
        block_num <= 0 || block_idx >= block_num) {
        return AicoreRootStatusV0::INVALID_CALLABLE;
    }
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task.task_id);
    int32_t tensor_count = *reinterpret_cast<__gm__ int32_t *>(payload + PTO2_TASKPAYLOAD_TENSOR_COUNT_OFFSET);
    int32_t scalar_count = *reinterpret_cast<__gm__ int32_t *>(payload + PTO2_TASKPAYLOAD_SCALAR_COUNT_OFFSET);
    if (tensor_count < 0 || tensor_count > MAX_TENSOR_ARGS || scalar_count < 0 || scalar_count > MAX_SCALAR_ARGS ||
        tensor_count + scalar_count > SPMD_LOCAL_CONTEXT_INDEX) {
        return AicoreRootStatusV0::INVALID_ARGUMENTS;
    }

    dispatch_payload->function_bin_addr = function_bin_address;

    int32_t n = 0;
    for (int32_t i = 0; i < tensor_count; ++i) {
        dispatch_payload->args[n++] = reinterpret_cast<uint64_t>(
            payload + PTO2_TASKPAYLOAD_TENSORS_OFFSET + static_cast<uint64_t>(i) * PTO2_TASKPAYLOAD_TENSOR_STRIDE
        );
    }
    __gm__ uint64_t *scalars = reinterpret_cast<__gm__ uint64_t *>(payload + PTO2_TASKPAYLOAD_SCALARS_OFFSET);
    for (int32_t i = 0; i < scalar_count; ++i)
        dispatch_payload->args[n++] = scalars[i];

    dispatch_payload->src_payload = 0;
    dispatch_payload->local_context.block_idx = block_idx;
    dispatch_payload->local_context.block_num = block_num;
    dispatch_payload->local_context.async_ctx.task_token.raw = static_cast<uint64_t>(task.task_id);
    dispatch_payload->args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload->local_context);
    dispatch_payload->args[PAYLOAD_GLOBAL_CONTEXT_INDEX] =
        reinterpret_cast<uint64_t>(&dispatch_payload->global_context);
    dispatch_payload->global_context.sub_block_id = task.subtask_slot == 2 ? 1 : 0;
    return AicoreRootStatusV0::OK;
}

inline __host__ __aicore__ AicoreRootStatusV0 aicore_materialize_task_payload_v0(
    const AicoreReadonlyGraphV0 &graph, const AicoreTaskInfoV0 &task, uint64_t callable_address,
    __gm__ PTO2DispatchPayload *dispatch_payload
) {
    if (callable_address == 0) return AicoreRootStatusV0::INVALID_CALLABLE;
    __gm__ uint8_t *callable = reinterpret_cast<__gm__ uint8_t *>(callable_address);
    const uint64_t function_bin_address =
        *reinterpret_cast<__gm__ uint64_t *>(callable + AICORE_CORE_CALLABLE_RESOLVED_ADDR_OFFSET_V0);
    return aicore_materialize_task_payload_resolved_v0(graph, task, function_bin_address, dispatch_payload);
}

inline __host__ __aicore__ AicoreRootStatusV0 aicore_materialize_root_payload_v0(
    const AicoreReadonlyGraphV0 &graph, const AicoreRootInfoV0 &root, uint64_t callable_address,
    __gm__ PTO2DispatchPayload *dispatch_payload
) {
    return aicore_materialize_task_payload_v0(graph, root, callable_address, dispatch_payload);
}
