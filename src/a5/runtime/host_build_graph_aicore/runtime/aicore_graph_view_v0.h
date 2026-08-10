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

static_assert(sizeof(AicoreReadonlyGraphV0) == 32, "read-only graph view layout changed");
static_assert(offsetof(AicoreReadonlyGraphV0, descriptors_address) == 0, "descriptor address offset changed");
static_assert(offsetof(AicoreReadonlyGraphV0, payloads_address) == 8, "payload address offset changed");
static_assert(sizeof(AicoreRootInfoV0) == 24, "root classification result layout changed");

inline __host__ __aicore__ __gm__ uint8_t *aicore_graph_descriptor_v0(
    const AicoreReadonlyGraphV0 &graph, int64_t task_id
) {
    uint64_t slot = static_cast<uint64_t>(task_id) & graph.task_window_mask;
    return reinterpret_cast<__gm__ uint8_t *>(graph.descriptors_address) +
           slot * AICORE_GRAPH_TASK_DESCRIPTOR_STRIDE_V0;
}
inline __host__ __aicore__ __gm__ uint8_t *aicore_graph_payload_v0(
    const AicoreReadonlyGraphV0 &graph, int64_t task_id
) {
    uint64_t slot = static_cast<uint64_t>(task_id) & graph.task_window_mask;
    return reinterpret_cast<__gm__ uint8_t *>(graph.payloads_address) + slot * AICORE_GRAPH_TASK_PAYLOAD_STRIDE_V0;
}

inline __host__ __aicore__ AicoreRootStatusV0
aicore_classify_single_root_v0(const AicoreReadonlyGraphV0 &graph, AicoreRootInfoV0 *root) {
    if (root == nullptr) return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
    *root = {-1, AICORE_GRAPH_INVALID_KERNEL_ID_V0, -1, AicoreRootCoreTypeV0::NONE};
    if (graph.task_count == 0) return AicoreRootStatusV0::EMPTY;
    if (graph.task_count != 1 || graph.descriptors_address == 0 || graph.payloads_address == 0) {
        return AicoreRootStatusV0::INVALID_TASK_COUNT;
    }

    __gm__ uint8_t *descriptor = aicore_graph_descriptor_v0(graph, 0);
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, 0);
    uint64_t task_id = *reinterpret_cast<__gm__ uint64_t *>(descriptor + AICORE_GRAPH_TASK_ID_OFFSET_V0);
    if (task_id != 0) return AicoreRootStatusV0::INVALID_TASK_ID;
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
    if (fanin_count != 0) return AicoreRootStatusV0::HAS_FANIN;

    __gm__ int32_t *kernel_ids =
        reinterpret_cast<__gm__ int32_t *>(descriptor + AICORE_GRAPH_KERNEL_IDS_OFFSET_V0);
    int32_t active_slot = -1;
    for (int32_t slot = 0; slot < 3; ++slot) {
        if (kernel_ids[slot] == AICORE_GRAPH_INVALID_KERNEL_ID_V0) continue;
        if (active_slot != -1) return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
        active_slot = slot;
        root->kernel_id = kernel_ids[slot];
    }
    if (active_slot == -1 || root->kernel_id < 0) return AicoreRootStatusV0::UNSUPPORTED_SHAPE;

    root->task_id = 0;
    root->subtask_slot = active_slot;
    root->core_type = active_slot == 0 ? AicoreRootCoreTypeV0::AIC : AicoreRootCoreTypeV0::AIV;
    return AicoreRootStatusV0::OK;
}

inline __host__ __aicore__ AicoreRootStatusV0 aicore_materialize_root_payload_v0(
    const AicoreReadonlyGraphV0 &graph, const AicoreRootInfoV0 &root, uint64_t callable_address,
    __gm__ PTO2DispatchPayload *dispatch_payload
) {
    if (dispatch_payload == nullptr || callable_address == 0 || root.task_id < 0) {
        return AicoreRootStatusV0::INVALID_CALLABLE;
    }
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, root.task_id);
    int32_t tensor_count = *reinterpret_cast<__gm__ int32_t *>(payload + PTO2_TASKPAYLOAD_TENSOR_COUNT_OFFSET);
    int32_t scalar_count = *reinterpret_cast<__gm__ int32_t *>(payload + PTO2_TASKPAYLOAD_SCALAR_COUNT_OFFSET);
    if (tensor_count < 0 || tensor_count > MAX_TENSOR_ARGS || scalar_count < 0 || scalar_count > MAX_SCALAR_ARGS ||
        tensor_count + scalar_count > SPMD_LOCAL_CONTEXT_INDEX) {
        return AicoreRootStatusV0::INVALID_ARGUMENTS;
    }

    __gm__ uint8_t *callable = reinterpret_cast<__gm__ uint8_t *>(callable_address);
    dispatch_payload->function_bin_addr =
        *reinterpret_cast<__gm__ uint64_t *>(callable + AICORE_CORE_CALLABLE_RESOLVED_ADDR_OFFSET_V0);
    if (dispatch_payload->function_bin_addr == 0) return AicoreRootStatusV0::INVALID_CALLABLE;

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
    dispatch_payload->local_context.block_idx = 0;
    dispatch_payload->local_context.block_num = 1;
    dispatch_payload->local_context.async_ctx.task_token.raw = static_cast<uint64_t>(root.task_id);
    dispatch_payload->args[PAYLOAD_LOCAL_CONTEXT_INDEX] =
        reinterpret_cast<uint64_t>(&dispatch_payload->local_context);
    dispatch_payload->args[PAYLOAD_GLOBAL_CONTEXT_INDEX] =
        reinterpret_cast<uint64_t>(&dispatch_payload->global_context);
    dispatch_payload->global_context.sub_block_id = root.subtask_slot == 2 ? 1 : 0;
    return AicoreRootStatusV0::OK;
}
