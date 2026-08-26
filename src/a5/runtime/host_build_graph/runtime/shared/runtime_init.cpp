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
/** Orchestrator initialization. */

#include <stdlib.h>
#include <string.h>

#include "orchestrator.h"
#include "runtime_core.h"
#include "task_allocator.h"
#include "shared_memory.h"
#include "tensormap.h"
bool OrchestratorState::init(void *sm_base, void *gm_heap, uint64_t heap_size, uint64_t max_tasks) {
    auto *orch = this;
    *orch = OrchestratorState{};

    always_assert(max_tasks > 0);

    orch->sm_header = reinterpret_cast<SharedMemoryHeader *>(sm_base);
    orch->fatal = false;
    auto *orch_err = sm_layout::orch_error_code_addr(sm_base);

    orch->task_allocator.init(static_cast<int32_t>(max_tasks), gm_heap, heap_size, orch_err);

    // The mirror's argument pools. Offset arithmetic on the same base as sm_header,
    // so it holds for whichever SM this orchestrator was pointed at. The cursors
    // reset with the rest of the state above.
    auto *sm_bytes = static_cast<char *>(sm_base);
    const auto pools = sm_layout::segment_offsets(sm_layout::mirror_extents(max_tasks));
    orch->fanin_pool = reinterpret_cast<int32_t *>(sm_bytes + pools.fanin_pool);
    orch->tensor_pool = reinterpret_cast<simpler::hbg::Tensor *>(sm_bytes + pools.tensor_pool);
    orch->scalar_pool = reinterpret_cast<uint64_t *>(sm_bytes + pools.scalar_pool);

    // Polling: no fanin-spill pool — producer ids are inline on the payload.
    const auto slots = static_cast<size_t>(max_tasks);
    orch->fanin_seen_epoch.reset(new (std::nothrow) uint32_t[slots]);
    if (orch->fanin_seen_epoch == nullptr) {
        LOG_ERROR("Orchestrator scratch allocation failed (max_tasks=%" PRIu64 ")", max_tasks);
        return false;
    }
    memset(orch->fanin_seen_epoch.get(), 0, slots * sizeof(uint32_t));

    if (!orch->tensor_map.init_default(static_cast<int32_t>(max_tasks))) {
        return false;
    }

    orch->scope_stack_top = -1;
    orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;

    return true;
}
