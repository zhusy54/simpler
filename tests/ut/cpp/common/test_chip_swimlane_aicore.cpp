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

#include "inner_kernel.h"
#undef OUT_OF_ORDER_STORE_BARRIER
#include "aicore/chip_swimlane_collector_aicore.h"
#include "aicore_task_profiling_v0.h"

#include <gtest/gtest.h>

#include <cstdint>

TEST(ChipSwimlaneAicoreTest, CommitUsesReservedBufferGeneration) {
    ChipSwimlaneAicoreTaskBuffer first{};
    ChipSwimlaneAicoreTaskBuffer second{};
    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(&first);
    head.current_buf_seq = 0;

    ChipSwimlaneAicoreLocalState local{};
    local.cached_buf_seq = UINT32_MAX;

    ChipSwimlaneAicoreTaskRecord *reserved = chip_swimlane_aicore_reserve_task_record(&head, &local);
    ASSERT_EQ(reserved, &first.records[0]);

    head.current_buf_ptr = reinterpret_cast<uint64_t>(&second);
    head.current_buf_seq = 1;

    chip_swimlane_aicore_commit_task_record(reserved, 0x1234, 17, 100, 120, 180);

    EXPECT_EQ(first.records[0].task_token_raw, 0x1234u);
    EXPECT_EQ(first.records[0].reg_task_id, 17u);
    EXPECT_EQ(first.records[0].start_time, 120u);
    EXPECT_EQ(first.records[0].end_time, 180u);
    EXPECT_EQ(first.records[0].receive_to_start_cycles, 20u);
    EXPECT_EQ(second.records[0].task_token_raw, 0u);

    ChipSwimlaneAicoreTaskRecord *next = chip_swimlane_aicore_reserve_task_record(&head, &local);
    EXPECT_EQ(next, &second.records[0]);
}

TEST(ChipSwimlaneAicoreTest, SidecarCountsOverflowWithoutWritingPastCapacity) {
    ChipSwimlaneAicoreTaskBuffer buffer{};
    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(&buffer);
    head.current_buf_seq = 0;

    AicoreWorkerContextV0 worker{};
    AicoreTaskProfilingStateV0 profiling{};
    aicore_task_profiling_init_v0(&profiling, true, &head);
    for (uint32_t task_id = 0; task_id < PLATFORM_AICORE_BUFFER_SIZE; ++task_id) {
        auto *record = aicore_task_profiling_reserve_v0(&profiling, &worker);
        ASSERT_NE(record, nullptr);
        aicore_task_profiling_commit_v0(record, &worker, task_id, 10 + task_id, 20 + task_id, 30 + task_id);
    }

    EXPECT_EQ(aicore_task_profiling_reserve_v0(&profiling, &worker), nullptr);
    EXPECT_EQ(worker.dfx_reserved[AICORE_PROFILE_ATTEMPTED_INDEX_V0], PLATFORM_AICORE_BUFFER_SIZE + 1);
    EXPECT_EQ(worker.dfx_reserved[AICORE_PROFILE_WRITTEN_INDEX_V0], PLATFORM_AICORE_BUFFER_SIZE);
    EXPECT_EQ(worker.dfx_reserved[AICORE_PROFILE_DROPPED_INDEX_V0], 1u);
    EXPECT_EQ(buffer.records[PLATFORM_AICORE_BUFFER_SIZE - 1].task_token_raw, PLATFORM_AICORE_BUFFER_SIZE - 1);
}
