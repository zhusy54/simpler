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

#include <gtest/gtest.h>
#include "utils/queue_peaks.h"

TEST(QueuePeakCounters, Full65536CapacityAndDistinctClassPeaks) {
    simpler::queue_peaks::Counters counters;
    counters.push(65536, 65536, true);
    counters.pop(65536, 65536);
    counters.push(65536, 0, true);
    counters.pop(65536);
    EXPECT_EQ(counters.current.load(), 0u);
    EXPECT_EQ(counters.peak.load(), 65536u);
    EXPECT_EQ(counters.spmd_peak.load(), 65536u);
    EXPECT_EQ(counters.normal_peak.load(), 65536u);
    counters.reset();
    EXPECT_EQ(counters.peak.load(), 0u);
}
