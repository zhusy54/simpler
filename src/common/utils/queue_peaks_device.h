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

#include "aicpu/cache_maintenance.h"
#include "utils/queue_peaks.h"

namespace simpler::queue_peaks {
inline void publish_output(const Workspace &workspace, uint64_t address, uint64_t epoch) {
    if (address == 0 || epoch == 0) return;
    auto *report = reinterpret_cast<Report *>(address);
    workspace.snapshot(*report);
    cache_flush_range(report, sizeof(*report));
    report->epoch = epoch;
    cache_flush_range(&report->epoch, sizeof(report->epoch));
}
}  // namespace simpler::queue_peaks
