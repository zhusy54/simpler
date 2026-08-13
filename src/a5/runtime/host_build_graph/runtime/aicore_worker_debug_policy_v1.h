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

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __host__
#define __host__
#endif

enum class AicoreWorkerDebugEventV1 {
    SEED_INITIALIZED,
    TICKET_CLAIMED,
    TASK_COMPLETED,
    WAITING_PRODUCER_CHANGED,
    CURSOR_EXHAUSTED,
    DRAINING,
};

inline __host__ __aicore__ constexpr bool aicore_worker_debug_requires_publish_v1(AicoreWorkerDebugEventV1 event) {
    return event == AicoreWorkerDebugEventV1::WAITING_PRODUCER_CHANGED ||
           event == AicoreWorkerDebugEventV1::CURSOR_EXHAUSTED || event == AicoreWorkerDebugEventV1::DRAINING;
}
