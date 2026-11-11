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
#include <cstdint>

namespace wse::temporary {
inline constexpr uint64_t SHARED_TILE_KEY_PREFIX = 0x5753455300000000ULL;
inline constexpr uint64_t SHARED_TILE_KEY_MASK = 0xFFFFFFFF00000000ULL;

inline void end_block_operations(uint32_t core, void (*release)(uint32_t, uint64_t, uint64_t)) {
    release(core, SHARED_TILE_KEY_PREFIX, SHARED_TILE_KEY_MASK);
}
}  // namespace wse::temporary
