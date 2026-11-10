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
#include "common/hardware_spec.h"
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace wse {
class SramStorage {
public:
    SramStorage();
    void configure(uint32_t cores, uint32_t pes, const uint64_t *capacities, const uint64_t *reserves);
    std::vector<uint64_t> configuration() const;
    void freeze();
    uint32_t core_count() const;
    void *allocate(size_t bytes, uint32_t core, size_t alignment);
    void release(void *ptr);
    bool touches(uint64_t address, uint64_t bytes) const;
    void check_application(uint64_t address, uint64_t bytes) const;
    std::vector<SramAllocation> allocations() const;
    std::vector<SramRegion> regions() const;
    std::vector<uint64_t> info(uint32_t core) const;
    void set_runtime_used(const std::vector<uint64_t> &used);
    void reset_runtime_used() noexcept;
    void clear();

private:
    struct Free {
        void operator()(std::byte *p) const noexcept;
    };
    struct Region {
        std::unique_ptr<std::byte, Free> data;
        std::map<uint64_t, uint64_t> available;
        uint64_t runtime_used = 0;
    };
    struct Allocation {
        uint32_t core;
        uint64_t offset;
        uint64_t extent;
        uint64_t bytes;
    };
    mutable std::mutex mutex_;
    std::vector<uint64_t> capacities_, reserves_;
    std::vector<Region> regions_;
    std::map<uint64_t, Allocation> allocations_;
    bool frozen_ = false;
    void initialize_locked();
};
}  // namespace wse
