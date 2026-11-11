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
#include "sram_storage.h"
#include "wse_hardware_defaults.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace wse {
void SramStorage::Free::operator()(std::byte *p) const noexcept { std::free(p); }
SramStorage::SramStorage() { configure(DEFAULT_CORES, DEFAULT_PES, DEFAULT_CAPACITIES, DEFAULT_RESERVES); }

void SramStorage::configure(uint32_t cores, uint32_t pes, const uint64_t *capacities, const uint64_t *reserves) {
    std::scoped_lock lock(mutex_);
    if (frozen_) throw std::invalid_argument("WSE configuration is frozen after first use");
    if (!cores || cores > MAX_CORE_COUNT || pes != PE_PER_CORE || !capacities || !reserves)
        throw std::invalid_argument("unsupported WSE Core/PE configuration");
    uint64_t total = 0;
    for (uint32_t i = 0; i < cores; ++i) {
        if (!reserves[i] || reserves[i] >= capacities[i] || capacities[i] % SRAM_ALIGNMENT ||
            reserves[i] % SRAM_ALIGNMENT || capacities[i] > SIZE_MAX - total)
            throw std::invalid_argument("invalid WSE SRAM capacity/reserve/alignment");
        total += capacities[i];
    }
    std::vector<uint64_t> c(capacities, capacities + cores), r(reserves, reserves + cores);
    capacities_.swap(c);
    reserves_.swap(r);
}
std::vector<uint64_t> SramStorage::configuration() const {
    std::scoped_lock lock(mutex_);
    std::vector<uint64_t> out{capacities_.size(), PE_PER_CORE, MAX_CORE_COUNT, frozen_};
    for (size_t i = 0; i < capacities_.size(); ++i) {
        out.push_back(capacities_[i]);
        out.push_back(reserves_[i]);
    }
    return out;
}
uint32_t SramStorage::core_count() const {
    std::scoped_lock lock(mutex_);
    return capacities_.size();
}
void SramStorage::initialize_locked() {
    if (!regions_.empty()) return;
    std::vector<Region> next(capacities_.size());
    for (size_t i = 0; i < next.size(); ++i) {
        void *ptr = nullptr;
        if (posix_memalign(&ptr, SRAM_ALIGNMENT, capacities_[i]) != 0) throw std::bad_alloc();
        next[i].data.reset(static_cast<std::byte *>(ptr));
        next[i].available.emplace(0, capacities_[i] - reserves_[i]);
    }
    regions_.swap(next);
    frozen_ = true;
}
void SramStorage::freeze() {
    std::scoped_lock lock(mutex_);
    initialize_locked();
}
void *SramStorage::allocate(size_t bytes, uint32_t core, size_t alignment) {
    std::scoped_lock lock(mutex_);
    if (!bytes || core >= capacities_.size() || !alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("invalid SRAM allocation size/core/alignment");
    initialize_locked();
    auto &region = regions_[core];
    const auto base = reinterpret_cast<uint64_t>(region.data.get());
    for (const auto &[offset, extent] : region.available) {
        if (base + offset > UINT64_MAX - (alignment - 1)) continue;
        const auto aligned = (base + offset + alignment - 1) & ~(uint64_t{alignment} - 1);
        const auto pad = aligned - base - offset;
        if (pad > extent || bytes > extent - pad) continue;
        auto remaining = region.available;
        remaining.erase(offset);
        if (extent > pad + bytes) remaining.emplace(offset + pad + bytes, extent - pad - bytes);
        const auto inserted = allocations_.emplace(aligned, Allocation{core, offset, pad + bytes, bytes});
        if (!inserted.second) throw std::logic_error("SRAM allocation overlap");
        region.available.swap(remaining);
        return reinterpret_cast<void *>(aligned);
    }
    throw std::runtime_error("SRAM application capacity exhausted on core " + std::to_string(core));
}
void SramStorage::release(void *ptr) {
    std::scoped_lock lock(mutex_);
    const auto it = allocations_.find(reinterpret_cast<uint64_t>(ptr));
    if (it == allocations_.end()) throw std::invalid_argument("SRAM free requires a live allocation base");
    const auto a = it->second;
    auto next = regions_[a.core].available;
    auto inserted = next.emplace(a.offset, a.extent).first;
    if (inserted != next.begin()) {
        auto prev = std::prev(inserted);
        if (prev->first + prev->second == inserted->first) {
            prev->second += inserted->second;
            next.erase(inserted);
            inserted = prev;
        }
    }
    auto following = std::next(inserted);
    if (following != next.end() && inserted->first + inserted->second == following->first) {
        inserted->second += following->second;
        next.erase(following);
    }
    regions_[a.core].available.swap(next);
    allocations_.erase(it);
}
void SramStorage::check_application(uint64_t address, uint64_t bytes) const {
    std::scoped_lock lock(mutex_);
    auto it = allocations_.upper_bound(address);
    if (it != allocations_.begin()) {
        --it;
        if (address - it->first <= it->second.bytes && bytes <= it->second.bytes - (address - it->first)) return;
    }
    throw std::invalid_argument("SRAM access is outside a live application allocation");
}
std::vector<SramAllocation> SramStorage::allocations() const {
    std::scoped_lock lock(mutex_);
    std::vector<SramAllocation> result;
    result.reserve(allocations_.size());
    for (const auto &[base, a] : allocations_)
        result.push_back({base, a.bytes, a.core, 0});
    return result;
}
std::vector<SramRegion> SramStorage::regions() const {
    std::scoped_lock lock(mutex_);
    std::vector<SramRegion> result;
    result.reserve(regions_.size());
    for (size_t i = 0; i < regions_.size(); ++i)
        result.push_back({reinterpret_cast<uint64_t>(regions_[i].data.get()), capacities_[i], reserves_[i]});
    return result;
}
std::vector<uint64_t> SramStorage::info(uint32_t core) const {
    std::scoped_lock lock(mutex_);
    if (core >= capacities_.size()) throw std::invalid_argument("invalid SRAM core");
    uint64_t free = capacities_[core] - reserves_[core], largest = free, used = 0, runtime = 0, base = 0;
    if (!regions_.empty()) {
        free = 0;
        largest = 0;
        for (const auto &[offset, bytes] : regions_[core].available) {
            (void)offset;
            free += bytes;
            largest = std::max(largest, bytes);
        }
        used = capacities_[core] - reserves_[core] - free;
        runtime = regions_[core].runtime_used;
        base = reinterpret_cast<uint64_t>(regions_[core].data.get());
    }
    return {base, capacities_[core], reserves_[core], used, free, largest, runtime};
}
void SramStorage::set_runtime_used(const std::vector<uint64_t> &used) {
    std::scoped_lock lock(mutex_);
    if (used.size() != regions_.size()) throw std::invalid_argument("SRAM runtime accounting shape");
    for (size_t i = 0; i < used.size(); ++i)
        if (used[i] > reserves_[i]) throw std::invalid_argument("SRAM runtime accounting overflow");
    for (size_t i = 0; i < used.size(); ++i)
        regions_[i].runtime_used = used[i];
}
void SramStorage::reset_runtime_used() noexcept {
    std::scoped_lock lock(mutex_);
    for (auto &region : regions_)
        region.runtime_used = 0;
}
size_t SramStorage::committed_bytes() const {
    std::scoped_lock lock(mutex_);
    size_t bytes = 0;
    if (!regions_.empty())
        for (auto capacity : capacities_)
            bytes += capacity;
    return bytes;
}
void SramStorage::clear() {
    std::scoped_lock lock(mutex_);
    allocations_.clear();
    regions_.clear();
}
}  // namespace wse
