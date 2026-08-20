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
/**
 * Runtime Builder - rt2 Implementation (host_build_graph: Host Orchestration)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions for rt2 runtime.
 * The HOST runs the orchestrator to completion, populates shared memory + the
 * prebuilt arena, and H2Ds the image; the device boots scheduler-only.
 *
 * init_runtime_impl:
 *   - Converts host tensor pointers to device pointers (all inputs copied H2D;
 *     only OUTPUT/INOUT tensors are copied back D2H)
 *   - dlopens the orchestration SO on the host and runs it to build the graph
 *   - Sets up runtime state for host orchestration
 *
 * validate_runtime_impl:
 *   - Copies OUTPUT/INOUT tensors back from device to host (read-only inputs
 *     are skipped)
 *   - Frees device memory
 */

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "../common/pto_runtime_status.h"
#include "../runtime/common.h"
#include "../runtime/aicore_graph_view_v0.h"
#include "../runtime/dep_gen_host_graph.h"
#include "../runtime/host_tensor_access.h"
#include "../runtime/pto_orchestrator.h"
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/pto_types.h"
#include "../runtime/runtime.h"
#include "../../../../common/runtime_status/error_log.h"
#include "../../../../common/task_interface/call_config.h"
#include "../../../../common/worker/pto_runtime_c_api.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/chip_swimlane_policy.h"
#include "common/unified_log.h"
#include "common/strace.h"
#include "host/platform_compile_info.h"
#include "utils/device_arena.h"
#include "prepare_callable_common.h"

extern "C" const PipelineContract *get_pipeline_contract(void) {
    // Host orchestration materializes this run's own graph into the image it
    // uploads, so every device-resident region carries per-run content.
    static const PipelineContract contract = {
        PTO_PIPELINE_CONTRACT_ABI_VERSION,
        5,
        2,
        {
            {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_GM_SM, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
        },
    };
    return &contract;
}

extern "C" int concurrent_native_prepare_supported_impl(void) {
    // HBG can materialize a complete graph into the lease-selected unpublished
    // arena bank. The common C API keeps collector-bearing configurations on
    // the sequential path until their state is per-epoch.
    return 1;
}

extern "C" int validate_chip_swimlane_level_impl(int32_t level) {
    if (level == 0 || level == static_cast<int32_t>(ChipSwimlaneLevel::AICORE_TIMING)) return 0;
    LOG_ERROR("A5 HBG AICore scheduler supports enable_chip_swimlane values 0 and 1 only, got %d", level);
    return -1;
}

extern "C" bool strict_chip_swimlane_validation_impl() { return true; }

extern "C" void configure_runtime_diagnostics_impl(Runtime *runtime, const CallConfig *config) {
    if (runtime == nullptr || config == nullptr) return;
    runtime->set_aicore_scheduler_trace(
        config->enable_chip_swimlane == static_cast<int32_t>(ChipSwimlaneLevel::AICORE_TIMING), config->output_prefix
    );
}

// RuntimeEnv (call_config.h) is the cross-runtime ABI for per-ring config and
// carries RUNTIME_ENV_RING_COUNT slots, shared with tensormap_and_ringbuffer.
// host_build_graph is single-ring (PTO2_MAX_RING_DEPTH == 1) and reads only the
// first slot; it must fit within the ABI's slot budget, not equal it.
static_assert(PTO2_MAX_RING_DEPTH <= RUNTIME_ENV_RING_COUNT, "PTO2 runtime ring depth must fit RuntimeEnv ring slots");
static_assert(sizeof(PTO2TaskDescriptor) == AICORE_GRAPH_TASK_DESCRIPTOR_STRIDE_V0);
static_assert(sizeof(PTO2TaskPayload) == AICORE_GRAPH_TASK_PAYLOAD_STRIDE_V0);
static_assert(offsetof(PTO2TaskDescriptor, task_id) == AICORE_GRAPH_TASK_ID_OFFSET_V0);
static_assert(offsetof(PTO2TaskDescriptor, kernel_id) == AICORE_GRAPH_KERNEL_IDS_OFFSET_V0);
static_assert(offsetof(PTO2TaskPayload, fanin_count) == AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
static_assert(offsetof(CoreCallable, resolved_addr_) == AICORE_CORE_CALLABLE_RESOLVED_ADDR_OFFSET_V0);

// Helper: return current time in milliseconds
static int64_t _now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

static bool is_power_of_2_u64(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

template <typename T>
static std::string format_ring_array(const T (&values)[PTO2_MAX_RING_DEPTH]) {
    std::string out = "[";
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; ++r) {
        if (r != 0) {
            out += ", ";
        }
        out += std::to_string(values[r]);
    }
    out += "]";
    return out;
}

static std::string trim_copy(const std::string &input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(begin, end - begin);
}

static bool parse_uint_token(
    const char *name, const std::string &raw, uint64_t min_val, uint64_t max_val, bool require_power_of_2, uint64_t *out
) {
    std::string token = trim_copy(raw);
    if (token.empty()) {
        LOG_WARN("%s has an empty value in '%s', ignored", name, raw.c_str());
        return false;
    }

    if (token[0] == '-') {
        LOG_WARN("%s=%s invalid (must be a non-negative integer), ignored", name, token.c_str());
        return false;
    }
    char *endptr = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(token.c_str(), &endptr, 10);
    if (errno == ERANGE || endptr == token.c_str() || *endptr != '\0') {
        LOG_WARN("%s=%s invalid (must be a non-negative integer), ignored", name, token.c_str());
        return false;
    }
    uint64_t val = static_cast<uint64_t>(parsed);

    if (val < min_val || val > max_val) {
        LOG_WARN(
            "%s=%s invalid (must be in [%" PRIu64 ", %" PRIu64 "]), ignored", name, token.c_str(), min_val, max_val
        );
        return false;
    }
    if (require_power_of_2 && !is_power_of_2_u64(val)) {
        LOG_WARN("%s=%s invalid (must be a power of 2), ignored", name, token.c_str());
        return false;
    }
    *out = val;
    return true;
}

static void apply_env_ring_values(
    const char *name, uint64_t min_val, uint64_t max_val, bool require_power_of_2, uint64_t out[PTO2_MAX_RING_DEPTH]
) {
    const char *env = std::getenv(name);
    if (!env) return;

    std::string text(env);
    if (text.find(',') == std::string::npos) {
        uint64_t value = 0;
        if (!parse_uint_token(name, text, min_val, max_val, require_power_of_2, &value)) {
            return;
        }
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            out[r] = value;
        }
        return;
    }

    uint64_t parsed[PTO2_MAX_RING_DEPTH]{};
    size_t pos = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        size_t comma = text.find(',', pos);
        std::string token = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!parse_uint_token(name, token, min_val, max_val, require_power_of_2, &parsed[r])) {
            return;
        }
        if (comma == std::string::npos) {
            if (r != PTO2_MAX_RING_DEPTH - 1) {
                LOG_WARN(
                    "%s=%s invalid (expected exactly %d comma-separated values), ignored", name, env,
                    PTO2_MAX_RING_DEPTH
                );
                return;
            }
            pos = text.size();
        } else {
            pos = comma + 1;
        }
    }
    if (pos < text.size() || (!text.empty() && text.back() == ',')) {
        LOG_WARN("%s=%s invalid (expected exactly %d comma-separated values), ignored", name, env, PTO2_MAX_RING_DEPTH);
        return;
    }
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        out[r] = parsed[r];
    }
}

// ring_task_window / ring_heap / ring_dep_pool point into the #pragma pack(1)
// RuntimeEnv wire struct (call_config.h), so the enclosing object provides no
// uint64_t alignment guarantee. Reading them as `base[idx]` can be an unaligned
// 8-byte load: UB, and fatal under UBSan (-fsanitize=alignment). Copy
// the bytes out instead. A null base means "no per-task overrides" -> 0 (unset).
static uint64_t read_ring_override(const uint64_t *base, int idx) {
    if (base == nullptr) {
        return 0;
    }
    uint64_t value;
    std::memcpy(&value, base + idx, sizeof(value));
    return value;
}

// Each of ring_task_window / ring_heap is a per-ring array of PTO2_MAX_RING_DEPTH
// entries (0 = unset). Precedence per ring: per-task entry > PTO2_RING_* env value
// > compile-time default. A "size all rings the same" request arrives already
// broadcast to every entry by the caller. (Polling has no dep_pool, so the former
// PTO2_RING_DEP_POOL knob is gone.)
static bool resolve_ring_config(
    const uint64_t *ring_task_window, const uint64_t *ring_heap, uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH],
    uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH]
) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        eff_task_window_sizes[r] = PTO2_TASK_WINDOW_SIZE;
        eff_heap_sizes[r] = PTO2_HEAP_SIZE;
    }

    apply_env_ring_values("PTO2_RING_TASK_WINDOW", 4, static_cast<uint64_t>(INT32_MAX), true, eff_task_window_sizes);
    apply_env_ring_values("PTO2_RING_HEAP", 1024, std::numeric_limits<uint64_t>::max(), false, eff_heap_sizes);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        const uint64_t task_window_override = read_ring_override(ring_task_window, r);
        const uint64_t heap_override = read_ring_override(ring_heap, r);
        if (task_window_override != 0) {
            eff_task_window_sizes[r] = task_window_override;
        }
        if (heap_override != 0) {
            eff_heap_sizes[r] = heap_override;
        }

        if (eff_task_window_sizes[r] < 4 || eff_task_window_sizes[r] > static_cast<uint64_t>(INT32_MAX) ||
            !is_power_of_2_u64(eff_task_window_sizes[r])) {
            LOG_ERROR(
                "ring_task_window[%d]=%" PRIu64 " must be a power of 2 in [4, INT32_MAX]", r, eff_task_window_sizes[r]
            );
            return false;
        }
        if (eff_heap_sizes[r] < 1024) {
            LOG_ERROR("ring_heap[%d]=%" PRIu64 " must be >= 1024", r, eff_heap_sizes[r]);
            return false;
        }
    }

    return true;
}

static int32_t pto2_read_runtime_status(Runtime *runtime, const HostApi *api, PTO2SharedMemoryHeader *host_header) {
    if (runtime == nullptr || api == nullptr || host_header == nullptr) {
        return 0;
    }

    void *pto2_sm = runtime->get_gm_sm_ptr();
    if (pto2_sm == nullptr) {
        return 0;
    }

    int hdr_rc = api->copy_from_device(host_header, pto2_sm, sizeof(PTO2SharedMemoryHeader));
    if (hdr_rc != 0) {
        LOG_WARN("Failed to copy PTO2 header from device");
        return 0;
    }

    int32_t orch_error_code = host_header->orch_error_code.load(std::memory_order_relaxed);
    int32_t sched_error_code = host_header->sched_error_code.load(std::memory_order_relaxed);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

namespace {

// host_build_graph is host-orchestration-first: the HOST dlopens the
// orchestration .so and runs it to completion. The shared memory + arena carry
// host-DDR cross-task pointers (slot_state.task/payload,
// payload.fanin_inline_slot_states[], dep_pool/ready queues); the host relocates them to
// their final device addresses (relocate_host_orch_image, below) BEFORE the H2D
// copy, so the device receives a fully device-addressed image and schedules
// only — no on-device pointer fixup.

bool write_all_bytes(int fd, const uint8_t *data, size_t size) {
    size_t total = 0;
    while (total < size) {
        ssize_t w = write(fd, data + total, size - total);
        if (w <= 0) {
            return false;
        }
        total += static_cast<size_t>(w);
    }
    return true;
}

// Materialize the orchestration .so bytes to a temp file so it can be dlopen'd
// on the host (dlopen needs a real path + the exec bit).
bool create_orch_so_tempfile(const uint8_t *data, size_t size, std::string *out_path) {
    char tmpl[] = "/tmp/orch_so_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        return false;
    }
    if (fchmod(fd, 0755) != 0) {
        close(fd);
        unlink(tmpl);
        return false;
    }
    bool ok = write_all_bytes(fd, data, size);
    if (close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        unlink(tmpl);
        return false;
    }
    *out_path = tmpl;
    return true;
}

// The orchestration .so exports these (PTO2 submit_task form).
typedef void (*OrchestrationEntryFunc)(const ChipTaskArgs &);
typedef void (*OrchestrationBindFunc)(PTO2Runtime *);
typedef PTO2OrchestrationConfig (*OrchestrationConfigFunc)(const ChipTaskArgs &);

// Resolved orchestration .so entry points. register_callable_impl allocates one
// of these (the entry, plus the .so's own framework_bind_runtime, which sets
// the .so-private g_current_runtime its inline rt_submit_* read) and stores its
// pointer in CallableArtifacts::host_orch_func_ptr. Owned for the callable's
// lifetime alongside host_dlopen_handle.
struct HostOrchEntryPoints {
    OrchestrationEntryFunc entry{nullptr};
    OrchestrationBindFunc bind{nullptr};
    OrchestrationConfigFunc config{nullptr};
};

// Run the orchestrator on the host. `rt` was built with its scheduler half
// pointing at the device SM; here we re-point ONLY the orchestrator half at a
// host SM mirror, run the orchestration entry against it, latch the submitted
// task count, and H2D the populated SM to the device (the device scheduler
// reads task descriptors from there). The device never dereferences the
// orchestrator's SM pointers, so leaving them host-side is safe. Returns the
// total task count (>= 0) on success, or -1 on failure.
// host_build_graph host-orch: the orchestrator built the task graph in a host
// SM mirror and (when wiring is folded into submit) the fanout adjacency in the
// host arena, storing host-DDR addresses into the cross-task pointers. Relocate
// them to their FINAL device addresses here on the host, BEFORE the SM/arena are
// copied to the device — so the device receives a fully device-addressed image
// and boots scheduler-only with no on-device pointer fixup.
//
// Relocated pointers span TWO regions with DIFFERENT deltas: the SM block
// (slot_state.task/.payload, fanin_inline_slot_states[], dep-entry.slot_state,
// ready-queue slot.slot_state) and the arena block (slot_state.fanout_head,
// dep-entry.next point into the SM but live in the arena).
// Rather than track which delta each field needs, reloc() classifies every
// pointer by the region it points INTO and applies that region's delta; foreign
// and null pointers pass through untouched. The fanout adjacency is wired inline
// during host submit, so dep_pool/ready are already populated here.
//
// The orchestrator's own task-allocator pointers are intentionally NOT relocated
// (the device runs scheduler-only and never dereferences them, and must not call
// rt_orchestration_done — the host already did). Multi-fanin spill is not yet
// relocated; a task exceeding PTO2_FANIN_INLINE_CAP producers latches fatal here
// (returns false) rather than shipping un-relocated host pointers to the device.
// Returns false on any unrelocatable pointer so the caller can fail the prepare.
static bool relocate_host_orch_image(
    PTO2SharedMemoryHandle &host_sm_handle, [[maybe_unused]] PTO2Runtime *rt, uint64_t host_sm, uint64_t sm_size,
    int64_t sm_delta, uint64_t host_arena, uint64_t arena_size, int64_t arena_delta
) {
    // host_build_graph is single-ring; the loops below iterate the lone ring and
    // index header->ring (singular). If the ring depth ever grows, those loops
    // would relocate the same ring N times (applying the delta repeatedly =
    // corruption), so pin the assumption here.
    static_assert(PTO2_MAX_RING_DEPTH == 1, "relocate_host_orch_image assumes a single ring");

    // SM and arena windows must not overlap — reloc classifies a pointer by
    // which window it falls in, so an overlap would misclassify and apply the
    // wrong delta. Both are independent malloc-backed host buffers in practice;
    // assert it so a future shared-buffer layout can't silently corrupt.
    if (!(host_sm + sm_size <= host_arena || host_arena + arena_size <= host_sm)) {
        LOG_ERROR(
            "host-orch: SM window [%#lx,+%#lx) overlaps arena window [%#lx,+%#lx); cannot relocate", host_sm, sm_size,
            host_arena, arena_size
        );
        return false;
    }

    bool ok = true;
    auto reloc = [&](auto *&p) {
        using Ptr = std::remove_reference_t<decltype(p)>;
        uint64_t v = reinterpret_cast<uint64_t>(p);
        if (v == 0) {
            return;
        }
        if (v >= host_sm && v < host_sm + sm_size) {
            p = reinterpret_cast<Ptr>(static_cast<uintptr_t>(v + sm_delta));
        } else if (v >= host_arena && v < host_arena + arena_size) {
            p = reinterpret_cast<Ptr>(static_cast<uintptr_t>(v + arena_delta));
        } else {
            // A non-null pointer in neither window is an external/host address
            // the device would dereference verbatim after H2D. No field should
            // legitimately carry one; latch fatal rather than ship a host VA to
            // the device (silent AICPU corruption otherwise).
            LOG_ERROR("host-orch: pointer %#lx is outside both SM and arena windows; cannot relocate for device", v);
            ok = false;
        }
    };

    PTO2SharedMemoryHeader *header = host_sm_handle.header;
    if (header != nullptr) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            PTO2SharedMemoryRingHeader &ring = header->ring;
            int32_t count = ring.fc.current_task_index.load(std::memory_order_acquire);
            for (int32_t slot = 0; slot < count; slot++) {
                PTO2TaskSlotState *ss = &ring.slot_states[slot];
                // Polling: fanin is a flat array of position-independent local-id
                // integers on the payload, so only the two per-slot arena/SM
                // pointers need relocating. There is no fanout_head/dep_pool graph
                // and no host-seeded ready queue (the device boot scan classifies),
                // so those relocation passes are gone.
                reloc(ss->task);
                reloc(ss->payload);
            }
        }
    }
    return ok;
}

void release_aicore_sidecar(Runtime *runtime, const HostApi *api) {
    if (runtime != nullptr && api != nullptr && runtime->aicore_sidecar_allocation != nullptr) {
        api->device_free(runtime->aicore_sidecar_allocation);
        runtime->aicore_sidecar_base = nullptr;
        runtime->aicore_sidecar_allocation = nullptr;
        runtime->aicore_sidecar_allocation_size = 0;
        runtime->aicore_sidecar_layout = {};
    }
}

bool append_aicore_scheduler_trace(
    const Runtime *runtime, const AicoreRunControlV1 *run_control, const AicpuCoreLifecycleTraceV1 *lifecycle_traces,
    const AicoreTaskTraceCellV1 *traces, const AicoreTaskControlV1 *task_controls, const AicoreWorkerContextV1 *contexts
) {
    if (runtime == nullptr || run_control == nullptr || lifecycle_traces == nullptr || traces == nullptr ||
        task_controls == nullptr || contexts == nullptr || !runtime->aicore_scheduler_trace_enabled()) {
        return true;
    }
    const std::filesystem::path path =
        std::filesystem::path(runtime->aicore_scheduler_trace_output_prefix()) / "chip_swimlane_records.json";
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        LOG_ERROR("A5 HBG AICore scheduler: profiling JSON is unavailable at %s", path.c_str());
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const size_t object_end = json.find_last_of('}');
    if (object_end == std::string::npos) {
        LOG_ERROR("A5 HBG AICore scheduler: profiling JSON has no root object terminator");
        return false;
    }

    std::ostringstream phases;
    phases << ",\n  \"aicore_scheduler_phases\": [";
    bool first = true;
    auto emit = [&](uint64_t worker_id, uint64_t core_type, uint64_t task_id, const char *phase, uint64_t start_cycles,
                    uint64_t end_cycles) {
        if (end_cycles < start_cycles || (start_cycles == 0 && end_cycles == 0)) return;
        phases << (first ? "\n" : ",\n") << "    {\"worker_id\": " << worker_id << ", \"core_type\": " << core_type
               << ", \"task_id\": " << task_id << ", \"phase\": \"" << phase << "\", \"start_cycles\": " << start_cycles
               << ", \"end_cycles\": " << end_cycles << "}";
        first = false;
    };
    auto emit_termination = [&](uint64_t worker_id, uint64_t core_type, const char *phase, uint64_t start_cycles,
                                uint64_t end_cycles, uint64_t atomic_count, uint64_t poll_count, uint64_t poll_cycles,
                                uint64_t backoff_cycles) {
        if (end_cycles < start_cycles || (start_cycles == 0 && end_cycles == 0)) return;
        phases << (first ? "\n" : ",\n") << "    {\"worker_id\": " << worker_id << ", \"core_type\": " << core_type
               << ", \"task_id\": " << UINT64_MAX << ", \"phase\": \"" << phase
               << "\", \"start_cycles\": " << start_cycles << ", \"end_cycles\": " << end_cycles;
        if (atomic_count != 0) phases << ", \"atomic_count\": " << atomic_count;
        if (poll_count != 0 || poll_cycles != 0 || backoff_cycles != 0) {
            phases << ", \"poll_count\": " << poll_count << ", \"poll_cycles\": " << poll_cycles
                   << ", \"backoff_cycles\": " << backoff_cycles;
        }
        phases << "}";
        first = false;
    };
    auto emit_completion_enqueue = [&](uint64_t worker_id, uint64_t core_type, uint64_t task_id, uint64_t start_cycles,
                                       uint64_t end_cycles, uint64_t completion_id, uint64_t inbox_index) {
        if (end_cycles < start_cycles || (start_cycles == 0 && end_cycles == 0)) return;
        phases << (first ? "\n" : ",\n") << "    {\"worker_id\": " << worker_id << ", \"core_type\": " << core_type
               << ", \"task_id\": " << task_id
               << ", \"phase\": \"CompletionEnqueue\", \"start_cycles\": " << start_cycles
               << ", \"end_cycles\": " << end_cycles << ", \"completion_id\": " << completion_id
               << ", \"inbox_index\": " << inbox_index << "}";
        first = false;
    };
    std::vector<std::vector<const AicoreTaskTraceCellV1 *>> worker_traces(static_cast<size_t>(runtime->worker_count));
    uint64_t execution_start_cycles = std::numeric_limits<uint64_t>::max();
    uint64_t descriptor_ready_cycles = 0;
    for (uint64_t task_id = 0; task_id < runtime->aicore_sidecar_layout.task_count; ++task_id) {
        const AicoreTaskTraceCellV1 &trace = traces[task_id];
        if (trace.valid == 0 || trace.aicore_entry_cycles == 0) continue;
        execution_start_cycles = std::min(execution_start_cycles, trace.claim_start_cycles);
        descriptor_ready_cycles = std::max(descriptor_ready_cycles, trace.descriptor_cache_observed_cycles);
    }
    execution_start_cycles = std::max(execution_start_cycles, descriptor_ready_cycles);
    for (uint64_t task_id = 0; task_id < runtime->aicore_sidecar_layout.task_count; ++task_id) {
        const AicoreTaskTraceCellV1 &trace = traces[task_id];
        if (trace.valid == 0) continue;
        if (trace.worker_id < static_cast<uint64_t>(runtime->worker_count)) {
            worker_traces[static_cast<size_t>(trace.worker_id)].push_back(&trace);
        }
        const uint64_t trace_core_type = trace.worker_id < static_cast<uint64_t>(runtime->worker_count) ?
                                             static_cast<uint64_t>(contexts[trace.worker_id].core_type) :
                                             UINT64_MAX;
        const char *claim_phase =
            trace.ready_source == static_cast<uint64_t>(AicoreReadySourceV1::STOLEN) ? "ReadySteal" : "ReadyPop";
        uint64_t claim_core_type = trace_core_type;
        if (trace.claim_worker_id < static_cast<uint64_t>(runtime->worker_count)) {
            claim_core_type = static_cast<uint64_t>(contexts[trace.claim_worker_id].core_type);
        }
        emit(
            trace.claim_worker_id, claim_core_type, trace.task_id, claim_phase, trace.claim_start_cycles,
            trace.claim_end_cycles
        );
        if (trace.aicore_entry_cycles != 0) {
            emit(
                trace.worker_id, trace_core_type, trace.task_id, "AICoreEntryToHandshake", trace.aicore_entry_cycles,
                trace.handshake_publish_cycles
            );
            emit(
                trace.worker_id, trace_core_type, trace.task_id, "HandshakeToRegisterRelease",
                trace.handshake_publish_cycles, trace.register_release_cycles
            );
            emit(
                trace.worker_id, trace_core_type, trace.task_id, "RegisterReleaseToDescriptorReady",
                trace.register_release_cycles, trace.descriptor_cache_observed_cycles
            );
            emit(
                trace.worker_id, trace_core_type, trace.task_id, "DescriptorReadyToReadyClaim",
                trace.descriptor_cache_observed_cycles, execution_start_cycles
            );
            const uint64_t first_ready_cycles = trace.ready_transition_cycles == 0 ?
                                                    execution_start_cycles :
                                                    std::max(execution_start_cycles, trace.ready_transition_cycles);
            emit(
                trace.worker_id, trace_core_type, trace.task_id, "ExecutionStartToFirstReady", execution_start_cycles,
                first_ready_cycles
            );
            emit(
                trace.worker_id, trace_core_type, trace.task_id, "FirstReadyToReadyClaim", first_ready_cycles,
                std::max(first_ready_cycles, trace.claim_start_cycles)
            );
        }
        emit(
            trace.worker_id, trace_core_type, trace.task_id, "Payload", trace.ready_observe_cycles,
            trace.kernel_start_cycles
        );
        emit(
            trace.worker_id, trace_core_type, trace.task_id, "Kernel", trace.kernel_start_cycles,
            trace.kernel_end_cycles
        );
        emit_completion_enqueue(
            trace.worker_id, trace_core_type, trace.task_id, trace.kernel_end_cycles, trace.completion_end_cycles,
            trace.completion_id, trace.completion_inbox_index
        );
        emit(
            trace.worker_id, trace_core_type, trace.task_id, "PostCompletion", trace.completion_end_cycles,
            trace.completion_bookkeeping_end_cycles
        );
        emit(
            trace.worker_id, trace_core_type, trace.task_id, "ReadyScan", trace.ready_scan_start_cycles,
            trace.ready_observe_cycles
        );
        const AicoreTaskControlV1 &control = task_controls[task_id];
        if (trace.completion_prepare_start_cycles != 0 &&
            trace.refill_resolver_worker_id < static_cast<uint64_t>(runtime->worker_count)) {
            const uint64_t resolver_worker_id = trace.refill_resolver_worker_id;
            const uint64_t resolver_core_type = static_cast<uint64_t>(contexts[resolver_worker_id].core_type);
            if (trace.refill_start_cycles >= trace.completion_prepare_start_cycles) {
                emit(
                    resolver_worker_id, resolver_core_type, trace.task_id, "CompletionBatchPrepare",
                    trace.completion_prepare_start_cycles, trace.refill_start_cycles
                );
            }
            if (trace.refill_start_cycles != 0 && trace.refill_task_id < runtime->aicore_sidecar_layout.task_count &&
                trace.refill_end_cycles >= trace.refill_start_cycles) {
                emit(
                    resolver_worker_id, resolver_core_type, trace.refill_task_id, "SlotRefill",
                    trace.refill_start_cycles, trace.refill_end_cycles
                );
            }
        }
        if (control.completion_resolve_start_cycles != 0 &&
            control.resolver_worker_id < static_cast<uint64_t>(runtime->worker_count)) {
            const AicoreWorkerContextV1 &resolver = contexts[control.resolver_worker_id];
            emit(
                control.resolver_worker_id, static_cast<uint64_t>(resolver.core_type), trace.task_id,
                "CompletionBatchClaim", control.completion_resolve_start_cycles, control.completion_resolve_start_cycles
            );
            emit(
                control.resolver_worker_id, static_cast<uint64_t>(resolver.core_type), trace.task_id, "WakeResolve",
                control.completion_resolve_start_cycles, control.completion_resolve_end_cycles
            );
        }
        emit(
            trace.worker_id, trace_core_type, trace.task_id, "ReadyPublish", control.ready_publish_cycles,
            control.ready_publish_cycles
        );
    }
    for (int32_t worker = 0; worker < runtime->worker_count; ++worker) {
        const AicoreWorkerContextV1 &context = contexts[worker];
        if (context.active == 0 || context.core_type != static_cast<int32_t>(AicoreRootCoreTypeV0::AIV)) continue;
        const uint64_t core_type = static_cast<uint64_t>(context.core_type);
        emit(
            context.worker_index, core_type, UINT64_MAX, "BootstrapGraphScan", context.bootstrap_start_cycles,
            context.bootstrap_scan_end_cycles
        );
        emit(
            context.worker_index, core_type, UINT64_MAX, "BootstrapBarrier", context.bootstrap_scan_end_cycles,
            context.bootstrap_end_cycles
        );
        uint64_t cursor = context.target_bootstrap_start_cycles;
        const uint64_t target_end = context.target_bootstrap_end_cycles;
        const uint64_t target_cycles[AICORE_CORE_TYPE_COUNT_V1]{
            context.bootstrap_target_aic_cycles,
            context.bootstrap_target_aiv_cycles,
        };
        const uint64_t ready_claim_cycles[AICORE_CORE_TYPE_COUNT_V1]{
            context.bootstrap_ready_claim_aic_cycles,
            context.bootstrap_ready_claim_aiv_cycles,
        };
        const uint64_t slot_fill_cycles[AICORE_CORE_TYPE_COUNT_V1]{
            context.bootstrap_slot_fill_aic_cycles,
            context.bootstrap_slot_fill_aiv_cycles,
        };
        const uint64_t free_advertise_cycles[AICORE_CORE_TYPE_COUNT_V1]{
            context.bootstrap_free_advertise_aic_cycles,
            context.bootstrap_free_advertise_aiv_cycles,
        };
        const char *ready_claim_phase[AICORE_CORE_TYPE_COUNT_V1]{
            "BootstrapAICReadyClaim",
            "BootstrapAIVReadyClaim",
        };
        const char *slot_fill_phase[AICORE_CORE_TYPE_COUNT_V1]{
            "BootstrapAICSlotFill",
            "BootstrapAIVSlotFill",
        };
        const char *free_advertise_phase[AICORE_CORE_TYPE_COUNT_V1]{
            "BootstrapAICFreeAdvertise",
            "BootstrapAIVFreeAdvertise",
        };
        const char *other_phase[AICORE_CORE_TYPE_COUNT_V1]{
            "BootstrapAICOther",
            "BootstrapAIVOther",
        };
        for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1 && cursor <= target_end; ++type) {
            uint64_t type_end = std::min(target_end, cursor + target_cycles[type]);
            auto emit_component = [&](const char *phase, uint64_t cycles) {
                if (cycles == 0) return;
                uint64_t component_end = std::min(type_end, cursor + cycles);
                emit(context.worker_index, core_type, UINT64_MAX, phase, cursor, component_end);
                cursor = component_end;
            };
            emit_component(ready_claim_phase[type], ready_claim_cycles[type]);
            emit_component(slot_fill_phase[type], slot_fill_cycles[type]);
            emit_component(free_advertise_phase[type], free_advertise_cycles[type]);
            emit_component(other_phase[type], type_end - cursor);
        }
        emit(context.worker_index, core_type, UINT64_MAX, "BootstrapTargetOther", cursor, target_end);
    }
    for (auto &worker_trace : worker_traces) {
        std::sort(
            worker_trace.begin(), worker_trace.end(),
            [](const AicoreTaskTraceCellV1 *lhs, const AicoreTaskTraceCellV1 *rhs) {
                return lhs->kernel_start_cycles < rhs->kernel_start_cycles;
            }
        );
        for (size_t index = 1; index < worker_trace.size(); ++index) {
            const AicoreTaskTraceCellV1 &previous = *worker_trace[index - 1];
            const AicoreTaskTraceCellV1 &current = *worker_trace[index];
            const uint64_t current_core_type = static_cast<uint64_t>(contexts[current.worker_id].core_type);
            const uint64_t transition_end = previous.completion_bookkeeping_end_cycles;
            uint64_t scheduler_start = transition_end;
            if (current.previous_trace_commit_end_cycles >= transition_end &&
                current.previous_trace_commit_end_cycles <= current.ready_scan_start_cycles) {
                emit(
                    current.worker_id, current_core_type, current.task_id, "TraceCommit", transition_end,
                    current.previous_trace_commit_end_cycles
                );
                scheduler_start = current.previous_trace_commit_end_cycles;
            }
            if (current.ready_scan_start_cycles < scheduler_start) continue;
            const uint64_t scheduler_end = current.ready_scan_start_cycles;
            const uint64_t detailed_cycles[] = {
                current.inter_task_completion_service_cycles,
                current.inter_task_dispatch_aic_cycles,
                current.inter_task_dispatch_aiv_cycles,
                current.inter_task_ready_poll_cycles,
                current.inter_task_backoff_cycles,
            };
            const char *detailed_phases[] = {
                "InterTaskCompletionService", "InterTaskDispatchAIC", "InterTaskDispatchAIV",
                "InterTaskReadyPoll",         "InterTaskBackoff",
            };
            bool has_detail = false;
            for (uint64_t cycles : detailed_cycles)
                has_detail = has_detail || cycles != 0;
            if (!has_detail) {
                emit(
                    current.worker_id, current_core_type, current.task_id, "InterTaskSchedule", scheduler_start,
                    scheduler_end
                );
                continue;
            }
            uint64_t cursor = scheduler_start;
            for (size_t detail = 0; detail < 5 && cursor < scheduler_end; ++detail) {
                if (detailed_cycles[detail] == 0) continue;
                uint64_t detail_end = std::min(scheduler_end, cursor + detailed_cycles[detail]);
                emit(
                    current.worker_id, current_core_type, current.task_id, detailed_phases[detail], cursor, detail_end
                );
                cursor = detail_end;
            }
            emit(current.worker_id, current_core_type, current.task_id, "InterTaskOther", cursor, scheduler_end);
        }
    }
    for (int32_t worker = 0; worker < runtime->worker_count; ++worker) {
        if (contexts[worker].active == 0) continue;
        emit_termination(
            contexts[worker].worker_index, static_cast<uint64_t>(contexts[worker].core_type), "WaitForExit",
            contexts[worker].exit_wait_start_cycles, contexts[worker].exit_observed_cycles, 0, 0, 0, 0
        );
        emit_termination(
            contexts[worker].worker_index, static_cast<uint64_t>(contexts[worker].core_type), "FinalStatsPublish",
            contexts[worker].final_stats_publish_start_cycles, contexts[worker].final_stats_publish_end_cycles, 0, 0, 0,
            0
        );
        emit(
            contexts[worker].worker_index, static_cast<uint64_t>(contexts[worker].core_type), UINT64_MAX,
            "ExitAckPublish", contexts[worker].exit_ack_publish_cycles, contexts[worker].exit_ack_publish_cycles
        );
        emit(
            contexts[worker].worker_index, static_cast<uint64_t>(contexts[worker].core_type), UINT64_MAX, "Drain",
            contexts[worker].drain_start_cycles, contexts[worker].drain_end_cycles
        );
    }
    phases << (first ? "" : "\n") << "  ]";

    phases << ",\n  \"aicpu_lifecycle_phases\": [";
    bool first_aicpu = true;
    auto emit_aicpu = [&](uint64_t worker_id, uint64_t thread_id, uint64_t core_type, const char *phase,
                          uint64_t start_cycles, uint64_t end_cycles, uint64_t poll_count = 0, uint64_t poll_cycles = 0,
                          uint64_t error_poll_count = 0) {
        if (end_cycles < start_cycles || (start_cycles == 0 && end_cycles == 0)) return;
        phases << (first_aicpu ? "\n" : ",\n") << "    {\"worker_id\": " << worker_id
               << ", \"aicpu_thread_id\": " << thread_id << ", \"core_type\": " << core_type << ", \"phase\": \""
               << phase << "\", \"start_cycles\": " << start_cycles << ", \"end_cycles\": " << end_cycles;
        if (poll_count != 0 || poll_cycles != 0 || error_poll_count != 0) {
            phases << ", \"poll_count\": " << poll_count << ", \"poll_cycles\": " << poll_cycles
                   << ", \"error_poll_count\": " << error_poll_count;
        }
        phases << "}";
        first_aicpu = false;
    };
    const uint64_t supervisor_thread =
        runtime->aicpu_thread_num > 0 ? static_cast<uint64_t>(runtime->aicpu_thread_num - 1) : 0;
    emit_aicpu(
        UINT64_MAX, supervisor_thread, UINT64_MAX, "WaitBootstrap", run_control->completion_wait_start_cycles,
        run_control->bootstrap_complete_cycles
    );
    emit_aicpu(
        UINT64_MAX, supervisor_thread, UINT64_MAX, "WaitResolved", run_control->bootstrap_complete_cycles,
        run_control->all_tasks_resolved_cycles, run_control->completion_poll_count, run_control->completion_poll_cycles,
        run_control->error_poll_count
    );
    emit_aicpu(
        UINT64_MAX, supervisor_thread, UINT64_MAX, "CompletionDecision", run_control->all_tasks_resolved_cycles,
        run_control->shutdown_ready_cycles
    );
    for (int32_t worker = 0; worker < runtime->worker_count; ++worker) {
        const AicpuCoreLifecycleTraceV1 &trace = lifecycle_traces[worker];
        emit_aicpu(
            trace.worker_id, trace.aicpu_thread_id, trace.core_type, "RegisterRelease", trace.register_release_cycles,
            trace.register_release_cycles
        );
        emit_aicpu(
            trace.worker_id, trace.aicpu_thread_id, trace.core_type, "ExitSignalToAck", trace.exit_signal_cycles,
            trace.exit_ack_cycles
        );
    }
    phases << (first_aicpu ? "" : "\n") << "  ]\n";
    json.insert(object_end, phases.str());

    const std::filesystem::path temporary = path.string() + ".scheduler.tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(json.data(), static_cast<std::streamsize>(json.size()));
    output.close();
    if (!output) {
        LOG_ERROR("A5 HBG AICore scheduler: failed to write profiling trace");
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return false;
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary, path, rename_error);
    if (rename_error) {
        LOG_ERROR("A5 HBG AICore scheduler: failed to publish profiling trace: %s", rename_error.message().c_str());
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return false;
    }
    return true;
}

bool create_aicore_sidecar_v1(
    Runtime *runtime, const HostApi *api, PTO2SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    uint64_t task_window_size
) {
    if (total_tasks < 0 || task_window_size == 0 || static_cast<uint64_t>(total_tasks) > task_window_size) {
        LOG_ERROR(
            "A5 HBG AICore scheduler: invalid graph size tasks=%d window=%" PRIu64, total_tasks, task_window_size
        );
        return false;
    }
    AicoreReadonlyGraphV0 host_graph{
        reinterpret_cast<uint64_t>(host_sm_handle.header->ring.task_descriptors),
        reinterpret_cast<uint64_t>(host_sm_handle.header->ring.task_payloads),
        static_cast<uint64_t>(total_tasks),
        task_window_size - 1,
    };
    std::vector<int64_t> inline_completed_task_ids;
    std::vector<AicoreTaskMetadataV1> task_metadata(static_cast<size_t>(total_tasks));
    uint64_t aic_task_count = 0;
    uint64_t aiv_task_count = 0;
    uint64_t executable_task_count = 0;
    uint64_t executable_subtask_count = 0;
    uint64_t gang_task_count = 0;
    uint64_t aic_worker_demand = 0;
    uint64_t aiv_worker_demand = 0;
    for (int64_t task_id = 0; task_id < total_tasks; ++task_id) {
        PTO2TaskSlotState &slot = host_sm_handle.header->ring.get_slot_state_by_task_id(task_id);
        AicoreTaskShapeV1 shape{};
        AicoreRootStatusV0 status = aicore_classify_task_shape_v1(host_graph, task_id, &shape);
        if (status != AicoreRootStatusV0::OK) {
            bool inline_completed_task = status == AicoreRootStatusV0::UNSUPPORTED_SHAPE &&
                                         slot.active_mask.raw() == 0 && slot.logical_block_num == 1 &&
                                         slot.total_required_subtasks == 0 &&
                                         slot.task_state.load(std::memory_order_acquire) == PTO2_TASK_COMPLETED &&
                                         slot.task_attrs.allow_early_resolve() &&
                                         !slot.task_attrs.requires_sync_start() && !slot.task_attrs.has_predicate();
            if (inline_completed_task) {
                inline_completed_task_ids.push_back(task_id);
                continue;
            }
            LOG_ERROR(
                "A5 HBG AICore scheduler: invalid v0 task id=%" PRId64 " status=%" PRIu64, task_id,
                static_cast<uint64_t>(status)
            );
            return false;
        }
        const uint8_t active_mask = shape.active_mask;
        const uint32_t active_subtasks = static_cast<uint32_t>(__builtin_popcount(active_mask));
        if (slot.logical_block_num <= 0) {
            LOG_ERROR("A5 HBG AICore scheduler: task id=%" PRId64 " has invalid block_num=%d", task_id,
                      slot.logical_block_num);
            return false;
        }
        const uint32_t logical_block_num = static_cast<uint32_t>(slot.logical_block_num);
        if (logical_block_num > UINT16_MAX / active_subtasks) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: task id=%" PRId64 " block/subtask product exceeds sidecar capacity",
                task_id
            );
            return false;
        }
        const uint32_t expected_subtasks = logical_block_num * active_subtasks;
        if (slot.active_mask.raw() != active_mask || expected_subtasks > UINT16_MAX ||
            slot.total_required_subtasks != expected_subtasks ||
            slot.task_state.load(std::memory_order_acquire) != PTO2_TASK_PENDING ||
            slot.task_attrs.allow_early_resolve() || slot.task_attrs.has_predicate()) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: task id=%" PRId64
                " has inconsistent shape/state or uses predicate/early-resolve",
                task_id
            );
            return false;
        }
        for (uint32_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((active_mask & (1U << subtask_slot)) == 0) continue;
            const int32_t kernel_id = shape.kernel_ids[subtask_slot];
            if (kernel_id < 0 || kernel_id >= RUNTIME_MAX_FUNC_ID ||
                runtime->get_function_bin_addr(kernel_id) == 0) {
                LOG_ERROR(
                    "A5 HBG AICore scheduler: task id=%" PRId64 " kernel id %d has no registered callable",
                    task_id, kernel_id
                );
                return false;
            }
        }
        __gm__ uint8_t *payload = aicore_graph_payload_v0(host_graph, task_id);
        int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
        if (fanin_count < 0) {
            LOG_ERROR("A5 HBG AICore scheduler: task id=%" PRId64 " has invalid fanin count %d", task_id, fanin_count);
            return false;
        }
        AicoreTaskMetadataV1 &metadata = task_metadata[static_cast<size_t>(task_id)];
        for (uint32_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            metadata.kernel_ids[subtask_slot] = (active_mask & (1U << subtask_slot)) != 0 ?
                                                    static_cast<uint16_t>(shape.kernel_ids[subtask_slot]) :
                                                    UINT16_MAX;
        }
        metadata.active_mask = active_mask;
        metadata.flags = AICORE_TASK_EXECUTABLE_V1;
        metadata.logical_block_num = static_cast<uint16_t>(logical_block_num);
        metadata.total_required_subtasks = static_cast<uint16_t>(expected_subtasks);
        if (fanin_count != 0) metadata.flags |= AICORE_TASK_HAS_FANIN_V1;
        if (active_subtasks > 1) metadata.flags |= AICORE_TASK_MIX_V1;
        if (logical_block_num > 1) metadata.flags |= AICORE_TASK_SPMD_V1;
        if (slot.task_attrs.requires_sync_start()) metadata.flags |= AICORE_TASK_SYNC_START_V1;
        if (aicore_task_requires_sync_start_v1(metadata.flags) && logical_block_num == 1) {
            LOG_ERROR("A5 HBG AICore scheduler: sync-start task id=%" PRId64 " must have block_num > 1", task_id);
            return false;
        }
        if ((active_mask & 1U) != 0) {
            ++aic_task_count;
            aic_worker_demand = std::max<uint64_t>(aic_worker_demand, logical_block_num);
        }
        const uint32_t active_aiv_subtasks = active_subtasks - ((active_mask & 1U) != 0 ? 1U : 0U);
        if (active_aiv_subtasks != 0) {
            ++aiv_task_count;
            aiv_worker_demand = std::max<uint64_t>(aiv_worker_demand, logical_block_num * active_aiv_subtasks);
        }
        if (aicore_task_is_gang_v1(metadata.flags)) ++gang_task_count;
        executable_subtask_count += expected_subtasks;
        ++executable_task_count;
    }

    if (executable_task_count + inline_completed_task_ids.size() != static_cast<uint64_t>(total_tasks)) {
        LOG_ERROR("A5 HBG AICore scheduler: task metadata does not cover the graph");
        return false;
    }

    AicoreExecutionSidecarLayoutV1 layout{};
    if (!aicore_sidecar_plan_v1(static_cast<uint64_t>(total_tasks), aic_task_count, aiv_task_count, &layout) ||
        layout.total_size > std::numeric_limits<uint64_t>::max() - (AICORE_SIDECAR_ALIGNMENT_V1 - 1)) {
        LOG_ERROR("A5 HBG AICore scheduler: sidecar layout overflow");
        return false;
    }
    layout.executable_task_count = executable_task_count;
    layout.executable_subtask_count = executable_subtask_count;
    layout.gang_task_count = gang_task_count;
    layout.aic_worker_demand = aic_worker_demand;
    layout.aiv_worker_demand = aiv_worker_demand;

    const uint64_t allocation_size = layout.total_size + AICORE_SIDECAR_ALIGNMENT_V1 - 1;
    void *allocation = api->device_malloc(static_cast<size_t>(allocation_size));
    if (allocation == nullptr) {
        LOG_ERROR("A5 HBG AICore scheduler: failed to allocate %" PRIu64 " sidecar bytes", allocation_size);
        return false;
    }
    const uintptr_t aligned_address = (reinterpret_cast<uintptr_t>(allocation) + AICORE_SIDECAR_ALIGNMENT_V1 - 1) &
                                      ~(static_cast<uintptr_t>(AICORE_SIDECAR_ALIGNMENT_V1) - 1);

    std::vector<uint8_t> storage(static_cast<size_t>(allocation_size));
    const uintptr_t host_aligned_address =
        (reinterpret_cast<uintptr_t>(storage.data()) + AICORE_SIDECAR_ALIGNMENT_V1 - 1) &
        ~(static_cast<uintptr_t>(AICORE_SIDECAR_ALIGNMENT_V1) - 1);
    void *host_base = reinterpret_cast<void *>(host_aligned_address);
    if (!aicore_sidecar_init_v1(host_base, layout)) {
        api->device_free(allocation);
        LOG_ERROR("A5 HBG AICore scheduler: failed to initialize sidecar");
        return false;
    }

    auto *task_controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(host_base, layout.task_controls_offset);
    for (int64_t task_id : inline_completed_task_ids) {
        task_controls[task_id].state = static_cast<int64_t>(AicoreTaskStateV1::DONE);
        task_controls[task_id].wake_list_head = AICORE_WAKE_LIST_CLOSED_V1;
    }
    static_assert(
        AICORE_CALLABLE_CAPACITY_V1 == RUNTIME_MAX_FUNC_ID, "sidecar callable table must cover the runtime table"
    );
    auto *callable_addresses = aicore_sidecar_at_v1<uint64_t>(host_base, layout.callable_addresses_offset);
    const bool cpu_sim = std::strcmp(get_platform(), "a5sim") == 0;
    for (uint32_t func_id = 0; func_id < AICORE_CALLABLE_CAPACITY_V1; ++func_id) {
        const uint64_t callable_address = runtime->get_function_bin_addr(static_cast<int32_t>(func_id));
        callable_addresses[func_id] = callable_address == 0 ?
                                          0 :
                                          (cpu_sim ? reinterpret_cast<const CoreCallable *>(callable_address)
                                                         ->resolved_addr() :
                                                     callable_address + CoreCallable::binary_data_offset());
    }
    auto *metadata = aicore_sidecar_at_v1<AicoreTaskMetadataV1>(host_base, layout.task_metadata_offset);
    std::copy(task_metadata.begin(), task_metadata.end(), metadata);

    const pto2_sm_layout::PTO2RingSegmentOffsets device_segments =
        pto2_sm_layout::ring_segment_offsets(task_window_size);
    const uint64_t device_sm_address = reinterpret_cast<uint64_t>(runtime->get_gm_sm_ptr());
    auto *run_control = aicore_sidecar_at_v1<AicoreRunControlV1>(host_base, layout.run_control_offset);
    run_control->expected_task_count = static_cast<uint64_t>(total_tasks);
    run_control->inline_completed_count = inline_completed_task_ids.size();
    run_control->claim_bindings_offset = layout.claim_bindings_offset;
    run_control->dispatch_payloads_offset = layout.dispatch_payloads_offset;
    run_control->task_metadata_offset = layout.task_metadata_offset;
    run_control->ready_inboxes_offset = layout.ready_inboxes_offset;
    run_control->ready_directory_offset = layout.ready_directory_offset;
    run_control->free_slot_directory_offset = layout.free_slot_directory_offset;
    run_control->gang_coordinator_offset = layout.gang_coordinator_offset;
    run_control->gang_cohorts_offset = layout.gang_cohorts_offset;
    run_control->error_task_id = UINT64_MAX;
    run_control->error_core_id = UINT64_MAX;
    run_control->error_core_type = UINT64_MAX;
    auto *gang_coordinator =
        aicore_sidecar_at_v1<AicoreGangCoordinatorV1>(host_base, layout.gang_coordinator_offset);
    gang_coordinator->gang_task_count = gang_task_count;

    auto *contexts = aicore_sidecar_at_v1<AicoreWorkerContextV1>(host_base, layout.worker_contexts_offset);
    int32_t aic_rank = 0;
    int32_t aiv_rank = 0;
    for (int32_t i = 0; i < runtime->get_worker_count(); ++i) {
        AicoreWorkerContextV1 &context = contexts[i];
        context.core_type = static_cast<int32_t>(runtime->workers[i].core_type);
        context.physical_core_id = -1;
        context.type_rank = context.core_type == static_cast<int32_t>(CoreType::AIC) ? aic_rank++ : aiv_rank++;
        context.active = 0;
        context.run_control_offset = layout.run_control_offset;
        context.task_controls_offset = layout.task_controls_offset;
        context.completion_inboxes_offset = layout.completion_inboxes_offset;
        context.task_metadata_offset = layout.task_metadata_offset;
        context.ready_inboxes_offset = layout.ready_inboxes_offset;
        context.ready_directory_offset = layout.ready_directory_offset;
        context.free_slot_directory_offset = layout.free_slot_directory_offset;
        context.worker_contexts_offset = layout.worker_contexts_offset;
        context.dispatch_slots_offset = layout.dispatch_slots_offset;
        context.callable_addresses_offset = layout.callable_addresses_offset;
        context.runtime_worker_count = static_cast<uint64_t>(runtime->get_worker_count());
        context.bootstrap_done = 0;
        context.gang_coordinator_offset = layout.gang_coordinator_offset;
        context.gang_cohorts_offset = layout.gang_cohorts_offset;
        context.gang_participants_offset = layout.gang_participants_offset;
        context.gang_commands_offset = layout.gang_commands_offset;
        context.graph_descriptors_address = device_sm_address + device_segments.descriptors;
        context.graph_payloads_address = device_sm_address + device_segments.payloads;
        context.sidecar_base_address = aligned_address;
        context.dispatch_payload_offset = layout.dispatch_payloads_offset + static_cast<uint64_t>(i) *
                                                                                AICORE_PENDING_SLOT_COUNT_V1 *
                                                                                sizeof(PTO2DispatchPayload);
        context.trace_cells_offset = layout.trace_cells_offset;
        context.task_window_mask = task_window_size - 1;
        context.graph_task_count = static_cast<uint64_t>(total_tasks);
        context.worker_index = static_cast<uint64_t>(i);
    }

    if (api->copy_to_device(
            reinterpret_cast<void *>(aligned_address), host_base, static_cast<size_t>(layout.total_size)
        ) != 0) {
        api->device_free(allocation);
        LOG_ERROR("A5 HBG AICore scheduler: failed to publish sidecar");
        return false;
    }
    runtime->aicore_sidecar_base = reinterpret_cast<void *>(aligned_address);
    runtime->aicore_sidecar_allocation = allocation;
    runtime->aicore_sidecar_allocation_size = allocation_size;
    runtime->aicore_sidecar_layout = layout;
    return true;
}

int32_t run_host_orchestration(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, PTO2Runtime *rt, DeviceArena &host_arena,
    const PTO2RuntimeArenaLayout &layout, void *device_sm, uint64_t sm_size, void *device_arena, void *gm_heap,
    const uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH], const uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH],
    void *host_orch_func_ptr, const ChipTaskArgs &orch_l2
) {
    // The dep_gen graph belongs to the orchestration that is about to run.
    dep_gen_host_graph_begin_capture();

    std::vector<uint8_t> host_sm_buf(sm_size, 0);
    void *host_sm = host_sm_buf.data();

    // Re-point the orchestrator half at the host SM (scheduler keeps device SM).
    // init_data_from_layout resets the orchestrator state, so this is safe.
    if (!rt->orchestrator.init_data_from_layout(
            layout.orch, host_arena, host_sm, gm_heap, eff_heap_sizes[0], eff_task_window_sizes[0]
        )) {
        LOG_ERROR("host-orch: orchestrator re-init against host SM failed");
        return -1;
    }
    rt->orchestrator.wire_arena_pointers(layout.orch, host_arena, &rt->scheduler);

    // Initialize the host SM header (ring flow control) so submit_task can run.
    PTO2SharedMemoryHandle host_sm_handle;
    if (!host_sm_handle.init_per_ring(host_sm, sm_size, eff_task_window_sizes, eff_heap_sizes)) {
        LOG_ERROR("host-orch: host SM init_per_ring failed");
        return -1;
    }

    // Install the ops table (host s_runtime_ops) and latch this run's cluster
    // counts. worker_count is published by DeviceRunner::prepare_launch_shape
    // before this bind, so the host orchestrator sees the same geometry the
    // AICPU re-derives from the handshake at boot.
    const int32_t block_dim = runtime->get_worker_count() / PLATFORM_CORES_PER_BLOCKDIM;
    if (block_dim < 1) {
        LOG_ERROR("host-orch: worker_count %d yields no clusters", runtime->get_worker_count());
        return -1;
    }
    runtime_finalize_after_wire(
        rt, block_dim * PLATFORM_AIC_CORES_PER_BLOCKDIM, block_dim * PLATFORM_AIV_CORES_PER_BLOCKDIM
    );
    rt->mode = PTO2_MODE_EXECUTE;
    // get_tensor_data/set_tensor_data resolve buffer.addr through the host
    // views registered at staging time (runtime/host_tensor_access.h), so the
    // host orchestrator can read control tensors (e.g. paged_attention's
    // context_lens/block_table) whether or not the platform maps device memory
    // into the host address space.

    const HostOrchEntryPoints *eps = reinterpret_cast<const HostOrchEntryPoints *>(host_orch_func_ptr);
    if (eps->bind != nullptr) {
        rt->tensor_access = &tensor_access;
        // Binds the orchestration .so's own framework_current_runtime, which its
        // inline rt_submit_* read. The host library links a same-named copy from
        // orchestration/common.cpp, but nothing outside the .so includes
        // pto_orchestration_api.h, so nothing reads that one — rt_scope_* and
        // rt_orchestration_done take the runtime as an argument.
        eps->bind(rt);
    } else {
        LOG_ERROR("host-orch: orch .so framework_bind_runtime was not resolved");
        return -1;
    }

    if (eps->config == nullptr) {
        LOG_ERROR("host-orch: orch .so aicpu_orchestration_config was not resolved");
        return -1;
    }
    (void)eps->config(orch_l2);

    rt_scope_begin(rt);
    eps->entry(orch_l2);
    rt_scope_end(rt);
    rt_orchestration_done(rt);

    int32_t total_tasks = pto2_sm_layout::ring_current_task_index_addr(host_sm)->load(std::memory_order_acquire);

    if (!create_aicore_sidecar_v1(runtime, api, host_sm_handle, total_tasks, eff_task_window_sizes[0])) {
        return -1;
    }

    // Relocate the host-DDR cross-task pointers to their final DEVICE addresses
    // on the host, before the SM and arena leave for the device. Pointers into
    // the SM shift by sm_delta; pointers into the arena (fanout adjacency, wiring
    // queue) shift by arena_delta. After this both the SM and arena carry device
    // addresses, so the device boots scheduler-only.
    const int64_t sm_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_sm)) -
                             static_cast<int64_t>(reinterpret_cast<uint64_t>(host_sm));
    const int64_t arena_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_arena)) -
                                static_cast<int64_t>(reinterpret_cast<uint64_t>(host_arena.base()));
    if (!relocate_host_orch_image(
            host_sm_handle, rt, reinterpret_cast<uint64_t>(host_sm), sm_size, sm_delta,
            reinterpret_cast<uint64_t>(host_arena.base()), layout.arena_size, arena_delta
        )) {
        LOG_ERROR("host-orch: relocation failed; refusing to H2D an image with unrelocated host pointers");
        release_aicore_sidecar(runtime, api);
        return -1;
    }

    if (api->copy_to_device(device_sm, host_sm, sm_size) != 0) {
        LOG_ERROR("host-orch: H2D of populated SM failed");
        release_aicore_sidecar(runtime, api);
        return -1;
    }
    return total_tasks;
}

}  // namespace

/**
 * Stage the per-callable resources (kernel binaries + orchestration SO) into
 * CallableArtifacts for subsequent per-run binding. Nothing here depends on
 * per-run argument values, so registration runs once per callable_id.
 *
 * @param callable  ChipCallable carrying the orch SO + child kernel binaries
 * @param api       Context-bound platform operations used during registration
 * @param out       Callable-owned artifacts retained across runs
 * @return 0 on success, -1 on failure
 */
extern "C" int register_callable_impl(const ChipCallable *callable, const HostApi *api, CallableArtifacts *out) {
    if (callable == nullptr) {
        LOG_ERROR("Callable pointer is null");
        return -1;
    }
    if (api == nullptr || out == nullptr) {
        LOG_ERROR("HostApi or out is null");
        return -1;
    }
    *out = CallableArtifacts{};
    out->signature.assign(callable->signature_, callable->signature_ + callable->sig_count());

    LOG_INFO("Registering %d kernel(s) in register_callable_impl", callable->child_count());
    if (upload_and_collect_child_addrs(
            callable, api, &out->kernel_addrs, &out->chip_buffer_dev, &out->chip_buffer_hash, &out->aicore_image_hash
        ) != 0) {
        LOG_ERROR("Failed to upload ChipCallable buffer");
        return -1;
    }
    for (const ChildKernelAddr &c : out->kernel_addrs) {
        if (c.func_id < 0 || c.func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d is out of range [0, %d)", c.func_id, RUNTIME_MAX_FUNC_ID);
            return -1;
        }
    }

    const uint8_t *orch_so_binary = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();

    if (orch_so_binary == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration SO binary is required for host orchestration");
        return -1;
    }

    out->orch_so_data = orch_so_binary;
    out->orch_so_size = orch_so_size;
    out->func_name = callable->func_name();
    out->config_name = callable->config_name();

    // host_build_graph host-orch: dlopen the orchestration .so ON THE HOST and
    // resolve its entry symbol now. The handle is held across the prepared
    // callable's lifetime (closed by DeviceRunner::unregister_callable via
    // host_dlopen_handle); bind_callable_to_runtime_impl invokes the resolved
    // entry per run. This is what makes the host-side dlopen observable
    // (host_dlopen_count) while the AICPU never dlopens the orch .so.
    {
        const char *orch_func_name = callable->func_name();
        if (orch_func_name == nullptr || orch_func_name[0] == '\0') {
            LOG_ERROR("host-orch: orchestration function name is empty");
            return -1;
        }
        std::string so_path;
        if (!create_orch_so_tempfile(orch_so_binary, orch_so_size, &so_path)) {
            LOG_ERROR("host-orch: failed to materialize orchestration .so");
            return -1;
        }
        void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            LOG_ERROR("host-orch: dlopen failed: %s", dlerror());
            return -1;
        }
        void *entry = dlsym(handle, orch_func_name);
        if (entry == nullptr) {
            LOG_ERROR("host-orch: dlsym('%s') failed: %s", orch_func_name, dlerror());
            dlclose(handle);
            return -1;
        }
        // The orch .so has its own framework_bind_runtime / g_current_runtime
        // (orchestration/common.cpp is compiled into it); resolve it now so the
        // per-run bind can set it before the .so's inline rt_submit_* run.
        void *bind_sym = dlsym(handle, "framework_bind_runtime");
        if (bind_sym == nullptr) {
            LOG_ERROR("host-orch: orch .so does not export framework_bind_runtime: %s", dlerror());
            dlclose(handle);
            return -1;
        }
        void *config_sym = dlsym(handle, "aicpu_orchestration_config");
        if (config_sym == nullptr) {
            LOG_ERROR("host-orch: orch .so does not export aicpu_orchestration_config: %s", dlerror());
            dlclose(handle);
            return -1;
        }
        // Safe to unlink now: the handle keeps the .so mapped regardless of path.
        unlink(so_path.c_str());
        auto *eps = new HostOrchEntryPoints{};
        eps->entry = reinterpret_cast<OrchestrationEntryFunc>(entry);
        eps->bind = reinterpret_cast<OrchestrationBindFunc>(bind_sym);
        eps->config = reinterpret_cast<OrchestrationConfigFunc>(config_sym);
        out->host_dlopen_handle = handle;
        out->host_orch_func_ptr = eps;
        LOG_INFO("host-orch: loaded orchestration entry '%s' on host", orch_func_name);
    }
    LOG_INFO("Orchestration SO: %zu bytes staged", orch_so_size);
    return 0;
}

/**
 * Per-run binding: build device-side argument storage (tensor copy-out, GM
 * heap, PTO2 shared memory) and publish it to the runtime. Assumes the
 * callable-side state (kernel binaries, orch SO bytes, func/config names)
 * is already populated by register_callable_impl.
 *
 * Splitting this from register_callable_impl matches the per-callable_id
 * design: register/run_prepared invokes this every call, while the prep
 * half runs only once per callable_id.
 *
 * @param runtime    Pointer to the per-run Runtime
 * @param api        Context-bound platform operations for this run
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    [[maybe_unused]] const uint64_t *ring_dep_pool  // polling has no dep_pool; kept for ABI stability
) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
        return -1;
    }
    if (orch_args == nullptr) {
        LOG_ERROR("orch_args pointer is null");
        return -1;
    }
    // host_build_graph host-orch: register_callable_impl resolved the
    // orchestration entry on the host and passed it here as host_orch_func_ptr;
    // it is run below (after the arena is built) against a host SM mirror.
    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();
    LOG_INFO("RT2 bind: %d tensors + %d scalars, host orchestration mode", tensor_count, scalar_count);

    int64_t t_total_start = _now_ms();

    uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH];
    if (!resolve_ring_config(ring_task_window, ring_heap, eff_task_window_sizes, eff_heap_sizes)) {
        return -1;
    }
    const std::string task_window_log = format_ring_array(eff_task_window_sizes);
    const std::string heap_log = format_ring_array(eff_heap_sizes);
    LOG_INFO("Ring buffer sizes: task_window=%s heap=%s", task_window_log.c_str(), heap_log.c_str());

    // Build device args: copy from input, replace host tensor pointers with device pointers
    ChipStorageTaskArgs device_args;

    // This run's host-view window. The accessor owns every mapping it
    // registers and releases them on every exit path, so no host view outlives
    // the point at which a task could make it stale.
    HostTensorAccessor tensor_access(api);

    int64_t t_args_start = _now_ms();
    for (int i = 0; i < tensor_count; i++) {
        ChipTensor t = orch_args->tensor(i);

        if (t.is_child_memory()) {
            LOG_DEBUG("  ChipTensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
            device_args.add_tensor(t);
            continue;
        }

        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
        size_t size = static_cast<size_t>(t.nbytes());

        void *dev_ptr = api->device_malloc(size);
        if (dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate device memory for tensor %d", i);
            return -1;
        }

        // Pure write-only OUTPUT buffers are never read by the kernel and hold
        // no meaningful host content, so they need no device staging — the
        // kernel defines what it writes and any unwritten bytes are undefined.
        // IN / INOUT (read-before-write) are staged H2D.
        bool is_pure_output = (signature != nullptr && i < sig_count && signature[i] == ArgDirection::OUT);
        if (!is_pure_output) {
            int rc = api->copy_to_device(dev_ptr, host_ptr, size);
            if (rc != 0) {
                LOG_ERROR("Failed to stage tensor %d to device", i);
                api->device_free(dev_ptr);
                return -1;
            }
        }
        // Read-only INPUT tensors are never written by the kernel, so there is
        // no point copying them back D2H at the end. Index the signature
        // by the orch tensor index `i` (child_memory tensors are skipped above
        // but do not consume a separate signature slot — scalars follow the
        // tensor entries). Anything not provably IN keeps the safe default of
        // copying back.
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        runtime->tensor_pairs_.push_back({host_ptr, dev_ptr, size, needs_copy_back});
        LOG_DEBUG("  ChipTensor %d: %zu bytes at %p", i, size, dev_ptr);

        // host_build_graph runs the orchestrator on the host, which may read
        // control tensors (e.g. paged_attention's context_lens/block_table) via
        // get_tensor_data to shape the graph. Give it a host view of this
        // buffer: the device buffer itself where the platform can map it into
        // the host address space (released in validate_runtime_impl before
        // device_free), otherwise the staging copy, which holds the same bytes
        // for the whole orchestration window and whose writes are pushed back
        // to the device. A tensor with neither is not host-accessible, so the
        // prepare fails here rather than the orchestrator dereferencing a
        // device address.
        if (!tensor_access.add(reinterpret_cast<uint64_t>(dev_ptr), size, host_ptr)) {
            LOG_ERROR("host-orch: no host view for tensor %d (dev_ptr %p, %zu bytes)", i, dev_ptr, size);
            return -1;
        }

        t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
        device_args.add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        device_args.add_scalar(orch_args->scalar(i));
    }
    int64_t t_args_end = _now_ms();

    // Lay out the per-Worker static device arena. GM heap, PTO2 shared memory,
    // and the prebuilt runtime arena use three independent pooled device
    // allocations committed together by setup_static_arena.
    // Owned by DeviceRunner across runs — do NOT record in tensor_pairs_; the
    // free is deferred to DeviceRunner::finalize(). The runtime-arena size is
    // determined by replaying the reserve sequence on a host-side arena.
    uint64_t total_heap_size = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (eff_heap_sizes[r] > std::numeric_limits<uint64_t>::max() - total_heap_size) {
            LOG_ERROR("Total ring heap size overflows uint64_t");
            return -1;
        }
        total_heap_size += eff_heap_sizes[r];
    }
    uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size_per_ring(eff_task_window_sizes);

    int64_t t_prebuilt_start = _now_ms();
    DeviceArena host_arena;  // libc malloc backend by default
    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(host_arena, eff_task_window_sizes, eff_heap_sizes);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return -1;
    }

    int64_t t_setup_start = _now_ms();
    if (api->setup_static_arena(total_heap_size, sm_size, layout.arena_size) != 0) {
        LOG_ERROR("Failed to setup pooled static arena");
        return -1;
    }
    int64_t t_setup_end = _now_ms();

    int64_t t_heap_start = _now_ms();
    void *gm_heap = api->acquire_pooled_gm_heap();
    int64_t t_heap_end = _now_ms();
    if (gm_heap == nullptr) {
        LOG_ERROR("Failed to acquire pooled GM heap");
        return -1;
    }
    runtime->set_gm_heap(gm_heap);

    int64_t t_sm_start = _now_ms();
    void *sm_ptr = api->acquire_pooled_gm_sm();
    int64_t t_sm_end = _now_ms();
    if (sm_ptr == nullptr) {
        LOG_ERROR("Failed to acquire pooled PTO2 shared memory");
        return -1;
    }
    runtime->set_gm_sm_ptr(sm_ptr);

    void *runtime_arena_dev = api->acquire_pooled_runtime_arena();
    if (runtime_arena_dev == nullptr) {
        LOG_ERROR("Failed to acquire pooled runtime arena");
        return -1;
    }

    // Set up orchestration state (consumed by the host orchestrator below)
    runtime->set_orch_args(device_args);

    // -------------------------------------------------------------------------
    // Build the prebuilt runtime-arena image on host.
    //
    // We pre-compute every byte the AICPU's runtime arena would otherwise have
    // to write at boot: layout offsets, sub-structure init data, and pointers
    // back to the SM / GM heap. Then we rtMemcpy the image into the pooled
    // runtime-arena region that DeviceRunner keeps alive across runs. AICPU
    // boot becomes attach + wire (cheap pointer fixup) + sm_handle->init (SM
    // reset) + a handful of device-only field fixups.
    // -------------------------------------------------------------------------
    PTO2Runtime *rt =
        runtime_init_data_from_layout(host_arena, layout, PTO2_MODE_EXECUTE, sm_ptr, sm_size, gm_heap, eff_heap_sizes);
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return -1;
    }
    runtime_wire_arena_pointers(host_arena, layout, rt);

    // host_build_graph host-orch: run the orchestrator on the host now, against
    // a host SM mirror, and ship the populated SM to the device. The arena
    // (copied to the device below) carries the resulting orchestrator/scheduler
    // state; the device boots scheduler-only. register_callable_impl guarantees
    // host_orch_func_ptr is non-null on success (it fails the whole prepare
    // otherwise), so this is an assertion-style guard, not a fallback path.
    if (host_orch_func_ptr == nullptr) {
        LOG_ERROR("host-orch: orchestration entry points were not resolved");
        return -1;
    }
    {
        ChipTaskArgs orch_l2;
        orch_l2.create_from_chip_args(device_args);
        int32_t total_tasks = run_host_orchestration(
            runtime, api, tensor_access, rt, host_arena, layout, sm_ptr, sm_size, runtime_arena_dev, gm_heap,
            eff_heap_sizes, eff_task_window_sizes, host_orch_func_ptr, orch_l2
        );
        // The orchestrator is the only host-view reader; from here the device
        // owns these buffers, so drop the window on both exits.
        tensor_access.close();
        if (total_tasks < 0) {
            LOG_ERROR("host-orch: orchestration run failed");
            return -1;
        }
        runtime->host_total_tasks = total_tasks;
        LOG_INFO("host-orch: submitted %d tasks on host", total_tasks);
    }

    // Stash the layout inside the PTO2Runtime image so the AICPU can recover
    // every arena-internal offset after rtMemcpy. The runtime arena's device
    // base does NOT travel in this image — it's on the host Runtime
    // (set_prebuilt_arena below), since the AICPU needs that pointer
    // *before* it can dereference the image.
    rt->prebuilt_layout = layout;

    int rc_upload = api->copy_to_device(runtime_arena_dev, host_arena.base(), layout.arena_size);
    if (rc_upload != 0) {
        LOG_ERROR("Failed to rtMemcpy prebuilt runtime arena to device (rc=%d)", rc_upload);
        release_aicore_sidecar(runtime, api);
        return -1;
    }
    runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);
    int64_t t_prebuilt_end = _now_ms();

    LOG_INFO("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

    int64_t t_total_end = _now_ms();
    LOG_INFO("TIMING: args_malloc_copy = %" PRId64 "ms", t_args_end - t_args_start);
    LOG_INFO("TIMING: static_arena_setup = %" PRId64 "ms", t_setup_end - t_setup_start);
    LOG_INFO("TIMING: gm_heap_acquire = %" PRId64 "ms", t_heap_end - t_heap_start);
    LOG_INFO("TIMING: shared_mem_acquire = %" PRId64 "ms", t_sm_end - t_sm_start);
    LOG_INFO("TIMING: prebuilt_runtime_arena = %" PRId64 "ms", t_prebuilt_end - t_prebuilt_start);
    LOG_INFO("TIMING: total_init_runtime_impl = %" PRId64 "ms", t_total_end - t_total_start);

    return 0;
}

/**
 * Validate runtime results and cleanup.
 *
 * This function:
 * 1. Copies recorded tensors from device back to host
 * 2. Frees device memory for recorded tensors
 * 3. Clears tensor pair state
 *
 * @param runtime       Pointer to Runtime
 * @param execution_rc  Device-runner drain status after successful enqueue,
 *                      or enqueue status on failure
 * @return 0 on success, -1 on failure
 */
extern "C" int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
        return -1;
    }

    int rc = 0;

    LOG_INFO("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorPair *tensor_pairs = runtime->tensor_pairs_.data();
    int tensor_pair_count = static_cast<int>(runtime->tensor_pairs_.size());

    LOG_INFO("ChipTensor pairs to process: %d", tensor_pair_count);

    const bool profiling_only_failure = execution_rc == SIMPLER_PROFILING_VALIDATION_ERROR;
    bool skip_tensor_copy_back = execution_rc != 0 && !profiling_only_failure;
    int32_t runtime_status = 0;
    PTO2SharedMemoryHeader host_header;
    memset(&host_header, 0, sizeof(host_header));

    if (execution_rc != 0 && !profiling_only_failure) {
        runtime_status = pto2_read_runtime_status(runtime, api, &host_header);
    }
    if (runtime_status != 0) {
        int32_t orch_error_code = host_header.orch_error_code.load(std::memory_order_relaxed);
        int32_t sched_error_code = host_header.sched_error_code.load(std::memory_order_relaxed);
        LOG_RUNTIME_FAILURE(orch_error_code, sched_error_code, runtime_status);
    }

    const bool execution_completed = execution_rc == 0 || profiling_only_failure;
    if (runtime->aicore_sidecar_base != nullptr && execution_completed) {
        const uint64_t sidecar_size = runtime->aicore_sidecar_layout.total_size;
        std::vector<uint8_t> sidecar_storage(static_cast<size_t>(sidecar_size + AICORE_SIDECAR_ALIGNMENT_V1 - 1));
        uintptr_t host_sidecar_address =
            (reinterpret_cast<uintptr_t>(sidecar_storage.data()) + AICORE_SIDECAR_ALIGNMENT_V1 - 1) &
            ~(static_cast<uintptr_t>(AICORE_SIDECAR_ALIGNMENT_V1) - 1);
        void *host_sidecar = reinterpret_cast<void *>(host_sidecar_address);
        int sidecar_rc =
            api->copy_from_device(host_sidecar, runtime->aicore_sidecar_base, static_cast<size_t>(sidecar_size));
        if (sidecar_rc != 0) {
            LOG_ERROR("A5 HBG AICore scheduler: failed to copy sidecar from device: %d", sidecar_rc);
            rc = sidecar_rc;
        } else {
            const auto *control = aicore_sidecar_at_v1<AicoreRunControlV1>(
                host_sidecar, runtime->aicore_sidecar_layout.run_control_offset
            );
            const auto *contexts = aicore_sidecar_at_v1<AicoreWorkerContextV1>(
                host_sidecar, runtime->aicore_sidecar_layout.worker_contexts_offset
            );
            const auto *lifecycle_traces = aicore_sidecar_at_v1<AicpuCoreLifecycleTraceV1>(
                host_sidecar, runtime->aicore_sidecar_layout.aicpu_lifecycle_traces_offset
            );
            const auto *task_controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(
                host_sidecar, runtime->aicore_sidecar_layout.task_controls_offset
            );
            const auto *completion_inboxes = aicore_sidecar_at_v1<AicoreCompletionInboxV1>(
                host_sidecar, runtime->aicore_sidecar_layout.completion_inboxes_offset
            );
            const auto *ready_inboxes = aicore_sidecar_at_v1<AicoreReadyInboxV1>(
                host_sidecar, runtime->aicore_sidecar_layout.ready_inboxes_offset
            );
            const auto *ready_directory = aicore_sidecar_at_v1<AicoreReadyDirectoryV1>(
                host_sidecar, runtime->aicore_sidecar_layout.ready_directory_offset
            );
            const auto *free_slot_directory = aicore_sidecar_at_v1<AicoreFreeSlotDirectoryV1>(
                host_sidecar, runtime->aicore_sidecar_layout.free_slot_directory_offset
            );
            const auto *dispatch_slots = aicore_sidecar_at_v1<AicoreDispatchSlotV1>(
                host_sidecar, runtime->aicore_sidecar_layout.dispatch_slots_offset
            );
            const auto *traces = aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(
                host_sidecar, runtime->aicore_sidecar_layout.trace_cells_offset
            );
            uint64_t active_workers = 0;
            uint64_t worker_executed = 0;
            uint64_t bootstrap_tasks = 0;
            uint64_t ready_enqueues = 0;
            uint64_t ready_batches = 0;
            uint64_t ready_pops = 0;
            uint64_t ready_steals = 0;
            uint64_t ready_cas_retries = 0;
            uint64_t ready_link_waits = 0;
            uint64_t ready_link_wait_max = 0;
            uint64_t free_slot_claims = 0;
            uint64_t free_slot_stale = 0;
            uint64_t state_polls = 0;
            uint64_t fanin_loads = 0;
            uint64_t wake_registers = 0;
            uint64_t wake_cas_retries = 0;
            uint64_t wake_closed_retries = 0;
            uint64_t wake_migrations = 0;
            uint64_t wake_closes = 0;
            uint64_t completion_enqueues = 0;
            uint64_t completion_batches = 0;
            uint64_t completion_resolves = 0;
            uint64_t completion_steals = 0;
            uint64_t completion_link_waits = 0;
            uint64_t completion_link_wait_max = 0;
            uint64_t completion_lag_cycles = 0;
            uint64_t completion_lag_max_cycles = 0;
            uint64_t ready_to_kernel_cycles = 0;
            uint64_t ready_to_kernel_max_cycles = 0;
            uint64_t idle_iterations = 0;
            uint64_t backoff_cycles = 0;
            uint64_t payload_cycles = 0;
            uint64_t kernel_cycles = 0;
            uint64_t completion_cycles = 0;
            const uint64_t executable_task_count = runtime->aicore_sidecar_layout.executable_task_count;
            const uint64_t executable_subtask_count = runtime->aicore_sidecar_layout.executable_subtask_count;
            const uint64_t ordinary_task_count = executable_task_count - runtime->aicore_sidecar_layout.gang_task_count;
            for (int32_t i = 0; i < runtime->worker_count; ++i) {
                if (contexts[i].active != 0) ++active_workers;
                worker_executed += contexts[i].executed_task_count;
                bootstrap_tasks += contexts[i].bootstrap_task_count;
                ready_enqueues += contexts[i].ready_enqueue_count;
                ready_batches += contexts[i].ready_batch_count;
                ready_pops += contexts[i].ready_pop_count;
                ready_steals += contexts[i].ready_steal_count;
                ready_cas_retries += contexts[i].ready_cas_retry_count;
                ready_link_waits += contexts[i].ready_link_wait_count;
                ready_link_wait_max = std::max(ready_link_wait_max, contexts[i].ready_link_wait_max);
                free_slot_claims += contexts[i].free_slot_claim_count;
                free_slot_stale += contexts[i].free_slot_stale_count;
                state_polls += contexts[i].task_state_poll_count;
                fanin_loads += contexts[i].fanin_state_load_count;
                wake_registers += contexts[i].wake_register_count;
                wake_cas_retries += contexts[i].wake_cas_retry_count;
                wake_closed_retries += contexts[i].wake_closed_retry_count;
                wake_migrations += contexts[i].wake_migrate_count;
                wake_closes += contexts[i].wake_close_count;
                completion_enqueues += contexts[i].completion_enqueue_count;
                completion_batches += contexts[i].completion_batch_count;
                completion_resolves += contexts[i].completion_resolve_count;
                completion_steals += contexts[i].completion_steal_count;
                completion_link_waits += contexts[i].completion_link_wait_count;
                completion_link_wait_max = std::max(completion_link_wait_max, contexts[i].completion_link_wait_max);
                completion_lag_cycles += contexts[i].completion_lag_cycles;
                completion_lag_max_cycles = std::max(completion_lag_max_cycles, contexts[i].completion_lag_max_cycles);
                ready_to_kernel_cycles += contexts[i].ready_to_kernel_cycles;
                ready_to_kernel_max_cycles =
                    std::max(ready_to_kernel_max_cycles, contexts[i].ready_to_kernel_max_cycles);
                idle_iterations += contexts[i].idle_iteration_count;
                backoff_cycles += contexts[i].backoff_cycles;
                payload_cycles += contexts[i].payload_cycles;
                kernel_cycles += contexts[i].kernel_cycles;
                completion_cycles += contexts[i].completion_enqueue_cycles;
            }
            bool completion_inboxes_empty = true;
            for (int32_t worker = 0; worker < runtime->worker_count; ++worker) {
                if (contexts[worker].active == 0) continue;
                if (completion_inboxes[worker].head != AICORE_INBOX_EMPTY_V1 ||
                    completion_inboxes[worker].completed_generations[0] != 0 ||
                    completion_inboxes[worker].completed_generations[1] != 0) {
                    LOG_ERROR(
                        "A5 HBG AICore scheduler: completion line=%d not empty head=%" PRId64
                        " generations={%" PRIu64 ",%" PRIu64 "}",
                        worker, completion_inboxes[worker].head,
                        completion_inboxes[worker].completed_generations[0],
                        completion_inboxes[worker].completed_generations[1]
                    );
                    completion_inboxes_empty = false;
                    break;
                }
            }
            bool ready_inboxes_empty = true;
            for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1 && ready_inboxes_empty; ++type) {
                for (uint64_t inbox = 0; inbox < control->resolver_count; ++inbox) {
                    uint64_t linear = static_cast<uint64_t>(type) * AICORE_WORKER_CAPACITY_V1 + inbox;
                    if (ready_inboxes[linear].head != AICORE_INBOX_EMPTY_V1) {
                        LOG_ERROR(
                            "A5 HBG AICore scheduler: ready type=%u inbox=%" PRIu64 " not empty head=%" PRId64, type,
                            inbox, ready_inboxes[linear].head
                        );
                        ready_inboxes_empty = false;
                        break;
                    }
                }
            }
            bool ready_directory_empty = true;
            for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
                for (uint32_t word = 0; word < AICORE_READY_DIRECTORY_WORD_COUNT_V1; ++word)
                    ready_directory_empty = ready_directory_empty && ready_directory->words[type][word] == 0;
            }
            bool task_controls_valid = true;
            for (int32_t task_id = 0; task_id < runtime->host_total_tasks; ++task_id) {
                if (task_controls[task_id].state != static_cast<int64_t>(AicoreTaskStateV1::DONE) ||
                    task_controls[task_id].wake_list_head != AICORE_WAKE_LIST_CLOSED_V1) {
                    LOG_ERROR(
                        "A5 HBG AICore scheduler: invalid task control id=%d state=%" PRId64 " wake_head=%" PRId64,
                        task_id, task_controls[task_id].state, task_controls[task_id].wake_list_head
                    );
                    task_controls_valid = false;
                    break;
                }
            }
            bool dispatch_slots_free = true;
            for (int32_t worker = 0; worker < runtime->worker_count && dispatch_slots_free; ++worker) {
                if (contexts[worker].active == 0) continue;
                if (contexts[worker].bootstrap_done == 0) {
                    LOG_ERROR("A5 HBG AICore scheduler: worker=%d dispatch bootstrap incomplete", worker);
                    dispatch_slots_free = false;
                    break;
                }
                for (uint32_t slot = 0; slot < AICORE_PENDING_SLOT_COUNT_V1; ++slot) {
                    uint64_t linear = static_cast<uint64_t>(worker) * AICORE_PENDING_SLOT_COUNT_V1 + slot;
                    const AicoreDispatchSlotV1 &dispatch = dispatch_slots[linear];
                    if (dispatch.task_id != AICORE_TASK_ID_INVALID_V1 ||
                        static_cast<AicoreDispatchPublicationV1>(dispatch.publication & UINT64_C(0xff)) !=
                            AicoreDispatchPublicationV1::FREE) {
                        LOG_ERROR(
                            "A5 HBG AICore scheduler: worker=%d slot=%u not free task=%" PRId64 " publication=%" PRIu64,
                            worker, slot, dispatch.task_id, dispatch.publication
                        );
                        dispatch_slots_free = false;
                        break;
                    }
                }
            }
            bool free_directory_exact = true;
            for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
                for (uint32_t word = 0; word < AICORE_FREE_SLOT_DIRECTORY_WORD_COUNT_V1; ++word) {
                    free_directory_exact = free_directory_exact && free_slot_directory->words[type][word] == 0;
                }
            }
            if (control->scheduler_error != 0 ||
                control->expected_task_count != static_cast<uint64_t>(runtime->host_total_tasks) ||
                worker_executed != executable_subtask_count || bootstrap_tasks != executable_task_count ||
                active_workers != control->active_worker_count ||
                control->bootstrap_arrived_count != control->resolver_count ||
                control->bootstrap_complete == 0 || control->resolved_task_count != executable_task_count ||
                wake_closes != executable_task_count || wake_registers != wake_migrations ||
                completion_enqueues != executable_subtask_count || completion_resolves != executable_task_count ||
                ready_enqueues != ordinary_task_count || ready_pops != ordinary_task_count ||
                !task_controls_valid || !completion_inboxes_empty || !ready_inboxes_empty || !ready_directory_empty ||
                !dispatch_slots_free || !free_directory_exact) {
                LOG_ERROR(
                    "A5 HBG AICore scheduler: invalid final state expected=%" PRIu64 " executed=%" PRIu64
                    " executable=%" PRIu64 " inline=%" PRIu64 " active=%" PRIu64 " bootstrap=%" PRIu64
                    " resolved=%" PRIu64 " wake_registers=%" PRIu64 " wake_migrations=%" PRIu64 " wake_closes=%" PRIu64
                    " completion_enqueues=%" PRIu64 " completion_resolves=%" PRIu64 " ready_enqueues=%" PRIu64
                    " ready_pops=%" PRIu64 " error=%" PRIu64,
                    control->expected_task_count, worker_executed, executable_task_count,
                    control->inline_completed_count, active_workers, bootstrap_tasks, control->resolved_task_count,
                    wake_registers, wake_migrations, wake_closes, completion_enqueues, completion_resolves,
                    ready_enqueues, ready_pops, control->scheduler_error
                );
                if (control->scheduler_error != 0) {
                    LOG_ERROR(
                        "A5 HBG AICore scheduler: first error task=%" PRIu64 " status=%" PRIu64 " core=%" PRIu64
                        " type=%" PRIu64 " graph_tasks=%" PRIu64 " descriptors=0x%" PRIx64 " payloads=0x%" PRIx64
                        " mask=0x%" PRIx64,
                        control->error_task_id, control->scheduler_error, control->error_core_id,
                        control->error_core_type, control->error_graph_task_count, control->error_descriptors_address,
                        control->error_payloads_address, control->error_task_window_mask
                    );
                }
                rc = -1;
            } else if (control->expected_task_count != 0) {
                LOG_INFO(
                    "A5 HBG AICore scheduler HOST TIMING: payload=%" PRIu64 " kernel=%" PRIu64 " completion=%" PRIu64
                    " backoff=%" PRIu64 " cycles",
                    payload_cycles, kernel_cycles, completion_cycles, backoff_cycles
                );
                LOG_INFO(
                    "A5 HBG AICore scheduler COUNTERS: bootstrap_tasks=%" PRIu64 " ready_enqueues=%" PRIu64
                    " ready_batches=%" PRIu64 " ready_pops=%" PRIu64 " ready_steals=%" PRIu64
                    " ready_cas_retries=%" PRIu64 " ready_link_waits=%" PRIu64 " ready_link_wait_max=%" PRIu64
                    " free_slot_claims=%" PRIu64 " free_slot_stale=%" PRIu64 " state_polls=%" PRIu64
                    " fanin_loads=%" PRIu64 " wake_registers=%" PRIu64 " wake_cas_retries=%" PRIu64
                    " wake_closed_retries=%" PRIu64 " wake_migrations=%" PRIu64 " wake_closes=%" PRIu64
                    " completion_enqueues=%" PRIu64 " completion_batches=%" PRIu64 " completion_resolves=%" PRIu64
                    " completion_steals=%" PRIu64 " completion_link_waits=%" PRIu64 " completion_link_wait_max=%" PRIu64
                    " completion_lag_cycles=%" PRIu64 " completion_lag_max=%" PRIu64 " ready_to_kernel_cycles=%" PRIu64
                    " ready_to_kernel_max=%" PRIu64 " idle_iterations=%" PRIu64,
                    bootstrap_tasks, ready_enqueues, ready_batches, ready_pops, ready_steals, ready_cas_retries,
                    ready_link_waits, ready_link_wait_max, free_slot_claims, free_slot_stale, state_polls, fanin_loads,
                    wake_registers, wake_cas_retries, wake_closed_retries, wake_migrations, wake_closes,
                    completion_enqueues, completion_batches, completion_resolves, completion_steals,
                    completion_link_waits, completion_link_wait_max, completion_lag_cycles, completion_lag_max_cycles,
                    ready_to_kernel_cycles, ready_to_kernel_max_cycles, idle_iterations
                );
            }
            if (!append_aicore_scheduler_trace(runtime, control, lifecycle_traces, traces, task_controls, contexts))
                rc = -1;
        }
    }

    if (skip_tensor_copy_back) {
        LOG_WARN("Skipping tensor copy-back because execution failed");
    } else {
        for (int i = 0; i < tensor_pair_count; i++) {
            const TensorPair &pair = tensor_pairs[i];

            // Skip if device pointer is null
            if (pair.dev_ptr == nullptr) {
                LOG_WARN("ChipTensor %d has null device pointer, skipping", i);
                continue;
            }

            // If host pointer is null, this is a device-only allocation (no copy-back)
            if (pair.host_ptr == nullptr) {
                LOG_DEBUG("ChipTensor %d: device-only allocation (no copy-back)", i);
                continue;
            }

            // Read-only INPUT tensors were uploaded H2D but the kernel never
            // wrote them — copying them back (potentially ~GB) is pure waste.
            // They are still device_free'd in the cleanup loop below.
            if (!pair.needs_copy_back) {
                LOG_DEBUG("ChipTensor %d: read-only input, skipping copy-back", i);
                continue;
            }

            int copy_rc = api->copy_from_device(pair.host_ptr, pair.dev_ptr, pair.size);
            if (copy_rc != 0) {
                LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
                rc = copy_rc;
            } else {
                LOG_DEBUG("ChipTensor %d: %zu bytes copied to host", i, pair.size);
            }
        }
    }

    // Cleanup device tensors
    LOG_INFO("=== Cleaning Up ===");
    for (int i = 0; i < tensor_pair_count; i++) {
        if (tensor_pairs[i].dev_ptr != nullptr) {
            api->device_free(tensor_pairs[i].dev_ptr);
        }
    }
    LOG_INFO("Freed %d device allocations", tensor_pair_count);
    release_aicore_sidecar(runtime, api);

    // Clear the per-run dispatch-table entries staged by register_callable_impl.
    // The underlying chip-callable device buffer is pool-managed by
    // DeviceRunner (keyed by content hash) and bulk-freed in
    // DeviceRunner::finalize(); re-running the same callable repeatedly
    // should not re-upload.
    int kernel_count = runtime->get_registered_kernel_count();
    for (int i = 0; i < kernel_count; i++) {
        int func_id = runtime->get_registered_kernel_func_id(i);
        runtime->set_function_bin_addr(func_id, 0);
    }
    if (kernel_count > 0) {
        LOG_INFO("Cleared %d kernel dispatch-table entries", kernel_count);
    }
    runtime->clear_registered_kernels();

    // Clear tensor pairs
    runtime->tensor_pairs_.clear();

    LOG_INFO("=== Finalize Complete ===");

    if (rc == 0 && runtime_status != 0) {
        rc = runtime_status;
    }
    if (rc == 0 && profiling_only_failure) {
        rc = execution_rc;
    }

    return rc;
}

// host_build_graph resolves orchestration on the host, so it exports no AICPU
// entries beyond the base {simpler_aicpu_exec, simpler_aicpu_init} — in
// particular it does not export simpler_aicpu_register_callable. Reporting an
// empty extra-symbol set keeps the common AICPU loader from looking for it.
extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count) {
    if (count != nullptr) {
        *count = 0;
    }
    return nullptr;
}
