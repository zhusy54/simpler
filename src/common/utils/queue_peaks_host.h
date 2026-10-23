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

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "host_log.h"
#include "utils/queue_peaks.h"
#include "utils/retained_temp_bump.h"

namespace simpler::queue_peaks {
inline int prepare_output(const HostApi *api, RetainedTempBump &bump, uint64_t &address, uint64_t &epoch) {
    static std::atomic<uint64_t> next_epoch{1};
    void *output = bump.acquire(sizeof(Report));
    if (output == nullptr) return -1;
    address = reinterpret_cast<uint64_t>(output);
    epoch = next_epoch.fetch_add(1, std::memory_order_relaxed);
    const uint64_t empty_epoch = 0;
    return api->copy_to_device(output, &empty_epoch, sizeof(empty_epoch));
}
// A report must not compete with per-task traces in the bounded async logger.
// This cold-path append uses the existing process log directory and no new knob.
inline void emit_report(const std::string &text) {
    const char *directory = std::getenv("ASCEND_PROCESS_LOG_PATH");
    if (directory == nullptr || directory[0] == '\0') {
        directory = HostLogger::get_instance().log_directory();
    }
    if (directory != nullptr && directory[0] != '\0') {
        const std::string path = std::string(directory) + "/queue_peaks." + std::to_string(getpid()) + ".log";
        std::ofstream output(path, std::ios::app);
        output << text;
        output.close();
        if (!output.fail()) return;
    }
    std::fwrite(text.data(), 1, text.size(), stderr);
    std::fflush(stderr);
}
inline void read_output(const HostApi *api, uint64_t address, uint64_t epoch, const char *runtime, int execution_rc) {
    const std::string key =
        std::string(" runtime=") + runtime + " pid=" + std::to_string(getpid()) + " epoch=" + std::to_string(epoch);
    if (address == 0 || epoch == 0) {
        emit_report("QPEAK_MISSING" + key + " reason=no_buffer\n");
        return;
    }
    Report report{};
    if (api->copy_from_device(&report, reinterpret_cast<const void *>(address), sizeof(report)) != 0 ||
        report.epoch != epoch || report.version != 1 || report.count > MAX_QUEUES) {
        emit_report("QPEAK_MISSING" + key + " reason=invalid_report\n");
        return;
    }
    std::ostringstream output;
    output << "QPEAK_BEGIN" << key << " count=" << report.count << " rc=" << execution_rc << '\n';
    for (uint32_t i = 0; i < report.count; ++i) {
        const auto &r = report.records[i];
        if (static_cast<uint32_t>(r.kind) > static_cast<uint32_t>(Kind::Mailbox)) {
            output << "QPEAK_MISSING" << key << " reason=invalid_kind\n";
            continue;
        }
        output << "QPEAK" << key << " queue=" << name(r.kind) << " index=" << r.index << " cap=" << r.capacity
               << " peak=" << r.peak << " spmd=" << r.spmd_peak << " normal=" << r.normal_peak
               << " classified=" << r.classified << '\n';
    }
    output << "QPEAK_END" << key << " count=" << report.count << " rc=" << execution_rc << '\n';
    emit_report(output.str());
}
}  // namespace simpler::queue_peaks
