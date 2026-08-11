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

#include "host/chip_swimlane_collector.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

void *test_alloc(size_t size) { return std::calloc(1, size); }

int test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::filesystem::path test_output(const char *suffix) {
    return std::filesystem::temp_directory_path() /
           ("chip_swimlane_collector_test_" + std::to_string(::getpid()) + "_" + suffix);
}

void collect_one_aicore_record(ChipSwimlaneCollector &collector, uint64_t task_id) {
    ChipSwimlaneAicoreTaskBuffer buffer{};
    buffer.count = 1;
    buffer.records[0].task_token_raw = task_id;
    buffer.records[0].reg_task_id = static_cast<uint32_t>(task_id);
    buffer.records[0].start_time = 100;
    buffer.records[0].end_time = 200;

    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.host_buffer_ptr = &buffer;
    collector.on_buffer_collected(info, 0);
}

void collect_task_and_resolve_records(ChipSwimlaneCollector &collector, uint64_t task_id) {
    ChipSwimlaneAicoreTaskBuffer buffer{};
    buffer.count = 2;
    buffer.records[0].task_token_raw = task_id;
    buffer.records[0].reg_task_id = static_cast<uint32_t>(task_id);
    buffer.records[0].start_time = 100;
    buffer.records[0].end_time = 200;
    buffer.records[1].task_token_raw = task_id;
    buffer.records[1].reg_task_id = UINT32_C(0x80000000) | static_cast<uint32_t>(task_id);
    buffer.records[1].start_time = 210;
    buffer.records[1].end_time = 240;

    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.host_buffer_ptr = &buffer;
    collector.on_buffer_collected(info, 0);
}

void initialize_memory_ops(ChipSwimlaneCollector &collector) {
    collector.start([](std::function<void()> fn) {
        return std::thread(std::move(fn));
    });
    collector.stop();
}

}  // namespace

TEST(ChipSwimlaneCollectorTest, StrictExportPublishesOnlyCompleteAicoreRecords) {
    const std::filesystem::path output = test_output("complete");
    std::filesystem::remove_all(output);

    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            1, 1, 0, ChipSwimlaneLevel::AICORE_TIMING, test_alloc, nullptr, test_free, output.string()
        ),
        0
    );
    initialize_memory_ops(collector);
    collector.set_strict_validation(true);
    collect_one_aicore_record(collector, 7);

    auto *state = get_aicore_buffer_state(collector.get_chip_swimlane_setup_device_ptr(), 1, 0);
    state->head.total_record_count = 1;
    state->head.dropped_record_count = 0;

    EXPECT_TRUE(collector.reconcile_counters());
    EXPECT_EQ(collector.export_swimlane_json(), 0);
    EXPECT_TRUE(std::filesystem::is_regular_file(output / "chip_swimlane_records.json"));
    EXPECT_FALSE(std::filesystem::exists(output / "chip_swimlane_records.json.tmp"));

    collector.finalize(nullptr, test_free);
    std::filesystem::remove_all(output);
}

TEST(ChipSwimlaneCollectorTest, StrictExportRejectsDroppedAicoreRecords) {
    const std::filesystem::path output = test_output("dropped");
    std::filesystem::remove_all(output);

    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            1, 1, 0, ChipSwimlaneLevel::AICORE_TIMING, test_alloc, nullptr, test_free, output.string()
        ),
        0
    );
    initialize_memory_ops(collector);
    collector.set_strict_validation(true);
    collect_one_aicore_record(collector, 9);

    auto *state = get_aicore_buffer_state(collector.get_chip_swimlane_setup_device_ptr(), 1, 0);
    state->head.total_record_count = 2;
    state->head.dropped_record_count = 1;

    EXPECT_FALSE(collector.reconcile_counters());
    EXPECT_NE(collector.export_swimlane_json(), 0);
    EXPECT_FALSE(std::filesystem::exists(output / "chip_swimlane_records.json"));
    EXPECT_FALSE(std::filesystem::exists(output / "chip_swimlane_records.json.tmp"));

    collector.finalize(nullptr, test_free);
    std::filesystem::remove_all(output);
}

TEST(ChipSwimlaneCollectorTest, ExportSeparatesAicoreResolveRecords) {
    const std::filesystem::path output = test_output("resolve");
    std::filesystem::remove_all(output);

    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            1, 1, 0, ChipSwimlaneLevel::AICORE_TIMING, test_alloc, nullptr, test_free, output.string()
        ),
        0
    );
    initialize_memory_ops(collector);
    collect_task_and_resolve_records(collector, 7);

    auto *state = get_aicore_buffer_state(collector.get_chip_swimlane_setup_device_ptr(), 1, 0);
    state->head.total_record_count = 2;

    ASSERT_TRUE(collector.reconcile_counters());
    ASSERT_EQ(collector.export_swimlane_json(), 0);
    std::ifstream input(output / "chip_swimlane_records.json");
    const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    EXPECT_NE(json.find("\"aicore_tasks\": [\n    [0, 7, 7, 100, 200, 0]\n  ]"), std::string::npos);
    EXPECT_NE(json.find("\"aicore_resolve_phases\": [\n    [0, 7, 210, 240]\n  ]"), std::string::npos);

    collector.finalize(nullptr, test_free);
    std::filesystem::remove_all(output);
}
