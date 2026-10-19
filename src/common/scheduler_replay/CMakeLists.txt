# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(simpler_scheduler_replay LANGUAGES CXX)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(SIMPLER_SRC "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(SIMPLER_ROOT "${SIMPLER_SRC}/.." ABSOLUTE)
include(${SIMPLER_ROOT}/cmake/profiling_config.cmake)
set(SIMPLER_DFX 0)
set(rt ${SIMPLER_SRC}/a5/runtime/host_build_graph)
add_executable(replay_a5_hbg hbg.cpp)
target_include_directories(replay_a5_hbg PRIVATE
    ${rt}/runtime ${rt}/common ${rt}/host
    ${SIMPLER_SRC}/a5/platform/include
    ${SIMPLER_SRC}/common/platform/include
    ${SIMPLER_SRC}/common/platform/sim/aicpu
    ${SIMPLER_SRC}/common/host_build_graph
    ${SIMPLER_SRC}/common/host_build_graph/host
    ${SIMPLER_SRC}/common/log/include
    ${SIMPLER_SRC}/common/task_interface
    ${SIMPLER_SRC}/common)
simpler_configure_profiling(replay_a5_hbg)

foreach(arch IN ITEMS a5 a2a3)
    set(rt ${SIMPLER_SRC}/${arch}/runtime/tensormap_and_ringbuffer)
    set(target replay_${arch}_tmr)
    add_executable(${target} tmr.cpp
        ${rt}/runtime/scheduler/scheduler.cpp
        ${rt}/runtime/scheduler/scheduler_dispatch.cpp
        ${rt}/runtime/scheduler/scheduler_completion.cpp
        ${rt}/runtime/scheduler/scheduler_cold_path.cpp
        ${rt}/runtime/shared/shared_memory.cpp
        ${rt}/runtime/shared/runtime_init.cpp
        ${rt}/runtime/shared/tensormap.cpp
        ${rt}/runtime/orchestrator.cpp
        ${rt}/runtime/ring_buffer.cpp
        platform.cpp)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/host ${CMAKE_CURRENT_SOURCE_DIR}
        ${rt}/runtime ${rt}/common ${rt}/orchestration
        ${SIMPLER_SRC}/${arch}/platform/include
        ${SIMPLER_SRC}/common/platform/include
        ${SIMPLER_SRC}/common/platform/sim/aicpu
        ${SIMPLER_SRC}/common/tensormap_and_ringbuffer
        ${SIMPLER_SRC}/common/log/include
        ${SIMPLER_SRC}/common/task_interface
        ${SIMPLER_SRC}/common)
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
    if(APPLE)
        target_link_options(${target} PRIVATE -Wl,-dead_strip)
    else()
        target_link_options(${target} PRIVATE -Wl,--gc-sections)
    endif()
    simpler_configure_profiling(${target})
    target_link_libraries(${target} PRIVATE Threads::Threads)
endforeach()
target_sources(replay_a5_tmr PRIVATE a5_registers.cpp)

foreach(arch IN ITEMS a5 a2a3)
    set(rt ${SIMPLER_SRC}/${arch}/runtime/host_build_graph)
    set(target replay_${arch}_hbg_aicpu)
    file(GLOB shared_sources CONFIGURE_DEPENDS ${SIMPLER_SRC}/common/host_build_graph/shared/*.cpp)
    list(REMOVE_ITEM shared_sources ${SIMPLER_SRC}/common/host_build_graph/shared/assert_support.cpp)
    add_executable(${target} hbg_aicpu.cpp platform.cpp
        ${rt}/runtime/scheduler/scheduler.cpp
        ${rt}/runtime/scheduler/scheduler_dispatch.cpp
        ${rt}/runtime/scheduler/scheduler_completion.cpp
        ${rt}/runtime/scheduler/scheduler_cold_path.cpp
        ${shared_sources}
        ${SIMPLER_SRC}/common/host_build_graph/host/tensormap.cpp)
    target_include_directories(${target} PRIVATE
        ${rt}/runtime ${rt}/common ${rt}/host
        ${SIMPLER_SRC}/${arch}/platform/include
        ${SIMPLER_SRC}/common/platform/include
        ${SIMPLER_SRC}/common/platform/sim/aicpu
        ${SIMPLER_SRC}/common/host_build_graph
        ${SIMPLER_SRC}/common/host_build_graph/host
        ${SIMPLER_SRC}/common/log/include
        ${SIMPLER_SRC}/common/task_interface ${SIMPLER_SRC}/common)
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
    if(APPLE)
        target_link_options(${target} PRIVATE -Wl,-dead_strip)
    else()
        target_link_options(${target} PRIVATE -Wl,--gc-sections)
    endif()
    simpler_configure_profiling(${target})
endforeach()
target_sources(replay_a5_hbg_aicpu PRIVATE a5_registers.cpp)
