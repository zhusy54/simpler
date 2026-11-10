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
#include "device_runner_base.h"
#include "sim_run_completion.h"
#include "wse_launch.h"

class DeviceRunner : public SimDeviceRunnerBase {
public:
    DeviceRunner();
    ~DeviceRunner() override;
    const char *cpu_sim_target_family() const override { return "wsesim"; }
    int prepare_launch_shape(Runtime &, const CallConfig &) override;
    int prepare_execution(
        Runtime &, const CallConfig &, uint32_t, const NativeRunIdentity &, std::unique_ptr<PreparedExecution> *
    ) override;
    void abandon_prepared_execution(PreparedExecution &) noexcept override;
    int poll_execution(const ActiveExecution &) override;
    DrainOutcome drain_execution(ActiveExecution &) override;
    int finalize() override;
    void arm_host_dep_gen_capture(bool enable) override;

private:
    LaunchTransactionResult launch_execution_impl(PreparedExecution &, LaunchPermit) override;
    int ensure_binaries_loaded() override;
    int invoke_device_register(const RegisterCallableArgs &) override { return PTO_RUNTIME_ERR_UNSUPPORTED; }
    void cleanup_run() noexcept;
    void unload_executor();
    struct ActiveRun;
    std::unique_ptr<ActiveRun> active_run_;
    simpler::common::sim_host::SimRunCompletion completion_;
    decltype(&wse_create_run) create_run_{nullptr};
    decltype(&wse_destroy_run) destroy_run_{nullptr};
    decltype(&wse_execute_pe) execute_pe_{nullptr};
    decltype(&wse_run_events) run_events_{nullptr};
    decltype(&wse_completed_tasks) completed_tasks_{nullptr};
};
