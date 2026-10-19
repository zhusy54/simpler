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

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace simpler::replay {

class SubmissionActor {
public:
    explicit SubmissionActor(std::function<void()> body) :
        worker_([this, body = std::move(body)] {
            active_ = this;
            try {
                await_turn();
                body();
            } catch (const Cancelled &) {
            } catch (...) {
                error_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                finished_ = true;
                running_ = false;
            }
            condition_.notify_all();
        }) {}
    ~SubmissionActor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        worker_.join();
    }
    SubmissionActor(const SubmissionActor &) = delete;
    SubmissionActor &operator=(const SubmissionActor &) = delete;
    void resume() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!finished_) {
            running_ = true;
            condition_.notify_all();
            condition_.wait(lock, [this] {
                return !running_;
            });
        }
        if (error_) std::rethrow_exception(error_);
    }
    static void yield() {
        auto *actor = active_;
        if (!actor) throw std::runtime_error("native scheduler blocked during a zero-cost scheduling pass");
        std::unique_lock<std::mutex> lock(actor->mutex_);
        actor->running_ = false;
        actor->condition_.notify_all();
        actor->condition_.wait(lock, [actor] {
            return actor->running_ || actor->stopping_;
        });
        if (actor->stopping_) throw Cancelled{};
    }

private:
    struct Cancelled {};
    void await_turn() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
            return running_ || stopping_;
        });
        if (stopping_) throw Cancelled{};
    }
    inline static thread_local SubmissionActor *active_{};
    std::mutex mutex_;
    std::condition_variable condition_;
    bool running_{};
    bool finished_{};
    bool stopping_{};
    std::exception_ptr error_;
    std::thread worker_;
};

}  // namespace simpler::replay
