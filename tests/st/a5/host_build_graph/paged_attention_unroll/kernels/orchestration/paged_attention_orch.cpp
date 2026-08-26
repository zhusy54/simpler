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

#include <algorithm>
#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr uint64_t kUnrollBlockCount = 64;
constexpr int32_t kQkMatmulFunc = 0;
constexpr int32_t kSoftmaxPrepareFunc = 1;
constexpr int32_t kPvMatmulFunc = 2;
constexpr int32_t kOnlineUpdateFunc = 3;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void build_paged_attention_unroll_graph(const ChipTaskArgs &orch_args) {
    const uint64_t batch = orch_args.tensor(0).ref().shapes[0];
    const uint64_t num_heads = orch_args.tensor(0).ref().shapes[1];
    const uint64_t head_dim = orch_args.tensor(0).ref().shapes[2];
    const DataType data_type = orch_args.tensor(0).ref().dtype;
    const uint64_t block_size = orch_args.tensor(1).ref().shapes[1];
    const uint64_t block_num = orch_args.tensor(3).ref().shapes[1];
    const uint64_t scale_value = orch_args.scalar(0);
    const uint64_t q_tile = std::min(num_heads, static_cast<uint64_t>(128));
    const uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;

    void *query_ptr = orch_args.tensor(0).ref().data_as<void>();
    void *key_cache_ptr = orch_args.tensor(1).ref().data_as<void>();
    void *value_cache_ptr = orch_args.tensor(2).ref().data_as<void>();
    void *out_ptr = orch_args.tensor(5).ref().data_as<void>();
    const uint64_t total_blocks = orch_args.tensor(1).ref().shapes[0];

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t cache_shapes[2] = {static_cast<uint32_t>(total_blocks * block_size), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    simpler::hbg::Tensor query = simpler::hbg::make_tensor_external(query_ptr, query_shapes, 2, data_type);
    simpler::hbg::Tensor key_cache = simpler::hbg::make_tensor_external(key_cache_ptr, cache_shapes, 2, data_type);
    simpler::hbg::Tensor value_cache = simpler::hbg::make_tensor_external(value_cache_ptr, cache_shapes, 2, data_type);
    simpler::hbg::Tensor out = simpler::hbg::make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);

    uint32_t block_table_shapes[2] = {static_cast<uint32_t>(batch), static_cast<uint32_t>(block_num)};
    simpler::hbg::Tensor block_table = simpler::hbg::make_tensor_external(
        orch_args.tensor(3).ref().data_as<void>(), block_table_shapes, 2, DataType::INT32, false
    );
    uint32_t context_lens_shapes[1] = {static_cast<uint32_t>(batch)};
    simpler::hbg::Tensor context_lens = simpler::hbg::make_tensor_external(
        orch_args.tensor(4).ref().data_as<void>(), context_lens_shapes, 1, DataType::INT32, false
    );

    uint32_t tile_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t scalar_shapes[1] = {static_cast<uint32_t>(q_tile)};
    TensorCreateInfo tile_create_info(tile_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo scalar_create_info(scalar_shapes, 1, DataType::FLOAT32);

    for (uint64_t batch_index = 0; batch_index < batch; ++batch_index) {
        uint32_t context_index[1] = {static_cast<uint32_t>(batch_index)};
        const uint64_t context_length = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, context_index));
        const uint64_t batch_block_count = (context_length + block_size - 1) / block_size;

        for (uint64_t q_index = 0; q_index < q_loop; ++q_index) {
            SIMPLER_SCOPE() {
                const uint64_t query_offset = batch_index * num_heads + q_index * q_tile;
                uint32_t view_offsets[2] = {static_cast<uint32_t>(query_offset), 0};
                simpler::hbg::Tensor query_view = query.view(tile_shapes, view_offsets);
                simpler::hbg::Tensor out_view = out.view(tile_shapes, view_offsets, true);

                TaskOutputTensors accumulators =
                    alloc_tensors(tile_create_info, scalar_create_info, scalar_create_info);
                const simpler::hbg::Tensor &output_accumulator = accumulators.get_ref(0);
                const simpler::hbg::Tensor &sum_accumulator = accumulators.get_ref(1);
                const simpler::hbg::Tensor &max_accumulator = accumulators.get_ref(2);

                CoreTaskArgs qk_args;
                CoreTaskArgs softmax_args;
                CoreTaskArgs pv_args;
                CoreTaskArgs update_args;

                for (uint64_t block_offset = 0; block_offset < batch_block_count; block_offset += kUnrollBlockCount) {
                    const uint64_t group_block_count = std::min(kUnrollBlockCount, batch_block_count - block_offset);
                    const uint64_t last_block_start = (block_offset + group_block_count - 1) * block_size;
                    const uint64_t last_block_valid_length = std::min(block_size, context_length - last_block_start);

                    uint32_t score_shapes[2] = {
                        static_cast<uint32_t>(q_tile), static_cast<uint32_t>(group_block_count * block_size)
                    };
                    TensorCreateInfo score_create_info(score_shapes, 2, DataType::FLOAT32);

                    qk_args.reset();
                    qk_args.add_input(query_view, key_cache, block_table);
                    qk_args.add_output(score_create_info);
                    qk_args.add_scalar(group_block_count, batch_index * block_num + block_offset);
                    TaskOutputTensors qk_outputs = rt_submit_aic_task(kQkMatmulFunc, qk_args);
                    const simpler::hbg::Tensor &scores = qk_outputs.get_ref(0);

                    TensorCreateInfo probabilities_create_info(score_shapes, 2, data_type);
                    softmax_args.reset();
                    softmax_args.add_input(scores);
                    softmax_args.add_output(probabilities_create_info, scalar_create_info, scalar_create_info);
                    softmax_args.add_scalar(scale_value, group_block_count, last_block_valid_length);
                    TaskOutputTensors softmax_outputs = rt_submit_aiv_task(kSoftmaxPrepareFunc, softmax_args);
                    const simpler::hbg::Tensor &probabilities = softmax_outputs.get_ref(0);
                    const simpler::hbg::Tensor &group_max = softmax_outputs.get_ref(1);
                    const simpler::hbg::Tensor &group_sum = softmax_outputs.get_ref(2);

                    pv_args.reset();
                    pv_args.add_input(probabilities, value_cache, block_table);
                    pv_args.add_output(tile_create_info);
                    pv_args.add_scalar(group_block_count, batch_index * block_num + block_offset);
                    TaskOutputTensors pv_outputs = rt_submit_aic_task(kPvMatmulFunc, pv_args);
                    const simpler::hbg::Tensor &group_output = pv_outputs.get_ref(0);

                    const uint64_t is_first = block_offset == 0 ? 1 : 0;
                    const uint64_t is_last = block_offset + group_block_count >= batch_block_count ? 1 : 0;
                    update_args.reset();
                    update_args.add_input(group_max, group_sum, group_output);
                    update_args.add_inout(max_accumulator, sum_accumulator, output_accumulator, out_view);
                    update_args.add_scalar(is_first, is_last);
                    rt_submit_aiv_task(kOnlineUpdateFunc, update_args);
                }
            }
        }
    }
}

}  // extern "C"
