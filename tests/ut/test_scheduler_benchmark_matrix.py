# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Matrix expansion and independent edge oracle for the fixed C1–C3 cases."""

from collections import Counter

from tests.st.scheduler_benchmark.matrix import (
    CASE_IDS,
    TASK_ITERS,
    expand_case,
    expected_edges,
    is_long,
    task_roles,
    work_iterations,
)


def test_all_33_configs_have_expected_size_and_edges():
    assert len(CASE_IDS) == len(set(CASE_IDS)) == 33
    for k_c, k_v in ((8, 16), (24, 48), (32, 64), (36, 72)):
        for case_id in CASE_IDS:
            spec = expand_case(case_id, k_c, k_v)
            roles = task_roles(spec)
            edges = expected_edges(spec)
            assert len(roles) == spec.task_count
            assert spec.task_count >= 2048
            assert all(pred < succ for pred, succ in edges)
            if spec.family == 1:
                assert len(edges) == spec.task_count - 1
            elif spec.family == 2:
                assert len(edges) == spec.width * (spec.length - 1)
                assert spec.task_count >= 4096
            elif spec.family == 3:
                assert not edges
            else:
                assert spec.task_count >= 4095
                assert roles.count("dummy") == spec.repeats - 1
                assert len(edges) == spec.repeats * spec.producers * spec.consumers + (spec.repeats - 1) * (
                    spec.producers + spec.consumers
                )


def test_reference_points_and_mixed_work_are_exact():
    for k_c, k_v in ((8, 16), (24, 48), (32, 64), (36, 72)):
        for index, producers, consumers, repeats, count in (
            (1, 1, 1, 1366, 4097),
            (2, 1, 4, 683, 4097),
            (3, 1, 8, 410, 4099),
            (4, 1, 32, 121, 4113),
            (5, 8, 1, 410, 4099),
            (6, 16, 1, 228, 4103),
            (7, 32, 1, 121, 4113),
            (8, 64, 1, 63, 4157),
        ):
            spec = expand_case(f"C3-{index:02d}", k_c, k_v)
            assert (spec.producers, spec.consumers, spec.repeats, spec.task_count) == (
                producers,
                consumers,
                repeats,
                count,
            )
    assert (expand_case("C2-01", 24, 48).width, expand_case("C2-01", 24, 48).length) == (12, 342)
    assert (expand_case("C2-08", 36, 72).width, expand_case("C2-08", 36, 72).length) == (288, 32)
    assert (expand_case("C3-10", 32, 64).producers, expand_case("C3-10", 32, 64).consumers) == (64, 64)
    for case_id, ratio in (("C2-13", (1, 4)), ("C2-14", (1, 1)), ("C2-15", (4, 1))):
        spec = expand_case(case_id, 32, 64)
        counts = Counter(is_long(spec, index) for index in range(spec.task_count))
        assert counts[True] * ratio[1] == counts[False] * ratio[0]


def test_fanin_unit_boundaries_have_only_gate_connections():
    spec = expand_case("C3-10", 32, 64)
    edges = expected_edges(spec)
    gate = spec.producers + spec.consumers
    next_producers = range(gate + 1, gate + 1 + spec.producers)
    first_consumers = range(spec.producers, gate)
    assert {(consumer, gate) for consumer in first_consumers} <= edges
    assert {(gate, producer) for producer in next_producers} <= edges
    assert all(
        (producer, consumer) not in edges
        for producer in range(spec.producers)
        for consumer in range(gate + 1 + spec.producers, gate + 1 + spec.producers + spec.consumers)
    )


def test_compute_granularity_preserves_long_and_mixed_cases():
    preserved = {f"C2-{i:02d}" for i in range(11, 16)}
    assert len(set(CASE_IDS) - preserved) == 28
    for case_id in CASE_IDS:
        spec = expand_case(case_id, 32, 64)
        for index, role in enumerate(task_roles(spec)):
            if role == "dummy":
                continue
            expected = (2048 if is_long(spec, index) else 1) if case_id in preserved else TASK_ITERS
            assert work_iterations(spec, index) == expected
