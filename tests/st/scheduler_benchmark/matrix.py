# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The 33 fixed scheduling DAG configurations from D200_docs/benchmark/README.md."""

from dataclasses import asdict, dataclass, replace
from math import ceil

CASE_IDS = tuple(
    [f"C1-{i:02d}" for i in range(1, 9)] + [f"C2-{i:02d}" for i in range(1, 16)] + [f"C3-{i:02d}" for i in range(1, 11)]
)

# The same binary is compiled once for AIC and once for AIV. These loop counts
# are fixed across chip generations and runtimes.
SHORT_ITERS = 1
LONG_ITERS = 2048


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    family: int  # 1=chain, 2=parallel chains, 3=independent roots, 4=fan-in/out units
    task_count: int
    width: int = 0
    length: int = 0
    producers: int = 0
    consumers: int = 0
    repeats: int = 0
    core_mode: int = 0  # 0=AIC, 1=AIV, 2=AIC-first alternate, 3=AIV-first alternate
    work_mode: int = 0  # 0=short, 1=long, 2=1:4, 3=1:1, 4=4:1

    def manifest(self) -> dict:
        return asdict(self)


def expand_case(case_id: str, k_c: int, k_v: int) -> CaseSpec:
    if case_id not in CASE_IDS:
        raise ValueError(f"unknown scheduling benchmark case {case_id!r}")
    if k_c < 1 or k_v < 1:
        raise ValueError(f"executor counts must be positive, got K_C={k_c}, K_V={k_v}")

    group, suffix = case_id.split("-")
    index = int(suffix)
    if group == "C1":
        length = 2048 if index % 2 else 8192
        return CaseSpec(case_id, 1, length, length=length, core_mode=(index - 1) // 2)

    if group == "C2":
        if index in (9, 10):
            return CaseSpec(case_id, 3, 4096, core_mode=index - 9)
        if index in (11, 12):
            width = 2 * (k_c if index == 11 else k_v)
            length = max(32, ceil(4096 / width))
            return CaseSpec(
                case_id, 2, width * length, width=width, length=length, core_mode=0 if index == 11 else 1, work_mode=1
            )
        if index >= 13:
            width = 2 * k_v
            length = 10 * ceil(max(50, 4096 / width) / 10)
            return CaseSpec(case_id, 2, width * length, width=width, length=length, core_mode=1, work_mode=index - 11)
        core_mode = 0 if index <= 4 else 1
        factor_index = (index - 1) % 4
        k = k_c if core_mode == 0 else k_v
        width = ceil(k / 2) if factor_index == 0 else (1 << (factor_index - 1)) * k
        length = max(32, ceil(4096 / width))
        return CaseSpec(case_id, 2, width * length, width=width, length=length, core_mode=core_mode)

    fanout = (ceil(k_c / 2), k_c, 2 * k_c, 4 * k_c)
    producers, consumers = (
        (1, fanout[index - 1])
        if index <= 4
        else (1 << (index - 2), k_c)
        if index <= 8
        else (32 if index == 9 else 64, 64)
    )
    repeats = max(32, ceil(4096 / (producers + consumers + 1)))
    count = repeats * (producers + consumers) + repeats - 1
    return CaseSpec(case_id, 4, count, producers=producers, consumers=consumers, repeats=repeats)


def is_long(spec: CaseSpec, index: int) -> bool:
    if spec.work_mode == 1:
        return True
    if spec.work_mode == 2:
        return index % 5 == 0
    if spec.work_mode == 3:
        return index % 2 == 0
    if spec.work_mode == 4:
        return index % 5 != 0
    return False


def scale_case(spec: CaseSpec, factor: int) -> CaseSpec:
    if factor < 1:
        raise ValueError("task scale must be positive")
    if spec.family == 1:
        result = replace(spec, length=spec.length * factor, task_count=spec.task_count * factor)
    elif spec.family == 2:
        result = replace(spec, length=spec.length * factor, task_count=spec.task_count * factor)
    elif spec.family == 3:
        result = replace(spec, task_count=spec.task_count * factor)
    else:
        repeats = spec.repeats * factor
        result = replace(spec, repeats=repeats, task_count=repeats * (spec.producers + spec.consumers + 1) - 1)
    if result.task_count > 32768:
        raise ValueError(f"scaled graph exceeds 32768 tasks: {result.task_count}")
    return result


def task_roles(spec: CaseSpec) -> list[str]:
    if spec.family == 1:
        modes = ("aic", "aiv")
        if spec.core_mode < 2:
            return [modes[spec.core_mode]] * spec.task_count
        return [modes[(index + spec.core_mode - 2) % 2] for index in range(spec.task_count)]
    if spec.family in (2, 3):
        return ["aic" if spec.core_mode == 0 else "aiv"] * spec.task_count
    roles = []
    for unit in range(spec.repeats):
        roles.extend(["aic"] * (spec.producers + spec.consumers))
        if unit + 1 < spec.repeats:
            roles.append("dummy")
    return roles


def expected_edges(spec: CaseSpec) -> set[tuple[int, int]]:
    if spec.family == 1:
        return {(i - 1, i) for i in range(1, spec.task_count)}
    if spec.family == 2:
        return {(i - spec.width, i) for i in range(spec.width, spec.task_count)}
    if spec.family == 3:
        return set()
    edges = set()
    offset = 0
    gate = None
    for unit in range(spec.repeats):
        producers = range(offset, offset + spec.producers)
        consumers = range(offset + spec.producers, offset + spec.producers + spec.consumers)
        if gate is not None:
            edges.update((gate, producer) for producer in producers)
        edges.update((producer, consumer) for producer in producers for consumer in consumers)
        offset += spec.producers + spec.consumers
        if unit + 1 < spec.repeats:
            edges.update((consumer, offset) for consumer in consumers)
            gate = offset
            offset += 1
    assert offset == spec.task_count
    return edges
