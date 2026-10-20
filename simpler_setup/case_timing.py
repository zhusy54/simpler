# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only phase accounting for scene and benchmark execution."""

from contextlib import contextmanager
from time import perf_counter_ns


@contextmanager
def measure_phase(timings, name):
    """Accumulate nanoseconds, including a phase that exits with an error."""
    if timings is None:
        yield
        return
    start = perf_counter_ns()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0) + perf_counter_ns() - start
