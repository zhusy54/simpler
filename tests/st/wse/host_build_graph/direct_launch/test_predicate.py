# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A skipped Kernel still completes its graph task and releases successors."""

from simpler_setup import scene_test
from tests.st.a5.host_build_graph.predicated_dispatch.test_predicated_dispatch import (
    TestPredicatedDispatchA5 as _A5Predicate,
)


@scene_test(level=2, runtime="host_build_graph")
class TestWsePredicate(_A5Predicate):
    CASES = [
        {"name": "predicate_false", "platforms": ["wsesim"], "params": {"case": 1}},
        {"name": "predicate_true", "platforms": ["wsesim"], "params": {"case": 2}},
    ]
