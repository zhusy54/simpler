# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE compatibility execution of the original A5 vector graph."""

from simpler_setup import SceneTestCase, scene_test
from tests.st.a5.host_build_graph.vector_example.test_vector_example import (
    TestVectorExampleHostBuildGraphA5 as _A5Vector,
)


@scene_test(level=2, runtime="host_build_graph")
class TestVectorExampleWse(_A5Vector):
    CASES = [{"name": "default", "platforms": ["wsesim"], "params": {}}]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
