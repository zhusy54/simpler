"""C1–C3 scheduling DAGs on A2/A3 host_build_graph."""

from simpler_setup import SceneTestCase, scene_test
from tests.st.scheduler_benchmark.scene import SchedulingBenchmarkScene, case_list


@scene_test(level=2, runtime="host_build_graph")
class TestSchedulerBenchmarkHbgA2A3(SchedulingBenchmarkScene):
    CASES = case_list(["a2a3sim", "a2a3"])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
