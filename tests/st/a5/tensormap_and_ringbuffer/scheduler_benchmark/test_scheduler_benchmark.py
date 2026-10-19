"""C1–C3 scheduling DAGs on A5 tensormap_and_ringbuffer."""

from simpler_setup import SceneTestCase, scene_test
from tests.st.scheduler_benchmark.scene import SchedulingBenchmarkScene, case_list


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSchedulerBenchmarkTmrA5(SchedulingBenchmarkScene):
    CASES = case_list(["a5sim", "a5"])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
