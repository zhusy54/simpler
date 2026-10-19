"""Fixed existing-workload cases in the scheduling benchmark matrix."""

import ast
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadSpec:
    case_id: str
    directory: str
    case: str
    params: dict

    def manifest(self) -> dict:
        return asdict(self)

    def test_file(self, repo: Path, arch: str, runtime: str) -> Path:
        for root in ("tests/st", "examples"):
            directory = repo / root / arch / runtime / self.directory
            matches = sorted(directory.glob("test_*.py"))
            if len(matches) == 1:
                return matches[0]
        raise FileNotFoundError(f"no test file for {self.case_id} on {arch}/{runtime}")


_PA1 = {
    "batch": 256,
    "num_heads": 16,
    "kv_head_num": 1,
    "head_dim": 128,
    "block_size": 128,
    "context_len": 8192,
    "max_model_len": 32768,
    "dtype": "bfloat16",
}
_PA2 = {**_PA1, "batch": 64, "num_heads": 64, "block_size": 64}

WORKLOADS = (
    WorkloadSpec(
        "W1-01", "alternating_matmul_add", "Case1", {"batch": 500, "M": 4, "N": 4, "matmul_batch": 4, "add_batch": 4}
    ),
    WorkloadSpec(
        "W2-01",
        "benchmark_bgemm",
        "Case0",
        {"matmul_add_task_num": 500, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
    ),
    WorkloadSpec("W3-01", "paged_attention_unroll", "Case1", _PA1),
    WorkloadSpec("W3-02", "paged_attention_unroll", "Case2", _PA2),
    WorkloadSpec("W3-03", "paged_attention_unroll_manual_scope", "Case1", _PA1),
    WorkloadSpec("W3-04", "paged_attention_unroll_manual_scope", "Case2", _PA2),
    WorkloadSpec("W4-01", "batch_paged_attention", "Case1", _PA1),
    WorkloadSpec(
        "W5-01", "qwen3_14b_decode", "runtime_case", {"batch": 16, "seq_len": 3500, "seed": 1234, "layers": 40}
    ),
)
WORKLOAD_BY_ID = {spec.case_id: spec for spec in WORKLOADS}


def workload_case(spec: WorkloadSpec, runtime: str) -> str:
    if spec.case_id == "W5-01":
        return "GraphExecutionBatch16Seq3500" if runtime == "host_build_graph" else "StressBatch16Seq3500"
    return spec.case


def supported_platforms(test_file: Path, case_name: str) -> tuple[str, ...]:
    if case_name in ("GraphExecutionBatch16Seq3500", "StressBatch16Seq3500"):
        return ("a2a3", "a5")
    tree = ast.parse(test_file.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        values = {key.value: value for key, value in zip(node.keys, node.values) if isinstance(key, ast.Constant)}
        if "name" in values and isinstance(values["name"], ast.Constant) and values["name"].value == case_name:
            return tuple(ast.literal_eval(values["platforms"]))
    raise ValueError(f"case {case_name!r} is not declared by {test_file}")
