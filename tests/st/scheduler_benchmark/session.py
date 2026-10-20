# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sequential benchmark cases sharing one process and one Worker per phase."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import sys
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from simpler_setup.case_timing import measure_phase
from tests.st.scheduler_benchmark.matrix import CASE_IDS, expand_case, scale_case
from tests.st.scheduler_benchmark.workloads import WORKLOAD_BY_ID, supported_platforms, workload_case

EXECUTION_MODEL = "persistent_process_worker_per_phase_v1"
_REPO = Path(__file__).resolve().parents[3]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _load_module(path):
    relative = Path(path).resolve().relative_to(_REPO)
    name = "_scheduler_benchmark_" + "_".join(relative.with_suffix("").parts)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import benchmark scene {relative}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


class HostLog:
    """Read each bounded, flushed slice of the process-owned log once."""

    def __init__(self, directory):
        from _task_interface import (  # noqa: PLC0415
            _flush_host_log,
            _host_log_dropped_records,
            _initialize_host_log,
            _set_host_log_directory,
        )

        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        bound = Path(_set_host_log_directory(str(self.directory))).resolve()
        if bound != self.directory:
            raise RuntimeError(f"process Host log is already bound to {bound}")
        if not _initialize_host_log(25):
            raise RuntimeError("cannot initialize benchmark Host log")
        self._flush = _flush_host_log
        self._drops = _host_log_dropped_records

    def checkpoint(self):
        if not self._flush(1000):
            raise ValueError("Host log flush did not complete")
        return {
            "offsets": {str(p): p.stat().st_size for p in self.directory.glob("host.*.log")},
            "drops": int(self._drops()),
        }

    def read_since(self, before):
        after = self.checkpoint()
        lines = []
        for name, end in sorted(after["offsets"].items()):
            start = before["offsets"].get(name, 0)
            if end < start:
                raise ValueError("Host log was truncated during a case")
            with Path(name).open("rb") as stream:
                stream.seek(start)
                lines.extend(stream.read(end - start).decode("utf-8", errors="replace").splitlines(True))
        return lines, after


class BenchmarkSession:
    def __init__(self, platform, runtime, device, k_c, k_v, directory, *, task_scale=1):
        self.platform, self.runtime, self.device = platform, runtime, device
        self.arch = "a5" if platform.startswith("a5") else "a2a3"
        self.k_c, self.k_v, self.task_scale = k_c, k_v, task_scale
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self.log = HostLog(self.directory / "host_logs")
        self.worker = None
        self.generation = 0
        self.adapters = {}
        self.handles = {}
        self.previous_handles = {}
        self.aborted = None
        self.unsafe_device = False
        self._original_counts = {k: os.environ.get(k) for k in ("SIMPLER_SCHED_BENCH_K_C", "SIMPLER_SCHED_BENCH_K_V")}
        os.environ["SIMPLER_SCHED_BENCH_K_C"] = str(k_c)
        os.environ["SIMPLER_SCHED_BENCH_K_V"] = str(k_v)

    def _prepare(self, case_id):
        if case_id in self.adapters:
            return self.adapters[case_id]
        if case_id in CASE_IDS:
            path = _REPO / "tests/st" / self.arch / self.runtime / "scheduler_benchmark/test_scheduler_benchmark.py"
            name = case_id
        else:
            spec = WORKLOAD_BY_ID[case_id]
            path = spec.test_file(_REPO, self.arch, self.runtime)
            name = workload_case(spec, self.runtime)
            if self.platform not in supported_platforms(path, name):
                self.adapters[case_id] = None
                return None
        module = _load_module(path)
        if case_id == "W5-01":
            driver = module._driver()
            base = driver._load_tmr_driver() if hasattr(driver, "_load_tmr_driver") else driver
            chip, spec = base.build_callable(
                self.platform,
                runtime=self.runtime,
                orchestration_source=getattr(driver, "ORCHESTRATION_SOURCE", None),
            )
            adapter = {"driver": driver, "class": None, "chip": chip, "callable_spec": spec}
        else:
            from simpler_setup.scene_test import SceneTestCase  # noqa: PLC0415

            matches = [
                (cls, case)
                for cls in vars(module).values()
                if isinstance(cls, type) and cls.__module__ == module.__name__ and issubclass(cls, SceneTestCase)
                for case in getattr(cls, "CASES", [])
                if case["name"] == name and self.platform in case["platforms"]
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one scene for {case_id}, found {len(matches)}")
            cls, case = matches[0]
            if cls._st_level != 2:
                raise ValueError(f"{case_id}: persistent benchmark requires L2")
            from simpler_setup.scene_test import _class_wants_sdma  # noqa: PLC0415

            if _class_wants_sdma(cls) or getattr(cls, "_st_collect_across_runs", False):
                raise ValueError(f"{case_id}: incompatible Worker capabilities")
            chip = cls.compile_chip_callable(self.platform)
            case = deepcopy(case)
            if case_id in CASE_IDS:
                case["params"]["task_scale"] = self.task_scale
            adapter = {"class": cls, "case": case, "chip": chip, "callable_spec": cls.CALLABLE}
        raw = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
        adapter["sha256"] = hashlib.sha256(raw).hexdigest()
        self.adapters[case_id] = adapter
        return adapter

    def _start_worker(self):
        from simpler.worker import Worker  # noqa: PLC0415

        self.generation += 1
        self.worker = Worker(level=2, platform=self.platform, runtime=self.runtime, device_id=self.device)
        self.worker.init()

    def _close_worker(self):
        try:
            if self.worker is not None:
                self.worker.close()
        except BaseException:
            self.unsafe_device = True
            raise
        finally:
            self.worker = None
            self.handles.clear()
            for cls, (existed, previous) in self.previous_handles.items():
                if existed:
                    cls._st_l2_handle = previous
                elif "_st_l2_handle" in cls.__dict__:
                    delattr(cls, "_st_l2_handle")
            self.previous_handles.clear()

    def _execute(self, case_id, adapter, rounds, deps, swimlane, prefix, timings):
        if adapter["class"] is None:
            return adapter["driver"].run(
                [self.device],
                self.platform,
                rounds=rounds,
                skip_golden=True,
                enable_dep_gen=deps,
                enable_chip_swimlane=swimlane,
                worker=self.worker,
                output_prefix=str(prefix) if deps or swimlane else "",
                finalize_outputs=False,
                phase_timings=timings,
            )
        cls = adapter["class"]
        if cls not in self.handles:
            with measure_phase(timings, "register"):
                self.previous_handles[cls] = ("_st_l2_handle" in cls.__dict__, cls.__dict__.get("_st_l2_handle"))
                self.handles[cls] = self.worker.register(adapter["chip"])
        cls._st_l2_handle = self.handles[cls]
        cls()._run_and_validate_l2(
            self.worker,
            adapter["chip"],
            adapter["case"],
            rounds=rounds,
            skip_golden=True,
            enable_dep_gen=deps,
            enable_chip_swimlane=swimlane,
            output_prefix=str(prefix) if deps or swimlane else "",
            phase_timings=timings,
        )

    def _resources(self):
        rss = None
        try:
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            pass
        committed = None
        query = getattr(self.worker, "committed_device_memory", None)
        if query is not None and self.aborted is None:
            committed = query()
        return {"host_rss_bytes": rss, "runtime_committed_device_bytes": committed}

    def _run_case(self, case_id, adapter, phase, rounds, deps, swimlane, exclude_first):
        from tests.st.scheduler_benchmark.run_cases import _timings_from_lines  # noqa: PLC0415

        directory = self.directory / phase / case_id
        directory.mkdir(parents=True, exist_ok=False)
        log_path = directory / "case.log"
        start, started = perf_counter_ns(), utc_now()
        timings = {}
        result = {
            "status": "pending",
            "exit_code": None,
            "measurement_mode": "diagnostic" if deps or swimlane else "timing",
            "golden_validation": "skipped",
            "rounds": [],
            "diagnostic_paths": [],
            "diagnostic_device_us": None,
            "scheduler_location": None,
            "kernel_execution_us": None,
            "log_path": str(log_path),
            "host_log_directory": str(self.log.directory),
            "session_id": self.session_id,
            "pid": os.getpid(),
            "worker_generation": self.generation,
            "phase": phase,
            "started_utc": started,
            "callable_sha256": adapter["sha256"],
        }
        from simpler_setup.scene_test import _dump_name_map, _extract_name_map  # noqa: PLC0415

        if deps or swimlane:
            _dump_name_map(_extract_name_map(adapter.get("callable_spec", {})), directory / "name_map_benchmark.json")
        before = None
        try:
            before = self.log.checkpoint()
            with measure_phase(timings, "resource_sample"):
                result["resources_before"] = self._resources()
            with log_path.open("w") as console, redirect_stdout(console), redirect_stderr(console):
                try:
                    self._execute(case_id, adapter, rounds, deps, swimlane, directory, timings)
                except AssertionError:
                    raise
                except Exception:
                    self.aborted = f"execution or cleanup failed at {phase}/{case_id}"
                    raise
            with measure_phase(timings, "resource_sample"):
                result["resources_after"] = self._resources()
            with measure_phase(timings, "log_parse"):
                lines, after = self.log.read_since(before)
                with log_path.open("a") as stream:
                    stream.writelines(lines)
                result["host_log_slice"] = {"begin": before, "end": after}
                if after["drops"] != before["drops"]:
                    raise ValueError("Host log records were dropped during this case")
                rows = _timings_from_lines(iter(lines), rounds)
                result["run_identities"] = [r["run_identity"] for r in rows]
            if deps or swimlane:
                with measure_phase(timings, "capture_validate"):
                    names = (["deps.json"] if deps else []) + (["chip_swimlane_records.json"] if swimlane else [])
                    result["diagnostic_paths"] = [
                        str(directory / name) for name in names if (directory / name).is_file()
                    ]
                    if len(result["diagnostic_paths"]) != len(names):
                        raise ValueError(f"missing capture file(s): {names}")
                    if deps and case_id in CASE_IDS:
                        from tests.st.scheduler_benchmark.scene import validate_deps  # noqa: PLC0415

                        spec = scale_case(expand_case(case_id, self.k_c, self.k_v), self.task_scale)
                        validate_deps(json.loads((directory / "deps.json").read_text()), spec)
                    result["diagnostic_device_us"] = rows[0]["device_us"]
            else:
                if exclude_first:
                    result["first_round"], rows = rows[0], rows[1:]
                result["rounds"] = rows
            result["status"] = "passed"
        except Exception as exc:
            result["status"], result["error"] = "failed", f"{type(exc).__name__}: {exc}"
            with log_path.open("a") as stream:
                stream.write(traceback.format_exc())
            if before is not None and "host_log_slice" not in result:
                try:
                    lines, after = self.log.read_since(before)
                    result["host_log_slice"] = {"begin": before, "end": after}
                    with log_path.open("a") as stream:
                        stream.writelines(lines)
                except ValueError:
                    pass
        result["ended_utc"] = utc_now()
        result["elapsed_ns"] = perf_counter_ns() - start
        result["phase_ns"] = timings
        result["phase_ns"]["other"] = result["elapsed_ns"] - sum(timings.values())
        return result

    def run_phase(self, manifest, output, *, exclude_first=False):
        output = Path(output)
        phase = "deps" if manifest["capture_deps"] else "events" if manifest["swimlane_level"] else "timing"
        start = perf_counter_ns()
        totals = {}
        manifest.update(
            schema_version=4,
            execution_model=EXECUTION_MODEL,
            session_id=self.session_id,
            pid=os.getpid(),
            round_policy="exclude_first" if exclude_first else "all",
            started_utc=utc_now(),
        )
        write_json(output, manifest)
        try:
            if self.aborted is None:
                with measure_phase(totals, "module_compile"):
                    for case_id in manifest["case_results"]:
                        try:
                            self._prepare(case_id)
                        except Exception as exc:
                            manifest["case_results"][case_id] = {"status": "failed", "error": f"prepare: {exc}"}
                if any(
                    self.adapters.get(cid) for cid, r in manifest["case_results"].items() if r["status"] == "not_run"
                ):
                    with measure_phase(totals, "worker_init"):
                        self._start_worker()
                for case_id, result in list(manifest["case_results"].items()):
                    if self.aborted is not None:
                        break
                    if result["status"] != "not_run":
                        continue
                    adapter = self.adapters.get(case_id)
                    if adapter is None:
                        result = {"status": "unsupported_platform"}
                    else:
                        with measure_phase(totals, "cases"):
                            result = self._run_case(
                                case_id,
                                adapter,
                                phase,
                                manifest["rounds_requested"],
                                manifest["capture_deps"],
                                manifest["swimlane_level"],
                                exclude_first,
                            )
                    manifest["case_results"][case_id] = result
                    print(f"{self.runtime}/{phase}/{case_id}: {result['status']}", flush=True)
                    with measure_phase(totals, "manifest_write"):
                        write_json(output, manifest)
        except Exception as exc:
            self.aborted = f"phase {phase}: {type(exc).__name__}: {exc}"
        finally:
            try:
                with measure_phase(totals, "worker_close"):
                    self._close_worker()
            except Exception as exc:
                self.aborted = f"Worker.close: {type(exc).__name__}: {exc}"
            manifest.update(
                ended_utc=utc_now(),
                elapsed_ns=perf_counter_ns() - start,
                phase_ns=totals,
                abort_reason=self.aborted,
                unsafe_device=self.unsafe_device,
                worker_generation=self.generation,
            )
            totals["other"] = manifest["elapsed_ns"] - sum(totals.values())
            manifest["exit_code"] = int(
                bool(self.aborted) or any(row["status"] == "failed" for row in manifest["case_results"].values())
            )
            write_json(output, manifest)
        return manifest

    def close(self):
        self._close_worker()
        for key, value in self._original_counts.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
