# A5 HBG AICore scheduling diagnostics

Host orchestration finishes before device launch. During device execution,
AICore workers own graph classification, dependency resolution, ready/completion
queues, and kernel execution. AICPU logs cover only initialization, run
supervision/waiting, profiling flush, and teardown; there is no AICPU scheduler
summary or scheduler phase breakdown for this runtime.

On hardware, AICPU lifecycle logs are written by CANN dlog under:

```text
$HOME/ascend/log/debug/device-<device_id>/device-<pid>_<timestamp>.log
```

For task timing, enable chip-swimlane level 1 and inspect
`<output_prefix>/chip_swimlane_records.json`. See
[profiling_levels.md](profiling_levels.md) for the schema and strict-failure
rules.

Dependency and execution counters are copied from the AICore sidecar during
host validation. They are the authoritative device scheduling diagnostics for
this runtime.
