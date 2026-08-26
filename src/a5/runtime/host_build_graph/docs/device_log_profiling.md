# A5 HBG AICore scheduling diagnostics

Host orchestration finishes before device launch. During device execution,
AICore workers own graph classification, dependency resolution, ready/completion
inboxes, Gang coordination, and kernel execution. AICPU logs cover only
initialization, run supervision/waiting, and teardown; there
is no AICPU scheduler summary or scheduler phase breakdown for this runtime.

On hardware, AICPU lifecycle logs are written by CANN dlog under:

```text
$HOME/ascend/log/debug/device-<device_id>/device-<pid>_<timestamp>.log
```

Dependency and execution counters are copied from the scheduler state during
host validation. They are the authoritative device scheduling diagnostics for
this runtime.
