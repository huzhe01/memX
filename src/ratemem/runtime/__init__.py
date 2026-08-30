"""Provider-neutral runtime selection for local and company accelerators."""

from ratemem.runtime.device import (
    DeviceRuntime,
    RuntimeProbe,
    observe_runtime_probe,
    resolve_runtime,
)
from ratemem.runtime.distributed import (
    DistributedContext,
    RankEnvironment,
    distributed_session,
)
from ratemem.runtime.preflight import build_preflight_receipt, validate_preflight

__all__ = [
    "DeviceRuntime",
    "DistributedContext",
    "RankEnvironment",
    "RuntimeProbe",
    "build_preflight_receipt",
    "distributed_session",
    "observe_runtime_probe",
    "resolve_runtime",
    "validate_preflight",
]
