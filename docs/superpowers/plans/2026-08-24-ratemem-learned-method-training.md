# RateMem-DiT Learned Method and Sequential Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the learned RateMem method that turns amortized adapter codes into reusable immutable multi-concept packet bundles, manages them causally under an exact serialized-byte cap, and meta-trains the codec and utility model on frozen lifecycle traces without exceeding two transformer passes per segment.

**Architecture:** The existing SANA support amortizer produces a target adapter code, then a blockwise base quantizer and a learned frozen group/RVQ direction dictionary encode it into a mandatory base plus content-addressed enhancement packets. The same dictionary key and payload may be selected by many concepts; compact quantized incidence gains remain concept-specific, and every allocation item contains one immutable packet plus its complete prespecified incidence list. A causal outer lifecycle controller reserves or evicts complete base records outside the theorem, while the existing certified allocator selects byte-exact packet bundles using a past-observable nonnegative utility calibrator.

**Tech Stack:** Python 3.11, `uv`, PyTorch 2.8.0, NumPy 2.2.6, Pydantic 2.11.7, safetensors 0.6.2, canonical CBOR from the core plan, JSON Schema Draft 2020-12 with `jsonschema==4.25.1`, pytest, Hypothesis, Ruff, mypy

---

## Novelty boundary and execution gate

This plan does not claim the support-to-adapter amortizer, RVQ, quantization, shared LoRA bases, or the standard partial-enumeration submodular allocator. Those are borrowed components. The implementation isolates the proposed contribution in a testable boundary: a learned dictionary that emits content-addressed packet directions, exact packet reuse across concept records with quantized per-concept gains, immutable all-incidence bundle proposals, and causal lifecycle decisions under the measured serialized byte count.

Execute the companion plans in this exact order:

1. complete `docs/superpowers/plans/2026-08-24-ratemem-core-memory.md`;
2. complete the free verification work in `docs/superpowers/plans/2026-08-24-ratemem-sana-modal-pilot.md`;
3. complete Tasks 1--7 of `docs/superpowers/plans/2026-08-24-ratemem-scientific-evaluation.md`, including sealed dataset, visible-trace, final-trace-envelope, and evaluator prerequisites;
4. execute `docs/superpowers/plans/2026-08-24-ratemem-matched-baselines.md`, using only the narrowly scoped pre-lock `baseline_fidelity` permit from scientific Task 8 when a real-checkpoint fidelity test requires an accelerator, and then seal the baseline lock in scientific Task 8;
5. implement this plan's Tasks 1--13 and run only their fixture/fake-provider tests; do not execute either deferred paid producer command;
6. execute Task 14 here and seal `artifacts/method/cpu-gate.json` with `provider_invocations: 0`;
7. return to scientific Task 9 for a separate authorization and reservation for one named RateMem training phase, run the deferred Task 12 producer once, reconcile its provider usage, and finalize its phase receipt before requesting another seed authorization;
8. obtain a new authorization and reservation for one named materialization phase, run the deferred Task 13 producer once to create the real shared-input bundle, reconcile its provider usage, and finalize its phase receipt;
9. only then continue the baseline-owned per-method search and remaining scientific evaluation tasks, followed by the paper plan.

Every command through the CPU gate is credential-free and starts no paid compute. Only the steps explicitly labelled **Deferred paid execution** may invoke the provider, and each such invocation is permitted only after scientific Task 9 has created a separate current authorization and reservation for that one named phase. The producer consumes that permit once, and the phase remains incomplete until provider usage is reconciled and a final phase receipt is sealed.

## File map

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Add the method extra and `ratemem-method` entry point without replacing the Python 3.11, SANA, or science configuration. |
| `configs/method/ratemem-v1.yaml` | Frozen architecture, hard-codec, causal-controller, calibration, loss, and segment limits. |
| `configs/method/ratemem-training-lock.yaml` | Hash binding from the method policy to the dataset, evaluation, baseline, and visible training-trace locks. |
| `schemas/ratemem-method-lock-v1.schema.json` | Generated strict schema for the method lock. |
| `src/ratemem/method/config.py` | Typed policy and lock validation. |
| `src/ratemem/method/base_quantizer.py` | Deterministic grouped symmetric integer base codec with exact binary format. |
| `src/ratemem/method/dictionary.py` | Trainable group/RVQ direction bank, hard and soft assignments, frozen digest, and canonical packet keys. |
| `src/ratemem/method/codec.py` | Deterministic hard codec, differentiable soft/STE codec, gain quantization, reconstruction, and soft--hard agreement. |
| `src/ratemem/method/proposal.py` | Forward-only candidate construction, exact resident-packet reuse, immutable all-incidence bundles, and exact bundle costs. |
| `src/ratemem/method/utility.py` | Past-observable request weights, nonnegative beta/gain calibration, calibration receipts, and coverage-oracle construction. |
| `src/ratemem/method/controller.py` | Outer admission/update/eviction/rejection policy, fixed-cohort packet allocation, atomic state replacement, and hard-budget enforcement. |
| `src/ratemem/method/adapter.py` | RateMem implementation of the locked scientific baseline adapter protocol. |
| `src/ratemem/training/segments.py` | Frozen visible-trace loader and bounded two-event segment construction. |
| `src/ratemem/training/functional_state.py` | Differentiable functional memory state and explicit segment-boundary detach. |
| `src/ratemem/training/losses.py` | Flow, reconstruction, rate, reuse, balance, commitment, and calibration objectives. |
| `src/ratemem/training/meta_trainer.py` | Sequential meta-training loop, transformer-pass accounting, and frozen-backbone guard. |
| `src/ratemem/method/checkpoint.py` | Trainable-only safetensors checkpoint plus strict provenance manifest and schema validation. |
| `src/ratemem/training/authorized.py` | One-shot, permit-guarded real learned-training producer with frozen-checkpoint validation. |
| `src/ratemem/training/modal_app.py` | Single-call Modal execution boundary for one bounded scientific training phase. |
| `src/ratemem/method/shared_provider.py` | Real `SharedInputProvider` backed by the frozen RateMem checkpoint and locked SANA artifacts. |
| `src/ratemem/method/materialize.py` | One-shot, permit-guarded production of a real baseline-owned shared-input bundle. |
| `src/ratemem/method/phase_receipts.py` | Attempt/final receipt binding for consumed permits, outputs, provider call IDs, and reconciliation. |
| `src/ratemem/method/cli.py` | Lock, CPU verification, checkpoint inspection, authorized training, materialization, and phase-finalization entry points. |
| `schemas/ratemem-training-request-v1.schema.json` | Strict schema for one bounded authorized learned-training request. |
| `schemas/ratemem-materialize-request-v1.schema.json` | Strict schema for one authorized real shared-input materialization request. |
| `schemas/ratemem-method-phase-attempt-v1.schema.json` | Strict immutable pre-reconciliation attempt receipt for a paid method phase. |
| `schemas/ratemem-method-phase-final-v1.schema.json` | Strict post-reconciliation final receipt accepted by downstream science. |
| `artifacts/method/cpu-gate.json` | Machine-readable proof that the method lock and clean commit passed the complete non-paid CPU gate with zero provider invocations. |
| `tests/unit/method/` | Fast deterministic codec, proposal, calibrator, controller, and checkpoint tests. |
| `tests/contract/method/` | Causality, immutability, exact-byte, novelty-boundary, and soft--hard contracts. |
| `tests/integration/method/` | Synthetic nonseparability and end-to-end lifecycle/meta-training contracts. |

### Task 1: Freeze the method policy after the scientific locks

**Depends on:** core plan; SANA pilot Tasks 1--8; scientific evaluation Tasks 1--8.

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `configs/method/ratemem-v1.yaml`
- Create: `src/ratemem/method/__init__.py`
- Create: `src/ratemem/method/config.py`
- Create: `src/ratemem/method/cli.py`
- Create: `schemas/ratemem-method-lock-v1.schema.json`
- Test: `tests/unit/method/test_config.py`
- Test: `tests/contract/method/test_method_lock_schema.py`

- [ ] **Step 1: Write the dependency and lock-failure tests**

```python
# tests/unit/method/test_config.py
from importlib.metadata import version
from pathlib import Path

import pytest

from ratemem.method.config import LockMismatch, MethodPolicy, freeze_method_lock


def test_method_dependencies_keep_the_locked_shared_versions() -> None:
    assert version("torch") == "2.8.0"
    assert version("numpy") == "2.2.6"
    assert version("pydantic") == "2.11.7"
    assert version("safetensors") == "0.6.2"
    assert version("jsonschema") == "4.25.1"


def test_policy_dimensions_match_the_sana_adapter_layout() -> None:
    policy = MethodPolicy.from_yaml(Path("configs/method/ratemem-v1.yaml"))
    assert policy.code.projection_count == 120
    assert policy.code.atom_count == 4
    assert policy.code.dimension == 480
    assert policy.code.dimension % policy.codec.group_size == 0
    assert policy.training.segment_length == 2
    assert policy.training.maximum_transformer_passes_per_segment == 2


def test_method_lock_rejects_a_changed_scientific_input(valid_lock_inputs) -> None:
    inputs = valid_lock_inputs.model_copy(update={"evaluation_lock_sha256": "0" * 64})
    with pytest.raises(LockMismatch, match="evaluation lock content hash"):
        freeze_method_lock(inputs)
```

- [ ] **Step 2: Run the focused tests and verify the missing method package**

Run: `uv run pytest tests/unit/method/test_config.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'ratemem.method'`.

- [ ] **Step 3: Extend, rather than replace, the shared dependency lock**

Add this extra and script alongside the existing core, pilot, and science entries:

```toml
[project.optional-dependencies]
method = [
  "jsonschema==4.25.1",
  "safetensors==0.6.2",
  "torch==2.8.0",
]

[project.scripts]
ratemem-method = "ratemem.method.cli:main"
```

Run: `uv lock && uv sync --all-extras --all-groups --frozen`

Expected: exit 0; `uv tree --depth 1` contains one resolved version for each of Torch, NumPy, Pydantic, safetensors, and jsonschema, with jsonschema exactly `4.25.1`.

- [ ] **Step 4: Add the fully specified version-one policy**

```yaml
# configs/method/ratemem-v1.yaml
schema_version: "1.0"
method_id: ratemem_v1
novelty_claim: learned_multi_concept_immutable_packet_bundles_with_causal_exact_byte_lifecycle
borrowed_components:
  - hyperlora_style_support_amortizer
  - grouped_residual_vector_quantization
  - symmetric_integer_quantization
  - partial_enumeration_submodular_knapsack_allocator
code:
  projection_count: 120
  atom_count: 4
  dimension: 480
codec:
  group_size: 16
  base_bits: 4
  rvq_stages: 2
  entries_per_stage: 64
  incidence_gain_step: 0.00390625
  maximum_packets_per_concept: 8
  sharing_rule: exact_payload_only
  packet_format_version: RTPKT001
soft_codec:
  initial_temperature: 1.0
  final_temperature: 0.1
  anneal_steps: 10000
  maximum_mean_code_error: 0.02
  maximum_assignment_disagreement: 0.05
  maximum_topk_disagreement: 0.0
  ste_forward_atol: 0.000001
utility:
  hidden_dimension: 64
  request_decay: 0.97
  calibration_bins: 10
  maximum_expected_calibration_error: 0.05
controller:
  outer_policy: request_density_size_aware
  allow_rejection: true
  whole_concept_eviction: true
  switching_penalty: 0.0
  certified_prescreen_max_bundles: 24
  theorem_scope: fixed_admitted_cohort_prescreened_packets_only
training:
  segment_length: 2
  maximum_query_events_per_segment: 2
  maximum_transformer_passes_per_segment: 2
  truncated_bptt_length: 2
  detach_at_segment_boundary: true
  precision: bfloat16
  activation_checkpointing: true
  training_seeds: [17, 29, 43]
  loss_weights:
    flow: 1.0
    reconstruction: 0.25
    rate: 0.01
    reuse_affinity: 0.05
    dictionary_balance: 0.01
    dictionary_commitment: 0.10
    utility_calibration: 0.10
```

- [ ] **Step 5: Implement strict policy and scientific-lock binding**

```python
# src/ratemem/method/config.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from ratemem.evaluation.canonical import file_sha256, semantic_sha256

Sha256 = str


class LockMismatch(ValueError):
    """Raised when an approved scientific input hash has changed."""


class CodePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    projection_count: PositiveInt
    atom_count: PositiveInt
    dimension: PositiveInt

    @model_validator(mode="after")
    def check_dimension(self) -> "CodePolicy":
        if self.dimension != self.projection_count * self.atom_count:
            raise ValueError("code dimension must equal projection_count times atom_count")
        return self


class CodecPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    group_size: PositiveInt
    base_bits: Literal[2, 4, 8]
    rvq_stages: PositiveInt
    entries_per_stage: PositiveInt
    incidence_gain_step: PositiveFloat
    maximum_packets_per_concept: PositiveInt
    sharing_rule: Literal["exact_payload_only"]
    packet_format_version: Literal["RTPKT001"]


class SoftCodecPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    initial_temperature: PositiveFloat
    final_temperature: PositiveFloat
    anneal_steps: PositiveInt
    maximum_mean_code_error: PositiveFloat
    maximum_assignment_disagreement: float = Field(ge=0.0, le=1.0)
    maximum_topk_disagreement: Literal[0.0]
    ste_forward_atol: Literal[0.000001]


class UtilityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hidden_dimension: PositiveInt
    request_decay: float = Field(gt=0.0, le=1.0)
    calibration_bins: PositiveInt
    maximum_expected_calibration_error: float = Field(ge=0.0, le=1.0)


class ControllerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    outer_policy: Literal["request_density_size_aware"]
    allow_rejection: Literal[True]
    whole_concept_eviction: Literal[True]
    switching_penalty: Literal[0.0]
    certified_prescreen_max_bundles: Literal[24]
    theorem_scope: Literal["fixed_admitted_cohort_prescreened_packets_only"]


class TrainingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    segment_length: Literal[2]
    maximum_query_events_per_segment: Literal[2]
    maximum_transformer_passes_per_segment: Literal[2]
    truncated_bptt_length: Literal[2]
    detach_at_segment_boundary: Literal[True]
    precision: Literal["bfloat16"]
    activation_checkpointing: Literal[True]
    training_seeds: tuple[int, int, int]
    loss_weights: dict[str, float]


class MethodPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"]
    method_id: Literal["ratemem_v1"]
    novelty_claim: Literal[
        "learned_multi_concept_immutable_packet_bundles_with_causal_exact_byte_lifecycle"
    ]
    borrowed_components: Sequence[str]
    code: CodePolicy
    codec: CodecPolicy
    soft_codec: SoftCodecPolicy
    utility: UtilityPolicy
    controller: ControllerPolicy
    training: TrainingPolicy

    @classmethod
    def from_yaml(cls, path: Path) -> "MethodPolicy":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def check_groups(self) -> "MethodPolicy":
        if self.code.dimension % self.codec.group_size:
            raise ValueError("code dimension must be divisible by codec group size")
        required = {
            "flow", "reconstruction", "rate", "reuse_affinity",
            "dictionary_balance", "dictionary_commitment", "utility_calibration",
        }
        if set(self.training.loss_weights) != required:
            raise ValueError("training loss weight keys do not match the locked objective")
        return self


class MethodLockInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_path: Path
    dataset_lock_path: Path
    evaluation_lock_path: Path
    baseline_lock_path: Path
    visible_trace_manifest_paths: Sequence[Path]
    expected_dataset_lock_sha256: Sha256
    expected_evaluation_lock_sha256: Sha256
    expected_baseline_lock_sha256: Sha256


class MethodTrainingLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    lock_id: Sha256
    method_id: Literal["ratemem_v1"]
    policy_sha256: Sha256
    dataset_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    visible_trace_manifest_sha256: Sequence[Sha256]


def freeze_method_lock(inputs: MethodLockInputs) -> MethodTrainingLock:
    actual = {
        "dataset": file_sha256(inputs.dataset_lock_path),
        "evaluation": file_sha256(inputs.evaluation_lock_path),
        "baseline": file_sha256(inputs.baseline_lock_path),
    }
    expected = {
        "dataset": inputs.expected_dataset_lock_sha256,
        "evaluation": inputs.expected_evaluation_lock_sha256,
        "baseline": inputs.expected_baseline_lock_sha256,
    }
    for name in actual:
        if actual[name] != expected[name]:
            raise LockMismatch(f"{name} lock content hash does not match approval")
    policy = MethodPolicy.from_yaml(inputs.policy_path)
    payload = {
        "schema_version": "1.0",
        "method_id": policy.method_id,
        "policy_sha256": file_sha256(inputs.policy_path),
        "dataset_lock_sha256": actual["dataset"],
        "evaluation_lock_sha256": actual["evaluation"],
        "baseline_lock_sha256": actual["baseline"],
        "visible_trace_manifest_sha256": tuple(
            file_sha256(path) for path in sorted(inputs.visible_trace_manifest_paths)
        ),
    }
    return MethodTrainingLock(lock_id=semantic_sha256(payload), **payload)
```

- [ ] **Step 6: Add the lock CLI and generated-schema contract**

```python
# src/ratemem/method/cli.py
from pathlib import Path

import typer

from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.method.config import MethodTrainingLock

app = typer.Typer(no_args_is_help=True, help="RateMem learned-method controls")


@app.command("schema")
def schema(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(MethodTrainingLock.model_json_schema()))
    typer.echo(f"PASS method-lock schema: {output}")


def main() -> None:
    app()
```

```python
# tests/contract/method/test_method_lock_schema.py
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.method.config import MethodTrainingLock


def test_committed_method_lock_schema_is_current() -> None:
    path = Path("schemas/ratemem-method-lock-v1.schema.json")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert path.read_bytes() == canonical_json_bytes(MethodTrainingLock.model_json_schema())
```

Run:

```bash
uv run ratemem-method schema --output schemas/ratemem-method-lock-v1.schema.json
uv run pytest tests/unit/method/test_config.py tests/contract/method/test_method_lock_schema.py -q
uv run ruff check src/ratemem/method tests/unit/method tests/contract/method
uv run mypy src/ratemem/method
```

Expected: all focused tests pass; schema validation, Ruff, and mypy exit 0.

- [ ] **Step 7: Seal the method lock from the already approved inputs and commit**

Add a `lock` command to `src/ratemem/method/cli.py` that accepts explicit paths and approved hashes, calls `freeze_method_lock`, and writes YAML atomically through `write_yaml_atomic`. Run it against `configs/scientific/dataset-lock.yaml`, `configs/scientific/evaluation-lock.yaml`, `configs/scientific/baseline-lock.yaml`, and only `configs/scientific/traces/train-*` plus `validation-*` manifests. The command rejects final-test payload paths.

```bash
uv run ratemem-method lock \
  --policy configs/method/ratemem-v1.yaml \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --visible-trace-dir configs/scientific/traces \
  --approval artifacts/scientific/freeze/method-lock-approval.json \
  --output configs/method/ratemem-training-lock.yaml
```

Expected: stdout matches `^PASS method-lock: [0-9a-f]{64}$`; any final-test event payload, changed scientific hash, missing approval, or mutable revision exits 2 without writing the lock.

```bash
git add pyproject.toml uv.lock configs/method src/ratemem/method/__init__.py src/ratemem/method/config.py src/ratemem/method/cli.py schemas/ratemem-method-lock-v1.schema.json tests/unit/method/test_config.py tests/contract/method/test_method_lock_schema.py
git commit -m "feat(method): freeze RateMem training contract"
```

### Task 2: Implement the deterministic blockwise base quantizer

**Files:**
- Create: `src/ratemem/method/base_quantizer.py`
- Test: `tests/unit/method/test_base_quantizer.py`
- Test: `tests/contract/method/test_base_payload_format.py`

- [ ] **Step 1: Write exact round-trip, bit-packing, and determinism tests**

```python
# tests/unit/method/test_base_quantizer.py
import numpy as np
import pytest

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer, decode_base_payload


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_blockwise_payload_is_deterministic_and_finite(bits: int) -> None:
    code = np.linspace(-1.25, 1.5, 48, dtype=np.float32)
    codec = BlockwiseBaseQuantizer(group_size=16, bits=bits)
    first = codec.encode(code)
    second = codec.encode(code.copy())
    assert first.payload == second.payload
    decoded = decode_base_payload(first.payload)
    assert decoded.shape == (48,)
    assert decoded.dtype == np.float32
    assert np.isfinite(decoded).all()


def test_more_base_bits_do_not_increase_error() -> None:
    code = np.array([0.91, -0.52, 0.33, -1.0] * 12, dtype=np.float32)
    errors = [
        np.mean((BlockwiseBaseQuantizer(16, bits).encode(code).decode() - code) ** 2)
        for bits in (2, 4, 8)
    ]
    assert errors[2] <= errors[1] <= errors[0]


def test_nonfinite_or_wrong_width_input_is_rejected() -> None:
    codec = BlockwiseBaseQuantizer(group_size=16, bits=4)
    with pytest.raises(ValueError, match="divisible"):
        codec.encode(np.zeros(17, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        codec.encode(np.array([float("nan")] * 16, dtype=np.float32))
```

- [ ] **Step 2: Run the base-code tests and verify the module is absent**

Run: `uv run pytest tests/unit/method/test_base_quantizer.py -q`

Expected: collection fails because `ratemem.method.base_quantizer` does not exist.

- [ ] **Step 3: Implement the fixed binary format and symmetric grouped quantizer**

```python
# src/ratemem/method/base_quantizer.py
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
UIntArray: TypeAlias = NDArray[np.uint8]
_MAGIC = b"RTBASE01"
_HEADER = struct.Struct("<8sBHI")


def _pack(values: UIntArray, bits: int) -> bytes:
    output = bytearray(math.ceil(len(values) * bits / 8))
    accumulator = 0
    occupied = 0
    cursor = 0
    for value in values.tolist():
        accumulator |= int(value) << occupied
        occupied += bits
        while occupied >= 8:
            output[cursor] = accumulator & 0xFF
            cursor += 1
            accumulator >>= 8
            occupied -= 8
    if occupied:
        output[cursor] = accumulator & 0xFF
    return bytes(output)


def _unpack(payload: bytes, count: int, bits: int) -> UIntArray:
    mask = (1 << bits) - 1
    output = np.empty(count, dtype=np.uint8)
    accumulator = 0
    occupied = 0
    cursor = 0
    for index in range(count):
        while occupied < bits:
            if cursor >= len(payload):
                raise ValueError("truncated packed base payload")
            accumulator |= payload[cursor] << occupied
            cursor += 1
            occupied += 8
        output[index] = accumulator & mask
        accumulator >>= bits
        occupied -= bits
    if cursor != len(payload) or accumulator:
        raise ValueError("noncanonical trailing base bits")
    return output


@dataclass(frozen=True, slots=True)
class QuantizedBase:
    payload: bytes

    def decode(self) -> FloatArray:
        return decode_base_payload(self.payload)

    def scales(self) -> FloatArray:
        return decode_base_scales(self.payload)


class BlockwiseBaseQuantizer:
    def __init__(self, group_size: int, bits: int) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if bits not in {2, 4, 8}:
            raise ValueError("bits must be 2, 4, or 8")
        self.group_size = group_size
        self.bits = bits

    def encode(self, code: FloatArray) -> QuantizedBase:
        flat = cast(FloatArray, np.asarray(code, dtype=np.float32).reshape(-1))
        if flat.size == 0 or flat.size % self.group_size:
            raise ValueError("code width must be nonempty and divisible by group_size")
        if not np.isfinite(flat).all():
            raise ValueError("code must be finite")
        groups = flat.reshape(-1, self.group_size)
        qmax = (1 << (self.bits - 1)) - 1
        scales = np.maximum(np.max(np.abs(groups), axis=1) / qmax, np.finfo(np.float16).tiny)
        scales_f16 = scales.astype("<f2")
        restored_scales = scales_f16.astype(np.float32)
        signed = np.rint(groups / restored_scales[:, None]).clip(-qmax, qmax).astype(np.int16)
        unsigned = cast(UIntArray, (signed + qmax).astype(np.uint8).reshape(-1))
        header = _HEADER.pack(_MAGIC, self.bits, self.group_size, flat.size)
        return QuantizedBase(header + scales_f16.tobytes(order="C") + _pack(unsigned, self.bits))


def decode_base_payload(payload: bytes) -> FloatArray:
    if len(payload) < _HEADER.size:
        raise ValueError("truncated base header")
    magic, bits, group_size, count = _HEADER.unpack_from(payload)
    if magic != _MAGIC or bits not in {2, 4, 8} or not group_size or count % group_size:
        raise ValueError("unsupported base payload")
    group_count = count // group_size
    scales_end = _HEADER.size + group_count * 2
    if scales_end > len(payload):
        raise ValueError("truncated base scales")
    scales = np.frombuffer(payload[_HEADER.size:scales_end], dtype="<f2").astype(np.float32)
    unsigned = _unpack(payload[scales_end:], count, bits).astype(np.int16)
    qmax = (1 << (bits - 1)) - 1
    if np.any(unsigned > 2 * qmax):
        raise ValueError("base payload contains an unused integer code")
    signed = unsigned - qmax
    decoded = signed.reshape(group_count, group_size).astype(np.float32) * scales[:, None]
    return cast(FloatArray, decoded.reshape(-1))


def decode_base_scales(payload: bytes) -> FloatArray:
    # Full decoding first enforces canonical length, padding, and integer range.
    decode_base_payload(payload)
    _, _, group_size, count = _HEADER.unpack_from(payload)
    group_count = count // group_size
    scales_end = _HEADER.size + group_count * 2
    return cast(
        FloatArray,
        np.frombuffer(payload[_HEADER.size:scales_end], dtype="<f2")
        .astype(np.float32)
        .copy(),
    )
```

- [ ] **Step 4: Lock the byte formula and malformed-input behavior**

```python
# tests/contract/method/test_base_payload_format.py
import math
import struct

import numpy as np
import pytest

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer, decode_base_payload


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_payload_length_matches_header_scales_and_packed_integers(bits: int) -> None:
    count, group_size = 480, 16
    payload = BlockwiseBaseQuantizer(group_size, bits).encode(
        np.linspace(-1.0, 1.0, count, dtype=np.float32)
    ).payload
    assert len(payload) == struct.calcsize("<8sBHI") + (count // group_size) * 2 + math.ceil(count * bits / 8)


def test_payload_rejects_truncation_and_noncanonical_suffix() -> None:
    payload = BlockwiseBaseQuantizer(16, 4).encode(np.ones(32, dtype=np.float32)).payload
    with pytest.raises(ValueError):
        decode_base_payload(payload[:-1])
    with pytest.raises(ValueError):
        decode_base_payload(payload + b"\x00")


def test_public_scale_decode_matches_the_thirty_production_groups() -> None:
    encoded = BlockwiseBaseQuantizer(16, 4).encode(
        np.linspace(-1.0, 1.0, 480, dtype=np.float32)
    )
    assert encoded.scales().shape == (30,)
    assert encoded.scales().dtype == np.float32
```

- [ ] **Step 5: Run the focused suite, static checks, and commit**

Run:

```bash
uv run pytest tests/unit/method/test_base_quantizer.py tests/contract/method/test_base_payload_format.py -q
uv run ruff check src/ratemem/method/base_quantizer.py tests/unit/method/test_base_quantizer.py tests/contract/method/test_base_payload_format.py
uv run mypy src/ratemem/method/base_quantizer.py
```

Expected: all focused tests pass; Ruff and mypy exit 0.

```bash
git add src/ratemem/method/base_quantizer.py tests/unit/method/test_base_quantizer.py tests/contract/method/test_base_payload_format.py
git commit -m "feat(method): add blockwise base codec"
```

### Task 3: Learn a reusable group/RVQ packet dictionary and freeze its identity

**Files:**
- Create: `src/ratemem/method/dictionary.py`
- Test: `tests/unit/method/test_dictionary.py`
- Test: `tests/contract/method/test_packet_identity.py`

- [ ] **Step 1: Write hard-assignment, gradient, and exact-reuse tests**

```python
# tests/unit/method/test_dictionary.py
import torch

from ratemem.method.dictionary import GroupRVQDictionary


def dictionary() -> GroupRVQDictionary:
    model = GroupRVQDictionary(group_count=2, group_size=4, stages=2, entries=3)
    with torch.no_grad():
        model.codebooks.copy_(torch.tensor([
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
             [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]],
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
             [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]],
        ]))
        model.normalize_codebooks_()
    return model


def test_hard_assignment_is_deterministic_and_residual_is_additive() -> None:
    model = dictionary()
    residual = torch.tensor([[[2.0, 0.1, 0.0, 0.0], [0.0, 1.5, 1.4, 0.0]]])
    first = model.hard_assign(residual)
    second = model.hard_assign(residual.clone())
    torch.testing.assert_close(first.reconstruction + first.residual, residual)
    assert torch.equal(first.indices, second.indices)
    torch.testing.assert_close(first.gains, second.gains)


def test_soft_assignment_reaches_dictionary_gradients() -> None:
    model = dictionary()
    residual = torch.randn(3, 2, 4, requires_grad=True)
    result = model.soft_assign(residual, temperature=0.5, straight_through=False)
    result.reconstruction.square().mean().backward()
    assert residual.grad is not None
    assert model.codebooks.grad is not None
```

```python
# tests/contract/method/test_packet_identity.py
import torch

from ratemem.method.dictionary import decode_packet_key, freeze_dictionary
from tests.unit.method.test_dictionary import dictionary


def test_same_dictionary_entry_has_one_exact_packet_payload_for_many_concepts() -> None:
    frozen = freeze_dictionary(dictionary())
    first = frozen.packet(group=0, stage=0, entry=1)
    second = frozen.packet(group=0, stage=0, entry=1)
    other = frozen.packet(group=0, stage=0, entry=2)
    assert first.packet_id == second.packet_id
    assert first.payload == second.payload
    assert first.packet_id != other.packet_id
    assert decode_packet_key(first.payload) == (frozen.revision_sha256, 0, 0, 1)
```

- [ ] **Step 2: Run the dictionary tests and verify the import failure**

Run: `uv run pytest tests/unit/method/test_dictionary.py tests/contract/method/test_packet_identity.py -q`

Expected: collection fails because `ratemem.method.dictionary` does not exist.

- [ ] **Step 3: Implement normalized direction assignment with hard, soft, and STE paths**

```python
# src/ratemem/method/dictionary.py
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ratemem.state.model import Packet
from ratemem.state.serialization import packet_from_payload

_PACKET_MAGIC = b"RTPKT001"
_PACKET_HEADER = struct.Struct("<8s32sHHH")


@dataclass(frozen=True)
class RVQAssignment:
    indices: Tensor
    gains: Tensor
    probabilities: Tensor
    reconstruction: Tensor
    residual: Tensor


class GroupRVQDictionary(nn.Module):
    def __init__(self, group_count: int, group_size: int, stages: int, entries: int) -> None:
        super().__init__()
        if min(group_count, group_size, stages, entries) < 1:
            raise ValueError("dictionary dimensions must be positive")
        self.group_count = group_count
        self.group_size = group_size
        self.stages = stages
        self.entries = entries
        self.codebooks = nn.Parameter(torch.randn(group_count, stages, entries, group_size))
        self.normalize_codebooks_()

    @torch.no_grad()
    def normalize_codebooks_(self) -> None:
        self.codebooks.copy_(F.normalize(self.codebooks.float(), dim=-1, eps=1e-8))

    def _validate(self, residual: Tensor) -> None:
        if residual.ndim != 3 or residual.shape[1:] != (self.group_count, self.group_size):
            raise ValueError("residual must have shape [batch, group_count, group_size]")
        if not torch.isfinite(residual).all():
            raise ValueError("residual must be finite")

    def _assign(self, residual: Tensor, temperature: float, straight_through: bool) -> RVQAssignment:
        self._validate(residual)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        remaining = residual.float()
        reconstruction = torch.zeros_like(remaining)
        index_rows: list[Tensor] = []
        gain_rows: list[Tensor] = []
        probability_rows: list[Tensor] = []
        for stage in range(self.stages):
            directions = F.normalize(self.codebooks[:, stage].float(), dim=-1, eps=1e-8)
            correlations = torch.einsum("bgd,ged->bge", remaining, directions)
            squared_error = remaining.square().sum(-1, keepdim=True) - correlations.square()
            soft = torch.softmax(-squared_error / temperature, dim=-1)
            indices = squared_error.argmin(dim=-1)
            hard = F.one_hot(indices, self.entries).to(soft.dtype)
            probabilities = hard + soft - soft.detach() if straight_through else soft
            selected = torch.einsum("bge,ged->bgd", probabilities, directions)
            gains = (remaining * selected).sum(dim=-1)
            contribution = gains.unsqueeze(-1) * selected
            remaining = remaining - contribution
            reconstruction = reconstruction + contribution
            index_rows.append(indices)
            gain_rows.append(gains)
            probability_rows.append(probabilities)
        return RVQAssignment(
            indices=torch.stack(index_rows, dim=2),
            gains=torch.stack(gain_rows, dim=2),
            probabilities=torch.stack(probability_rows, dim=2),
            reconstruction=reconstruction,
            residual=remaining,
        )

    def hard_assign(self, residual: Tensor) -> RVQAssignment:
        return self._assign(residual, temperature=1.0, straight_through=True)

    def soft_assign(self, residual: Tensor, temperature: float, straight_through: bool) -> RVQAssignment:
        return self._assign(residual, temperature=temperature, straight_through=straight_through)


@dataclass(frozen=True)
class FrozenGroupRVQDictionary:
    codebooks: Tensor
    revision_sha256: str

    def packet(self, group: int, stage: int, entry: int) -> Packet:
        group_count, stages, entries, _ = self.codebooks.shape
        if not 0 <= group < group_count or not 0 <= stage < stages or not 0 <= entry < entries:
            raise IndexError("dictionary packet index is out of range")
        payload = _PACKET_HEADER.pack(
            _PACKET_MAGIC, bytes.fromhex(self.revision_sha256), group, stage, entry
        )
        return packet_from_payload(payload)

    def direction(self, group: int, stage: int, entry: int) -> Tensor:
        return self.codebooks[group, stage, entry]


def freeze_dictionary(dictionary: GroupRVQDictionary) -> FrozenGroupRVQDictionary:
    codebooks = F.normalize(dictionary.codebooks.detach().float().cpu(), dim=-1, eps=1e-8).contiguous()
    shape = struct.pack("<IIII", *codebooks.shape)
    digest = hashlib.sha256(shape + codebooks.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
    return FrozenGroupRVQDictionary(codebooks=codebooks, revision_sha256=digest)


def decode_packet_key(payload: bytes) -> tuple[str, int, int, int]:
    if len(payload) != _PACKET_HEADER.size:
        raise ValueError("packet payload has the wrong byte length")
    magic, revision, group, stage, entry = _PACKET_HEADER.unpack(payload)
    if magic != _PACKET_MAGIC:
        raise ValueError("unsupported packet payload version")
    return revision.hex(), group, stage, entry
```

- [ ] **Step 4: Add explicit anti-collapse and immutability contracts**

Append tests that verify every normalized codeword has unit norm after `normalize_codebooks_`, `FrozenGroupRVQDictionary.codebooks.requires_grad` is false, changing one trainable codeword changes the frozen revision, and a packet built under one revision cannot be decoded with a different frozen dictionary. Add `FrozenGroupRVQDictionary.validate_packet(packet)` to compare both the packet hash and embedded revision before returning the direction.

```python
def validate_packet(self, packet: Packet) -> tuple[int, int, int]:
    if packet_from_payload(packet.payload).packet_id != packet.packet_id:
        raise ValueError("packet content address does not match payload")
    revision, group, stage, entry = decode_packet_key(packet.payload)
    if revision != self.revision_sha256:
        raise ValueError("packet belongs to another frozen dictionary")
    self.direction(group, stage, entry)
    return group, stage, entry
```

- [ ] **Step 5: Run the dictionary suite and commit**

Run:

```bash
uv run pytest tests/unit/method/test_dictionary.py tests/contract/method/test_packet_identity.py -q
uv run ruff check src/ratemem/method/dictionary.py tests/unit/method/test_dictionary.py tests/contract/method/test_packet_identity.py
uv run mypy src/ratemem/method/dictionary.py
```

Expected: all focused tests pass; the same `(revision, group, stage, entry)` always produces the same payload and packet hash.

```bash
git add src/ratemem/method/dictionary.py tests/unit/method/test_dictionary.py tests/contract/method/test_packet_identity.py
git commit -m "feat(method): add reusable RVQ packet dictionary"
```

### Task 4: Make the deterministic hard codec and differentiable STE top-k identical

**Files:**
- Create: `src/ratemem/method/codec.py`
- Test: `tests/unit/method/test_codec.py`
- Test: `tests/contract/method/test_soft_hard_agreement.py`
- Test: `tests/contract/method/test_production_topk.py`

- [ ] **Step 1: Write failing gain, canonical tie, and actual-decode tests**

```python
# tests/unit/method/test_codec.py
import numpy as np

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer
from ratemem.method.codec import (
    PacketCandidateKey,
    RateMemHardCodec,
    select_packet_topk,
)
from ratemem.method.dictionary import freeze_dictionary
from tests.unit.method.test_dictionary import dictionary


def test_topk_ties_use_group_stage_entry_order_after_quantized_gain() -> None:
    rows = (
        PacketCandidateKey(group=1, stage=0, entry=2, gain_q=8),
        PacketCandidateKey(group=0, stage=1, entry=1, gain_q=-8),
        PacketCandidateKey(group=0, stage=0, entry=2, gain_q=8),
        PacketCandidateKey(group=0, stage=0, entry=1, gain_q=7),
    )
    selected = select_packet_topk(rows, maximum_packets=3)
    assert [(row.group, row.stage, row.entry, row.gain_q) for row in selected] == [
        (0, 0, 2, 8),
        (0, 1, 1, -8),
        (1, 0, 2, 8),
    ]


def test_hard_codec_decodes_only_the_ranked_quantized_topk() -> None:
    frozen = freeze_dictionary(dictionary())
    codec = RateMemHardCodec(
        BlockwiseBaseQuantizer(4, 4),
        frozen,
        gain_step=1 / 256,
        maximum_packets=2,
    )
    code = np.array(
        [1.0, 0.1, 0.0, 0.0, 0.0, 1.2, 0.8, 0.0],
        dtype=np.float32,
    )
    encoded = codec.encode("a", code)
    assert len(encoded.incidences) == 2
    decoded = codec.decode(encoded.base_payload, encoded.incidences)
    assert decoded.shape == code.shape
    assert all(-32768 <= row.incidence.gain_q <= 32767 for row in encoded.incidences)
    assert tuple(row.key for row in encoded.incidences) == select_packet_topk(
        tuple(row.key for row in encoded.all_candidates),
        maximum_packets=2,
    )
```

- [ ] **Step 2: Run the hard-codec tests and verify the missing module**

Run: `uv run pytest tests/unit/method/test_codec.py -q`

Expected: collection fails because `ratemem.method.codec` does not exist.

- [ ] **Step 3: Implement one quantized ranking shared by deployment and training**

```python
# src/ratemem/method/codec.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from ratemem.method.base_quantizer import (
    BlockwiseBaseQuantizer,
    decode_base_payload,
)
from ratemem.method.dictionary import (
    FrozenGroupRVQDictionary,
    GroupRVQDictionary,
    freeze_dictionary,
)
from ratemem.state.model import Incidence, Packet


def quantize_gain(value: float, step: float) -> int:
    if not np.isfinite(value) or step <= 0.0:
        raise ValueError("gain and step must be finite with positive step")
    return int(np.clip(np.rint(value / step), -32768, 32767))


def dequantize_gain(value: int, step: float) -> float:
    if not -32768 <= value <= 32767 or step <= 0.0:
        raise ValueError("gain_q must fit int16 and step must be positive")
    return value * step


@dataclass(frozen=True, slots=True, order=True)
class PacketCandidateKey:
    group: int
    stage: int
    entry: int
    gain_q: int


def select_packet_topk(
    rows: Sequence[PacketCandidateKey],
    maximum_packets: int,
) -> tuple[PacketCandidateKey, ...]:
    if maximum_packets < 1:
        raise ValueError("maximum_packets must be positive")
    if len({(row.group, row.stage) for row in rows}) != len(rows):
        raise ValueError("candidate stream repeats a group-stage position")
    ranked = sorted(
        rows,
        key=lambda row: (
            -(row.gain_q * row.gain_q),
            row.group,
            row.stage,
            row.entry,
        ),
    )
    return tuple(ranked[:maximum_packets])


@dataclass(frozen=True, slots=True)
class HardIncidence:
    incidence: Incidence
    packet: Packet
    group: int
    stage: int
    entry: int
    residual_reduction: float

    @property
    def key(self) -> PacketCandidateKey:
        return PacketCandidateKey(
            self.group,
            self.stage,
            self.entry,
            self.incidence.gain_q,
        )


@dataclass(frozen=True, slots=True)
class HardConceptEncoding:
    handle: str
    base_payload: bytes
    all_candidates: Sequence[HardIncidence]
    incidences: Sequence[HardIncidence]


class RateMemHardCodec:
    def __init__(
        self,
        base_quantizer: BlockwiseBaseQuantizer,
        dictionary: FrozenGroupRVQDictionary,
        gain_step: float,
        maximum_packets: int,
    ) -> None:
        self.base_quantizer = base_quantizer
        self.dictionary = dictionary
        self.gain_step = gain_step
        self.maximum_packets = maximum_packets

    def encode(
        self,
        handle: str,
        code: NDArray[np.float32],
    ) -> HardConceptEncoding:
        base = self.base_quantizer.encode(code)
        base_vector = base.decode()
        residual = torch.from_numpy(
            (
                np.asarray(code, dtype=np.float32) - base_vector
            ).reshape(
                1,
                self.dictionary.codebooks.shape[0],
                self.dictionary.codebooks.shape[-1],
            )
        )
        trainable = GroupRVQDictionary(
            self.dictionary.codebooks.shape[0],
            self.dictionary.codebooks.shape[-1],
            self.dictionary.codebooks.shape[1],
            self.dictionary.codebooks.shape[2],
        )
        with torch.no_grad():
            trainable.codebooks.copy_(self.dictionary.codebooks)
            assigned = trainable.hard_assign(residual)
        rows = []
        for group in range(trainable.group_count):
            for stage in range(trainable.stages):
                entry = int(assigned.indices[0, group, stage])
                gain_q = quantize_gain(
                    float(assigned.gains[0, group, stage]),
                    self.gain_step,
                )
                packet = self.dictionary.packet(group, stage, entry)
                rows.append(
                    HardIncidence(
                        incidence=Incidence(
                            handle,
                            packet.packet_id,
                            gain_q,
                        ),
                        packet=packet,
                        group=group,
                        stage=stage,
                        entry=entry,
                        residual_reduction=(
                            dequantize_gain(gain_q, self.gain_step) ** 2
                        ),
                    )
                )
        keys = select_packet_topk(
            tuple(row.key for row in rows),
            self.maximum_packets,
        )
        selected_key_set = set(keys)
        selected = tuple(
            row
            for row in sorted(
                rows,
                key=lambda row: (
                    -(row.incidence.gain_q * row.incidence.gain_q),
                    row.group,
                    row.stage,
                    row.entry,
                ),
            )
            if row.key in selected_key_set
        )
        return HardConceptEncoding(
            handle,
            base.payload,
            tuple(rows),
            selected,
        )

    def decode(
        self,
        base_payload: bytes,
        incidences: Sequence[HardIncidence],
    ) -> NDArray[np.float32]:
        output = decode_base_payload(base_payload).copy()
        group_size = self.dictionary.codebooks.shape[-1]
        for row in incidences:
            packet_key = self.dictionary.validate_packet(row.packet)
            if packet_key != (row.group, row.stage, row.entry):
                raise ValueError("incidence metadata does not match packet payload")
            start = row.group * group_size
            direction = self.dictionary.direction(
                row.group,
                row.stage,
                row.entry,
            ).numpy()
            output[start:start + group_size] += (
                dequantize_gain(
                    row.incidence.gain_q,
                    self.gain_step,
                )
                * direction
            )
        return output
```

Every hard encoding exposes all `group_count * stages` candidates for diagnostics but stores and decodes only the canonical top-k. At the locked production shape this ranks 60 candidates and retains exactly eight.

- [ ] **Step 4: Implement an STE whose forward value comes from the actual hard codec**

Add these types and methods to `src/ratemem/method/codec.py`:

```python
@dataclass(frozen=True)
class DifferentiableEncoding:
    reconstruction: Tensor
    base_reconstruction: Tensor
    assignment_probabilities: Tensor
    hard_indices: Tensor
    quantized_gains: Tensor
    selected_mask: Tensor
    selected_keys: tuple[tuple[PacketCandidateKey, ...], ...]


@dataclass(frozen=True)
class SoftHardAgreement:
    mean_code_error: float
    maximum_code_error: float
    assignment_disagreement: float
    topk_disagreement: float


def _ste_gain(values: Tensor, step: float) -> Tensor:
    hard = torch.round(values / step).clamp(-32768, 32767) * step
    return hard + values - values.detach()


def _hard_mask(
    keys: tuple[tuple[PacketCandidateKey, ...], ...],
    *,
    group_count: int,
    stages: int,
    device: torch.device,
) -> Tensor:
    mask = torch.zeros(
        len(keys),
        group_count,
        stages,
        dtype=torch.float32,
        device=device,
    )
    for batch_index, selected in enumerate(keys):
        for row in selected:
            mask[batch_index, row.group, row.stage] = 1.0
    return mask


class RateMemDifferentiableCodec(torch.nn.Module):
    def __init__(
        self,
        dictionary: GroupRVQDictionary,
        group_size: int,
        base_bits: int,
        gain_step: float,
        maximum_packets: int,
    ) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.group_size = group_size
        self.base_bits = base_bits
        self.gain_step = gain_step
        self.maximum_packets = maximum_packets

    def _actual_hard(
        self,
        code: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[tuple[PacketCandidateKey, ...], ...]]:
        hard_codec = RateMemHardCodec(
            BlockwiseBaseQuantizer(self.group_size, self.base_bits),
            freeze_dictionary(self.dictionary),
            self.gain_step,
            self.maximum_packets,
        )
        reconstructions = []
        bases = []
        selected_keys = []
        for batch_index, row in enumerate(code.detach().float().cpu().numpy()):
            encoded = hard_codec.encode(f"ste-{batch_index}", row)
            reconstructions.append(
                torch.from_numpy(
                    hard_codec.decode(
                        encoded.base_payload,
                        encoded.incidences,
                    )
                )
            )
            bases.append(torch.from_numpy(decode_base_payload(encoded.base_payload)))
            selected_keys.append(tuple(item.key for item in encoded.incidences))
        return (
            torch.stack(reconstructions).to(code.device),
            torch.stack(bases).to(code.device),
            tuple(selected_keys),
        )

    def forward(
        self,
        code: Tensor,
        *,
        temperature: float,
        mode: Literal["soft", "ste"],
    ) -> DifferentiableEncoding:
        if (
            code.ndim != 2
            or code.shape[1]
            != self.dictionary.group_count * self.group_size
        ):
            raise ValueError("code has the wrong width")
        if self.maximum_packets > self.dictionary.group_count * self.dictionary.stages:
            raise ValueError("maximum_packets exceeds the candidate count")
        actual, hard_base, selected_keys = self._actual_hard(code)
        base_surrogate = code.float()
        base = hard_base + base_surrogate - base_surrogate.detach()
        residual = (code.float() - base).reshape(
            code.shape[0],
            self.dictionary.group_count,
            self.group_size,
        )
        assignment = self.dictionary.soft_assign(
            residual,
            temperature=temperature,
            straight_through=(mode == "ste"),
        )
        gains = _ste_gain(assignment.gains, self.gain_step)
        hard_selected = _hard_mask(
            selected_keys,
            group_count=self.dictionary.group_count,
            stages=self.dictionary.stages,
            device=code.device,
        )
        score = gains.square().to(torch.float64)
        candidate_count = score.shape[1] * score.shape[2]
        flat_tie_rank = torch.arange(
            candidate_count,
            device=score.device,
            dtype=score.dtype,
        ).reshape(1, score.shape[1], score.shape[2])
        # Quantized squared gains differ by at least gain_step**2. This smaller
        # fixed bonus breaks exact ties in flattened (group, stage) order
        # without changing any unequal deployed gain ordering.
        tie_unit = (self.gain_step * self.gain_step) / (2 * (candidate_count + 1))
        ranked_score = score + (candidate_count - 1 - flat_tie_rank) * tie_unit
        threshold = torch.topk(
            ranked_score.reshape(code.shape[0], -1),
            k=self.maximum_packets,
            dim=-1,
        ).values[:, -1].reshape(-1, 1, 1)
        soft_selected = torch.sigmoid(
            (ranked_score - threshold.detach())
            / max(temperature, 1e-6)
        ).to(gains.dtype)
        selection = (
            hard_selected
            + soft_selected
            - soft_selected.detach()
            if mode == "ste"
            else soft_selected
        )
        directions = F.normalize(
            self.dictionary.codebooks.float(),
            dim=-1,
            eps=1e-8,
        )
        surrogate = base.reshape(
            code.shape[0],
            self.dictionary.group_count,
            self.group_size,
        )
        for stage in range(self.dictionary.stages):
            chosen = torch.einsum(
                "bge,ged->bgd",
                assignment.probabilities[:, :, stage],
                directions[:, stage],
            )
            surrogate = surrogate + (
                gains[:, :, stage].unsqueeze(-1)
                * selection[:, :, stage].unsqueeze(-1)
                * chosen
            )
        surrogate = surrogate.reshape_as(code)
        reconstruction = (
            actual + surrogate - surrogate.detach()
            if mode == "ste"
            else surrogate
        )
        hard_indices = torch.full(
            (
                code.shape[0],
                self.dictionary.group_count,
                self.dictionary.stages,
            ),
            -1,
            dtype=torch.long,
            device=code.device,
        )
        hard_gain_q = torch.zeros_like(hard_indices)
        for batch_index, selected in enumerate(selected_keys):
            for row in selected:
                hard_indices[batch_index, row.group, row.stage] = row.entry
                hard_gain_q[batch_index, row.group, row.stage] = row.gain_q
        return DifferentiableEncoding(
            reconstruction=reconstruction,
            base_reconstruction=base,
            assignment_probabilities=assignment.probabilities,
            hard_indices=hard_indices,
            quantized_gains=hard_gain_q,
            selected_mask=selection,
            selected_keys=selected_keys,
        )

    def hard_reference(self, code: Tensor) -> DifferentiableEncoding:
        return self.forward(
            code,
            temperature=1.0,
            mode="ste",
        )
```

The STE selection mask returned to the rate loss is `selection`: its forward value is the deployed hard top-k, while its backward value uses the temperature-controlled soft membership. Pure `mode="soft"` is diagnostic only and is never used for a deployment receipt. Quantized int16 gains, candidate entry IDs, tie order, and the reconstructed STE forward tensor all come from `RateMemHardCodec.encode` followed by the real `RateMemHardCodec.decode`.

- [ ] **Step 5: Contract-test production 60-to-8 agreement and gradients**

```python
# tests/contract/method/test_production_topk.py
import numpy as np
import torch

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer
from ratemem.method.codec import RateMemDifferentiableCodec, RateMemHardCodec
from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary


def test_production_ste_matches_actual_hard_60_candidate_top8() -> None:
    torch.manual_seed(20260824)
    dictionary = GroupRVQDictionary(
        group_count=30,
        group_size=16,
        stages=2,
        entries=64,
    )
    differentiable = RateMemDifferentiableCodec(
        dictionary,
        group_size=16,
        base_bits=4,
        gain_step=1 / 256,
        maximum_packets=8,
    )
    code = torch.randn(2, 480, requires_grad=True)
    ste = differentiable(code, temperature=0.25, mode="ste")
    hard = RateMemHardCodec(
        BlockwiseBaseQuantizer(16, 4),
        freeze_dictionary(dictionary),
        gain_step=1 / 256,
        maximum_packets=8,
    )
    for batch_index in range(2):
        encoded = hard.encode(
            f"h{batch_index}",
            code[batch_index].detach().numpy().astype(np.float32),
        )
        actual = torch.from_numpy(
            hard.decode(encoded.base_payload, encoded.incidences)
        )
        assert len(encoded.all_candidates) == 60
        assert len(encoded.incidences) == 8
        assert ste.selected_keys[batch_index] == tuple(
            row.key for row in encoded.incidences
        )
        assert int(ste.selected_mask[batch_index].detach().sum().item()) == 8
        for row in encoded.incidences:
            assert (
                int(
                    ste.quantized_gains[
                        batch_index,
                        row.group,
                        row.stage,
                    ].item()
                )
                == row.key.gain_q
            )
        torch.testing.assert_close(
            ste.reconstruction[batch_index].detach().cpu(),
            actual,
            rtol=0.0,
            atol=1e-6,
        )
    ste.reconstruction.square().mean().backward()
    assert code.grad is not None
    assert dictionary.codebooks.grad is not None


def test_production_zero_gain_ties_match_deployed_lexicographic_top8() -> None:
    torch.manual_seed(20260824)
    dictionary = GroupRVQDictionary(30, 16, 2, 64)
    differentiable = RateMemDifferentiableCodec(
        dictionary,
        group_size=16,
        base_bits=4,
        gain_step=1 / 256,
        maximum_packets=8,
    )
    code = torch.zeros(1, 480, requires_grad=True)
    ste = differentiable(code, temperature=0.25, mode="ste")
    hard = RateMemHardCodec(
        BlockwiseBaseQuantizer(16, 4),
        freeze_dictionary(dictionary),
        gain_step=1 / 256,
        maximum_packets=8,
    ).encode("zero-tie", np.zeros(480, dtype=np.float32))
    assert ste.selected_keys[0] == tuple(row.key for row in hard.incidences)
    assert [(row.group, row.stage) for row in ste.selected_keys[0]] == [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (2, 0), (2, 1), (3, 0), (3, 1),
    ]
    assert int(ste.selected_mask.detach().sum().item()) == 8
```

```python
# tests/contract/method/test_soft_hard_agreement.py
import pytest
import torch

from ratemem.method.codec import (
    RateMemDifferentiableCodec,
    enforce_agreement,
    measure_soft_hard_agreement,
)
from tests.unit.method.test_dictionary import dictionary


def test_locked_soft_hard_agreement_includes_topk_membership() -> None:
    codec = RateMemDifferentiableCodec(
        dictionary(),
        group_size=4,
        base_bits=4,
        gain_step=1 / 256,
        maximum_packets=2,
    )
    code = torch.tensor(
        [[1.5, 0.0, 0.0, 0.0, 0.0, 1.3, 1.2, 0.0]]
    )
    soft = codec(code, temperature=0.01, mode="soft")
    hard = codec(code, temperature=1.0, mode="ste")
    report = measure_soft_hard_agreement(soft, hard)
    enforce_agreement(
        report,
        maximum_mean_code_error=0.02,
        maximum_assignment_disagreement=0.05,
        maximum_topk_disagreement=0.0,
    )


def test_release_gate_rejects_any_topk_disagreement() -> None:
    with pytest.raises(RuntimeError, match="soft-hard agreement"):
        enforce_agreement(
            mean_code_error=0.0,
            assignment_disagreement=0.0,
            topk_disagreement=0.01,
            maximum_mean_code_error=0.02,
            maximum_assignment_disagreement=0.05,
            maximum_topk_disagreement=0.0,
        )
```

Implement `measure_soft_hard_agreement` so assignment disagreement is measured only on deployed top-k positions and top-k disagreement is the symmetric difference divided by eight. `enforce_agreement` accepts either the report object or explicit values, rejects nonfinite values, and reads the locked thresholds without modifying them.

- [ ] **Step 6: Thread top-k into the differentiable rate objective**

Change `expected_rate_loss` in Task 10 to accept `selected_mask: Tensor` and `candidate_cost_bytes: Tensor` with shape `[group_count, stages]`. It computes:

```python
def expected_rate_loss(
    selected_mask: Tensor,
    candidate_cost_bytes: Tensor,
    budget_bytes: int,
) -> Tensor:
    if (
        budget_bytes <= 0
        or selected_mask.ndim != 3
        or candidate_cost_bytes.shape != selected_mask.shape[1:]
    ):
        raise ValueError("rate tensors or byte budget have the wrong shape")
    expected = (
        selected_mask.float()
        * candidate_cost_bytes.to(selected_mask).unsqueeze(0)
    ).sum()
    return expected / (selected_mask.shape[0] * budget_bytes)
```

Pass `encoding.selected_mask`, not all RVQ entry probabilities, from `SequentialMetaTrainer`. Add a test that a 60-candidate tensor with exactly eight selected positions charges exactly those eight canonical bundle costs.

- [ ] **Step 7: Run the codec contracts and commit**

Run:

```bash
uv run pytest tests/unit/method/test_codec.py tests/contract/method/test_soft_hard_agreement.py tests/contract/method/test_production_topk.py -q
uv run ruff check src/ratemem/method/codec.py tests/unit/method/test_codec.py tests/contract/method/test_soft_hard_agreement.py tests/contract/method/test_production_topk.py
uv run mypy src/ratemem/method/codec.py
```

Expected: all tests pass; the production test proves that STE forward uses the same eight packet keys, int16 gains, and tie order as the actual hard decoder, and matches its float32 reconstruction within the locked absolute tolerance `1e-6` with zero relative tolerance.

```bash
git add src/ratemem/method/codec.py tests/unit/method/test_codec.py tests/contract/method/test_soft_hard_agreement.py tests/contract/method/test_production_topk.py
git commit -m "feat(method): align STE and deployed packet top-k"
```


### Task 5: Propose causal immutable packet bundles with exact resident reuse

**Files:**
- Create: `src/ratemem/method/proposal.py`
- Test: `tests/unit/method/test_proposal.py`
- Test: `tests/contract/method/test_proposal_causality.py`

- [ ] **Step 1: Write reuse, bundle-closure, and exact-cost tests**

```python
# tests/unit/method/test_proposal.py
import pytest

from ratemem.method.proposal import CausalCandidateProposer
from ratemem.state.model import Incidence, MemoryState


def test_existing_packet_is_stored_once_and_bundle_contains_every_incidence(hard_codec, shared_codes) -> None:
    proposer = CausalCandidateProposer(hard_codec)
    first = proposer.propose(MemoryState(), "a", shared_codes[0], event_index=1)
    state = MemoryState(
        bases={"a": first.base_record},
        packets={bundle.packet.packet_id: bundle.packet for bundle in first.bundles},
        incidences={
            (edge.handle, edge.packet_id): edge
            for bundle in first.bundles for edge in bundle.incidences
        },
    )
    second = proposer.propose(state, "b", shared_codes[1], event_index=2)
    reused = [bundle for bundle in second.bundles if {edge.handle for edge in bundle.incidences} == {"a", "b"}]
    assert reused
    assert reused[0].packet.packet_id in state.packets
    assert reused[0].cost_bytes == reused[0].measured_cost_bytes()


def test_bundle_is_frozen_and_incidence_order_is_canonical(hard_codec, shared_codes) -> None:
    bundle = CausalCandidateProposer(hard_codec).propose(
        MemoryState(), "b", shared_codes[1], event_index=2
    ).bundles[0]
    assert tuple(edge.handle for edge in bundle.incidences) == tuple(
        sorted(edge.handle for edge in bundle.incidences)
    )
    with pytest.raises((AttributeError, TypeError)):
        bundle.incidences += (Incidence("x", bundle.packet.packet_id, 1),)
```

- [ ] **Step 2: Run proposal tests and verify the module is absent**

Run: `uv run pytest tests/unit/method/test_proposal.py -q`

Expected: collection fails because `ratemem.method.proposal` does not exist.

- [ ] **Step 3: Implement immutable complete bundle proposals**

```python
# src/ratemem/method/proposal.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ratemem.method.codec import HardConceptEncoding, RateMemHardCodec
from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import bundle_cost_bytes


@dataclass(frozen=True, slots=True)
class ImmutableBundleProposal:
    packet: Packet
    incidences: Sequence[Incidence]
    cost_bytes: int

    def __post_init__(self) -> None:
        canonical = tuple(sorted(self.incidences, key=lambda edge: (edge.handle, edge.packet_id)))
        if tuple(self.incidences) != canonical:
            raise ValueError("bundle incidences must use canonical order")
        object.__setattr__(self, "incidences", canonical)
        if any(edge.packet_id != self.packet.packet_id for edge in self.incidences):
            raise ValueError("bundle incidence references another packet")
        if len({edge.handle for edge in self.incidences}) != len(self.incidences):
            raise ValueError("bundle repeats one concept")
        if self.cost_bytes != bundle_cost_bytes(self.packet, self.incidences):
            raise ValueError("bundle cost does not match canonical serialization")

    def measured_cost_bytes(self) -> int:
        return bundle_cost_bytes(self.packet, self.incidences)


@dataclass(frozen=True, slots=True)
class ConceptProposal:
    handle: str
    event_index: int
    base_record: BaseRecord
    bundles: Sequence[ImmutableBundleProposal]


class CausalCandidateProposer:
    def __init__(self, codec: RateMemHardCodec) -> None:
        self.codec = codec

    def propose(
        self,
        state: MemoryState,
        handle: str,
        current_target_code: NDArray[np.float32],
        event_index: int,
    ) -> ConceptProposal:
        if event_index < 0:
            raise ValueError("event_index must be nonnegative")
        encoded: HardConceptEncoding = self.codec.encode(handle, current_target_code)
        previous = state.bases.get(handle)
        base = BaseRecord(
            handle=handle,
            payload=encoded.base_payload,
            reads=0 if previous is None else previous.reads,
            created_at=event_index if previous is None else previous.created_at,
        )
        by_packet: dict[str, tuple[Packet, dict[str, Incidence]]] = {}
        for packet_id, packet in state.packets.items():
            edges = {
                edge.handle: edge
                for edge in state.incidences.values()
                if edge.packet_id == packet_id and edge.handle != handle
            }
            by_packet[packet_id] = (packet, edges)
        for row in encoded.incidences:
            packet, edges = by_packet.get(row.packet.packet_id, (row.packet, {}))
            edges[row.incidence.handle] = row.incidence
            by_packet[row.packet.packet_id] = (packet, edges)
        bundles = []
        for packet_id, (packet, edges) in sorted(by_packet.items()):
            if not edges:
                continue
            incidences = tuple(sorted(edges.values(), key=lambda edge: (edge.handle, edge.packet_id)))
            bundles.append(ImmutableBundleProposal(
                packet=packet,
                incidences=incidences,
                cost_bytes=bundle_cost_bytes(packet, incidences),
            ))
        return ConceptProposal(handle, event_index, base, tuple(bundles))
```

- [ ] **Step 4: Add a forward-only API and source contract**

```python
# tests/contract/method/test_proposal_causality.py
import ast
import inspect

from ratemem.method.proposal import CausalCandidateProposer


def test_proposal_api_has_no_future_trace_or_query_images() -> None:
    parameters = set(inspect.signature(CausalCandidateProposer.propose).parameters)
    assert parameters == {"self", "state", "handle", "current_target_code", "event_index"}


def test_proposal_source_cannot_import_evaluator_or_final_trace() -> None:
    tree = ast.parse(inspect.getsource(__import__("ratemem.method.proposal", fromlist=["*"])))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith("ratemem.evaluation.final_trace") for name in imported)
    assert not any("evaluator" in name for name in imported)
```

Also add a test that a concept update replaces only that concept's incidence within a reused bundle, preserves all other incidences byte-for-byte, and emits a new immutable proposal object rather than changing the previous proposal.

- [ ] **Step 5: Run proposal contracts and commit**

Run:

```bash
uv run pytest tests/unit/method/test_proposal.py tests/contract/method/test_proposal_causality.py -q
uv run ruff check src/ratemem/method/proposal.py tests/unit/method/test_proposal.py tests/contract/method/test_proposal_causality.py
uv run mypy src/ratemem/method/proposal.py
```

Expected: all focused tests pass; no future-trace or evaluator dependency appears in the proposal module.

```bash
git add src/ratemem/method/proposal.py tests/unit/method/test_proposal.py tests/contract/method/test_proposal_causality.py
git commit -m "feat(method): propose immutable shared packet bundles"
```

### Task 6: Calibrate a causal nonnegative coverage utility

**Files:**
- Create: `src/ratemem/method/utility.py`
- Create: `schemas/ratemem-utility-calibration-v1.schema.json`
- Test: `tests/unit/method/test_utility.py`
- Test: `tests/contract/method/test_utility_calibration.py`

- [ ] **Step 1: Write history isolation, nonnegativity, and absent-incidence tests**

```python
# tests/unit/method/test_utility.py
import pytest
import torch

from ratemem.method.utility import (
    CausalFeatureBatch,
    CausalRequestHistory,
    NonnegativeUtilityCalibrator,
)


def test_request_weight_uses_only_operational_reads_before_allocation() -> None:
    history = CausalRequestHistory(decay=0.97)
    history = history.observe_read("a", event_index=2, operational=True)
    history = history.observe_read("a", event_index=3, operational=False)
    assert history.weight("a", allocation_event_index=4) == pytest.approx(0.97)
    with pytest.raises(ValueError, match="future"):
        history.weight("a", allocation_event_index=2)


def test_beta_and_packet_gain_outputs_are_finite_and_nonnegative() -> None:
    model = NonnegativeUtilityCalibrator(concept_features=4, incidence_features=5, hidden=8, groups=3)
    batch = CausalFeatureBatch(
        concept=torch.randn(2, 4),
        incidence=torch.randn(2, 4, 5),
        incidence_mask=torch.tensor([[True, False, True, False], [False, True, True, False]]),
        maximum_source_event_index=torch.tensor([5, 7]),
        allocation_event_index=torch.tensor([5, 8]),
    )
    result = model(batch)
    assert torch.isfinite(result.beta).all() and torch.all(result.beta >= 0)
    assert torch.isfinite(result.value).all() and torch.all(result.value >= 0)
    assert torch.all(result.value[~batch.incidence_mask] == 0)
```

- [ ] **Step 2: Run utility tests and verify the module is absent**

Run: `uv run pytest tests/unit/method/test_utility.py -q`

Expected: collection fails because `ratemem.method.utility` does not exist.

- [ ] **Step 3: Implement immutable request history and the nonnegative model**

```python
# src/ratemem/method/utility.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CausalRequestHistory:
    decay: float
    reads: Mapping[str, Sequence[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("request decay must be in (0, 1]")
        object.__setattr__(self, "reads", MappingProxyType({key: tuple(value) for key, value in self.reads.items()}))

    def observe_read(self, handle: str, event_index: int, operational: bool) -> "CausalRequestHistory":
        if not operational:
            return self
        rows = dict(self.reads)
        rows[handle] = tuple(sorted((*rows.get(handle, ()), event_index)))
        return CausalRequestHistory(self.decay, rows)

    def weight(self, handle: str, allocation_event_index: int) -> float:
        events = self.reads.get(handle, ())
        if any(index >= allocation_event_index for index in events):
            raise ValueError("future or current-event read reached allocation history")
        return float(sum(self.decay ** (allocation_event_index - 1 - index) for index in events))


@dataclass(frozen=True)
class CausalFeatureBatch:
    concept: Tensor
    incidence: Tensor
    incidence_mask: Tensor
    maximum_source_event_index: Tensor
    allocation_event_index: Tensor


@dataclass(frozen=True)
class UtilityPrediction:
    beta: Tensor
    value: Tensor


class NonnegativeUtilityCalibrator(nn.Module):
    def __init__(self, concept_features: int, incidence_features: int, hidden: int, groups: int) -> None:
        super().__init__()
        self.groups = groups
        self.concept_net = nn.Sequential(nn.Linear(concept_features, hidden), nn.SiLU(), nn.Linear(hidden, groups))
        self.incidence_net = nn.Sequential(nn.Linear(incidence_features, hidden), nn.SiLU(), nn.Linear(hidden, groups))

    def forward(self, batch: CausalFeatureBatch) -> UtilityPrediction:
        if torch.any(batch.maximum_source_event_index > batch.allocation_event_index):
            raise ValueError("future feature reached the utility calibrator")
        if batch.incidence_mask.shape != batch.incidence.shape[:2]:
            raise ValueError("incidence mask shape does not match feature rows")
        beta = F.softplus(self.concept_net(batch.concept.float()))
        raw = F.softplus(self.incidence_net(batch.incidence.float()))
        value = raw * batch.incidence_mask.unsqueeze(-1).to(raw.dtype)
        return UtilityPrediction(beta=beta, value=value)


class CalibrationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    split: Literal["calibration"]
    method_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bin_edges: list[float]
    bin_count: list[int]
    predicted_mean: list[float]
    observed_mean: list[float]
    expected_calibration_error: float = Field(ge=0.0)
    maximum_allowed_ece: float = Field(ge=0.0)


def calibration_receipt(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    bins: int,
    method_lock_sha256: str,
    feature_manifest_sha256: str,
    label_artifact_sha256: str,
    maximum_allowed_ece: float,
) -> CalibrationReceipt:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 1 or predicted.size == 0:
        raise ValueError("calibration arrays must be nonempty aligned vectors")
    if np.any(predicted < 0) or np.any(observed < 0) or not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("calibration gains must be finite and nonnegative")
    edges = np.linspace(0.0, max(1.0, float(predicted.max())), bins + 1)
    assignments = np.minimum(np.digitize(predicted, edges[1:-1]), bins - 1)
    counts, predicted_means, observed_means = [], [], []
    weighted_error = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        p_mean = float(predicted[mask].mean()) if count else 0.0
        o_mean = float(observed[mask].mean()) if count else 0.0
        counts.append(count)
        predicted_means.append(p_mean)
        observed_means.append(o_mean)
        weighted_error += count * abs(p_mean - o_mean)
    ece = weighted_error / predicted.size
    return CalibrationReceipt(
        split="calibration", method_lock_sha256=method_lock_sha256,
        feature_manifest_sha256=feature_manifest_sha256,
        label_artifact_sha256=label_artifact_sha256,
        bin_edges=[float(value) for value in edges], bin_count=counts,
        predicted_mean=predicted_means, observed_mean=observed_means,
        expected_calibration_error=ece, maximum_allowed_ece=maximum_allowed_ece,
    )
```

- [ ] **Step 4: Convert calibrated predictions to the exact coverage oracle**

Add `build_coverage_oracle` with this contract: it accepts a fixed cohort, immutable bundle proposals, `CausalRequestHistory`, allocation event index, one prediction per incidence, and one beta vector per concept. It returns core `PacketBundle` objects whose `cost_bytes` equals each proposal's measured canonical cost, whose gain tuple is zero for absent incidences/groups, and whose request weights are read exclusively through `history.weight(handle, allocation_event_index)`. Add `1.0` as the cold-start request weight only for a concept created by the current event; record that rule in the returned `UtilityAudit`.

```python
@dataclass(frozen=True, slots=True)
class UtilityAudit:
    allocation_event_index: int
    maximum_feature_event_index: int
    cold_start_handles: Sequence[str]
    request_weights: Mapping[str, float]


def enforce_calibration(receipt: CalibrationReceipt) -> None:
    if receipt.expected_calibration_error > receipt.maximum_allowed_ece:
        raise RuntimeError("utility calibration ECE exceeds the locked threshold")
```

The builder calls `enforce_calibration` before constructing the oracle and rejects a receipt from any split other than `calibration` through Pydantic validation.

- [ ] **Step 5: Generate and verify the calibration schema**

```python
# tests/contract/method/test_utility_calibration.py
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.method.utility import CalibrationReceipt, calibration_receipt, enforce_calibration


def test_calibration_schema_is_current_and_threshold_is_fail_closed() -> None:
    path = Path("schemas/ratemem-utility-calibration-v1.schema.json")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert path.read_bytes() == canonical_json_bytes(CalibrationReceipt.model_json_schema())
    receipt = calibration_receipt(
        predicted=[0.0, 1.0], observed=[1.0, 0.0], bins=2,
        method_lock_sha256="a" * 64, feature_manifest_sha256="b" * 64,
        label_artifact_sha256="c" * 64, maximum_allowed_ece=0.05,
    )
    with pytest.raises(RuntimeError, match="ECE"):
        enforce_calibration(receipt)
```

Run:

```bash
uv run python -c 'from pathlib import Path; from ratemem.evaluation.canonical import canonical_json_bytes; from ratemem.method.utility import CalibrationReceipt; Path("schemas/ratemem-utility-calibration-v1.schema.json").write_bytes(canonical_json_bytes(CalibrationReceipt.model_json_schema()))'
uv run pytest tests/unit/method/test_utility.py tests/contract/method/test_utility_calibration.py -q
```

Expected: all focused tests pass; an over-threshold or non-calibration receipt blocks oracle construction.

- [ ] **Step 6: Run static checks and commit**

Run: `uv run ruff check src/ratemem/method/utility.py tests/unit/method/test_utility.py tests/contract/method/test_utility_calibration.py && uv run mypy src/ratemem/method/utility.py`

Expected: Ruff and mypy exit 0.

```bash
git add src/ratemem/method/utility.py schemas/ratemem-utility-calibration-v1.schema.json tests/unit/method/test_utility.py tests/contract/method/test_utility_calibration.py
git commit -m "feat(method): calibrate causal packet utility"
```

### Task 7: Add the empirical outer lifecycle controller around the certified allocator

**Files:**
- Create: `src/ratemem/method/controller.py`
- Test: `tests/unit/method/test_controller.py`
- Test: `tests/contract/method/test_controller_theorem_boundary.py`

- [ ] **Step 1: Write admission, eviction, rejection, and exact-budget tests**

```python
# tests/unit/method/test_controller.py
from collections import Counter

import pytest

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.method import controller as controller_module
from ratemem.method.controller import RateMemController
from ratemem.method.proposal import ConceptProposal, ImmutableBundleProposal
from ratemem.state.model import BaseRecord, Incidence, MemoryState
from ratemem.state.serialization import bundle_cost_bytes, packet_from_payload


def value_oracle(cohort, bundles):
    return CoverageOracle(
        bundles={
            row.packet.packet_id: PacketBundle(
                row.packet.packet_id,
                row.cost_bytes,
                {edge.handle: (1.0,) for edge in row.incidences},
            )
            for row in bundles
        },
        request_weights={handle: 1.0 for handle in cohort},
        group_weights={handle: (1.0,) for handle in cohort},
    )


@pytest.fixture
def four_concept_32_bundle_case():
    bases = {}
    packets = {}
    incidences = {}
    bundles = []
    proposed_base = None
    for event_index, handle in enumerate(("a", "b", "c", "d"), start=1):
        base = BaseRecord(handle, f"base-{handle}".encode(), 0, event_index)
        if handle == "d":
            proposed_base = base
        else:
            bases[handle] = base
        for packet_index in range(8):
            packet = packet_from_payload(f"{handle}-packet-{packet_index}".encode())
            incidence = Incidence(handle, packet.packet_id, 1)
            bundles.append(ImmutableBundleProposal(
                packet, (incidence,), bundle_cost_bytes(packet, (incidence,))
            ))
            if handle != "d":
                packets[packet.packet_id] = packet
                incidences[(handle, packet.packet_id)] = incidence
    assert proposed_base is not None
    state = MemoryState(bases=bases, packets=packets, incidences=incidences)
    proposal = ConceptProposal("d", 4, proposed_base, tuple(bundles))
    assert len(proposal.bundles) == 32
    assert Counter(
        edge.handle
        for bundle in proposal.bundles
        for edge in bundle.incidences
    ) == Counter({"a": 8, "b": 8, "c": 8, "d": 8})
    return RateMemController(1_000_000, value_oracle), state, proposal


def test_controller_prescreens_four_concepts_with_eight_distinct_packets_each(
    monkeypatch, four_concept_32_bundle_case
) -> None:
    controller, state, proposal = four_concept_32_bundle_case
    observed_prescreens: list[tuple[int, int]] = []
    observed_allocations: list[tuple[int, int]] = []
    real_prescreen = controller_module.prescreen_certified_oracle
    real_allocate = controller_module.allocate_snapshot

    def recording_prescreen(oracle, budget_bytes, *, max_bundles=24):
        observed_prescreens.append((len(oracle.bundles), max_bundles))
        return real_prescreen(oracle, budget_bytes, max_bundles=max_bundles)

    def recording_allocate(oracle, budget_bytes, *, max_bundles=24):
        observed_allocations.append((len(oracle.bundles), max_bundles))
        return real_allocate(oracle, budget_bytes, max_bundles=max_bundles)

    monkeypatch.setattr(
        controller_module, "prescreen_certified_oracle", recording_prescreen
    )
    monkeypatch.setattr(controller_module, "allocate_snapshot", recording_allocate)
    decision = controller.apply_create(state, proposal)
    assert decision.outcome == "created"
    assert len(proposal.bundles) == 32
    assert observed_prescreens == [(32, 24)]
    assert observed_allocations == [(24, 24)]
    assert decision.state.serialized_bytes <= 1_000_000


def test_controller_never_exceeds_serialized_budget(proposer, shared_codes) -> None:
    controller = RateMemController(budget_bytes=700, oracle_factory=value_oracle)
    first = proposer.propose(MemoryState(), "a", shared_codes[0], event_index=1)
    after_first = controller.apply_create(MemoryState(), first)
    second = proposer.propose(after_first.state, "b", shared_codes[1], event_index=2)
    after_second = controller.apply_create(after_first.state, second)
    assert after_second.state.serialized_bytes <= 700
    assert after_second.theorem_scope == "fixed_admitted_cohort_prescreened_packets_only"


def test_oversized_base_is_rejected_without_mutating_old_state(proposer, shared_codes) -> None:
    controller = RateMemController(budget_bytes=64, oracle_factory=value_oracle)
    proposal = proposer.propose(MemoryState(), "a", shared_codes[0], event_index=1)
    result = controller.apply_create(MemoryState(), proposal)
    assert result.outcome == "rejected"
    assert result.state == MemoryState()


def test_delete_collects_only_packets_without_remaining_dependents(shared_state) -> None:
    controller = RateMemController(budget_bytes=4096, oracle_factory=value_oracle)
    after_a = controller.delete(shared_state, "a")
    assert after_a.outcome == "deleted"
    assert after_a.state.packets
    after_b = controller.delete(after_a.state, "b")
    assert after_b.state.packets == {}
```

- [ ] **Step 2: Run controller tests and verify the module is absent**

Run: `uv run pytest tests/unit/method/test_controller.py -q`

Expected: collection fails because `ratemem.method.controller` does not exist.

- [ ] **Step 3: Implement deterministic whole-base admission outside the theorem**

```python
# src/ratemem/method/controller.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from ratemem.allocation.objective import CoverageOracle
from ratemem.allocation.snapshot import allocate_snapshot, prescreen_certified_oracle
from ratemem.method.proposal import ConceptProposal, ImmutableBundleProposal
from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet

OracleFactory = Callable[[Sequence[str], Sequence[ImmutableBundleProposal]], CoverageOracle]


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    state: MemoryState
    outcome: Literal["created", "updated", "deleted", "rejected", "read", "stale_handle"]
    evicted_handles: Sequence[str] = ()
    selected_packet_ids: Sequence[str] = ()
    theorem_scope: Literal["fixed_admitted_cohort_prescreened_packets_only"] = (
        "fixed_admitted_cohort_prescreened_packets_only"
    )


def _without_handle(state: MemoryState, handle: str) -> MemoryState:
    bases = {key: row for key, row in state.bases.items() if key != handle}
    incidences = {key: row for key, row in state.incidences.items() if row.handle != handle}
    referenced = {row.packet_id for row in incidences.values()}
    packets = {key: row for key, row in state.packets.items() if key in referenced}
    return MemoryState(bases=bases, packets=packets, incidences=incidences)


def _base_only(bases: dict[str, BaseRecord]) -> MemoryState:
    return MemoryState(bases=bases, packets={}, incidences={})


def _base_increment_bytes(bases: dict[str, BaseRecord], handle: str) -> int:
    with_row = _base_only(bases).serialized_bytes
    without_row = _base_only({key: row for key, row in bases.items() if key != handle}).serialized_bytes
    return with_row - without_row


class RateMemController:
    def __init__(
        self,
        budget_bytes: int,
        oracle_factory: OracleFactory,
        certified_prescreen_max_bundles: Literal[24] = 24,
    ) -> None:
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be nonnegative")
        if (
            type(certified_prescreen_max_bundles) is not int
            or certified_prescreen_max_bundles != 24
        ):
            raise ValueError("certified prescreen cap must equal the locked value 24")
        self.budget_bytes = budget_bytes
        self.oracle_factory = oracle_factory
        self.certified_prescreen_max_bundles = certified_prescreen_max_bundles

    def _admit_bases(
        self, state: MemoryState, proposal: ConceptProposal
    ) -> tuple[dict[str, BaseRecord] | None, Sequence[str]]:
        bases = dict(state.bases)
        bases[proposal.handle] = proposal.base_record
        evicted: list[str] = []
        while _base_only(bases).serialized_bytes > self.budget_bytes:
            candidates = [handle for handle in bases if handle != proposal.handle]
            if not candidates:
                return None, tuple(evicted)
            victim = min(
                candidates,
                key=lambda handle: (
                    (bases[handle].reads + 1) / _base_increment_bytes(bases, handle),
                    bases[handle].created_at,
                    handle,
                ),
            )
            del bases[victim]
            evicted.append(victim)
        return bases, tuple(evicted)

    @staticmethod
    def _project_bundles(
        bundles: Sequence[ImmutableBundleProposal], cohort: set[str]
    ) -> Sequence[ImmutableBundleProposal]:
        projected = []
        for bundle in bundles:
            incidences = tuple(edge for edge in bundle.incidences if edge.handle in cohort)
            if incidences:
                from ratemem.state.serialization import bundle_cost_bytes

                projected.append(ImmutableBundleProposal(
                    bundle.packet, incidences, bundle_cost_bytes(bundle.packet, incidences)
                ))
        return tuple(projected)

    def _apply(self, state: MemoryState, proposal: ConceptProposal, outcome: Literal["created", "updated"]) -> ControllerDecision:
        admitted, evicted = self._admit_bases(state, proposal)
        if admitted is None:
            return ControllerDecision(state=state, outcome="rejected")
        base_state = _base_only(admitted)
        cohort = tuple(sorted(admitted))
        bundles = self._project_bundles(proposal.bundles, set(cohort))
        oracle = self.oracle_factory(cohort, bundles)
        if set(oracle.bundles) != {row.packet.packet_id for row in bundles}:
            raise ValueError("oracle and immutable proposal ground sets differ")
        residual_budget = self.budget_bytes - base_state.serialized_bytes
        certified_oracle = prescreen_certified_oracle(
            oracle,
            residual_budget,
            max_bundles=self.certified_prescreen_max_bundles,
        )
        selected = allocate_snapshot(
            certified_oracle,
            residual_budget,
            max_bundles=self.certified_prescreen_max_bundles,
        )
        packets: dict[str, Packet] = {}
        incidences: dict[tuple[str, str], Incidence] = {}
        by_id = {row.packet.packet_id: row for row in bundles}
        for packet_id in sorted(selected):
            bundle = by_id[packet_id]
            packets[packet_id] = bundle.packet
            for edge in bundle.incidences:
                incidences[(edge.handle, edge.packet_id)] = edge
        result = MemoryState(bases=admitted, packets=packets, incidences=incidences)
        if result.serialized_bytes > self.budget_bytes:
            raise RuntimeError("allocator produced a state above the exact byte budget")
        return ControllerDecision(
            state=result,
            outcome=outcome,
            evicted_handles=evicted,
            selected_packet_ids=tuple(sorted(selected)),
        )

    def apply_create(self, state: MemoryState, proposal: ConceptProposal) -> ControllerDecision:
        if proposal.handle in state.bases:
            raise ValueError("create received an active handle")
        return self._apply(state, proposal, "created")

    def apply_update(self, state: MemoryState, proposal: ConceptProposal) -> ControllerDecision:
        if proposal.handle not in state.bases:
            return ControllerDecision(state=state, outcome="stale_handle")
        return self._apply(state, proposal, "updated")

    def delete(self, state: MemoryState, handle: str) -> ControllerDecision:
        if handle not in state.bases:
            return ControllerDecision(state=state, outcome="stale_handle")
        return ControllerDecision(state=_without_handle(state, handle), outcome="deleted")
```

- [ ] **Step 4: Add read semantics and the theorem-boundary receipt**

Add this method to `RateMemController`; it copies only the requested `BaseRecord` with `reads + 1` when usage updates are enabled and returns the identical state object for a scoring read:

```python
def read(self, state: MemoryState, handle: str, update_usage: bool) -> ControllerDecision:
    if handle not in state.bases:
        return ControllerDecision(state=state, outcome="stale_handle")
    if not update_usage:
        return ControllerDecision(state=state, outcome="read")
    old = state.bases[handle]
    bases = dict(state.bases)
    bases[handle] = BaseRecord(old.handle, old.payload, old.reads + 1, old.created_at)
    updated = MemoryState(bases=bases, packets=state.packets, incidences=state.incidences)
    if updated.serialized_bytes != state.serialized_bytes:
        raise RuntimeError("fixed-width usage update changed serialized state length")
    return ControllerDecision(state=updated, outcome="read")
```

```python
# tests/contract/method/test_controller_theorem_boundary.py
import inspect

from ratemem.method.controller import ControllerDecision, RateMemController


def test_outer_policy_cannot_be_reported_as_theorem_covered() -> None:
    assert ControllerDecision.__dataclass_fields__["theorem_scope"].default == (
        "fixed_admitted_cohort_prescreened_packets_only"
    )
    source = inspect.getsource(RateMemController._admit_bases)
    assert "allocate_snapshot" not in source
    assert "switching" not in source


def test_read_only_probe_returns_identical_state(controller, populated_state) -> None:
    decision = controller.read(populated_state, "a", update_usage=False)
    assert decision.state is populated_state
    assert decision.state.serialized_bytes == populated_state.serialized_bytes
```

The two-argument `RateMemController` construction remains valid through the locked default. At the
method-composition boundary, pass
`policy.controller.certified_prescreen_max_bundles` as the third constructor argument; the
controller stores it and passes it explicitly to both pre-screening and certified allocation. The
core helper removes individually infeasible bundles, sorts the rest by descending exact singleton
density, and retains the highest-density 24 with deterministic packet-ID ties. The
`four_concept_32_bundle_case` fixture constructs a controller and contains three resident concepts
with eight pairwise-distinct packets each and a fourth create proposal with eight further packets;
the proposal therefore carries exactly 32 complete bundles. Document in the controller docstring
that whole-base admission/eviction/rejection, cohort projection after eviction, causal pre-screening
loss relative to the full pool, and any future hysteresis are empirical outer-policy operations.
The certified claim begins only after the cohort, reduced bundle list, costs, and residual budget
are fixed.

- [ ] **Step 5: Run controller contracts and commit**

Run:

```bash
uv run pytest tests/unit/method/test_controller.py tests/contract/method/test_controller_theorem_boundary.py tests/allocation -q
uv run ruff check src/ratemem/method/controller.py tests/unit/method/test_controller.py tests/contract/method/test_controller_theorem_boundary.py
uv run mypy src/ratemem/method/controller.py
```

Expected: all focused and allocator tests pass; every resulting state is within its serialized byte budget.

```bash
git add src/ratemem/method/controller.py tests/unit/method/test_controller.py tests/contract/method/test_controller_theorem_boundary.py
git commit -m "feat(method): add bounded causal lifecycle controller"
```

### Task 8: Implement the one canonical baseline adapter without redefining its protocol

**Depends on:** Task 2 of `docs/superpowers/plans/2026-08-24-ratemem-matched-baselines.md`, which exclusively owns `src/ratemem/baselines/protocol.py` and `src/ratemem/baselines/ledger.py`.

**Files:**
- Create: `src/ratemem/method/adapter.py`
- Test: `tests/unit/method/test_adapter.py`
- Test: `tests/contract/method/test_adapter_protocol.py`
- Test: `tests/integration/method/test_canonical_baseline_adapter.py`

- [ ] **Step 1: Write failing canonical protocol and state-roundtrip tests**

```python
# tests/contract/method/test_adapter_protocol.py
from ratemem.baselines.protocol import (
    BaselineAdapter,
    CausalEventView,
    ExactByteLedger,
)
from ratemem.method.adapter import RateMemAdapter


def require_canonical_adapter(value: BaselineAdapter) -> BaselineAdapter:
    return value


def test_ratemem_is_the_canonical_causal_adapter(adapter: RateMemAdapter) -> None:
    typed = require_canonical_adapter(adapter)
    assert isinstance(typed, BaselineAdapter)
    assert typed.method_id == "ratemem_v1"
    assert typed.role == "causal"


def test_export_import_restores_identical_future_receipt(
    adapter_factory, contract, create_event, read_event
) -> None:
    original = adapter_factory()
    original.initialize(contract)
    original.apply_event(create_event, CausalEventView((create_event,), current_index=0))
    payload = original.export_online_state()
    ledger = original.state_ledger()
    assert isinstance(ledger, ExactByteLedger)
    assert ledger.online_state_bytes == len(payload)
    assert ledger.online_state_sha256 == original.copy_snapshot().state_sha256

    restored = adapter_factory()
    restored.initialize(contract)
    restored.import_online_state(payload)
    view = CausalEventView((create_event, read_event), current_index=1)
    assert restored.apply_event(read_event, view) == original.apply_event(read_event, view)
```

```python
# tests/unit/method/test_adapter.py
from ratemem.baselines.protocol import CausalEventView


def test_create_read_update_delete_and_stale_handle(
    adapter, contract, lifecycle_events
) -> None:
    adapter.initialize(contract)
    receipts = []
    for index, event in enumerate(lifecycle_events):
        receipts.append(
            adapter.apply_event(
                event,
                CausalEventView(lifecycle_events, current_index=index),
            )
        )
    assert [row.outcome for row in receipts] == [
        "created", "read", "updated", "deleted", "stale_handle"
    ]
    assert all(row.ledger.online_state_bytes <= contract.byte_budget for row in receipts)
    assert all(row.method_id == "ratemem_v1" for row in receipts)
    assert all(row.trace_id == contract.trace_id for row in receipts)
    assert all(row.candidate_stream_sha256 == contract.candidate_stream_sha256 for row in receipts)
```

- [ ] **Step 2: Run the tests and verify the adapter is absent**

Run:

```bash
uv run pytest tests/unit/method/test_adapter.py tests/contract/method/test_adapter_protocol.py -q
```

Expected: collection fails because `ratemem.method.adapter` does not exist. It must not fail because a second protocol was added under `ratemem.method` or `ratemem.evaluation`.

- [ ] **Step 3: Define injected runtime boundaries and canonical state components**

```python
# src/ratemem/method/adapter.py
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

import cbor2
import numpy as np
from numpy.typing import NDArray

from ratemem.baselines.ledger import (
    ONLINE_COMPONENT_NAMES,
    export_state,
    ledger_from_export,
)
from ratemem.baselines.protocol import (
    BaselineAdapter,
    CausalEventView,
    EventReceipt,
    ExactByteLedger,
    FrozenComparisonContract,
    MethodSnapshot,
    ProbeResult,
)
from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.evaluation.traces import (
    CreateEvent,
    DeleteEvent,
    LifecycleEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)
from ratemem.method.codec import HardIncidence, RateMemHardCodec
from ratemem.method.controller import RateMemController
from ratemem.method.proposal import CausalCandidateProposer
from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet


@runtime_checkable
class TargetCodePredictor(Protocol):
    def predict(
        self,
        support_image_ids: Sequence[str],
        description_id: str | None,
    ) -> NDArray[np.float32]:
        raise NotImplementedError


@runtime_checkable
class GenerationBackend(Protocol):
    def generate(
        self,
        adapter_code: NDArray[np.float32],
        prompt_id: str,
        seed: int,
    ) -> bytes:
        raise NotImplementedError


def _component_records(
    state: MemoryState,
    *,
    budget_bytes: int,
    last_event_index: int,
) -> Mapping[str, Sequence[object]]:
    handles = tuple(sorted(state.bases))
    packet_ids = tuple(sorted(state.packets))
    reference_counts = {
        packet_id: sum(edge.packet_id == packet_id for edge in state.incidences.values())
        for packet_id in packet_ids
    }
    records: dict[str, Sequence[object]] = {
        name: () for name in ONLINE_COMPONENT_NAMES
    }
    records.update({
        "base_codes": tuple([state.bases[handle].payload] for handle in handles),
        "packet_payloads": tuple([state.packets[packet_id].payload] for packet_id in packet_ids),
        "packet_hashes": tuple([packet_id] for packet_id in packet_ids),
        "incidences_gains": tuple(
            [edge.handle, edge.packet_id, edge.gain_q]
            for edge in sorted(
                state.incidences.values(),
                key=lambda row: (row.handle, row.packet_id),
            )
        ),
        "handles": tuple([handle] for handle in handles),
        "usage_age": tuple(
            [state.bases[handle].reads, state.bases[handle].created_at]
            for handle in handles
        ),
        "reference_counts": tuple(
            [packet_id, reference_counts[packet_id]] for packet_id in packet_ids
        ),
        "controller_state": (["budget_bytes", budget_bytes],),
        "allocator_state": (
            ["last_event_index", last_event_index],
            ["selected_packet_ids", list(packet_ids)],
        ),
    })
    return records


def _decode_component_records(payload: bytes) -> Mapping[str, Sequence[object]]:
    top = cbor2.loads(payload)
    if top.get("format") != "ratemem-baseline-cbor-v1":
        raise ValueError("unsupported canonical baseline state")
    framed = top.get("components")
    if not isinstance(framed, dict) or set(framed) != set(ONLINE_COMPONENT_NAMES):
        raise ValueError("canonical baseline state components are incomplete")
    records = {}
    for name in ONLINE_COMPONENT_NAMES:
        row = cbor2.loads(framed[name])
        if row != {"component": name, "records": row.get("records")}:
            raise ValueError("canonical component frame has the wrong identity")
        records[name] = tuple(row["records"])
    if export_state(records) != payload:
        raise ValueError("online state is not in canonical form")
    return records
```

This module imports canonical types; it never defines or monkey-patches `BaselineAdapter`, `EventReceipt`, `ExactByteLedger`, `MethodSnapshot`, or `ProbeResult`.

- [ ] **Step 4: Implement canonical export, import, ledger, and token snapshots**

Add these methods and helpers to `src/ratemem/method/adapter.py`:

```python
def _restore_state(records: Mapping[str, Sequence[object]]) -> tuple[MemoryState, int, int]:
    handles = [str(row[0]) for row in records["handles"]]
    bases_payload = [bytes(row[0]) for row in records["base_codes"]]
    usage = [(int(row[0]), int(row[1])) for row in records["usage_age"]]
    if not (len(handles) == len(bases_payload) == len(usage)):
        raise ValueError("base, handle, and usage records are misaligned")
    bases = {
        handle: BaseRecord(handle, payload, reads, created_at)
        for handle, payload, (reads, created_at) in zip(handles, bases_payload, usage)
    }
    packet_ids = [str(row[0]) for row in records["packet_hashes"]]
    packet_payloads = [bytes(row[0]) for row in records["packet_payloads"]]
    if len(packet_ids) != len(packet_payloads):
        raise ValueError("packet hashes and payloads are misaligned")
    packets = {
        packet_id: Packet(packet_id, packet_payload)
        for packet_id, packet_payload in zip(packet_ids, packet_payloads)
    }
    incidences = {
        (str(row[0]), str(row[1])): Incidence(str(row[0]), str(row[1]), int(row[2]))
        for row in records["incidences_gains"]
    }
    expected_references = {
        str(row[0]): int(row[1]) for row in records["reference_counts"]
    }
    actual_references = {
        packet_id: sum(edge.packet_id == packet_id for edge in incidences.values())
        for packet_id in packets
    }
    if expected_references != actual_references:
        raise ValueError("packet reference counts do not match incidences")
    controller = dict(records["controller_state"])
    allocator = dict(records["allocator_state"])
    return (
        MemoryState(bases=bases, packets=packets, incidences=incidences),
        int(controller["budget_bytes"]),
        int(allocator["last_event_index"]),
    )


class RateMemAdapter(BaselineAdapter):
    method_id: str = "ratemem_v1"
    role: Literal["causal", "upper_reference", "latency_control"] = "causal"

    def __init__(
        self,
        predictor: TargetCodePredictor,
        generation_backend: GenerationBackend,
        codec: RateMemHardCodec,
        controller_factory: Callable[[int], RateMemController],
        *,
        shared_trained_bytes: int,
    ) -> None:
        self.predictor = predictor
        self.generation_backend = generation_backend
        self.codec = codec
        self.proposer = CausalCandidateProposer(codec)
        self.controller_factory = controller_factory
        self.shared_trained_bytes = shared_trained_bytes
        self.contract: FrozenComparisonContract | None = None
        self.controller: RateMemController | None = None
        self.state = MemoryState()
        self.last_event_index = 0
        self._snapshot_states: dict[str, MemoryState] = {}

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self.contract is not None:
            raise RuntimeError("adapter is already initialized")
        self.contract = contract
        self.controller = self.controller_factory(contract.byte_budget)
        self.state = MemoryState()
        self.last_event_index = 0
        self._snapshot_states.clear()

    def _require_runtime(self) -> tuple[FrozenComparisonContract, RateMemController]:
        if self.contract is None or self.controller is None:
            raise RuntimeError("adapter is not initialized")
        return self.contract, self.controller

    def export_online_state(self) -> bytes:
        contract, _ = self._require_runtime()
        return export_state(
            _component_records(
                self.state,
                budget_bytes=contract.byte_budget,
                last_event_index=self.last_event_index,
            )
        )

    def import_online_state(self, payload: bytes) -> None:
        contract, _ = self._require_runtime()
        state, budget_bytes, last_event_index = _restore_state(
            _decode_component_records(payload)
        )
        if budget_bytes != contract.byte_budget:
            raise ValueError("imported state belongs to another byte budget")
        self.state = state
        self.last_event_index = last_event_index
        self._snapshot_states.clear()
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            raise ValueError("imported canonical state exceeds the byte budget")

    def state_ledger(self) -> ExactByteLedger:
        return ledger_from_export(
            self.export_online_state(),
            shared_trained_bytes=self.shared_trained_bytes,
            external_support_bytes=0,
        )

    def copy_snapshot(self) -> MethodSnapshot:
        contract, _ = self._require_runtime()
        ledger = self.state_ledger()
        token = "ratemem-snapshot-" + ledger.online_state_sha256
        self._snapshot_states[token] = self.state
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self.last_event_index,
            state_sha256=ledger.online_state_sha256,
            online_state_bytes=ledger.online_state_bytes,
            opaque_snapshot_token=token,
        )
```

Snapshot tokens refer only to copied immutable `MemoryState` objects used by evaluation probes. They are cleared on import and close, are never used by lifecycle decisions, and are not online personalization state.

- [ ] **Step 5: Implement the exact causal event and probe signatures**

Add this code to `RateMemAdapter`:

```python
def _decoded_code(self, handle: str, state: MemoryState) -> NDArray[np.float32]:
    base = state.bases[handle]
    rows = []
    for edge in sorted(
        state.incidences.values(),
        key=lambda row: (row.handle, row.packet_id),
    ):
        if edge.handle == handle:
            packet = state.packets[edge.packet_id]
            group, stage, entry = self.codec.dictionary.validate_packet(packet)
            rows.append(HardIncidence(edge, packet, group, stage, entry, 0.0))
    return self.codec.decode(base.payload, tuple(rows))


def apply_event(
    self,
    event: LifecycleEvent,
    view: CausalEventView,
) -> EventReceipt:
    contract, controller = self._require_runtime()
    if len(view) == 0 or view.history()[-1] != event:
        raise ValueError("causal event view does not end at the supplied event")
    before = self.state_ledger().online_state_sha256
    generated: bytes | None = None
    decoded: NDArray[np.float32] | None = None
    if isinstance(event, CreateEvent):
        code = self.predictor.predict(event.support_image_ids, event.description_id)
        decision = controller.apply_create(
            self.state,
            self.proposer.propose(self.state, event.handle, code, event.event_index),
        )
    elif isinstance(event, UpdateEvent):
        code = self.predictor.predict(event.support_image_ids, None)
        decision = controller.apply_update(
            self.state,
            self.proposer.propose(self.state, event.handle, code, event.event_index),
        )
    elif isinstance(event, ReadEvent):
        decision = controller.read(self.state, event.handle, update_usage=True)
        if decision.outcome != "stale_handle":
            decoded = self._decoded_code(event.handle, decision.state)
            generated = self.generation_backend.generate(
                decoded,
                event.prompt_id,
                event.generation_seed,
            )
    elif isinstance(event, DeleteEvent):
        decision = controller.delete(self.state, event.handle)
    elif isinstance(event, ProbeEvent):
        raise ValueError("probe events must use score_probe")
    else:
        raise TypeError(f"unsupported lifecycle event: {type(event).__name__}")
    self.state = decision.state
    self.last_event_index = event.event_index
    ledger = self.state_ledger()
    input_commitment = hashlib.sha256(
        canonical_json_bytes({
            "contract": contract.model_dump(mode="json"),
            "event": event.model_dump(mode="json"),
        })
    ).hexdigest()
    return EventReceipt(
        method_id=self.method_id,
        trace_id=contract.trace_id,
        event_index=event.event_index,
        event_kind=event.kind,
        input_commitment_sha256=input_commitment,
        method_state_sha256_before=before,
        method_state_sha256_after=ledger.online_state_sha256,
        candidate_stream_sha256=contract.candidate_stream_sha256,
        outcome=decision.outcome,
        affected_handles=tuple(sorted({event.handle, *decision.evicted_handles})),
        evicted_handles=tuple(decision.evicted_handles),
        decoded_code_sha256=(
            hashlib.sha256(np.asarray(decoded, dtype="<f4").tobytes()).hexdigest()
            if decoded is not None else None
        ),
        generated_sample_sha256=(
            hashlib.sha256(generated).hexdigest() if generated is not None else None
        ),
        ledger=ledger,
    )


def score_probe(
    self,
    snapshot: MethodSnapshot,
    probe: ProbeEvent,
) -> ProbeResult:
    contract, _ = self._require_runtime()
    if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
        raise ValueError("snapshot identity does not match RateMem runtime")
    state = self._snapshot_states[snapshot.opaque_snapshot_token]
    before = self.state_ledger()
    decoded = self._decoded_code(probe.handle, state)
    generated = self.generation_backend.generate(
        decoded,
        probe.prompt_id,
        probe.generation_seed,
    )
    after = self.state_ledger()
    if before != after:
        raise RuntimeError("probe mutated online RateMem state")
    return ProbeResult(
        method_id=self.method_id,
        trace_id=contract.trace_id,
        probe_event_index=probe.event_index,
        snapshot_state_sha256=snapshot.state_sha256,
        input_commitment_sha256=hashlib.sha256(
            canonical_json_bytes(probe.model_dump(mode="json"))
        ).hexdigest(),
        generated_sample_sha256=hashlib.sha256(generated).hexdigest(),
        update_usage=False,
    )


def close(self) -> None:
    self.state = MemoryState()
    self.last_event_index = 0
    self._snapshot_states.clear()
    self.controller = None
    self.contract = None
```

The method must not inspect `view` beyond `view.history()` and the current event when constructing request weights. Any access after the current index is blocked by the canonical `CausalEventView`.

- [ ] **Step 6: Add an integration and mypy proof against the canonical protocol**

```python
# tests/integration/method/test_canonical_baseline_adapter.py
from ratemem.baselines.protocol import BaselineAdapter, CausalEventView
from ratemem.method.adapter import RateMemAdapter


def execute_as_canonical(
    adapter: BaselineAdapter,
    contract,
    events,
) -> tuple[str, bytes]:
    adapter.initialize(contract)
    for index, event in enumerate(events):
        adapter.apply_event(event, CausalEventView(events, current_index=index))
    snapshot = adapter.copy_snapshot()
    payload = adapter.export_online_state()
    assert adapter.state_ledger().online_state_sha256 == snapshot.state_sha256
    return snapshot.opaque_snapshot_token, payload


def test_concrete_ratemem_instantiates_and_roundtrips_as_canonical_adapter(
    adapter_factory,
    contract,
    lifecycle_events,
) -> None:
    adapter: RateMemAdapter = adapter_factory()
    token, payload = execute_as_canonical(adapter, contract, lifecycle_events)
    assert token.startswith("ratemem-snapshot-")
    restored: RateMemAdapter = adapter_factory()
    restored.initialize(contract)
    restored.import_online_state(payload)
    assert restored.export_online_state() == payload
```

Run:

```bash
uv run pytest tests/unit/method/test_adapter.py tests/contract/method/test_adapter_protocol.py tests/integration/method/test_canonical_baseline_adapter.py tests/baselines/test_protocol.py tests/baselines/test_ledger.py -q
uv run ruff check src/ratemem/method/adapter.py tests/unit/method/test_adapter.py tests/contract/method/test_adapter_protocol.py tests/integration/method/test_canonical_baseline_adapter.py
uv run mypy src/ratemem/method/adapter.py tests/integration/method/test_canonical_baseline_adapter.py
```

Expected: all tests pass; mypy accepts a concrete `RateMemAdapter` wherever the sole canonical `BaselineAdapter` is required; export/import preserves the exact canonical bytes and next receipt.

- [ ] **Step 7: Commit the canonical adapter only**

```bash
git add src/ratemem/method/adapter.py tests/unit/method/test_adapter.py tests/contract/method/test_adapter_protocol.py tests/integration/method/test_canonical_baseline_adapter.py
git commit -m "feat(method): implement canonical RateMem adapter"
```


### Task 9: Build bounded training segments from frozen visible lifecycle traces

**Files:**
- Create: `src/ratemem/training/__init__.py`
- Create: `src/ratemem/training/segments.py`
- Create: `src/ratemem/training/functional_state.py`
- Test: `tests/unit/training/test_segments.py`
- Test: `tests/unit/training/test_functional_state.py`
- Test: `tests/contract/training/test_visible_trace_only.py`

- [ ] **Step 1: Write trace-hash, two-event, and final-payload rejection tests**

```python
# tests/unit/training/test_segments.py
from pathlib import Path

import pytest

from ratemem.training.segments import SegmentPolicy, load_visible_trace, segment_trace


def test_segment_builder_is_deterministic_and_caps_events_and_queries(visible_trace) -> None:
    policy = SegmentPolicy(length=2, maximum_queries=2)
    first = segment_trace(visible_trace, policy)
    second = segment_trace(visible_trace, policy)
    assert first == second
    assert all(len(segment.events) <= 2 for segment in first)
    assert all(sum(event.has_training_query for event in segment.events) <= 2 for segment in first)


@pytest.mark.parametrize("split", ["final_test", "test"])
def test_training_loader_rejects_nonvisible_split(tmp_path: Path, split: str) -> None:
    manifest = tmp_path / f"{split}-manifest.json"
    manifest.write_text('{"split":"' + split + '"}', encoding="utf-8")
    with pytest.raises(ValueError, match="train or validation"):
        load_visible_trace(manifest, expected_manifest_sha256="0" * 64)
```

```python
# tests/unit/training/test_functional_state.py
import torch

from ratemem.training.functional_state import FunctionalMemoryState


def test_updates_are_out_of_place_and_boundary_detach_cuts_history() -> None:
    code = torch.randn(4, requires_grad=True)
    empty = FunctionalMemoryState()
    updated = empty.upsert("a", code, event_index=1)
    detached = updated.detach_boundary()
    assert "a" not in empty.codes
    assert updated.codes["a"] is code
    assert detached.codes["a"].grad_fn is None
    assert not detached.codes["a"].requires_grad
```

- [ ] **Step 2: Run segment tests and verify the training package is absent**

Run: `uv run pytest tests/unit/training/test_segments.py tests/unit/training/test_functional_state.py -q`

Expected: collection fails because `ratemem.training` does not exist.

- [ ] **Step 3: Implement immutable functional memory with explicit detach**

```python
# src/ratemem/training/functional_state.py
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True, slots=True)
class FunctionalMemoryState:
    codes: Mapping[str, Tensor] = field(default_factory=dict)
    last_event: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "codes", MappingProxyType(dict(self.codes)))
        object.__setattr__(self, "last_event", MappingProxyType(dict(self.last_event)))

    def upsert(self, handle: str, code: Tensor, event_index: int) -> "FunctionalMemoryState":
        codes, events = dict(self.codes), dict(self.last_event)
        codes[handle], events[handle] = code, event_index
        return FunctionalMemoryState(codes, events)

    def delete(self, handle: str) -> "FunctionalMemoryState":
        codes, events = dict(self.codes), dict(self.last_event)
        codes.pop(handle, None)
        events.pop(handle, None)
        return FunctionalMemoryState(codes, events)

    def detach_boundary(self) -> "FunctionalMemoryState":
        return FunctionalMemoryState(
            {handle: code.detach() for handle, code in self.codes.items()},
            self.last_event,
        )
```

- [ ] **Step 4: Implement hash-checked visible-trace loading and deterministic segmentation**

```python
# src/ratemem/training/segments.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class SegmentPolicy:
    length: Literal[2]
    maximum_queries: Literal[2]


@dataclass(frozen=True, slots=True)
class FrozenTrainingEvent:
    event_index: int
    kind: Literal["create", "update", "read", "delete"]
    handle: str
    support_image_ids: Sequence[str] = ()
    description_id: str | None = None
    prompt_id: str | None = None
    generation_seed: int | None = None
    has_training_query: bool = False


@dataclass(frozen=True, slots=True)
class FrozenVisibleTrace:
    trace_id: str
    split: Literal["train", "validation"]
    manifest_sha256: str
    payload_sha256: str
    events: Sequence[FrozenTrainingEvent]


@dataclass(frozen=True, slots=True)
class TrainingSegment:
    trace_id: str
    segment_index: int
    events: Sequence[FrozenTrainingEvent]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_visible_trace(manifest_path: Path, expected_manifest_sha256: str) -> FrozenVisibleTrace:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("split") not in {"train", "validation"}:
        raise ValueError("training accepts only train or validation traces")
    if "final" in manifest_path.name.lower() or _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("visible trace manifest path or hash is not approved")
    payload_path = Path(raw["payload_path"])
    if _sha256(payload_path) != raw["payload_sha256"]:
        raise ValueError("visible trace payload hash mismatch")
    rows = [json.loads(line) for line in payload_path.read_text(encoding="utf-8").splitlines()]
    events = tuple(
        FrozenTrainingEvent(
            event_index=row["event_index"], kind=row["kind"], handle=row["handle"],
            support_image_ids=tuple(row.get("support_image_ids", ())),
            description_id=row.get("description_id"), prompt_id=row.get("prompt_id"),
            generation_seed=row.get("generation_seed"),
            has_training_query=row["kind"] in {"read", "update"},
        )
        for row in rows
        if row["kind"] != "probe"
    )
    return FrozenVisibleTrace(
        trace_id=raw["trace_id"], split=raw["split"],
        manifest_sha256=expected_manifest_sha256,
        payload_sha256=raw["payload_sha256"], events=events,
    )


def segment_trace(trace: FrozenVisibleTrace, policy: SegmentPolicy) -> Sequence[TrainingSegment]:
    output = []
    cursor = 0
    while cursor < len(trace.events):
        rows = []
        query_count = 0
        while cursor < len(trace.events) and len(rows) < policy.length:
            event = trace.events[cursor]
            if event.has_training_query and query_count == policy.maximum_queries:
                break
            rows.append(event)
            query_count += int(event.has_training_query)
            cursor += 1
        if not rows:
            raise RuntimeError("segment policy made no progress")
        output.append(TrainingSegment(trace.trace_id, len(output), tuple(rows)))
    return tuple(output)
```

- [ ] **Step 5: Add an AST contract that blocks final-trace access**

```python
# tests/contract/training/test_visible_trace_only.py
import ast
from pathlib import Path


def test_training_tree_has_no_final_trace_import_or_final_evaluation_literal() -> None:
    for path in Path("src/ratemem/training").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ratemem.evaluation.final_trace"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "final-test-envelope" not in node.value
                assert node.value != "final_evaluation"
```

- [ ] **Step 6: Run training-state contracts and commit**

Run:

```bash
uv run pytest tests/unit/training/test_segments.py tests/unit/training/test_functional_state.py tests/contract/training/test_visible_trace_only.py -q
uv run ruff check src/ratemem/training tests/unit/training tests/contract/training
uv run mypy src/ratemem/training
```

Expected: all focused tests pass; no final-test access appears under `src/ratemem/training`.

```bash
git add src/ratemem/training tests/unit/training tests/contract/training/test_visible_trace_only.py
git commit -m "feat(train): build bounded visible-trace segments"
```

### Task 10: Meta-train the codec and utility on sequential segments with at most two passes

**Files:**
- Create: `src/ratemem/training/losses.py`
- Create: `src/ratemem/training/meta_trainer.py`
- Test: `tests/unit/training/test_losses.py`
- Test: `tests/unit/training/test_meta_trainer.py`
- Test: `tests/contract/training/test_compute_bound.py`
- Test: `tests/contract/training/test_gradient_boundary.py`

- [ ] **Step 1: Write exact loss and anti-collapse tests**

```python
# tests/unit/training/test_losses.py
import torch

from ratemem.training.losses import (
    dictionary_balance_loss,
    expected_rate_loss,
    reuse_affinity_loss,
)


def test_reuse_affinity_rewards_matching_assignments_for_similar_residuals() -> None:
    probabilities = torch.tensor([
        [[[[0.9, 0.1]]]],
        [[[[0.9, 0.1]]]],
        [[[[0.1, 0.9]]]],
    ])
    residuals = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    aligned = reuse_affinity_loss(probabilities, residuals, similarity_center=0.8, similarity_width=0.1)
    permuted = reuse_affinity_loss(probabilities[[0, 2, 1]], residuals, 0.8, 0.1)
    assert aligned < permuted


def test_balance_loss_penalizes_single_entry_collapse() -> None:
    uniform = torch.full((4, 2, 1, 4), 0.25)
    collapsed = torch.zeros_like(uniform)
    collapsed[:, :, :, 0] = 1.0
    assert dictionary_balance_loss(uniform) < dictionary_balance_loss(collapsed)


def test_rate_loss_charges_exactly_eight_selected_candidate_costs() -> None:
    selected = torch.zeros(1, 30, 2)
    selected.reshape(-1)[:8] = 1.0
    costs = torch.arange(1, 61, dtype=torch.float32).reshape(30, 2)
    actual = expected_rate_loss(selected, costs, budget_bytes=1000)
    assert actual == torch.tensor(sum(range(1, 9)) / 1000)
```

- [ ] **Step 2: Run loss tests and verify the modules are absent**

Run: `uv run pytest tests/unit/training/test_losses.py tests/unit/training/test_meta_trainer.py -q`

Expected: collection fails because `ratemem.training.losses` and `ratemem.training.meta_trainer` do not exist.

- [ ] **Step 3: Implement the declared objective terms without hidden image sampling**

```python
# src/ratemem/training/losses.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


def reconstruction_loss(target: Tensor, reconstructed: Tensor) -> Tensor:
    return F.mse_loss(reconstructed.float(), target.float())


def expected_rate_loss(
    selected_mask: Tensor,
    candidate_cost_bytes: Tensor,
    budget_bytes: int,
) -> Tensor:
    if (
        budget_bytes <= 0
        or selected_mask.ndim != 3
        or candidate_cost_bytes.shape != selected_mask.shape[1:]
    ):
        raise ValueError("rate tensors or byte budget have the wrong shape")
    expected = (
        selected_mask.float()
        * candidate_cost_bytes.to(selected_mask).unsqueeze(0)
    ).sum()
    return expected / (selected_mask.shape[0] * budget_bytes)


def reuse_affinity_loss(
    probabilities: Tensor,
    residuals: Tensor,
    similarity_center: float,
    similarity_width: float,
) -> Tensor:
    if residuals.shape[0] < 2:
        return probabilities.sum() * 0.0
    normalized = F.normalize(residuals.float().reshape(residuals.shape[0], -1), dim=-1, eps=1e-8)
    target = torch.sigmoid((normalized @ normalized.T - similarity_center) / similarity_width)
    flat = probabilities.float().reshape(probabilities.shape[0], -1, probabilities.shape[-1])
    match = torch.einsum("bke,cke->bck", flat, flat).mean(dim=-1).clamp(1e-6, 1 - 1e-6)
    mask = ~torch.eye(match.shape[0], dtype=torch.bool, device=match.device)
    return F.binary_cross_entropy(match[mask], target[mask])


def dictionary_balance_loss(probabilities: Tensor) -> Tensor:
    usage = probabilities.float().mean(dim=tuple(range(probabilities.ndim - 1))).clamp_min(1e-8)
    uniform = torch.full_like(usage, 1.0 / usage.numel())
    return torch.sum(usage * (usage.log() - uniform.log()))


def dictionary_commitment_loss(residual: Tensor, packet_reconstruction: Tensor) -> Tensor:
    encoder_term = F.mse_loss(residual.float(), packet_reconstruction.detach().float())
    dictionary_term = F.mse_loss(residual.detach().float(), packet_reconstruction.float())
    return encoder_term + dictionary_term


def nonnegative_calibration_loss(predicted: Tensor, observed: Tensor, mask: Tensor) -> Tensor:
    if torch.any(predicted < 0) or torch.any(observed < 0):
        raise ValueError("utility calibration values must be nonnegative")
    selected = mask.to(torch.bool)
    return F.mse_loss(predicted[selected].float(), observed[selected].float())


@dataclass(frozen=True)
class LossWeights:
    flow: float
    reconstruction: float
    rate: float
    reuse_affinity: float
    dictionary_balance: float
    dictionary_commitment: float
    utility_calibration: float


def combine_losses(terms: dict[str, Tensor], weights: LossWeights) -> Tensor:
    if set(terms) != set(weights.__dataclass_fields__):
        raise ValueError("loss terms do not match the locked method objective")
    return sum(terms[name] * getattr(weights, name) for name in sorted(terms))
```

- [ ] **Step 4: Implement one-step query injection and sequential functional updates**

```python
# src/ratemem/training/meta_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ratemem.method.codec import RateMemDifferentiableCodec
from ratemem.method.utility import CausalFeatureBatch, NonnegativeUtilityCalibrator
from ratemem.training.functional_state import FunctionalMemoryState
from ratemem.training.losses import LossWeights, combine_losses
from ratemem.training.segments import TrainingSegment


class PreparedSegmentResolver(Protocol):
    def target_code(self, trace_id: str, event_index: int) -> Tensor:
        raise NotImplementedError

    def one_timestep_flow_loss(self, trace_id: str, event_index: int, adapter_code: Tensor) -> Tensor:
        raise NotImplementedError

    def utility_supervision(
        self, trace_id: str, event_index: int
    ) -> tuple[CausalFeatureBatch, Tensor, Tensor]:
        raise NotImplementedError


@dataclass(frozen=True)
class MetaStepReceipt:
    trace_id: str
    segment_index: int
    transformer_passes: int
    event_count: int
    total_loss: float
    detached_state: FunctionalMemoryState


class SequentialMetaTrainer:
    def __init__(
        self,
        codec: RateMemDifferentiableCodec,
        utility: NonnegativeUtilityCalibrator,
        optimizer: torch.optim.Optimizer,
        resolver: PreparedSegmentResolver,
        weights: LossWeights,
        maximum_transformer_passes: int = 2,
    ) -> None:
        self.codec = codec
        self.utility = utility
        self.optimizer = optimizer
        self.resolver = resolver
        self.weights = weights
        self.maximum_transformer_passes = maximum_transformer_passes

    def train_segment(
        self,
        segment: TrainingSegment,
        state: FunctionalMemoryState,
        *,
        temperature: float,
        candidate_cost_bytes: Tensor,
        budget_bytes: int,
    ) -> MetaStepReceipt:
        self.optimizer.zero_grad(set_to_none=True)
        terms: dict[str, Tensor] = {}
        transformer_passes = 0
        current = state
        encodings = []
        targets = []
        for event in segment.events:
            if event.kind in {"create", "update"}:
                target = self.resolver.target_code(segment.trace_id, event.event_index).float()
                encoded = self.codec(target, temperature=temperature, mode="ste")
                current = current.upsert(event.handle, encoded.reconstruction, event.event_index)
                encodings.append(encoded)
                targets.append(target)
            elif event.kind == "delete":
                current = current.delete(event.handle)
            if event.has_training_query:
                if event.handle not in current.codes:
                    continue
                transformer_passes += 1
                if transformer_passes > self.maximum_transformer_passes:
                    raise RuntimeError("segment exceeded the locked transformer-pass cap")
                flow = self.resolver.one_timestep_flow_loss(
                    segment.trace_id, event.event_index, current.codes[event.handle]
                )
                terms["flow"] = terms.get("flow", flow * 0.0) + flow
        if not encodings:
            zero = next(self.codec.parameters()).sum() * 0.0
            terms.update({name: terms.get(name, zero) for name in self.weights.__dataclass_fields__})
        else:
            from ratemem.training.losses import (
                dictionary_balance_loss, dictionary_commitment_loss,
                expected_rate_loss, nonnegative_calibration_loss,
                reconstruction_loss, reuse_affinity_loss,
            )
            target_batch = torch.cat(targets, dim=0)
            reconstruction_batch = torch.cat([row.reconstruction for row in encodings], dim=0)
            probabilities = torch.cat([row.assignment_probabilities for row in encodings], dim=0)
            base_batch = torch.cat([row.base_reconstruction for row in encodings], dim=0)
            residuals = target_batch - base_batch
            packet_reconstruction = reconstruction_batch - base_batch
            utility_features, observed, mask = self.resolver.utility_supervision(
                segment.trace_id, segment.events[-1].event_index
            )
            predicted = self.utility(utility_features).value
            terms.update({
                "reconstruction": reconstruction_loss(target_batch, reconstruction_batch),
                "rate": expected_rate_loss(
                    torch.cat([row.selected_mask for row in encodings], dim=0),
                    candidate_cost_bytes,
                    budget_bytes,
                ),
                "reuse_affinity": reuse_affinity_loss(probabilities, residuals, 0.8, 0.1),
                "dictionary_balance": dictionary_balance_loss(probabilities),
                "dictionary_commitment": dictionary_commitment_loss(residuals, packet_reconstruction),
                "utility_calibration": nonnegative_calibration_loss(predicted, observed, mask),
            })
            zero = reconstruction_batch.sum() * 0.0
            terms["flow"] = terms.get("flow", zero)
        total = combine_losses(terms, self.weights)
        total.backward()
        self.optimizer.step()
        self.codec.dictionary.normalize_codebooks_()
        detached = current.detach_boundary()
        return MetaStepReceipt(
            trace_id=segment.trace_id, segment_index=segment.segment_index,
            transformer_passes=transformer_passes, event_count=len(segment.events),
            total_loss=float(total.detach()), detached_state=detached,
        )
```

The production `PreparedSegmentResolver` wraps the already pinned SANA amortizer and flow helpers. It uses precomputed frozen support features, text embeddings, and VAE latents; each `one_timestep_flow_loss` samples one flow timestep and calls the frozen transformer exactly once under BF16 autocast and activation checkpointing. It never calls a denoising pipeline or retains a sampled image graph.

- [ ] **Step 5: Contract-test the pass cap, frozen backbone, and boundary detach**

```python
# tests/contract/training/test_compute_bound.py
def test_two_query_segment_uses_exactly_two_transformer_passes(meta_trainer, two_query_segment, state, costs) -> None:
    receipt = meta_trainer.train_segment(
        two_query_segment, state, temperature=0.5,
        candidate_cost_bytes=costs, budget_bytes=2048,
    )
    assert receipt.transformer_passes == 2
    assert meta_trainer.resolver.transformer_calls == 2
    assert meta_trainer.resolver.full_denoising_calls == 0


def test_third_query_is_rejected_before_transformer_call(meta_trainer, three_query_segment, state, costs) -> None:
    before = meta_trainer.resolver.transformer_calls
    try:
        meta_trainer.train_segment(
            three_query_segment, state, temperature=0.5,
            candidate_cost_bytes=costs, budget_bytes=2048,
        )
    except RuntimeError as exc:
        assert "pass cap" in str(exc)
    else:
        raise AssertionError("three-query segment was accepted")
    assert meta_trainer.resolver.transformer_calls - before == 2
```

```python
# tests/contract/training/test_gradient_boundary.py
def test_gradients_reach_amortizer_dictionary_utility_and_atoms_but_not_backbone(real_tiny_trainer, segment) -> None:
    receipt = real_tiny_trainer.train_segment(segment)
    assert receipt.detached_state.codes
    assert all(not code.requires_grad and code.grad_fn is None for code in receipt.detached_state.codes.values())
    assert all(parameter.grad is None for parameter in real_tiny_trainer.backbone.parameters())
    assert any(parameter.grad is not None for parameter in real_tiny_trainer.amortizer.parameters())
    assert any(parameter.grad is not None for parameter in real_tiny_trainer.codec.dictionary.parameters())
    assert any(parameter.grad is not None for parameter in real_tiny_trainer.utility.parameters())
    assert any(parameter.grad is not None for parameter in real_tiny_trainer.adapter_bank.parameters())
```

- [ ] **Step 6: Run trainer tests, static checks, and commit**

Run:

```bash
uv run pytest tests/unit/training/test_losses.py tests/unit/training/test_meta_trainer.py tests/contract/training/test_compute_bound.py tests/contract/training/test_gradient_boundary.py -q
uv run ruff check src/ratemem/training tests/unit/training tests/contract/training
uv run mypy src/ratemem/training
```

Expected: all focused tests pass; a valid two-query segment records two transformer calls, and a third query is blocked before an additional call.

```bash
git add src/ratemem/training/losses.py src/ratemem/training/meta_trainer.py tests/unit/training/test_losses.py tests/unit/training/test_meta_trainer.py tests/contract/training/test_compute_bound.py tests/contract/training/test_gradient_boundary.py
git commit -m "feat(train): add bounded sequential meta training"
```

### Task 11: Save a trainable-only method checkpoint with complete provenance

**Files:**
- Create: `src/ratemem/method/checkpoint.py`
- Create: `schemas/ratemem-method-checkpoint-v1.schema.json`
- Test: `tests/unit/method/test_checkpoint.py`
- Test: `tests/contract/method/test_checkpoint_provenance.py`

- [ ] **Step 1: Write exclusion, round-trip, and provenance-binding tests**

```python
# tests/unit/method/test_checkpoint.py
from pathlib import Path

import torch

from ratemem.method.checkpoint import load_method_checkpoint, save_method_checkpoint


def test_checkpoint_round_trip_excludes_backbone_optimizer_and_online_state(tmp_path: Path, trainable_method) -> None:
    tensor_path = tmp_path / "ratemem.safetensors"
    manifest_path = tmp_path / "ratemem.manifest.json"
    expected = trainable_method.fixed_prediction().detach().clone()
    save_method_checkpoint(tensor_path, manifest_path, trainable_method, trainable_method.provenance)
    keys = trainable_method.checkpoint_keys(tensor_path)
    assert all(not key.startswith(("backbone.", "optimizer.", "online_state.")) for key in keys)
    trainable_method.randomize_trainable_state()
    load_method_checkpoint(tensor_path, manifest_path, trainable_method, trainable_method.provenance)
    torch.testing.assert_close(trainable_method.fixed_prediction(), expected)
```

```python
# tests/contract/method/test_checkpoint_provenance.py
from dataclasses import replace

import pytest

from ratemem.method.checkpoint import ProvenanceMismatch, load_method_checkpoint


def test_changed_dataset_trace_or_model_revision_blocks_load(checkpoint_files, method, provenance) -> None:
    for field, value in (
        ("dataset_lock_sha256", "0" * 64),
        ("visible_trace_set_sha256", "1" * 64),
        ("backbone_revision", "2" * 40),
    ):
        with pytest.raises(ProvenanceMismatch):
            load_method_checkpoint(
                *checkpoint_files, method, replace(provenance, **{field: value})
            )
```

- [ ] **Step 2: Run checkpoint tests and verify the module is absent**

Run: `uv run pytest tests/unit/method/test_checkpoint.py tests/contract/method/test_checkpoint_provenance.py -q`

Expected: collection fails because `ratemem.method.checkpoint` does not exist.

- [ ] **Step 3: Define the strict deployment and training provenance**

```python
# src/ratemem/method/checkpoint.py
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from ratemem.evaluation.canonical import canonical_json_bytes


class ProvenanceMismatch(ValueError):
    """Raised when checkpoint bytes do not match approved training provenance."""


@dataclass(frozen=True, slots=True)
class MethodProvenance:
    git_commit: str
    git_diff_sha256: str
    backbone_model_id: str
    backbone_revision: str
    support_encoder_revision: str
    method_lock_sha256: str
    dataset_lock_sha256: str
    evaluation_lock_sha256: str
    baseline_lock_sha256: str
    visible_trace_set_sha256: str
    training_seed: int
    torch_version: str
    diffusers_version: str


class MethodCheckpointManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    tensor_file: str
    tensor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tensor_keys: list[str]
    dictionary_revision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backbone_model_id: str
    backbone_revision: str
    support_encoder_revision: str
    method_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visible_trace_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_seed: int
    torch_version: str
    diffusers_version: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect(module: nn.Module) -> dict[str, Tensor]:
    allowed = ("adapter_bank.", "amortizer.", "dictionary.", "utility.")
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in module.state_dict().items()
        if key.startswith(allowed)
    }
    forbidden = ("backbone.", "optimizer.", "online_state.")
    if not tensors or any(key.startswith(forbidden) for key in tensors):
        raise ValueError("method checkpoint contains a forbidden or empty tensor set")
    return dict(sorted(tensors.items()))
```

- [ ] **Step 4: Implement atomic save/load and dictionary revision verification**

Add these functions to `src/ratemem/method/checkpoint.py`:

```python
def save_method_checkpoint(
    tensor_path: Path,
    manifest_path: Path,
    method: nn.Module,
    provenance: MethodProvenance,
) -> MethodCheckpointManifest:
    tensors = _collect(method)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    save_file(tensors, temporary, metadata={"format": "ratemem-method-v1"})
    temporary.replace(tensor_path)
    dictionary_revision = method.frozen_dictionary_revision()
    manifest = MethodCheckpointManifest(
        tensor_file=tensor_path.name, tensor_sha256=_sha256(tensor_path),
        tensor_keys=list(tensors), dictionary_revision_sha256=dictionary_revision,
        **asdict(provenance),
    )
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    manifest_tmp.replace(manifest_path)
    return manifest


def load_method_checkpoint(
    tensor_path: Path,
    manifest_path: Path,
    method: nn.Module,
    expected: MethodProvenance,
) -> MethodCheckpointManifest:
    manifest = MethodCheckpointManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.tensor_sha256 != _sha256(tensor_path):
        raise ProvenanceMismatch("checkpoint tensor hash mismatch")
    expected_fields = asdict(expected)
    for field, value in expected_fields.items():
        if getattr(manifest, field) != value:
            raise ProvenanceMismatch(f"checkpoint provenance mismatch: {field}")
    with safe_open(tensor_path, framework="pt", device="cpu") as stream:
        if stream.metadata().get("format") != "ratemem-method-v1":
            raise ProvenanceMismatch("checkpoint format mismatch")
        keys = tuple(sorted(stream.keys()))
    if list(keys) != manifest.tensor_keys or keys != tuple(_collect(method)):
        raise ProvenanceMismatch("checkpoint tensor key set mismatch")
    tensors = load_file(tensor_path, device="cpu")
    current = method.state_dict()
    current.update(tensors)
    method.load_state_dict(current, strict=True)
    if method.frozen_dictionary_revision() != manifest.dictionary_revision_sha256:
        raise ProvenanceMismatch("frozen dictionary revision mismatch")
    return manifest
```

- [ ] **Step 5: Generate the schema, inspect a synthetic checkpoint, and commit**

Extend `ratemem-method` with `checkpoint-schema` and `checkpoint-inspect`. The first writes canonical `MethodCheckpointManifest.model_json_schema()`. The second validates the manifest with `jsonschema==4.25.1`, verifies the tensor hash/key list without loading arbitrary pickle data, and prints only revisions, hashes, and tensor shapes.

Run:

```bash
uv run ratemem-method checkpoint-schema --output schemas/ratemem-method-checkpoint-v1.schema.json
uv run pytest tests/unit/method/test_checkpoint.py tests/contract/method/test_checkpoint_provenance.py -q
uv run ruff check src/ratemem/method/checkpoint.py tests/unit/method/test_checkpoint.py tests/contract/method/test_checkpoint_provenance.py
uv run mypy src/ratemem/method/checkpoint.py
```

Expected: all focused tests pass; the schema is valid Draft 2020-12 and byte-matches the generated model schema.

```bash
git add src/ratemem/method/checkpoint.py src/ratemem/method/cli.py schemas/ratemem-method-checkpoint-v1.schema.json tests/unit/method/test_checkpoint.py tests/contract/method/test_checkpoint_provenance.py
git commit -m "feat(method): checkpoint learned RateMem state"
```

### Task 12: Own the one-shot authorized scientific-training producer

**Depends on:** Tasks 1--11 for implementation and fake-provider tests. The last step is deferred until Task 14 passes and scientific Task 9 has issued a distinct authorization/reservation for exactly one `meta_train_seed_{17,29,43}` phase.

**Files:**
- Create: `src/ratemem/training/authorized.py`
- Create: `src/ratemem/training/modal_app.py`
- Create: `src/ratemem/method/phase_receipts.py`
- Modify: `src/ratemem/method/cli.py`
- Create: `schemas/ratemem-training-request-v1.schema.json`
- Create: `schemas/ratemem-method-phase-attempt-v1.schema.json`
- Create: `schemas/ratemem-method-phase-final-v1.schema.json`
- Test: `tests/unit/training/test_authorized_training.py`
- Test: `tests/contract/training/test_training_producer_guard.py`
- Test: `tests/contract/training/test_scientific_modal_app.py`
- Test: `tests/contract/method/test_phase_receipts.py`

- [ ] **Step 1: Write the failing one-use, ordering, output, and reconciliation tests**

```python
# tests/unit/training/test_authorized_training.py
from ratemem.method.checkpoint import load_method_checkpoint
from ratemem.method.phase_receipts import MethodPhaseAttemptReceipt
from ratemem.training.authorized import run_authorized_training


def test_permit_is_consumed_immediately_before_one_launch_and_checkpoint_is_frozen(
    authorized_training_fixture,
) -> None:
    calls: list[str] = []
    guard = authorized_training_fixture.guard(record_into=calls)
    launcher = authorized_training_fixture.launcher(record_into=calls)
    receipt = run_authorized_training(
        request=authorized_training_fixture.request(training_seed=17),
        permit_paths=authorized_training_fixture.permit_paths(),
        checkpoint_output=authorized_training_fixture.checkpoint_path,
        checkpoint_manifest_output=authorized_training_fixture.manifest_path,
        receipt_output=authorized_training_fixture.attempt_path,
        permit_guard=guard,
        launcher=launcher,
    )
    assert calls == ["consume-permit", "launch-once"]
    assert launcher.calls == 1
    assert receipt.status == "success"
    assert receipt.training_seed == 17
    assert receipt.consumed_permit_sha256 == guard.consumed_sha256
    assert receipt.launch_receipt_sha256 == guard.launch_receipt_sha256
    assert receipt.output_artifact_sha256 == {
        "checkpoint": authorized_training_fixture.checkpoint_sha256,
        "checkpoint_manifest": authorized_training_fixture.manifest_sha256,
    }
    load_method_checkpoint(
        authorized_training_fixture.checkpoint_path,
        authorized_training_fixture.manifest_path,
        authorized_training_fixture.fresh_method(),
        authorized_training_fixture.provenance,
    )
    assert MethodPhaseAttemptReceipt.model_validate_json(
        authorized_training_fixture.attempt_path.read_text()
    ) == receipt


def test_missing_stale_wrong_phase_or_reused_permit_never_calls_launcher(
    authorized_training_fixture,
) -> None:
    for permit_paths in authorized_training_fixture.invalid_permit_paths(
        cases=("missing", "stale_cpu_gate", "wrong_phase", "wrong_workspace", "already_consumed")
    ):
        launcher = authorized_training_fixture.launcher()
        with authorized_training_fixture.denied():
            run_authorized_training(
                request=authorized_training_fixture.request(training_seed=17),
                permit_paths=permit_paths,
                checkpoint_output=authorized_training_fixture.checkpoint_path,
                checkpoint_manifest_output=authorized_training_fixture.manifest_path,
                receipt_output=authorized_training_fixture.attempt_path,
                launcher=launcher,
            )
        assert launcher.calls == 0


def test_seed_and_trace_must_be_locked_before_permit_consumption(
    authorized_training_fixture,
) -> None:
    for request in (
        authorized_training_fixture.request(training_seed=0),
        authorized_training_fixture.request(trace_manifest_sha256=("0" * 64,)),
    ):
        guard = authorized_training_fixture.guard()
        with authorized_training_fixture.denied():
            run_authorized_training(
                request=request,
                permit_paths=authorized_training_fixture.permit_paths(),
                checkpoint_output=authorized_training_fixture.checkpoint_path,
                checkpoint_manifest_output=authorized_training_fixture.manifest_path,
                receipt_output=authorized_training_fixture.attempt_path,
                permit_guard=guard,
                launcher=authorized_training_fixture.launcher(),
            )
        assert guard.calls == 0
```

```python
# tests/contract/method/test_phase_receipts.py
import pytest

from ratemem.method.phase_receipts import finalize_method_phase


def test_final_receipt_requires_matching_reconciliation_and_outputs(phase_fixture) -> None:
    final = finalize_method_phase(
        kind="training",
        attempt_path=phase_fixture.success_attempt,
        authorization_path=phase_fixture.authorization,
        reservation_path=phase_fixture.reservation,
        reconciliation_path=phase_fixture.reconciliation,
        output_path=phase_fixture.final_receipt,
    )
    assert final.provider_call_ids == phase_fixture.reconciled_provider_call_ids
    assert final.output_artifact_sha256 == phase_fixture.output_hashes
    for mutation in (
        phase_fixture.wrong_authorization,
        phase_fixture.wrong_reservation,
        phase_fixture.wrong_provider_ids,
        phase_fixture.tampered_output,
    ):
        with pytest.raises(ValueError):
            mutation.finalize()
```

- [ ] **Step 2: Run the focused tests and verify the producers are absent**

Run:

```bash
uv run pytest tests/unit/training/test_authorized_training.py tests/contract/training/test_training_producer_guard.py tests/contract/method/test_phase_receipts.py -q
```

Expected: collection fails importing `ratemem.training.authorized` or `ratemem.method.phase_receipts`. No provider command runs.

- [ ] **Step 3: Define strict request, attempt, and final receipt contracts**

```python
# src/ratemem/method/phase_receipts.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from ratemem.evaluation.canonical import Sha256

PhaseKind = Literal["training", "materialization"]


class AuthorizedTrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    phase_id: str = Field(pattern=r"^meta_train_seed_(17|29|43)$")
    workspace_id: str = Field(min_length=1)
    split: Literal["train"]
    training_seed: Literal[17, 29, 43]
    trace_manifest_sha256: tuple[Sha256, ...]
    dataset_lock_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_lock_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_lock_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    method_lock_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    method_cpu_gate_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    segment_length: Literal[2]
    maximum_transformer_passes_per_segment: Literal[2]
    maximum_segments: PositiveInt
    maximum_optimizer_steps: PositiveInt

    @model_validator(mode="after")
    def phase_matches_seed(self) -> "AuthorizedTrainingRequest":
        if self.phase_id != f"meta_train_seed_{self.training_seed}":
            raise ValueError("training phase id does not match its locked seed")
        if not self.trace_manifest_sha256:
            raise ValueError("training request needs at least one frozen trace manifest")
        return self


class MethodPhaseAttemptReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    kind: PhaseKind
    phase_id: str
    workspace_id: str
    training_seed: Literal[17, 29, 43]
    status: Literal["success", "failed"]
    request_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    reservation_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    consumed_permit_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    launch_receipt_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_lock_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_lock_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_lock_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    method_lock_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    method_cpu_gate_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    input_artifact_sha256: dict[str, Sha256]
    output_artifact_sha256: dict[str, Sha256]
    provider_call_ids: tuple[str, ...]
    execution_receipt_count: NonNegativeInt
    execution_receipt_semantics: Literal["lower_bound_may_miss_precommit_reschedule"]
    error_code: str | None
    started_at_utc: AwareDatetime
    finished_at_utc: AwareDatetime

    @model_validator(mode="after")
    def success_has_complete_output(self) -> "MethodPhaseAttemptReceipt":
        if self.status == "success":
            if set(self.output_artifact_sha256) not in (
                {"checkpoint", "checkpoint_manifest"},
                {"train_manifest", "validation_manifest", "bundle_receipt"},
            ):
                raise ValueError("successful method phase has the wrong output set")
            if not self.provider_call_ids or self.error_code is not None:
                raise ValueError("successful method phase needs provider IDs and no error")
        elif self.error_code is None:
            raise ValueError("failed method phase needs a stable error code")
        return self


class MethodPhaseFinalReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    kind: PhaseKind
    phase_id: str
    workspace_id: str
    training_seed: Literal[17, 29, 43]
    attempt_receipt_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    reservation_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    consumed_permit_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    launch_receipt_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    reconciliation_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    provider_call_ids: tuple[str, ...]
    output_artifact_sha256: dict[str, Sha256]
    metered_delta_usd: Decimal = Field(ge=Decimal("0.00"), decimal_places=2)
    finalized_at_utc: AwareDatetime
```

Generate each schema directly from its Pydantic model with `jsonschema==4.25.1` validation. `finalize_method_phase` loads the canonical `ScientificComputeAuthorization`, `ScientificCostReservation`, `ConsumedPermit` launch receipt, and `ScientificCostReconciliation` types from `ratemem.evaluation.compute`. It recomputes every file hash, requires the same phase/workspace/authorization/reservation throughout, requires exact equality between attempt and reconciliation provider-call ID sets, rejects failed attempts, and atomically creates the final receipt with exclusive creation. It never edits the attempt or reconciliation.

- [ ] **Step 4: Implement guard-first bounded training and one synchronous Modal call**

`run_authorized_training(...)` must:

1. load the method lock, CPU gate, train-only trace manifests, and request; verify seed `17`, `29`, or `43` is present in the method lock and recompute all bound hashes before consuming anything;
2. call the canonical `require_scientific_compute_permit(authorization_path, reservation_path, dataset_lock_path, baseline_lock_path, evaluation_lock_path, method_lock_path, method_cpu_gate_path, expected_phase_id, expected_workspace_id, launch_receipt_path)`;
3. make `launcher.run(request, consumed_permit)` the immediately following effectful statement and invoke it exactly once;
4. run Task 10's bounded segment trainer with detach at every segment boundary and reject a receipt above two transformer passes;
5. save only Task 11's permitted tensors, then reload the produced safetensors and provenance manifest against the current locks before declaring success;
6. atomically write a `MethodPhaseAttemptReceipt` on success or after any consumed-permit failure. Failures before permit consumption write no attempt receipt and make no provider call.

```python
# src/ratemem/training/authorized.py
@dataclass(frozen=True, slots=True)
class ScientificPermitPaths:
    authorization_path: Path
    reservation_path: Path
    dataset_lock_path: Path
    baseline_lock_path: Path
    evaluation_lock_path: Path
    method_lock_path: Path
    method_cpu_gate_path: Path
    launch_receipt_path: Path


@dataclass(frozen=True, slots=True)
class TrainingLaunchResult:
    checkpoint_source: Path
    checkpoint_manifest_source: Path
    checkpoint_sha256: Sha256
    checkpoint_manifest_sha256: Sha256
    provider_call_ids: tuple[str, ...]
    execution_receipt_count: int
    execution_receipt_semantics: Literal[
        "lower_bound_may_miss_precommit_reschedule"
    ]


class ScientificTrainingLauncher(Protocol):
    def run(
        self,
        request: AuthorizedTrainingRequest,
        consumed_permit: ConsumedPermit,
    ) -> TrainingLaunchResult:
        raise NotImplementedError


class ScientificPermitGuard(Protocol):
    def __call__(
        self,
        authorization_path: Path,
        reservation_path: Path,
        dataset_lock_path: Path,
        baseline_lock_path: Path,
        evaluation_lock_path: Path,
        method_lock_path: Path,
        method_cpu_gate_path: Path,
        expected_phase_id: str,
        expected_workspace_id: str,
        launch_receipt_path: Path,
    ) -> ConsumedPermit:
        raise NotImplementedError


def run_authorized_training(
    *,
    request: AuthorizedTrainingRequest,
    permit_paths: ScientificPermitPaths,
    checkpoint_output: Path,
    checkpoint_manifest_output: Path,
    receipt_output: Path,
    launcher: ScientificTrainingLauncher,
    permit_guard: ScientificPermitGuard = require_scientific_compute_permit,
) -> MethodPhaseAttemptReceipt:
    verified = verify_training_request_before_consumption(request, permit_paths)
    consumed = permit_guard(
        permit_paths.authorization_path,
        permit_paths.reservation_path,
        permit_paths.dataset_lock_path,
        permit_paths.baseline_lock_path,
        permit_paths.evaluation_lock_path,
        permit_paths.method_lock_path,
        permit_paths.method_cpu_gate_path,
        request.phase_id,
        request.workspace_id,
        permit_paths.launch_receipt_path,
    )
    result = launcher.run(verified, consumed)
    return validate_downloaded_checkpoint_and_write_attempt(
        verified,
        consumed,
        result,
        checkpoint_output,
        checkpoint_manifest_output,
        receipt_output,
    )
```

`src/ratemem/training/modal_app.py` defines one ephemeral `modal.App("ratemem-scientific-training")` function with exactly one L40S, `retries=0`, `max_containers=1`, `single_use_containers=True`, and the locked timeout/resources. It contains exactly one synchronous `.remote()` call and no `spawn`, `map`, deployment, schedule, detached run, retry loop, seed loop, workspace fallback, or credential argument. The remote function records `modal.current_function_call_id()`, input/task IDs, and the lower-bound execution receipt before training; it writes the checkpoint and manifest to a fixed pre-existing scientific artifact volume and returns hashes and remote relative paths. The local launcher downloads exactly those paths, rejects any hash mismatch, and never chooses a workspace or GPU.

- [ ] **Step 5: Add the exact `train-scientific` and finalization CLIs**

`ratemem-method train-scientific` accepts exactly:

```text
--method-lock PATH
--method-cpu-gate PATH
--dataset-lock PATH
--baseline-lock PATH
--evaluation-lock PATH
--workspace-selection PATH
--trace-dir PATH
--split train
--training-seed {17,29,43}
--compute-authorization PATH
--cost-reservation PATH
--launch-receipt PATH
--checkpoint-output PATH
--checkpoint-manifest-output PATH
--receipt-output PATH
```

It derives `expected_phase_id=f"meta_train_seed_{training_seed}"` and the exact workspace ID from the explicit selection record, builds and hashes `AuthorizedTrainingRequest`, and calls `run_authorized_training`. Stdout must match `^PASS RateMem scientific training: seed=(17|29|43) checkpoint=[0-9a-f]{64} receipt=[0-9a-f]{64}$`. It exposes no token, alternate GPU, fallback workspace, cap override, loop, or retry flag.

`ratemem-method finalize-phase --kind training` accepts `--attempt`, `--authorization`, `--reservation`, `--reconciliation`, and `--output`; it calls only `finalize_method_phase` and prints the final receipt hash.

- [ ] **Step 6: Run all non-paid producer contracts, generate schemas, and commit**

Run:

```bash
uv run ratemem-method phase-schema \
  --training-request-output schemas/ratemem-training-request-v1.schema.json \
  --attempt-output schemas/ratemem-method-phase-attempt-v1.schema.json \
  --final-output schemas/ratemem-method-phase-final-v1.schema.json
uv run pytest tests/unit/training/test_authorized_training.py tests/contract/training/test_training_producer_guard.py tests/contract/training/test_scientific_modal_app.py tests/contract/method/test_phase_receipts.py -q
uv run ruff check src/ratemem/training/authorized.py src/ratemem/training/modal_app.py src/ratemem/method/phase_receipts.py src/ratemem/method/cli.py tests/unit/training/test_authorized_training.py tests/contract/training/test_training_producer_guard.py tests/contract/training/test_scientific_modal_app.py tests/contract/method/test_phase_receipts.py
uv run mypy src/ratemem/training/authorized.py src/ratemem/training/modal_app.py src/ratemem/method/phase_receipts.py src/ratemem/method/cli.py
```

Expected: all tests pass with fake launchers; AST inspection proves one guarded synchronous remote call and no fan-out; generated schemas byte-match their models; provider invocation count remains zero.

```bash
git add src/ratemem/training/authorized.py src/ratemem/training/modal_app.py src/ratemem/method/phase_receipts.py src/ratemem/method/cli.py schemas/ratemem-training-request-v1.schema.json schemas/ratemem-method-phase-attempt-v1.schema.json schemas/ratemem-method-phase-final-v1.schema.json tests/unit/training/test_authorized_training.py tests/contract/training/test_training_producer_guard.py tests/contract/training/test_scientific_modal_app.py tests/contract/method/test_phase_receipts.py
git commit -m "feat(train): add one-shot authorized RateMem producer"
```

- [ ] **Step 7: Deferred paid execution -- produce and reconcile one frozen checkpoint**

Do not run this step while implementing Task 12. Run it only after Task 14's CPU receipt and a current scientific Task 9 authorization/reservation exist for the named seed. The scientific plan owns the per-seed loop; this producer owns one iteration only.

```bash
uv run ratemem-method train-scientific \
  --method-lock configs/method/ratemem-training-lock.yaml \
  --method-cpu-gate artifacts/method/cpu-gate.json \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --trace-dir configs/scientific/traces \
  --split train \
  --training-seed 17 \
  --compute-authorization artifacts/scientific/method/training/seed-17/compute/authorization.json \
  --cost-reservation artifacts/scientific/method/training/seed-17/compute/reservation.json \
  --launch-receipt artifacts/scientific/method/training/seed-17/compute/launch-receipt.json \
  --checkpoint-output artifacts/scientific/method/training/seed-17/checkpoint.safetensors \
  --checkpoint-manifest-output artifacts/scientific/method/training/seed-17/checkpoint.manifest.json \
  --receipt-output artifacts/scientific/method/training/seed-17/attempt-receipt.json
uv run ratemem-eval compute reconcile \
  --authorization artifacts/scientific/method/training/seed-17/compute/authorization.json \
  --reservation artifacts/scientific/method/training/seed-17/compute/reservation.json \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
  --ledger /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl \
  --output artifacts/scientific/method/training/seed-17/compute/reconciliation.json
uv run ratemem-method finalize-phase \
  --kind training \
  --attempt artifacts/scientific/method/training/seed-17/attempt-receipt.json \
  --authorization artifacts/scientific/method/training/seed-17/compute/authorization.json \
  --reservation artifacts/scientific/method/training/seed-17/compute/reservation.json \
  --reconciliation artifacts/scientific/method/training/seed-17/compute/reconciliation.json \
  --output artifacts/scientific/method/training/seed-17/receipt.json
```

Expected: training prints the locked seed, checkpoint hash, and attempt-receipt hash; reconciliation prints the metered delta; finalization prints a final receipt hash. A new seed authorization is forbidden until reconciliation/finalization succeeds. Repeat with fresh phase-specific records for seeds 29 and 43; never reuse a permit.

### Task 13: Produce canonical real shared inputs from the frozen checkpoint

**Depends on:** matched-baseline shared-input Task 3. That task is the sole owner of the concrete provider-neutral types `SharedProviderMetadata`, `ProviderPacketKey`, `ProviderPacketCandidate`, `ProviderEventOutput`, and `SharedInputProvider` in `src/ratemem/baselines/shared_inputs.py`. Implementation and fake-provider tests precede Task 14; real materialization is deferred until the corresponding Task 12 final training receipt and a new scientific Task 9 authorization/reservation exist.

The learned implementation requires the exact baseline-owned frozen fields below before this task starts; the table is a dependency check, not another model definition.

| Canonical type | Exact baseline-owned fields |
|---|---|
| `SharedProviderMetadata` | `provider_id: str`, `provider_revision_sha256: Sha256`, `backbone_id: Literal["sana_1_5_1_6b"]`, `backbone_revision: GitCommit`, `adapter_layout_sha256: Sha256`, `projection_count: Literal[120]`, `code_dim: Literal[480]`, `amortizer_sha256: Sha256`, `adapter_basis_sha256: Sha256`, `codec_dictionary_sha256: Sha256`, `support_pool_sha256: Sha256`, `incidence_gain_step: float` |
| `ProviderPacketKey` | `dictionary_revision_sha256: Sha256`, `group: int`, `stage: int`, `entry: int` |
| `ProviderPacketCandidate` | `key: ProviderPacketKey`, `packet_id: Sha256`, `packet_payload: bytes`, `gain_q: int` |
| `ProviderEventOutput` | `event_index: int`, `handle: str`, `target_code: NDArray[np.float32]`, `base_code: NDArray[np.float32]`, `quantizer_scales: NDArray[np.float32]`, `candidates: tuple[ProviderPacketCandidate, ...]` |

The two exact protocol signatures are `manifest_metadata(self) -> SharedProviderMetadata` and `record_for_event(self, event: LifecycleEvent) -> ProviderEventOutput`.

The baseline implementation validates nonnegative indices, nonempty handles, exact float32 shapes `(480,)`/`(480,)`/`(30,)`, finite tensors, int16 `gain_q`, `sha256(packet_payload) == packet_id`, and candidate key revision equal to metadata `codec_dictionary_sha256`. It aggregates equal packet IDs across event outputs into the manifest's dependent handles and `gain_q_by_handle` map. The candidate key deliberately excludes `gain_q`: dictionary identity is reusable while the signed coefficient is concept-specific. This plan must not redefine any of these types.

**Files:**
- Create: `src/ratemem/method/shared_provider.py`
- Create: `src/ratemem/method/materialize.py`
- Modify: `src/ratemem/method/cli.py`
- Create: `schemas/ratemem-materialize-request-v1.schema.json`
- Test: `tests/unit/method/test_shared_provider.py`
- Test: `tests/contract/method/test_real_shared_input_provider.py`
- Test: `tests/integration/method/test_materialize_shared_inputs.py`
- Test: `tests/contract/method/test_materialize_producer_guard.py`

- [ ] **Step 1: Write failing canonical-provider and real-bundle tests**

```python
# tests/contract/method/test_real_shared_input_provider.py
from ratemem.baselines.shared_inputs import (
    ProviderPacketKey,
    SharedInputProvider,
)
from ratemem.method.shared_provider import RateMemSharedInputProvider


def accepts_canonical_provider(value: SharedInputProvider) -> SharedInputProvider:
    return value


def test_ratemem_provider_has_the_only_canonical_signatures(real_provider_fixture) -> None:
    provider = accepts_canonical_provider(real_provider_fixture.provider())
    assert isinstance(provider, RateMemSharedInputProvider)
    assert provider.manifest_metadata() == real_provider_fixture.expected_metadata
    output = provider.record_for_event(real_provider_fixture.create_event)
    assert output.event_index == real_provider_fixture.create_event.event_index
    assert output.handle == real_provider_fixture.create_event.handle
    assert output.target_code.shape == (480,)
    assert output.base_code.shape == (480,)
    assert output.quantizer_scales.shape == (30,)
    assert len(output.candidates) == 8
    assert tuple(row.packet_payload for row in output.candidates) == (
        real_provider_fixture.hard_codec_selected_payloads
    )
    assert all(isinstance(row.key, ProviderPacketKey) for row in output.candidates)


def test_provider_uses_actual_hard_decode_and_production_topk(real_provider_fixture) -> None:
    provider = real_provider_fixture.provider(groups=30, stages=2, maximum_packets=8)
    output = provider.record_for_event(real_provider_fixture.create_event)
    hard = real_provider_fixture.codec.encode(
        real_provider_fixture.create_event.handle,
        output.target_code,
    )
    actual = real_provider_fixture.codec.decode(hard.base_payload, hard.incidences)
    assert tuple(row.packet_payload for row in output.candidates) == tuple(
        row.packet.payload for row in hard.incidences
    )
    assert tuple(row.gain_q for row in output.candidates) == tuple(
        row.incidence.gain_q for row in hard.incidences
    )
    assert tuple(
        (
            row.key.dictionary_revision_sha256,
            row.key.group,
            row.key.stage,
            row.key.entry,
        )
        for row in output.candidates
    ) == tuple(
        (
            real_provider_fixture.codec.dictionary.revision_sha256,
            row.group,
            row.stage,
            row.entry,
        )
        for row in hard.incidences
    )
    assert len(output.candidates) == 8
    real_provider_fixture.assert_saved_base_and_candidates_reconstruct(
        output,
        hard,
        actual,
    )
```

```python
# tests/integration/method/test_materialize_shared_inputs.py
from ratemem.baselines.shared_inputs import SharedInputReader
from ratemem.method.materialize import materialize_authorized_shared_inputs


def test_one_real_provider_invocation_writes_reproducible_train_and_validation_bundles(
    materialize_fixture,
) -> None:
    attempt = materialize_authorized_shared_inputs(
        **materialize_fixture.arguments("first")
    )
    second_streams = materialize_fixture.fixture_only_replay("second")
    assert attempt.kind == "materialization"
    assert attempt.status == "success"
    assert materialize_fixture.launcher.calls == 1
    assert "synthetic" not in attempt.model_dump_json()
    for split in ("train", "validation"):
        reader = SharedInputReader(
            materialize_fixture.output_root / split,
            method_id="ratemem_v1",
        )
        assert reader.manifest.code_dim == 480
        assert reader.manifest.candidate_stream_sha256 == second_streams[split]
```

- [ ] **Step 2: Run the focused tests and verify the real provider is absent**

Run:

```bash
uv run pytest tests/unit/method/test_shared_provider.py tests/contract/method/test_real_shared_input_provider.py tests/integration/method/test_materialize_shared_inputs.py -q
```

Expected: collection fails importing `ratemem.method.shared_provider` or `ratemem.method.materialize`.

- [ ] **Step 3: Implement the canonical provider without a second shared-input schema**

`RateMemSharedInputProvider` imports `SharedInputProvider`, `SharedProviderMetadata`, `ProviderPacketKey`, `ProviderPacketCandidate`, and `ProviderEventOutput` solely from `ratemem.baselines.shared_inputs`. It defines exactly these methods:

```python
class RateMemSharedInputProvider(SharedInputProvider):
    def manifest_metadata(self) -> SharedProviderMetadata:
        return self._metadata

    def record_for_event(self, event: LifecycleEvent) -> ProviderEventOutput:
        if not isinstance(event, (CreateEvent, UpdateEvent)):
            raise ValueError("shared inputs exist only for create/update events")
        target = self._predict_from_visible_support(event)
        hard = self._codec.encode(event.handle, target)
        if len(hard.incidences) > self._maximum_packets:
            raise RuntimeError("hard codec exceeded the deployed packet cap")
        decoded = self._codec.decode(hard.base_payload, hard.incidences)
        return self._canonical_output(event, target, decoded, hard)
```

`_canonical_output` constructs the baseline-owned objects directly, mapping each learned-private `HardIncidence` to a provider-neutral candidate:

```python
base = self._codec.base_quantizer.encode(target)
candidates = tuple(
    ProviderPacketCandidate(
        key=ProviderPacketKey(
            dictionary_revision_sha256=self._codec.dictionary.revision_sha256,
            group=row.group,
            stage=row.stage,
            entry=row.entry,
        ),
        packet_id=row.packet.packet_id,
        packet_payload=row.packet.payload,
        gain_q=row.incidence.gain_q,
    )
    for row in hard.incidences
)
return ProviderEventOutput(
    event_index=event.event_index,
    handle=event.handle,
    target_code=np.asarray(target, dtype=np.float32),
    base_code=base.decode(),
    quantizer_scales=base.scales(),
    candidates=candidates,
)
```

No `PacketCandidateKey`, `HardIncidence`, wrapper, alternate manifest, duplicate packet record, `from_*` helper, or RateMem-only output field crosses the canonical boundary. The provider:

- loads Task 11's checkpoint only through `load_method_checkpoint` and verifies the final Task 12 receipt, checkpoint manifest, method lock, SANA adapter layout, amortizer, basis, and frozen dictionary hashes;
- calls the frozen support amortizer on only the current event's support/description IDs;
- uses Task 2's hard base payload/decoded base/scales and Task 4's actual deterministic hard codec;
- emits the same eight selected packet keys, payloads, signed quantized gains, and decoded code as deployment from the 30-group, two-stage candidate set;
- is stateless across events except for immutable loaded weights, so the baseline-owned writer controls event order and candidate-stream hashing.

Add a mypy contract that assigns `RateMemSharedInputProvider` to `SharedInputProvider` and inspects `inspect.signature` to require exactly `manifest_metadata(self)` and `record_for_event(self, event)`. Tests must fail if `src/ratemem/method` defines another `SharedInputProvider`, `SharedInputManifest`, or `ProviderEventOutput`.

- [ ] **Step 4: Implement one guarded materialization invocation and receipt**

Define `AuthorizedMaterializeRequest` with `phase_id` matching `^materialize_shared_inputs_seed_(17|29|43)$`, exact workspace ID, `training_seed`, splits exactly `("train", "validation")`, trace-manifest hashes, finalized training-receipt hash, checkpoint/manifest hashes, all five lock/gate hashes, code dimension `480`, group/stage/maximum packet literals `30/2/8`, and canonical shared-input schema hash. Generate `schemas/ratemem-materialize-request-v1.schema.json` from this model.

```python
class AuthorizedMaterializeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    phase_id: str = Field(
        pattern=r"^materialize_shared_inputs_seed_(17|29|43)$"
    )
    workspace_id: str = Field(min_length=1)
    training_seed: Literal[17, 29, 43]
    splits: tuple[Literal["train"], Literal["validation"]]
    trace_manifest_sha256: dict[Literal["train", "validation"], Sha256]
    training_receipt_sha256: Sha256
    checkpoint_sha256: Sha256
    checkpoint_manifest_sha256: Sha256
    dataset_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    method_lock_sha256: Sha256
    method_cpu_gate_sha256: Sha256
    shared_input_schema_sha256: Sha256
    code_dimension: Literal[480]
    group_count: Literal[30]
    rvq_stages: Literal[2]
    maximum_packets: Literal[8]

    @model_validator(mode="after")
    def phase_and_splits_are_exact(self) -> "AuthorizedMaterializeRequest":
        if self.phase_id != (
            f"materialize_shared_inputs_seed_{self.training_seed}"
        ):
            raise ValueError("materialization phase id does not match its seed")
        if self.splits != ("train", "validation"):
            raise ValueError("materialization is train then validation only")
        if set(self.trace_manifest_sha256) != {"train", "validation"}:
            raise ValueError("both visible split manifests are required")
        return self
```

Use this exact producer boundary:

```python
@dataclass(frozen=True, slots=True)
class MaterializationLaunchResult:
    output_source: Path
    train_manifest_sha256: Sha256
    validation_manifest_sha256: Sha256
    bundle_receipt_sha256: Sha256
    candidate_stream_sha256: dict[Literal["train", "validation"], Sha256]
    provider_call_ids: tuple[str, ...]
    execution_receipt_count: int
    execution_receipt_semantics: Literal[
        "lower_bound_may_miss_precommit_reschedule"
    ]


class SharedInputMaterializationLauncher(Protocol):
    def run(
        self,
        request: AuthorizedMaterializeRequest,
        consumed_permit: ConsumedPermit,
    ) -> MaterializationLaunchResult:
        raise NotImplementedError


def materialize_authorized_shared_inputs(
    *,
    request: AuthorizedMaterializeRequest,
    permit_paths: ScientificPermitPaths,
    output: Path,
    receipt_output: Path,
    launcher: SharedInputMaterializationLauncher,
    permit_guard: ScientificPermitGuard = require_scientific_compute_permit,
) -> MethodPhaseAttemptReceipt:
    verified = verify_materialize_request_before_consumption(request, permit_paths)
    consumed = permit_guard(
        permit_paths.authorization_path,
        permit_paths.reservation_path,
        permit_paths.dataset_lock_path,
        permit_paths.baseline_lock_path,
        permit_paths.evaluation_lock_path,
        permit_paths.method_lock_path,
        permit_paths.method_cpu_gate_path,
        request.phase_id,
        request.workspace_id,
        permit_paths.launch_receipt_path,
    )
    result = launcher.run(verified, consumed)
    return validate_bundles_and_write_attempt(
        verified,
        consumed,
        result,
        output,
        receipt_output,
    )
```

`materialize_authorized_shared_inputs(...)` verifies every input and confirms the training receipt is final and reconciled before calling the same canonical `require_scientific_compute_permit(...)`. Its immediately following effectful statement is one `launcher.run(request, consumed_permit)` call. That one remote phase instantiates `RateMemSharedInputProvider` once and calls the baseline-owned `write_shared_input_bundle` once per requested split, passing a visible event iterator rather than a trace path. It rejects `final`/`final_test`, path traversal, duplicate events, a second provider construction, a second permit use, a fixture provider, or any edited output. The resulting attempt receipt uses `kind="materialization"` and binds the exact train/validation manifests, combined bundle receipt, candidate-stream hashes, checkpoint/final-training-receipt hashes, provider IDs, consumed permit, launch receipt, and locks.

- [ ] **Step 5: Add the exact `materialize-shared-inputs` CLI**

The CLI accepts exactly the scientific-plan handoff plus the lock paths needed by the canonical guard:

```text
--method-lock PATH
--method-cpu-gate PATH
--dataset-lock PATH
--baseline-lock PATH
--evaluation-lock PATH
--workspace-selection PATH
--checkpoint PATH
--checkpoint-manifest PATH
--training-receipt PATH
--trace-dir PATH
--splits train,validation
--schema PATH
--compute-authorization PATH
--cost-reservation PATH
--launch-receipt PATH
--output PATH
--receipt-output PATH
```

It derives the seed and `f"materialize_shared_inputs_seed_{training_seed}"` phase from the finalized training receipt, never from an unchecked CLI string. Stdout must match `^PASS real shared-inputs: seed=(17|29|43) candidate_stream=[0-9a-f]{64} receipt=[0-9a-f]{64}$`. Finalize it with the existing `ratemem-method finalize-phase --kind materialization` command after scientific reconciliation.

- [ ] **Step 6: Run all non-paid provider/materializer tests, generate the schema, and commit**

Run:

```bash
uv run ratemem-method materialize-request-schema --output schemas/ratemem-materialize-request-v1.schema.json
uv run pytest tests/unit/method/test_shared_provider.py tests/contract/method/test_real_shared_input_provider.py tests/integration/method/test_materialize_shared_inputs.py tests/contract/method/test_materialize_producer_guard.py -q
uv run ruff check src/ratemem/method/shared_provider.py src/ratemem/method/materialize.py src/ratemem/method/cli.py tests/unit/method/test_shared_provider.py tests/contract/method/test_real_shared_input_provider.py tests/integration/method/test_materialize_shared_inputs.py tests/contract/method/test_materialize_producer_guard.py
uv run mypy src/ratemem/method/shared_provider.py src/ratemem/method/materialize.py src/ratemem/method/cli.py
```

Expected: all tests pass using frozen fixture weights and fake launchers; the canonical baseline reader opens both bundles; tampered locks/checkpoints/final receipts fail before provider invocation; provider invocation count remains zero.

```bash
git add src/ratemem/method/shared_provider.py src/ratemem/method/materialize.py src/ratemem/method/cli.py schemas/ratemem-materialize-request-v1.schema.json tests/unit/method/test_shared_provider.py tests/contract/method/test_real_shared_input_provider.py tests/integration/method/test_materialize_shared_inputs.py tests/contract/method/test_materialize_producer_guard.py
git commit -m "feat(method): add real shared-input producer"
```

- [ ] **Step 7: Deferred paid execution -- materialize, reconcile, and finalize one seed**

Run only after Task 14 and a new materialization-specific scientific Task 9 authorization/reservation:

```bash
uv run ratemem-method materialize-shared-inputs \
  --method-lock configs/method/ratemem-training-lock.yaml \
  --method-cpu-gate artifacts/method/cpu-gate.json \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --checkpoint artifacts/scientific/method/training/seed-17/checkpoint.safetensors \
  --checkpoint-manifest artifacts/scientific/method/training/seed-17/checkpoint.manifest.json \
  --training-receipt artifacts/scientific/method/training/seed-17/receipt.json \
  --trace-dir configs/scientific/traces \
  --splits train,validation \
  --schema schemas/ratemem-shared-input-bundle-v1.schema.json \
  --compute-authorization artifacts/scientific/baselines/execution/shared-input/seed-17/compute/authorization.json \
  --cost-reservation artifacts/scientific/baselines/execution/shared-input/seed-17/compute/reservation.json \
  --launch-receipt artifacts/scientific/baselines/execution/shared-input/seed-17/compute/launch-receipt.json \
  --output artifacts/scientific/baselines/execution/shared-input/seed-17 \
  --receipt-output artifacts/scientific/baselines/execution/shared-input/seed-17/attempt-receipt.json
uv run ratemem-eval compute reconcile \
  --authorization artifacts/scientific/baselines/execution/shared-input/seed-17/compute/authorization.json \
  --reservation artifacts/scientific/baselines/execution/shared-input/seed-17/compute/reservation.json \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
  --ledger /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl \
  --output artifacts/scientific/baselines/execution/shared-input/seed-17/compute/reconciliation.json
uv run ratemem-method finalize-phase \
  --kind materialization \
  --attempt artifacts/scientific/baselines/execution/shared-input/seed-17/attempt-receipt.json \
  --authorization artifacts/scientific/baselines/execution/shared-input/seed-17/compute/authorization.json \
  --reservation artifacts/scientific/baselines/execution/shared-input/seed-17/compute/reservation.json \
  --reconciliation artifacts/scientific/baselines/execution/shared-input/seed-17/compute/reconciliation.json \
  --output artifacts/scientific/baselines/execution/shared-input/seed-17/receipt.json
```

Expected: the canonical bundle validates, reconciliation binds the same provider call IDs, and the finalized receipt becomes the only materialization receipt accepted by baseline search. Repeat for seeds 29 and 43 with new authorizations/reservations; baseline per-method search remains owned by the matched-baseline/scientific plans.

### Task 14: Prove synthetic nonseparability and freeze the CPU end-to-end contract

**Files:**
- Create: `src/ratemem/method/synthetic.py`
- Create: `tests/integration/method/test_nonseparability.py`
- Create: `tests/integration/method/test_method_lifecycle.py`
- Create: `tests/integration/training/test_meta_training_smoke.py`
- Create: `tests/contract/method/test_novelty_boundary.py`
- Modify: `src/ratemem/method/cli.py`
- Create: `schemas/ratemem-method-cpu-gate-v1.schema.json`
- Create: `docs/method/ratemem-v1-interface.md`

- [ ] **Step 1: Write the synthetic shared-packet falsification test**

```python
# tests/integration/method/test_nonseparability.py
import numpy as np

from ratemem.method.synthetic import build_shared_direction_fixture, private_copy_state
from ratemem.state.model import MemoryState


def test_one_retained_packet_improves_two_concepts_for_one_payload_cost() -> None:
    fixture = build_shared_direction_fixture()
    state = fixture.encode_all()
    shared = [
        packet_id for packet_id in state.packets
        if {edge.handle for edge in state.incidences.values() if edge.packet_id == packet_id} == {"a", "b"}
    ]
    assert len(shared) == 1
    packet_id = shared[0]
    full_error = fixture.total_code_error(state)
    removed = MemoryState(
        bases=state.bases,
        packets={key: row for key, row in state.packets.items() if key != packet_id},
        incidences={key: row for key, row in state.incidences.items() if row.packet_id != packet_id},
    )
    assert fixture.code_error("a", removed) > fixture.code_error("a", state)
    assert fixture.code_error("b", removed) > fixture.code_error("b", state)
    assert fixture.total_code_error(removed) > full_error
    private = private_copy_state(state)
    assert state.serialized_bytes < private.serialized_bytes
```

- [ ] **Step 2: Run the nonseparability test and verify the fixture is absent**

Run: `uv run pytest tests/integration/method/test_nonseparability.py -q`

Expected: collection fails because `ratemem.method.synthetic` does not exist.

- [ ] **Step 3: Implement the deterministic fixture without reporting it as scientific evidence**

```python
# src/ratemem/method/synthetic.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer
from ratemem.method.codec import HardIncidence, RateMemHardCodec
from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary
from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import packet_from_payload


@dataclass(frozen=True)
class SharedDirectionFixture:
    codec: RateMemHardCodec
    targets: dict[str, np.ndarray]

    def encode_all(self) -> MemoryState:
        bases, packets, incidences = {}, {}, {}
        for index, (handle, code) in enumerate(sorted(self.targets.items())):
            encoded = self.codec.encode(handle, code)
            bases[handle] = BaseRecord(handle, encoded.base_payload, 0, index)
            for row in encoded.incidences:
                packets[row.packet.packet_id] = row.packet
                incidences[(handle, row.packet.packet_id)] = row.incidence
        return MemoryState(bases, packets, incidences)

    def code_error(self, handle: str, state: MemoryState) -> float:
        rows = []
        for edge in state.incidences.values():
            if edge.handle == handle:
                packet = state.packets[edge.packet_id]
                group, stage, entry = self.codec.dictionary.validate_packet(packet)
                rows.append(HardIncidence(edge, packet, group, stage, entry, 0.0))
        decoded = self.codec.decode(state.bases[handle].payload, tuple(rows))
        return float(np.mean((decoded - self.targets[handle]) ** 2))

    def total_code_error(self, state: MemoryState) -> float:
        return sum(self.code_error(handle, state) for handle in sorted(self.targets))


def build_shared_direction_fixture() -> SharedDirectionFixture:
    dictionary = GroupRVQDictionary(group_count=1, group_size=4, stages=1, entries=2)
    with torch.no_grad():
        dictionary.codebooks.copy_(torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]]))
    dictionary.normalize_codebooks_()
    codec = RateMemHardCodec(
        BlockwiseBaseQuantizer(group_size=4, bits=2),
        freeze_dictionary(dictionary), gain_step=1 / 256, maximum_packets=1,
    )
    return SharedDirectionFixture(
        codec=codec,
        targets={
            "a": np.array([0.80, 0.06, 0.0, 0.0], dtype=np.float32),
            "b": np.array([0.72, -0.04, 0.0, 0.0], dtype=np.float32),
        },
    )


def private_copy_state(shared: MemoryState) -> MemoryState:
    packets, incidences = {}, {}
    for edge in sorted(shared.incidences.values(), key=lambda row: (row.handle, row.packet_id)):
        source = shared.packets[edge.packet_id]
        private_payload = bytearray(source.payload)
        digest = hashlib.sha256(edge.handle.encode()).digest()
        private_payload[8:40] = bytes(left ^ right for left, right in zip(private_payload[8:40], digest))
        private = packet_from_payload(bytes(private_payload))
        packets[private.packet_id] = private
        incidences[(edge.handle, private.packet_id)] = Incidence(edge.handle, private.packet_id, edge.gain_q)
    return MemoryState(shared.bases, packets, incidences)
```

This fixture is a mechanistic contract only. Its errors, byte difference, and pass status never enter a paper result table or scientific artifact.

- [ ] **Step 4: Add full lifecycle and training-smoke contracts**

```python
# tests/integration/method/test_method_lifecycle.py
def test_hard_method_lifecycle_is_deterministic_byte_bounded_and_collateral_safe(method_fixture, trace) -> None:
    first = method_fixture.replay(trace)
    second = method_fixture.replay(trace)
    assert first.canonical_bytes == second.canonical_bytes
    assert all(row.online_state_bytes <= trace.byte_budget for row in first.receipts)
    assert first.outcomes == (
        "created", "created", "read", "updated", "deleted", "stale_handle"
    )
    assert first.unrelated_code_hash_before_delete == first.unrelated_code_hash_after_delete


def test_packet_removal_changes_only_declared_dependents(method_fixture) -> None:
    state = method_fixture.shared_state()
    packet_id = next(iter(state.packets))
    declared = {edge.handle for edge in state.incidences.values() if edge.packet_id == packet_id}
    before, after = method_fixture.decode_all(state), method_fixture.remove_packet_and_decode(state, packet_id)
    assert {handle for handle in before if before[handle] != after[handle]} == declared
```

```python
# tests/integration/training/test_meta_training_smoke.py
def test_two_segment_cpu_training_reduces_locked_synthetic_objective(tiny_training_fixture) -> None:
    before = tiny_training_fixture.evaluate_objective()
    receipts = tiny_training_fixture.train_segments(count=2)
    after = tiny_training_fixture.evaluate_objective()
    assert len(receipts) == 2
    assert all(receipt.transformer_passes <= 2 for receipt in receipts)
    assert after < before
    assert tiny_training_fixture.full_denoising_calls == 0
```

Use a fixed CPU seed, tiny adapter bank, fake one-timestep flow backend, and the frozen two-event trace fixture. The test asserts reduction only on its own synthetic objective and makes no image-quality or benchmark claim.

- [ ] **Step 5: Lock the novelty boundary in source and documentation**

```python
# tests/contract/method/test_novelty_boundary.py
from pathlib import Path


def test_interface_document_names_borrowed_and_proposed_parts() -> None:
    text = Path("docs/method/ratemem-v1-interface.md").read_text(encoding="utf-8")
    for borrowed in ("support-to-adapter amortization", "RVQ", "quantization", "submodular allocator"):
        assert borrowed in text
    for proposed in (
        "learned multi-concept immutable packet bundles",
        "exact serialized-byte lifecycle",
        "causal packet reuse",
    ):
        assert proposed in text
    assert "competitive ratio over future traces" not in text
    assert "dynamic regret guarantee" not in text
```

Write `docs/method/ratemem-v1-interface.md` with the exact public types, tensor shapes, hard packet format, incidence gain rule, state-ledger components, causal feature timestamps, controller/theorem boundary, training pass cap, checkpoint fields, canonical baseline adapter signatures, authorized producer inputs/outputs, and failure behavior defined in Tasks 1--13. State that the synthetic contract is not empirical evidence and that all result claims require the locked scientific pipeline.

- [ ] **Step 6: Add a CPU-only verification command**

Extend `src/ratemem/method/cli.py` with `verify-cpu`. It invokes pytest as a subprocess with an explicit test list, strips provider-related environment variables from the child environment, requires a clean commit, and writes a canonical machine-readable receipt. It refuses `--paid`, GPU selection, workspace selection, or credential inputs because those flags are not defined.

```python
class MethodCpuGateReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass"]
    method_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    clean_diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_invocations: Literal[0]
    created_at_utc: AwareDatetime


@app.command("verify-cpu")
def verify_cpu(
    method_lock: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    receipt: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    import hashlib
    import os
    import subprocess
    import sys
    from datetime import datetime, timezone

    env = {
        key: value for key, value in os.environ.items()
        if not any(token in key.upper() for token in ("MODAL_TOKEN", "HF_TOKEN", "WANDB_API"))
    }
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/unit/method", "tests/contract/method", "tests/integration/method",
        "tests/unit/training", "tests/contract/training", "tests/integration/training",
    ]
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)
    if status:
        typer.echo("BLOCKED RateMem CPU method gate: git worktree is not clean", err=True)
        raise typer.Exit(2)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    row = MethodCpuGateReceipt(
        status="pass",
        method_lock_sha256=file_sha256(method_lock),
        git_commit=commit,
        clean_diff_sha256=hashlib.sha256(diff).hexdigest(),
        test_command_sha256=hashlib.sha256("\0".join(command).encode()).hexdigest(),
        provider_invocations=0,
        created_at_utc=datetime.now(timezone.utc),
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_bytes(canonical_json_bytes(row.model_dump(mode="json")))
    typer.echo(f"PASS RateMem CPU method gate: receipt={receipt}")
```

Import `Annotated`, `AwareDatetime`, `BaseModel`, `ConfigDict`, `Field`, and `Literal`, plus the existing `canonical_json_bytes` and `file_sha256`, at module scope. Add `cpu-gate-schema` to write `MethodCpuGateReceipt.model_json_schema()` canonically. A `verify-cpu-receipt` command validates the schema, current method-lock hash, clean commit, empty-diff hash, `provider_invocations == 0`, and test-command hash before printing `PASS RateMem CPU receipt: <sha256>`.

- [ ] **Step 7: Run the complete CPU release gate and credential scan**

Run:

```bash
uv run ratemem-method cpu-gate-schema --output schemas/ratemem-method-cpu-gate-v1.schema.json
git add src/ratemem/method/synthetic.py src/ratemem/method/cli.py schemas/ratemem-method-cpu-gate-v1.schema.json tests/integration/method tests/integration/training/test_meta_training_smoke.py tests/contract/method/test_novelty_boundary.py docs/method/ratemem-v1-interface.md
git commit -m "test(method): freeze RateMem nonseparability contract"
uv run ratemem-method verify-cpu \
  --method-lock configs/method/ratemem-training-lock.yaml \
  --receipt artifacts/method/cpu-gate.json
uv run ratemem-method verify-cpu-receipt \
  --method-lock configs/method/ratemem-training-lock.yaml \
  --receipt artifacts/method/cpu-gate.json
uv run pytest -q -m "not cuda and not real_sana and not paid_modal"
uv run ruff check src tests
uv run mypy src/ratemem
git diff --check
uv run python - <<'PY'
import re
from pathlib import Path

pattern = re.compile("(?:a" + "k|a" + "s)-[A-Za-z0-9_-]{20,}")
matches = [str(path) for path in Path(".").rglob("*") if path.is_file() and pattern.search(path.read_text(errors="ignore"))]
raise SystemExit("credential-shaped value found in: " + ", ".join(matches) if matches else 0)
PY
```

Expected: the CPU method gate and full non-paid suite pass; `artifacts/method/cpu-gate.json` validates against the generated schema and records the current method-lock hash, current clean commit, empty-diff hash, exact test-command hash, and `provider_invocations: 0`; Ruff, mypy, and `git diff --check` exit 0; the credential scan returns no match. No provider command has run.

- [ ] **Step 8: Freeze the method interface and stop before scientific payment**

```bash
uv run ratemem-method verify-cpu-receipt \
  --method-lock configs/method/ratemem-training-lock.yaml \
  --receipt artifacts/method/cpu-gate.json
git status --short
```

Expected: commit succeeds and `git status --short` prints nothing. This receipt is the handoff between sealed scientific Task 8 and scientific Task 9. Resume at scientific Task 9, which must produce a new authorization and exact cost reservation for one named phase before invoking the deferred Task 12 or Task 13 producer. The CPU receipt remains immutable with `provider_invocations: 0`; paid attempt/final receipts are separate artifacts and never rewrite it.

## Self-review checklist completed while authoring

- The method plan covers the blockwise base codec, learned reusable group/RVQ dictionary, per-concept quantized gains, deterministic hard and differentiable soft/STE paths, deterministic quantized-gain 60-to-8 selection with deployed tie order and actual hard decoding, measured soft--hard agreement, forward-only candidate generation, resident exact reuse, immutable full-incidence bundles, nonnegative causal utility calibration, exact serialized costs, empirical outer lifecycle control, the sole canonical baseline adapter contract, bounded sequential meta-training, checkpoint provenance, one-shot authorized training/materialization producers, reconciliation-bound final receipts, synthetic nonseparability, and end-to-end lifecycle contracts.
- The theorem boundary is consistent with the core plan: only fixed-cohort packet selection from
  the causal pre-screened set `C_t` uses the certified allocator and its ratio is only against the
  exact optimum on that same `C_t`; full-pool `G_t` pre-screen loss, base admission, whole-concept
  eviction, rejection, cohort projection, switching behavior, and future-trace comparisons remain
  outside it.
- All tensor shapes use code dimension 480, 30 groups of width 16 in the locked scientific policy, 120 projections, and four atoms. Tiny tests declare their smaller dimensions explicitly.
- Packet identities consistently use `(dictionary revision, group, stage, entry)`; incidence coefficients consistently use signed int16 storage with the frozen `1/256` gain step.
- The implementation adds to the Python 3.11 `uv` project and retains `jsonschema==4.25.1`; it does not introduce a second environment or replace shared dependency pins.
- Scientific data/model launches remain blocked until dataset, evaluation, baseline, trace, method, paid-workspace, and phase-cost records are all present and current.
- The learned adapter imports the exact baseline-owned `BaselineAdapter`, `EventReceipt`, `ExactByteLedger`, snapshot, and probe types; the real shared-input provider imports the exact baseline-owned provider metadata/key/candidate/output types and keeps `gain_q` outside the reusable dictionary address.
- The `train-scientific` and `materialize-shared-inputs` flags, seed set, artifact paths, attempt/final receipt split, reconciliation commands, and stdout regexes match scientific Task 10 exactly; each phase consumes a distinct Task 9 permit.
