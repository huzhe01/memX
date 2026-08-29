# RateMem SANA Integration and Modal Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the engineering-only SANA-1.5 dynamic-adapter path, support amortizer, one-timestep flow trainer, byte-checksummed pilot artifacts, and one fail-closed Modal L40S pilot submission under the hard USD 28 workspace cap and the separate USD 27 pre-launch admission bound.

**Architecture:** A new Python 3.11 `ratemem` package wraps only SANA q/k/v projections with contract-tested low-rank atom modules, predicts per-example coefficients from frozen support and description features, and trains the atoms plus amortizer through exactly one random flow-matching timestep per query. A separate pilot layer owns immutable pins, held-in Subjects200K preprocessing, artifact validation, workspace attestation, Decimal cost reservations, and a single synchronous Modal call. Private mode-0700 state directories and mode-0600 immutable files hold one global slot, one launch permit, and one `O_EXCL` submission receipt whose attempt ID, workspace, and source hash must match exactly before `.remote()`; the pilot CLI exposes no scientific-evaluation, deployment, fan-out, fallback-GPU, or publication path.

**Tech Stack:** Python 3.11, uv, PyTorch 2.13.0, Torchvision 0.28.0, Diffusers 0.40.0, PEFT 0.20.0, Transformers 5.16.1, Hugging Face Hub/Datasets, safetensors, JSON Schema Draft 2020-12, Typer, pytest 9.0.3, Modal 1.5.4, NVIDIA L40S/BF16.

---

## Scope and locked external inputs

This plan implements engineering probes only. The accepted outputs are checkpoint/API compatibility, numerical and gradient contracts, frozen-backbone integrity, a one-step inference, a one-random-timestep backward pass, CUDA peak memory, p50/p95 step time, a p95-derived held-in step cap, and an observed tiny held-in flow-loss change. Every artifact has `scope="engineering_pilot_only"` and `publication_eligible=false`; no code in this plan computes a CVPR endpoint, compares memory policies, runs a 5--50-concept lifecycle, opens final-test data, or exercises GMM, composition, autonomous lookup, or augmentation.

Lock these identities in committed configuration and pass the full revisions to every Hub load:

| Resource | Locked identity |
|---|---|
| SANA checkpoint | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers@b77948f2b4eed5c728e9b828ccff07f7427b43cc` |
| Frozen support encoder | `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| Held-in engineering data | `Yuanshi/Subjects200K@0d1cf6536239888f1a8e218790649344810067bc`, streamed rows 0--7 only |
| SANA adapter layout | 20 blocks x 2 attention kinds x q/k/v x 4 atoms = 480 FP32 predictor logits |
| Dynamic atoms | rank 4, atom count 4, 120 wrapped projections, 8,601,600 trainable atom parameters |
| Modal profile/environment | profile `ratemem-pilot`, environment `main`; profile is never made globally active |
| Modal resources | one `L40S` requested per execution, 4 physical CPU cores, 32 GiB requested RAM, `retries=0` for user-code failures, `max_containers=1` concurrent container |
| Cost policy | workspace usage budget exactly USD 28.00; internal metered-usage ceiling USD 27.00; first pilot USD 21.00 = USD 2.00 setup/cache/inference/backward + USD 3.00 timing + USD 16.00 held-in pilot; remaining USD 6.00 is an unallocated safety buffer and authorizes no rerun |

The model revision is a content commit, not the moving `main` branch. Training constructs `FlowMatchEulerDiscreteScheduler` directly from the immutable 1000-step, shift-1 contract; only inference retains the pinned checkpoint DPM scheduler. The public SANA and DINO checkpoints require no Hugging Face credential.

The launch contract is exactly one synchronous `.remote()` submission. Modal may still reschedule a container after an infrastructure crash or preemption even with `retries=0`; therefore neither `max_containers=1` nor the USD 27 client ledger proves a single physical execution or hard-caps realized spend. The USD 27 rule is conservative admission accounting. The verified USD 28 Workspace usage budget is the hard outer stop, every available runtime execution receipt is preserved as a lower bound on container attempts, and actual pre-credit metered usage is reconciled before any later launch.

## File responsibility map

The legacy TensorFlow files remain untouched. Add the modern implementation as a separate package:

- `.gitignore` — extends the core ignore rules with pilot caches, credentials, downloaded data, and generated artifacts.
- `pyproject.toml` — extends the core Python 3.11/uv project with compatible exact SANA pins and pilot markers without removing core dependencies or scripts; Task 13 adds the pilot entry point only when its target module exists.
- `uv.lock` — updates the core resolved graph and is reused unchanged inside the Modal image.
- `configs/pilot/sana-1.5-1.6b.json` — immutable model/support-encoder revisions and the 20x2x3x4 adapter layout.
- `configs/pilot/subjects200k-held-in.json` — immutable dataset revision, composite-image geometry, and rows 0--7.
- `configs/pilot/modal-budget.json` — exact profile, environment, GPU/resource, timeout, storage, and USD bounds.
- `schemas/ratemem-pilot-attempt-v1.schema.json` — artifact contract, including source, revisions, costs, timings, Modal IDs, metrics, and checkpoint checksums.
- `src/ratemem/adapters/dynamic_atom_linear.py` — dynamic low-rank linear execution with no dense delta weight.
- `src/ratemem/adapters/sana_layout.py` — stable projection order, SANA wrapper installation, and coefficient activation.
- `src/ratemem/adapters/checkpoint.py` — trainable-only safetensors save/load and metadata validation.
- `src/ratemem/sana/components.py` — pinned SANA/DINO loading and frozen-component assertions.
- `src/ratemem/sana/flow.py` — one-timestep flow sampling, objective, and one-pass train step.
- `src/ratemem/support/amortizer.py` — permutation-invariant Set Transformer and bounded per-projection coefficients.
- `src/ratemem/support/features.py` — frozen DINO and SANA-description feature extraction.
- `src/ratemem/pilot/config.py` — typed, exact-key parsing of the three committed pilot configs.
- `src/ratemem/pilot/data.py` — deterministic Subjects200K composite split and precomputed tensor cache.
- `src/ratemem/pilot/artifacts.py` — atomic artifact writing, schema checks, hashes, and finalization.
- `src/ratemem/pilot/private_io.py` — owner/mode/symlink checks, canonical private JSON writes, directory fsync, and process-safe file locking.
- `src/ratemem/pilot/costs.py` — normalized rates, conservative bounds, and append-only hash-chained cost ledger.
- `src/ratemem/pilot/workspace.py` — isolated-profile, billing, evidence, and freshness checks.
- `src/ratemem/pilot/probes.py` — allowed probe enumeration, timers, peak-memory recorder, and p95 step-cap calculation.
- `src/ratemem/pilot/runner.py` — ordered first-pilot execution and fail-safe artifact emission.
- `src/ratemem/pilot/modal_app.py` — one ephemeral Modal function and one local entry point.
- `src/ratemem/pilot/one_shot.py` — immutable global-slot, permit, and submission-receipt identity protocol.
- `src/ratemem/pilot/cli.py` — attestation, reservation, reconciliation, validation, and credential-scan commands.
- `scripts/run_modal_pilot.sh` — the only supported paid-launch command; one `.remote()` call and no retry loop.
- `docs/runbooks/ratemem-sana-modal-pilot.md` — exact dashboard, authentication, launch, reconciliation, and cleanup procedure.
- `tests/unit/` — CPU-fast tensor, parsing, cost, data, schema, and runner tests.
- `tests/contract/` — dynamic adapter, tiny randomized SANA, Modal AST, and CUDA memory contracts.
- `tests/integration/` — opt-in real-checkpoint tests executed inside the paid pilot only.
- `tests/fixtures/modal/` — credential-free Modal profile, billing-summary, and rate JSON fixtures.

### Task 1: Extend the core package with the locked SANA pilot environment

**Depends on:** the core RateMem implementation plan has already created the Python 3.11 uv project, `.gitignore`, `pyproject.toml`, `uv.lock`, `src/ratemem/`, and its package smoke test.

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/ratemem/adapters/__init__.py` only if the core plan did not create it
- Create: `src/ratemem/sana/__init__.py`
- Create: `src/ratemem/support/__init__.py`
- Create: `src/ratemem/pilot/__init__.py`
- Create: `tests/unit/test_sana_dependency_contract.py`

- [ ] **Step 1: Inspect the completed core scaffold before editing shared files**

Run: `sed -n '1,260p' pyproject.toml && sed -n '1,220p' .gitignore && uv tree --depth 1`

Expected: the existing project targets Python 3.11 and already contains the core dependencies, scripts, pytest settings, and package layout. Record the existing `[project]`, `[project.scripts]`, `[dependency-groups]`, and pytest marker entries so the following merge preserves every one of them.

- [ ] **Step 2: Add default-versus-extra dependency contract tests before changing the lock**

```python
# tests/unit/test_sana_dependency_contract.py (essential contract)
from importlib.metadata import PackageNotFoundError, version

import pytest


DEFAULT_EXPECTED = {
    "accelerate": "1.14.0",
    "cbor2": "6.1.4",
    "datasets": "5.0.1",
    "diffusers": "0.40.0",
    "filelock": "3.32.4",
    "huggingface-hub": "1.29.0",
    "jsonschema": "4.26.0",
    "peft": "0.20.0",
    "pillow": "12.3.0",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.16.1",
    "typer": "0.27.2",
}


def installed(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


@pytest.mark.parametrize(("distribution", "expected"), DEFAULT_EXPECTED.items())
def test_default_sana_dependency_has_exact_version(distribution: str, expected: str) -> None:
    assert installed(distribution) == expected


@pytest.mark.skipif(installed("modal") is None, reason="install the Modal extra")
def test_installed_modal_extra_has_exact_version() -> None:
    assert installed("modal") == "1.5.4"
```

Also parse `pyproject.toml` with `tomllib` and assert that Modal is absent from default dependencies, present exactly once under the `modal` optional dependency, and that the pilot script is not registered before Task 13 creates its target module.

- [ ] **Step 3: Run the dependency contract and confirm the pilot stack is not installed yet**

Run: `uv run pytest tests/unit/test_sana_dependency_contract.py -q`

Expected: FAIL on at least one missing distribution or version mismatch before the SANA dependencies are merged. After an ordinary default sync, the Modal-specific runtime assertion must skip rather than fail.

- [ ] **Step 4: Merge exact SANA dependencies without replacing core configuration**

Run these additive uv commands from the repository root:

```bash
uv add accelerate==1.14.0 cbor2==6.1.4 datasets==5.0.1 diffusers==0.40.0 huggingface-hub==1.29.0 peft==0.20.0 pillow==12.3.0 safetensors==0.8.0 torch==2.13.0 torchvision==0.28.0 transformers==5.16.1
uv add --optional modal modal==1.5.4
uv add filelock==3.32.4 jsonschema==4.26.0 typer==0.27.2
uv add --dev pytest==9.0.3
```

These commands update existing dependency arrays in place. The `cbor2` pin is deliberately raised because the previous core pin is affected by GHSA-3c37-wwvx-h642; run the core canonical-CBOR byte contracts after the upgrade. Do not delete or repin the core plan's Pydantic, PyYAML, or Hypothesis dependencies, do not replace existing `[project.scripts]` entries, and do not rewrite the core Ruff, mypy, build-backend, or pytest settings. Do not register the pilot script in Task 1: its CLI module does not exist until Task 13.

Append only absent pytest markers to the existing marker list:

```toml
markers = [
  "cuda: requires a CUDA device",
  "real_sana: downloads and loads the pinned SANA checkpoint",
  "paid_modal: runs only inside the authorized Modal pilot",
]
```

If the core marker list already has entries, merge these three strings into that list rather than declaring a second `markers` key.

- [ ] **Step 5: Extend ignore rules without replacing core entries**

Append only missing lines to `.gitignore`:

```gitignore
.env
.env.*
*.pem
*.key
.modal.toml
artifacts/pilot/
data/cache/
hf-cache/
wandb/
```

- [ ] **Step 6: Verify both the ordinary default environment and the explicit Modal extra**

Run:

```bash
uv sync --frozen --all-groups
uv run --frozen pytest tests/unit/test_sana_dependency_contract.py -q
uv sync --frozen --all-groups --extra modal
uv run --frozen --extra modal pytest tests/unit/test_sana_dependency_contract.py -q
uv sync --frozen --all-groups
```

Expected: both test runs exit 0. The default run skips only the installed-Modal assertion and never requires Modal; the extra run exercises it at 1.5.4. The final default sync removes Modal again. `uv tree --depth 1` lists one resolved Torch/Diffusers/Transformers version each, and Torch 2.13.0 is paired exactly with Torchvision 0.28.0.

- [ ] **Step 7: Run the pre-existing core suite to catch shared-file regressions**

Run: `uv run pytest tests/codec tests/state/test_serialization.py -q && uv run pytest -q`

Expected: the canonical-CBOR and serialization byte contracts pass under cbor2 6.1.4; every core test still passes and only tests explicitly marked for CUDA, a real checkpoint, or paid Modal are skipped.

- [ ] **Step 8: Commit only the additive environment delta**

```bash
git add .gitignore pyproject.toml uv.lock src/ratemem/adapters/__init__.py src/ratemem/sana/__init__.py src/ratemem/support/__init__.py src/ratemem/pilot/__init__.py tests/unit/test_sana_dependency_contract.py
git commit -m "build: add locked sana pilot dependencies"
```

### Task 2: Implement the `DynamicAtomLinear` numerical contract

**Files:**
- Create: `src/ratemem/adapters/dynamic_atom_linear.py`
- Create: `tests/unit/test_dynamic_atom_linear.py`
- Create: `tests/contract/test_dynamic_atom_linear_contract.py`
- Create: `tests/contract/test_dynamic_atom_linear_cuda_memory.py`

- [ ] **Step 1: Write the complete validation, lifecycle, and numerical contract first**

The tests must cover an exact `nn.Linear` base (reject subclasses and `nn.LazyLinear` before touching their parameters), exact built-in positive integers for rank/count (including rejection of booleans and `int` subclasses), atom device/dtype inheritance, preservation and freezing of the original base object, and transient coefficients that never enter parameters, buffers, or `state_dict` even when the caller passes an `nn.Parameter`. Test nested-context rejection before inspecting the inner value, exception cleanup/reuse, no-context equivalence, exact zero behavior, bias/no-bias, global coefficients with all native Linear leading dimensions, and batched coefficients for inputs with a batch dimension. Invalid coefficient rank/width, scalar input, feature width, and coefficient batch size must fail before the base executes. Each activation also owns a unique transient token and records the coefficient tensor version: re-entering with the same tensor must not validate an older graph, and any in-place coefficient mutation from context entry through backward must fail closed.

```python
# tests/unit/test_dynamic_atom_linear.py
import torch
from torch import nn
from torch.nn import functional as F

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear


def _layer() -> DynamicAtomLinear:
    torch.manual_seed(7)
    return DynamicAtomLinear(nn.Linear(5, 7, bias=True), rank=2, atom_count=3)


def test_alpha_zero_exactly_matches_frozen_linear() -> None:
    layer = _layer()
    x = torch.randn(2, 4, 5)
    expected = layer.base(x)
    with layer.use_coefficients(torch.zeros(3)):
        actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_batch_one_matches_explicit_dense_weight() -> None:
    layer = _layer()
    x = torch.randn(1, 4, 5)
    alpha = torch.tensor([0.5, -0.25, 0.125])
    delta_weight = torch.einsum("a,aor,ari->oi", alpha, layer.atom_up, layer.atom_down)
    expected = F.linear(x, layer.base.weight + delta_weight, layer.base.bias)
    with layer.use_coefficients(alpha):
        actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_per_example_coefficients_match_explicit_dense_weights() -> None:
    layer = _layer()
    x = torch.randn(2, 4, 5)
    alpha = torch.tensor([[0.5, -0.25, 0.125], [-0.4, 0.2, 0.3]])
    expected = []
    for sample, sample_alpha in zip(x, alpha, strict=True):
        delta_weight = torch.einsum("a,aor,ari->oi", sample_alpha, layer.atom_up, layer.atom_down)
        expected.append(F.linear(sample, layer.base.weight + delta_weight, layer.base.bias))
    with layer.use_coefficients(alpha):
        actual = layer(x)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run the numerical tests and observe the missing class**

Run: `uv run pytest tests/unit/test_dynamic_atom_linear.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'ratemem.adapters.dynamic_atom_linear'`.

- [ ] **Step 3: Implement direct low-rank accumulation without a dense delta weight**

```python
# src/ratemem/adapters/dynamic_atom_linear.py
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_UNTRACKED_TENSOR_VERSION = -1


class DynamicAtomLinear(nn.Module):
    """Frozen linear layer plus dynamically weighted low-rank atoms."""

    _coefficients: Tensor | None
    _activation_token: object | None
    _coefficient_version: int | None

    def __init__(self, base: nn.Linear, *, rank: int, atom_count: int) -> None:
        super().__init__()
        if type(base) is not nn.Linear:
            raise TypeError("base must be an exact nn.Linear")
        if type(rank) is not int:
            raise TypeError("rank must be an int")
        if rank < 1:
            raise ValueError("rank must be positive")
        if type(atom_count) is not int:
            raise TypeError("atom_count must be an int")
        if atom_count < 1:
            raise ValueError("atom_count must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.atom_count = atom_count
        self.atom_down = nn.Parameter(
            base.weight.new_empty((atom_count, rank, base.in_features))
        )
        self.atom_up = nn.Parameter(
            base.weight.new_empty((atom_count, base.out_features, rank))
        )
        nn.init.normal_(self.atom_down, mean=0.0, std=0.01)
        nn.init.normal_(self.atom_up, mean=0.0, std=0.01)
        object.__setattr__(self, "_coefficients", None)
        object.__setattr__(self, "_activation_token", None)
        object.__setattr__(self, "_coefficient_version", None)

    @staticmethod
    def _tensor_version(coefficients: Tensor) -> int:
        try:
            return int(coefficients._version)
        except RuntimeError:
            if not coefficients.is_inference():
                raise
            return _UNTRACKED_TENSOR_VERSION

    @classmethod
    def _require_unmodified(cls, coefficients: Tensor, version: int) -> None:
        if version == _UNTRACKED_TENSOR_VERSION:
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "untracked inference coefficients require disabled gradients"
                )
            return
        if cls._tensor_version(coefficients) != version:
            raise RuntimeError("coefficients were modified in-place during activation")

    @contextmanager
    def use_coefficients(self, coefficients: Tensor) -> Iterator[None]:
        if self._coefficients is not None:
            raise RuntimeError("coefficients are already active")
        if not isinstance(coefficients, Tensor):
            raise TypeError("coefficients must be a Tensor")
        if coefficients.ndim not in (1, 2):
            raise ValueError("coefficients must be 1D or 2D")
        if coefficients.shape[-1] != self.atom_count:
            raise ValueError(f"coefficient atom dimension must be {self.atom_count}")
        activation_token = object()
        coefficient_version = self._tensor_version(coefficients)
        object.__setattr__(self, "_coefficients", coefficients)
        object.__setattr__(self, "_activation_token", activation_token)
        object.__setattr__(self, "_coefficient_version", coefficient_version)
        try:
            yield
        finally:
            object.__setattr__(self, "_coefficients", None)
            object.__setattr__(self, "_activation_token", None)
            object.__setattr__(self, "_coefficient_version", None)

    def _validate_input(self, x: Tensor, coefficients: Tensor | None) -> None:
        if not isinstance(x, Tensor):
            raise TypeError("input must be a Tensor")
        if x.ndim < 1:
            raise ValueError("input must have at least one dimension")
        if x.shape[-1] != self.base.in_features:
            raise ValueError(f"input feature dimension must be {self.base.in_features}")
        if coefficients is not None and coefficients.ndim == 2:
            if x.ndim < 2:
                raise ValueError("batched coefficients require input with a batch dimension")
            coefficient_batch = coefficients.shape[0]
            input_batch = x.shape[0]
            if coefficient_batch != input_batch:
                raise ValueError(
                    f"coefficient batch {coefficient_batch} does not match "
                    f"input batch {input_batch}"
                )

    def _guard_backward_context(
        self,
        output: Tensor,
        coefficients: Tensor,
        activation_token: object,
        coefficient_version: int,
    ) -> Tensor:
        if not output.requires_grad:
            return output

        def require_active_context(gradient: Tensor) -> Tensor:
            if (
                self._activation_token is not activation_token
                or self._coefficients is not coefficients
            ):
                raise RuntimeError("coefficient context must remain active through backward")
            self._require_unmodified(coefficients, coefficient_version)
            return gradient

        output.register_hook(require_active_context)  # type: ignore[no-untyped-call]
        return output

    def forward(self, x: Tensor) -> Tensor:
        coefficients = self._coefficients
        activation_token = self._activation_token
        coefficient_version = self._coefficient_version
        self._validate_input(x, coefficients)
        if coefficients is None:
            return self.base(x)
        if activation_token is None or coefficient_version is None:
            raise RuntimeError("coefficient activation state is inconsistent")
        self._require_unmodified(coefficients, coefficient_version)
        output: Tensor = self.base(x)
        dynamic = torch.zeros_like(output)
        for atom_index in range(self.atom_count):
            low_rank = F.linear(x, self.atom_down[atom_index])
            atom_output = F.linear(low_rank, self.atom_up[atom_index])
            scale = coefficients[atom_index] if coefficients.ndim == 1 else coefficients[:, atom_index]
            scale = scale.to(device=atom_output.device, dtype=atom_output.dtype)
            if scale.ndim == 1:
                scale = scale.reshape(scale.shape[0], *([1] * (atom_output.ndim - 1)))
            dynamic = dynamic + atom_output * scale
        self._require_unmodified(coefficients, coefficient_version)
        return self._guard_backward_context(
            output + dynamic,
            coefficients,
            activation_token,
            coefficient_version,
        )
```

- [ ] **Step 4: Run the numerical tests**

Run: `uv run pytest tests/unit/test_dynamic_atom_linear.py -q`

Expected: every constructor, context, shape, zero, no-context, and dense-equivalence case passes for global and batched coefficients without narrowing the base Linear leading-dimension semantics.

- [ ] **Step 5: Add registration, gradient, autocast, checkpoint, serialization, and allocation contracts**

```python
# tests/contract/test_dynamic_atom_linear_contract.py
import torch
from pytest import MonkeyPatch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear


def test_gradients_reach_code_and_both_atom_factors_but_not_base() -> None:
    layer = DynamicAtomLinear(nn.Linear(5, 7), rank=2, atom_count=3)
    x = torch.randn(2, 4, 5)
    alpha = torch.randn(2, 3, requires_grad=True)
    with layer.use_coefficients(alpha):
        layer(x).square().mean().backward()
    assert alpha.grad is not None and torch.count_nonzero(alpha.grad) > 0
    assert layer.atom_down.grad is not None and torch.count_nonzero(layer.atom_down.grad) > 0
    assert layer.atom_up.grad is not None and torch.count_nonzero(layer.atom_up.grad) > 0
    assert all(parameter.grad is None for parameter in layer.base.parameters())


def test_context_clears_coefficients_after_exception() -> None:
    layer = DynamicAtomLinear(nn.Linear(5, 7), rank=2, atom_count=3)
    try:
        with layer.use_coefficients(torch.ones(3)):
            raise RuntimeError("contract sentinel")
    except RuntimeError as exc:
        assert str(exc) == "contract sentinel"
    assert layer._coefficients is None


def test_coefficients_stay_active_during_checkpoint_recompute() -> None:
    layer = DynamicAtomLinear(nn.Linear(5, 7), rank=2, atom_count=3)
    x = torch.randn(2, 4, 5, requires_grad=True)
    alpha = torch.randn(2, 3, requires_grad=True)
    with layer.use_coefficients(alpha):
        checkpoint(layer, x, use_reentrant=False).sum().backward()
    assert alpha.grad is not None
```

Expand these representative tests into the exact contract in `tests/contract/test_dynamic_atom_linear_contract.py`: an active `nn.Parameter` coefficient plus its activation token/version must not appear in `named_parameters()`, `named_buffers()`, or `state_dict()`; strict state-dict roundtrip must preserve outputs, frozen base parameters, and trainable atoms. Gradients must reach `x`, the original coefficient leaf, and both atom factors but never the base. Under CPU BF16 autocast, FP32 input and FP32 coefficients must still produce BF16 output, with the cast gradient returning to the original FP32 coefficient leaf. Compare eager and `checkpoint(..., use_reentrant=False)` outputs and every gradient, and use a forward hook to prove forward and recomputation observe the same coefficient object. Backward after leaving the coefficient context must raise `coefficient context must remain active through backward`; leaving and re-entering with that same tensor still carries a different activation token and must not make the old graph valid. In-place mutation after entry (both before forward and between forward/backward) must raise `coefficients were modified in-place`, and every failure must leave a later fresh context usable. Coefficients created inside `torch.inference_mode()` have no version counter, so record an untracked sentinel: permit it only while gradients are disabled, preserving inference-only dynamic forwards, and fail closed before any grad-enabled forward.

```python
# tests/contract/test_dynamic_atom_linear_cuda_memory.py
from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear


WIDTH = 2240
DENSE_WEIGHT_BYTES = WIDTH * WIDTH * torch.bfloat16.itemsize
MINIMUM_GAP_BYTES = DENSE_WEIGHT_BYTES // 2


def _peak_bytes(callable_: Callable[[], torch.Tensor]) -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = callable_()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del result
    return peak


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contract runs in the Modal pilot")
def test_dynamic_path_has_a_repeatable_gap_below_explicit_dense_delta() -> None:
    base = nn.Linear(WIDTH, WIDTH, bias=False, device="cuda", dtype=torch.bfloat16)
    layer = DynamicAtomLinear(base, rank=4, atom_count=4)
    x = torch.randn(1, 128, WIDTH, device="cuda", dtype=torch.bfloat16)
    coefficients = torch.randn(4, device="cuda", dtype=torch.bfloat16)

    def dynamic() -> torch.Tensor:
        with layer.use_coefficients(coefficients):
            return layer(x)

    def explicit() -> torch.Tensor:
        delta = torch.einsum(
            "a,aor,ari->oi", coefficients, layer.atom_up, layer.atom_down
        )
        return F.linear(x, layer.base.weight + delta)

    dynamic()
    explicit()
    torch.cuda.synchronize()
    dynamic_peaks = [_peak_bytes(dynamic)]
    explicit_peaks = [_peak_bytes(explicit)]
    explicit_peaks.append(_peak_bytes(explicit))
    dynamic_peaks.append(_peak_bytes(dynamic))
    assert max(dynamic_peaks) + MINIMUM_GAP_BYTES <= min(explicit_peaks)
```

The CUDA contracts are marked `cuda` and are not run locally. One small contract passes a CPU coefficient leaf to a CUDA BF16 layer and proves both output placement and gradient return to the original CPU leaf. For the peak contract, warm both paths, synchronize before resetting and after each call, measure in dynamic/explicit and explicit/dynamic order, and require at least half one dense BF16 2240x2240 weight of separation so allocator noise cannot turn a one-byte comparison into a pass.

Add this deterministic allocation-shape test to `tests/contract/test_dynamic_atom_linear_contract.py`; the separate CUDA test supplies the required peak-memory instrumentation:

```python
def test_dynamic_path_never_passes_a_dense_delta_weight_to_linear(monkeypatch: MonkeyPatch) -> None:
    from torch.nn import functional as functional

    layer = DynamicAtomLinear(nn.Linear(5, 7), rank=2, atom_count=3)
    x = torch.randn(2, 4, 5)
    observed_shapes: list[tuple[int, ...]] = []
    original_linear = functional.linear

    def traced_linear(input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        observed_shapes.append(tuple(weight.shape))
        return original_linear(input_, weight, bias)

    monkeypatch.setattr(functional, "linear", traced_linear)
    with layer.use_coefficients(torch.ones(2, 3)):
        layer(x)
    assert observed_shapes.count((7, 5)) == 1
    assert observed_shapes.count((2, 5)) == 3
    assert observed_shapes.count((7, 2)) == 3
```

- [ ] **Step 6: Run all CPU contracts**

Run: `uv run pytest tests/unit/test_dynamic_atom_linear.py tests/contract/test_dynamic_atom_linear_contract.py -q`

Expected: all CPU tests pass; exact base validation rejects Linear subclasses/LazyLinear, direct/global/batched numerical cases match dense references, zero is exact, transient activation state never serializes, inference tensors work only for no-backward forwards, autocast does not promote BF16 output, eager/checkpoint outputs and gradients agree, old graphs cannot be revived by re-entering the same tensor, coefficient in-place mutation fails closed, no base parameter has a gradient, and the coefficient context is empty and reusable after every exit.

- [ ] **Step 7: Commit the dynamic linear contract**

```bash
git add src/ratemem/adapters/dynamic_atom_linear.py tests/unit/test_dynamic_atom_linear.py tests/contract/test_dynamic_atom_linear_contract.py tests/contract/test_dynamic_atom_linear_cuda_memory.py
git commit -m "feat: add dynamic atom linear contract"
```

### Task 3: Install dynamic atoms into the exact SANA q/k/v layout

**Files:**
- Create: `src/ratemem/adapters/sana_layout.py`
- Create: `tests/unit/test_sana_layout.py`
- Create: `tests/contract/test_sana_cpu_integration.py`
- Create: `tests/contract/test_sana_layout_cuda.py`

- [x] **Step 1: Write the complete canonical-layout and transaction RED suite**

Define one immutable contract shared with Task 4: `SANA_LAYOUT_VERSION = "sana-qkv-v1"`, `ATTENTION_KINDS = ("attn1", "attn2")`, and `TARGET_MODULES = ("to_q", "to_k", "to_v")`. The production layout is block-major/attention-major/projection-major: 20 blocks, 120 projections, a 480-value code for four atoms, 240 atom tensors, and exactly 8,601,600 trainable atom values at width 2240/rank 4.

The generic install tests must use a frozen, recursively-eval toy transformer and snapshot every module identity/training flag, parameter identity/requires-grad/device/dtype, and state tensor. Inventory the entire layout before construction, construct every wrapper before commit, and commit only after all validation passes. Missing/wrong final targets and an injected Nth constructor failure leave the snapshot unchanged. A module-level commit-helper fault that writes the current wrapper and then raises must restore every target, including the current one; a stronger fault that swaps the owner's live `_modules` dictionary, writes the wrapper there, and raises must also restore the captured registry identity and clear the displaced live dictionary. Adversarial `__setattr__` and qkv `__getattribute__` hooks that would mutate unrelated weights, buffers, and modes must never run. Reject root or deep-child train mode, any unfrozen parameter, exact-Linear violations, an existing wrapper/second install, and block/attention/Linear/Parameter/storage aliases. Require exact built-in `_modules` dictionaries and read the hierarchy only through direct canonical dictionary entries at the root `transformer_blocks`, each `ModuleList` index, each block attention, and each qkv target. At every segment, inspect raw instance dictionaries, `_parameters`, `_buffers`, and every MRO class namespace without invoking a descriptor; competing instance, Parameter, buffer, property, or descriptor registrations fail before construction. Canonical qkv modules and their weight/bias Parameters must have one module/Parameter/storage owner across the whole transformer, including non-qkv module, Parameter, and registered-buffer paths; empty buffers must not alias merely because their data pointer is zero. Generic install also validates positive `in_features`/`out_features`, exact weight `(out, in)` and bias `(out,)` shapes/numel, rejects meta/offloaded targets, mixed target device/dtype, and bias placement that differs from its weight.

- [x] **Step 2: Run RED before the module exists**

Run: `uv run pytest tests/unit/test_sana_layout.py tests/contract/test_sana_cpu_integration.py -q`

Expected: collection fails with `ModuleNotFoundError: ratemem.adapters.sana_layout`.

- [x] **Step 3: Implement inventory -> validate -> construct -> commit**

Keep the public APIs `SanaAdapterLayout`, `SanaDynamicAdapterBank`, and `install_sana_dynamic_atoms(transformer, rank, atom_count, expected_blocks)`. Require exact positive integer dimensions. Inventory canonical qkv paths and global ownership/placement first; require all transformer parameters frozen and every submodule already in eval. Construct detached `DynamicAtomLinear` wrappers and call `eval()` on them before mutation. Commit in canonical order through one module-level CAS helper: pass the captured exact registry explicitly, confirm through the raw owner instance dictionary that it is still the live `_modules` object and still maps the attribute to the expected base, then perform one direct dictionary assignment. Never call owner `__setattr__`. Include the failed current target in rollback, clear every attempted target from both the displaced live registry and captured registry, and restore each owner's captured `_modules` identity on any commit or Bank-construction exception.

`SanaDynamicAdapterBank` is deliberately not an `nn.Module`, and its public constructor remains exactly `SanaDynamicAdapterBank(layout, wrappers)`: direct callers receive the original non-owning weak-wrapper controller semantics. The installer alone uses a private factory to bind the resulting Bank to weak references for the root transformer, each canonical owner, and each wrapper. Keeping either kind of Bank alone must not keep wrappers or transformer owners alive. Every entry point on an installer-bound Bank resolves the complete direct `_modules` path from the weak root, repeats the static instance/Parameter/buffer/MRO shadow checks, and confirms the owner/attribute/wrapper binding; target, attention, block swaps, or newly-added shadows fail closed before touching state. `parameters()`/`named_parameters()` yield each atom exactly once under canonical transformer paths. `state_dict()` is trainable-only and contains no `base.*`, positional `wrappers.*`, or frozen duplicates. Before `load_state_dict()` mutates any target, validate all supplied entries and independently materialize every value with `detach().to(target device/dtype).clone()`; then copy transactionally and restore every original value on an injected mid-copy failure. This makes alias swaps exact, prevents a later meta/materialization error from changing an earlier parameter or version counter, and leaves independent source tensors and gradients untouched. The installed transformer's ordinary state dict remains the full strict-roundtrip boundary.

- [x] **Step 4: Implement all-or-nothing scoped activation**

Accept only 1D `[code_dim]` and 2D `[batch, code_dim]` tensors. Prevalidate code shape and confirm every wrapper's coefficient/token/version transient tuple is inactive before entering any context. Reshape to `[projection, atom]` or `[batch, projection, atom]` and map slices in canonical order. Snapshot that exact tuple before each `__enter__`; immediately after every successful enter require the exact coefficient view identity, a token, and the tensor's real version. If an intermediate enter mutates the current wrapper and then raises, close earlier contexts and restore every attempted tuple in reverse order. Nested activation and a genuinely externally-active middle wrapper must fail before touching another wrapper and must preserve all three external fields exactly. Validate active identities and versions in a `finally` that runs even while a body exception is propagating, then use an outer `finally` to restore every attempted pre-state after successful exit, enter failure, body failure, or a context exit that poisons state and raises. Thus body/exit corruption is surfaced rather than silently hidden, while the Bank remains reusable.

Use exhaustive arange codes for both global and batched mapping, and backpropagate through the views to prove every input code value receives exactly one gradient contribution.

- [x] **Step 5: Separate the fixed production validator from the tiny installer**

`validate_production_sana_layout(...)` enforces rank 4, four atoms, 20 blocks, each target's `in_features == out_features == 2240`, all 120 weights exactly `[2240, 2240]`, no bias on `attn1`, an exact `[2240]`/2240-value bias on `attn2`, 240 atom tensors, and 8,601,600 atom values. CPU structural tests may use uniform BF16 meta modules with `require_cuda_bfloat16=False`; the default real contract also requires every weight and bias to be CUDA BF16. Keep that real randomized allocation test behind both the `cuda` marker and `RATEMEM_RUN_PRODUCTION_SANA_LAYOUT=1`. Do not hardcode production width/device into the generic two-block Diffusers test path.

- [x] **Step 6: Prove Diffusers 0.40 integration without a checkpoint**

Use a two-block randomized `SanaTransformer2DModel`, freeze it, recursively set eval, and install twelve wrappers. Assert zero code is bit-exact to the pre-install transformer, non-qkv modules keep identity, batched codes equal independent per-example forwards, and code plus both atom factors on every wrapper receive gradients while every base remains grad-free. Strictly load the full installed transformer state into a separately installed model, assert all 24 canonical Bank keys and values exactly, and compare activated outputs bit-for-bit. This suite must not access the network or load a real checkpoint.

- [x] **Step 7: Run the complete Task 3 and free verification suites**

Run:

```bash
uv run pytest tests/unit/test_sana_layout.py tests/contract/test_sana_cpu_integration.py -q
uv run pytest -m "not cuda and not real_sana and not paid_modal" -q
uv run ruff check src tests
uv run mypy src
```

Expected: all Task 3 CPU tests and all free tests pass; CUDA production validation remains collected but opt-in only.

- [x] **Step 8: Commit the SANA adapter layout**

```bash
git add src/ratemem/adapters/sana_layout.py tests/unit/test_sana_layout.py tests/contract/test_sana_cpu_integration.py tests/contract/test_sana_layout_cuda.py docs/superpowers/plans/2026-08-24-ratemem-sana-modal-pilot.md
git commit -m "fix: close sana adapter transaction gaps"
```

### Task 4: Lock and load the exact SANA and support checkpoints

**Files:**
- Create: `configs/pilot/sana-1.5-1.6b.json`
- Create: `src/ratemem/pilot/config.py`
- Create: `src/ratemem/sana/components.py`
- Create: `tests/unit/test_pilot_config.py`
- Create: `tests/unit/test_sana_components.py`
- Create: `tests/integration/test_real_sana_checkpoint.py`
- Modify: `docs/superpowers/plans/2026-08-24-ratemem-sana-modal-pilot.md` (Task 7 scheduler handoff)

- [x] **Step 1: RED — specify an exact immutable pilot config**

Pin schema `1.0.0`, SANA
`Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers@b77948f2b4eed5c728e9b828ccff07f7427b43cc`,
and DINOv2
`facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`.
The training object additionally pins
`FlowMatchEulerDiscreteScheduler`, `num_train_timesteps=1000`,
`flow_shift=1.0`, and `use_dynamic_shifting=false`.

The RED suite must reject duplicate JSON keys, NaN and both infinities, non-object
root/nested values, reordered keys, every changed leaf, and Python equal-value
type substitutions such as `20.0`, `1`, and `0` for canonical integer,
float, and boolean leaves. It must also reject coordinated plausible drift such
as 10 blocks with 8 atoms or width 1120 with rank 8 even when derived totals are
preserved.

- [x] **Step 2: GREEN — implement one canonical config boundary**

`SanaPilotConfig.load()` reads UTF-8 and uses
`object_pairs_hook` plus `parse_constant`. The frozen, slotted dataclass
revalidates in `__post_init__`; its public `validate()` reconstructs the
canonical payload with exact built-in types, values, and order. Both direct
construction and `dataclasses.replace()` therefore remain fail-closed, and
hydration/loading revalidate an exact `SanaPilotConfig` before I/O.

Derived values are fixed at:

```text
code_shape             = (20, 2, 3, 4)
projection_count       = 120
code_dim                = 480
atom_tensor_count       = 240
atom_parameter_count    = 8,601,600
```

RED evidence: collection initially failed because `ratemem.pilot.config` did
not exist. GREEN command:

```bash
uv run pytest tests/unit/test_pilot_config.py -q
```

- [x] **Step 3: RED — specify a two-phase, allowlisted Hub boundary**

The exact required file tuples are:

```python
SANA_FILES = (
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
DINO_FILES = (
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
)
```

There are no wildcards, Python files, pickle weights, or moving revisions.
For Transformers 5.16.1, the explicit `GemmaTokenizer` consumes
`tokenizer.json`; the upstream `tokenizer.model` and
`special_tokens_map.json` are optional and deliberately excluded. Root model
metadata and DINO `pytorch_model.bin` are also unnecessary and excluded.

Nine behavior-control files additionally have committed SHA-256 manifests:

| Snapshot file | SHA-256 |
|---|---|
| `scheduler/scheduler_config.json` | `f9256042828841b26561487c7e0c33fff8717e98ac0fef5c1f6d05bfdd66e908` |
| `transformer/config.json` | `70863bf60b87cbeab5780c9827ffc5b880cd1ec9ce22bf033409b7e257e8fc68` |
| `vae/config.json` | `ba6f3d3e44d75d44fdd3760097c069173b5b925e6d14604d5d3582628d09cca6` |
| `text_encoder/config.json` | `733f241a6692770dfba10383e2c5a56a4f88b320732d9ee8fa16118737eca84d` |
| `text_encoder/model.safetensors.index.json` | `92764588f700e36874c52f9f05bba143857e5069fc69b14450f907a1cdf879ed` |
| `tokenizer/tokenizer_config.json` | `cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89` |
| `tokenizer/tokenizer.json` | `5f7eee611703c5ce5d1eee32d9cdcfe465647b8aff0c1dfb3bed7ad7dbb05060` |
| DINO `config.json` | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| DINO `preprocessor_config.json` | `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828` |

Hydration verifies each repository's control hashes before the next download;
offline loading repeats every hash before the first loader and after the final
loader. This protects the
model/tokenizer/processor behavior and the sharded text weight map even when a
local directory has the expected SHA basename. Large safetensors remain under
the fixed public Hub revision, Hugging Face cache-integrity, and safetensors
loader trust boundary rather than being rehashed by this application. The
private HF cache trust model excludes concurrent malicious writers; the second
control-file pass narrows accidental or non-malicious time-of-check/time-of-use
drift but is not a filesystem isolation primitive.

`hydrate_pinned_snapshots()` is the sole network-enabled function. It makes
exactly two `snapshot_download` calls with `repo_type="model"`, the full
revision, exact allowlist, `token=False`, `local_files_only=False`, and
`force_download=False`. It validates and strict-resolves the SANA result
before making the DINO request, verifies the resolved basename is the full SHA,
and checks every required file. A missing/wrong SANA result therefore causes
only one network call; a SHA-named symlink to another revision is rejected.

- [x] **Step 4: GREEN — implement explicit offline component loading**

`load_pinned_components()` accepts only an exact, revalidated config and exact
`PinnedSnapshotPaths`. It strict-resolves and fully validates both snapshots
before the first loader. Every loader receives a local snapshot path,
`local_files_only=True`, `token=False`, and `force_download=False`.

Only these concrete classes may load:

- `DPMSolverMultistepScheduler`
- `SanaTransformer2DModel`
- `AutoencoderDC`
- `GemmaTokenizer`
- `Gemma2Model`
- `BitImageProcessor`
- `Dinov2Model`

All weight-bearing calls use `use_safetensors=True`; Transformers model calls
also use `weights_only=True` and `trust_remote_code=False`. The tokenizer
uses `trust_remote_code=False`; `BitImageProcessor` does not receive that
unsupported kwarg. SANA transformer/text load as BF16, while VAE/DINO load as
FP32. No Auto class, `DiffusionPipeline`, pipeline `from_pretrained`,
`custom_pipeline`, `hf_hub_download`, requests/httpx/urllib fallback, or
dynamic remote code is allowed.

The normalized semantic surface excludes only provenance/version metadata.
Transformer validation covers every Diffusers 0.40 constructor field that can
change its forward pass, including the fixed raw fields plus the new
`guidance_embeds_scale=0.1` and `timestep_scale=1.0` defaults. VAE validation
covers its complete fixed block/channel/layer/qkv/up/down/norm/activation
architecture plus the 0.40 shortcut and convolution-activation defaults. The
inference DPM config covers all fixed solver/beta/threshold/final/spacing/
variance/offset/rescale flags and the 0.40 dynamic/time-shift defaults.
Gemma, tokenizer, DINO, and Bit processor core fields are also validated with
exact runtime types. This includes Gemma special-token IDs and left padding,
plus Bit processor RGB/resize/crop, ImageNet mean/std, `resample=3`, and the
Transformers 5.16.1 normalized `SizeDict`/tuple forms. The control-file hashes
close semantic fields not duplicated as normalized assertions.

- [x] **Step 5: Pin the training FlowMatch constructor and immutable arrays**

Training never hydrates a scheduler config and never calls `set_timesteps`.
Construct it directly:

```python
training_scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000,
    shift=1.0,
    use_dynamic_shifting=False,
)
```

Canonical immutable arrays use the scheduler's fixed float32 construction:

```python
sigmas = torch.linspace(1, 1000, 1000, dtype=torch.float32).flip(0) / 1000
timesteps = sigmas * 1000
training_timesteps = tuple(float(value) for value in timesteps)
training_sigmas = tuple(float(value) for value in sigmas)
```

The scheduler tensors must remain one-dimensional float32 CPU tensors and both
tensor values and saved tuples must equal these arrays at every index. This
detects synchronized mid-array mutation rather than checking endpoints only.
Task 7 receives these immutable tuples and copies them to the training device
without regenerating scheduler state.

- [x] **Step 6: Build and validate the frozen bundle**

`PinnedComponents` has exact concrete field types. Construction validates
every pinned component/config, freezes all four modules, recursively sets eval,
and checks one shared device with transformer/text BF16 and VAE/DINO FP32.
`inference_pipeline()` directly constructs exactly one `SanaPipeline` from
the five already-loaded objects and verifies every object identity.

Unit tests patch every loader and the socket/network boundary. They prove all
kwargs, exact returned-class rejection, config/snapshot revalidation before
network or loaders, missing-file failure before any loader, exact config leaf
and equal-type tamper rejection, immutable scheduler arrays, and the AST
prohibitions above. Free/local unit verification performs **no network** access;
hydration is the sole separately invoked explicit network boundary.

- [x] **Step 7: Add the explicit paid real-checkpoint test**

The `real_sana + paid_modal + cuda` integration test is locally skipped unless
`RATEMEM_RUN_REAL_SANA=1` and CUDA are both present. Only after that guard does
it hydrate the two fixed revisions, then load solely from the validated local
paths. It repeats the complete pinned semantic surface, class, scheduler,
placement, dtype, frozen/eval, processor, and tokenizer checks; validates the production
20-block SANA layout; installs 120 wrappers and 240 unique BF16 CUDA atom
tensors; verifies exactly 8,601,600 trainables and Bank/trainable ID equality;
and checks exact `SanaPipeline` object identity. Normal local verification
must not opt in and must not download a checkpoint.

- [x] **Step 8: Run Task 4 and free verification**

```bash
uv run pytest tests/unit/test_pilot_config.py tests/unit/test_sana_components.py \
  tests/integration/test_real_sana_checkpoint.py -q
uv run pytest -m "not cuda and not real_sana and not paid_modal" -q
uv run ruff check src tests
uv run mypy src
```

Expected: config/component tests pass, the paid real test reports one skip, all
free tests and static checks pass, and no local model download or Modal call
occurs.

- [x] **Step 9: Commit the pinned offline boundary**

```bash
git add configs/pilot/sana-1.5-1.6b.json \
  src/ratemem/pilot/config.py src/ratemem/sana/components.py \
  tests/unit/test_pilot_config.py tests/unit/test_sana_components.py \
  tests/integration/test_real_sana_checkpoint.py \
  docs/superpowers/plans/2026-08-24-ratemem-sana-modal-pilot.md
git commit -m "feat: pin and load sana components offline"
```

### Task 5: Build the frozen support feature path and permutation-invariant amortizer

**Files:**
- Create: `src/ratemem/support/features.py`
- Create: `src/ratemem/support/amortizer.py`
- Create: `tests/unit/test_support_features.py`
- Create: `tests/unit/test_support_amortizer.py`

> **Implementation reconciliation (complete):** The snippets below are the
> original RED sketch; the executable 101-test contract is normative. The
> final implementation adds an immutable, hashed amortizer architecture
> identity; complete behavioral-topology validation; exact FP32/device/
> trainability/finite-state checks; coefficients-only and tiny-SANA Bank
> gradient proofs; explicit permutation-invariant multiset semantics; and
> inference-safe cache tensors. Frozen DINO encoding snapshots recursive
> modules, parameters, buffers, storage, tensor versions, values, and mode
> before any processor/config getter, then revalidates in `finally` on both
> success and failure. A real local `BitImageProcessor` normalization path is
> covered without network access. Identity-based topology validation
> deliberately supports reconstruction plus strict state loading rather than
> deepcopy/full-module pickle interchange.

- [x] **Step 1: Write support-feature and set-order contracts**

```python
# tests/unit/test_support_amortizer.py
import torch

from ratemem.support.amortizer import SupportAmortizer


def amortizer() -> SupportAmortizer:
    torch.manual_seed(13)
    return SupportAmortizer(
        support_dim=384,
        description_dim=2304,
        hidden_dim=256,
        projection_count=120,
        atom_count=4,
        layers=2,
        heads=8,
    ).eval()


def test_support_order_is_permutation_invariant() -> None:
    model = amortizer()
    support = torch.randn(2, 3, 384)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    descriptions = torch.randn(2, 2304)
    permutation = torch.tensor([1, 0, 2])
    first = model(support, mask, descriptions).coefficients
    second = model(support[:, permutation], mask[:, permutation], descriptions).coefficients
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_prediction_contract_is_fp32_bounded_and_differentiable() -> None:
    model = amortizer().train()
    support = torch.randn(2, 2, 384, requires_grad=True)
    mask = torch.ones(2, 2, dtype=torch.bool)
    descriptions = torch.randn(2, 2304)
    prediction = model(support, mask, descriptions)
    assert prediction.logits.shape == prediction.coefficients.shape == (2, 120, 4)
    assert prediction.logits.dtype == prediction.coefficients.dtype == torch.float32
    assert prediction.scales.shape == (120, 1)
    assert torch.all(prediction.scales > 0)
    assert torch.all(prediction.coefficients.abs() <= prediction.scales.unsqueeze(0))
    prediction.coefficients.square().mean().backward()
    assert support.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_empty_support_set_is_rejected() -> None:
    model = amortizer()
    support = torch.randn(1, 2, 384)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    descriptions = torch.randn(1, 2304)
    try:
        model(support, mask, descriptions)
    except ValueError as exc:
        assert str(exc) == "each concept requires at least one support image"
    else:
        raise AssertionError("empty support set was accepted")
```

```python
# tests/unit/test_support_features.py
import torch
from torch import nn

from ratemem.support.features import masked_mean_description, verify_frozen_encoder


def test_description_pool_uses_only_unmasked_tokens() -> None:
    tokens = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    torch.testing.assert_close(masked_mean_description(tokens, mask), torch.tensor([[3.0, 5.0]]))


def test_frozen_encoder_contract_rejects_trainable_weight() -> None:
    encoder = nn.Linear(3, 2)
    try:
        verify_frozen_encoder(encoder)
    except RuntimeError as exc:
        assert "requires_grad" in str(exc)
    else:
        raise AssertionError("trainable support encoder was accepted")
```

- [x] **Step 2: Run the support tests and observe missing modules**

Run: `uv run pytest tests/unit/test_support_amortizer.py tests/unit/test_support_features.py -q`

Expected: collection fails because the support modules do not exist.

- [x] **Step 3: Implement frozen image/description feature helpers**

```python
# src/ratemem/support/features.py
from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from torch import Tensor, nn


def verify_frozen_encoder(module: nn.Module) -> None:
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError("support encoder parameter has requires_grad=True")
    if module.training:
        raise RuntimeError("support encoder must be in eval mode")


@torch.inference_mode()
def encode_support_images(
    images: Sequence[Image.Image], *, processor: object, encoder: nn.Module, device: torch.device
) -> Tensor:
    verify_frozen_encoder(encoder)
    processed = processor(images=list(images), return_tensors="pt")  # type: ignore[operator]
    pixel_values = processed["pixel_values"].to(device=device, dtype=torch.float32)
    output = encoder(pixel_values=pixel_values)
    return output.last_hidden_state[:, 0].float().cpu()


def masked_mean_description(token_features: Tensor, attention_mask: Tensor) -> Tensor:
    weights = attention_mask.to(token_features.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (token_features.float() * weights.float()).sum(dim=1) / denominator.float()
```

- [x] **Step 4: Implement the Set Transformer amortizer with positive scales**

```python
# src/ratemem/support/amortizer.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AdapterPrediction:
    logits: Tensor
    scales: Tensor
    coefficients: Tensor


class SupportAmortizer(nn.Module):
    def __init__(
        self,
        *,
        support_dim: int,
        description_dim: int,
        hidden_dim: int,
        projection_count: int,
        atom_count: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.projection_count = projection_count
        self.atom_count = atom_count
        self.support_projection = nn.Linear(support_dim, hidden_dim)
        self.description_projection = nn.Linear(description_dim, hidden_dim)
        self.support_type = nn.Parameter(torch.zeros(hidden_dim))
        self.description_type = nn.Parameter(torch.zeros(hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool = nn.MultiheadAttention(hidden_dim, heads, dropout=0.0, batch_first=True)
        self.head = nn.Linear(hidden_dim, projection_count * atom_count)
        self.raw_projection_scale = nn.Parameter(torch.zeros(projection_count, 1))

    def forward(self, support_features: Tensor, support_mask: Tensor, description_features: Tensor) -> AdapterPrediction:
        if support_features.ndim != 3 or support_mask.shape != support_features.shape[:2]:
            raise ValueError("support features and mask have incompatible shapes")
        if torch.any(support_mask.sum(dim=1) == 0):
            raise ValueError("each concept requires at least one support image")
        support_tokens = self.support_projection(support_features.float()) + self.support_type
        description_token = self.description_projection(description_features.float()).unsqueeze(1) + self.description_type
        tokens = torch.cat((support_tokens, description_token), dim=1)
        token_mask = torch.cat(
            (support_mask, torch.ones(support_mask.shape[0], 1, dtype=torch.bool, device=support_mask.device)), dim=1
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        query = self.pool_query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=~token_mask, need_weights=False)
        logits = self.head(pooled[:, 0].float()).reshape(-1, self.projection_count, self.atom_count).float()
        scales = F.softplus(self.raw_projection_scale.float()) + 1e-6
        coefficients = torch.tanh(logits) * scales.unsqueeze(0)
        return AdapterPrediction(logits=logits, scales=scales, coefficients=coefficients)
```

- [x] **Step 5: Run the support contracts**

Run: `uv run pytest tests/unit/test_support_amortizer.py tests/unit/test_support_features.py -q`

Expected: `101 passed`; permuting valid support tokens leaves coefficients unchanged within tolerance.

- [x] **Step 6: Commit the support-to-code path**

```bash
git add src/ratemem/support/features.py src/ratemem/support/amortizer.py tests/unit/test_support_features.py tests/unit/test_support_amortizer.py docs/superpowers/plans/2026-08-24-ratemem-sana-modal-pilot.md
git commit -m "feat: add permutation invariant support amortizer"
```

### Task 6: Add the deterministic held-in Subjects200K pilot loader and cache

**Files:**
- Create: `configs/pilot/subjects200k-held-in.json`
- Modify: `src/ratemem/pilot/config.py`
- Create: `src/ratemem/pilot/data.py`
- Create: `tests/unit/test_pilot_data.py`

> **Implementation reconciliation (complete):** The snippets below are the
> original RED sketch; the executable 154-test Task 6 contract is normative.
> The final implementation pins the exact Subjects200K revision, shard LFS
> identity, schema order, rows 0--7, composite geometry, and official SANA/DINO
> preprocessing. Production precompute/build entry points accept only an exact,
> validated `PinnedComponents`; synthetic components are confined to private
> test seams. A canonical config hash and an externally retained
> `PilotCacheReceipt` bind the identity, canonical manifest bytes, and exact
> safetensors bytes, so a coherently rewritten feature/manifest/marker bundle
> cannot authenticate itself. Cache publication is owner-only, single-writer,
> fsynced, and atomic with complete failure cleanup; thread and spawned-process
> tests prove one compute winner. Real Subjects hydration and the complete
> SANA/DINO precompute path are explicit opt-in integration contracts. These
> eight public training rows are an engineering smoke set only: they are held-in,
> publication-ineligible, and provide no scientific or CVPR claim evidence. The
> pinned streaming Parquet path explicitly closes an early-stopped iterator and
> executes the dependency's five-second I/O-thread shutdown barrier; the real
> hydration integration must exit cleanly, not merely pass before interpreter
> teardown.

- [x] **Step 1: Write composite-split, manifest, and held-in-only tests**

```python
# tests/unit/test_pilot_data.py
from pathlib import Path

from PIL import Image
import pytest

from ratemem.pilot.config import SubjectsPilotConfig
from ratemem.pilot.data import build_example, split_composite_pair


def test_composite_pair_uses_official_subjects200k_crop_boxes() -> None:
    image = Image.new("RGB", (1056, 528), "black")
    image.paste(Image.new("RGB", (512, 512), "red"), (8, 8))
    image.paste(Image.new("RGB", (512, 512), "blue"), (528, 8))
    support, query = split_composite_pair(image, image_size=512, padding_pixels=8)
    assert support.size == query.size == (512, 512)
    assert support.getpixel((10, 10)) == (255, 0, 0)
    assert query.getpixel((10, 10)) == (0, 0, 255)


def test_composite_pair_rejects_dimensions_outside_pinned_contract() -> None:
    with pytest.raises(ValueError, match="1056x528"):
        split_composite_pair(
            Image.new("RGB", (1055, 528), "white"), image_size=512, padding_pixels=8
        )


def test_example_hash_is_deterministic() -> None:
    row = {
        "image": Image.new("RGB", (1056, 528), "white"),
        "collection": "collection_1",
        "description": {"item": "chair", "description_0": "chair in a room", "description_1": "chair outside"},
    }
    first = build_example(3, row, image_size=512, padding_pixels=8)
    second = build_example(3, row, image_size=512, padding_pixels=8)
    assert first.sha256 == second.sha256
    assert first.concept_description == "chair"
    assert first.query_prompt == "chair outside"


def test_committed_dataset_config_is_engineering_only() -> None:
    config = SubjectsPilotConfig.load(Path("configs/pilot/subjects200k-held-in.json"))
    assert config.revision == "0d1cf6536239888f1a8e218790649344810067bc"
    assert config.row_indices == tuple(range(8))
    assert config.split == "train"
    assert config.image_size == 512 and config.padding_pixels == 8
    assert config.held_in is True
    assert config.publication_eligible is False
```

- [x] **Step 2: Run the data tests and observe missing configuration/data code**

Run: `uv run pytest tests/unit/test_pilot_data.py -q`

Expected: collection fails because `SubjectsPilotConfig` and `ratemem.pilot.data` do not exist.

- [x] **Step 3: Commit the exact held-in data selection**

```json
{
  "schema_version": "1.0.0",
  "dataset_id": "Yuanshi/Subjects200K",
  "revision": "0d1cf6536239888f1a8e218790649344810067bc",
  "split": "train",
  "streaming": true,
  "row_indices": [0, 1, 2, 3, 4, 5, 6, 7],
  "image_size": 512,
  "padding_pixels": 8,
  "held_in": true,
  "publication_eligible": false,
  "license": "apache-2.0"
}
```

- [x] **Step 4: Add `SubjectsPilotConfig` to the central config module**

```python
# append to src/ratemem/pilot/config.py
@dataclass(frozen=True)
class SubjectsPilotConfig:
    dataset_id: str
    revision: str
    split: str
    streaming: bool
    row_indices: tuple[int, ...]
    image_size: int
    padding_pixels: int
    held_in: bool
    publication_eligible: bool
    license: str

    @classmethod
    def load(cls, path: Path) -> "SubjectsPilotConfig":
        payload = json.loads(path.read_text())
        _exact_keys(
            payload,
            {"schema_version", "dataset_id", "revision", "split", "streaming", "row_indices", "image_size", "padding_pixels", "held_in", "publication_eligible", "license"},
            "Subjects200K pilot config",
        )
        if not COMMIT_PATTERN.fullmatch(payload["revision"]):
            raise ValueError("dataset revision must be a 40-character lowercase commit")
        config = cls(
            dataset_id=payload["dataset_id"], revision=payload["revision"], split=payload["split"],
            streaming=payload["streaming"], row_indices=tuple(payload["row_indices"]),
            image_size=payload["image_size"], padding_pixels=payload["padding_pixels"],
            held_in=payload["held_in"],
            publication_eligible=payload["publication_eligible"], license=payload["license"],
        )
        if (
            config.row_indices != tuple(range(8))
            or config.image_size != 512
            or config.padding_pixels != 8
            or not config.held_in
            or config.publication_eligible
        ):
            raise ValueError("pilot data must remain the locked held-in rows 0 through 7")
        return config
```

- [x] **Step 5: Implement deterministic row extraction and a checksummed precompute cache**

```python
# src/ratemem/pilot/data.py (public contract and core extraction)
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from torch import Tensor

from ratemem.pilot.config import SanaPilotConfig, SubjectsPilotConfig
from ratemem.sana.components import PinnedComponents
from ratemem.support.features import encode_support_images, masked_mean_description


@dataclass(frozen=True)
class PilotExample:
    row_index: int
    support_image: Image.Image
    query_image: Image.Image
    concept_description: str
    query_prompt: str
    sha256: str


def split_composite_pair(
    image: Image.Image, *, image_size: int, padding_pixels: int
) -> tuple[Image.Image, Image.Image]:
    expected_size = (2 * image_size + 4 * padding_pixels, image_size + 2 * padding_pixels)
    if image.size != expected_size:
        raise ValueError(
            f"Subjects200K composite must be {expected_size[0]}x{expected_size[1]}, got {image.width}x{image.height}"
        )
    left_box = (padding_pixels, padding_pixels, padding_pixels + image_size, padding_pixels + image_size)
    right_left = image_size + 2 * padding_pixels
    right_box = (right_left, padding_pixels, right_left + image_size, padding_pixels + image_size)
    support = image.crop(left_box).convert("RGB")
    query = image.crop(right_box).convert("RGB")
    return support, query


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def build_example(
    row_index: int, row: dict[str, Any], *, image_size: int, padding_pixels: int
) -> PilotExample:
    support, query = split_composite_pair(
        row["image"], image_size=image_size, padding_pixels=padding_pixels
    )
    description = row["description"]
    concept = str(description["item"])
    prompt = str(description["description_1"])
    digest = hashlib.sha256()
    digest.update(str(row_index).encode())
    digest.update(_png_bytes(support))
    digest.update(_png_bytes(query))
    digest.update(json.dumps({"concept": concept, "prompt": prompt}, sort_keys=True).encode())
    return PilotExample(row_index, support, query, concept, prompt, digest.hexdigest())


def stream_locked_examples(config: SubjectsPilotConfig) -> list[PilotExample]:
    rows = load_dataset(
        config.dataset_id,
        revision=config.revision,
        split=config.split,
        streaming=True,
    )
    wanted = set(config.row_indices)
    examples: list[PilotExample] = []
    for row_index, row in enumerate(rows):
        if row_index in wanted:
            examples.append(
                build_example(
                    row_index,
                    row,
                    image_size=config.image_size,
                    padding_pixels=config.padding_pixels,
                )
            )
        if row_index >= max(wanted):
            break
    if tuple(example.row_index for example in examples) != config.row_indices:
        raise RuntimeError("stream did not return every locked held-in row")
    return examples


def save_precomputed_cache(path: Path, tensors: dict[str, torch.Tensor], manifest: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}, path / "features.safetensors")
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class PrecomputedPilotData:
    root: Path
    tensors: dict[str, Tensor]
    manifest: dict[str, Any]


def _query_pixels(image: Image.Image, resolution: int) -> Tensor:
    fitted = ImageOps.fit(
        image.convert("RGB"),
        (resolution, resolution),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    array = torch.frombuffer(bytearray(fitted.tobytes()), dtype=torch.uint8)
    pixels = array.reshape(resolution, resolution, 3).permute(2, 0, 1).float()
    return pixels.div(127.5).sub(1.0)


@torch.inference_mode()
def _encode_text(
    texts: list[str], components: PinnedComponents, *, max_length: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    tokenized = components.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    hidden = components.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).last_hidden_state
    return hidden.float().cpu(), attention_mask.cpu()


def _manifest_identity(
    examples: list[PilotExample], sana_config: SanaPilotConfig, dataset_config: SubjectsPilotConfig
) -> dict[str, Any]:
    identity = {
        "schema_version": "1.0.0",
        "sana_revision": sana_config.revision,
        "support_revision": sana_config.support_revision,
        "dataset_revision": dataset_config.revision,
        "row_indices": [example.row_index for example in examples],
        "row_sha256": [example.sha256 for example in examples],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return identity | {"manifest_sha256": hashlib.sha256(canonical).hexdigest()}


@torch.inference_mode()
def precompute_examples(
    examples: list[PilotExample],
    components: PinnedComponents,
    output_dir: Path,
    sana_config: SanaPilotConfig,
    dataset_config: SubjectsPilotConfig,
) -> PrecomputedPilotData:
    manifest = _manifest_identity(examples, sana_config, dataset_config)
    if output_dir.exists():
        existing = json.loads((output_dir / "manifest.json").read_text())
        if existing != manifest:
            raise ValueError("existing pilot cache does not match immutable revisions or row hashes")
        tensors = load_file(output_dir / "features.safetensors", device="cpu")
        return PrecomputedPilotData(output_dir, tensors, existing)

    vae_device = next(components.vae.parameters()).device
    support_device = next(components.support_encoder.parameters()).device
    text_device = next(components.text_encoder.parameters()).device
    query_pixels = torch.stack([
        _query_pixels(example.query_image, sana_config.resolution) for example in examples
    ])
    latent_output = components.vae.encode(query_pixels.to(vae_device, dtype=torch.float32))
    clean_latents = latent_output.latent.float() * float(components.vae.config.scaling_factor)
    prompt_embeddings, prompt_attention_mask = _encode_text(
        [example.query_prompt for example in examples],
        components,
        max_length=sana_config.max_sequence_length,
        device=text_device,
    )
    description_tokens, description_mask = _encode_text(
        [example.concept_description for example in examples],
        components,
        max_length=sana_config.max_sequence_length,
        device=text_device,
    )
    support_features = encode_support_images(
        [example.support_image for example in examples],
        processor=components.support_processor,
        encoder=components.support_encoder,
        device=support_device,
    ).unsqueeze(1)
    tensors = {
        "clean_latents": clean_latents.cpu(),
        "prompt_embeddings": prompt_embeddings,
        "prompt_attention_mask": prompt_attention_mask,
        "support_features": support_features,
        "support_mask": torch.ones(len(examples), 1, dtype=torch.bool),
        "description_features": masked_mean_description(description_tokens, description_mask),
    }
    save_precomputed_cache(output_dir, tensors, manifest)
    return PrecomputedPilotData(output_dir, tensors, manifest)
```

The training loop consumes only `PrecomputedPilotData.tensors`; it never receives a PIL image, VAE, text encoder, or DINO encoder after this function returns.

- [x] **Step 6: Run the deterministic data tests**

Run: `uv run pytest tests/unit/test_pilot_data.py -q`

Expected: `4 passed`; no network access occurs because tests use synthetic images.

- [x] **Step 7: Commit the held-in pilot data path**

```bash
git add configs/pilot/subjects200k-held-in.json src/ratemem/pilot/config.py src/ratemem/pilot/data.py tests/unit/test_pilot_data.py
git commit -m "feat: add pinned held-in pilot data loader"
```

### Task 7: Implement one-random-timestep flow training with one transformer pass

> **Implementation reconciliation (complete):** The executable contract is
> intentionally stronger than the illustrative snippets below.  The trainer
> now consumes the exact Task 4 schedule and the canonical, serialized AdamW
> configuration; accepts only an empty-state optimizer with the exact atom and
> amortizer parameters; performs exactly one SANA call; and requires CPU FP32
> or CUDA BF16 outputs.  Bounded, full-value SHA-256 inventories cover every
> atom, amortizer tensor, frozen transformer parameter/buffer, optimizer tensor,
> and global storage alias, so `.data` mutations and mutation during failed
> train/evaluate calls cannot bypass the guard.  A detected mutation
> permanently poisons the trainer.  The completed suite is 228 passed and one
> explicit CUDA-only skip.  The deliberate cost of scanning the full frozen
> 1.6B backbone remains a measured-pilot item rather than an unverified paper
> efficiency claim.

**Files:**
- Modify: `configs/pilot/sana-1.5-1.6b.json`
- Modify: `src/ratemem/pilot/config.py`
- Create: `src/ratemem/sana/flow.py`
- Modify: `tests/unit/test_pilot_config.py`
- Create: `tests/unit/test_flow_matching.py`
- Create: `tests/contract/test_flow_gradient_contract.py`

- [x] **Step 1: Write the flow interpolation/target and one-pass tests**

```python
# tests/unit/test_flow_matching.py
import torch

from ratemem.sana.flow import flow_interpolate, flow_target, sigma_for_timesteps


def test_flow_endpoints_and_target() -> None:
    clean = torch.tensor([[[[1.0]]], [[[2.0]]]])
    noise = torch.tensor([[[[5.0]]], [[[7.0]]]])
    sigma = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1)
    actual = flow_interpolate(clean, noise, sigma)
    torch.testing.assert_close(actual[0], clean[0])
    torch.testing.assert_close(actual[1], noise[1])
    torch.testing.assert_close(flow_target(clean, noise), noise - clean)


def test_sigma_lookup_preserves_batch_order() -> None:
    schedule_timesteps = torch.tensor([1000.0, 500.0, 1.0])
    schedule_sigmas = torch.tensor([1.0, 0.5, 0.001])
    selected = torch.tensor([1.0, 1000.0])
    sigma = sigma_for_timesteps(selected, schedule_timesteps, schedule_sigmas, n_dim=4)
    torch.testing.assert_close(sigma[:, 0, 0, 0], torch.tensor([0.001, 1.0]))
```

```python
# tests/contract/test_flow_gradient_contract.py
import torch
from diffusers import SanaTransformer2DModel

from ratemem.adapters.sana_layout import install_sana_dynamic_atoms
from ratemem.sana.flow import FlowBatch, OneTimestepFlowTrainer
from ratemem.support.amortizer import SupportAmortizer


def tiny_sana() -> SanaTransformer2DModel:
    return SanaTransformer2DModel(
        in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=4,
        num_layers=1, num_cross_attention_heads=2, cross_attention_head_dim=4,
        cross_attention_dim=8, caption_channels=8, mlp_ratio=1.0, sample_size=4, patch_size=1,
    )


def test_train_step_uses_one_transformer_pass_and_preserves_backbone() -> None:
    torch.manual_seed(17)
    transformer = tiny_sana()
    transformer.requires_grad_(False)
    bank = install_sana_dynamic_atoms(transformer, rank=2, atom_count=4, expected_blocks=1)
    amortizer = SupportAmortizer(
        support_dim=6, description_dim=8, hidden_dim=16,
        projection_count=6, atom_count=4, layers=1, heads=4,
    )
    training_sigmas_tensor = torch.linspace(1, 10, 10, dtype=torch.float32).flip(0) / 10
    training_timesteps = tuple(float(value) for value in training_sigmas_tensor * 10)
    training_sigmas = tuple(float(value) for value in training_sigmas_tensor)
    trainable = [
        parameter
        for parameter in (*bank.parameters(), *amortizer.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    trainer = OneTimestepFlowTrainer(
        transformer,
        bank,
        amortizer,
        training_timesteps,
        training_sigmas,
        optimizer,
    )
    batch = FlowBatch(
        clean_latents=torch.randn(2, 4, 4, 4),
        prompt_embeddings=torch.randn(2, 3, 8),
        prompt_attention_mask=torch.ones(2, 3),
        support_features=torch.randn(2, 2, 6),
        support_mask=torch.ones(2, 2, dtype=torch.bool),
        description_features=torch.randn(2, 8),
    )
    forward_calls = 0

    def count_forward(_module: object, _inputs: object, _output: object) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = transformer.register_forward_hook(count_forward)
    result = trainer.train_step(batch, generator=torch.Generator().manual_seed(19))
    handle.remove()
    assert forward_calls == 1
    assert result.loss > 0 and result.timestep_count == 2
    assert all(parameter.grad is None for parameter in trainer.frozen_parameters)
    assert any(parameter.grad is not None for parameter in bank.parameters() if parameter.requires_grad)
    assert any(parameter.grad is not None for parameter in amortizer.parameters())
```

- [x] **Step 2: Run the flow tests and observe the missing module**

Run: `uv run pytest tests/unit/test_flow_matching.py tests/contract/test_flow_gradient_contract.py -q`

Expected: collection fails because `ratemem.sana.flow` does not exist.

- [x] **Step 3: Implement exact flow interpolation, timestep lookup, and frozen-version guard**

```python
# src/ratemem/sana/flow.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ratemem.adapters.sana_layout import SanaDynamicAdapterBank
from ratemem.support.amortizer import SupportAmortizer


@dataclass(frozen=True)
class FlowBatch:
    clean_latents: Tensor
    prompt_embeddings: Tensor
    prompt_attention_mask: Tensor
    support_features: Tensor
    support_mask: Tensor
    description_features: Tensor


@dataclass(frozen=True)
class FlowStepResult:
    loss: float
    timesteps: tuple[float, ...]
    timestep_count: int


class FrozenVersionGuard:
    def __init__(self, module: nn.Module) -> None:
        self._versions = {
            name: parameter._version
            for name, parameter in module.named_parameters()
            if not parameter.requires_grad
        }

    def assert_unchanged(self, module: nn.Module) -> None:
        changed = [
            name for name, parameter in module.named_parameters()
            if not parameter.requires_grad and self._versions[name] != parameter._version
        ]
        if changed:
            raise RuntimeError(f"frozen parameter versions changed: {changed[:5]}")


def flow_interpolate(clean: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    return (1.0 - sigma) * clean + sigma * noise


def flow_target(clean: Tensor, noise: Tensor) -> Tensor:
    return noise - clean


def sigma_for_timesteps(
    timesteps: Tensor, schedule_timesteps: Tensor, schedule_sigmas: Tensor, *, n_dim: int
) -> Tensor:
    indices = []
    for timestep in timesteps:
        matches = (schedule_timesteps == timestep).nonzero(as_tuple=False).flatten()
        if matches.numel() != 1:
            raise ValueError("each sampled timestep must occur exactly once in the scheduler")
        indices.append(int(matches.item()))
    sigma = schedule_sigmas[torch.tensor(indices, device=schedule_sigmas.device)].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class OneTimestepFlowTrainer:
    def __init__(
        self,
        transformer: nn.Module,
        adapter_bank: SanaDynamicAdapterBank,
        amortizer: SupportAmortizer,
        training_timesteps: tuple[float, ...],
        training_sigmas: tuple[float, ...],
        optimizer: torch.optim.Optimizer,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        self.transformer = transformer
        self.adapter_bank = adapter_bank
        self.amortizer = amortizer
        if (
            type(training_timesteps) is not tuple
            or type(training_sigmas) is not tuple
            or not training_timesteps
            or len(training_timesteps) != len(training_sigmas)
        ):
            raise ValueError("training schedule arrays must be non-empty aligned tuples")
        self.training_timesteps = training_timesteps
        self.training_sigmas = training_sigmas
        self.optimizer = optimizer
        self.autocast_dtype = autocast_dtype
        self.frozen_parameters = tuple(parameter for parameter in transformer.parameters() if not parameter.requires_grad)
        self.frozen_guard = FrozenVersionGuard(transformer)

    def train_step(self, batch: FlowBatch, *, generator: torch.Generator) -> FlowStepResult:
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.amortizer(
            batch.support_features, batch.support_mask, batch.description_features
        )
        clean = batch.clean_latents
        noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
        schedule_timesteps = torch.tensor(
            self.training_timesteps, device=clean.device, dtype=torch.float32
        )
        schedule_sigmas = torch.tensor(
            self.training_sigmas, device=clean.device, dtype=clean.dtype
        )
        indices = torch.randint(
            0, len(self.training_timesteps), (clean.shape[0],),
            generator=generator, device=clean.device,
        )
        timesteps = schedule_timesteps[indices]
        sigma = schedule_sigmas[indices].flatten()
        while sigma.ndim < clean.ndim:
            sigma = sigma.unsqueeze(-1)
        noisy = flow_interpolate(clean, noise, sigma)
        flat_coefficients = prediction.coefficients.reshape(clean.shape[0], -1)
        # Backward remains inside the activation context so gradient-checkpoint recomputation sees the code.
        with self.adapter_bank.activate(flat_coefficients):
            with torch.autocast(
                device_type=clean.device.type,
                dtype=self.autocast_dtype,
                enabled=self.autocast_dtype is not None,
            ):
                model_prediction = self.transformer(
                    hidden_states=noisy,
                    encoder_hidden_states=batch.prompt_embeddings,
                    encoder_attention_mask=batch.prompt_attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
            loss = (model_prediction.float() - flow_target(clean, noise).float()).square().mean()
            loss.backward()
        self.optimizer.step()
        self.frozen_guard.assert_unchanged(self.transformer)
        return FlowStepResult(
            loss=float(loss.detach()),
            timesteps=tuple(float(value) for value in timesteps.detach().cpu()),
            timestep_count=timesteps.numel(),
        )
```

The trainer receives the exact immutable `training_timesteps` and `training_sigmas`
tuples from Task 4. One sampled index selects both values; it must never call
`set_timesteps` or regenerate either array. The CPU contract uses the default
`autocast_dtype=None`; `RealSanaPilotBackend` passes `torch.bfloat16`. Keep
autocast scoped only to the SANA transformer call, keep amortizer logits/scales
and loss accumulation FP32, and enable
`transformer.enable_gradient_checkpointing()` before constructing the trainer.
Never move `loss.backward()` outside `adapter_bank.activate(...)`.

- [x] **Step 4: Run the flow and gradient contracts**

Run: `uv run pytest tests/unit/test_flow_matching.py tests/contract/test_flow_gradient_contract.py -q`

Expected: `3 passed`; the hook records exactly one transformer forward for one training query.

- [x] **Step 5: Commit the one-timestep trainer**

```bash
git add src/ratemem/sana/flow.py tests/unit/test_flow_matching.py tests/contract/test_flow_gradient_contract.py
git commit -m "feat: add one timestep sana flow trainer"
```

### Task 8: Save and reload only trainable atom and amortizer state

> **Implementation reconciliation (complete):** The executable contract is
> intentionally stronger than the illustrative snippets below.  Checkpoints
> contain only exact dynamic-atom and amortizer tensors plus canonical metadata
> bound to pinned model/support revisions, the installed layout, tensor specs,
> and amortizer architecture.  Save is an owner-only, create-only single-file
> transaction anchored to one `O_DIRECTORY|O_NOFOLLOW` parent descriptor; the
> returned `CheckpointFileIdentity` is recomputed from the final published
> inode.  Load inventories the complete transformer and all global storage
> aliases, validates the entire file before mutation, and jointly commits Bank
> and amortizer state while preserving parameter/storage/gradient identities.
> A failed rollback persistently poisons either component across future
> pairings.  The weakref registry cleans only its original record keys, so
> Python object-ID reuse cannot clear an unrelated live poison.  Directory-wide
> artifact publication remains Task 9's boundary.  The completed contract is
> 71 passed, with an independent final review reporting no findings.

**Files:**
- Create: `src/ratemem/adapters/checkpoint.py`
- Create: `tests/contract/test_trainable_checkpoint.py`

- [x] **Step 1: Write a save/load equivalence test**

```python
# tests/contract/test_trainable_checkpoint.py
from pathlib import Path

import torch
from diffusers import SanaTransformer2DModel

from ratemem.adapters.checkpoint import load_trainable_checkpoint, save_trainable_checkpoint
from ratemem.adapters.sana_layout import install_sana_dynamic_atoms
from ratemem.support.amortizer import SupportAmortizer


def tiny_sana() -> SanaTransformer2DModel:
    return SanaTransformer2DModel(
        in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=4,
        num_layers=1, num_cross_attention_heads=2, cross_attention_head_dim=4,
        cross_attention_dim=8, caption_channels=8, mlp_ratio=1.0,
        sample_size=4, patch_size=1,
    )


def test_trainable_checkpoint_excludes_backbone_and_restores_output(tmp_path: Path) -> None:
    torch.manual_seed(23)
    first_transformer = tiny_sana()
    first_transformer.requires_grad_(False)
    first_bank = install_sana_dynamic_atoms(first_transformer, rank=2, atom_count=4, expected_blocks=1)
    first_amortizer = SupportAmortizer(
        support_dim=6, description_dim=8, hidden_dim=16, projection_count=6, atom_count=4, layers=1, heads=4
    ).eval()
    support = torch.randn(1, 2, 6)
    mask = torch.ones(1, 2, dtype=torch.bool)
    description = torch.randn(1, 8)
    expected = first_amortizer(support, mask, description).coefficients
    checkpoint_path = tmp_path / "trainable.safetensors"
    digest = save_trainable_checkpoint(
        checkpoint_path,
        adapter_bank=first_bank,
        amortizer=first_amortizer,
        model_id="test/sana",
        model_revision="1" * 40,
        layout_version="sana-qkv-v1",
    )
    assert len(digest) == 64

    torch.manual_seed(29)
    second_transformer = tiny_sana()
    second_transformer.requires_grad_(False)
    second_bank = install_sana_dynamic_atoms(second_transformer, rank=2, atom_count=4, expected_blocks=1)
    second_amortizer = SupportAmortizer(
        support_dim=6, description_dim=8, hidden_dim=16, projection_count=6, atom_count=4, layers=1, heads=4
    ).eval()
    metadata = load_trainable_checkpoint(
        checkpoint_path, adapter_bank=second_bank, amortizer=second_amortizer,
        expected_model_id="test/sana", expected_model_revision="1" * 40,
        expected_layout_version="sana-qkv-v1",
    )
    actual = second_amortizer(support, mask, description).coefficients
    torch.testing.assert_close(actual, expected)
    for first_wrapper, second_wrapper in zip(first_bank.wrappers, second_bank.wrappers, strict=True):
        torch.testing.assert_close(second_wrapper.atom_down, first_wrapper.atom_down)
        torch.testing.assert_close(second_wrapper.atom_up, first_wrapper.atom_up)
    assert metadata["model_revision"] == "1" * 40
    assert all("base" not in key for key in metadata["tensor_keys"].split(","))
```

- [x] **Step 2: Run the checkpoint test and observe the missing module**

Run: `uv run pytest tests/contract/test_trainable_checkpoint.py -q`

Expected: collection fails because `ratemem.adapters.checkpoint` does not exist.

- [x] **Step 3: Implement deterministic trainable-only safetensors serialization**

```python
# src/ratemem/adapters/checkpoint.py
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from ratemem.adapters.sana_layout import SanaDynamicAdapterBank
from ratemem.support.amortizer import SupportAmortizer


def _state(adapter_bank: SanaDynamicAdapterBank, amortizer: SupportAmortizer) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for index, wrapper in enumerate(adapter_bank.wrappers):
        tensors[f"adapters.{index}.atom_down"] = wrapper.atom_down.detach().cpu().contiguous()
        tensors[f"adapters.{index}.atom_up"] = wrapper.atom_up.detach().cpu().contiguous()
    for name, tensor in amortizer.state_dict().items():
        tensors[f"amortizer.{name}"] = tensor.detach().cpu().contiguous()
    return dict(sorted(tensors.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_trainable_checkpoint(
    path: Path,
    *,
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    model_id: str,
    model_revision: str,
    layout_version: str,
) -> str:
    tensors = _state(adapter_bank, amortizer)
    metadata = {
        "format_version": "1.0.0",
        "model_id": model_id,
        "model_revision": model_revision,
        "layout_version": layout_version,
        "tensor_keys": ",".join(tensors),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata=metadata)
    return _sha256(path)


def load_trainable_checkpoint(
    path: Path,
    *,
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    expected_model_id: str,
    expected_model_revision: str,
    expected_layout_version: str,
) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    expected_metadata = {
        "model_id": expected_model_id,
        "model_revision": expected_model_revision,
        "layout_version": expected_layout_version,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("checkpoint metadata does not match the pinned SANA layout")
    tensors = load_file(path, device="cpu")
    expected_keys = set(_state(adapter_bank, amortizer))
    if set(tensors) != expected_keys:
        raise ValueError("checkpoint tensor keys do not exactly match trainable state")
    for index, wrapper in enumerate(adapter_bank.wrappers):
        wrapper.atom_down.data.copy_(tensors[f"adapters.{index}.atom_down"])
        wrapper.atom_up.data.copy_(tensors[f"adapters.{index}.atom_up"])
    amortizer.load_state_dict({key.removeprefix("amortizer."): value for key, value in tensors.items() if key.startswith("amortizer.")}, strict=True)
    return metadata
```

- [x] **Step 4: Run the save/load contract**

Run: `uv run pytest tests/contract/test_trainable_checkpoint.py -q`

Expected: `1 passed`; the saved key list contains atom factors and amortizer weights only.

- [x] **Step 5: Commit trainable checkpoint interchange**

```bash
git add src/ratemem/adapters/checkpoint.py tests/contract/test_trainable_checkpoint.py
git commit -m "feat: serialize trainable sana adapter state"
```

### Task 9: Define and enforce the pilot attempt artifact schema

**Files:**
- Create: `schemas/ratemem-pilot-attempt-v1.schema.json`
- Create: `src/ratemem/pilot/artifacts.py`
- Create: `tests/unit/test_pilot_artifacts.py`

- [ ] **Step 1: Write tests for scope constants, unknown fields, atomic checksums, and reconciliation**

```python
# tests/unit/test_pilot_artifacts.py
import hashlib
import json
from pathlib import Path

import pytest

from ratemem.pilot.artifacts import ArtifactWriter, validate_attempt


def valid_attempt() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "scope": "engineering_pilot_only",
        "publication_eligible": False,
        "attempt_id": "019d0000-0000-7000-8000-000000000001",
        "phase": "first_pilot",
        "status": "succeeded",
        "started_at": "2026-08-24T00:00:00Z",
        "ended_at": "2026-08-24T01:00:00Z",
        "source": {"git_commit": "1" * 40, "git_diff_sha256": "2" * 64, "config_sha256": "3" * 64},
        "software": {"python": "3.11.13", "torch": "2.13.0", "diffusers": "0.40.0", "peft": "0.20.0", "transformers": "5.16.1", "modal": "1.5.4", "container_image_id": "im-test"},
        "model": {"model_id": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers", "revision": "b77948f2b4eed5c728e9b828ccff07f7427b43cc", "support_model_id": "facebook/dinov2-small", "support_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056"},
        "dataset": {"dataset_id": "Yuanshi/Subjects200K", "revision": "0d1cf6536239888f1a8e218790649344810067bc", "manifest_sha256": "4" * 64, "row_indices": list(range(8)), "held_in": True},
        "runtime": {"seed": 20260824, "requested_gpu": "L40S", "observed_gpu": "NVIDIA L40S", "gpu_count": 1, "cpu_cores": 4, "memory_gib": 32, "timeout_seconds": 7200, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200},
        "modal": {"profile": "ratemem-pilot", "workspace": "authorized-workspace", "environment": "main", "launch_attempt_id": "019d0000-0000-7000-8000-000000000001", "launch_source_sha256": hashlib.sha256(("1" * 40).encode()).hexdigest(), "pilot_slot_sha256": "8" * 64, "submission_receipt_sha256": "9" * 64, "function_call_id": "fc-test", "input_id": "in-test", "task_id": "ta-test", "execution_receipt_count": 1, "execution_receipt_semantics": "lower_bound_may_miss_precommit_reschedule", "retries": 0, "detached": False},
        "cost": {"workspace_budget_usd": "28.00", "internal_limit_usd": "27.00", "known_usage_before_usd": "0.00", "pending_worst_case_usd": "9.50", "phase_bound_usd": "9.50", "estimated_cost_usd": "1.25", "reconciliation_status": "pending", "reconciled_cost_usd": None, "rates_sha256": "5" * 64},
        "probes": {"allowed_probe_names": ["checkpoint_compatibility"], "results": {"checkpoint_compatibility": {"status": "pass"}}, "warmup_steps": 10, "measured_steps": 20, "p50_step_seconds": 1.0, "p95_step_seconds": 1.2, "held_in_step_cap": 40, "initial_flow_loss": 1.1, "final_flow_loss": 0.9, "transformer_passes_per_step": 1},
        "checkpoint": {"path": "trainable.safetensors", "sha256": "6" * 64, "bytes": 1234},
        "files": {"checksums_sha256": "7" * 64},
        "error": None,
    }


def test_schema_forbids_scientific_scope_and_unknown_fields() -> None:
    payload = valid_attempt()
    validate_attempt(payload)
    payload["publication_eligible"] = True
    with pytest.raises(ValueError, match="publication_eligible"):
        validate_attempt(payload)
    payload = valid_attempt()
    payload["headline_identity_score"] = 0.9
    with pytest.raises(ValueError, match="Additional properties"):
        validate_attempt(payload)
    payload = valid_attempt()
    payload["probes"]["results"]["scientific_comparison"] = {"status": "pass"}
    with pytest.raises(ValueError, match="Additional properties"):
        validate_attempt(payload)


def test_writer_hashes_files_and_finalization_requires_reconciled_cost(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path / "attempt", valid_attempt())
    writer.write_json("metrics.json", {"loss": 0.9})
    writer.write_bytes("trainable.safetensors", b"checkpoint")
    writer.write_pending()
    assert (writer.root / "attempt.pending.json").exists()
    assert "metrics.json" in (writer.root / "checksums.sha256").read_text()
    final_path = writer.finalize(reconciled_cost_usd="1.31")
    final = json.loads(final_path.read_text())
    assert final["cost"]["reconciliation_status"] == "reconciled"
    assert final["cost"]["reconciled_cost_usd"] == "1.31"


def test_launch_identity_cross_fields_reject_tampering() -> None:
    payload = valid_attempt()
    payload["modal"]["launch_attempt_id"] = "019d0000-0000-7000-8000-000000000002"
    with pytest.raises(ValueError, match="launch attempt identity"):
        validate_attempt(payload)

    payload = valid_attempt()
    payload["modal"]["launch_source_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="launch source identity"):
        validate_attempt(payload)
```

- [ ] **Step 2: Run the artifact tests and observe the missing module/schema**

Run: `uv run pytest tests/unit/test_pilot_artifacts.py -q`

Expected: collection fails because `ratemem.pilot.artifacts` does not exist.

- [ ] **Step 3: Add the complete Draft 2020-12 artifact schema**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://ratemem.local/schemas/ratemem-pilot-attempt-v1.schema.json",
  "title": "RateMem engineering pilot attempt",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "scope", "publication_eligible", "attempt_id", "phase", "status", "started_at", "ended_at", "source", "software", "model", "dataset", "runtime", "modal", "cost", "probes", "checkpoint", "files", "error"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "scope": {"const": "engineering_pilot_only"},
    "publication_eligible": {"const": false},
    "attempt_id": {"type": "string", "pattern": "^[0-9a-f-]{36}$"},
    "phase": {"enum": ["first_pilot"]},
    "status": {"enum": ["succeeded", "probe_failed", "oom", "exception"]},
    "started_at": {"type": "string", "format": "date-time"},
    "ended_at": {"type": "string", "format": "date-time"},
    "source": {
      "type": "object", "additionalProperties": false,
      "required": ["git_commit", "git_diff_sha256", "config_sha256"],
      "properties": {
        "git_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "git_diff_sha256": {"$ref": "#/$defs/sha256"},
        "config_sha256": {"$ref": "#/$defs/sha256"}
      }
    },
    "software": {
      "type": "object", "additionalProperties": false,
      "required": ["python", "torch", "diffusers", "peft", "transformers", "modal", "container_image_id"],
      "properties": {
        "python": {"type": "string"}, "torch": {"const": "2.13.0"}, "diffusers": {"const": "0.40.0"},
        "peft": {"const": "0.20.0"}, "transformers": {"const": "5.16.1"}, "modal": {"const": "1.5.4"},
        "container_image_id": {"type": "string", "minLength": 1}
      }
    },
    "model": {
      "type": "object", "additionalProperties": false,
      "required": ["model_id", "revision", "support_model_id", "support_revision"],
      "properties": {
        "model_id": {"const": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"},
        "revision": {"const": "b77948f2b4eed5c728e9b828ccff07f7427b43cc"},
        "support_model_id": {"const": "facebook/dinov2-small"},
        "support_revision": {"const": "ed25f3a31f01632728cabb09d1542f84ab7b0056"}
      }
    },
    "dataset": {
      "type": "object", "additionalProperties": false,
      "required": ["dataset_id", "revision", "manifest_sha256", "row_indices", "held_in"],
      "properties": {
        "dataset_id": {"const": "Yuanshi/Subjects200K"},
        "revision": {"const": "0d1cf6536239888f1a8e218790649344810067bc"},
        "manifest_sha256": {"$ref": "#/$defs/sha256"},
        "row_indices": {"const": [0, 1, 2, 3, 4, 5, 6, 7]},
        "held_in": {"const": true}
      }
    },
    "runtime": {
      "type": "object", "additionalProperties": false,
      "required": ["seed", "requested_gpu", "observed_gpu", "gpu_count", "cpu_cores", "memory_gib", "timeout_seconds", "peak_allocated_bytes", "peak_reserved_bytes"],
      "properties": {
        "seed": {"type": "integer"}, "requested_gpu": {"const": "L40S"}, "observed_gpu": {"type": "string", "pattern": "L40S"},
        "gpu_count": {"const": 1}, "cpu_cores": {"const": 4}, "memory_gib": {"const": 32}, "timeout_seconds": {"const": 7200},
        "peak_allocated_bytes": {"type": "integer", "minimum": 0}, "peak_reserved_bytes": {"type": "integer", "minimum": 0}
      }
    },
    "modal": {
      "type": "object", "additionalProperties": false,
      "required": ["profile", "workspace", "environment", "launch_attempt_id", "launch_source_sha256", "pilot_slot_sha256", "submission_receipt_sha256", "function_call_id", "input_id", "task_id", "execution_receipt_count", "execution_receipt_semantics", "retries", "detached"],
      "properties": {
        "profile": {"const": "ratemem-pilot"}, "workspace": {"type": "string", "minLength": 1}, "environment": {"const": "main"},
        "launch_attempt_id": {"type": "string", "pattern": "^[0-9a-f-]{36}$"},
        "launch_source_sha256": {"$ref": "#/$defs/sha256"},
        "pilot_slot_sha256": {"$ref": "#/$defs/sha256"},
        "submission_receipt_sha256": {"$ref": "#/$defs/sha256"},
        "function_call_id": {"type": "string", "minLength": 1}, "input_id": {"type": "string", "minLength": 1},
        "task_id": {"oneOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "execution_receipt_count": {"type": "integer", "minimum": 1},
        "execution_receipt_semantics": {"const": "lower_bound_may_miss_precommit_reschedule"},
        "retries": {"const": 0}, "detached": {"const": false}
      }
    },
    "cost": {
      "type": "object", "additionalProperties": false,
      "required": ["workspace_budget_usd", "internal_limit_usd", "known_usage_before_usd", "pending_worst_case_usd", "phase_bound_usd", "estimated_cost_usd", "reconciliation_status", "reconciled_cost_usd", "rates_sha256"],
      "properties": {
        "workspace_budget_usd": {"const": "28.00"}, "internal_limit_usd": {"const": "27.00"},
        "known_usage_before_usd": {"$ref": "#/$defs/usd"}, "pending_worst_case_usd": {"$ref": "#/$defs/usd"},
        "phase_bound_usd": {"$ref": "#/$defs/usd"}, "estimated_cost_usd": {"$ref": "#/$defs/usd"},
        "reconciliation_status": {"enum": ["pending", "reconciled"]},
        "reconciled_cost_usd": {"oneOf": [{"$ref": "#/$defs/usd"}, {"type": "null"}]},
        "rates_sha256": {"$ref": "#/$defs/sha256"}
      },
      "allOf": [{"if": {"properties": {"reconciliation_status": {"const": "reconciled"}}}, "then": {"properties": {"reconciled_cost_usd": {"$ref": "#/$defs/usd"}}}}]
    },
    "probes": {
      "type": "object", "additionalProperties": false,
      "required": ["allowed_probe_names", "results", "warmup_steps", "measured_steps", "p50_step_seconds", "p95_step_seconds", "held_in_step_cap", "initial_flow_loss", "final_flow_loss", "transformer_passes_per_step"],
      "properties": {
        "allowed_probe_names": {"type": "array", "items": {"enum": ["checkpoint_compatibility", "dynamic_numerics", "gradient_flow", "frozen_backbone", "peak_memory", "one_step_inference", "one_timestep_backward", "step_timing", "held_in_loss"]}, "uniqueItems": true},
        "results": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "checkpoint_compatibility": {"type": "object"},
            "dynamic_numerics": {"type": "object"},
            "gradient_flow": {"type": "object"},
            "frozen_backbone": {"type": "object"},
            "peak_memory": {"type": "object"},
            "one_step_inference": {"type": "object"},
            "one_timestep_backward": {"type": "object"},
            "step_timing": {"type": "object"},
            "held_in_loss": {"type": "object"}
          }
        },
        "warmup_steps": {"const": 10}, "measured_steps": {"const": 20},
        "p50_step_seconds": {"type": "number", "exclusiveMinimum": 0}, "p95_step_seconds": {"type": "number", "exclusiveMinimum": 0},
        "held_in_step_cap": {"type": "integer", "minimum": 0}, "initial_flow_loss": {"type": "number"}, "final_flow_loss": {"type": "number"},
        "transformer_passes_per_step": {"const": 1}
      }
    },
    "checkpoint": {
      "type": "object", "additionalProperties": false,
      "required": ["path", "sha256", "bytes"],
      "properties": {"path": {"const": "trainable.safetensors"}, "sha256": {"$ref": "#/$defs/sha256"}, "bytes": {"type": "integer", "minimum": 1}}
    },
    "files": {
      "type": "object", "additionalProperties": false,
      "required": ["checksums_sha256"], "properties": {"checksums_sha256": {"$ref": "#/$defs/sha256"}}
    },
    "error": {"oneOf": [{"type": "null"}, {"type": "object", "additionalProperties": false, "required": ["type", "message"], "properties": {"type": {"type": "string"}, "message": {"type": "string"}}}]}
  },
  "$defs": {
    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "usd": {"type": "string", "pattern": "^(0|[1-9][0-9]*)\\.[0-9]{2,6}$"}
  }
}
```

- [ ] **Step 4: Implement canonical, atomic artifact writing and validation**

```python
# src/ratemem/pilot/artifacts.py (core contract)
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_PATH = Path("schemas/ratemem-pilot-attempt-v1.schema.json")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_attempt(payload: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(f"{'.'.join(map(str, error.path))}: {error.message}" for error in errors)
        raise ValueError(detail)
    if payload["modal"]["launch_attempt_id"] != payload["attempt_id"]:
        raise ValueError("launch attempt identity differs from artifact attempt_id")
    expected_source = hashlib.sha256(payload["source"]["git_commit"].encode()).hexdigest()
    if payload["modal"]["launch_source_sha256"] != expected_source:
        raise ValueError("launch source identity differs from the exact HEAD commit hash")


class ArtifactWriter:
    def __init__(self, root: Path, attempt: dict[str, Any]) -> None:
        self.root = root
        self.attempt = copy.deepcopy(attempt)
        self.root.mkdir(parents=True, exist_ok=False)

    def _atomic(self, path: Path, content: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def write_json(self, relative_path: str, value: Any) -> None:
        self._atomic(self.root / relative_path, _canonical(value))

    def write_bytes(self, relative_path: str, value: bytes) -> None:
        self._atomic(self.root / relative_path, value)

    def _write_checksums(self) -> str:
        entries = []
        for path in sorted(self.root.iterdir()):
            if path.is_file() and path.name not in {"checksums.sha256", "attempt.pending.json", "attempt.json"}:
                entries.append(f"{_sha256(path)}  {path.name}")
        self._atomic(self.root / "checksums.sha256", ("\n".join(entries) + "\n").encode())
        return _sha256(self.root / "checksums.sha256")

    def write_pending(self) -> Path:
        self.attempt["files"]["checksums_sha256"] = self._write_checksums()
        validate_attempt(self.attempt)
        path = self.root / "attempt.pending.json"
        self._atomic(path, _canonical(self.attempt))
        return path

    def finalize(self, *, reconciled_cost_usd: str) -> Path:
        self.attempt["cost"]["reconciliation_status"] = "reconciled"
        self.attempt["cost"]["reconciled_cost_usd"] = reconciled_cost_usd
        validate_attempt(self.attempt)
        path = self.root / "attempt.json"
        self._atomic(path, _canonical(self.attempt))
        return path
```

The real runner writes `config.json`, `rates.json`, `dataset-manifest.json`, `execution-receipts.jsonl`, `metrics.jsonl`, and `trainable.safetensors` before `write_pending()`. `execution-receipts.jsonl` is copied from the separately committed append-only runtime receipt, and its count populates the lower-bound field. Each runtime receipt and the artifact's `modal` object carry the launch attempt ID, workspace, source hash, pilot-slot hash, and submission-receipt hash received from the locally consumed request. `validate_attempt()` cross-checks the launch attempt against top-level `attempt_id` and the launch source against the exact clean HEAD commit. `attempt.pending.json` remains immutable evidence; reconciliation creates `attempt.json` rather than modifying the pending file.

- [ ] **Step 5: Run the artifact tests**

Run: `uv run pytest tests/unit/test_pilot_artifacts.py -q`

Expected: `3 passed`; an unknown scientific metric and either launch-identity mismatch are rejected,
and finalization records a non-null reconciled cost.

- [ ] **Step 6: Commit the artifact contract**

```bash
git add schemas/ratemem-pilot-attempt-v1.schema.json src/ratemem/pilot/artifacts.py tests/unit/test_pilot_artifacts.py
git commit -m "feat: validate pilot attempt artifacts"
```

### Task 10: Enforce workspace identity, the USD 28 cap, and the USD 27 internal ledger

> **Implementation reconciliation (complete):** The executable boundary is
> intentionally stronger than the illustrative snippets below.  Modal queries
> run only with a verified owner-only config path, an exact named profile, and
> a rebuilt environment that cannot inherit token/OAuth overrides.  Missing,
> unauthorized, warning-bearing, malformed, or semantically inconsistent
> billing responses fail closed; metered usage is checked before credits.
> Because Modal exposes no documented Workspace-budget read API, the outer USD
> 28 boundary is explicitly an operator-attested dashboard contract rather than
> an API claim: canonical private evidence binds the exact workspace, budget,
> UTC freshness, fixed confirmation, and a hashed dashboard capture.  The USD
> 27 admission ledger uses exact Decimal arithmetic plus a create-only receipt
> for every hash-chained entry, so a single-file rollback, truncation, deletion,
> or interrupted receipt publication permanently blocks another reservation.
> No purely local design can detect a same-UID actor deleting the ledger and all
> receipts together; that capability boundary is explicit.  The final review
> passed 66 focused tests plus five targeted replays with no findings, and no
> real Modal configuration, credential, network query, or paid call was used.

**Files:**
- Create: `configs/pilot/modal-budget.json`
- Modify: `src/ratemem/pilot/config.py`
- Create: `src/ratemem/pilot/private_io.py`
- Create: `src/ratemem/pilot/workspace.py`
- Create: `src/ratemem/pilot/costs.py`
- Create: `tests/fixtures/modal/profile-list.json`
- Create: `tests/fixtures/modal/billing-summary.json`
- Create: `tests/fixtures/modal/rates.json`
- Create: `tests/unit/test_workspace_guard.py`
- Create: `tests/unit/test_private_io.py`
- Create: `tests/unit/test_cost_ledger.py`

- [x] **Step 1: Write fail-closed workspace tests**

```python
# tests/unit/test_workspace_guard.py
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from ratemem.pilot.config import ModalBudgetConfig
from ratemem.pilot.workspace import WorkspaceSnapshot, verify_workspace_snapshot


def snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    evidence = tmp_path / "usage-budget-28.png"
    evidence.write_bytes(b"budget evidence")
    evidence.chmod(0o600)
    return WorkspaceSnapshot(
        profile="ratemem-pilot",
        workspace="authorized-workspace",
        environment="main",
        workspace_budget_usd="28.00",
        known_metered_usage_usd="1.25",
        verified_at=datetime.now(timezone.utc),
        evidence_path=evidence,
        evidence_sha256="",
        rates={"gpu_l40s_per_second": "0.000542", "cpu_core_per_second": "0.0000131", "memory_gib_per_second": "0.00000222", "volume_gib_month": "0.09"},
    ).with_evidence_hash()


def test_exact_workspace_budget_and_profile_pass(tmp_path: Path) -> None:
    verified = verify_workspace_snapshot(snapshot(tmp_path), expected_workspace="authorized-workspace", max_age_seconds=900)
    assert verified.workspace_budget_usd == "28.00"


@pytest.mark.parametrize("field,value", [("profile", "default"), ("workspace_budget_usd", "28.01"), ("workspace", "another-workspace")])
def test_profile_budget_or_workspace_mismatch_fails_closed(tmp_path: Path, field: str, value: str) -> None:
    payload = snapshot(tmp_path).__dict__ | {field: value}
    candidate = WorkspaceSnapshot(**payload)
    with pytest.raises(ValueError):
        verify_workspace_snapshot(candidate, expected_workspace="authorized-workspace", max_age_seconds=900)


def test_stale_attestation_fails_closed(tmp_path: Path) -> None:
    payload = snapshot(tmp_path).__dict__ | {"verified_at": datetime.now(timezone.utc) - timedelta(seconds=901)}
    with pytest.raises(ValueError, match="stale"):
        verify_workspace_snapshot(WorkspaceSnapshot(**payload), expected_workspace="authorized-workspace", max_age_seconds=900)


def test_committed_budget_keeps_outer_cap_internal_bound_and_phase_split() -> None:
    config = ModalBudgetConfig.load(Path("configs/pilot/modal-budget.json"))
    assert config.workspace_budget_usd == Decimal("28.00")
    assert config.internal_limit_usd == Decimal("27.00")
    assert config.first_pilot_allocation_usd == Decimal("2.00") + Decimal("3.00") + Decimal("16.00")
    assert config.first_pilot_allocation_usd + config.unallocated_safety_buffer_usd == Decimal("27.00")
```

```python
# tests/unit/test_private_io.py
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ratemem.pilot.private_io import (
    ensure_private_directory,
    private_lock,
    read_private_json,
    write_exclusive_private_json,
)


def test_private_directory_and_file_require_exact_owner_modes(tmp_path: Path) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    target = private / "record.json"
    write_exclusive_private_json(target, {"value": 1})
    assert private.stat().st_mode & 0o777 == 0o700
    assert target.stat().st_mode & 0o777 == 0o600
    assert read_private_json(target) == {"value": 1}


def test_permissive_parent_file_and_symlink_fail_closed(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    target = private / "record.json"
    target.write_text(json.dumps({"value": 1}))
    target.chmod(0o600)

    private.chmod(0o755)
    with pytest.raises(PermissionError, match="directory must be owned by the current uid with mode 0700"):
        read_private_json(target)
    private.chmod(0o700)

    target.chmod(0o644)
    with pytest.raises(PermissionError, match="file must be owned by the current uid with mode 0600"):
        read_private_json(target)
    target.chmod(0o600)

    link = private / "link.json"
    link.symlink_to(target)
    with pytest.raises(PermissionError, match="regular non-symlink"):
        read_private_json(link)


def test_private_lock_serializes_threads_and_has_mode_0600(tmp_path: Path) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    lock_path = private / "state.lock"
    order: list[int] = []

    def append(value: int) -> None:
        with private_lock(lock_path):
            order.append(value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(append, (1, 2)))
    assert sorted(order) == [1, 2]
    assert lock_path.stat().st_mode & 0o777 == 0o600
    assert lock_path.stat().st_uid == os.getuid()


def test_private_lock_rejects_instead_of_repairing_existing_permissive_file(tmp_path: Path) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    lock_path = private / "state.lock"
    lock_path.write_bytes(b"")
    lock_path.chmod(0o644)
    with pytest.raises(PermissionError, match="file must be a regular non-symlink"):
        with private_lock(lock_path):
            raise AssertionError("unreachable")
    assert lock_path.stat().st_mode & 0o777 == 0o644
```

- [x] **Step 2: Write Decimal ledger tests for open reservations and the internal bound**

```python
# tests/unit/test_cost_ledger.py
from decimal import Decimal
from pathlib import Path

import pytest

from ratemem.pilot.costs import CostLedger, CostRates, ResourceContract, conservative_bound


def rates() -> CostRates:
    return CostRates(
        gpu_l40s_per_second=Decimal("0.000542"),
        cpu_core_per_second=Decimal("0.0000131"),
        memory_gib_per_second=Decimal("0.00000222"),
        volume_gib_month=Decimal("0.09"),
    )


def test_bound_includes_gpu_cpu_requested_ram_startup_and_storage() -> None:
    resources = ResourceContract(
        gpu_count=1, cpu_cores=4, memory_gib=32, timeout_seconds=7200,
        startup_timeout_seconds=1800, storage_gib_bound=24,
        non_gpu_setup_allowance_usd=Decimal("2.00"),
    )
    bound = conservative_bound(rates(), resources)
    assert bound == Decimal("10.15")
    assert bound <= Decimal("21.00")


def test_reservation_obeys_known_plus_pending_plus_new_at_27(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "cost-ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    ledger.reserve("attempt-one", known_usage=Decimal("1.00"), phase_bound=Decimal("20.00"), rates_sha256="1" * 64)
    with pytest.raises(ValueError, match="internal USD 27.00 limit"):
        ledger.reserve("attempt-two", known_usage=Decimal("1.00"), phase_bound=Decimal("6.01"), rates_sha256="2" * 64)
    ledger.reconcile("attempt-one", reconciled_cost=Decimal("2.50"), known_usage_after=Decimal("3.50"))
    ledger.verify_hash_chain()
```

- [x] **Step 3: Run the guard/ledger tests and observe missing modules**

Run: `uv run pytest tests/unit/test_private_io.py tests/unit/test_workspace_guard.py tests/unit/test_cost_ledger.py -q`

Expected: collection fails because the private-I/O, workspace, and cost modules do not exist.

- [x] **Step 4: Commit the hard cost/resource configuration**

```json
{
  "schema_version": "1.0.0",
  "profile": "ratemem-pilot",
  "environment": "main",
  "workspace_budget_usd": "28.00",
  "internal_limit_usd": "27.00",
  "first_pilot_allocation_usd": "21.00",
  "setup_probe_allocation_usd": "2.00",
  "timing_probe_allocation_usd": "3.00",
  "held_in_pilot_allocation_usd": "16.00",
  "unallocated_safety_buffer_usd": "6.00",
  "attestation_max_age_seconds": 900,
  "gpu": "L40S",
  "gpu_count": 1,
  "cpu_cores": 4,
  "memory_gib": 32,
  "timeout_seconds": 7200,
  "startup_timeout_seconds": 1800,
  "storage_gib_bound": 24,
  "non_gpu_setup_allowance_usd": "2.00",
  "retries": 0,
  "max_containers": 1,
  "detached": false,
  "cache_volume": "ratemem-sana-cache",
  "artifact_volume": "ratemem-pilot-artifacts"
}
```

Save this as `configs/pilot/modal-budget.json`, add `Decimal` to the imports in `src/ratemem/pilot/config.py`, and append this exact-key parser:

```python
@dataclass(frozen=True)
class ModalBudgetConfig:
    profile: str
    environment: str
    workspace_budget_usd: Decimal
    internal_limit_usd: Decimal
    first_pilot_allocation_usd: Decimal
    setup_probe_allocation_usd: Decimal
    timing_probe_allocation_usd: Decimal
    held_in_pilot_allocation_usd: Decimal
    unallocated_safety_buffer_usd: Decimal
    attestation_max_age_seconds: int
    gpu: str
    gpu_count: int
    cpu_cores: int
    memory_gib: int
    timeout_seconds: int
    startup_timeout_seconds: int
    storage_gib_bound: int
    non_gpu_setup_allowance_usd: Decimal
    retries: int
    max_containers: int
    detached: bool
    cache_volume: str
    artifact_volume: str

    @classmethod
    def load(cls, path: Path) -> "ModalBudgetConfig":
        payload = json.loads(path.read_text())
        locked = {
            "schema_version": "1.0.0",
            "profile": "ratemem-pilot", "environment": "main",
            "workspace_budget_usd": "28.00", "internal_limit_usd": "27.00",
            "first_pilot_allocation_usd": "21.00", "setup_probe_allocation_usd": "2.00",
            "timing_probe_allocation_usd": "3.00", "held_in_pilot_allocation_usd": "16.00",
            "unallocated_safety_buffer_usd": "6.00", "attestation_max_age_seconds": 900,
            "gpu": "L40S", "gpu_count": 1, "cpu_cores": 4, "memory_gib": 32,
            "timeout_seconds": 7200, "startup_timeout_seconds": 1800,
            "storage_gib_bound": 24, "non_gpu_setup_allowance_usd": "2.00",
            "retries": 0, "max_containers": 1, "detached": False,
            "cache_volume": "ratemem-sana-cache", "artifact_volume": "ratemem-pilot-artifacts",
        }
        if json.dumps(payload, sort_keys=True) != json.dumps(locked, sort_keys=True):
            raise ValueError("Modal budget config differs from the locked one-L40S contract")
        decimal_fields = {
            "workspace_budget_usd", "internal_limit_usd", "first_pilot_allocation_usd",
            "setup_probe_allocation_usd", "timing_probe_allocation_usd",
            "held_in_pilot_allocation_usd", "unallocated_safety_buffer_usd",
            "non_gpu_setup_allowance_usd",
        }
        values = {
            key: Decimal(value) if key in decimal_fields else value
            for key, value in payload.items()
            if key != "schema_version"
        }
        return cls(**values)
```

This exact payload comparison rejects a list or a GPU string containing a count or fallback. The three first-pilot buckets are recorded separately in `rates.json` and `metrics.jsonl` even though they run in the one authorized synchronous invocation; they do not authorize three invocations. The USD 6.00 safety buffer is never reserved and authorizes no second submission.

- [x] **Step 5: Add credential-free Modal response fixtures**

```json
[
  {"name": "ratemem-pilot", "workspace": "authorized-workspace", "active": true}
]
```

```json
{
  "metered_cost": "1.25",
  "billed_cost": "0.00",
  "adjustments": {"credits": "-1.25"},
  "metered_cost_breakdown": {"compute": "1.25"}
}
```

```json
{
  "gpu_l40s_per_second": "0.000542",
  "cpu_core_per_second": "0.0000131",
  "memory_gib_per_second": "0.00000222",
  "volume_gib_month": "0.09"
}
```

These files contain neither token IDs nor token secrets.

- [x] **Step 6: Implement owner-only atomic state I/O and locking**

```python
# src/ratemem/pilot/private_io.py
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise PermissionError("directory must be owned by the current uid with mode 0700")


def _validate_private_file(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise PermissionError("file must be a regular non-symlink owned by the current uid with mode 0600")


def read_private_bytes(path: Path) -> bytes:
    ensure_private_directory(path.parent)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise PermissionError("file must be a regular non-symlink owned by the current uid with mode 0600") from error
    try:
        _validate_private_file(os.fstat(descriptor))
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def read_private_json(path: Path) -> dict[str, Any]:
    payload = json.loads(read_private_bytes(path))
    if not isinstance(payload, dict):
        raise ValueError("private JSON root must be an object")
    return payload


def file_sha256(path: Path) -> str:
    return hashlib.sha256(read_private_bytes(path)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_private_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def write_atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_private_directory(path.parent)
    if path.exists() or path.is_symlink():
        read_private_bytes(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    write_exclusive_private_json(temporary, payload)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@contextmanager
def private_lock(path: Path) -> Iterator[None]:
    ensure_private_directory(path.parent)
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        created = True
    except FileExistsError:
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        if created:
            os.fchmod(descriptor, 0o600)
        _validate_private_file(os.fstat(descriptor))
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
```

`ensure_private_directory` never repairs an existing permissive or foreign-owned directory; it stops. Reads use `O_NOFOLLOW`, require a regular single-link owner file, and validate the open descriptor rather than trusting a pre-open path check. Every immutable write is mode 0600, fsyncs the file, and fsyncs its directory. The lock file is itself owner-only and is shared by slot claim and submission consumption.

- [x] **Step 7: Implement the attested workspace snapshot and non-secret Modal queries**

```python
# src/ratemem/pilot/workspace.py (core contract)
from __future__ import annotations

import json
import os
import stat
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from ratemem.pilot.private_io import file_sha256, read_private_json


@dataclass(frozen=True)
class WorkspaceSnapshot:
    profile: str
    workspace: str
    environment: str
    workspace_budget_usd: str
    known_metered_usage_usd: str
    verified_at: datetime
    evidence_path: Path
    evidence_sha256: str
    rates: dict[str, str]

    def with_evidence_hash(self) -> "WorkspaceSnapshot":
        digest = file_sha256(self.evidence_path)
        return replace(self, evidence_sha256=digest)

    def to_json(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "workspace": self.workspace,
            "environment": self.environment,
            "workspace_budget_usd": self.workspace_budget_usd,
            "known_metered_usage_usd": self.known_metered_usage_usd,
            "verified_at": self.verified_at.isoformat(),
            "evidence_path": str(self.evidence_path),
            "evidence_sha256": self.evidence_sha256,
            "rates": self.rates,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "WorkspaceSnapshot":
        expected = {
            "profile", "workspace", "environment", "workspace_budget_usd",
            "known_metered_usage_usd", "verified_at", "evidence_path",
            "evidence_sha256", "rates",
        }
        if set(payload) != expected or not isinstance(payload["rates"], dict):
            raise ValueError("workspace attestation has unexpected keys or rates")
        return cls(
            profile=str(payload["profile"]),
            workspace=str(payload["workspace"]),
            environment=str(payload["environment"]),
            workspace_budget_usd=str(payload["workspace_budget_usd"]),
            known_metered_usage_usd=str(payload["known_metered_usage_usd"]),
            verified_at=datetime.fromisoformat(str(payload["verified_at"])),
            evidence_path=Path(str(payload["evidence_path"])),
            evidence_sha256=str(payload["evidence_sha256"]),
            rates={str(key): str(value) for key, value in payload["rates"].items()},
        )


def _modal_json(profile: str, arguments: list[str]) -> object:
    environment = os.environ.copy()
    environment["MODAL_PROFILE"] = profile
    completed = subprocess.run(
        ["modal", *arguments, "--json"], capture_output=True, text=True,
        check=True, env=environment,
    )
    return json.loads(completed.stdout)


def capture_workspace_snapshot(*, evidence_path: Path, confirmed_budget: str) -> WorkspaceSnapshot:
    if confirmed_budget != "28.00":
        raise ValueError("workspace usage budget must be exactly USD 28.00")
    config_path = Path.home() / ".modal.toml"
    config_metadata = config_path.lstat()
    if (
        not stat.S_ISREG(config_metadata.st_mode)
        or config_metadata.st_uid != os.getuid()
        or stat.S_IMODE(config_metadata.st_mode) != 0o600
        or config_metadata.st_nlink != 1
    ):
        raise PermissionError("Modal config must be an owner mode-0600 regular non-symlink file")
    profiles = _modal_json("ratemem-pilot", ["profile", "list"])
    matches = [entry for entry in profiles if entry["name"] == "ratemem-pilot" and entry["active"]]
    if len(matches) != 1:
        raise ValueError("the ratemem-pilot profile is not the sole explicitly selected profile")
    billing = _modal_json("ratemem-pilot", ["billing", "summary", "--for", "this month"])
    rates = _modal_json("ratemem-pilot", ["billing", "rates"])
    return WorkspaceSnapshot(
        profile="ratemem-pilot", workspace=matches[0]["workspace"], environment="main",
        workspace_budget_usd=confirmed_budget, known_metered_usage_usd=billing["metered_cost"],
        verified_at=datetime.now(timezone.utc), evidence_path=evidence_path,
        evidence_sha256="", rates=rates,
    ).with_evidence_hash()


def verify_workspace_snapshot(
    snapshot: WorkspaceSnapshot, *, expected_workspace: str, max_age_seconds: int
) -> WorkspaceSnapshot:
    if snapshot.profile != "ratemem-pilot" or snapshot.workspace != expected_workspace:
        raise ValueError("Modal profile/workspace mismatch")
    if snapshot.environment != "main" or snapshot.workspace_budget_usd != "28.00":
        raise ValueError("Modal environment or workspace budget mismatch")
    age = (datetime.now(timezone.utc) - snapshot.verified_at).total_seconds()
    if age < 0 or age > max_age_seconds:
        raise ValueError("workspace attestation is stale")
    if snapshot.with_evidence_hash().evidence_sha256 != snapshot.evidence_sha256:
        raise ValueError("workspace-budget evidence hash changed")
    return snapshot


def verify_fresh_attestation_file(path: Path) -> WorkspaceSnapshot:
    snapshot = WorkspaceSnapshot.from_json(read_private_json(path))
    profiles = _modal_json("ratemem-pilot", ["profile", "list"])
    selected = [
        entry
        for entry in profiles  # type: ignore[union-attr]
        if entry["name"] == "ratemem-pilot" and entry["active"]
    ]
    if len(selected) != 1:
        raise ValueError("ratemem-pilot is not the explicitly selected profile")
    verified = verify_workspace_snapshot(
        snapshot, expected_workspace=selected[0]["workspace"], max_age_seconds=900
    )
    billing = _modal_json(
        "ratemem-pilot", ["billing", "summary", "--for", "this month"]
    )
    rates = _modal_json("ratemem-pilot", ["billing", "rates"])
    if not isinstance(billing, dict) or not isinstance(rates, dict):
        raise ValueError("Modal billing commands returned unexpected JSON")
    return replace(
        verified,
        known_metered_usage_usd=str(billing["metered_cost"]),
        rates={str(key): str(value) for key, value in rates.items()},
    )
```

`capture_workspace_snapshot` is called only after the operator has selected the intended first authorized workspace in the Modal dashboard, set its Workspace usage budget to exactly USD 28.00, and saved the Usage & Billing evidence file outside the repository. It records `metered_cost` before credits because the Workspace usage budget is also pre-credit. Raw `modal token info`, environment dumps, and Modal config contents are never captured.

- [x] **Step 8: Implement normalized rates, conservative bounds, and the append-only ledger**

```python
# src/ratemem/pilot/costs.py (types and admission contract)
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from filelock import FileLock


@dataclass(frozen=True)
class CostRates:
    gpu_l40s_per_second: Decimal
    cpu_core_per_second: Decimal
    memory_gib_per_second: Decimal
    volume_gib_month: Decimal

    @classmethod
    def normalize(cls, raw: dict[str, str]) -> "CostRates":
        required = {"gpu_l40s_per_second", "cpu_core_per_second", "memory_gib_per_second", "volume_gib_month"}
        if set(raw) != required:
            raise ValueError(f"unexpected Modal rate keys: {sorted(raw)}")
        return cls(**{key: Decimal(value) for key, value in raw.items()})


@dataclass(frozen=True)
class ResourceContract:
    gpu_count: int
    cpu_cores: int
    memory_gib: int
    timeout_seconds: int
    startup_timeout_seconds: int
    storage_gib_bound: int
    non_gpu_setup_allowance_usd: Decimal


def conservative_bound(rates: CostRates, resources: ResourceContract) -> Decimal:
    billed_seconds = Decimal(resources.timeout_seconds + resources.startup_timeout_seconds)
    compute = billed_seconds * (
        rates.gpu_l40s_per_second * resources.gpu_count
        + rates.cpu_core_per_second * resources.cpu_cores
        + rates.memory_gib_per_second * resources.memory_gib
    )
    storage = rates.volume_gib_month * resources.storage_gib_bound
    total = compute + storage + resources.non_gpu_setup_allowance_usd
    return total.quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class CostLedger:
    def __init__(self, path: Path, *, internal_limit_usd: Decimal) -> None:
        self.path = path
        self.internal_limit_usd = internal_limit_usd
        self.lock = FileLock(str(path) + ".lock")

    def _entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def _append(self, body: dict[str, Any]) -> None:
        entries = self._entries()
        previous_hash = entries[-1]["entry_sha256"] if entries else "0" * 64
        entry = body | {"sequence": len(entries), "previous_sha256": previous_hash}
        entry["entry_sha256"] = hashlib.sha256(_canonical(entry).encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical(entry) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, attempt_id: str, *, known_usage: Decimal, phase_bound: Decimal, rates_sha256: str) -> None:
        with self.lock:
            entries = self._entries()
            open_bounds: dict[str, Decimal] = {}
            for entry in entries:
                if entry["kind"] == "reserve":
                    open_bounds[entry["attempt_id"]] = Decimal(entry["phase_bound_usd"])
                elif entry["kind"] == "reconcile":
                    open_bounds.pop(entry["attempt_id"], None)
            if attempt_id in open_bounds:
                raise ValueError("attempt already has an open reservation")
            pending = sum(open_bounds.values(), Decimal("0"))
            if known_usage + pending + phase_bound > self.internal_limit_usd:
                raise ValueError("launch would exceed the internal USD 27.00 limit")
            self._append({
                "kind": "reserve", "attempt_id": attempt_id, "at": datetime.now(timezone.utc).isoformat(),
                "known_usage_usd": str(known_usage), "pending_before_usd": str(pending),
                "phase_bound_usd": str(phase_bound), "rates_sha256": rates_sha256,
            })

    def reconcile(self, attempt_id: str, *, reconciled_cost: Decimal, known_usage_after: Decimal) -> None:
        with self.lock:
            entries = self._entries()
            reservations = [
                entry for entry in entries
                if entry["kind"] == "reserve" and entry["attempt_id"] == attempt_id
            ]
            reconciliations = [
                entry for entry in entries
                if entry["kind"] == "reconcile" and entry["attempt_id"] == attempt_id
            ]
            if len(reservations) != 1 or reconciliations:
                raise ValueError("attempt must have exactly one open reservation")
            if reconciled_cost < 0 or known_usage_after < Decimal(reservations[0]["known_usage_usd"]):
                raise ValueError("reconciled metered costs must be nonnegative and monotonic")
            self._append({
                "kind": "reconcile", "attempt_id": attempt_id, "at": datetime.now(timezone.utc).isoformat(),
                "reconciled_cost_usd": str(reconciled_cost), "known_usage_after_usd": str(known_usage_after),
            })

    def verify_hash_chain(self) -> None:
        previous = "0" * 64
        for sequence, entry in enumerate(self._entries()):
            claimed = entry.pop("entry_sha256")
            assert entry["sequence"] == sequence and entry["previous_sha256"] == previous
            assert hashlib.sha256(_canonical(entry).encode()).hexdigest() == claimed
            previous = claimed
```

Implement reconciliation only when a fresh `modal billing summary --for "this month" --json` shows metered usage at least as high as the pre-launch value. If Modal billing is lagging, leave the reservation open, print `PENDING: billing data has not caught up; another launch is forbidden`, and exit 3. Never substitute `billed_cost`, credits, or the runner's estimate for the reconciled metered cost.

- [x] **Step 9: Run private-I/O, workspace, and ledger tests**

Run: `uv run pytest tests/unit/test_private_io.py tests/unit/test_workspace_guard.py tests/unit/test_cost_ledger.py -q`

Expected: all tests pass; permissive/symlinked state fails closed, mismatched/stale cap evidence fails closed, and `known + pending + new > 27.00` is rejected.

- [x] **Step 10: Commit the private state and two-layer cost guard**

```bash
git add configs/pilot/modal-budget.json src/ratemem/pilot/config.py src/ratemem/pilot/private_io.py src/ratemem/pilot/workspace.py src/ratemem/pilot/costs.py tests/fixtures/modal tests/unit/test_private_io.py tests/unit/test_workspace_guard.py tests/unit/test_cost_ledger.py
git commit -m "feat: enforce modal pilot cost guards"
```

### Task 11: Orchestrate only the allowed pilot probes and derive the held-in step cap from p95

**Files:**
- Create: `src/ratemem/pilot/probes.py`
- Create: `src/ratemem/pilot/runner.py`
- Create: `tests/unit/test_pilot_probes.py`
- Create: `tests/unit/test_pilot_runner.py`

- [ ] **Step 1: Write allowed-probe, percentile, and measured-cap tests**

```python
# tests/unit/test_pilot_probes.py
from decimal import Decimal

from ratemem.pilot.probes import ALLOWED_PROBES, held_in_step_cap, percentile


def test_probe_set_has_no_scientific_endpoint() -> None:
    assert ALLOWED_PROBES == (
        "checkpoint_compatibility", "dynamic_numerics", "gradient_flow", "frozen_backbone",
        "peak_memory", "one_step_inference", "one_timestep_backward", "step_timing", "held_in_loss",
    )
    forbidden = {"identity", "kid", "fid", "memory_policy", "lifecycle", "composition", "augmentation"}
    assert not any(token in probe for token in forbidden for probe in ALLOWED_PROBES)


def test_percentiles_use_nearest_rank() -> None:
    values = [float(value) for value in range(1, 21)]
    assert percentile(values, 0.50) == 10.0
    assert percentile(values, 0.95) == 19.0


def test_step_cap_is_minimum_of_dollars_and_timeout() -> None:
    cap = held_in_step_cap(
        p95_step_seconds=Decimal("2.0"), remaining_compute_usd=Decimal("4.00"),
        requested_resource_usd_per_second=Decimal("0.001"), remaining_timeout_seconds=3000,
        shutdown_reserve_seconds=120,
    )
    assert cap == 1440
```

- [ ] **Step 2: Run the probe tests and observe the missing module**

Run: `uv run pytest tests/unit/test_pilot_probes.py -q`

Expected: collection fails because `ratemem.pilot.probes` does not exist.

- [ ] **Step 3: Implement the closed probe set, CUDA recorder, and cap formula**

```python
# src/ratemem/pilot/probes.py
from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal

import torch

ALLOWED_PROBES = (
    "checkpoint_compatibility", "dynamic_numerics", "gradient_flow", "frozen_backbone",
    "peak_memory", "one_step_inference", "one_timestep_backward", "step_timing", "held_in_loss",
)


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0 < quantile <= 1:
        raise ValueError("percentile requires samples and a quantile in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def held_in_step_cap(
    *,
    p95_step_seconds: Decimal,
    remaining_compute_usd: Decimal,
    requested_resource_usd_per_second: Decimal,
    remaining_timeout_seconds: int,
    shutdown_reserve_seconds: int,
) -> int:
    if p95_step_seconds <= 0 or requested_resource_usd_per_second <= 0:
        raise ValueError("p95 and resource rate must be positive")
    usable_timeout = max(0, remaining_timeout_seconds - shutdown_reserve_seconds)
    seconds_from_cost = remaining_compute_usd / requested_resource_usd_per_second
    usable_seconds = min(Decimal(usable_timeout), seconds_from_cost)
    return max(0, int(usable_seconds // p95_step_seconds))


@dataclass(frozen=True)
class CudaPeak:
    allocated_bytes: int
    reserved_bytes: int


@contextmanager
def cuda_peak() -> Iterator[dict[str, CudaPeak]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paid pilot")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result: dict[str, CudaPeak] = {}
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        result["peak"] = CudaPeak(
            allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.max_memory_reserved(),
        )


def timed(callable_: object) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()  # type: ignore[operator]
    torch.cuda.synchronize()
    return time.perf_counter() - started
```

- [ ] **Step 4: Write a fake-backed runner sequencing test**

```python
# tests/unit/test_pilot_runner.py
from decimal import Decimal
from pathlib import Path

from ratemem.pilot.runner import PilotLimits, PilotRunner


class FakeBackend:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.losses = [1.2, 0.8]

    def compatibility(self) -> dict[str, object]: self.events.append("compatibility"); return {"status": "pass"}
    def inference(self) -> dict[str, object]: self.events.append("inference"); return {"status": "pass"}
    def backward(self) -> float: self.events.append("backward"); return 1.0
    def training_step(self) -> float: self.events.append("training_step"); return 0.01
    def evaluate_loss(self) -> float: self.events.append("evaluate_loss"); return self.losses.pop(0)
    def save_checkpoint(self, path: Path) -> str: self.events.append("checkpoint"); path.write_bytes(b"state"); return "1" * 64


def test_runner_has_one_inference_one_probe_backward_and_exact_timing_counts(tmp_path: Path) -> None:
    backend = FakeBackend()
    runner = PilotRunner(
        backend=backend,
        limits=PilotLimits(
            warmup_steps=10,
            measured_steps=20,
            held_in_allocation_usd=Decimal("0.03"),
            resource_usd_per_second=Decimal("1.00"),
            timeout_seconds=7200,
            shutdown_reserve_seconds=120,
        ),
    )
    result = runner.run(tmp_path)
    assert backend.events.count("inference") == 1
    assert backend.events.count("backward") == 1
    assert backend.events.count("training_step") == 33
    assert backend.events.count("evaluate_loss") == 2
    assert result["initial_flow_loss"] == 1.2 and result["final_flow_loss"] == 0.8
    assert result["held_in_loss"]["status"] == "pass"
```

- [ ] **Step 5: Implement the ordered runner with artifact-on-failure semantics**

```python
# src/ratemem/pilot/runner.py (orchestration boundary)
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from ratemem.pilot.probes import ALLOWED_PROBES, held_in_step_cap, percentile


class PilotBackend(Protocol):
    def compatibility(self) -> dict[str, object]:
        raise NotImplementedError

    def inference(self) -> dict[str, object]:
        raise NotImplementedError

    def backward(self) -> float:
        raise NotImplementedError

    def training_step(self) -> float:
        raise NotImplementedError

    def evaluate_loss(self) -> float:
        raise NotImplementedError

    def save_checkpoint(self, path: Path) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class PilotLimits:
    warmup_steps: int
    measured_steps: int
    held_in_allocation_usd: Decimal
    resource_usd_per_second: Decimal
    timeout_seconds: int
    shutdown_reserve_seconds: int


class PilotRunner:
    def __init__(self, *, backend: PilotBackend, limits: PilotLimits) -> None:
        if limits.warmup_steps != 10 or limits.measured_steps != 20:
            raise ValueError("pilot timing contract requires 10 warm-up and 20 measured steps")
        self.backend = backend
        self.limits = limits

    def run(self, artifact_dir: Path) -> dict[str, object]:
        run_started = time.monotonic()
        results: dict[str, object] = {"allowed_probe_names": list(ALLOWED_PROBES)}
        results["checkpoint_compatibility"] = self.backend.compatibility()
        results["one_step_inference"] = self.backend.inference()
        results["one_timestep_backward"] = {"status": "pass", "loss": self.backend.backward()}
        for _ in range(self.limits.warmup_steps):
            self.backend.training_step()
        measured: list[float] = []
        for _ in range(self.limits.measured_steps):
            measured.append(self.backend.training_step())
        p50 = percentile(measured, 0.50)
        p95 = percentile(measured, 0.95)
        elapsed_seconds = math.ceil(time.monotonic() - run_started)
        cap = held_in_step_cap(
            p95_step_seconds=Decimal(str(p95)),
            remaining_compute_usd=self.limits.held_in_allocation_usd,
            requested_resource_usd_per_second=self.limits.resource_usd_per_second,
            remaining_timeout_seconds=max(0, self.limits.timeout_seconds - elapsed_seconds),
            shutdown_reserve_seconds=self.limits.shutdown_reserve_seconds,
        )
        initial_loss = self.backend.evaluate_loss()
        for _ in range(cap):
            self.backend.training_step()
        final_loss = self.backend.evaluate_loss()
        checkpoint_path = artifact_dir / "trainable.safetensors"
        checkpoint_sha256 = self.backend.save_checkpoint(checkpoint_path)
        results.update({
            "warmup_steps": 10, "measured_steps": 20,
            "p50_step_seconds": p50, "p95_step_seconds": p95,
            "held_in_step_cap": cap,
            "initial_flow_loss": initial_loss, "final_flow_loss": final_loss,
            "held_in_loss": {"status": "pass" if final_loss < initial_loss else "fail"},
            "transformer_passes_per_step": 1, "checkpoint_sha256": checkpoint_sha256,
        })
        return results
```

Implement `RealSanaPilotBackend` in the same module with these exact behaviors:

1. `compatibility()` loads the pinned components, verifies the resolved SANA and DINO commit SHAs, asserts 20 blocks/120 qkv wrappers/8,601,600 atom parameters, enables activation checkpointing, records package versions, and snapshots every frozen parameter version counter.
2. `inference()` calls the pinned pipeline exactly once with prompt `A studio photograph of a red ceramic teapot on a plain gray table.`, seed `20260824`, height/width 1024, guidance 4.5, and `num_inference_steps=1`; it records latency and peak CUDA bytes but makes no quality claim.
3. `backward()` consumes one cached held-in query, creates a CUDA generator with seed `20260825`, and calls `OneTimestepFlowTrainer.train_step()` once. It asserts code/amortizer/atom gradients are present, backbone gradients are absent, and all frozen version counters are unchanged.
4. `training_step()` consumes the next cached held-in row cyclically, passes exactly one `train_step()` to `ratemem.pilot.probes.timed()`, and returns that synchronized CUDA wall time; BF16 and activation checkpointing remain enabled.
5. `evaluate_loss()` uses fixed row order, fixed timestep indices, fixed noise seeds, no optimizer update, and one transformer pass per row so the before/after losses are paired.
6. `save_checkpoint()` uses `save_trainable_checkpoint()` and writes no frozen SANA, VAE, text, or DINO weight.

Wrap `PilotRunner.run()` at the remote boundary in `try/except/finally`: classify CUDA OOM as `status="oom"`, every other exception as `status="exception"`, always write and commit `attempt.pending.json`, and never substitute A100 or another GPU. A held-in loss that does not fall is `status="probe_failed"`; it falsifies this implementation/configuration only and remains in the artifact.

- [ ] **Step 6: Run the probe and runner tests**

Run: `uv run pytest tests/unit/test_pilot_probes.py tests/unit/test_pilot_runner.py -q`

Expected: `4 passed`; the fake backend records 1 inference, 1 standalone backward, 10 warm-up steps, 20 measured steps, and only the measured p95-derived held-in count.

- [ ] **Step 7: Commit the pilot-only orchestration**

```bash
git add src/ratemem/pilot/probes.py src/ratemem/pilot/runner.py tests/unit/test_pilot_probes.py tests/unit/test_pilot_runner.py
git commit -m "feat: orchestrate capped sana pilot probes"
```

### Task 12: Define one synchronous, single-L40S, zero-retry Modal job

**Files:**
- Create: `src/ratemem/pilot/modal_app.py`
- Create: `tests/contract/test_modal_app_contract.py`

- [ ] **Step 1: Write a static Modal resource/invocation contract**

```python
# tests/contract/test_modal_app_contract.py
import ast
from pathlib import Path


def test_modal_source_has_one_sync_call_and_no_fanout_or_deployment() -> None:
    source = Path("src/ratemem/pilot/modal_app.py").read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    assert attributes.count("remote") == 1
    assert not ({"spawn", "map", "starmap", "deploy", "detach"} & set(attributes))
    assert 'gpu="L40S"' in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert "timeout=7200" in source
    assert "startup_timeout=1800" in source
    assert "create_if_missing=False" in source
    assert "execution-receipts" in source
    assert "lower_bound_may_miss_precommit_reschedule" in source
    assert '"attempt_id": request["attempt_id"]' in source
    assert '"workspace": request["workspace"]' in source
    assert '"source_sha256": request["source_sha256"]' in source
    assert '"slot_sha256": request["slot_sha256"]' in source
    assert '"submission_receipt_sha256": request["submission_receipt_sha256"]' in source
    assert "GLOBAL_SLOT_PATH" in source
    assert "GLOBAL_SUBMISSION_RECEIPT_PATH" in source
    assert "MODAL_ENVIRONMENT" in source
    assert "permit_path" not in source
```

- [ ] **Step 2: Run the Modal contract and observe the missing module**

Run: `uv run pytest tests/contract/test_modal_app_contract.py -q`

Expected: FAIL because `src/ratemem/pilot/modal_app.py` does not exist.

- [ ] **Step 3: Implement the immutable image and single remote function**

```python
# src/ratemem/pilot/modal_app.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "ratemem-sana-pilot"
CACHE_VOLUME_NAME = "ratemem-sana-cache"
ARTIFACT_VOLUME_NAME = "ratemem-pilot-artifacts"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(".", extras=["modal"], groups=[], frozen=True, uv_version="0.8.14")
    .add_local_python_source("ratemem")
    .add_local_dir("configs/pilot", "/opt/ratemem/configs/pilot")
    .add_local_dir("schemas", "/opt/ratemem/schemas")
    .workdir("/opt/ratemem")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "WANDB_MODE": "disabled"})
)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=False)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=7200,
    startup_timeout=1800,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
def run_first_pilot(request: dict[str, object]) -> dict[str, object]:
    import torch

    from ratemem.pilot.runner import run_real_pilot

    forbidden_names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY")
    present = [name for name in forbidden_names if os.environ.get(name)]
    if present:
        raise RuntimeError(f"forbidden credential variables are present: {present}")
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("pilot requires exactly one observed NVIDIA L40S")
    function_call_id = modal.current_function_call_id()
    input_id = modal.current_input_id()
    receipt_path = Path("/artifacts/execution-receipts") / f"{request['attempt_id']}.jsonl"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "attempt_id": request["attempt_id"],
        "workspace": request["workspace"],
        "source_sha256": request["source_sha256"],
        "slot_sha256": request["slot_sha256"],
        "submission_receipt_sha256": request["submission_receipt_sha256"],
        "function_call_id": function_call_id,
        "input_id": input_id,
        "task_id": os.environ.get("MODAL_TASK_ID"),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "semantics": "lower_bound_may_miss_precommit_reschedule",
    }
    with receipt_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    artifact_volume.commit()
    execution_receipt_count = len(receipt_path.read_text().splitlines())
    result = run_real_pilot(
        request=request,
        cache_root=Path("/cache"),
        artifact_root=Path("/artifacts"),
        modal_ids={
            "profile": "ratemem-pilot",
            "workspace": request["workspace"],
            "environment": "main",
            "launch_attempt_id": request["attempt_id"],
            "launch_source_sha256": request["source_sha256"],
            "pilot_slot_sha256": request["slot_sha256"],
            "submission_receipt_sha256": request["submission_receipt_sha256"],
            "function_call_id": function_call_id,
            "input_id": input_id,
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "container_image_id": os.environ["MODAL_IMAGE_ID"],
            "execution_receipt_path": str(receipt_path),
            "execution_receipt_count": execution_receipt_count,
            "execution_receipt_semantics": "lower_bound_may_miss_precommit_reschedule",
        },
    )
    artifact_volume.commit()
    cache_volume.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    from ratemem.pilot.cli import source_tree_sha256
    from ratemem.pilot.one_shot import (
        GLOBAL_SLOT_PATH,
        GLOBAL_SUBMISSION_RECEIPT_PATH,
        PERMIT_PATH,
        consume_launch_request,
    )
    from ratemem.pilot.workspace import verify_fresh_attestation_file

    if os.environ.get("MODAL_PROFILE") != "ratemem-pilot":
        raise RuntimeError("MODAL_PROFILE must be ratemem-pilot")
    if os.environ.get("MODAL_ENVIRONMENT") != "main":
        raise RuntimeError("MODAL_ENVIRONMENT must be main")
    snapshot = verify_fresh_attestation_file(
        Path("artifacts/pilot/workspace-attestation.json")
    )
    request = consume_launch_request(
        PERMIT_PATH,
        slot=GLOBAL_SLOT_PATH,
        receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
        expected_workspace=snapshot.workspace,
        current_source_sha256=source_tree_sha256(),
    )
    result = run_first_pilot.remote(request)
    print(json.dumps(result, sort_keys=True))
```

The local entry point has no path override: it re-attests the workspace, hashes the current clean
source, and atomically creates the global submission receipt only after the slot, permit, workspace,
and source identities match exactly. This check finishes before `.remote()`; a crash after receipt
creation therefore fails closed and cannot be resubmitted. The code contains no GPU
list, deployment, schedule, detached execution, map/fan-out, or client-requested rerun. `retries=0`
disables retries for user-code failures, but Modal can reschedule infrastructure failures;
`execution-receipts` records every execution that reaches and commits the receipt, so its count is
explicitly a lower bound rather than proof of exactly one container attempt. There is no diagnosed
rerun path in this engineering-pilot plan.

- [ ] **Step 4: Run the static Modal contract**

Run: `uv run pytest tests/contract/test_modal_app_contract.py -q`

Expected: `1 passed`; AST inspection finds exactly one synchronous `.remote()` call.

- [ ] **Step 5: Commit the Modal job definition**

```bash
git add src/ratemem/pilot/modal_app.py tests/contract/test_modal_app_contract.py
git commit -m "feat: define single l40s pilot job"
```

### Task 13: Add the guarded CLI, one-shot launch script, reconciliation, and runbook

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/ratemem/pilot/one_shot.py`
- Create: `src/ratemem/pilot/cli.py`
- Create: `scripts/run_modal_pilot.sh`
- Create: `docs/runbooks/ratemem-sana-modal-pilot.md`
- Create: `tests/unit/test_pilot_cli.py`
- Create: `tests/unit/test_one_shot.py`
- Create: `tests/contract/test_launch_script.py`

- [ ] **Step 1: Write CLI/launch-script fail-closed tests**

```python
# tests/contract/test_launch_script.py
from pathlib import Path


def test_launch_script_has_preflight_before_exactly_one_modal_run() -> None:
    source = Path("scripts/run_modal_pilot.sh").read_text()
    assert source.count("modal run") == 1
    assert "ratemem-pilot preflight" in source
    assert source.index("ratemem-pilot preflight") < source.index("modal run")
    assert "MODAL_PROFILE=ratemem-pilot" in source
    assert "ratemem-pilot preflight" in source  # preflight claims the immutable global slot
    assert "--permit-path" not in source
    for forbidden in ("modal deploy", "modal serve", ".spawn", ".map", "--detach", "while ", "until "):
        assert forbidden not in source
```

```python
# tests/unit/test_pilot_cli.py
from pathlib import Path
from types import SimpleNamespace

import pytest
import ratemem.pilot.cli as pilot_cli

from ratemem.pilot.cli import (
    credential_findings,
    source_tree_sha256,
)


def test_source_hash_requires_a_clean_tree(monkeypatch) -> None:
    monkeypatch.setattr(
        pilot_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=b"?? untracked.py\n"),
    )
    with pytest.raises(ValueError, match="tracked, staged, or untracked"):
        source_tree_sha256()


def test_scanner_accepts_redacted_metadata_and_flags_secret_environment_assignment(tmp_path: Path) -> None:
    safe = tmp_path / "safe.json"
    safe.write_text('{"modal_profile":"ratemem-pilot","credential_values":"redacted"}')
    unsafe = tmp_path / "unsafe.env"
    unsafe.write_text("WANDB_API_" + "KEY" + chr(61) + "fixture-value")
    assert credential_findings([safe]) == []
    assert credential_findings([unsafe]) == [unsafe]
```

```python
# tests/unit/test_one_shot.py
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pytest

from ratemem.pilot.one_shot import (
    GLOBAL_LOCK_NAME,
    PilotIdentity,
    claim_global_pilot_slot,
    consume_launch_request,
    create_launch_permit,
    validate_consumed_launch_evidence,
)
from ratemem.pilot.private_io import file_sha256, read_private_json


ATTEMPT_A = "11111111-1111-4111-8111-111111111111"
ATTEMPT_B = "22222222-2222-4222-8222-222222222222"
SOURCE_A = "1" * 64
SOURCE_B = "2" * 64
WORKSPACE = "authorized-workspace"


def _consume_in_process(
    slot: str,
    permit: str,
    receipt: str,
    output: Any,
) -> None:
    try:
        consume_launch_request(
            Path(permit),
            slot=Path(slot),
            receipt=Path(receipt),
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE_A,
        )
    except FileExistsError:
        output.put("blocked")
    else:
        output.put("submitted")


def paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    state = tmp_path / "state"
    permits = tmp_path / "permits"
    state.mkdir(mode=0o700)
    permits.mkdir(mode=0o700)
    return (
        state / "modal-pilot-slot.json",
        permits / "launch-permit.json",
        state / "modal-pilot-submitted.json",
    )


def prepared(tmp_path: Path) -> tuple[Path, Path, Path, PilotIdentity]:
    slot, permit, receipt = paths(tmp_path)
    identity = PilotIdentity(ATTEMPT_A, WORKSPACE, SOURCE_A)
    claim_global_pilot_slot(slot, identity=identity)
    create_launch_permit(
        permit,
        slot=slot,
        receipt=receipt,
        identity=identity,
        known_usage_before_usd="1.25",
        phase_bound_usd="10.15",
        rates={
            "gpu_l40s_per_second": "0.000542",
            "cpu_core_per_second": "0.0000131",
            "memory_gib_per_second": "0.00000222",
            "volume_gib_month": "0.09",
        },
        rates_sha256="3" * 64,
    )
    return slot, permit, receipt, identity


def test_slot_and_receipt_bind_exact_attempt_workspace_source_and_hashes(tmp_path: Path) -> None:
    slot, permit, receipt, identity = prepared(tmp_path)
    claimed = read_private_json(slot)
    authorized = read_private_json(permit)
    assert (claimed["attempt_id"], claimed["workspace"], claimed["source_sha256"]) == (
        identity.attempt_id,
        identity.workspace,
        identity.source_sha256,
    )
    assert (authorized["attempt_id"], authorized["workspace"], authorized["source_sha256"]) == (
        identity.attempt_id,
        identity.workspace,
        identity.source_sha256,
    )
    assert authorized["slot_sha256"] == file_sha256(slot)
    request = consume_launch_request(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE_A,
    )
    submitted = read_private_json(receipt)
    assert (submitted["attempt_id"], submitted["workspace"], submitted["source_sha256"]) == (
        identity.attempt_id,
        identity.workspace,
        identity.source_sha256,
    )
    assert submitted["slot_sha256"] == file_sha256(slot)
    assert submitted["permit_sha256"] == file_sha256(permit)
    assert request["submission_receipt_sha256"] == file_sha256(receipt)
    assert slot.stat().st_mode & 0o777 == 0o600
    assert permit.stat().st_mode & 0o777 == 0o600
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert slot.with_name(GLOBAL_LOCK_NAME).stat().st_mode & 0o777 == 0o600
    assert receipt.parent.stat().st_mode & 0o777 == 0o700
    evidence = validate_consumed_launch_evidence(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE_A,
    )
    assert evidence == {
        "attempt_id": ATTEMPT_A,
        "workspace": WORKSPACE,
        "source_sha256": SOURCE_A,
        "slot_sha256": file_sha256(slot),
        "permit_sha256": file_sha256(permit),
        "submission_receipt_sha256": file_sha256(receipt),
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("attempt_id", ATTEMPT_B), ("workspace", "another-workspace"), ("source_sha256", SOURCE_B)],
)
def test_consume_rejects_any_permit_slot_identity_mismatch_before_receipt(
    tmp_path: Path, field: str, replacement: str
) -> None:
    slot, permit, receipt, _ = prepared(tmp_path)
    payload = read_private_json(permit)
    payload[field] = replacement
    permit.unlink()
    permit.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    permit.chmod(0o600)
    with pytest.raises(ValueError, match="slot, permit, workspace, and source identities must match exactly"):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE_A,
        )
    assert not receipt.exists()


@pytest.mark.parametrize(
    ("expected_workspace", "current_source_sha256"),
    [("another-workspace", SOURCE_A), (WORKSPACE, SOURCE_B)],
)
def test_consume_rejects_attested_workspace_or_current_source_mismatch_before_receipt(
    tmp_path: Path,
    expected_workspace: str,
    current_source_sha256: str,
) -> None:
    slot, permit, receipt, _ = prepared(tmp_path)
    with pytest.raises(ValueError, match="slot, permit, workspace, and source identities must match exactly"):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=expected_workspace,
            current_source_sha256=current_source_sha256,
        )
    assert not receipt.exists()


def test_consume_rejects_permissive_parent_or_file_before_receipt(tmp_path: Path) -> None:
    slot, permit, receipt, _ = prepared(tmp_path)
    permit.chmod(0o644)
    with pytest.raises(PermissionError):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE_A,
        )
    assert not receipt.exists()

    permit.chmod(0o600)
    receipt.parent.chmod(0o755)
    with pytest.raises(PermissionError):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE_A,
        )
    assert not receipt.exists()


def test_serial_second_consume_fails_closed(tmp_path: Path) -> None:
    slot, permit, receipt, _ = prepared(tmp_path)
    arguments = {
        "slot": slot,
        "receipt": receipt,
        "expected_workspace": WORKSPACE,
        "current_source_sha256": SOURCE_A,
    }
    consume_launch_request(permit, **arguments)
    with pytest.raises(FileExistsError, match="submission receipt already exists"):
        consume_launch_request(permit, **arguments)


def test_concurrent_consumers_produce_one_receipt_and_one_failure(tmp_path: Path) -> None:
    slot, permit, receipt, _ = prepared(tmp_path)

    def consume() -> str:
        try:
            consume_launch_request(
                permit,
                slot=slot,
                receipt=receipt,
                expected_workspace=WORKSPACE,
                current_source_sha256=SOURCE_A,
            )
        except FileExistsError:
            return "blocked"
        return "submitted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: consume(), range(2)))
    assert sorted(outcomes) == ["blocked", "submitted"]
    assert read_private_json(receipt)["attempt_id"] == ATTEMPT_A


def test_cross_process_consumers_produce_one_receipt_and_one_failure(tmp_path: Path) -> None:
    slot, permit, receipt, _ = prepared(tmp_path)
    context = get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_consume_in_process,
            args=(str(slot), str(permit), str(receipt), output),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert not process.is_alive()
        assert process.exitcode == 0
    assert sorted(output.get(timeout=5) for _ in processes) == ["blocked", "submitted"]
    assert read_private_json(receipt)["attempt_id"] == ATTEMPT_A


def test_global_slot_rejects_serial_and_concurrent_second_attempts(tmp_path: Path) -> None:
    slot, _, _ = paths(tmp_path)
    first = PilotIdentity(ATTEMPT_A, WORKSPACE, SOURCE_A)
    second = PilotIdentity(ATTEMPT_B, WORKSPACE, SOURCE_A)
    claim_global_pilot_slot(slot, identity=first)
    with pytest.raises(FileExistsError, match="pilot slot has already been claimed"):
        claim_global_pilot_slot(slot, identity=second)

    concurrent_state = tmp_path / "concurrent-state"
    concurrent_state.mkdir(mode=0o700)
    concurrent_slot = concurrent_state / "modal-pilot-slot.json"

    def claim(identity: PilotIdentity) -> str:
        try:
            claim_global_pilot_slot(concurrent_slot, identity=identity)
        except FileExistsError:
            return "blocked"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, (first, second)))
    assert sorted(outcomes) == ["blocked", "claimed"]
```

- [ ] **Step 2: Run CLI/script tests and observe missing files**

Run: `uv run pytest tests/unit/test_pilot_cli.py tests/unit/test_one_shot.py tests/contract/test_launch_script.py -q`

Expected: collection/file failures because the one-shot module, CLI, and launch script do not exist.

- [ ] **Step 3: Implement the immutable one-shot identity and consumption protocol**

```python
# src/ratemem/pilot/one_shot.py
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from ratemem.pilot.private_io import (
    canonical_json_bytes,
    file_sha256,
    private_lock,
    read_private_json,
    write_exclusive_private_json,
)

GLOBAL_STATE_DIRECTORY = Path("/home/ubuntu/.local/state/ratemem")
GLOBAL_SLOT_PATH = GLOBAL_STATE_DIRECTORY / "modal-pilot-slot.json"
GLOBAL_SUBMISSION_RECEIPT_PATH = GLOBAL_STATE_DIRECTORY / "modal-pilot-submitted.json"
GLOBAL_LOCK_NAME = "modal-pilot-one-shot.lock"
PERMIT_PATH = Path("artifacts/pilot/launch-permit.json")

RATE_KEYS = {
    "gpu_l40s_per_second",
    "cpu_core_per_second",
    "memory_gib_per_second",
    "volume_gib_month",
}
SLOT_KEYS = {
    "schema_version", "kind", "attempt_id", "workspace", "source_sha256", "claimed_at_utc",
}
PERMIT_KEYS = {
    "schema_version", "kind", "attempt_id", "workspace", "source_sha256", "slot_sha256",
    "profile", "environment", "workspace_budget_usd", "internal_limit_usd",
    "known_usage_before_usd", "phase_bound_usd", "rates", "rates_sha256",
}
RECEIPT_KEYS = {
    "schema_version", "kind", "attempt_id", "workspace", "source_sha256",
    "slot_sha256", "permit_sha256", "submitted_at_utc",
}


@dataclass(frozen=True, slots=True)
class PilotIdentity:
    attempt_id: str
    workspace: str
    source_sha256: str

    def __post_init__(self) -> None:
        parsed = UUID(self.attempt_id)
        if parsed.version != 4 or str(parsed) != self.attempt_id:
            raise ValueError("attempt_id must be a canonical UUID4")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", self.workspace):
            raise ValueError("workspace must be a nonempty Modal workspace slug")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be lowercase SHA-256")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PilotIdentity":
        return cls(
            attempt_id=str(payload["attempt_id"]),
            workspace=str(payload["workspace"]),
            source_sha256=str(payload["source_sha256"]),
        )

    def fields(self) -> dict[str, str]:
        return {
            "attempt_id": self.attempt_id,
            "workspace": self.workspace,
            "source_sha256": self.source_sha256,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_paths(slot: Path, permit: Path, receipt: Path) -> None:
    if slot.name != "modal-pilot-slot.json":
        raise ValueError("unexpected global pilot-slot path")
    if permit.name != "launch-permit.json":
        raise ValueError("unexpected launch-permit path")
    if receipt != slot.with_name("modal-pilot-submitted.json"):
        raise ValueError("submission receipt must share the slot directory and fixed filename")


def _receipt_exists(receipt: Path) -> bool:
    return receipt.exists() or receipt.is_symlink()


def _read_slot(slot: Path) -> tuple[dict[str, Any], PilotIdentity]:
    payload = read_private_json(slot)
    if set(payload) != SLOT_KEYS or payload["schema_version"] != "1.0" or payload["kind"] != "pilot_slot":
        raise ValueError("global pilot slot has unexpected schema or keys")
    return payload, PilotIdentity.from_payload(payload)


def claim_global_pilot_slot(path: Path, *, identity: PilotIdentity) -> dict[str, Any]:
    if path.name != "modal-pilot-slot.json":
        raise ValueError("unexpected global pilot-slot path")
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "pilot_slot",
        **identity.fields(),
        "claimed_at_utc": _now(),
    }
    with private_lock(path.with_name(GLOBAL_LOCK_NAME)):
        if _receipt_exists(path.with_name("modal-pilot-submitted.json")):
            raise FileExistsError(
                "submission receipt already exists; no second pilot slot is allowed"
            )
        try:
            write_exclusive_private_json(path, payload)
        except FileExistsError as error:
            raise FileExistsError(
                "pilot slot has already been claimed; no second submission is allowed"
            ) from error
    return payload


def create_launch_permit(
    path: Path,
    *,
    slot: Path,
    receipt: Path,
    identity: PilotIdentity,
    known_usage_before_usd: str,
    phase_bound_usd: str,
    rates: Mapping[str, str],
    rates_sha256: str,
) -> dict[str, Any]:
    _validate_paths(slot, path, receipt)
    normalized_rates = {str(key): str(value) for key, value in rates.items()}
    if set(normalized_rates) != RATE_KEYS:
        raise ValueError("launch permit rates have unexpected keys")
    if hashlib.sha256(canonical_json_bytes(normalized_rates)).hexdigest() != rates_sha256:
        raise ValueError("launch permit rates hash mismatch")
    if Decimal(known_usage_before_usd) < 0 or Decimal(phase_bound_usd) <= 0:
        raise ValueError("launch permit costs must be nonnegative with a positive phase bound")
    with private_lock(slot.with_name(GLOBAL_LOCK_NAME)):
        if _receipt_exists(receipt):
            raise FileExistsError("submission receipt already exists; no second submission is allowed")
        _, slot_identity = _read_slot(slot)
        if slot_identity != identity:
            raise ValueError("slot and permit identities must match exactly")
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "kind": "launch_permit",
            **identity.fields(),
            "slot_sha256": file_sha256(slot),
            "profile": "ratemem-pilot",
            "environment": "main",
            "workspace_budget_usd": "28.00",
            "internal_limit_usd": "27.00",
            "known_usage_before_usd": known_usage_before_usd,
            "phase_bound_usd": phase_bound_usd,
            "rates": normalized_rates,
            "rates_sha256": rates_sha256,
        }
        write_exclusive_private_json(path, payload)
        return payload


def _validated_unsubmitted_locked(
    path: Path,
    *,
    slot: Path,
    receipt: Path,
    expected_workspace: str,
    current_source_sha256: str,
) -> tuple[dict[str, Any], PilotIdentity]:
    if _receipt_exists(receipt):
        raise FileExistsError("submission receipt already exists; no second submission is allowed")
    slot_payload, slot_identity = _read_slot(slot)
    permit = read_private_json(path)
    if set(permit) != PERMIT_KEYS or permit["schema_version"] != "1.0" or permit["kind"] != "launch_permit":
        raise ValueError("launch permit has unexpected schema or keys")
    permit_identity = PilotIdentity.from_payload(permit)
    expected_identity = PilotIdentity(
        attempt_id=permit_identity.attempt_id,
        workspace=expected_workspace,
        source_sha256=current_source_sha256,
    )
    if slot_identity != permit_identity or permit_identity != expected_identity:
        raise ValueError("slot, permit, workspace, and source identities must match exactly")
    if permit["slot_sha256"] != hashlib.sha256(canonical_json_bytes(slot_payload)).hexdigest():
        raise ValueError("launch permit slot hash mismatch")
    if permit["profile"] != "ratemem-pilot" or permit["environment"] != "main":
        raise ValueError("launch permit profile or environment changed")
    if permit["workspace_budget_usd"] != "28.00" or Decimal(str(permit["internal_limit_usd"])) != Decimal("27.00"):
        raise ValueError("launch permit budget constants changed")
    rates = permit["rates"]
    if not isinstance(rates, dict) or set(rates) != RATE_KEYS:
        raise ValueError("launch permit rates have unexpected keys")
    if hashlib.sha256(canonical_json_bytes(rates)).hexdigest() != permit["rates_sha256"]:
        raise ValueError("launch permit rates hash mismatch")
    return permit, permit_identity


def validate_unsubmitted_launch_permit(
    path: Path,
    *,
    slot: Path,
    receipt: Path,
    expected_workspace: str,
    current_source_sha256: str,
) -> dict[str, Any]:
    _validate_paths(slot, path, receipt)
    with private_lock(slot.with_name(GLOBAL_LOCK_NAME)):
        permit, _ = _validated_unsubmitted_locked(
            path,
            slot=slot,
            receipt=receipt,
            expected_workspace=expected_workspace,
            current_source_sha256=current_source_sha256,
        )
        return permit


def consume_launch_request(
    path: Path,
    *,
    slot: Path,
    receipt: Path,
    expected_workspace: str,
    current_source_sha256: str,
) -> dict[str, Any]:
    _validate_paths(slot, path, receipt)
    with private_lock(slot.with_name(GLOBAL_LOCK_NAME)):
        permit, identity = _validated_unsubmitted_locked(
            path,
            slot=slot,
            receipt=receipt,
            expected_workspace=expected_workspace,
            current_source_sha256=current_source_sha256,
        )
        submitted: dict[str, Any] = {
            "schema_version": "1.0",
            "kind": "submission_receipt",
            **identity.fields(),
            "slot_sha256": file_sha256(slot),
            "permit_sha256": file_sha256(path),
            "submitted_at_utc": _now(),
        }
        try:
            write_exclusive_private_json(receipt, submitted)
        except FileExistsError as error:
            raise FileExistsError(
                "submission receipt already exists; no second submission is allowed"
            ) from error
        return dict(permit) | {"submission_receipt_sha256": file_sha256(receipt)}


def validate_consumed_launch_evidence(
    path: Path,
    *,
    slot: Path,
    receipt: Path,
    expected_workspace: str,
    current_source_sha256: str,
) -> dict[str, str]:
    _validate_paths(slot, path, receipt)
    with private_lock(slot.with_name(GLOBAL_LOCK_NAME)):
        slot_payload, slot_identity = _read_slot(slot)
        permit = read_private_json(path)
        submitted = read_private_json(receipt)
        if (
            set(permit) != PERMIT_KEYS
            or permit["schema_version"] != "1.0"
            or permit["kind"] != "launch_permit"
        ):
            raise ValueError("launch permit has unexpected schema or keys")
        if (
            set(submitted) != RECEIPT_KEYS
            or submitted["schema_version"] != "1.0"
            or submitted["kind"] != "submission_receipt"
        ):
            raise ValueError("submission receipt has unexpected schema or keys")
        permit_identity = PilotIdentity.from_payload(permit)
        receipt_identity = PilotIdentity.from_payload(submitted)
        current_identity = PilotIdentity(
            permit_identity.attempt_id,
            expected_workspace,
            current_source_sha256,
        )
        if not (
            slot_identity == permit_identity == receipt_identity == current_identity
        ):
            raise ValueError("slot, permit, receipt, workspace, and source identities must match exactly")
        slot_sha256 = file_sha256(slot)
        permit_sha256 = file_sha256(path)
        receipt_sha256 = file_sha256(receipt)
        if permit["slot_sha256"] != slot_sha256:
            raise ValueError("launch permit slot hash mismatch")
        if submitted["slot_sha256"] != slot_sha256:
            raise ValueError("submission receipt slot hash mismatch")
        if submitted["permit_sha256"] != permit_sha256:
            raise ValueError("submission receipt permit hash mismatch")
        if slot_sha256 != hashlib.sha256(canonical_json_bytes(slot_payload)).hexdigest():
            raise ValueError("global pilot slot is not canonical")
        return {
            **current_identity.fields(),
            "slot_sha256": slot_sha256,
            "permit_sha256": permit_sha256,
            "submission_receipt_sha256": receipt_sha256,
        }
```

The slot, permit, and receipt are immutable. The global lock covers the complete read-compare-create critical section, while `O_EXCL` remains the final cross-process arbiter. Every identity comparison and permission check happens before the submission receipt is created. A crash after receipt creation intentionally burns the only submission; neither a new attempt ID, a second permit, a different workspace, nor unchanged source can authorize another `.remote()` call. Artifact validation later reopens all three owner-only records with `O_NOFOLLOW`, rechecks exact keys/identities/hashes under the same lock, and compares that evidence with the downloaded artifact.

- [ ] **Step 4: Implement exact Typer commands around the one-shot protocol**

```python
# src/ratemem/pilot/cli.py (command surface)
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import typer

from ratemem.pilot.costs import CostLedger, CostRates, ResourceContract, conservative_bound
from ratemem.pilot.artifacts import validate_attempt
from ratemem.pilot.one_shot import (
    GLOBAL_SLOT_PATH,
    GLOBAL_SUBMISSION_RECEIPT_PATH,
    PERMIT_PATH,
    PilotIdentity,
    claim_global_pilot_slot,
    create_launch_permit,
    validate_consumed_launch_evidence,
    validate_unsubmitted_launch_permit,
)
from ratemem.pilot.private_io import (
    canonical_json_bytes,
    read_private_json,
    write_atomic_private_json,
)
from ratemem.pilot.workspace import (
    capture_workspace_snapshot,
    verify_fresh_attestation_file,
    verify_workspace_snapshot,
)

app = typer.Typer(no_args_is_help=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def credential_findings(paths: list[Path]) -> list[Path]:
    assignment = re.compile(r"(?:MODAL_TOKEN_ID|MODAL_TOKEN_SECRET|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|WANDB_API_KEY)=[^\s]+")
    opaque_prefix = re.compile(
        r"(?:" + "a" + r"k-|" + "a" + r"s-|" + "h" + r"f_)[A-Za-z0-9_-]{12,}"
    )
    return [
        path
        for path in paths
        if path.is_file()
        and (
            assignment.search(path.read_text(errors="ignore"))
            or opaque_prefix.search(path.read_text(errors="ignore"))
        )
    ]


def source_tree_sha256() -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        check=True,
    )
    if completed.stdout:
        raise ValueError("pilot source tree has tracked, staged, or untracked changes")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True).stdout.strip()
    return _sha256_bytes(commit)


@app.command("attest-workspace")
def attest_workspace(evidence: Path, output: Path = Path("artifacts/pilot/workspace-attestation.json")) -> None:
    expected_workspace = typer.prompt("Type the exact authorized Modal workspace slug")
    confirmation = typer.prompt("Type the exact workspace usage budget shown in the dashboard")
    snapshot = capture_workspace_snapshot(evidence_path=evidence, confirmed_budget=confirmation)
    verify_workspace_snapshot(snapshot, expected_workspace=expected_workspace, max_age_seconds=900)
    write_atomic_private_json(output, snapshot.to_json())
    typer.echo(f"PASS workspace={expected_workspace} usage_budget_usd=28.00")


@app.command("preflight")
def preflight(
    attestation: Path = Path("artifacts/pilot/workspace-attestation.json"),
) -> None:
    snapshot = verify_fresh_attestation_file(attestation)
    rates = CostRates.normalize(snapshot.rates)
    resources = ResourceContract(1, 4, 32, 7200, 1800, 24, Decimal("2.00"))
    bound = conservative_bound(rates, resources)
    if bound > Decimal("21.00"):
        typer.echo("current rates exceed the USD 21.00 first-pilot allocation", err=True)
        raise typer.Exit(code=2)
    attempt_id = str(uuid.uuid4())
    source_sha256 = source_tree_sha256()
    identity = PilotIdentity(attempt_id, snapshot.workspace, source_sha256)
    claim_global_pilot_slot(GLOBAL_SLOT_PATH, identity=identity)
    rates_sha256 = _sha256_bytes(canonical_json_bytes(snapshot.rates))
    ledger = CostLedger(Path("artifacts/pilot/cost-ledger.jsonl"), internal_limit_usd=Decimal("27.00"))
    ledger.reserve(attempt_id, known_usage=Decimal(snapshot.known_metered_usage_usd), phase_bound=bound, rates_sha256=rates_sha256)
    create_launch_permit(
        PERMIT_PATH,
        slot=GLOBAL_SLOT_PATH,
        receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
        identity=identity,
        known_usage_before_usd=snapshot.known_metered_usage_usd,
        phase_bound_usd=str(bound),
        rates=snapshot.rates,
        rates_sha256=rates_sha256,
    )
    typer.echo(f"PASS attempt={attempt_id} bound_usd={bound} internal_limit_usd=27.00")


def main() -> None:
    app()
```

The slot claim deliberately precedes ledger reservation and permit creation. Any local crash or
validation failure after that claim burns the single pilot authorization and requires an explicit
author review; implementation must never delete, replace, or silently recover the marker.

Insert these helpers and commands before `main()` in the same module. Provisioning is free but is
still bound to the same attested workspace, immutable slot, exact source, and unsubmitted permit;
the field reader remains available after submission for download and reconciliation only:

```python
def _verified_unsubmitted_permit(
    attestation: Path = Path("artifacts/pilot/workspace-attestation.json"),
) -> dict[str, Any]:
    snapshot = verify_fresh_attestation_file(attestation)
    return validate_unsubmitted_launch_permit(
        PERMIT_PATH,
        slot=GLOBAL_SLOT_PATH,
        receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
        expected_workspace=snapshot.workspace,
        current_source_sha256=source_tree_sha256(),
    )


def _modal_cli_json(arguments: list[str]) -> object:
    environment = os.environ.copy()
    environment["MODAL_PROFILE"] = "ratemem-pilot"
    completed = subprocess.run(
        ["modal", *arguments, "--json"],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def _modal_cli(arguments: list[str]) -> None:
    environment = os.environ.copy()
    environment["MODAL_PROFILE"] = "ratemem-pilot"
    subprocess.run(["modal", *arguments], check=True, env=environment)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _validate_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    validate_attempt(payload)
    permit = read_private_json(PERMIT_PATH)
    launch = validate_consumed_launch_evidence(
        PERMIT_PATH,
        slot=GLOBAL_SLOT_PATH,
        receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
        expected_workspace=str(permit["workspace"]),
        current_source_sha256=source_tree_sha256(),
    )
    expected_launch = {
        "attempt_id": payload["attempt_id"],
        "workspace": payload["modal"]["workspace"],
        "source_sha256": payload["modal"]["launch_source_sha256"],
        "slot_sha256": payload["modal"]["pilot_slot_sha256"],
        "submission_receipt_sha256": payload["modal"]["submission_receipt_sha256"],
    }
    observed_launch = {key: launch[key] for key in expected_launch}
    if observed_launch != expected_launch:
        raise ValueError("artifact launch identity or receipt hashes differ from local evidence")
    root = path.parent
    checksum_path = root / "checksums.sha256"
    if _file_sha256(checksum_path) != payload["files"]["checksums_sha256"]:
        raise ValueError("checksums.sha256 digest does not match the artifact")
    for line in checksum_path.read_text().splitlines():
        digest, relative_name = line.split("  ", maxsplit=1)
        component = root / relative_name
        if component.parent != root or _file_sha256(component) != digest:
            raise ValueError(f"artifact checksum mismatch: {relative_name}")
    checkpoint = root / payload["checkpoint"]["path"]
    if _file_sha256(checkpoint) != payload["checkpoint"]["sha256"]:
        raise ValueError("trainable checkpoint checksum mismatch")
    findings = credential_findings(_artifact_files(root))
    if findings:
        raise ValueError(f"credential assignments found in {len(findings)} artifact files")
    return payload


@app.command("provision-volumes")
def provision_volumes(
    attestation: Path = Path("artifacts/pilot/workspace-attestation.json"),
) -> None:
    _verified_unsubmitted_permit(attestation)
    raw = _modal_cli_json(["volume", "list", "--env", "main"])
    if not isinstance(raw, list):
        raise ValueError("Modal volume list returned unexpected JSON")
    existing = {
        str(entry.get("name", entry.get("Name")))
        for entry in raw
        if isinstance(entry, dict)
    }
    required = ("ratemem-sana-cache", "ratemem-pilot-artifacts")
    for name in required:
        if name not in existing:
            _modal_cli(["volume", "create", "--env", "main", name])
    typer.echo("PASS volumes=ratemem-sana-cache,ratemem-pilot-artifacts")


@app.command("permit-field")
def permit_field(
    field: str,
) -> None:
    if field not in {"attempt_id", "workspace"}:
        raise typer.BadParameter("field must be attempt_id or workspace")
    typer.echo(str(read_private_json(PERMIT_PATH)[field]))


@app.command("validate-artifact")
def validate_artifact(path: Path) -> None:
    _validate_artifact(path)
    typer.echo(f"PASS artifact={path}")


@app.command("reconcile")
def reconcile(
    path: Path,
    attestation: Path = Path("artifacts/pilot/workspace-attestation.json"),
) -> None:
    payload = _validate_artifact(path)
    snapshot = verify_fresh_attestation_file(attestation)
    if payload["modal"]["workspace"] != snapshot.workspace:
        raise ValueError("artifact workspace differs from the current attested workspace")
    before = Decimal(payload["cost"]["known_usage_before_usd"])
    after = Decimal(snapshot.known_metered_usage_usd)
    if after <= before:
        typer.echo(
            "PENDING: billing data has not caught up; another launch is forbidden",
            err=True,
        )
        raise typer.Exit(code=3)
    reconciled_cost = after - before
    ledger = CostLedger(
        Path("artifacts/pilot/cost-ledger.jsonl"),
        internal_limit_usd=Decimal("27.00"),
    )
    ledger.reconcile(
        str(payload["attempt_id"]),
        reconciled_cost=reconciled_cost,
        known_usage_after=after,
    )
    final_payload = json.loads(json.dumps(payload))
    final_payload["cost"]["reconciliation_status"] = "reconciled"
    final_payload["cost"]["reconciled_cost_usd"] = str(reconciled_cost)
    validate_attempt(final_payload)
    final_path = path.with_name("attempt.json")
    if final_path.exists():
        raise FileExistsError(final_path)
    temporary = final_path.with_suffix(".json.tmp")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(final_payload, sort_keys=True, separators=(",", ":")).encode())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(final_path)
    typer.echo(f"PASS reconciled_cost_usd={reconciled_cost}")


@app.command("security-scan")
def security_scan(paths: list[Path]) -> None:
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    candidates = [Path(name) for name in changed]
    for path in paths:
        candidates.extend(_artifact_files(path) if path.is_dir() else [path])
    findings = credential_findings(sorted(set(candidates)))
    if findings:
        for finding in findings:
            typer.echo(str(finding), err=True)
        raise typer.Exit(code=4)
    typer.echo(f"PASS security_scan_files={len(set(candidates))}")
```

No CLI command accepts a token value, changes the global Modal profile, deploys an app, selects a GPU, changes the USD constants, or launches a remote call.

- [ ] **Step 5: Register the pilot CLI only after its target module exists**

After `src/ratemem/pilot/cli.py` and its unit tests are implemented, add the entry alongside the existing core script:

```toml
[project.scripts]
ratemem-pilot = "ratemem.pilot.cli:main"
```

Run: `uv lock && uv run ratemem-pilot --help`

Expected: exit 0 and Typer help headed by the guarded pilot commands. This smoke is intentionally in Task 13, not Task 1, so a default Task 1 environment never installs a broken console script whose target module has not been created.

- [ ] **Step 6: Add the only supported one-shot paid-launch script**

```bash
#!/usr/bin/env bash
# scripts/run_modal_pilot.sh
set -euo pipefail

export MODAL_PROFILE=ratemem-pilot
export MODAL_ENVIRONMENT=main

uv run ratemem-pilot preflight
uv run ratemem-pilot provision-volumes
RATEMEM_ATTEMPT_ID="$(uv run ratemem-pilot permit-field attempt_id)"
uv run modal run -m ratemem.pilot.modal_app
uv run modal volume get --env main --force ratemem-pilot-artifacts "attempts/${RATEMEM_ATTEMPT_ID}" "artifacts/pilot/${RATEMEM_ATTEMPT_ID}"
uv run ratemem-pilot validate-artifact "artifacts/pilot/${RATEMEM_ATTEMPT_ID}/attempt.pending.json"
uv run ratemem-pilot security-scan "artifacts/pilot/${RATEMEM_ATTEMPT_ID}"
```

Run: `chmod 0755 scripts/run_modal_pilot.sh`

Expected: no output. The script has one `modal run`, waits synchronously, and has no retry loop. If any step fails, `set -euo pipefail` stops immediately and never submits another function invocation.

- [ ] **Step 7: Write the exact dashboard/authentication/run/reconcile runbook**

Create `docs/runbooks/ratemem-sana-modal-pilot.md` with these ordered gates and expected output:

1. Before opening the dashboard, run `uv run python -c 'from pathlib import Path; from ratemem.pilot.private_io import ensure_private_directory; [ensure_private_directory(Path(value)) for value in ("/home/ubuntu/.local/share/ratemem/modal", "/home/ubuntu/.local/state/ratemem", "artifacts/pilot")]'`. Expected: exit 0; each absent directory is created mode 0700, while an existing symlink, foreign-owned directory, or directory with any mode other than 0700 fails without being repaired. Then, in a browser, select the one intended first authorized workspace. Before creating or selecting CLI credentials, open **Usage & Billing**, set **Workspace usage budget** to exactly **USD 28.00**, confirm the current billing-cycle metered usage, and save a new screenshot to `/home/ubuntu/.local/share/ratemem/modal/usage-budget-28.png`. If the dashboard cannot show and save the USD 28.00 cap, stop; no CLI authentication, volume creation, image build, or paid job is allowed.
2. Run `uv run python -c 'import os, stat; path="/home/ubuntu/.local/share/ratemem/modal/usage-budget-28.png"; descriptor=os.open(path, os.O_RDONLY | os.O_NOFOLLOW); metadata=os.fstat(descriptor); assert stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid() and metadata.st_nlink == 1; os.fchmod(descriptor, 0o600); os.fsync(descriptor); os.close(descriptor)'`, followed by `stat -c '%a %U %n' /home/ubuntu/.local/share/ratemem/modal /home/ubuntu/.local/state/ratemem artifacts/pilot /home/ubuntu/.local/share/ratemem/modal/usage-budget-28.png`. Expected: every directory reports mode `700` and the evidence file reports mode `600`, all owned by the current user. Any mismatch is a stop; do not chmod an existing foreign-owned or unexpectedly permissive one-shot state directory in place.
3. Configure exactly one of the user-supplied account-token pairs with `modal token set --profile ratemem-pilot --no-activate --verify`. Enter the token ID and token secret only at Modal's hidden interactive prompts. Expected: verification succeeds and the profile is not made globally active. No credential value is passed in an argument, written in the repository, echoed, logged, or recorded in shell history. Do not rotate through another supplied pair or select another workspace if verification fails; stop and ask the author. `modal token new --profile ratemem-pilot --no-activate --verify` is an optional browser-login alternative only when the author explicitly chooses it.
4. Run `chmod 600 /home/ubuntu/.modal.toml && MODAL_PROFILE=ratemem-pilot modal profile current`. Expected: exactly `ratemem-pilot`.
5. Run `MODAL_PROFILE=ratemem-pilot modal profile list --json`. Expected: the `ratemem-pilot` entry names the same workspace visible in the screenshot; do not redirect this command to a repository file.
6. Run `MODAL_PROFILE=ratemem-pilot uv run ratemem-pilot attest-workspace --evidence /home/ubuntu/.local/share/ratemem/modal/usage-budget-28.png`. At the prompts, type the displayed workspace slug and `28.00`. Expected: output starts with `PASS workspace=` and ends with `usage_budget_usd=28.00`; `artifacts/pilot/workspace-attestation.json` has mode 0600 and is ignored by version control.
7. Run the full local suite and static contracts: `uv run pytest -q -m "not paid_modal and not real_sana and not cuda" && uv run ruff check src tests && uv run mypy src/ratemem`. Expected: all selected tests pass and static tools exit 0.
8. Review `artifacts/pilot/workspace-attestation.json` by key name only, then run `scripts/run_modal_pilot.sh` once. Expected preflight atomically creates mode-0600 `/home/ubuntu/.local/state/ratemem/modal-pilot-slot.json` and `artifacts/pilot/launch-permit.json`; both bind the exact same `attempt_id`, `workspace`, and `source_sha256`, and the permit also binds the slot hash and `internal_limit_usd=27.00`. The local entry point re-attests the workspace and source, compares those values and hashes under the same mode-0600 global lock, then atomically creates mode-0600 `modal-pilot-submitted.json` before its one synchronous `.remote()` submission requesting one L40S. The receipt binds the same identity plus exact slot and permit hashes. A concurrent consumer and every serial later paid-script invocation fail closed before `.remote()`, even with a new UUID or after a client crash. There is no deployment or detached app. Modal may reschedule a container after infrastructure failure even though user-code retries are zero; do not treat `max_containers=1` as proof of one physical execution.
9. After the command returns, run `RATEMEM_ATTEMPT_ID="$(uv run ratemem-pilot permit-field attempt_id)"` and then `MODAL_PROFILE=ratemem-pilot uv run ratemem-pilot reconcile "artifacts/pilot/${RATEMEM_ATTEMPT_ID}/attempt.pending.json"`. Expected: either output starts with `PASS reconciled_cost_usd=` and a validated `attempt.json` exists, or the command exits 3 with `PENDING: billing data has not caught up; another launch is forbidden`. In the latter case, rerun only the free reconciliation command after Modal updates; never rerun the paid script.
10. On OOM, retain the artifact, release no reservation until billing reconciles, and stop. Do not edit the GPU string or substitute A100. Any A100 attempt requires an explicit new cost calculation and author authorization.
11. After the artifact and trainable checkpoint are downloaded, checksummed, and reconciled, delete the cache volume with `MODAL_PROFILE=ratemem-pilot modal volume delete --env main --yes ratemem-sana-cache`. This deletion is irreversible but the public frozen checkpoint is recoverable by revision; record the deletion time in the ledger. Keep the artifact volume until its local checksum is independently verified, then delete it with the analogous exact command. The unallocated USD 6 safety buffer authorizes no rerun.

The runbook must also state that reported billing can lag, storage may be billed for up to four days after deletion, W&B remains disabled, no Hugging Face token is needed, and no unredacted environment/configuration dump is permitted. It must identify the USD 27 ledger as a pre-launch admission bound rather than a realized-spend hard cap, identify the verified USD 28 Workspace usage budget as the hard outer stop, and require inspection of `execution_receipt_count` plus reconciliation of actual pre-credit metered usage after possible infrastructure rescheduling.

- [ ] **Step 8: Run CLI and launch-script contracts**

Run: `uv run pytest tests/unit/test_pilot_cli.py tests/unit/test_one_shot.py tests/contract/test_launch_script.py -q`

Expected: all tests pass; dirty or untracked source invalidates preflight, every attempt/workspace/source
mismatch is rejected before receipt creation, serial and concurrent second slot claims or submission
consumers fail atomically, private parent/file modes are exact, and the shell script has one guarded
`modal run` with no permit-path override.

- [ ] **Step 9: Commit the guarded operating surface**

```bash
git add pyproject.toml uv.lock src/ratemem/pilot/one_shot.py src/ratemem/pilot/cli.py scripts/run_modal_pilot.sh docs/runbooks/ratemem-sana-modal-pilot.md tests/unit/test_pilot_cli.py tests/unit/test_one_shot.py tests/contract/test_launch_script.py
git commit -m "feat: add guarded modal pilot workflow"
```

### Task 14: Execute the free verification gate and stop before payment

**Files:**
- Modify only if a test exposes a defect: files introduced by Tasks 1--13

- [ ] **Step 1: Run all free CPU tests**

Run: `uv run pytest -q -m "not paid_modal and not real_sana and not cuda"`

Expected: exit 0; every selected test passes and only explicitly excluded hardware/paid contracts are deselected.

- [ ] **Step 2: Run static analysis**

Run: `uv run ruff check src tests && uv run mypy src/ratemem`

Expected: exit 0, zero Ruff findings, and `Success: no issues found`.

- [ ] **Step 3: Verify exact pins, scope guards, and no credential-bearing files**

Run:

```bash
uv run pytest tests/unit/test_sana_dependency_contract.py tests/unit/test_pilot_config.py tests/unit/test_pilot_artifacts.py tests/unit/test_workspace_guard.py tests/unit/test_private_io.py tests/unit/test_cost_ledger.py tests/unit/test_one_shot.py tests/contract/test_modal_app_contract.py tests/contract/test_launch_script.py -q
uv run ratemem-pilot security-scan src tests configs schemas scripts
git status --short
```

Expected: all named tests pass; security scan output starts with `PASS security_scan_files=`; `git status --short` contains only the intended implementation files and no `artifacts/pilot`, cache, dataset, model, Modal config, or credential file.

- [ ] **Step 4: Audit the complete lock against both PyPI advisories and OSV**

Run:

```bash
uvx --from pip-audit==2.10.1 pip-audit --strict --no-deps -r <(uv export --frozen --all-groups --all-extras --no-emit-project --format requirements.txt --no-hashes)
uvx --from pip-audit==2.10.1 pip-audit --strict --no-deps --vulnerability-service osv -r <(uv export --frozen --all-groups --all-extras --no-emit-project --format requirements.txt --no-hashes)
```

Expected: both commands exit 0 with `No known vulnerabilities found`. The frozen export includes every dependency group and optional extra; `--no-deps` tells pip-audit to audit that already-complete exact graph without attempting a second resolution. Do not add an ignore for a direct or transitive advisory merely to make this gate pass; update the exact compatible pin and regenerate `uv.lock`, or document a true upstream no-fix blocker before proceeding. The second command is the independent OSV-backed view of the same complete uv lock.

- [ ] **Step 5: Run the CUDA contracts only inside the already-budgeted first pilot**

The first-pilot remote runner invokes:

Run: `RATEMEM_RUN_REAL_SANA=1 uv run pytest tests/contract/test_dynamic_atom_linear_cuda_memory.py tests/integration/test_real_sana_checkpoint.py -q`

Expected: both tests pass and each observed execution uses one L40S. Their runtime, peak memory, and exit status are included in the same first-pilot attempt artifact; this is not a separate client submission.

- [ ] **Step 6: Stop at the payment gate**

Do not run `scripts/run_modal_pilot.sh` during implementation or code review. Hand the completed free-test output and `docs/runbooks/ratemem-sana-modal-pilot.md` to the author. The paid invocation begins only after the dashboard cap evidence and exact-workspace attestation in Task 13; inability to verify either is a hard stop.

- [ ] **Step 7: Commit any verification-only corrections, then confirm a clean tree**

```bash
git add .gitignore pyproject.toml uv.lock configs/pilot schemas src/ratemem scripts docs/runbooks tests
git commit -m "test: verify sana modal pilot gates"
git status --short
```

Expected: the commit succeeds if verification required corrections; otherwise skip the commit. Final `git status --short` is empty.

## Paid pilot acceptance record

After separate author authorization and successful execution of the runbook, the first pilot is accepted only when all of the following are true:

- the dashboard evidence predates authentication and proves the selected workspace's usage budget is USD 28.00;
- the committed source tree is clean, including staged and non-ignored untracked files; its receipt binds the exact HEAD commit;
- mode-0700 owner-only parents contain one immutable mode-0600 global pilot slot, permit, lock, and submission receipt; slot and receipt bind the exact attempt ID, workspace, and source hash, the permit binds the slot hash, the receipt binds the slot and permit hashes, every value is compared before `.remote()`, and serial or concurrent second preflight/submission attempts fail even with another identity;
- the attested profile is `ratemem-pilot`, the workspace matches exactly, and every Modal process explicitly sets that profile;
- the cost ledger proves `known metered usage + all open worst-case reservations + the new bound <= USD 27.00` before any paid resource is created;
- the client makes exactly one synchronous `.remote()` submission requesting one L40S, with `retries=0` for user-code failures, `max_containers=1` concurrent container, and no fallback, fan-out, deployment, schedule, or detached execution; possible Modal infrastructure rescheduling is represented by a lower-bound execution receipt count and reconciled actual metered cost;
- checkpoint revision, 120-wrapper layout, zero/dense numerical equivalence, per-example coefficients, gradient routing, frozen parameters, no-dense-delta peak memory, and trainable save/load contracts pass;
- the run records one inference and one random-timestep backward, then 10 warm-up and 20 measured one-timestep steps; the p95 measurement determines the held-in step cap;
- a failed loss-reduction or OOM probe is preserved as a failure artifact and does not trigger another job;
- `attempt.pending.json`, all component files, and `trainable.safetensors` pass schema/hash/credential scans; the artifact launch identity and receipt hashes match the reopened local slot, permit, and submission receipt exactly; and `attempt.json` is created only from reconciled metered billing;
- the artifact remains `engineering_pilot_only` and `publication_eligible=false`, so no pilot number can enter the manuscript or support a scientific claim.
