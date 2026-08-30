# memX Company PPU Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first complete, reproducible `bootstrap -> data -> smoke -> train -> evaluate -> report` execution path and a fail-closed PPU/PCCL launch surface without changing the existing Modal pilot.

**Architecture:** Add provider-neutral runtime, data, experiment, checkpoint, and reporting packages around the tested RateMem core. A deterministic fixture implements the same experiment protocol as the later SANA learner, which lets local CI prove the full orchestration path while real ZW810E validation remains an explicit company-hardware gate.

**Tech Stack:** Python 3.11, PyTorch, torch.distributed, Pydantic, PyYAML, safetensors, Hugging Face Datasets, Typer, GNU Make, Bash, Docker/OCI, pytest, Ruff, mypy.

---

## File map

- `src/ratemem/runtime/device.py` owns accelerator discovery and strict CPU/NVIDIA/PPU selection.
- `src/ratemem/runtime/distributed.py` owns rank environment validation and process-group lifetime.
- `src/ratemem/data/manifest.py` owns immutable YAML dataset manifests.
- `src/ratemem/data/fixture.py` owns deterministic offline smoke episodes.
- `src/ratemem/data/prepare.py` owns atomic prepared-index publication.
- `src/ratemem/experiment/config.py` owns canonical run configuration and hashes.
- `src/ratemem/experiment/checkpoint.py` owns atomic resumable fixture checkpoints.
- `src/ratemem/experiment/fixture.py` owns the deterministic protocol implementation.
- `src/ratemem/experiment/runner.py` owns train/evaluate/report orchestration.
- `src/ratemem/experiment/cli.py` exposes stable commands.
- `configs/data/smoke.yaml` and `configs/experiments/smoke.yaml` are runnable locked defaults.
- `docker/Dockerfile.ppu` builds on a caller-selected vendor PPU image without replacing torch.
- `scripts/bootstrap.sh`, `scripts/launch_train.sh`, and `Makefile` form the clone-and-run surface.
- `docs/runbooks/company-ppu.md` records exact operator and validation commands.

### Task 1: Add strict runtime discovery

**Files:**
- Create: `src/ratemem/runtime/__init__.py`
- Create: `src/ratemem/runtime/device.py`
- Create: `tests/unit/runtime/test_device.py`

- [x] **Step 1: Write the failing runtime tests**

```python
from ratemem.runtime.device import RuntimeProbe, resolve_runtime


def test_explicit_ppu_uses_pccl_without_marketing_name_dependency() -> None:
    runtime = resolve_runtime(
        "ppu",
        RuntimeProbe(True, 8, ("accelerator-0",) * 8, True, ("gloo", "pccl")),
    )
    assert runtime.kind == "ppu"
    assert runtime.device.type == "cuda"
    assert runtime.distributed_backend == "pccl"


def test_missing_ppu_never_falls_back_to_cpu() -> None:
    with pytest.raises(RuntimeError, match="requested PPU"):
        resolve_runtime("ppu", RuntimeProbe(False, 0, (), False, ("gloo",)))
```

- [x] **Step 2: Verify the tests fail for the missing package**

Run: `uv run pytest tests/unit/runtime/test_device.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'ratemem.runtime'`.

- [x] **Step 3: Implement the immutable resolver**

```python
@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    accelerator_available: bool
    device_count: int
    device_names: tuple[str, ...]
    bf16_supported: bool
    available_backends: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeviceRuntime:
    kind: Literal["cpu", "nvidia", "ppu"]
    device: torch.device
    distributed_backend: str
    device_count: int
    device_names: tuple[str, ...]
    bf16_supported: bool


def resolve_runtime(requested: str, probe: RuntimeProbe, *, backend_override: str | None = None) -> DeviceRuntime:
    if requested not in {"auto", "cpu", "nvidia", "ppu"}:
        raise ValueError("device must be one of auto, cpu, nvidia, or ppu")
    if requested == "cpu":
        return DeviceRuntime("cpu", torch.device("cpu"), backend_override or "gloo", 0, (), False)
    if not probe.accelerator_available or probe.device_count < 1:
        if requested != "auto":
            raise RuntimeError(f"requested {requested.upper()} accelerator is unavailable")
        return DeviceRuntime("cpu", torch.device("cpu"), backend_override or "gloo", 0, (), False)
    kind = "ppu" if requested == "ppu" else "nvidia"
    backend = backend_override or ("pccl" if kind == "ppu" else "nccl")
    if backend not in probe.available_backends:
        raise RuntimeError(f"distributed backend {backend} is unavailable")
    return DeviceRuntime(
        kind, torch.device("cuda"), backend, probe.device_count,
        probe.device_names, probe.bf16_supported,
    )
```

- [x] **Step 4: Run the focused runtime suite**

Run: `uv run pytest tests/unit/runtime/test_device.py -q`

Expected: all runtime tests pass.

- [x] **Step 5: Commit the runtime resolver**

```bash
git add src/ratemem/runtime tests/unit/runtime
git commit -m "feat: add provider-neutral accelerator runtime"
```

### Task 2: Add distributed environment and PPU preflight contracts

**Files:**
- Create: `src/ratemem/runtime/distributed.py`
- Create: `src/ratemem/runtime/preflight.py`
- Create: `tests/unit/runtime/test_distributed.py`
- Create: `tests/unit/runtime/test_preflight.py`

- [x] **Step 1: Write failing rank and preflight tests**

```python
def test_rank_environment_requires_consistent_world_size() -> None:
    with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE"):
        RankEnvironment.from_mapping({
            "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "4"
        }, visible_devices=8)


def test_preflight_rejects_world_size_larger_than_visible_devices() -> None:
    runtime = ppu_runtime(device_count=4)
    with pytest.raises(RuntimeError, match="visible device count"):
        validate_preflight(runtime, world_size=8, local_world_size=8)
```

- [x] **Step 2: Verify expected missing-symbol failures**

Run: `uv run pytest tests/unit/runtime/test_distributed.py tests/unit/runtime/test_preflight.py -q`

Expected: collection fails because the new modules do not exist.

- [x] **Step 3: Implement strict rank parsing and process-group ownership**

```python
@dataclass(frozen=True, slots=True)
class RankEnvironment:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, str], *, visible_devices: int) -> "RankEnvironment":
        names = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        present = tuple(name for name in names if name in values)
        if not present:
            return cls(0, 0, 1, 1)
        if present != names:
            missing = ", ".join(name for name in names if name not in values)
            raise ValueError(f"distributed environment is incomplete: {missing}")
        result = cls(*(int(values[name]) for name in names))
        if result.world_size < 1 or result.local_world_size < 1:
            raise ValueError("WORLD_SIZE and LOCAL_WORLD_SIZE must be positive")
        if not 0 <= result.rank < result.world_size:
            raise ValueError("RANK is outside WORLD_SIZE")
        if not 0 <= result.local_rank < result.local_world_size:
            raise ValueError("LOCAL_RANK is outside LOCAL_WORLD_SIZE")
        if visible_devices and result.local_world_size > visible_devices:
            raise ValueError("LOCAL_WORLD_SIZE exceeds visible devices")
        return result


@contextmanager
def distributed_session(runtime: DeviceRuntime, ranks: RankEnvironment) -> Iterator[DistributedContext]:
    owned = False
    if runtime.device.type == "cuda":
        torch.cuda.set_device(ranks.local_rank)
    if ranks.world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=runtime.distributed_backend, rank=ranks.rank, world_size=ranks.world_size
        )
        owned = True
    context = DistributedContext(runtime, ranks)
    try:
        yield context
        if ranks.world_size > 1:
            torch.distributed.barrier()
    finally:
        if owned and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
```

- [x] **Step 4: Implement preflight receipt generation**

```python
def validate_preflight(runtime: DeviceRuntime, *, world_size: int, local_world_size: int) -> None:
    if world_size < 1 or local_world_size < 1 or local_world_size > world_size:
        raise ValueError("distributed sizes are inconsistent")
    if runtime.kind != "cpu" and not runtime.bf16_supported:
        raise RuntimeError("production accelerator requires BF16 support")
    if runtime.kind != "cpu" and local_world_size > runtime.device_count:
        raise RuntimeError("local world size exceeds visible device count")
```

- [x] **Step 5: Run focused tests and commit**

```bash
uv run pytest tests/unit/runtime -q
git add src/ratemem/runtime tests/unit/runtime
git commit -m "feat: validate distributed ppu launches"
```

### Task 3: Add immutable data manifests and deterministic preparation

**Files:**
- Create: `src/ratemem/data/__init__.py`
- Create: `src/ratemem/data/manifest.py`
- Create: `src/ratemem/data/fixture.py`
- Create: `src/ratemem/data/prepare.py`
- Create: `configs/data/smoke.yaml`
- Create: `tests/unit/data/test_manifest.py`
- Create: `tests/integration/data/test_prepare_smoke.py`

- [x] **Step 1: Write failing manifest validation tests**

```python
def test_manifest_rejects_mutable_revision(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, revision="main")
    with pytest.raises(ValueError, match="immutable revision"):
        DatasetManifest.load(path)


def test_manifest_rejects_split_overlap(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, train=("a",), validation=("a",))
    with pytest.raises(ValueError, match="disjoint"):
        DatasetManifest.load(path)
```

- [x] **Step 2: Verify the data modules are missing**

Run: `uv run pytest tests/unit/data/test_manifest.py tests/integration/data/test_prepare_smoke.py -q`

Expected: collection fails for `ratemem.data`.

- [x] **Step 3: Implement strict YAML loading and canonical hashing**

```python
class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["memx-dataset-v1"]
    name: str
    revision: str
    license_spdx: str
    profile: Literal["smoke", "training", "evaluation"]
    splits: dict[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        result = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        identities = [item for values in result.splits.values() for item in values]
        if len(identities) != len(set(identities)):
            raise ValueError("dataset splits must be disjoint")
        if len(result.revision) != 40 or any(c not in "0123456789abcdef" for c in result.revision):
            raise ValueError("dataset requires an immutable revision")
        return result

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
```

- [x] **Step 4: Implement deterministic fixture episodes and atomic index publication**

```python
def prepare_smoke_dataset(manifest: DatasetManifest, root: Path) -> PreparedDataset:
    staging = root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, mode=0o700)
    records = tuple(
        generate_fixture_episode(name, staging) for name in manifest.splits["train"]
    )
    index = b"".join(
        canonical_json_bytes(record.as_dict()) + b"\n" for record in records
    )
    (staging / "episodes.jsonl").write_bytes(index)
    destination = root / manifest.sha256
    staging.rename(destination)
    return load_prepared_dataset(
        destination, expected_manifest_sha256=manifest.sha256
    )
```

- [x] **Step 5: Prove byte-for-byte deterministic preparation**

Run: `uv run pytest tests/unit/data tests/integration/data -q`

Expected: two independent prepared roots have identical index and image hashes.

- [x] **Step 6: Commit data preparation**

```bash
git add src/ratemem/data configs/data tests/unit/data tests/integration/data
git commit -m "feat: add deterministic dataset preparation"
```

### Task 4: Add canonical experiment configuration and atomic resume

**Files:**
- Create: `src/ratemem/experiment/__init__.py`
- Create: `src/ratemem/experiment/config.py`
- Create: `src/ratemem/experiment/checkpoint.py`
- Create: `configs/experiments/smoke.yaml`
- Create: `tests/unit/experiment/test_config.py`
- Create: `tests/unit/experiment/test_checkpoint.py`

- [x] **Step 1: Write failing config and checkpoint tests**

```python
def test_config_hash_changes_when_training_steps_change() -> None:
    first = smoke_config(max_steps=4)
    second = smoke_config(max_steps=5)
    assert first.sha256 != second.sha256


def test_incomplete_checkpoint_is_not_latest(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.write_interrupted_fixture(step=2)
    assert store.latest() is None
```

- [x] **Step 2: Run and observe missing experiment package failures**

Run: `uv run pytest tests/unit/experiment/test_config.py tests/unit/experiment/test_checkpoint.py -q`

Expected: collection fails for `ratemem.experiment`.

- [x] **Step 3: Implement strict configuration**

```python
class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["memx-experiment-v1"]
    profile: Literal["smoke", "sana-ratemem"]
    seed: int
    max_steps: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    checkpoint_every: int
    dataset_manifest: Path

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
```

- [x] **Step 4: Implement atomic checkpoint directories**

```python
class CheckpointStore:
    def save(self, state: CheckpointState) -> Path:
        staging = self.root / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, mode=0o700)
        write_checkpoint_payload(staging, state)
        validate_checkpoint_payload(staging, state.config_sha256)
        final = self.root / f"step-{state.step:08d}"
        staging.rename(final)
        write_atomic_json(self.root / "latest.json", {"path": final.name})
        return final

    def latest(self) -> CheckpointState | None:
        pointer = self.root / "latest.json"
        if not pointer.exists():
            return None
        return load_checkpoint_payload(self.root / read_latest_name(pointer))

# save() writes tensors and state into `.staging-<uuid>`, validates hashes,
# renames to `step-00000004`, then atomically replaces `latest.json`.
```

- [x] **Step 5: Verify corruption, config mismatch, and exact resume behavior**

Run: `uv run pytest tests/unit/experiment/test_config.py tests/unit/experiment/test_checkpoint.py -q`

Expected: all focused tests pass.

- [x] **Step 6: Commit configuration and resume storage**

```bash
git add src/ratemem/experiment configs/experiments tests/unit/experiment
git commit -m "feat: add immutable experiment checkpoints"
```

### Task 5: Implement the full offline train/evaluate/report slice

**Files:**
- Create: `src/ratemem/experiment/protocol.py`
- Create: `src/ratemem/experiment/fixture.py`
- Create: `src/ratemem/experiment/runner.py`
- Create: `src/ratemem/experiment/report.py`
- Create: `src/ratemem/experiment/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/integration/experiment/test_smoke_pipeline.py`
- Create: `tests/contract/experiment/test_cli.py`

- [x] **Step 1: Write a failing uninterrupted-versus-resume integration test**

```python
def test_resumed_fixture_matches_uninterrupted_run(tmp_path: Path) -> None:
    full = run_fixture(tmp_path / "full", stop_after=None)
    run_fixture(tmp_path / "resumed", stop_after=2)
    resumed = run_fixture(tmp_path / "resumed", resume="auto", stop_after=None)
    assert resumed.model_sha256 == full.model_sha256
    assert resumed.metrics_sha256 == full.metrics_sha256
```

- [x] **Step 2: Verify the pipeline API is absent**

Run: `uv run pytest tests/integration/experiment/test_smoke_pipeline.py -q`

Expected: collection fails for missing experiment runner symbols.

- [x] **Step 3: Define one experiment protocol and fixture implementation**

```python
class Experiment(Protocol):
    def train_step(self, batch: EpisodeBatch) -> StepMetrics:
        raise NotImplementedError

    def evaluate(self, batch: EpisodeBatch) -> EvaluationMetrics:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Tensor]:
        raise NotImplementedError

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        raise NotImplementedError


class FixtureExperiment:
    def __init__(self, seed: int, device: torch.device) -> None:
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(6, 8), nn.SiLU(), nn.Linear(8, 3)
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def train_step(self, batch: EpisodeBatch) -> StepMetrics:
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(batch.features)
        loss = torch.nn.functional.mse_loss(prediction, batch.targets)
        loss.backward()
        self.optimizer.step()
        return StepMetrics(loss=float(loss.detach().cpu()))
```

- [x] **Step 4: Implement rank-zero JSONL artifacts and deterministic resume**

```python
def train(config: ExperimentConfig, prepared: PreparedDataset, run_root: Path, runtime: DeviceRuntime, *, resume: str) -> TrainResult:
    experiment = FixtureExperiment(config.seed, runtime.device)
    store = CheckpointStore(run_root / "checkpoints")
    start = restore_if_requested(experiment, store, resume, config.sha256)
    metrics = MetricsWriter(run_root / "metrics.jsonl")
    for step in range(start, config.max_steps):
        batch = prepared.batch_for_step(
            step, seed=config.seed, device=runtime.device
        )
        observed = experiment.train_step(batch)
        metrics.append(step, observed)
        if (step + 1) % config.checkpoint_every == 0 or step + 1 == config.max_steps:
            store.save(capture_checkpoint(experiment, step + 1, config.sha256))
    return finalize_train_result(experiment, metrics, config, prepared)
```

- [x] **Step 5: Add evaluate/report commands and entry point**

```toml
[project.scripts]
memx = "ratemem.experiment.cli:main"
```

Run: `uv run memx data prepare --config configs/data/smoke.yaml --root /tmp/memx-data`

Expected: canonical JSON with `status="prepared"`.

- [x] **Step 6: Run the complete offline slice twice**

Run: `uv run pytest tests/integration/experiment tests/contract/experiment -q`

Expected: uninterrupted and resumed output hashes match.

- [x] **Step 7: Commit the executable experiment slice**

```bash
git add pyproject.toml uv.lock src/ratemem/experiment tests/integration/experiment tests/contract/experiment
git commit -m "feat: run resumable memx experiments"
```

### Task 6: Add container, launcher, Make targets, and operator documentation

**Files:**
- Create: `requirements/ppu.txt`
- Create: `docker/Dockerfile.ppu`
- Create: `scripts/bootstrap.sh`
- Create: `scripts/launch_train.sh`
- Create: `Makefile`
- Rewrite: `README.md`
- Create: `docs/runbooks/company-ppu.md`
- Create: `tests/contract/test_company_launch_surface.py`

- [ ] **Step 1: Write failing source-contract tests**

```python
def test_ppu_container_does_not_install_or_replace_torch() -> None:
    dockerfile = Path("docker/Dockerfile.ppu").read_text()
    assert "ARG PPU_BASE_IMAGE" in dockerfile
    assert not re.search(r"(?:pip|uv).*(?:install|sync).*torch", dockerfile)


def test_makefile_exposes_complete_operator_surface() -> None:
    makefile = Path("Makefile").read_text()
    for target in ("bootstrap:", "data:", "smoke:", "train:", "evaluate:", "report:"):
        assert target in makefile
```

- [ ] **Step 2: Verify launch-surface tests fail because files are absent**

Run: `uv run pytest tests/contract/test_company_launch_surface.py -q`

Expected: failure reading `docker/Dockerfile.ppu` or `Makefile`.

- [ ] **Step 3: Add vendor-preserving container and bootstrap**

```dockerfile
ARG PPU_BASE_IMAGE=ppu-training:1.7.0-pytorch2.8-ppu-py312-cu129-ubuntu24.04
FROM ${PPU_BASE_IMAGE}
WORKDIR /workspace/memx
COPY requirements/ppu.txt /tmp/requirements/ppu.txt
RUN python -m pip install --require-hashes -r /tmp/requirements/ppu.txt
COPY . /workspace/memx
RUN python -m pip install --no-deps -e /workspace/memx
ENTRYPOINT ["memx"]
```

- [ ] **Step 4: Add generic torchrun launcher**

```bash
exec torchrun \
  --nnodes="${NNODES:-1}" \
  --nproc-per-node="${LOCAL_WORLD_SIZE:-${WORLD_SIZE:-1}}" \
  --node-rank="${NODE_RANK:-0}" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-29500}" \
  -m ratemem.experiment.cli train "$@"
```

- [ ] **Step 5: Rewrite the README around the exact six-command path**

Document prerequisites, clone command, PPU image build, public/company-mirror data preparation,
single-node and multi-node examples, resume, outputs, test commands, and the explicit distinction
between locally verified CPU behavior and company-only PPU gates.

- [ ] **Step 6: Run shell, Make, and contract checks**

```bash
bash -n scripts/bootstrap.sh scripts/launch_train.sh
make help
uv run pytest tests/contract/test_company_launch_surface.py -q
```

Expected: shell syntax exits zero, help lists all six targets, and contracts pass.

- [ ] **Step 7: Commit the operator surface**

```bash
git add requirements docker scripts Makefile README.md docs/runbooks/company-ppu.md tests/contract/test_company_launch_surface.py
git commit -m "docs: add clone-to-run ppu workflow"
```

### Task 7: Run release-one verification and push

**Files:**
- Modify: `docs/superpowers/plans/2026-08-30-memx-company-ppu-pipeline.md`

- [ ] **Step 1: Execute the offline acceptance path in a temporary root**

```bash
make data PROFILE=smoke DATA_ROOT="$(mktemp -d)"
make smoke DEVICE=cpu DATA_ROOT="<same-root>" RUN_ROOT="$(mktemp -d)"
make evaluate DEVICE=cpu DATA_ROOT="<same-root>" RUN_ROOT="<same-run-root>"
make report RUN_ROOT="<same-run-root>"
```

Expected: every command exits zero and the report is marked `publication_eligible=false`.

- [ ] **Step 2: Run all local quality gates**

```bash
uv sync --all-extras --frozen
uv run ruff check src tests
uv run mypy src/ratemem
uv run pytest -q -m 'not paid_modal and not real_sana and not cuda and not ppu'
bash -n scripts/*.sh
git diff --check
```

Expected: every command exits zero; the pre-existing skip remains explicit.

- [ ] **Step 3: Record unobserved hardware gates truthfully**

Mark the CPU acceptance items complete. Leave single-, eight-, and sixteen-PPU receipts absent and
state in the runbook that they require execution on company ZW810E hardware.

- [ ] **Step 4: Commit verification documentation**

```bash
git add docs/superpowers/plans/2026-08-30-memx-company-ppu-pipeline.md docs/runbooks/company-ppu.md
git commit -m "docs: record memx release-one verification"
```

- [ ] **Step 5: Push the non-default development branch**

```bash
git push origin codex/ratemem-implementation
git ls-remote --heads origin refs/heads/codex/ratemem-implementation
```

Expected: the remote feature SHA equals local `HEAD`; `master` remains unchanged.
