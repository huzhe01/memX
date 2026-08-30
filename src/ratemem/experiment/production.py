"""Company-queue SANA/RateMem training orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from ratemem.adapters.sana_layout import (
    SanaDynamicAdapterBank,
    install_sana_dynamic_atoms,
)
from ratemem.data.subjects200k import (
    PreparedSubjects200KSnapshot,
    Subjects200KManifest,
)
from ratemem.evaluation.canonical import file_sha256
from ratemem.experiment.checkpoint import CheckpointState, CheckpointStore
from ratemem.experiment.production_config import ProductionExperimentConfig
from ratemem.method.codec import RateMemDifferentiableCodec
from ratemem.method.config import MethodPolicy
from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary
from ratemem.method.model import RateMemTrainableMethod
from ratemem.method.utility import NonnegativeUtilityCalibrator
from ratemem.pilot.config import SanaPilotConfig
from ratemem.runtime.distributed import (
    DistributedContext,
    all_reduce_gradients,
)
from ratemem.sana.components import (
    PinnedComponents,
    PinnedSnapshotPaths,
    hydrate_pinned_snapshots,
    load_pinned_components,
)
from ratemem.state.model import Incidence
from ratemem.state.serialization import bundle_cost_bytes
from ratemem.support.amortizer import SupportAmortizer
from ratemem.training.functional_state import FunctionalMemoryState
from ratemem.training.losses import LossWeights
from ratemem.training.meta_trainer import SequentialMetaTrainer
from ratemem.training.sana_meta import SanaMetaResolver, preprocess_subjects_batch
from ratemem.training.segments import FrozenTrainingEvent, TrainingSegment
from ratemem.training.subjects_data import SubjectsPair, iter_subjects_pairs

_RESULT_SCHEMA = "memx-ratemem-engineering-train-v1"
_PATH_TYPE = type(Path())


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _barrier(context: DistributedContext) -> None:
    if context.ranks.world_size > 1:
        if not torch.distributed.is_initialized():
            raise RuntimeError("distributed production training requires a process group")
        torch.distributed.barrier()


def _snapshot_path(data_root: Path, manifest: Subjects200KManifest) -> Path:
    return data_root / f"{manifest.name}-{manifest.sha256[:16]}"


def _hydrate_shared_models(
    config: SanaPilotConfig,
    model_root: Path,
    context: DistributedContext,
) -> PinnedSnapshotPaths:
    payload: list[str | None] = [None, None]
    if context.is_primary:
        paths = hydrate_pinned_snapshots(config, cache_dir=model_root)
        payload[:] = [str(paths.sana), str(paths.dino)]
    if context.ranks.world_size > 1:
        torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0] is None or payload[1] is None:
        raise RuntimeError("rank zero did not publish pinned model snapshot paths")
    return PinnedSnapshotPaths(sana=Path(payload[0]), dino=Path(payload[1]))


def _loss_weights(policy: MethodPolicy) -> LossWeights:
    return LossWeights(**policy.training.loss_weights)


def _candidate_costs(
    dictionary: GroupRVQDictionary,
    *,
    device: torch.device,
) -> Tensor:
    frozen = freeze_dictionary(dictionary)
    packet = frozen.packet(0, 0, 0)
    incidence = Incidence("h_training_slot_0000", packet.packet_id, 0)
    one_incidence_bundle_bytes = bundle_cost_bytes(packet, (incidence,))
    return torch.full(
        (dictionary.group_count, dictionary.stages),
        float(one_incidence_bundle_bytes),
        device=device,
        dtype=torch.float32,
    )


def _temperature(policy: MethodPolicy, step: int) -> float:
    progress = min(max(step, 0), policy.soft_codec.anneal_steps)
    fraction = progress / policy.soft_codec.anneal_steps
    initial = policy.soft_codec.initial_temperature
    final = policy.soft_codec.final_temperature
    return float(initial * (final / initial) ** fraction)


@dataclass(slots=True)
class ProductionStack:
    components: PinnedComponents
    adapter_bank: SanaDynamicAdapterBank
    method: RateMemTrainableMethod
    codec: RateMemDifferentiableCodec
    resolver: SanaMetaResolver
    optimizer: torch.optim.AdamW
    trainer: SequentialMetaTrainer
    candidate_cost_bytes: Tensor


def build_production_stack(
    config: ProductionExperimentConfig,
    method_policy: MethodPolicy,
    sana_config: SanaPilotConfig,
    snapshots: PinnedSnapshotPaths,
    context: DistributedContext,
) -> ProductionStack:
    """Load frozen encoders and construct every approved trainable component."""

    if method_policy.code.dimension != sana_config.code_dim:
        raise ValueError("method code dimension differs from the pinned SANA adapter layout")
    if method_policy.training.training_seeds != (17, 29, 43):
        raise ValueError("method policy training seeds changed")
    if config.seed not in method_policy.training.training_seeds:
        raise ValueError("experiment seed is absent from the method policy")
    torch.manual_seed(config.seed)
    components = load_pinned_components(
        sana_config,
        snapshots=snapshots,
        device=context.device,
    )
    transformer = components.transformer
    adapter_bank = install_sana_dynamic_atoms(
        cast(nn.Module, transformer),
        rank=sana_config.rank,
        atom_count=sana_config.atom_count,
        expected_blocks=sana_config.num_blocks,
    )
    enable_checkpointing = getattr(transformer, "enable_gradient_checkpointing", None)
    if not callable(enable_checkpointing):
        raise TypeError("SANA transformer does not expose gradient checkpointing")
    enable_checkpointing()
    cast(nn.Module, transformer).eval()
    amortizer = SupportAmortizer(
        support_dim=sana_config.support_feature_dim,
        description_dim=sana_config.text_feature_dim,
        hidden_dim=256,
        projection_count=sana_config.projection_count,
        atom_count=sana_config.atom_count,
        layers=2,
        heads=8,
    ).to(device=context.device, dtype=torch.float32)
    amortizer.train()
    group_count = method_policy.code.dimension // method_policy.codec.group_size
    dictionary = GroupRVQDictionary(
        group_count,
        method_policy.codec.group_size,
        method_policy.codec.rvq_stages,
        method_policy.codec.entries_per_stage,
    ).to(device=context.device, dtype=torch.float32)
    utility = NonnegativeUtilityCalibrator(
        concept_features=4,
        incidence_features=4,
        hidden=method_policy.utility.hidden_dimension,
        groups=group_count,
    ).to(device=context.device, dtype=torch.float32)
    method = RateMemTrainableMethod(
        tuple(adapter_bank.parameters()),
        amortizer,
        dictionary,
        utility,
    )
    codec = RateMemDifferentiableCodec(
        method.dictionary,
        group_size=method_policy.codec.group_size,
        base_bits=method_policy.codec.base_bits,
        gain_step=method_policy.codec.incidence_gain_step,
        maximum_packets=method_policy.codec.maximum_packets_per_concept,
    )
    resolver = SanaMetaResolver(
        cast(nn.Module, transformer),
        adapter_bank,
        method.amortizer,
        components.training_timesteps,
        components.training_sigmas,
        seed=config.seed,
        group_size=method_policy.codec.group_size,
        autocast_dtype=torch.bfloat16,
    )
    optimizer = torch.optim.AdamW(
        tuple(method.parameters()),
        lr=config.learning_rate,
        betas=sana_config.optimizer_betas,
        eps=sana_config.optimizer_eps,
        weight_decay=sana_config.optimizer_weight_decay,
    )
    trainer = SequentialMetaTrainer(
        codec,
        utility,
        optimizer,
        resolver,
        _loss_weights(method_policy),
        maximum_transformer_passes=method_policy.training.maximum_transformer_passes_per_segment,
        gradient_synchronizer=lambda parameters: all_reduce_gradients(parameters, context),
    )
    return ProductionStack(
        components=components,
        adapter_bank=adapter_bank,
        method=method,
        codec=codec,
        resolver=resolver,
        optimizer=optimizer,
        trainer=trainer,
        candidate_cost_bytes=_candidate_costs(dictionary, device=context.device),
    )


def _segment(
    config: ProductionExperimentConfig,
    context: DistributedContext,
    step: int,
    pairs: tuple[SubjectsPair, ...],
) -> TrainingSegment:
    identity = "\0".join(pair.row_sha256 for pair in pairs)
    trace_id = hashlib.sha256(
        f"ratemem-train\0{config.seed}\0{context.ranks.rank}\0{step}\0{identity}".encode()
    ).hexdigest()
    create_index = step * 2
    handle = f"h_train_r{context.ranks.rank:04d}_s{step % config.active_handle_slots:04d}"
    return TrainingSegment(
        trace_id=trace_id,
        segment_index=step,
        events=(
            FrozenTrainingEvent(
                create_index,
                "create",
                handle,
                support_image_ids=tuple(pair.support_sha256 for pair in pairs),
                description_id=hashlib.sha256(
                    "\0".join(pair.support_prompt for pair in pairs).encode()
                ).hexdigest(),
            ),
            FrozenTrainingEvent(
                create_index + 1,
                "read",
                handle,
                prompt_id=hashlib.sha256(
                    "\0".join(pair.query_prompt for pair in pairs).encode()
                ).hexdigest(),
                generation_seed=config.seed + step,
                has_training_query=True,
            ),
        ),
    )


def _next_batch(iterator: Any, batch_size: int) -> tuple[SubjectsPair, ...]:
    rows: list[SubjectsPair] = []
    for _ in range(batch_size):
        try:
            row = next(iterator)
        except StopIteration as error:
            raise RuntimeError("Subjects200K stream ended before max_steps") from error
        if type(row) is not SubjectsPair:
            raise TypeError("Subjects200K stream returned an invalid row")
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class ProductionTrainResult:
    status: Literal["completed"]
    step: int
    result_path: Path
    checkpoint_path: Path
    model_sha256: str


@dataclass(frozen=True, slots=True)
class ProductionEvaluationResult:
    status: Literal["completed"]
    result_path: Path
    model_sha256: str
    validation_flow_mse: float
    validation_batches: int


def train_production(
    config: ProductionExperimentConfig,
    data_root: Path,
    model_root: Path,
    run_root: Path,
    context: DistributedContext,
    *,
    resume: Literal["never", "auto"],
) -> ProductionTrainResult:
    """Run real SANA/RateMem optimization; all outputs remain engineering-only."""

    if type(config) is not ProductionExperimentConfig:
        raise TypeError("production training requires ProductionExperimentConfig")
    if any(type(path) is not _PATH_TYPE for path in (data_root, model_root, run_root)):
        raise TypeError("production roots must be exact pathlib.Path values")
    if resume not in {"never", "auto"}:
        raise ValueError("resume must be never or auto")
    if run_root.exists() or run_root.is_symlink():
        metadata = run_root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or run_root.is_symlink():
            raise ValueError("production run root must be a real directory")
        if resume == "never" and any(run_root.iterdir()):
            raise FileExistsError("production run exists; use RESUME=auto")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = Subjects200KManifest.load(config.dataset_manifest)
    snapshot = PreparedSubjects200KSnapshot.load(
        _snapshot_path(data_root, manifest),
        manifest,
    )
    sana_config = SanaPilotConfig.load(config.sana_config)
    method_policy = MethodPolicy.from_yaml(config.method_policy)
    snapshots = _hydrate_shared_models(sana_config, model_root, context)
    stack = build_production_stack(
        config,
        method_policy,
        sana_config,
        snapshots,
        context,
    )
    store = CheckpointStore(run_root / "checkpoints")
    loaded = (
        store.latest(
            expected_config_sha256=config.sha256,
            expected_dataset_sha256=snapshot.manifest_sha256,
        )
        if resume == "auto"
        else None
    )
    start_step = 0
    if loaded is not None:
        stack.method.load_state_dict(loaded.model_state, strict=True)
        stack.optimizer.load_state_dict(loaded.optimizer_state)
        torch.set_rng_state(loaded.torch_rng_state)
        start_step = loaded.step
    stream = iter_subjects_pairs(
        snapshot,
        manifest,
        partition="train",
        seed=config.seed,
        rank=context.ranks.rank,
        world_size=context.ranks.world_size,
        shuffle_buffer=config.shuffle_buffer,
    )
    iterator = iter(stream)
    for _ in range(start_step):
        _next_batch(iterator, config.batch_size)
    state = FunctionalMemoryState()
    metrics_path = run_root / f"metrics-rank-{context.ranks.rank:04d}.jsonl"
    if start_step == 0 and metrics_path.exists():
        raise FileExistsError("production metric journal exists without a resumed checkpoint")
    if start_step and len(metrics_path.read_bytes().splitlines()) != start_step:
        raise ValueError("production metric journal length differs from resumed checkpoint")
    final_checkpoint: Path | None = None
    for step in range(start_step, config.max_steps):
        started = time.monotonic()
        pairs = _next_batch(iterator, config.batch_size)
        batch = preprocess_subjects_batch(pairs, stack.components, device=context.device)
        segment = _segment(config, context, step, pairs)
        stack.resolver.bind(segment.trace_id, segment.events[0].event_index, batch)
        receipt = stack.trainer.train_segment(
            segment,
            state,
            temperature=_temperature(method_policy, step),
            candidate_cost_bytes=stack.candidate_cost_bytes,
            budget_bytes=config.memory_budget_bytes,
        )
        state = receipt.detached_state
        metric = {
            "schema_version": "memx-ratemem-step-v1",
            "step": step + 1,
            "trace_id": segment.trace_id,
            "loss": receipt.total_loss,
            "temperature": _temperature(method_policy, step),
            "transformer_passes": receipt.transformer_passes,
            "rank": context.ranks.rank,
            "wall_seconds": float(time.monotonic() - started),
        }
        with metrics_path.open("ab") as stream_handle:
            stream_handle.write(_canonical_json(metric) + b"\n")
            stream_handle.flush()
            os.fsync(stream_handle.fileno())
        checkpoint_step = step + 1
        must_checkpoint = (
            checkpoint_step % config.checkpoint_every == 0
            or checkpoint_step == config.max_steps
        )
        if must_checkpoint and context.is_primary:
            final_checkpoint = store.save(
                CheckpointState(
                    step=checkpoint_step,
                    config_sha256=config.sha256,
                    dataset_sha256=snapshot.manifest_sha256,
                    model_state=dict(stack.method.state_dict()),
                    optimizer_state=stack.optimizer.state_dict(),
                    torch_rng_state=torch.get_rng_state(),
                )
            )
        if must_checkpoint:
            _barrier(context)
    if context.is_primary:
        if final_checkpoint is None:
            raise RuntimeError("production training did not create a final checkpoint")
        checkpoint_manifest: Any = json.loads(
            (final_checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
        result_payload = {
            "schema_version": _RESULT_SCHEMA,
            "status": "completed",
            "publication_eligible": False,
            "reason": "requires real-hardware validation and frozen scientific evaluation",
            "step": config.max_steps,
            "config_sha256": config.sha256,
            "dataset_manifest_sha256": snapshot.manifest_sha256,
            "model_sha256": checkpoint_manifest["model_sha256"],
            "checkpoint": final_checkpoint.name,
            "runtime": context.runtime.as_manifest(),
            "distributed": context.ranks.as_manifest(),
            "trainable_parameters": stack.method.trainable_parameter_count(),
        }
        _atomic_write(run_root / "train-result.json", _canonical_json(result_payload))
    _barrier(context)
    result: Any = json.loads((run_root / "train-result.json").read_text(encoding="utf-8"))
    checkpoint_name = result["checkpoint"]
    return ProductionTrainResult(
        status="completed",
        step=result["step"],
        result_path=run_root / "train-result.json",
        checkpoint_path=run_root / "checkpoints" / checkpoint_name,
        model_sha256=result["model_sha256"],
    )


def evaluate_production(
    config: ProductionExperimentConfig,
    data_root: Path,
    model_root: Path,
    run_root: Path,
    context: DistributedContext,
) -> ProductionEvaluationResult:
    """Measure deterministic held-out concept flow loss from a completed checkpoint."""

    manifest = Subjects200KManifest.load(config.dataset_manifest)
    snapshot = PreparedSubjects200KSnapshot.load(
        _snapshot_path(data_root, manifest),
        manifest,
    )
    sana_config = SanaPilotConfig.load(config.sana_config)
    method_policy = MethodPolicy.from_yaml(config.method_policy)
    snapshots = _hydrate_shared_models(sana_config, model_root, context)
    stack = build_production_stack(
        config,
        method_policy,
        sana_config,
        snapshots,
        context,
    )
    checkpoint = CheckpointStore(run_root / "checkpoints").latest(
        expected_config_sha256=config.sha256,
        expected_dataset_sha256=snapshot.manifest_sha256,
    )
    if checkpoint is None or checkpoint.step != config.max_steps:
        raise RuntimeError("production evaluation requires the completed configured checkpoint")
    stack.method.load_state_dict(checkpoint.model_state, strict=True)
    stack.method.eval()
    stream = iter_subjects_pairs(
        snapshot,
        manifest,
        partition="validation",
        seed=config.seed + 10_000,
        rank=context.ranks.rank,
        world_size=context.ranks.world_size,
        shuffle_buffer=config.shuffle_buffer,
    )
    iterator = iter(stream)
    local_sum = 0.0
    local_count = 0
    for global_batch_index in range(
        context.ranks.rank,
        config.validation_batches,
        context.ranks.world_size,
    ):
        pairs = _next_batch(iterator, config.batch_size)
        batch = preprocess_subjects_batch(pairs, stack.components, device=context.device)
        segment = _segment(
            config,
            context,
            config.max_steps + global_batch_index,
            pairs,
        )
        stack.resolver.bind(segment.trace_id, segment.events[0].event_index, batch)
        with torch.no_grad():
            target = stack.resolver.target_code(
                segment.trace_id,
                segment.events[0].event_index,
            )
            encoding = stack.codec(
                target,
                temperature=method_policy.soft_codec.final_temperature,
                mode="ste",
            )
            loss = stack.resolver.one_timestep_flow_loss(
                segment.trace_id,
                segment.events[1].event_index,
                encoding.reconstruction,
            )
            stack.resolver.release_without_backward()
        local_sum += float(loss)
        local_count += 1
    aggregate = torch.tensor(
        (local_sum, float(local_count)),
        dtype=torch.float64,
        device=context.device,
    )
    if context.ranks.world_size > 1:
        torch.distributed.all_reduce(aggregate, op=torch.distributed.ReduceOp.SUM)
    total_sum = float(aggregate[0].cpu())
    total_count = int(aggregate[1].cpu())
    if total_count != config.validation_batches:
        raise RuntimeError("distributed validation batch count changed")
    metric = total_sum / total_count
    model_sha256 = file_sha256(
        run_root / "checkpoints" / f"step-{checkpoint.step:08d}" / "model.safetensors"
    )
    result_path = run_root / "evaluation-engineering.json"
    if context.is_primary:
        _atomic_write(
            result_path,
            _canonical_json(
                {
                    "schema_version": "memx-ratemem-engineering-evaluation-v1",
                    "status": "completed",
                    "publication_eligible": False,
                    "reason": "flow-loss engineering validation is not the frozen CVPR protocol",
                    "config_sha256": config.sha256,
                    "dataset_manifest_sha256": snapshot.manifest_sha256,
                    "model_sha256": model_sha256,
                    "validation_flow_mse": metric,
                    "validation_batches": total_count,
                    "runtime": context.runtime.as_manifest(),
                    "distributed": context.ranks.as_manifest(),
                }
            ),
        )
    _barrier(context)
    return ProductionEvaluationResult(
        status="completed",
        result_path=result_path,
        model_sha256=model_sha256,
        validation_flow_mse=metric,
        validation_batches=total_count,
    )


def prepare_models(sana_config_path: Path, model_root: Path) -> PinnedSnapshotPaths:
    """Hydrate and verify both pinned model repositories on a data/login worker."""

    return hydrate_pinned_snapshots(
        SanaPilotConfig.load(sana_config_path),
        cache_dir=model_root,
    )


__all__ = [
    "ProductionStack",
    "ProductionEvaluationResult",
    "ProductionTrainResult",
    "build_production_stack",
    "evaluate_production",
    "prepare_models",
    "train_production",
]
