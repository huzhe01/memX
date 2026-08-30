from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from ratemem.data.prepare import PreparedDataset, PreparedEpisode
from ratemem.experiment.config import ExperimentConfig
from ratemem.experiment.protocol import EpisodeBatch, EvaluationMetrics, StepMetrics
from ratemem.runtime.distributed import DistributedContext


def _image_statistics(path: Path) -> tuple[float, float, float, float, float, float]:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (32, 32):
            raise ValueError("fixture images must be exact 32x32 RGB images")
        pixels = np.array(image, dtype=np.float32, copy=True) / np.float32(255.0)
    means = pixels.mean(axis=(0, 1), dtype=np.float64)
    standard_deviations = pixels.std(axis=(0, 1), dtype=np.float64)
    return cast(
        tuple[float, float, float, float, float, float],
        tuple(float(value) for value in np.concatenate((means, standard_deviations))),
    )


def _query_target(path: Path) -> tuple[float, float, float]:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (32, 32):
            raise ValueError("fixture images must be exact 32x32 RGB images")
        pixels = np.array(image, dtype=np.float32, copy=True) / np.float32(255.0)
    return cast(
        tuple[float, float, float],
        tuple(float(value) for value in pixels.mean(axis=(0, 1), dtype=np.float64)),
    )


def _batch(
    dataset: PreparedDataset,
    episodes: tuple[PreparedEpisode, ...],
    *,
    device: torch.device,
) -> EpisodeBatch:
    features = torch.tensor(
        [_image_statistics(dataset.root / episode.support_path) for episode in episodes],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        [_query_target(dataset.root / episode.query_path) for episode in episodes],
        dtype=torch.float32,
        device=device,
    )
    return EpisodeBatch(features=features, targets=targets)


def training_batch(
    dataset: PreparedDataset,
    config: ExperimentConfig,
    context: DistributedContext,
    *,
    zero_based_step: int,
) -> EpisodeBatch:
    if type(zero_based_step) is not int or zero_based_step < 0:
        raise ValueError("training step must be a nonnegative exact int")
    pool = tuple(episode for episode in dataset.episodes if episode.split == "train")
    if not pool:
        raise ValueError("prepared dataset has no training episodes")
    global_batch = config.batch_size * context.ranks.world_size
    offset = (
        config.seed
        + zero_based_step * global_batch
        + context.ranks.rank * config.batch_size
    ) % len(pool)
    selected = tuple(pool[(offset + index) % len(pool)] for index in range(config.batch_size))
    return _batch(dataset, selected, device=context.device)


def evaluation_batches(
    dataset: PreparedDataset,
    config: ExperimentConfig,
    context: DistributedContext,
    *,
    split: str,
) -> Iterator[EpisodeBatch]:
    if split not in {"validation", "test"}:
        raise ValueError("fixture evaluation split must be validation or test")
    pool = tuple(episode for episode in dataset.episodes if episode.split == split)
    local = pool[context.ranks.rank :: context.ranks.world_size]
    for offset in range(0, len(local), config.batch_size):
        yield _batch(
            dataset,
            local[offset : offset + config.batch_size],
            device=context.device,
        )


class FixtureExperiment:
    """Small deterministic learner that verifies orchestration, never paper quality."""

    def __init__(
        self,
        config: ExperimentConfig,
        context: DistributedContext,
    ) -> None:
        if type(config) is not ExperimentConfig or config.profile != "smoke":
            raise TypeError("fixture experiment requires an exact smoke ExperimentConfig")
        if type(context) is not DistributedContext:
            raise TypeError("fixture experiment requires an exact DistributedContext")
        self.config = config
        self.context = context
        torch.manual_seed(config.seed)
        base_model = nn.Sequential(
            nn.Linear(6, 8),
            nn.SiLU(),
            nn.Linear(8, 3),
        ).to(device=context.device, dtype=torch.float32)
        self.base_model = base_model
        if context.ranks.world_size > 1:
            device_ids = (
                [context.ranks.local_rank]
                if context.runtime.device.type == "cuda"
                else None
            )
            self.model: nn.Module = DistributedDataParallel(
                base_model,
                device_ids=device_ids,
            )
        else:
            self.model = base_model
        self.optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=config.learning_rate,
        )

    def train_step(self, batch: EpisodeBatch) -> StepMetrics:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(batch.features)
        loss = torch.nn.functional.mse_loss(prediction, batch.targets)
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        return StepMetrics(loss=float(loss.detach().cpu()))

    def evaluate(self, batch: EpisodeBatch) -> EvaluationMetrics:
        self.model.eval()
        with torch.inference_mode():
            prediction = self.model(batch.features)
            squared_error = torch.square(prediction - batch.targets).mean(dim=1).sum()
        return EvaluationMetrics(
            squared_error_sum=float(squared_error.cpu()),
            example_count=batch.features.shape[0],
        )

    def model_state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in sorted(self.base_model.state_dict().items())
        }

    def optimizer_state_dict(self) -> dict[str, object]:
        return cast(dict[str, object], self.optimizer.state_dict())

    def load_state_dicts(
        self,
        model_state: Mapping[str, Tensor],
        optimizer_state: Mapping[str, object],
    ) -> None:
        self.base_model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(cast(dict[str, Any], dict(optimizer_state)))

    def model_sha256(self) -> str:
        return hashlib.sha256(save(self.model_state_dict())).hexdigest()
