"""Real SANA preprocessing and one-timestep resolver for RateMem meta-training."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import Tensor, nn

from ratemem.method.utility import CausalFeatureBatch
from ratemem.pilot.data import preprocess_query_image
from ratemem.sana.components import PinnedComponents
from ratemem.sana.flow import FlowBatch, flow_interpolate, flow_target
from ratemem.support.amortizer import AdapterPrediction, SupportAmortizer
from ratemem.support.features import encode_support_images, masked_mean_description
from ratemem.training.subjects_data import SubjectsPair


@runtime_checkable
class ActivatableAdapterBank(Protocol):
    def activate(self, coefficients: Tensor) -> AbstractContextManager[None]: ...


def _module_device(module: nn.Module, name: str) -> torch.device:
    devices = {tensor.device for tensor in (*module.parameters(), *module.buffers())}
    if len(devices) != 1:
        raise ValueError(f"{name} tensors must share exactly one device")
    return next(iter(devices))


def _text_features(
    texts: list[str],
    *,
    tokenizer: Any,
    encoder: nn.Module,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        max_length=300,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    if not isinstance(encoded, Mapping) or set(encoded) != {"input_ids", "attention_mask"}:
        raise TypeError("SANA tokenizer output fields changed")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
        raise TypeError("SANA tokenizer outputs must be tensors")
    if input_ids.shape != (len(texts), 300) or attention_mask.shape != input_ids.shape:
        raise ValueError("SANA tokenizer output shape changed")
    with torch.inference_mode():
        output: Any = encoder(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            return_dict=True,
        )
    hidden = getattr(output, "last_hidden_state", None)
    if not isinstance(hidden, Tensor) or hidden.shape != (len(texts), 300, 2304):
        raise ValueError("SANA text encoder output shape changed")
    if not bool(torch.isfinite(hidden).all().item()):
        raise ValueError("SANA text encoder produced non-finite features")
    return hidden.float(), attention_mask.to(device=device, dtype=torch.int64)


def preprocess_subjects_batch(
    pairs: Sequence[SubjectsPair],
    components: PinnedComponents,
    *,
    device: torch.device,
) -> FlowBatch:
    """Encode a local Subjects200K batch with frozen VAE, text, and DINO models."""

    values = tuple(pairs)
    if not values or any(type(pair) is not SubjectsPair for pair in values):
        raise TypeError("preprocessing requires a nonempty SubjectsPair sequence")
    if type(components) is not PinnedComponents:
        raise TypeError("components must be an exact PinnedComponents bundle")
    for module, name in (
        (cast(nn.Module, components.vae), "VAE"),
        (cast(nn.Module, components.text_encoder), "text encoder"),
        (cast(nn.Module, components.support_encoder), "support encoder"),
    ):
        if _module_device(module, name) != device:
            raise ValueError(f"{name} and preprocessing device differ")

    pixels = torch.stack(
        [preprocess_query_image(pair.query, resolution=1024) for pair in values]
    ).to(device=device, dtype=torch.float32)
    vae = cast(Any, components.vae)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        encoded: Any = vae.encode(pixels, return_dict=True)
    latent = getattr(encoded, "latent", None)
    if not isinstance(latent, Tensor) or latent.shape != (len(values), 32, 32, 32):
        raise ValueError("SANA VAE latent shape changed")
    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
    if type(scaling_factor) is not float or scaling_factor != 0.41407:
        raise ValueError("SANA VAE scaling factor changed")
    clean_latents = latent.float() * scaling_factor

    tokenizer = components.tokenizer
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ValueError("SANA tokenizer must begin with left padding")
    try:
        tokenizer.padding_side = "right"
        prompt, prompt_mask = _text_features(
            [pair.query_prompt.lower().strip() for pair in values],
            tokenizer=tokenizer,
            encoder=cast(nn.Module, components.text_encoder),
            device=device,
        )
        description_tokens, description_mask = _text_features(
            [pair.support_prompt.lower().strip() for pair in values],
            tokenizer=tokenizer,
            encoder=cast(nn.Module, components.text_encoder),
            device=device,
        )
    finally:
        tokenizer.padding_side = "left"
    description = masked_mean_description(description_tokens, description_mask)
    support = encode_support_images(
        [pair.support for pair in values],
        processor=components.support_processor,
        encoder=cast(nn.Module, components.support_encoder),
        device=device,
    ).to(device=device).unsqueeze(1)
    return FlowBatch(
        clean_latents=clean_latents,
        prompt_embeddings=prompt,
        prompt_attention_mask=prompt_mask,
        support_features=support,
        support_mask=torch.ones((len(values), 1), dtype=torch.bool, device=device),
        description_features=description,
    )


class SanaMetaResolver:
    """Bind one lifecycle segment to one prepared batch without extra transformer passes."""

    def __init__(
        self,
        transformer: nn.Module,
        adapter_bank: ActivatableAdapterBank,
        amortizer: SupportAmortizer,
        training_timesteps: tuple[float, ...],
        training_sigmas: tuple[float, ...],
        *,
        seed: int,
        group_size: int,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        if not isinstance(transformer, nn.Module):
            raise TypeError("transformer must be an nn.Module")
        if not isinstance(adapter_bank, ActivatableAdapterBank):
            raise TypeError("adapter_bank must implement activation")
        if type(amortizer) is not SupportAmortizer:
            raise TypeError("amortizer must be an exact SupportAmortizer")
        if len(training_timesteps) != 1000 or len(training_sigmas) != 1000:
            raise ValueError("SANA training schedule must contain exactly 1000 entries")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("resolver seed must be a nonnegative signed 64-bit integer")
        if type(group_size) is not int or group_size < 1:
            raise ValueError("resolver group size must be positive")
        if amortizer.projection_count * amortizer.atom_count % group_size:
            raise ValueError("amortizer code width must be divisible by group size")
        if autocast_dtype not in {None, torch.bfloat16}:
            raise ValueError("autocast dtype must be None or bfloat16")
        self.transformer = transformer
        self.adapter_bank = adapter_bank
        self.amortizer = amortizer
        self.timesteps = training_timesteps
        self.sigmas = training_sigmas
        self.seed = seed
        self.group_size = group_size
        self.autocast_dtype = autocast_dtype
        self._batches: dict[str, FlowBatch] = {}
        self._create_indices: dict[str, int] = {}
        self._targets: dict[str, Tensor] = {}
        self._active_context: AbstractContextManager[None] | None = None

    def bind(self, trace_id: str, create_event_index: int, batch: FlowBatch) -> None:
        if type(trace_id) is not str or len(trace_id) != 64:
            raise ValueError("bound trace id must be a SHA-256")
        if type(create_event_index) is not int or create_event_index < 0:
            raise ValueError("create event index must be nonnegative")
        if type(batch) is not FlowBatch:
            raise TypeError("bound batch must be an exact FlowBatch")
        if self._active_context is not None:
            raise RuntimeError("cannot bind a new batch before the previous backward")
        self._batches[trace_id] = batch
        self._create_indices[trace_id] = create_event_index
        self._targets.pop(trace_id, None)

    def target_code(self, trace_id: str, event_index: int) -> Tensor:
        batch = self._batches.get(trace_id)
        if batch is None or self._create_indices.get(trace_id) != event_index:
            raise KeyError("target code request is not bound to the current create event")
        prediction = self.amortizer(
            batch.support_features,
            batch.support_mask,
            batch.description_features,
        )
        if type(prediction) is not AdapterPrediction:
            raise TypeError("support amortizer returned an invalid prediction")
        target = prediction.coefficients.reshape(prediction.coefficients.shape[0], -1)
        self._targets[trace_id] = target
        return target

    def _draw(self, trace_id: str, event_index: int, batch: FlowBatch) -> tuple[Tensor, Tensor]:
        digest = hashlib.sha256(
            f"{self.seed}\0{trace_id}\0{event_index}".encode("ascii")
        ).digest()
        draw_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device=batch.clean_latents.device).manual_seed(draw_seed)
        noise = torch.randn(
            batch.clean_latents.shape,
            generator=generator,
            device=batch.clean_latents.device,
            dtype=torch.float32,
        )
        indices = torch.randint(
            0,
            1000,
            (batch.clean_latents.shape[0],),
            generator=generator,
            device=batch.clean_latents.device,
        )
        return noise, indices

    def one_timestep_flow_loss(
        self,
        trace_id: str,
        event_index: int,
        adapter_code: Tensor,
    ) -> Tensor:
        batch = self._batches.get(trace_id)
        if batch is None:
            raise KeyError("flow query is not bound to a prepared batch")
        if self._active_context is not None:
            raise RuntimeError("SANA resolver supports one bounded query before backward")
        device = batch.clean_latents.device
        noise, indices = self._draw(trace_id, event_index, batch)
        timesteps = torch.tensor(self.timesteps, device=device, dtype=torch.float32)[indices]
        sigmas = torch.tensor(self.sigmas, device=device, dtype=torch.float32)[indices]
        shaped = sigmas.reshape(sigmas.shape[0], 1, 1, 1)
        noisy = flow_interpolate(batch.clean_latents, noise, shaped)
        target = flow_target(batch.clean_latents, noise)
        activation = self.adapter_bank.activate(adapter_code)
        activation.__enter__()
        self._active_context = activation
        try:
            with torch.autocast(
                device_type=device.type,
                dtype=self.autocast_dtype,
                enabled=self.autocast_dtype is not None,
            ):
                output: Any = self.transformer(
                    hidden_states=noisy,
                    encoder_hidden_states=batch.prompt_embeddings,
                    encoder_attention_mask=batch.prompt_attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )
        except BaseException as error:
            self._active_context = None
            activation.__exit__(type(error), error, error.__traceback__)
            raise
        try:
            if type(output) is not tuple or len(output) != 1 or not isinstance(output[0], Tensor):
                raise TypeError("SANA transformer must return one tensor tuple")
            prediction = output[0]
            if prediction.shape != target.shape or not bool(
                torch.isfinite(prediction).all().item()
            ):
                raise ValueError("SANA transformer prediction is invalid")
            return (prediction.float() - target).square().flatten(start_dim=1).mean()
        except BaseException as error:
            self._active_context = None
            activation.__exit__(type(error), error, error.__traceback__)
            raise

    def backward(self, total: Tensor) -> None:
        """Run backward before releasing dynamic coefficients needed by checkpoint replay."""

        activation = self._active_context
        if activation is None:
            raise RuntimeError("SANA backward requires one active transformer query")
        try:
            torch.autograd.backward(total)
        except BaseException as error:
            self._active_context = None
            activation.__exit__(type(error), error, error.__traceback__)
            raise
        else:
            self._active_context = None
            activation.__exit__(None, None, None)

    def release_without_backward(self) -> None:
        """Release an inference-only activation after a no-grad flow measurement."""

        if torch.is_grad_enabled():
            raise RuntimeError("release_without_backward requires torch.no_grad()")
        activation = self._active_context
        if activation is None:
            raise RuntimeError("no inference activation is active")
        self._active_context = None
        activation.__exit__(None, None, None)

    def utility_supervision(
        self,
        trace_id: str,
        event_index: int,
    ) -> tuple[CausalFeatureBatch, Tensor, Tensor]:
        target = self._targets.get(trace_id)
        if target is None:
            raise KeyError("utility supervision requires the current target code")
        detached = target.detach().float()
        groups = detached.reshape(detached.shape[0], -1, self.group_size)
        energy = groups.square().mean(dim=-1)
        absolute = detached.abs()
        concept = torch.stack(
            (
                absolute.mean(dim=-1),
                detached.square().mean(dim=-1).sqrt(),
                absolute.amax(dim=-1),
                torch.ones(detached.shape[0], device=detached.device),
            ),
            dim=-1,
        )
        stages = torch.tensor((0.0, 1.0), device=detached.device).reshape(1, 2)
        incidence = torch.stack(
            (
                absolute.mean(dim=-1, keepdim=True).expand(-1, 2),
                detached.square().mean(dim=-1, keepdim=True).sqrt().expand(-1, 2),
                stages.expand(detached.shape[0], -1),
                torch.ones((detached.shape[0], 2), device=detached.device),
            ),
            dim=-1,
        )
        observed = torch.stack((energy * 0.65, energy * 0.35), dim=1)
        indices = torch.full(
            (detached.shape[0],),
            max(0, event_index - 1),
            dtype=torch.int64,
            device=detached.device,
        )
        allocation = torch.full_like(indices, event_index)
        features = CausalFeatureBatch(
            concept=concept,
            incidence=incidence,
            incidence_mask=torch.ones(
                incidence.shape[:2], dtype=torch.bool, device=detached.device
            ),
            maximum_source_event_index=indices,
            allocation_event_index=allocation,
        )
        return features, observed, torch.ones_like(observed, dtype=torch.bool)


__all__ = [
    "ActivatableAdapterBank",
    "SanaMetaResolver",
    "preprocess_subjects_batch",
]
