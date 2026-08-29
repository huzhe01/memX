from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AdapterPrediction:
    logits: Tensor
    scales: Tensor
    coefficients: Tensor


@dataclass(frozen=True, slots=True)
class SupportAmortizerArchitecture:
    support_dim: int
    description_dim: int
    hidden_dim: int
    projection_count: int
    atom_count: int
    layers: int
    heads: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "support_dim",
            "description_dim",
            "hidden_dim",
            "projection_count",
            "atom_count",
            "layers",
            "heads",
        ):
            _positive_exact_int(name, getattr(self, name))
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

    @property
    def canonical(self) -> str:
        self.validate()
        return json.dumps(
            {
                "schema_version": "ratemem-support-amortizer-v1",
                "support_dim": self.support_dim,
                "description_dim": self.description_dim,
                "hidden_dim": self.hidden_dim,
                "projection_count": self.projection_count,
                "atom_count": self.atom_count,
                "layers": self.layers,
                "heads": self.heads,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def signature(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()


def _positive_exact_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class SupportAmortizer(nn.Module):
    """Predict dynamic adapter codes from a permutation-invariant support multiset.

    Support order and masked padding are irrelevant, while duplicate valid entries are
    intentional separate observations: multiplicity is preserved rather than deduplicated.
    """

    _ARCHITECTURE_FIELDS = frozenset(
        {
            "support_dim",
            "description_dim",
            "hidden_dim",
            "projection_count",
            "atom_count",
            "layers",
            "heads",
        }
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._ARCHITECTURE_FIELDS and "_architecture" in self.__dict__:
            raise AttributeError(f"{name} is immutable architecture state")
        super().__setattr__(name, value)

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
        architecture = SupportAmortizerArchitecture(
            support_dim=_positive_exact_int("support_dim", support_dim),
            description_dim=_positive_exact_int("description_dim", description_dim),
            hidden_dim=_positive_exact_int("hidden_dim", hidden_dim),
            projection_count=_positive_exact_int("projection_count", projection_count),
            atom_count=_positive_exact_int("atom_count", atom_count),
            layers=_positive_exact_int("layers", layers),
            heads=_positive_exact_int("heads", heads),
        )
        if architecture.hidden_dim % architecture.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        object.__setattr__(self, "_architecture", architecture)
        object.__setattr__(self, "_construction_architecture", architecture.canonical)

        self.support_projection = nn.Linear(self.support_dim, self.hidden_dim)
        self.description_projection = nn.Linear(self.description_dim, self.hidden_dim)
        self.support_type = nn.Parameter(torch.zeros(self.hidden_dim))
        self.description_type = nn.Parameter(torch.zeros(self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.layers,
            enable_nested_tensor=False,
        )
        self.pool_query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pool = nn.MultiheadAttention(
            self.hidden_dim,
            self.heads,
            dropout=0.0,
            batch_first=True,
        )
        self.head = nn.Linear(self.hidden_dim, self.projection_count * self.atom_count)
        self.raw_projection_scale = nn.Parameter(torch.zeros(self.projection_count, 1))
        object.__setattr__(self, "_construction_topology", self._topology_fingerprint())

    @property
    def architecture(self) -> SupportAmortizerArchitecture:
        architecture = self.__dict__.get("_architecture")
        if type(architecture) is not SupportAmortizerArchitecture:
            raise RuntimeError("amortizer architecture or topology was mutated")
        architecture.validate()
        construction = self.__dict__.get("_construction_architecture")
        if construction is not None and architecture.canonical != construction:
            raise RuntimeError("amortizer architecture or topology was mutated")
        return architecture

    @property
    def architecture_canonical(self) -> str:
        self._validate_architecture()
        return self.architecture.canonical

    @property
    def architecture_signature(self) -> str:
        self._validate_architecture()
        return self.architecture.signature

    @property
    def support_dim(self) -> int:
        return self.architecture.support_dim

    @property
    def description_dim(self) -> int:
        return self.architecture.description_dim

    @property
    def hidden_dim(self) -> int:
        return self.architecture.hidden_dim

    @property
    def projection_count(self) -> int:
        return self.architecture.projection_count

    @property
    def atom_count(self) -> int:
        return self.architecture.atom_count

    @property
    def layers(self) -> int:
        return self.architecture.layers

    @property
    def heads(self) -> int:
        return self.architecture.heads

    def _topology_fingerprint(self) -> tuple[object, ...]:
        def linear_configuration(linear: nn.Linear) -> tuple[object, ...]:
            return (
                type(linear),
                linear.in_features,
                linear.out_features,
                linear.bias is not None,
            )

        def dropout_configuration(dropout: nn.Dropout) -> tuple[object, ...]:
            return (type(dropout), dropout.p, dropout.inplace)

        def norm_configuration(norm: nn.LayerNorm) -> tuple[object, ...]:
            return (
                type(norm),
                tuple(norm.normalized_shape),
                norm.eps,
                norm.elementwise_affine,
                norm.weight is not None,
                norm.bias is not None,
            )

        def attention_configuration(attention: nn.MultiheadAttention) -> tuple[object, ...]:
            return (
                type(attention),
                attention.embed_dim,
                attention.num_heads,
                attention.dropout,
                attention.batch_first,
                attention.kdim,
                attention.vdim,
                attention._qkv_same_embed_dim,
                attention.head_dim,
                attention.add_zero_attn,
                attention.bias_k is not None,
                attention.bias_v is not None,
                attention.in_proj_weight is not None,
                attention.in_proj_bias is not None,
                attention.q_proj_weight is not None,
                attention.k_proj_weight is not None,
                attention.v_proj_weight is not None,
                linear_configuration(attention.out_proj),
            )

        module_topology = tuple(
            (name, id(module), type(module))
            for name, module in self.named_modules(remove_duplicate=False)
        )
        parameter_topology = tuple(
            (name, id(parameter), tuple(parameter.shape))
            for name, parameter in self.named_parameters(remove_duplicate=False)
        )
        buffer_topology = tuple(
            (name, id(buffer), tuple(buffer.shape))
            for name, buffer in self.named_buffers(remove_duplicate=False)
        )
        layer_configuration = tuple(
            (
                type(layer),
                attention_configuration(layer.self_attn),
                linear_configuration(layer.linear1),
                linear_configuration(layer.linear2),
                norm_configuration(layer.norm1),
                norm_configuration(layer.norm2),
                layer.norm_first,
                dropout_configuration(layer.dropout),
                dropout_configuration(layer.dropout1),
                dropout_configuration(layer.dropout2),
                id(layer.activation),
                getattr(layer.activation, "__module__", None),
                getattr(layer.activation, "__qualname__", None),
                layer.activation_relu_or_gelu,
            )
            for layer in self.encoder.layers
        )
        encoder_configuration = (
            type(self.encoder),
            self.encoder.num_layers,
            self.encoder.enable_nested_tensor,
            self.encoder.use_nested_tensor,
            self.encoder.mask_check,
            self.encoder.norm is None,
            None
            if self.encoder.norm is None
            else norm_configuration(cast(nn.LayerNorm, self.encoder.norm)),
        )
        configured_dimensions = (
            linear_configuration(self.support_projection),
            linear_configuration(self.description_projection),
            encoder_configuration,
            attention_configuration(self.pool),
            linear_configuration(self.head),
            tuple(self.support_type.shape),
            tuple(self.description_type.shape),
            tuple(self.pool_query.shape),
            tuple(self.raw_projection_scale.shape),
        )
        return (
            module_topology,
            parameter_topology,
            buffer_topology,
            layer_configuration,
            configured_dimensions,
        )

    def _validate_architecture(self) -> None:
        try:
            architecture = self.architecture
            current_topology = self._topology_fingerprint()
        except BaseException as error:
            raise RuntimeError("amortizer architecture or topology was mutated") from error
        if (
            architecture.canonical != self.__dict__.get("_construction_architecture")
            or current_topology != self.__dict__.get("_construction_topology")
        ):
            raise RuntimeError("amortizer architecture or topology was mutated")

    def assert_architecture_signature(self, expected: str) -> None:
        if type(expected) is not str:
            raise TypeError("expected architecture signature must be an exact str")
        self._validate_architecture()
        if expected != self.architecture.signature:
            raise ValueError("amortizer architecture signature does not match")

    def _validate_tensor_health(self) -> None:
        for name, tensor in (
            *(self.named_parameters(remove_duplicate=False)),
            *(self.named_buffers(remove_duplicate=False)),
        ):
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"amortizer floating tensors must be finite; {name} is nonfinite"
                )

    def _validate_model_placement(self) -> torch.device:
        parameters = tuple(self.named_parameters(remove_duplicate=False))
        if not parameters:
            raise RuntimeError("amortizer must own trainable parameters")
        devices = {parameter.device for _, parameter in parameters}
        if len(devices) != 1:
            raise ValueError("all amortizer parameters must be on one device")
        device = next(iter(devices))
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("amortizer parameters must be on cpu or cuda")
        if any(not parameter.requires_grad for _, parameter in parameters):
            raise ValueError("every amortizer parameter must require gradients")
        if any(parameter.is_inference() for _, parameter in parameters):
            raise ValueError("amortizer parameters must not be inference tensors")
        parameter_identities = [id(parameter) for _, parameter in parameters]
        if len(set(parameter_identities)) != len(parameter_identities):
            raise ValueError("amortizer has duplicate parameter objects")
        storage_identities = [
            (
                parameter.device,
                parameter.untyped_storage().data_ptr(),
                parameter.untyped_storage().nbytes(),
            )
            for _, parameter in parameters
        ]
        if len(set(storage_identities)) != len(storage_identities):
            raise ValueError("amortizer parameter storage aliases are forbidden")
        for name, parameter in parameters:
            if parameter.dtype != torch.float32:
                raise ValueError(
                    "amortizer parameters must have dtype torch.float32; "
                    f"{name} has {parameter.dtype}"
                )
        for name, buffer in self.named_buffers():
            if buffer.device != device:
                raise ValueError(f"all amortizer tensors must be on {device}; {name} is not")
            if buffer.is_inference():
                raise ValueError("amortizer buffers must not be inference tensors")
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                raise ValueError(
                    "amortizer floating buffers must have dtype torch.float32; "
                    f"{name} has {buffer.dtype}"
                )
        return device

    def _validate_inputs(
        self,
        support_features: Tensor,
        support_mask: Tensor,
        description_features: Tensor,
    ) -> torch.device:
        if not isinstance(support_features, Tensor):
            raise TypeError("support_features must be a Tensor")
        if not isinstance(support_mask, Tensor):
            raise TypeError("support_mask must be a Tensor")
        if not isinstance(description_features, Tensor):
            raise TypeError("description_features must be a Tensor")
        if support_features.ndim != 3:
            raise ValueError("support features must be rank-3")
        if support_mask.ndim != 2 or support_mask.shape != support_features.shape[:2]:
            raise ValueError(
                "support mask shape must match support feature batch and set dimensions"
            )
        if description_features.ndim != 2:
            raise ValueError("description features must be rank-2")
        if support_features.shape[0] <= 0 or support_features.shape[1] <= 0:
            raise ValueError("support features must have positive batch and support dimensions")
        if support_features.shape[2] != self.support_dim:
            raise ValueError(f"support feature dimension must be {self.support_dim}")
        if description_features.shape[1] != self.description_dim:
            raise ValueError(f"description feature dimension must be {self.description_dim}")
        if description_features.shape[0] != support_features.shape[0]:
            raise ValueError("support and description batch sizes must match")
        if support_features.dtype != torch.float32:
            raise TypeError("support features must have dtype torch.float32")
        if support_mask.dtype != torch.bool:
            raise TypeError("support mask must have dtype torch.bool")
        if description_features.dtype != torch.float32:
            raise TypeError("description features must have dtype torch.float32")
        if not (
            support_features.device == support_mask.device == description_features.device
        ):
            raise ValueError(
                "support features, mask, and description features must be on the same device"
            )
        if torch.is_grad_enabled() and (
            support_features.is_inference() or description_features.is_inference()
        ):
            raise ValueError(
                "inference support or description features cannot be used with gradients enabled"
            )

        model_device = self._validate_model_placement()
        if support_features.device != model_device:
            raise ValueError("amortizer inputs and parameters must be on the same device")
        if bool((support_mask.sum(dim=1) == 0).any()):
            raise ValueError("each concept requires at least one support image")
        if not bool(torch.isfinite(support_features[support_mask]).all()):
            raise ValueError("unmasked support features must be finite")
        if not bool(torch.isfinite(description_features).all()):
            raise ValueError("description features must be finite")
        return model_device

    def forward(
        self,
        support_features: Tensor,
        support_mask: Tensor,
        description_features: Tensor,
    ) -> AdapterPrediction:
        self._validate_model_placement()
        self._validate_tensor_health()
        self._validate_architecture()
        normalized_mask = support_mask
        if (
            isinstance(support_mask, Tensor)
            and support_mask.is_inference()
            and torch.is_grad_enabled()
        ):
            with torch.inference_mode(False):
                normalized_mask = support_mask.clone()
        device = self._validate_inputs(
            support_features, normalized_mask, description_features
        )
        sanitized_support = torch.where(
            normalized_mask.unsqueeze(-1),
            support_features,
            torch.zeros_like(support_features),
        )
        with torch.autocast(device_type=device.type, enabled=False):
            support_tokens = self.support_projection(sanitized_support) + self.support_type
            description_token = (
                self.description_projection(description_features).unsqueeze(1)
                + self.description_type
            )
            tokens = torch.cat((support_tokens, description_token), dim=1)
            description_mask = torch.ones(
                normalized_mask.shape[0],
                1,
                dtype=torch.bool,
                device=device,
            )
            token_mask = torch.cat((normalized_mask, description_mask), dim=1)
            encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
            query = self.pool_query.expand(tokens.shape[0], -1, -1)
            pooled, _ = self.pool(
                query,
                encoded,
                encoded,
                key_padding_mask=~token_mask,
                need_weights=False,
            )
            logits = self.head(pooled[:, 0]).reshape(
                support_features.shape[0],
                self.projection_count,
                self.atom_count,
            )
            scales = F.softplus(self.raw_projection_scale) + 1e-6
            coefficients = torch.tanh(logits) * scales.unsqueeze(0)

        expected_shape = (
            support_features.shape[0],
            self.projection_count,
            self.atom_count,
        )
        if logits.shape != expected_shape or coefficients.shape != expected_shape:
            raise RuntimeError("amortizer produced an invalid coefficient shape")
        if scales.shape != (self.projection_count, 1):
            raise RuntimeError("amortizer produced an invalid scale shape")
        if logits.dtype != torch.float32 or scales.dtype != torch.float32:
            raise RuntimeError("amortizer logits and scales must remain torch.float32")
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(scales).all()):
            raise RuntimeError("amortizer prediction must be finite")
        if not bool((scales > 0).all()):
            raise RuntimeError("amortizer scales must be positive")
        return AdapterPrediction(logits=logits, scales=scales, coefficients=coefficients)
