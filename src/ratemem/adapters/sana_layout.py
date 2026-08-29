from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Final
from weakref import ReferenceType, ref

import torch
from torch import Tensor, nn

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear

_DYNAMIC_ATOM_LINEAR_TYPE = DynamicAtomLinear

SANA_LAYOUT_VERSION: Final = "sana-qkv-v1"
ATTENTION_KINDS: Final = ("attn1", "attn2")
TARGET_MODULES: Final = ("to_q", "to_k", "to_v")

PRODUCTION_BLOCK_COUNT: Final = 20
PRODUCTION_WIDTH: Final = 2240
PRODUCTION_RANK: Final = 4
PRODUCTION_ATOM_COUNT: Final = 4


def _positive_exact_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class SanaAdapterLayout:
    """Canonical block-major SANA q/k/v dynamic-code layout."""

    num_blocks: int
    atom_count: int

    def __post_init__(self) -> None:
        _positive_exact_integer("num_blocks", self.num_blocks)
        _positive_exact_integer("atom_count", self.atom_count)

    @property
    def attention_kinds(self) -> tuple[str, ...]:
        return ATTENTION_KINDS

    @property
    def target_modules(self) -> tuple[str, ...]:
        return TARGET_MODULES

    @property
    def code_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_blocks,
            len(self.attention_kinds),
            len(self.target_modules),
            self.atom_count,
        )

    @property
    def projection_count(self) -> int:
        return self.num_blocks * len(self.attention_kinds) * len(
            self.target_modules
        )

    @property
    def code_dim(self) -> int:
        return self.projection_count * self.atom_count

    @property
    def atom_tensor_count(self) -> int:
        return self.projection_count * 2

    @property
    def projection_names(self) -> tuple[str, ...]:
        return tuple(
            f"transformer_blocks.{block}.{attention}.{projection}"
            for block in range(self.num_blocks)
            for attention in self.attention_kinds
            for projection in self.target_modules
        )

    def trainable_parameter_count(self, *, width: int, rank: int) -> int:
        checked_width = _positive_exact_integer("width", width)
        checked_rank = _positive_exact_integer("rank", rank)
        return (
            self.projection_count
            * self.atom_count
            * checked_rank
            * (checked_width + checked_width)
        )


@dataclass(frozen=True, slots=True)
class SanaStateLoadResult:
    missing_keys: list[str]
    unexpected_keys: list[str]


@dataclass(frozen=True, slots=True)
class _CanonicalWrapperBinding:
    path: str
    owner_ref: ReferenceType[nn.Module]
    attribute: str
    wrapper_ref: ReferenceType[DynamicAtomLinear]


_ActivationState = tuple[Tensor | None, object | None, int | None]


def _direct_module_at_path(root: nn.Module, path: str) -> nn.Module:
    current = root
    for segment in path.split("."):
        if type(current._modules) is not dict:
            raise RuntimeError(
                f"canonical module registry changed before {path}"
            )
        child = current._modules.get(segment)
        if not isinstance(child, nn.Module):
            raise RuntimeError(f"canonical module path no longer resolves: {path}")
        current = child
    return current


def _activation_state(wrapper: DynamicAtomLinear) -> _ActivationState:
    return (
        wrapper._coefficients,
        wrapper._activation_token,
        wrapper._coefficient_version,
    )


def _same_activation_state(
    first: _ActivationState, second: _ActivationState
) -> bool:
    return (
        first[0] is second[0]
        and first[1] is second[1]
        and first[2] == second[2]
    )


def _require_exact_activation(
    wrapper: DynamicAtomLinear,
    coefficients: Tensor,
    state: _ActivationState,
) -> None:
    active_coefficients, activation_token, coefficient_version = state
    if (
        active_coefficients is not coefficients
        or activation_token is None
        or coefficient_version is None
    ):
        raise RuntimeError("wrapper failed to activate the exact coefficient slice")
    if (
        type(coefficient_version) is not int
        or coefficient_version != wrapper._tensor_version(coefficients)
    ):
        raise RuntimeError("wrapper activated an invalid coefficient version")
    wrapper._require_unmodified(active_coefficients, coefficient_version)


def _require_unchanged_activation(
    wrapper: DynamicAtomLinear, expected: _ActivationState
) -> None:
    if not _same_activation_state(_activation_state(wrapper), expected):
        raise RuntimeError(
            "adapter activation state was modified inside the body"
        )
    active_coefficients, _activation_token, coefficient_version = expected
    if active_coefficients is None or coefficient_version is None:
        raise RuntimeError("adapter activation state is inconsistent")
    wrapper._require_unmodified(active_coefficients, coefficient_version)


def _restore_activation_state(
    wrapper: DynamicAtomLinear, state: _ActivationState
) -> None:
    coefficients, activation_token, coefficient_version = state
    object.__setattr__(wrapper, "_coefficients", coefficients)
    object.__setattr__(wrapper, "_activation_token", activation_token)
    object.__setattr__(wrapper, "_coefficient_version", coefficient_version)


def _stage_state_value(parameter: nn.Parameter, value: Tensor) -> Tensor:
    return (
        value.detach()
        .to(device=parameter.device, dtype=parameter.dtype)
        .clone()
    )


def _copy_staged_state_value(parameter: nn.Parameter, value: Tensor) -> None:
    parameter.copy_(value)


class SanaDynamicAdapterBank:
    """A non-owning controller for transformer-owned dynamic q/k/v wrappers."""

    def __init__(
        self,
        layout: SanaAdapterLayout,
        wrappers: list[DynamicAtomLinear] | tuple[DynamicAtomLinear, ...],
        *,
        transformer: nn.Module,
        bindings: tuple[_CanonicalWrapperBinding, ...],
    ) -> None:
        if type(layout) is not SanaAdapterLayout:
            raise TypeError("layout must be an exact SanaAdapterLayout")
        wrapper_tuple = tuple(wrappers)
        if len(wrapper_tuple) != layout.projection_count:
            raise ValueError("wrapper count does not match the SANA layout")
        if any(
            type(wrapper) is not _DYNAMIC_ATOM_LINEAR_TYPE
            for wrapper in wrapper_tuple
        ):
            raise TypeError("every wrapper must be an exact DynamicAtomLinear")
        if len({id(wrapper) for wrapper in wrapper_tuple}) != len(wrapper_tuple):
            raise ValueError("wrapper alias is forbidden")
        if any(wrapper.atom_count != layout.atom_count for wrapper in wrapper_tuple):
            raise ValueError("wrapper atom_count does not match the SANA layout")
        if not isinstance(transformer, nn.Module):
            raise TypeError("transformer must be an nn.Module")
        if len(bindings) != layout.projection_count:
            raise ValueError("canonical binding count does not match the SANA layout")
        for path, wrapper, binding in zip(
            layout.projection_names, wrapper_tuple, bindings, strict=True
        ):
            if binding.path != path:
                raise ValueError("canonical binding order does not match the SANA layout")
            if binding.wrapper_ref() is not wrapper:
                raise ValueError(f"canonical wrapper binding is invalid at {path}")

        self.layout = layout
        self._transformer_ref: ReferenceType[nn.Module] = ref(transformer)
        self._bindings = bindings
        self._resolve_wrappers()

    def _resolve_wrappers(self) -> tuple[DynamicAtomLinear, ...]:
        transformer = self._transformer_ref()
        if transformer is None:
            raise RuntimeError("canonical transformer was released")
        resolved: list[DynamicAtomLinear] = []
        for expected_path, binding in zip(
            self.layout.projection_names, self._bindings, strict=True
        ):
            if binding.path != expected_path:
                raise RuntimeError("canonical adapter binding order changed")
            owner = binding.owner_ref()
            wrapper = binding.wrapper_ref()
            if owner is None:
                raise RuntimeError(
                    f"canonical adapter owner was released at {expected_path}"
                )
            if wrapper is None:
                raise RuntimeError(
                    f"canonical adapter wrapper was released at {expected_path}"
                )
            if type(owner._modules) is not dict:
                raise RuntimeError(
                    f"canonical owner registry changed at {expected_path}"
                )
            if owner._modules.get(binding.attribute) is not wrapper:
                raise RuntimeError(
                    f"canonical owner binding changed at {expected_path}"
                )
            if _direct_module_at_path(transformer, expected_path) is not wrapper:
                raise RuntimeError(
                    f"canonical module path changed at {expected_path}"
                )
            resolved.append(wrapper)
        return tuple(resolved)

    @property
    def wrappers(self) -> tuple[DynamicAtomLinear, ...]:
        return self._resolve_wrappers()

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        seen: set[int] = set()
        for path, wrapper in zip(
            self.layout.projection_names, self.wrappers, strict=True
        ):
            for atom_name, parameter in (
                ("atom_down", wrapper.atom_down),
                ("atom_up", wrapper.atom_up),
            ):
                identity = id(parameter)
                if identity in seen:
                    raise RuntimeError("adapter trainable parameter alias detected")
                seen.add(identity)
                yield f"{path}.{atom_name}", parameter

    def parameters(self) -> Iterator[nn.Parameter]:
        for _name, parameter in self.named_parameters():
            yield parameter

    def state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (name, parameter.detach())
            for name, parameter in self.named_parameters()
        )

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], *, strict: bool = True
    ) -> SanaStateLoadResult:
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        expected = OrderedDict(self.named_parameters())
        supplied_keys = tuple(state_dict)
        missing = [name for name in expected if name not in state_dict]
        unexpected = [name for name in supplied_keys if name not in expected]
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"missing keys: {missing}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected}")
            raise RuntimeError("strict adapter state load failed; " + "; ".join(details))

        validated: list[tuple[nn.Parameter, Tensor]] = []
        for name, parameter in expected.items():
            if name not in state_dict:
                continue
            value = state_dict[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"state value {name} must be a Tensor")
            if value.shape != parameter.shape:
                raise ValueError(
                    f"state value {name} has shape {tuple(value.shape)}, "
                    f"expected {tuple(parameter.shape)}"
                )
            validated.append((parameter, value))

        copies = [
            (parameter, _stage_state_value(parameter, value))
            for parameter, value in validated
        ]

        originals = [parameter.detach().clone() for parameter, _value in copies]
        try:
            with torch.no_grad():
                for parameter, value in copies:
                    _copy_staged_state_value(parameter, value)
        except Exception:
            with torch.no_grad():
                for (parameter, _value), original in zip(
                    copies, originals, strict=True
                ):
                    parameter.copy_(original)
            raise
        return SanaStateLoadResult(missing, unexpected)

    @contextmanager
    def activate(self, coefficients: Tensor) -> Iterator[None]:
        wrappers = self.wrappers
        if any(
            not _same_activation_state(
                _activation_state(wrapper), (None, None, None)
            )
            for wrapper in wrappers
        ):
            raise RuntimeError("adapter coefficients are already active")
        if not isinstance(coefficients, Tensor):
            raise TypeError("coefficients must be a Tensor")
        if coefficients.ndim not in (1, 2):
            raise ValueError("coefficients must be 1D or 2D flat codes")
        if coefficients.shape[-1] != self.layout.code_dim:
            raise ValueError(
                f"code dimension must be {self.layout.code_dim}, "
                f"got {coefficients.shape[-1]}"
            )

        if coefficients.ndim == 1:
            shaped = coefficients.reshape(
                self.layout.projection_count, self.layout.atom_count
            )
            slices = tuple(shaped[index] for index in range(len(wrappers)))
        else:
            shaped = coefficients.reshape(
                coefficients.shape[0],
                self.layout.projection_count,
                self.layout.atom_count,
            )
            slices = tuple(shaped[:, index] for index in range(len(wrappers)))

        attempted: list[tuple[DynamicAtomLinear, _ActivationState]] = []
        active: list[tuple[DynamicAtomLinear, _ActivationState]] = []
        try:
            with ExitStack() as stack:
                for wrapper, alpha in zip(wrappers, slices, strict=True):
                    attempted.append((wrapper, _activation_state(wrapper)))
                    stack.enter_context(wrapper.use_coefficients(alpha))
                    active_state = _activation_state(wrapper)
                    _require_exact_activation(wrapper, alpha, active_state)
                    active.append((wrapper, active_state))
                try:
                    yield
                finally:
                    for wrapper, expected_state in active:
                        _require_unchanged_activation(wrapper, expected_state)
            for wrapper, expected_state in attempted:
                if not _same_activation_state(
                    _activation_state(wrapper), expected_state
                ):
                    raise RuntimeError(
                        "adapter context exit did not restore activation state"
                    )
        finally:
            for wrapper, state in reversed(attempted):
                _restore_activation_state(wrapper, state)


@dataclass(frozen=True, slots=True)
class _TargetInventory:
    path: str
    attention_name: str
    owner: nn.Module
    registry: dict[str, nn.Module | None]
    attribute: str
    base: nn.Linear


def _storage_identity(tensor: Tensor) -> tuple[torch.device, int] | None:
    if tensor.device.type == "meta" or tensor.numel() == 0:
        return None
    return tensor.device, tensor.untyped_storage().data_ptr()


def _inventory_targets(
    transformer: nn.Module, *, expected_blocks: int, allow_meta: bool = False
) -> tuple[_TargetInventory, ...]:
    if not isinstance(transformer, nn.Module):
        raise TypeError("transformer must be an nn.Module")
    if type(transformer._modules) is not dict:
        raise TypeError("transformer._modules must be an exact built-in dict")
    blocks = transformer._modules.get("transformer_blocks")
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(
            "transformer_blocks must be a canonically registered nn.ModuleList"
        )
    if type(blocks._modules) is not dict:
        raise TypeError(
            "transformer_blocks._modules must be an exact built-in dict"
        )
    if len(blocks) != expected_blocks:
        raise ValueError(
            f"expected {expected_blocks} transformer blocks, got {len(blocks)}"
        )

    module_occurrences = Counter(
        id(module)
        for _name, module in transformer.named_modules(remove_duplicate=False)
    )
    parameter_occurrences = Counter(
        id(parameter)
        for _name, parameter in transformer.named_parameters(
            remove_duplicate=False
        )
    )
    storage_occurrences: Counter[tuple[torch.device, int]] = Counter()
    for _name, parameter in transformer.named_parameters(
        remove_duplicate=False
    ):
        storage_identity = _storage_identity(parameter)
        if storage_identity is not None:
            storage_occurrences[storage_identity] += 1
    for _name, buffer in transformer.named_buffers(remove_duplicate=False):
        storage_identity = _storage_identity(buffer)
        if storage_identity is not None:
            storage_occurrences[storage_identity] += 1

    block_ids: set[int] = set()
    attention_ids: set[int] = set()
    linear_ids: set[int] = set()
    parameter_ids: set[int] = set()
    storage_ids: set[tuple[torch.device, int]] = set()
    inventory: list[_TargetInventory] = []
    target_placement: tuple[torch.device, torch.dtype] | None = None

    canonical_blocks: list[nn.Module] = []
    for block_index in range(expected_blocks):
        block = blocks._modules.get(str(block_index))
        if not isinstance(block, nn.Module):
            raise TypeError(
                f"transformer_blocks.{block_index} must be canonically registered"
            )
        canonical_blocks.append(block)

    for block_index, block in enumerate(canonical_blocks):
        if type(block._modules) is not dict:
            raise TypeError(
                f"transformer_blocks.{block_index}._modules must be an "
                "exact built-in dict"
            )
        if module_occurrences[id(block)] != 1 or id(block) in block_ids:
            raise ValueError("transformer block module alias is forbidden")
        block_ids.add(id(block))
        for attention_name in ATTENTION_KINDS:
            attention = block._modules.get(attention_name)
            attention_path = f"transformer_blocks.{block_index}.{attention_name}"
            if not isinstance(attention, nn.Module):
                raise TypeError(
                    f"{attention_path} must be a canonically registered nn.Module"
                )
            if type(attention._modules) is not dict:
                raise TypeError(
                    f"{attention_path}._modules must be an exact built-in dict"
                )
            if (
                module_occurrences[id(attention)] != 1
                or id(attention) in attention_ids
            ):
                raise ValueError(f"attention module alias is forbidden at {attention_path}")
            attention_ids.add(id(attention))
            for target_name in TARGET_MODULES:
                path = f"{attention_path}.{target_name}"
                target = attention._modules.get(target_name)
                if isinstance(target, _DYNAMIC_ATOM_LINEAR_TYPE):
                    raise ValueError(f"{path} is already wrapped")
                if type(target) is not nn.Linear:
                    raise TypeError(f"{path} must be an exact nn.Linear")
                if (
                    module_occurrences[id(target)] != 1
                    or id(target) in linear_ids
                ):
                    raise ValueError(f"Linear module alias is forbidden at {path}")
                linear_ids.add(id(target))

                if (
                    type(target.in_features) is not int
                    or target.in_features < 1
                    or type(target.out_features) is not int
                    or target.out_features < 1
                ):
                    raise ValueError(
                        f"{path} feature metadata must be positive exact integers"
                    )
                if target._parameters.get("weight") is not target.weight:
                    raise ValueError(
                        f"directly registered weight parameter required at {path}"
                    )
                if (
                    "bias" not in target._parameters
                    or target._parameters["bias"] is not target.bias
                ):
                    raise ValueError(
                        f"directly registered bias parameter required at {path}"
                    )
                if type(target.weight) is not nn.Parameter:
                    raise TypeError(f"{path}.weight must be an exact Parameter")
                expected_weight_shape = (
                    target.out_features,
                    target.in_features,
                )
                expected_weight_numel = (
                    target.out_features * target.in_features
                )
                if (
                    tuple(target.weight.shape) != expected_weight_shape
                    or target.weight.numel() != expected_weight_numel
                ):
                    raise ValueError(
                        f"{path} weight shape must match in_features "
                        f"{target.in_features} and out_features "
                        f"{target.out_features}; expected {expected_weight_shape} "
                        f"with numel {expected_weight_numel}, got "
                        f"{tuple(target.weight.shape)} with numel "
                        f"{target.weight.numel()}"
                    )
                if target.bias is not None:
                    if type(target.bias) is not nn.Parameter:
                        raise TypeError(f"{path}.bias must be an exact Parameter")
                    expected_bias_shape = (target.out_features,)
                    if (
                        tuple(target.bias.shape) != expected_bias_shape
                        or target.bias.numel() != target.out_features
                    ):
                        raise ValueError(
                            f"{path} bias shape must be {expected_bias_shape} "
                            f"with numel {target.out_features}, got "
                            f"{tuple(target.bias.shape)} with numel "
                            f"{target.bias.numel()}"
                        )

                weight_placement = (target.weight.device, target.weight.dtype)
                if not allow_meta and target.weight.device.type == "meta":
                    raise ValueError(
                        f"{path} must have materialized weight placement"
                    )
                if target.bias is not None and (
                    target.bias.device,
                    target.bias.dtype,
                ) != weight_placement:
                    raise ValueError(
                        f"{path} bias placement must match its weight"
                    )
                if target_placement is None:
                    target_placement = weight_placement
                elif weight_placement != target_placement:
                    raise ValueError(
                        "all SANA target weights must share one device and dtype"
                    )

                for parameter_name, parameter in target.named_parameters(
                    recurse=False, remove_duplicate=False
                ):
                    parameter_path = f"{path}.{parameter_name}"
                    if (
                        parameter_occurrences[id(parameter)] != 1
                        or id(parameter) in parameter_ids
                    ):
                        raise ValueError(
                            f"Parameter alias is forbidden at {parameter_path}"
                        )
                    parameter_ids.add(id(parameter))
                    storage_identity = _storage_identity(parameter)
                    if storage_identity is not None:
                        if (
                            storage_occurrences[storage_identity] != 1
                            or storage_identity in storage_ids
                        ):
                            raise ValueError(
                                f"parameter storage alias is forbidden at {parameter_path}"
                            )
                        storage_ids.add(storage_identity)
                inventory.append(
                    _TargetInventory(
                        path=path,
                        attention_name=attention_name,
                        owner=attention,
                        registry=attention._modules,
                        attribute=target_name,
                        base=target,
                    )
                )

    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise ValueError("entire transformer must be frozen before adapter install")
    if any(module.training for module in transformer.modules()):
        raise ValueError("every transformer module must be in eval before adapter install")
    return tuple(inventory)


def _commit_target_module(
    owner: nn.Module,
    attribute: str,
    expected: nn.Module,
    replacement: nn.Module,
) -> None:
    if type(owner._modules) is not dict:
        raise RuntimeError("target module registry changed before commit")
    if owner._modules.get(attribute) is not expected:
        raise RuntimeError("target module changed before commit")
    owner._modules[attribute] = replacement


def install_sana_dynamic_atoms(
    transformer: nn.Module,
    *,
    rank: int,
    atom_count: int,
    expected_blocks: int,
) -> SanaDynamicAdapterBank:
    """Install all adapters transactionally after complete layout validation."""

    checked_rank = _positive_exact_integer("rank", rank)
    checked_atom_count = _positive_exact_integer("atom_count", atom_count)
    checked_blocks = _positive_exact_integer("expected_blocks", expected_blocks)
    layout = SanaAdapterLayout(checked_blocks, checked_atom_count)

    inventory = _inventory_targets(transformer, expected_blocks=checked_blocks)
    if len(inventory) != layout.projection_count:
        raise ValueError("target inventory does not match the canonical SANA layout")

    wrappers: list[DynamicAtomLinear] = []
    for target in inventory:
        wrapper = DynamicAtomLinear(
            target.base, rank=checked_rank, atom_count=checked_atom_count
        )
        wrapper.eval()
        wrappers.append(wrapper)

    committed: list[_TargetInventory] = []
    try:
        for target, wrapper in zip(inventory, wrappers, strict=True):
            committed.append(target)
            _commit_target_module(
                target.owner,
                target.attribute,
                target.base,
                wrapper,
            )
        bindings = tuple(
            _CanonicalWrapperBinding(
                path=target.path,
                owner_ref=ref(target.owner),
                attribute=target.attribute,
                wrapper_ref=ref(wrapper),
            )
            for target, wrapper in zip(inventory, wrappers, strict=True)
        )
        bank = SanaDynamicAdapterBank(
            layout,
            wrappers,
            transformer=transformer,
            bindings=bindings,
        )
    except Exception:
        for target in reversed(committed):
            target.registry[target.attribute] = target.base
        raise
    return bank


def validate_production_sana_layout(
    transformer: nn.Module,
    *,
    rank: int,
    atom_count: int,
    require_cuda_bfloat16: bool = True,
) -> SanaAdapterLayout:
    """Validate the fixed production shape separately from the generic installer."""

    checked_rank = _positive_exact_integer("rank", rank)
    checked_atom_count = _positive_exact_integer("atom_count", atom_count)
    if type(require_cuda_bfloat16) is not bool:
        raise TypeError("require_cuda_bfloat16 must be a bool")
    if checked_rank != PRODUCTION_RANK:
        raise ValueError(f"production rank must be {PRODUCTION_RANK}")
    if checked_atom_count != PRODUCTION_ATOM_COUNT:
        raise ValueError(
            f"production atom_count must be {PRODUCTION_ATOM_COUNT}"
        )

    layout = SanaAdapterLayout(PRODUCTION_BLOCK_COUNT, checked_atom_count)
    inventory = _inventory_targets(
        transformer, expected_blocks=PRODUCTION_BLOCK_COUNT, allow_meta=True
    )
    actual_atom_parameters = 0
    for target in inventory:
        weight = target.base.weight
        if (
            target.base.in_features != PRODUCTION_WIDTH
            or target.base.out_features != PRODUCTION_WIDTH
        ):
            raise ValueError(
                f"{target.path} in_features must be {PRODUCTION_WIDTH} and "
                f"out_features must be {PRODUCTION_WIDTH}, got "
                f"{target.base.in_features} and {target.base.out_features}"
            )
        if tuple(weight.shape) != (PRODUCTION_WIDTH, PRODUCTION_WIDTH):
            raise ValueError(
                f"{target.path} weight shape must be "
                f"({PRODUCTION_WIDTH}, {PRODUCTION_WIDTH}), got {tuple(weight.shape)}"
            )
        if target.attention_name == "attn1" and target.base.bias is not None:
            raise ValueError(f"{target.path} attn1 projection must be without bias")
        if target.attention_name == "attn2" and target.base.bias is None:
            raise ValueError(f"{target.path} attn2 projection must be with bias")
        if target.base.bias is not None and (
            tuple(target.base.bias.shape) != (PRODUCTION_WIDTH,)
            or target.base.bias.numel() != PRODUCTION_WIDTH
        ):
            raise ValueError(
                f"{target.path} bias shape must be ({PRODUCTION_WIDTH},) "
                f"and numel must be {PRODUCTION_WIDTH}, got "
                f"shape {tuple(target.base.bias.shape)} and "
                f"numel {target.base.bias.numel()}"
            )
        if require_cuda_bfloat16 and (
            weight.device.type != "cuda" or weight.dtype is not torch.bfloat16
        ):
            raise ValueError(
                f"{target.path} production target must be CUDA bfloat16"
            )
        if (
            require_cuda_bfloat16
            and target.base.bias is not None
            and (
                target.base.bias.device.type != "cuda"
                or target.base.bias.dtype is not torch.bfloat16
            )
        ):
            raise ValueError(
                f"{target.path} production bias must be CUDA bfloat16"
            )
        actual_atom_parameters += checked_atom_count * checked_rank * (
            target.base.in_features + target.base.out_features
        )

    expected_atom_parameters = layout.trainable_parameter_count(
        width=PRODUCTION_WIDTH, rank=checked_rank
    )
    if actual_atom_parameters != expected_atom_parameters:
        raise ValueError(
            f"production atom parameter numel must be {expected_atom_parameters}, "
            f"got {actual_atom_parameters}"
        )
    if layout.atom_tensor_count != 240:
        raise ValueError(
            f"production atom tensor count must be 240, got {layout.atom_tensor_count}"
        )
    return layout
