"""Source-hashed runtime registry for the fifteen prespecified controls."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ratemem.baselines.catalog import REQUIRED_CONTROL_IDS, BaselineCatalog
from ratemem.baselines.protocol import BaselineAdapter
from ratemem.evaluation.canonical import canonical_json_bytes

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class RegistryError(RuntimeError):
    """Raised when a runtime factory is missing, mutable, or source-mismatched."""


class FactoryDescriptor(BaseModel):
    model_config = _MODEL_CONFIG

    method_id: str = Field(min_length=1)
    implementation_mode: Literal["native", "external_jsonl"]
    import_path: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*$")
    fixed_kwargs: dict[str, Any]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeRegistryLock(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    baseline_lock_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[FactoryDescriptor, ...]
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_entries(self) -> RuntimeRegistryLock:
        ids = tuple(row.method_id for row in self.entries)
        if ids != tuple(sorted(REQUIRED_CONTROL_IDS)):
            raise ValueError("runtime registry lock must cover all controls in sorted order")
        return self

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("registry_sha256")
        return canonical_json_bytes(payload)


@dataclass(frozen=True, slots=True)
class RegisteredFactory:
    descriptor: FactoryDescriptor
    constructor: type[Any]

    @property
    def factory_importable(self) -> bool:
        return True

    @property
    def factory_sha256(self) -> str:
        return self.descriptor.source_sha256

    def create(self, **dependencies: object) -> BaselineAdapter:
        overlap = set(self.descriptor.fixed_kwargs) & set(dependencies)
        if overlap:
            raise RegistryError(
                f"cannot override fixed factory arguments for {self.descriptor.method_id}: "
                f"{sorted(overlap)}"
            )
        arguments = {**self.descriptor.fixed_kwargs, **dependencies}
        try:
            adapter = self.constructor(**arguments)
        except TypeError as error:
            raise RegistryError(
                f"factory dependencies are invalid for {self.descriptor.method_id}: {error}"
            ) from error
        if not isinstance(adapter, BaselineAdapter):
            raise RegistryError(
                f"factory did not produce the canonical adapter protocol: "
                f"{self.descriptor.method_id}"
            )
        if adapter.method_id != self.descriptor.method_id:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
            raise RegistryError("factory produced an adapter with the wrong method id")
        return adapter


class BaselineRegistry:
    def __init__(
        self,
        factories: Mapping[str, RegisteredFactory],
        *,
        catalog_sha256: str,
        baseline_lock_id: str,
    ) -> None:
        if set(factories) != REQUIRED_CONTROL_IDS:
            raise RegistryError("runtime registry differs from the fifteen controls")
        self._factories = {key: factories[key] for key in sorted(factories)}
        self.catalog_sha256 = catalog_sha256
        self.baseline_lock_id = baseline_lock_id

    @property
    def method_ids(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def __getitem__(self, method_id: str) -> RegisteredFactory:
        try:
            return self._factories[method_id]
        except KeyError as error:
            raise RegistryError(f"unknown baseline method: {method_id}") from error

    def create(self, method_id: str, **dependencies: object) -> BaselineAdapter:
        return self[method_id].create(**dependencies)

    def lock(self) -> RuntimeRegistryLock:
        provisional = RuntimeRegistryLock(
            baseline_lock_id=self.baseline_lock_id,
            catalog_sha256=self.catalog_sha256,
            entries=tuple(row.descriptor for row in self._factories.values()),
            registry_sha256="0" * 64,
        )
        return provisional.model_copy(
            update={
                "registry_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
            }
        )


_FACTORY_TARGETS: dict[str, tuple[str, dict[str, object]]] = {
    "independent_fifo": (
        "ratemem.baselines.independent:IndependentCodeCacheAdapter",
        {"method_id": "independent_fifo", "policy": "fifo"},
    ),
    "independent_lru": (
        "ratemem.baselines.independent:IndependentCodeCacheAdapter",
        {"method_id": "independent_lru", "policy": "lru"},
    ),
    "independent_lrua": (
        "ratemem.baselines.independent:IndependentCodeCacheAdapter",
        {"method_id": "independent_lrua", "policy": "lrua"},
    ),
    "private_progressive_size_aware": (
        "ratemem.baselines.private_progressive:PrivateProgressiveAdapter",
        {"method_id": "private_progressive_size_aware", "policy": "size_aware"},
    ),
    "private_progressive_separable_rate": (
        "ratemem.baselines.private_progressive:PrivateProgressiveAdapter",
        {
            "method_id": "private_progressive_separable_rate",
            "policy": "separable_rate",
        },
    ),
    "shared_packet_plain_greedy": (
        "ratemem.baselines.shared_greedy:SharedPacketGreedyAdapter",
        {},
    ),
    "cts_style_static": (
        "ratemem.baselines.static_shared:StaticSharedAdapter",
        {"method_id": "cts_style_static"},
    ),
    "vb_lora_style_static": (
        "ratemem.baselines.static_shared:StaticSharedAdapter",
        {"method_id": "vb_lora_style_static"},
    ),
    "share_style_online": (
        "ratemem.baselines.online_share:OnlineShareAdapter",
        {},
    ),
    "dreamcache_feature_cache": (
        "ratemem.baselines.external_jsonl:ExternalJsonlAdapter",
        {},
    ),
    "hyperlora_upstream": (
        "ratemem.baselines.external_jsonl:ExternalJsonlAdapter",
        {},
    ),
    "stateless_amortizer": (
        "ratemem.baselines.stateless:StatelessAmortizerAdapter",
        {},
    ),
    "per_concept_lora": (
        "ratemem.baselines.lora_reference:LoRAOptimizationAdapter",
        {},
    ),
    "exact_append_only_quantized": (
        "ratemem.baselines.oracles:ExactAppendOnlyAdapter",
        {},
    ),
    "exact_future_trace_packets": (
        "ratemem.baselines.oracles:FutureTracePacketAdapter",
        {},
    ),
}


def _resolve(import_path: str) -> type[Any]:
    module_name, symbol_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
    except (ImportError, AttributeError) as error:
        raise RegistryError(f"factory is not importable: {import_path}") from error
    if not inspect.isclass(symbol):
        raise RegistryError(f"factory target must be a class: {import_path}")
    source_file = inspect.getsourcefile(symbol)
    if source_file is None or "/tests/" in source_file or "<" in source_file:
        raise RegistryError(f"factory source is not a production module: {import_path}")
    return symbol


def _source_sha256(symbol: type[Any]) -> str:
    try:
        source = inspect.getsource(symbol).encode("utf-8")
    except (OSError, TypeError) as error:
        raise RegistryError("factory source cannot be inspected") from error
    return hashlib.sha256(source).hexdigest()


def build_registry(
    catalog: BaselineCatalog,
    *,
    baseline_lock_id: str = "0" * 64,
    expected_factory_sha256: Mapping[str, str] | None = None,
) -> BaselineRegistry:
    """Resolve and source-hash every fixed factory without instantiating dependencies."""

    if type(catalog) is not BaselineCatalog:
        raise TypeError("catalog must be an exact BaselineCatalog")
    if len(baseline_lock_id) != 64 or any(
        character not in "0123456789abcdef" for character in baseline_lock_id
    ):
        raise RegistryError("baseline lock id must be a lowercase SHA-256")
    catalog_modes = {row.id: row.implementation_mode for row in catalog.controls}
    if set(_FACTORY_TARGETS) != set(catalog_modes):
        raise RegistryError("factory targets differ from the comparator catalog")
    factories: dict[str, RegisteredFactory] = {}
    for method_id in sorted(_FACTORY_TARGETS):
        import_path, fixed_kwargs = _FACTORY_TARGETS[method_id]
        constructor = _resolve(import_path)
        source_sha = _source_sha256(constructor)
        if expected_factory_sha256 is not None:
            expected = expected_factory_sha256.get(method_id)
            if expected != source_sha:
                raise RegistryError(f"factory source hash changed for {method_id}")
        descriptor = FactoryDescriptor(
            method_id=method_id,
            implementation_mode=catalog_modes[method_id],
            import_path=import_path,
            fixed_kwargs=fixed_kwargs,
            source_sha256=source_sha,
        )
        factories[method_id] = RegisteredFactory(descriptor, constructor)
    if expected_factory_sha256 is not None and set(expected_factory_sha256) != set(factories):
        raise RegistryError("expected factory hash registry is incomplete")
    return BaselineRegistry(
        factories,
        catalog_sha256=catalog.sha256,
        baseline_lock_id=baseline_lock_id,
    )


__all__ = [
    "BaselineRegistry",
    "FactoryDescriptor",
    "RegisteredFactory",
    "RegistryError",
    "RuntimeRegistryLock",
    "build_registry",
]
