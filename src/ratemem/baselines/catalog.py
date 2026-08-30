"""Prespecified comparator and literature disposition catalog."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, model_validator

from ratemem.evaluation.canonical import canonical_json_bytes

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")

REQUIRED_CONTROL_IDS = frozenset(
    {
        "independent_fifo",
        "independent_lru",
        "independent_lrua",
        "private_progressive_size_aware",
        "private_progressive_separable_rate",
        "shared_packet_plain_greedy",
        "cts_style_static",
        "vb_lora_style_static",
        "share_style_online",
        "dreamcache_feature_cache",
        "hyperlora_upstream",
        "stateless_amortizer",
        "per_concept_lora",
        "exact_append_only_quantized",
        "exact_future_trace_packets",
    }
)


class CatalogError(ValueError):
    """Raised when the fixed comparator disposition is invalid or misused."""


class ComparisonClass(StrEnum):
    MATCHED_REQUIRED = "matched_required"
    CONTEXTUAL_ONLY = "contextual_only"
    INCOMPATIBLE = "incompatible"


class ControlEntry(BaseModel):
    model_config = _MODEL_CONFIG

    id: str
    implementation_mode: Literal["native", "external_jsonl"]
    family: str
    primary_table: bool
    roles: tuple[str, ...]

    @model_validator(mode="after")
    def validate_identifiers(self) -> ControlEntry:
        values = (self.id, self.family, *self.roles)
        if any(_IDENTIFIER.fullmatch(value) is None for value in values):
            raise ValueError("control identifiers must be canonical lowercase text")
        if not self.roles or len(self.roles) != len(set(self.roles)):
            raise ValueError("control roles must be non-empty and unique")
        return self


class LiteratureEntry(BaseModel):
    model_config = _MODEL_CONFIG

    citation_key: str
    title: str
    comparison_class: ComparisonClass
    port_mode: str
    primary_table: bool
    allowed_claims: tuple[str, ...]
    reason_code: str

    @model_validator(mode="after")
    def validate_disposition(self) -> LiteratureEntry:
        values = (
            self.citation_key,
            self.port_mode,
            self.reason_code,
            *self.allowed_claims,
        )
        if any(_IDENTIFIER.fullmatch(value) is None for value in values):
            raise ValueError("literature disposition identifiers must be canonical")
        if not self.title.strip() or not self.allowed_claims:
            raise ValueError("literature entries require a title and allowed claims")
        if self.primary_table and self.comparison_class is not ComparisonClass.MATCHED_REQUIRED:
            raise ValueError("only matched_required literature can enter a primary table")
        return self


class BaselineCatalog(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    controls: tuple[ControlEntry, ...]
    literature: tuple[LiteratureEntry, ...]

    @model_validator(mode="after")
    def validate_partition(self) -> BaselineCatalog:
        control_ids = self.control_ids
        citations = tuple(item.citation_key for item in self.literature)
        if not control_ids or len(control_ids) != len(set(control_ids)):
            raise ValueError("control ids must be non-empty and unique")
        if set(control_ids) != REQUIRED_CONTROL_IDS:
            raise ValueError("catalog controls differ from the prespecified registry")
        if len(citations) != len(set(citations)):
            raise ValueError("literature citation keys must be unique")
        if {item.comparison_class for item in self.literature} != set(ComparisonClass):
            raise ValueError("literature must include every explicit comparison class")
        return self

    @property
    def control_ids(self) -> tuple[str, ...]:
        return tuple(control.id for control in self.controls)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()

    def require_primary_eligible(self, citation_key: str) -> LiteratureEntry:
        candidates = [
            item for item in self.literature if item.citation_key == citation_key
        ]
        if len(candidates) != 1:
            raise CatalogError(f"unknown literature citation: {citation_key}")
        item = candidates[0]
        if (
            item.comparison_class is not ComparisonClass.MATCHED_REQUIRED
            or not item.primary_table
        ):
            raise CatalogError(
                f"{citation_key} is not eligible for primary matched table"
            )
        return item


def load_catalog(path: Path) -> BaselineCatalog:
    """Load and validate the complete fixed comparator catalog."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return BaselineCatalog.model_validate(payload)
    except (OSError, ValueError) as error:
        raise CatalogError(f"invalid baseline catalog: {error}") from error


__all__ = [
    "BaselineCatalog",
    "CatalogError",
    "ComparisonClass",
    "ControlEntry",
    "LiteratureEntry",
    "REQUIRED_CONTROL_IDS",
    "load_catalog",
]
