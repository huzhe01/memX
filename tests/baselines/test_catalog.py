from pathlib import Path

import pytest

from ratemem.baselines.catalog import (
    REQUIRED_CONTROL_IDS,
    CatalogError,
    ComparisonClass,
    load_catalog,
)

CATALOG = Path("configs/baselines/literature-classification.yaml")


def test_every_required_control_is_present_once_and_has_an_implementation_mode() -> None:
    catalog = load_catalog(CATALOG)
    assert set(catalog.control_ids) == REQUIRED_CONTROL_IDS
    assert len(catalog.control_ids) == len(set(catalog.control_ids))
    assert all(
        control.implementation_mode in {"native", "external_jsonl"}
        for control in catalog.controls
    )


def test_literature_is_explicitly_partitioned_and_never_auto_promoted() -> None:
    catalog = load_catalog(CATALOG)
    assert len(catalog.literature) >= 24
    assert {item.comparison_class for item in catalog.literature} == set(ComparisonClass)
    assert all(item.reason_code and item.allowed_claims for item in catalog.literature)
    assert not any(
        item.primary_table
        for item in catalog.literature
        if item.comparison_class != ComparisonClass.MATCHED_REQUIRED
    )


def test_incompatible_or_contextual_method_cannot_enter_primary_table() -> None:
    catalog = load_catalog(CATALOG)
    for citation_key in (
        "vsm_diffusion_neurips2023",
        "moblora_acl2026",
        "rqt_acl2025",
        "sinelora_delta_aaai2026",
    ):
        with pytest.raises(CatalogError, match="not eligible for primary matched table"):
            catalog.require_primary_eligible(citation_key)


def test_non_sana_sinelora_delta_is_contextual_only_and_has_no_claimed_port() -> None:
    catalog = load_catalog(CATALOG)
    item = next(
        row
        for row in catalog.literature
        if row.citation_key == "sinelora_delta_aaai2026"
    )
    assert item.comparison_class == ComparisonClass.CONTEXTUAL_ONLY
    assert item.primary_table is False
    assert item.port_mode == "citation_only_sd3_medium"
    assert "sine_lora_delta_sdxl" not in catalog.control_ids
