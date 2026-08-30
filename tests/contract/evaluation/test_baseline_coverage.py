import subprocess
from pathlib import Path

from ratemem.baselines.catalog import load_catalog
from ratemem.evaluation.baselines import load_requirements

REQUIREMENTS = Path("configs/scientific/baseline-requirements.yaml")
CATALOG = Path("configs/baselines/literature-classification.yaml")


def test_required_ids_equal_the_companion_catalog_controls() -> None:
    requirements = load_requirements(REQUIREMENTS)
    catalog = load_catalog(CATALOG)
    assert len(requirements.runnable_registry) == 15
    assert set(requirements.runnable_registry) == {
        control.id for control in catalog.controls
    }
    assert requirements.postlock_execution_required == requirements.runnable_registry


def test_sinelora_delta_is_contextual_sd3_literature_not_a_control() -> None:
    requirements = load_requirements(REQUIREMENTS)
    catalog = load_catalog(CATALOG)
    record = next(
        row
        for row in catalog.literature
        if row.citation_key == "sinelora_delta_aaai2026"
    )
    assert record.comparison_class == "contextual_only"
    assert record.port_mode == "citation_only_sd3_medium"
    assert requirements.contextual_literature_citation_keys == (
        "sinelora_delta_aaai2026",
    )
    assert "sine_lora_delta_sdxl" not in requirements.runnable_registry
    assert all(control.id != "sine_lora_delta_sdxl" for control in catalog.controls)


def test_evaluation_reexports_the_one_canonical_protocol() -> None:
    from ratemem.baselines.protocol import BaselineAdapter as CanonicalBaselineAdapter
    from ratemem.evaluation.baselines import BaselineAdapter

    assert BaselineAdapter is CanonicalBaselineAdapter


def test_committed_catalog_and_baseline_lock_schemas_match_models(
    tmp_path: Path,
) -> None:
    catalog_schema = tmp_path / "catalog.schema.json"
    lock_schema = tmp_path / "baseline-lock.schema.json"
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-baselines",
            "catalog",
            "schema",
            "--output",
            str(catalog_schema),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "baselines",
            "schema",
            "--output",
            str(lock_schema),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert catalog_schema.read_bytes() == Path(
        "schemas/ratemem-baseline-catalog-v1.schema.json"
    ).read_bytes()
    assert lock_schema.read_bytes() == Path(
        "schemas/scientific-baseline-lock.schema.json"
    ).read_bytes()
