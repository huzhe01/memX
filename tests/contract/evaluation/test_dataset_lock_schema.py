from __future__ import annotations

import subprocess
from pathlib import Path

import jsonschema

from ratemem.evaluation.dataset_lock import DatasetLock, load_inventory, seal_dataset_lock


def test_committed_dataset_lock_schema_matches_model(tmp_path: Path) -> None:
    output = tmp_path / "dataset-lock.schema.json"
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "data",
            "schema",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    committed = Path("schemas/dataset-lock.schema.json")
    assert output.read_bytes() == committed.read_bytes()


def test_synthetic_lock_validates_against_committed_schema() -> None:
    inventory = load_inventory(Path("tests/fixtures/scientific/source-inventory.json"))
    lock = seal_dataset_lock(
        inventory,
        policy_path=Path("configs/scientific/dataset-policy.yaml"),
        mode="synthetic",
    )
    schema = DatasetLock.model_json_schema()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(lock.model_dump(mode="json"), schema)


def test_seal_cli_blocks_missing_scientific_inventory_without_partial_outputs(
    tmp_path: Path,
) -> None:
    lock_output = tmp_path / "dataset-lock.yaml"
    card_output = tmp_path / "data-card.md"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "data",
            "seal",
            "--inventory",
            str(tmp_path / "missing.json"),
            "--policy",
            "configs/scientific/dataset-policy.yaml",
            "--lock-output",
            str(lock_output),
            "--card-output",
            str(card_output),
            "--mode",
            "scientific",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr.endswith(
        "BLOCKED dataset-lock: audited source inventory is missing\n"
    )
    assert not lock_output.exists()
    assert not card_output.exists()


def test_seal_cli_writes_a_schema_valid_synthetic_lock(tmp_path: Path) -> None:
    lock_output = tmp_path / "dataset-lock.yaml"
    card_output = tmp_path / "data-card.md"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "data",
            "seal",
            "--inventory",
            "tests/fixtures/scientific/source-inventory.json",
            "--policy",
            "configs/scientific/dataset-policy.yaml",
            "--lock-output",
            str(lock_output),
            "--card-output",
            str(card_output),
            "--mode",
            "synthetic",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.startswith("PASS dataset-lock sealed: ")
    assert lock_output.is_file()
    assert card_output.is_file()
