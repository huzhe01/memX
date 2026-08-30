from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jsonschema

from ratemem.evaluation.pools import (
    PoolManifestLine,
    build_locked_pools,
    read_image_records,
    read_prompt_templates,
)

IMAGES = Path("tests/fixtures/scientific/images.jsonl")
PROMPTS = Path("tests/fixtures/scientific/prompts.jsonl")


def test_pool_schema_matches_model_and_validates_every_public_line(tmp_path: Path) -> None:
    generated = tmp_path / "pool.schema.json"
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "data",
            "pool-schema",
            "--output",
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    committed = Path("schemas/scientific-pool-manifest.schema.json")
    assert generated.read_bytes() == committed.read_bytes()

    result = build_locked_pools(
        read_image_records(IMAGES),
        read_prompt_templates(PROMPTS),
        split_seed=87321,
        output_dir=tmp_path / "pools",
    )
    schema = PoolManifestLine.model_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    for path in result.manifest_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            jsonschema.validate(json.loads(line), schema)


def test_pool_outputs_are_byte_identical_across_builds(tmp_path: Path) -> None:
    records = read_image_records(IMAGES)
    prompts = read_prompt_templates(PROMPTS)
    first = build_locked_pools(records, prompts, split_seed=87321, output_dir=tmp_path / "a")
    second = build_locked_pools(records, prompts, split_seed=87321, output_dir=tmp_path / "b")

    first_files = {
        path.relative_to(first.output_dir): path.read_bytes()
        for path in sorted(first.output_dir.iterdir())
    }
    second_files = {
        path.relative_to(second.output_dir): path.read_bytes()
        for path in sorted(second.output_dir.iterdir())
    }
    assert first_files == second_files
    assert first.manifest_sha256 == second.manifest_sha256


def test_build_pools_cli_joins_explicit_split_assignments(tmp_path: Path) -> None:
    output = tmp_path / "cli-pools"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "data",
            "build-pools",
            "--source-catalog",
            str(IMAGES),
            "--prompt-catalog",
            str(PROMPTS),
            "--split-assignments",
            "tests/fixtures/scientific/split-assignments.jsonl",
            "--split-seed",
            "87321",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.endswith(
        "PASS pools: train/validation/final_test concept pools and prompt namespaces are disjoint\n"
    )
    assert output.is_dir()
