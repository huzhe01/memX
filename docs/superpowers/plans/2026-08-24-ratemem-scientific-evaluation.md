# RateMem-DiT Scientific Data and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, byte-accurate scientific data and evaluation pipeline whose locked traces, controls, statistics, human-study exports, gates, and paper tables can support or falsify the RateMem-DiT claims without test leakage or hand-entered results.

**Architecture:** A typed locking layer canonicalizes dataset, trace, baseline, and evaluation records and binds them by SHA-256. Deterministic replay adapters run every eligible method on the same immutable lifecycle events, write schema-validated raw artifacts, and derive paired hierarchical statistics, gates, and paper releases exclusively from those artifacts. The final-test event payload remains X25519/ChaCha20-Poly1305 encrypted until a signed freeze permit is consumed once; optional augmentation has a separate lock and cannot run until the core hard-budget gates pass.

**Tech Stack:** Python 3.11, `uv`, Pydantic 2, Typer, PyYAML, JSON Schema, pandas/Parquet, NumPy/SciPy, scikit-learn, statsmodels, Pillow/ImageHash/OpenCV, `cryptography`, Jinja2, pytest, Ruff, mypy

---

## Execution boundary

Execute `docs/superpowers/plans/2026-08-24-ratemem-core-memory.md` first; it owns and creates `.python-version`, `.gitignore`, `pyproject.toml`, `uv.lock`, the base `src/ratemem/` package, lifecycle interfaces, and credential-safe core artifacts. Execute `docs/superpowers/plans/2026-08-24-ratemem-sana-modal-pilot.md` second; it additively extends that frozen scaffold with SANA/amortizer interfaces and `schemas/ratemem-pilot-attempt-v1.schema.json`. Then use this exact cross-plan order:

1. Complete this plan's Tasks 1--7 and Task 8 Steps 1--5. At that boundary the dataset/traces/evaluator policy, provider-neutral baseline registry/protocol, frozen comparator/search-budget policy, and narrow `baseline_fidelity` authorization exist, while `configs/scientific/evaluation-lock.yaml` is intentionally absent.
2. Execute `docs/superpowers/plans/2026-08-24-ratemem-matched-baselines.md`. Its pre-lock audit may use a locked synthetic `SharedInputBundle` to prove the provider-neutral protocol and byte ledger, and may use the narrow permit only for source/real-checkpoint fidelity; it must not consume validation outcomes, the final trace, or a learned RateMem dictionary.
3. Resume Task 8 Steps 6--8 to validate the companion handoff, seal `configs/scientific/baseline-lock.yaml`, and then seal `configs/scientific/evaluation-lock.yaml`.
4. Execute `docs/superpowers/plans/2026-08-24-ratemem-learned-method-training.md` through its non-paid CPU gate. Produce and immediately revalidate `artifacts/method/cpu-gate.json`; do not launch paid training yet.
5. Complete this plan's Task 9. Only its distinct full scientific authorization may fund learned training, real shared-input materialization, baseline tuning/search, or comparative evaluation. Freeze those post-gate receipts before model selection or final-trace opening, then complete Tasks 10--18.
6. Execute the paper plan last. It may consume only the validated paper release and may not contain hand-entered result values.

This plan extends its predecessors without replacing their dependencies, scripts, tool settings, ignore rules, or tests, and must not change the legacy TensorFlow/GAN files.

Scientific runs are fail-closed. Source licenses, immutable revisions, evaluator revisions, margins, byte budgets, trace hashes, baseline fidelity records, and final-test approvals are generated from audited inputs; a missing or mutable value blocks sealing rather than being represented by a sentinel string. Synthetic fixtures live only under `tests/fixtures/scientific/` and are never accepted by a run whose mode is `scientific`.

The engineering-pilot authorization is not a scientific-compute authorization. Scientific CPU work may run locally, but every paid scientific phase requires a newly issued, scope-bound authorization for one explicitly operator-selected workspace and one named phase. The pre-lock `baseline_fidelity` scope is restricted to held-in or dedicated calibration inputs and source/real-checkpoint fidelity; the post-lock `scientific` scope requires the dataset lock, baseline/evaluation locks, and learned-method CPU receipt. Neither scope may automatically discover, reuse, rotate, or fall back across supplied workspaces. Both verify that the selected workspace's outer usage cap is exactly USD 28.00 and atomically reserve `known usage + pending worst-case + new phase bound <= USD 27.00` in that workspace's append-only ledger before launch. If the current workspace can admit no further phase, execution stops; another workspace requires a separate explicit author decision, new named profile and selection record, fresh cap/usage evidence, and a distinct workspace partition in the append-only ledger, never an automatic continuation after failure. Credentials and raw provider configuration remain outside the repository, command arguments, environment dumps, logs, and artifacts.

## Locked file map

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Add the `science` dependency extra and `ratemem-eval` entry point. |
| `src/ratemem/evaluation/types.py` | Strict-mypy-compatible named `Annotated` string aliases shared by every scientific Pydantic model. |
| `src/ratemem/evaluation/canonical.py` | Canonical JSON/YAML bytes, SHA-256, atomic writes, schema validation. |
| `src/ratemem/evaluation/dataset_lock.py` | Source inventories, dataset lock, immutable pool manifests, and data-card rendering. |
| `src/ratemem/evaluation/leakage.py` | Exact/decoded/perceptual/crop/burst near-duplicate evidence, global components, and split enforcement. |
| `src/ratemem/evaluation/traces.py` | Typed lifecycle events and deterministic concept-disjoint trace construction. |
| `src/ratemem/evaluation/final_trace.py` | Public-key final-trace envelope, signed freeze permit, stream-only one-time opening ledger. |
| `src/ratemem/evaluation/evaluation_lock.py` | Evaluator, formula, margin, workload, budget, power, and multiplicity lock. |
| `src/ratemem/evaluation/baselines.py` | Required-control registry, canonical companion-protocol re-export, audit-handoff validation, execution-receipt freeze, and strongest-control selection. |
| `src/ratemem/evaluation/compute.py` | Narrow pre-lock fidelity and full post-lock scientific permits sharing one explicit-workspace USD 28 cap/USD 27 aggregate reservation ledger. |
| `src/ratemem/evaluation/replay.py` | Immutable no-pressure, budget-pressure, and autonomous-lookup replay. |
| `src/ratemem/evaluation/metrics.py` | Locked lifecycle, byte, latency, lookup, deletion, and diversity estimands. |
| `src/ratemem/evaluation/statistics.py` | Paired inference-unit aggregation, hierarchical bootstrap, Holm correction, and CI-width/power calculation. |
| `src/ratemem/evaluation/artifacts.py` | Attempt/result validation, checksums, provenance joins, and artifact-only aggregation. |
| `src/ratemem/evaluation/human_study.py` | Balanced blinded-pair export, encrypted blinding key, response freeze/import, and paired analysis rows. |
| `src/ratemem/evaluation/publish.py` | Exact CSV/JSON paper release with a checksummed manifest. |
| `src/ratemem/evaluation/gates.py` | Core and optional gate predicates and reason codes. |
| `src/ratemem/evaluation/augmentation.py` | Class-disjoint optional augmentation lock, controls, statistics, and all-dataset gate. |
| `src/ratemem/evaluation/cli.py` | Typer commands that expose each auditable transition. |
| `configs/scientific/*.yaml` | Prespecified policies and sealed locks. |
| `configs/scientific/compute-policy.yaml` | Scientific authorization scope, cap, reservation, freshness, and credential-exclusion policy. |
| `configs/scientific/traces/*` | Plain train/validation manifests and encrypted final-test envelope. |
| `schemas/*scientific*.schema.json` | Generated JSON Schemas checked for drift in tests. |
| `docs/data/ratemem-scientific-data-card.md` | Generated source, license, pool, contamination, and leakage disclosure. |
| `artifacts/scientific/<run_id>/` | Immutable raw/derived attempt artifacts; never a source file for hand edits. |
| `artifacts/paper/<release_id>/` | Validated tables, curves, study summary, qualitative selection, and release manifest consumed by the paper build. |

### Task 1: Establish canonical scientific records and the CLI

> Execution note (2026-08-30): the repository dependency lock had already advanced beyond the
> versions in this August 24 draft. This task preserved the verified shared pins (including Torch
> 2.13.0, Pillow 12.3.0, safetensors 0.8.0, and jsonschema 4.26.0) and added only compatible
> scientific dependencies; it did not downgrade the working SANA/company runtime.

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/ratemem/evaluation/__init__.py`
- Create: `src/ratemem/evaluation/types.py`
- Create: `src/ratemem/evaluation/canonical.py`
- Create: `src/ratemem/evaluation/cli.py`
- Test: `tests/unit/evaluation/test_canonical.py`

- [x] **Step 1: Verify the completed core and SANA/pilot scaffold**

Run: `sed -n '1,260p' pyproject.toml && uv tree --depth 1 && uv run ratemem --help && uv run ratemem-pilot --help`

Expected: exit 0; Python is constrained to 3.11; the existing graph includes core `cbor2`, NumPy, Pydantic, and PyYAML plus the pinned SANA/pilot stack; both `ratemem` and `ratemem-pilot` scripts work. Stop if either predecessor is incomplete rather than recreating its files or package smoke tests.

- [x] **Step 2: Add the pinned scientific dependencies and console script**

Extend the pilot plan's existing Python 3.11/`uv` `pyproject.toml` and `uv.lock`; add this extra and entry point without replacing existing project metadata, dependency pins, scripts, or extras:

```toml
[project.optional-dependencies]
science = [
  "cryptography==45.0.6",
  "ImageHash==4.3.2",
  "Jinja2==3.1.6",
  "jsonschema==4.25.1",
  "numpy==2.2.6",
  "opencv-python-headless==4.12.0.88",
  "pandas==2.3.2",
  "Pillow==11.3.0",
  "pyarrow==21.0.0",
  "pydantic==2.11.7",
  "PyYAML==6.0.2",
  "scikit-learn==1.7.1",
  "scipy==1.16.1",
  "statsmodels==0.14.5",
]

[project.scripts]
ratemem-eval = "ratemem.evaluation.cli:main"
```

Run: `uv lock && uv sync --all-extras --frozen`

Expected: exit 0 and `uv run ratemem-eval --help` lists the root command after Step 5.

- [x] **Step 3: Write the failing canonicalization tests**

```python
# tests/unit/evaluation/test_canonical.py
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from ratemem.evaluation.canonical import (
    MutableLockValueError,
    canonical_json_bytes,
    semantic_sha256,
    write_yaml_atomic,
)
from ratemem.evaluation.types import ConceptToken, GitCommit, PhaseId, ScientificProfile, Sha256


def test_canonical_hash_ignores_mapping_order_and_seal_metadata(tmp_path: Path) -> None:
    left = {"b": 2, "a": 1, "lock_id": "old", "sealed_at_utc": "2026-01-01T00:00:00Z"}
    right = {"a": 1, "b": 2, "lock_id": "new", "sealed_at_utc": "2026-08-24T00:00:00Z"}
    assert canonical_json_bytes(left) != canonical_json_bytes(right)
    assert semantic_sha256(left) == semantic_sha256(right)
    out = tmp_path / "nested" / "lock.yaml"
    write_yaml_atomic(out, {"a": 1, "b": 2})
    assert out.read_text() == "a: 1\nb: 2\n"


@pytest.mark.parametrize("value", ["latest", "main", "unknown", "", "unresolved"])
def test_mutable_lock_values_are_rejected(value: str) -> None:
    from ratemem.evaluation.canonical import require_immutable_value

    with pytest.raises(MutableLockValueError):
        require_immutable_value("revision", value)


def test_named_string_aliases_enforce_their_exact_patterns() -> None:
    assert TypeAdapter(Sha256).validate_python("a" * 64) == "a" * 64
    assert TypeAdapter(GitCommit).validate_python("b" * 40) == "b" * 40
    assert TypeAdapter(ConceptToken).validate_python("<concept_000123>") == "<concept_000123>"
    assert TypeAdapter(ScientificProfile).validate_python("ratemem-scientific-study-a") == "ratemem-scientific-study-a"
    assert TypeAdapter(PhaseId).validate_python("meta_train_seed_0") == "meta_train_seed_0"
    with pytest.raises(ValidationError):
        TypeAdapter(Sha256).validate_python("not-a-hash")
```

- [x] **Step 4: Run the canonicalization tests and verify the expected failure**

Run: `uv run pytest tests/unit/evaluation/test_canonical.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'ratemem.evaluation'`.

- [x] **Step 5: Implement named constrained-string aliases, canonical writes, hashes, and the Typer root**

```python
# src/ratemem/evaluation/types.py
from typing import Annotated

from pydantic import StringConstraints

Sha256 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
ConceptToken = Annotated[str, StringConstraints(strict=True, pattern=r"^<concept_[0-9]{6}>$")]
ScientificProfile = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^ratemem-scientific-[a-z0-9_-]+$"),
]
PhaseId = Annotated[str, StringConstraints(strict=True, pattern=r"^[a-z0-9_-]+$")]
```

```python
# src/ratemem/evaluation/canonical.py
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

MUTABLE_VALUES = {"", "latest", "main", "master", "unknown", "unresolved", "to be determined", "not set"}
SEAL_METADATA = {"lock_id", "sealed_at_utc"}


class MutableLockValueError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def semantic_sha256(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key not in SEAL_METADATA}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_immutable_value(field: str, value: str) -> str:
    normalized = value.strip().lower()
    if normalized in MUTABLE_VALUES:
        raise MutableLockValueError(f"{field} must be an immutable resolved value")
    return value


def write_yaml_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(value, sort_keys=True, allow_unicode=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
```

```python
# src/ratemem/evaluation/cli.py
import typer

app = typer.Typer(no_args_is_help=True, help="RateMem-DiT locked scientific evaluation")


def main() -> None:
    app()
```

Export only stable public types from `src/ratemem/evaluation/__init__.py`; begin with `__all__: list[str] = []`.

- [x] **Step 6: Run the focused test and static checks**

Run: `uv run pytest tests/unit/evaluation/test_canonical.py -q && uv run ruff check src/ratemem/evaluation tests/unit/evaluation && uv run mypy src/ratemem/evaluation`

Expected: `6 passed`, followed by Ruff and mypy exit 0.

- [x] **Step 7: Commit the canonical layer**

```bash
git add pyproject.toml uv.lock src/ratemem/evaluation tests/unit/evaluation/test_canonical.py
git commit -m "feat(eval): add canonical scientific records"
```

### Task 2: Seal exact dataset inventories and render the data card

**Files:**
- Create: `configs/scientific/dataset-policy.yaml`
- Create: `src/ratemem/evaluation/dataset_lock.py`
- Create: `schemas/dataset-lock.schema.json`
- Create: `tests/fixtures/scientific/source-inventory.json`
- Test: `tests/unit/evaluation/test_dataset_lock.py`
- Test: `tests/contract/evaluation/test_dataset_lock_schema.py`

- [x] **Step 1: Add the fail-closed dataset policy**

```yaml
# configs/scientific/dataset-policy.yaml
schema_version: "1.0"
required_roles:
  engineering_pilot: [subjects200k_pilot]
  meta_training: [subjects200k]
  primary_one_shot_evaluation: [dreambench_plus_plus]
  multi_shot_evaluation: [customconcept101_eligible]
  controlled_post_checkpoint_evaluation: [controlled_post_checkpoint]
optional_roles:
  multi_image_training: [syncd]
  historical_stress: [omniglot]
anonymous_token_pattern: "^<concept_[0-9]{6}>$"
forbid_real_identity_names: true
require_cluster_disjoint_splits: true
require_immutable_support_query_pools: true
require_license_attestation: true
require_pretraining_contamination_disclosure_for:
  - sana_backbone
  - support_encoder
  - identity_evaluator
  - prompt_evaluator
role_to_allowed_splits:
  engineering_pilot: [train]
  meta_training: [train, validation]
  multi_image_training: [train, validation]
  primary_one_shot_evaluation: [final_test]
  multi_shot_evaluation: [final_test]
  controlled_post_checkpoint_evaluation: [final_test]
  historical_stress: [final_test]
within_training_source_concept_split: {train: 0.90, validation: 0.10}
scientific_modes: [calibration, validation, final_test]
```

- [x] **Step 2: Write failing lock and data-card tests with a fully resolved synthetic inventory**

The synthetic inventory must contain one resolved source for each of the six required source IDs, immutable 40-character revisions, SPDX licenses, provenance URLs, exact concept/image/pair counts, min/median/max image dimensions, caption/mask counts, immutable support/query pool paths and hashes, and backbone/evaluator contamination statements. Test these behaviors:

```python
def test_seal_dataset_lock_binds_every_pool_and_renders_card(tmp_path: Path) -> None:
    inventory = load_inventory(Path("tests/fixtures/scientific/source-inventory.json"))
    lock = seal_dataset_lock(inventory, policy_path=Path("configs/scientific/dataset-policy.yaml"))
    assert lock.lock_id == semantic_sha256(lock.model_dump(mode="json"))
    assert {pool.kind for source in lock.sources for pool in source.pools} >= {"support", "query"}
    card = render_data_card(lock)
    assert "## Licenses and allowed uses" in card
    assert "## Duplicate, derivative, and contamination audit" in card
    assert "synthetic-primary" in card


def test_scientific_lock_rejects_missing_post_checkpoint_source(resolved_inventory: dict) -> None:
    resolved_inventory["sources"] = [
        source for source in resolved_inventory["sources"]
        if source["role"] != "controlled_post_checkpoint_evaluation"
    ]
    with pytest.raises(DatasetLockError, match="controlled_post_checkpoint_evaluation"):
        seal_dataset_lock(SourceInventory.model_validate(resolved_inventory), POLICY)
```

- [x] **Step 3: Run the dataset-lock tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_dataset_lock.py tests/contract/evaluation/test_dataset_lock_schema.py -q`

Expected: collection fails because `ratemem.evaluation.dataset_lock` does not exist.

- [x] **Step 4: Implement strict source, pool, license, statistics, and contamination models**

Implement these signatures in `src/ratemem/evaluation/dataset_lock.py`:

```python
from ratemem.evaluation.types import Sha256


class LicenseAttestation(BaseModel):
    spdx_id: str
    license_url: AnyUrl
    verified_by: str
    verified_at_utc: AwareDatetime
    research_use_allowed: Literal[True]
    redistribution_allowed: bool


class PoolLock(BaseModel):
    kind: Literal["support", "query", "caption", "mask", "prompt"]
    manifest_path: str
    sha256: Sha256
    record_count: PositiveInt
    concept_count: PositiveInt


class SourceLock(BaseModel):
    source_id: str
    role: Literal[
        "engineering_pilot", "meta_training", "multi_image_training",
        "primary_one_shot_evaluation", "multi_shot_evaluation",
        "controlled_post_checkpoint_evaluation", "historical_stress",
    ]
    upstream_uri: AnyUrl
    immutable_revision: str
    provenance_uri: AnyUrl
    license: LicenseAttestation
    concept_unit: str
    concept_count: PositiveInt
    image_count: PositiveInt
    pair_count: NonNegativeInt
    image_width_min: PositiveInt
    image_width_median: PositiveFloat
    image_width_max: PositiveInt
    image_height_min: PositiveInt
    image_height_median: PositiveFloat
    image_height_max: PositiveInt
    caption_count: NonNegativeInt
    mask_count: NonNegativeInt
    pools: list[PoolLock]
    allowed_uses: list[str]
    pretraining_contamination: dict[str, str]
    evaluation_pool_semantics: Literal["held_out_query", "reference_prompt_only", "training_pairs", "historical_stress"]
    eligible_support_shots: list[Literal[1, 3, 5]]
    limitations: list[str]


class DatasetLock(BaseModel):
    schema_version: Literal["1.0"]
    lock_id: Sha256
    sealed_at_utc: AwareDatetime
    global_image_manifest_path: str
    global_image_manifest_sha256: Sha256
    duplicate_report_path: str
    duplicate_report_sha256: Sha256
    split_assignment_path: str
    split_assignment_sha256: Sha256
    sources: list[SourceLock]


```

Implement the exact signatures `load_inventory(path: Path) -> SourceInventory`, `seal_dataset_lock(inventory: SourceInventory, policy_path: Path) -> DatasetLock`, `render_data_card(lock: DatasetLock) -> str`, and `write_dataset_lock_and_card(lock: DatasetLock, lock_path: Path, card_path: Path) -> None`.

Every revision, URL, license attestation, pool hash, audit hash, and split hash is required. Call `require_immutable_value` for revisions and contamination statements, check that support/query image identifiers are disjoint, require the roles in `dataset-policy.yaml`, and reject any `mode: synthetic` inventory when the requested output mode is scientific. Require DreamBench++ to record `reference_prompt_only` rather than describing it as a real held-out-query set; derive the eligible CustomConcept101 shot list from distinct-image counts; record Subjects200K's pair/mostly-one-shot limitation; and omit SynCD unless its license/provenance attestation passes.

- [x] **Step 5: Generate and contract-test the JSON Schema**

Add `ratemem-eval data schema --output schemas/dataset-lock.schema.json`, implemented as `DatasetLock.model_json_schema()` serialized with canonical JSON. The contract test regenerates to a temporary path and byte-compares it to the committed schema.

Run: `uv run ratemem-eval data schema --output schemas/dataset-lock.schema.json && uv run pytest tests/unit/evaluation/test_dataset_lock.py tests/contract/evaluation/test_dataset_lock_schema.py -q`

Expected: all tests pass and the command prints `PASS dataset-lock schema: schemas/dataset-lock.schema.json`.

- [x] **Step 6: Add the sealing command and exercise it on the resolved synthetic fixture**

Expose this exact command:

```bash
uv run ratemem-eval data seal \
  --inventory artifacts/scientific/dataset-audit/source-inventory.json \
  --policy configs/scientific/dataset-policy.yaml \
  --lock-output configs/scientific/dataset-lock.yaml \
  --card-output docs/data/ratemem-scientific-data-card.md \
  --mode scientific
```

Expected against the synthetic fixture with `--mode synthetic`: exit 0 and a schema-valid temporary lock/card. Expected for the shown scientific command before Tasks 3–4 produce the audited inventory: exit 2 with `BLOCKED dataset-lock: audited source inventory is missing`. The command never creates a partial lock or card. Run the real scientific invocation only in Task 4, after duplicate components and immutable pools exist.

- [x] **Step 7: Commit the dataset lock machinery**

```bash
git add configs/scientific/dataset-policy.yaml src/ratemem/evaluation/dataset_lock.py schemas/dataset-lock.schema.json tests/fixtures/scientific/source-inventory.json tests/unit/evaluation/test_dataset_lock.py tests/contract/evaluation/test_dataset_lock_schema.py
git commit -m "feat(eval): seal scientific dataset inventories"
```

### Task 3: Build immutable image/prompt pools and derivative lineage

**Files:**
- Create: `src/ratemem/evaluation/pools.py`
- Create: `schemas/scientific-pool-manifest.schema.json`
- Create: `tests/fixtures/scientific/images.jsonl`
- Test: `tests/unit/evaluation/test_pools.py`
- Test: `tests/contract/evaluation/test_pool_schema.py`

- [x] **Step 1: Write the failing pool-manifest tests**

```python
def test_pool_builder_anonymizes_names_and_keeps_support_query_disjoint(tmp_path: Path) -> None:
    records = read_image_records(Path("tests/fixtures/scientific/images.jsonl"))
    result = build_locked_pools(records, split_seed=87321, output_dir=tmp_path)
    assert set(result.support_image_ids).isdisjoint(result.query_image_ids)
    assert all(re.fullmatch(r"<concept_[0-9]{6}>", token) for token in result.concept_tokens)
    assert not any("Ada" in prompt for prompt in result.rendered_prompts)


def test_derivative_of_query_cannot_enter_training_pool(image_records: list[ImageRecord]) -> None:
    image_records[0].split = "train"
    image_records[0].derivative_of = image_records[-1].image_id
    image_records[-1].split = "final_test"
    with pytest.raises(PoolLeakageError, match="derivative ancestry crosses splits"):
        build_locked_pools(image_records, split_seed=87321)
```

- [x] **Step 2: Run the pool tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_pools.py -q`

Expected: collection fails because `ratemem.evaluation.pools` does not exist.

- [x] **Step 3: Implement the exact per-image record and deterministic pool writer**

```python
from ratemem.evaluation.types import ConceptToken, Sha256


class ImageRecord(BaseModel):
    image_id: str
    source_id: str
    concept_id: str
    content_sha256: Sha256
    decoded_pixel_sha256: Sha256
    width: PositiveInt
    height: PositiveInt
    caption_sha256: Sha256 | None = None
    mask_sha256: Sha256 | None = None
    derivative_of: str | None = None
    capture_group: str | None = None
    split: Literal["train", "validation", "final_test"]
    eligible_for_support: bool
    eligible_for_query: bool


class PromptRecord(BaseModel):
    prompt_id: str
    template_id: str
    split: Literal["train", "validation", "final_test"]
    template_sha256: Sha256
    anonymous_concept_token: ConceptToken


def build_locked_pools(
    records: Sequence[ImageRecord],
    prompts: Sequence[PromptTemplate],
    split_seed: int,
    output_dir: Path,
) -> PoolBuildResult:
    return PoolBuilder(split_seed=split_seed).build(records, prompts, output_dir)
```

Sort by `(split, source_id, concept_id, image_id)`, derive anonymous tokens with `HMAC-SHA256(split_seed, concept_id)[:6]`, preserve the private concept-to-token map only in the restricted audit artifact, traverse every `derivative_of` chain, and write one canonical JSONL manifest per `(source, split, support/query)` pool. Prompt templates are split before rendering, and their hashes—not identity-bearing text—enter the public trace manifest.

- [x] **Step 4: Add schema generation and verify deterministic bytes**

Run the builder twice into two temporary directories in the contract test and assert identical filenames, record order, SHA-256 values, and pool summary. Generate `schemas/scientific-pool-manifest.schema.json` from `ImageRecord` plus a manifest header.

Run: `uv run pytest tests/unit/evaluation/test_pools.py tests/contract/evaluation/test_pool_schema.py -q`

Expected: all tests pass, including byte-identical pool output across both builds.

- [x] **Step 5: Add the pool command and exercise it on synthetic split assignments**

```bash
uv run ratemem-eval data build-pools \
  --source-catalog artifacts/scientific/dataset-audit/normalized-images.parquet \
  --prompt-catalog artifacts/scientific/dataset-audit/prompt-templates.jsonl \
  --split-assignments artifacts/scientific/dataset-audit/split-assignments.parquet \
  --split-seed 87321 \
  --output artifacts/scientific/dataset-audit/pools
```

Expected on the synthetic fixture: `PASS pools: train/validation/final_test concept pools and prompt namespaces are disjoint`; any derivative/caption/mask/prompt crossing exits 2 and writes no pool manifests. Defer the shown real-data invocation until Task 4 writes audited split assignments.

- [x] **Step 6: Commit the immutable pool builder**

```bash
git add src/ratemem/evaluation/pools.py schemas/scientific-pool-manifest.schema.json tests/fixtures/scientific/images.jsonl tests/unit/evaluation/test_pools.py tests/contract/evaluation/test_pool_schema.py
git commit -m "feat(eval): lock support query and prompt pools"
```

### Task 4: Audit exact duplicates, crops, recompressions, burst neighbors, and feature leakage

> Execution note (2026-08-30): Steps 1--5 are verified with deterministic synthetic images and a
> frozen injected encoder contract. The real catalog audit and scientific dataset seal in Steps
> 6--7 remain deliberately unclaimed until the audited company catalog and locked DINOv2 weight
> bytes are available.

**Files:**
- Create: `configs/scientific/duplicate-policy.yaml`
- Create: `src/ratemem/evaluation/leakage.py`
- Create: `schemas/scientific-duplicate-report.schema.json`
- Test: `tests/unit/evaluation/test_leakage.py`
- Test: `tests/integration/evaluation/test_duplicate_audit.py`

- [x] **Step 1: Pin the pre-split candidate and adjudication policy**

```yaml
# configs/scientific/duplicate-policy.yaml
schema_version: "1.0"
decoded_pixel_hash: sha256_exif_transposed_rgb_v1
perceptual_hash: phash_64_v1
phash_max_hamming: 4
feature_encoder:
  name: facebook/dinov2-large
  immutable_lock_path: artifacts/scientific/dataset-audit/duplicate-feature-encoder-lock.json
  preprocessing: resize_shorter_518_center_crop_rgb_v1
feature_cosine_min: 0.985
crop_verification:
  detector: opencv_sift_homography_v1
  minimum_inliers: 12
  minimum_inlier_ratio: 0.35
burst_neighbor:
  maximum_seconds: 2.0
  feature_cosine_min: 0.97
ambiguous_pair_review:
  independent_reviewers: 2
  disagreement_resolution: third_reviewer_majority
cluster_assignment: connected_components_v1
```

The DINO revision and thresholds are frozen before any split. If calibration changes them, commit a new policy version and rerun the entire global audit before model selection; never patch clusters in place.

Before the audit, bind the locally verified repository commit and weight bytes from the audited model inventory:

```bash
uv run ratemem-eval data lock-feature-encoder \
  --model-id dinov2_large_duplicate_audit \
  --model-inventory artifacts/scientific/dataset-audit/model-inventory.json \
  --output artifacts/scientific/dataset-audit/duplicate-feature-encoder-lock.json
```

Expected: exit 0 and stdout matches `^PASS duplicate-feature-encoder lock: revision=[0-9a-f]{40,64} weights=[0-9a-f]{64}$`; mutable branches or weights without a local SHA-256 exit 2.

- [x] **Step 2: Write failing unit tests for evidence linking and whole-component splitting**

```python
def test_any_strong_duplicate_evidence_links_records(policy: DuplicatePolicy) -> None:
    evidence = PairEvidence(
        left_id="a", right_id="b", exact_sha256=False, decoded_pixel_equal=False,
        phash_hamming=3, feature_cosine=0.99, sift_inliers=15,
        sift_inlier_ratio=0.5, capture_delta_seconds=None,
    )
    assert classify_pair(evidence, policy) == PairDecision.LINK


def test_component_crossing_preassigned_splits_is_rejected() -> None:
    records = [record("a", "concept-a", "train"), record("b", "concept-b", "final_test")]
    with pytest.raises(CrossSplitComponentError, match="a.*b"):
        assign_components(records, linked_pairs=[("a", "b")])


def test_concepts_connected_by_duplicate_images_share_one_split() -> None:
    assignment = allocate_component_splits(
        concept_components=[{"c1", "c2"}, {"c3"}], ratios={"train": 0.5, "validation": 0.25, "final_test": 0.25}, seed=87321
    )
    assert assignment["c1"] == assignment["c2"]
```

- [x] **Step 3: Run the leakage tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_leakage.py -q`

Expected: collection fails because `ratemem.evaluation.leakage` does not exist.

- [x] **Step 4: Implement fingerprints, pair evidence, union-find, and adjudication validation**

Implement these stable interfaces:

```python
class PairDecision(str, Enum):
    LINK = "link"
    REVIEW = "review"
    DISTINCT = "distinct"


@dataclass(frozen=True)
class PairEvidence:
    left_id: str
    right_id: str
    exact_sha256: bool
    decoded_pixel_equal: bool
    phash_hamming: int | None
    feature_cosine: float | None
    sift_inliers: int | None
    sift_inlier_ratio: float | None
    capture_delta_seconds: float | None


```

Implement the exact signatures `fingerprint_image(path: Path, encoder: FrozenFeatureEncoder) -> ImageFingerprint`, `classify_pair(evidence: PairEvidence, policy: DuplicatePolicy) -> PairDecision`, `validate_adjudications(candidates: Sequence[PairEvidence], reviews: Sequence[Review]) -> list[tuple[str, str]]`, `assign_components(records: Sequence[ImageRecord], linked_pairs: Sequence[tuple[str, str]]) -> ComponentReport`, and `allocate_component_splits(concept_components: Sequence[set[str]], ratios: Mapping[str, float], seed: int) -> dict[str, str]`.

Use exact content and decoded-pixel hashes first, LSH only to propose pHash/feature candidates, SIFT homography to verify crop candidates, and capture-group/time evidence for burst/video neighbors. The final connected components must include accepted two-reviewer adjudications. Record every candidate rule, score, immutable encoder repository/revision/weights hash from `duplicate-feature-encoder-lock.json`, reviewer decision hash, and component membership in the report.

- [x] **Step 5: Add a tiny end-to-end audit fixture**

Create four small test images: one original, one JPEG recompression, one crop, and one visually different image. Assert the first three share a component, the last does not, and a preassigned train/final collision exits before writing split assignments. Do not assert exact DINO floating-point values; assert the declared tolerance and recorded encoder revision.

Run: `uv run pytest tests/unit/evaluation/test_leakage.py tests/integration/evaluation/test_duplicate_audit.py -q`

Expected: all tests pass; the integration report validates against `schemas/scientific-duplicate-report.schema.json`.

- [ ] **Step 6: Expose the global audit command**

```bash
uv run ratemem-eval data audit \
  --images artifacts/scientific/dataset-audit/normalized-images.parquet \
  --policy configs/scientific/duplicate-policy.yaml \
  --adjudications artifacts/scientific/dataset-audit/adjudications.jsonl \
  --output artifacts/scientific/dataset-audit/duplicate-report.json \
  --splits-output artifacts/scientific/dataset-audit/split-assignments.parquet
```

Expected on success: stdout matches `^PASS dataset-audit: zero cross-split components; report=[0-9a-f]{64}$`. With unresolved reviews, exit 2 and stdout matches `^BLOCKED dataset-audit: [0-9]+ ambiguous pairs require adjudication$`; no split file is written.

- [ ] **Step 7: Run the real audit, build pools, seal the dataset, and commit the lock**

Run the Task 4 audit command, then the Task 3 `data build-pools` command, then the Task 2 `data seal --mode scientific` command. Verify each reports `PASS` and that the data-card source table explicitly marks DreamBench++ as reference/prompt evaluation rather than held-out-query evidence.

```bash
git add configs/scientific/duplicate-policy.yaml src/ratemem/evaluation/leakage.py schemas/scientific-duplicate-report.schema.json tests/unit/evaluation/test_leakage.py tests/integration/evaluation/test_duplicate_audit.py
git commit -m "feat(eval): enforce global duplicate leakage audit"
git add configs/scientific/dataset-lock.yaml docs/data/ratemem-scientific-data-card.md
git commit -m "data: freeze audited scientific dataset lock"
```

### Task 5: Freeze sample size and generate concept-disjoint lifecycle trace manifests

**Files:**
- Create: `configs/scientific/trace-policy.yaml`
- Create: `src/ratemem/evaluation/statistics.py`
- Create: `src/ratemem/evaluation/traces.py`
- Create: `schemas/scientific-calibration-record.schema.json`
- Create: `schemas/scientific-trace-manifest.schema.json`
- Create: `tests/fixtures/scientific/concept-pools.json`
- Test: `tests/unit/evaluation/test_power_planning.py`
- Test: `tests/unit/evaluation/test_traces.py`
- Test: `tests/contract/evaluation/test_trace_manifest.py`

- [ ] **Step 1: Write the failing CI-width and power-planning tests**

```python
def test_required_units_uses_larger_ci_or_power_requirement(calibration_record: CalibrationRecord) -> None:
    result = plan_required_units(
        calibration_record,
        maximum_half_width=0.02,
        minimum_effect=0.03,
        alpha=0.05,
        power=0.80,
        minimum_units=12,
        simulation_seed=314159,
    )
    assert result.required_units == max(result.ci_required_units, result.power_required_units)
    assert result.calibration_pool_sha256 == calibration_record.pool_sha256


def test_power_record_rejects_final_test_concepts(calibration_record: CalibrationRecord) -> None:
    calibration_record.split = "final_test"
    with pytest.raises(CalibrationLeakageError):
        plan_required_units(calibration_record, 0.02, 0.03, 0.05, 0.80, 12, 314159)
```

- [ ] **Step 2: Run the power-planning test and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_power_planning.py -q`

Expected: collection fails because `ratemem.evaluation.statistics` does not exist.

- [ ] **Step 3: Implement the calibration record and deterministic planning simulation**

Define `CalibrationRecord` with immutable dataset/evaluator/pool hashes, `split: Literal["calibration"]`, paired pilot effect rows, inference-unit ID, metric ID, and source artifact hashes. Implement `plan_required_units(record: CalibrationRecord, maximum_half_width: float, minimum_effect: float, alpha: float, power: float, minimum_units: int, simulation_seed: int) -> RequiredUnits`; use cluster resampling of inference units, find the smallest unit count whose simulated 95% CI half-width is at most 0.02 and power is at least 0.80, then select the larger CI/power count. Record the whole search curve, Monte Carlo draws, seed, and calibration hash.

- [ ] **Step 4: Freeze the required unit count before building any comparative trace**

```bash
uv run ratemem-eval stats plan-units \
  --calibration-record artifacts/scientific/calibration/calibration-record.json \
  --maximum-half-width 0.02 \
  --minimum-effect 0.03 \
  --alpha 0.05 \
  --power 0.80 \
  --minimum-units 12 \
  --simulation-seed 314159 \
  --output configs/scientific/required-units.json
```

Expected: exit 0 and stdout matches `^PASS power-plan: final deployment episodes=[0-9]+; target_half_width=0.02; power=0.80$`. The calibration record schema rejects final-test pools, architecture/model-selection outputs, or mutable evaluator revisions.

- [ ] **Step 5: Prespecify event generation and namespaces**

```yaml
# configs/scientific/trace-policy.yaml
schema_version: "1.0"
builder_revision: lifecycle_trace_v1
seed_namespaces:
  train: "ratemem:trace:train:v1"
  validation: "ratemem:trace:validation:v1"
  final_test: "ratemem:trace:final-test:v1"
event_probabilities:
  create: 0.20
  update: 0.10
  read: 0.55
  delete: 0.15
support_shots: [1, 3, 5]
maximum_update_support: 10
locked_active_set_size: 20
events_per_deployment_episode: 240
request_regimes:
  uniform: {kind: uniform}
  zipf: {kind: zipf, exponent: 1.2}
protocols: [no_pressure, budget_pressure, autonomous_lookup]
prompt_seed_pairing: strict
probe_update_usage: false
handle_format: "h_<trace-prefix>_<event-index>"
final_payload_visibility: encrypted_until_signed_freeze
```

Trace count and event count do not live in this policy: they are copied from `configs/scientific/required-units.json`, produced before comparative model development; the later evaluation lock binds that record and every generated trace hash. This prevents both an arbitrary trace ceiling and a trace/evaluation-lock dependency cycle.

- [ ] **Step 6: Write failing event and trace-separation tests**

```python
def test_trace_is_deterministic_and_read_probe_semantics_differ(pools: ConceptPools) -> None:
    first = build_trace(split="validation", trace_index=3, pools=pools, policy=POLICY, event_count=40)
    second = build_trace(split="validation", trace_index=3, pools=pools, policy=POLICY, event_count=40)
    assert first.model_dump_json() == second.model_dump_json()
    assert all(event.update_usage for event in first.events if event.kind == "read")
    assert all(not event.update_usage for event in first.events if event.kind == "probe")


def test_train_validation_final_concepts_trace_ids_and_seed_namespaces_are_disjoint(all_pools: AllPools) -> None:
    manifests = build_trace_set(all_pools, POLICY, counts={"train": 4, "validation": 3, "final_test": 3}, event_count=40)
    assert_pairwise_disjoint(manifest.concept_ids for manifest in manifests.values())
    assert_pairwise_disjoint(manifest.trace_ids for manifest in manifests.values())
    assert_pairwise_disjoint(manifest.generation_seeds for manifest in manifests.values())


def test_update_is_labeled_and_delete_never_reuses_handle(pools: ConceptPools) -> None:
    trace = build_trace(split="validation", trace_index=1, pools=pools, policy=POLICY, event_count=100)
    assert all(event.handle for event in trace.events if event.kind in {"update", "delete"})
    deleted = {event.handle for event in trace.events if event.kind == "delete"}
    later_creates = {event.handle for event in trace.events if event.kind == "create"}
    assert deleted.isdisjoint(later_creates)
```

- [ ] **Step 7: Run the trace tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_traces.py -q`

Expected: collection fails because `ratemem.evaluation.traces` does not exist.

- [ ] **Step 8: Implement the discriminated event schema**

```python
class CreateEvent(BaseModel):
    kind: Literal["create"]
    event_index: NonNegativeInt
    handle: str
    concept_token: str
    support_image_ids: list[str]
    description_id: str


class UpdateEvent(BaseModel):
    kind: Literal["update"]
    event_index: NonNegativeInt
    handle: str
    support_image_ids: list[str]


class ReadEvent(BaseModel):
    kind: Literal["read"]
    event_index: NonNegativeInt
    handle: str
    prompt_id: str
    generation_seed: int
    update_usage: Literal[True] = True


class DeleteEvent(BaseModel):
    kind: Literal["delete"]
    event_index: NonNegativeInt
    handle: str


class ProbeEvent(BaseModel):
    kind: Literal["probe"]
    event_index: NonNegativeInt
    snapshot_event_index: NonNegativeInt
    handle: str
    prompt_id: str
    generation_seed: int
    update_usage: Literal[False] = False


LifecycleEvent = Annotated[
    CreateEvent | UpdateEvent | ReadEvent | DeleteEvent | ProbeEvent,
    Field(discriminator="kind"),
]
```

Define `TraceManifest` with `trace_id`, `split`, `dataset_lock_id`, `trace_builder_revision`, `trace_seed`, `seed_namespace`, `request_regime`, `protocol`, `concept_pool_sha256`, `prompt_pool_sha256`, `payload_path`, `payload_sha256`, event counts by kind, and sorted concept/trace/seed commitments. Validate legal handle state transitions during construction.

- [ ] **Step 9: Implement deterministic generation and manifest hashes**

Provide these concrete interfaces: `build_trace(split: Split, trace_index: int, pools: ConceptPools, policy: TracePolicy, event_count: int) -> Trace`; `build_trace_set(pools: AllPools, policy: TracePolicy, counts: Mapping[Split, int], event_count: int) -> dict[Split, TraceSet]`; and `write_trace_set(trace_set: TraceSet, output_dir: Path) -> list[TraceManifest]`.

Derive each PRNG with `numpy.random.SeedSequence([dataset_lock_prefix, split_namespace_hash, trace_index])`; never share a generator across splits. Insert deterministic `PROBE` schedules separately from operational events so they cannot change event probabilities. Hash canonical JSONL payload bytes and validate the manifest against the generated schema.

- [ ] **Step 10: Generate the schemas and verify manifest tamper detection**

The contract test writes a manifest, changes one prompt seed in the JSONL payload, and asserts `TraceHashMismatch` before replay.

Run: `uv run ratemem-eval stats schema-calibration --output schemas/scientific-calibration-record.schema.json && uv run ratemem-eval traces schema --output schemas/scientific-trace-manifest.schema.json && uv run pytest tests/unit/evaluation/test_power_planning.py tests/unit/evaluation/test_traces.py tests/contract/evaluation/test_trace_manifest.py -q`

Expected: all tests pass and the schema command prints `PASS trace-manifest schema: schemas/scientific-trace-manifest.schema.json`.

- [ ] **Step 11: Build only development-visible train and validation traces**

```bash
uv run ratemem-eval traces build-visible \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --policy configs/scientific/trace-policy.yaml \
  --power-record configs/scientific/required-units.json \
  --splits train,validation \
  --output configs/scientific/traces
```

Expected: `PASS traces: train and validation manifests have disjoint concepts, ids, and seeds`. This command refuses `final_test`.

- [ ] **Step 12: Commit the power record, visible trace builder, and manifests**

```bash
git add configs/scientific/trace-policy.yaml src/ratemem/evaluation/statistics.py src/ratemem/evaluation/traces.py schemas/scientific-calibration-record.schema.json schemas/scientific-trace-manifest.schema.json tests/fixtures/scientific/concept-pools.json tests/unit/evaluation/test_power_planning.py tests/unit/evaluation/test_traces.py tests/contract/evaluation/test_trace_manifest.py
git commit -m "feat(eval): build locked lifecycle traces"
git add configs/scientific/required-units.json
git commit -m "science: freeze trace count from power target"
git add configs/scientific/traces/train-* configs/scientific/traces/validation-*
git commit -m "data: freeze development trace manifests"
```

### Task 6: Keep the final trace encrypted and consume it exactly once

**Files:**
- Modify: `.gitignore`
- Create: `src/ratemem/evaluation/final_trace.py`
- Create: `schemas/scientific-final-trace-envelope.schema.json`
- Create: `schemas/scientific-final-freeze.schema.json`
- Test: `tests/unit/evaluation/test_final_trace.py`
- Test: `tests/contract/evaluation/test_final_trace_access.py`
- Test: `tests/integration/evaluation/test_final_trace_roundtrip.py`

- [ ] **Step 1: Write failing encryption, access, and one-time-ledger tests**

```python
def test_final_trace_round_trip_requires_private_key_and_binds_manifest(tmp_path: Path) -> None:
    private_key, public_key = generate_x25519_keypair()
    envelope = seal_final_trace(b'{"kind":"read"}\n', public_key, associated_manifest=b"manifest")
    with pytest.raises(InvalidTag):
        open_final_trace(envelope, generate_x25519_keypair()[0], associated_manifest=b"manifest")
    assert open_final_trace(envelope, private_key, associated_manifest=b"manifest").read() == b'{"kind":"read"}\n'


def test_training_and_model_selection_cannot_request_final_payload(final_envelope: FinalTraceEnvelope) -> None:
    for purpose in (AccessPurpose.TRAINING, AccessPurpose.MODEL_SELECTION, AccessPurpose.COMPARATIVE_VALIDATION):
        with pytest.raises(FinalTraceAccessDenied):
            acquire_final_trace(final_envelope, purpose=purpose, permit=None, ledger_path=Path("unused"))


def test_open_ledger_is_created_before_decryption_and_blocks_second_attempt(tmp_path: Path, valid_permit: FinalEvaluationPermit) -> None:
    ledger = tmp_path / "final-open-ledger.json"
    with acquire_final_trace(ENVELOPE, AccessPurpose.FINAL_EVALUATION, valid_permit, ledger) as stream:
        assert stream.read(1)
    with pytest.raises(FinalTraceAlreadyOpened):
        acquire_final_trace(ENVELOPE, AccessPurpose.FINAL_EVALUATION, valid_permit, ledger)
    assert json.loads(ledger.read_text())["status"] == "opened"
```

- [ ] **Step 2: Run the final-trace tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_final_trace.py tests/contract/evaluation/test_final_trace_access.py -q`

Expected: collection fails because `ratemem.evaluation.final_trace` does not exist.

- [ ] **Step 3: Implement the versioned public-key envelope**

Use X25519 ephemeral-static key exchange, HKDF-SHA256 with info `b"ratemem-final-trace-v1"`, and ChaCha20-Poly1305. Bind the canonical public manifest bytes as associated data. Define this serialized envelope:

```json
{
  "schema_version": "1.0",
  "algorithm": "X25519-HKDF-SHA256-CHACHA20POLY1305",
  "recipient_public_key_sha256": "64 lowercase hex characters",
  "ephemeral_public_key_base64": "base64 text",
  "nonce_base64": "base64 text",
  "ciphertext_base64": "base64 text",
  "plaintext_sha256": "64 lowercase hex characters",
  "associated_manifest_sha256": "64 lowercase hex characters",
  "event_count": 240
}
```

Implement `generate_x25519_keypair() -> tuple[X25519PrivateKey, X25519PublicKey]`, `seal_final_trace(plaintext: bytes, recipient: X25519PublicKey, associated_manifest: bytes) -> FinalTraceEnvelope`, and `open_final_trace(envelope: FinalTraceEnvelope, private_key: X25519PrivateKey, associated_manifest: bytes) -> BytesIO`. Plaintext final events may exist only in memory or a permission-restricted temporary file removed by the same sealing process; no API accepts a repository plaintext output path.

- [ ] **Step 4: Implement signed freeze permits and atomic one-time opening**

```python
from ratemem.evaluation.types import GitCommit, Sha256


class FinalEvaluationPermit(BaseModel):
    schema_version: Literal["1.0"]
    freeze_id: str
    git_commit: GitCommit
    clean_diff_sha256: Sha256
    dataset_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    method_lock_sha256: Sha256
    method_cpu_gate_sha256: Sha256
    comparative_execution_freeze_sha256: Sha256
    final_envelope_sha256: Sha256
    approved_at_utc: AwareDatetime
    approver_public_key_sha256: Sha256
    ed25519_signature_base64: str
    paid_compute: bool
    scientific_compute_authorization_sha256: Sha256 | None
    scientific_cost_reservation_sha256: Sha256 | None


class AccessPurpose(str, Enum):
    TRAINING = "training"
    MODEL_SELECTION = "model_selection"
    COMPARATIVE_VALIDATION = "comparative_validation"
    FINAL_EVALUATION = "final_evaluation"
```

`acquire_final_trace` verifies every current hash, including the comparative execution freeze and current learned-method CPU gate, and the Ed25519 signature; it opens the ledger with `os.O_CREAT | os.O_EXCL`, writes and `fsync`s status `opened` before decrypting, returns `BytesIO`, and finally overwrites the in-memory plaintext buffer. Any failure after the ledger is created remains a consumed final attempt and records `exit_status` plus the error-class name.

When `paid_compute=true`, permit validation also requires both scientific-compute hashes and Task 9's consumed one-phase launch receipt before any provider invocation. When `paid_compute=false`, both hashes must be absent. An engineering-pilot authorization hash is never accepted in either branch.

- [ ] **Step 5: Add a contract test preventing training imports**

Walk the AST under `src/ratemem/training/` and fail if a module imports `ratemem.evaluation.final_trace`, references `AccessPurpose.FINAL_EVALUATION`, or opens a path containing `final-test-envelope`. Also assert `ratemem-eval traces build-visible --splits final_test` exits 2.

Run: `uv run pytest tests/unit/evaluation/test_final_trace.py tests/contract/evaluation/test_final_trace_access.py tests/integration/evaluation/test_final_trace_roundtrip.py -q`

Expected: all tests pass, including a failed second-open attempt.

- [ ] **Step 6: Add key generation and final sealing commands**

Keep transient decrypted material out of Git by adding `artifacts/scientific/**/*.plaintext` and `artifacts/scientific/**/*.key` to `.gitignore`; final permits and opening ledgers remain publishable evidence. Generate the final-trace key outside the repository without echoing key material:

```bash
umask 077
uv run ratemem-eval traces keygen \
  --private-key /home/ubuntu/.config/ratemem/final-trace-x25519.key \
  --public-key configs/scientific/traces/final-trace-recipient.pem
uv run ratemem-eval traces seal-final \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --policy configs/scientific/trace-policy.yaml \
  --power-record configs/scientific/required-units.json \
  --recipient configs/scientific/traces/final-trace-recipient.pem \
  --manifest-output configs/scientific/traces/final-test-manifest.json \
  --envelope-output configs/scientific/traces/final-test-envelope.json
```

Expected: exit 0 and stdout matches `^PASS final-trace sealed: plaintext retained=false envelope=[0-9a-f]{64}$`. Scan with `git grep -n '"kind":"\(create\|update\|read\|delete\|probe\)"' -- configs/scientific/traces/final-*`; expected no plaintext event match.

- [ ] **Step 7: Commit the unopened final trace before comparative development**

```bash
git add .gitignore src/ratemem/evaluation/final_trace.py schemas/scientific-final-trace-envelope.schema.json schemas/scientific-final-freeze.schema.json tests/unit/evaluation/test_final_trace.py tests/contract/evaluation/test_final_trace_access.py tests/integration/evaluation/test_final_trace_roundtrip.py configs/scientific/traces/final-trace-recipient.pem configs/scientific/traces/final-test-manifest.json configs/scientific/traces/final-test-envelope.json
git commit -m "data: freeze unopened final lifecycle trace"
```

### Task 7: Freeze evaluator revisions, formulas, margins, budgets, workloads, and power

**Files:**
- Create: `configs/scientific/evaluation-policy.yaml`
- Create: `src/ratemem/evaluation/evaluation_lock.py`
- Create: `schemas/evaluation-lock.schema.json`
- Test: `tests/unit/evaluation/test_evaluation_lock.py`
- Test: `tests/contract/evaluation/test_evaluation_lock_schema.py`

- [ ] **Step 1: Add the prespecified claim and metric policy**

```yaml
# configs/scientific/evaluation-policy.yaml
schema_version: "1.0"
alpha: 0.05
confidence_level: 0.95
minimum_training_seeds: 3
bootstrap_resamples: 10000
bootstrap_seed: 271828
multiplicity_method: holm
ci_width_target:
  metric_id: request_weighted_identity
  maximum_half_width: 0.02
power_target:
  power: 0.80
  two_sided_alpha: 0.05
  minimum_detectable_effect: 0.03
primary_grid:
  budget_fractions: [0.25, 0.50, 0.75]
  required_pass_fraction: 0.50
  request_regimes: [uniform, zipf]
metric_formulas:
  identity: identity_mean_v1
  prompt: prompt_mean_v1
  request_weighted_identity: request_weighted_identity_v1
  request_weighted_utility: equal_weight_identity_prompt_v1
  retention_auc: normalized_event_trapezoid_v1
  active_state_drift: acquisition_delta_v1
  maximum_active_degradation: maximum_acquisition_drop_v1
  oracle_regret: future_oracle_utility_gap_v1
  lookup_aurc: lookup_risk_coverage_v1
  diversity: thresholded_conditional_diversity_v1
claims:
  shared_packet_representation:
    primary_endpoint: request_weighted_identity
    inference_unit: deployment_episode
    required_controls: [private_progressive_size_aware, private_progressive_separable_rate, cts_style_static, vb_lora_style_static, share_style_online, dreamcache_feature_cache]
    constraint_metric: prompt
    pass_rule: positive_paired_ci_and_prompt_noninferiority
  causal_packet_allocator:
    primary_endpoint: request_weighted_utility
    inference_unit: deployment_episode
    required_controls: [independent_lrua, private_progressive_size_aware, private_progressive_separable_rate, shared_packet_plain_greedy]
    constraint_metric: average_active_quality
    pass_rule: positive_paired_ci_lower_regret_and_quality_noninferiority
  allocator_guarantee:
    primary_endpoint: certified_reduced_set_approximation_ratio
    inference_unit: allocator_instance
    required_controls: [exact_reduced_set_optimum]
    ground_set_scope: causal_singleton_density_prescreen_C_t_max24
    allocator_boundary:
      fixture_id: four_concepts_eight_packets_each_v1
      proposal_count: 32
      prescreen_input_count: 32
      allocator_input_count: 24
      deterministic_tie_break: lexicographically_larger_packet_id_wins
    pass_rule: proof_and_exhaustive_reduced_set_instance_certificate
  optimization_free_tradeoff:
    primary_endpoint: identity
    inference_unit: concept
    required_controls: [per_concept_lora, dreamcache_feature_cache]
    pass_rule: quality_noninferiority_and_insertion_latency_advantage
  autonomous_lookup:
    primary_endpoint: lookup_aurc
    inference_unit: concept_conditioned_lookup_episode
    required_controls: [nearest_key_threshold, learned_novelty]
    pass_rule: lower_aurc
prelock_baseline_evidence:
  require_source_and_license_inventory: true
  require_real_checkpoint_fidelity: true
  require_state_ledger_roundtrip: true
  require_provider_neutral_shared_input_schema: true
  require_synthetic_provider_contract: true
  require_frozen_search_policy: true
  forbid_learned_shared_input_bundle: true
  forbid_search_outcomes: true
postlock_execution_freeze:
  required_before: [strongest_control_selection, final_trace_open]
  required_receipts: [ratemem_shared_input_bundle, method_train_receipts, method_search_ledgers, search_budget_compliance]
  require_scientific_compute_reconciliation_for_paid_receipts: true
ablations:
  design: one_factor_at_a_time_from_frozen_primary
  representation_fixed_allocator: guarantee_bearing_allocator
  allocator_fixed_candidate_stream: true
  factors:
    packet_group_size: [8, 16, 32]
    base_code_bits_per_coefficient: [2, 4, 8]
    packet_precision_bits: [2, 4, 8]
    sharing_rule: [exact_payload_only]
    maximum_incidences_per_concept: [1, 2, 4, 8]
    packet_ownership: [private_progressive, shared]
    sharing_enabled: [false, true]
    allocator: [density_greedy, guarantee_bearing]
    hysteresis: [disabled, locked_switch_cost]
    budget_label: [25pct, 50pct, 75pct]
    support_shots: [1, 3, 5]
    request_pattern: [uniform, zipf, locked_drift]
    update_delete_rate_multiplier: [0.5, 1.0, 2.0]
    distortion: [code_mse, one_step_diffusion]
```

- [ ] **Step 2: Write failing freeze tests**

```python
def test_budget_bytes_are_derived_from_locked_independent_cache_ledger() -> None:
    budgets = derive_budget_cells(reference_total_bytes=20_480, active_set_size=20, fractions=(0.25, 0.5, 0.75), ledger_sha256="a" * 64)
    assert [(cell.label, cell.bytes) for cell in budgets] == [("25pct", 5_120), ("50pct", 10_240), ("75pct", 15_360)]


def test_freeze_rejects_mutable_evaluator_revision_and_uncalibrated_margin(valid_draft: EvaluationLockDraft) -> None:
    valid_draft.evaluators[0].revision = "main"
    valid_draft.margins[0].calibration_artifact_sha256 = None
    with pytest.raises(EvaluationLockError):
        freeze_evaluation_lock(valid_draft)


def test_training_identity_representation_cannot_be_sole_headline_evaluator(valid_draft: EvaluationLockDraft) -> None:
    valid_draft.evaluators = [evaluator("training_identity_encoder", roles=["training_loss", "headline_identity"])]
    with pytest.raises(EvaluationLockError, match="independent headline identity evaluator"):
        freeze_evaluation_lock(valid_draft)


@pytest.mark.parametrize("scope", [None, "full_G_t"])
def test_allocator_guarantee_lock_requires_reduced_ground_set_scope(
    valid_draft: EvaluationLockDraft, scope: str | None
) -> None:
    claim = valid_draft.claims["allocator_guarantee"]
    valid_draft.claims["allocator_guarantee"] = claim.model_copy(
        update={"ground_set_scope": scope}
    )
    with pytest.raises(EvaluationLockError, match="allocator guarantee ground-set scope"):
        freeze_evaluation_lock(valid_draft)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proposal_count", 31),
        ("prescreen_input_count", 31),
        ("allocator_input_count", 25),
        (
            "deterministic_tie_break",
            "lexicographically_smaller_packet_id_wins",
        ),
    ],
)
def test_allocator_guarantee_lock_requires_exact_controller_boundary(
    valid_draft: EvaluationLockDraft, field: str, value: object
) -> None:
    claim = valid_draft.claims["allocator_guarantee"]
    assert claim.allocator_boundary is not None
    changed_boundary = claim.allocator_boundary.model_copy(update={field: value})
    valid_draft.claims["allocator_guarantee"] = claim.model_copy(
        update={"allocator_boundary": changed_boundary}
    )

    with pytest.raises(EvaluationLockError, match="allocator boundary"):
        freeze_evaluation_lock(valid_draft)


def test_allocator_guarantee_lock_rejects_missing_controller_boundary(
    valid_draft: EvaluationLockDraft,
) -> None:
    claim = valid_draft.claims["allocator_guarantee"]
    valid_draft.claims["allocator_guarantee"] = claim.model_copy(
        update={"allocator_boundary": None}
    )

    with pytest.raises(EvaluationLockError, match="allocator boundary"):
        freeze_evaluation_lock(valid_draft)
```

- [ ] **Step 3: Run the evaluation-lock tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_evaluation_lock.py -q`

Expected: collection fails because `ratemem.evaluation.evaluation_lock` does not exist.

- [ ] **Step 4: Implement strict lock models and budget derivation**

```python
from ratemem.evaluation.types import Sha256


class AllocatorBoundaryLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fixture_id: Literal["four_concepts_eight_packets_each_v1"]
    proposal_count: Literal[32]
    prescreen_input_count: Literal[32]
    allocator_input_count: Literal[24]
    deterministic_tie_break: Literal[
        "lexicographically_larger_packet_id_wins"
    ]


class ClaimLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    primary_endpoint: str
    inference_unit: str
    required_controls: list[str]
    constraint_metric: str | None = None
    pass_rule: str
    ground_set_scope: Literal[
        "causal_singleton_density_prescreen_C_t_max24"
    ] | None = None
    allocator_boundary: AllocatorBoundaryLock | None = None


class EvaluatorLock(BaseModel):
    evaluator_id: str
    repository: AnyUrl
    revision: str
    weights_sha256: Sha256
    preprocessing_id: str
    preprocessing_sha256: Sha256
    roles: set[Literal["training_loss", "filtering", "headline_identity", "headline_prompt", "diversity"]]


class MarginLock(BaseModel):
    claim_id: str
    metric_id: str
    value: NonNegativeFloat
    direction: Literal["higher", "lower"]
    source_kind: Literal["published_reliability", "separate_calibration"]
    source_reference: str
    calibration_pool_sha256: Sha256
    calibration_artifact_sha256: Sha256


class BudgetCell(BaseModel):
    label: Literal["25pct", "50pct", "75pct"]
    fraction: Literal[0.25, 0.5, 0.75]
    bytes: PositiveInt
    active_set_size: PositiveInt
    independent_cache_ledger_sha256: Sha256


class EvaluationLock(BaseModel):
    schema_version: Literal["1.0"]
    lock_id: Sha256
    sealed_at_utc: AwareDatetime
    dataset_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    baseline_audit_receipt_sha256: Sha256
    comparator_catalog_sha256: Sha256
    shared_input_schema_sha256: Sha256
    synthetic_provider_report_sha256: Sha256
    search_policy_sha256: Sha256
    trace_manifest_sha256: dict[str, Sha256]
    evaluators: list[EvaluatorLock]
    metric_formulas: dict[str, str]
    margins: list[MarginLock]
    budget_cells: list[BudgetCell]
    workload_distributions: list[WorkloadLock]
    generation: GenerationLock
    latency: LatencyLock
    human_study: HumanStudyLock
    claims: dict[str, ClaimLock]
    bootstrap: BootstrapLock
    power: PowerLock
    approvals: list[Approval]
```

`derive_budget_cells` uses decimal half-up rounding and stores the exact source byte-ledger hash.
`freeze_evaluation_lock` verifies immutable 40-hex or content-addressed revisions,
evaluator/preprocessing weight hashes, fixed prompt/noise pairing,
hardware/warm-up/batch/resolution/sampler/steps, both request regimes, all three budgets, at least
three training seeds, margin provenance excluded from model selection, the
strongest-eligible-control selector, and Task 8's pre-lock source/fidelity/protocol/search-policy
receipt. It additionally requires the `allocator_guarantee` claim to name
`exact_reduced_set_optimum` and the exact non-null
`ground_set_scope=causal_singleton_density_prescreen_C_t_max24`; a missing field or any full-`G_t`
scope is rejected. The same claim must carry the exact `AllocatorBoundaryLock` above, freezing the
learned-controller regression at `proposal_count=32`, `prescreen_input_count=32`, and
`allocator_input_count=24`, with deterministic exact-density ties specified as
`lexicographically_larger_packet_id_wins` (lexicographically larger packet ID wins). Other claims
must leave `allocator_boundary` null. It rejects any learned shared-input materialization or
validation search outcome at this stage and instead binds the required post-lock receipt types.

Add `require_scientific_training_lock(dataset_lock: Path, evaluation_lock: Path, requested_split: str) -> None` and call it from the scientific training entry point. It accepts only `train`, verifies both lock schemas/hashes, and rejects a launch if the evaluation lock is absent, unsigned, or newer inputs have invalidated it.

- [ ] **Step 5: Implement calibration-to-lock compilation and verify it blocks before the baseline lock**

Expose `ratemem-eval lock evaluation` with explicit inputs for evaluator inventory, independent-cache byte ledger, margin calibration, CI-width/power record, dataset lock, baseline lock, its audit receipt, and all three trace commitments. The command computes exact byte budgets, copies rather than recomputes margins, and requires two approval records created before comparative validation.

```bash
uv run ratemem-eval lock evaluation \
  --policy configs/scientific/evaluation-policy.yaml \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --baseline-audit-receipt artifacts/scientific/baselines/audit-receipt.json \
  --trace-dir configs/scientific/traces \
  --evaluator-inventory artifacts/scientific/calibration/evaluators.json \
  --byte-ledger artifacts/scientific/calibration/independent-cache-ledger.json \
  --margin-record artifacts/scientific/calibration/margins.json \
  --power-record configs/scientific/required-units.json \
  --approvals artifacts/scientific/calibration/evaluation-approvals.json \
  --output configs/scientific/evaluation-lock.yaml
```

Expected at this task boundary: exit 2 with `BLOCKED evaluation-lock: baseline lock is missing` and no output. Task 8 runs the same command after baseline audit and approvals; then stdout must match `^PASS evaluation-lock: [0-9a-f]{64}$`.

- [ ] **Step 6: Generate the schema and test semantic immutability**

Run: `uv run ratemem-eval lock schema --kind evaluation --output schemas/evaluation-lock.schema.json && uv run pytest tests/unit/evaluation/test_evaluation_lock.py tests/contract/evaluation/test_evaluation_lock_schema.py -q`

Expected: all tests pass. Add a test that changing a margin, evaluator preprocessing, trace hash, budget byte, request exponent, or generation seed changes `lock_id`.

- [ ] **Step 7: Commit the evaluation-lock policy and code**

```bash
git add configs/scientific/evaluation-policy.yaml src/ratemem/evaluation/evaluation_lock.py schemas/evaluation-lock.schema.json tests/unit/evaluation/test_evaluation_lock.py tests/contract/evaluation/test_evaluation_lock_schema.py
git commit -m "feat(eval): freeze scientific evaluation protocol"
```

### Task 8: Register the comparator contract, authorize pre-lock fidelity narrowly, and seal the companion audit

**Files:**
- Create: `configs/scientific/baseline-requirements.yaml`
- Create: `configs/scientific/baseline-fidelity-compute-policy.yaml`
- Create: `src/ratemem/evaluation/baselines.py`
- Create: `src/ratemem/evaluation/compute.py`
- Create: `schemas/scientific-baseline-lock.schema.json`
- Create: `schemas/scientific-baseline-fidelity-authorization.schema.json`
- Test: `tests/unit/evaluation/test_baselines.py`
- Test: `tests/unit/evaluation/test_baseline_fidelity_authorization.py`
- Test: `tests/contract/evaluation/test_baseline_coverage.py`
- Test: `tests/contract/evaluation/test_baseline_fidelity_boundary.py`

- [ ] **Step 1: Encode the exact matched-control and pre-lock evidence requirements**

Create the following registry. The matched-baselines companion plan owns every concrete adapter and source-specific fidelity runner; this task owns only the required set, the consumer-facing protocol contract, and fail-closed sealing.

```yaml
# configs/scientific/baseline-requirements.yaml
schema_version: "1.0"
catalog: configs/baselines/literature-classification.yaml
runnable_registry:
  - independent_fifo
  - independent_lru
  - independent_lrua
  - private_progressive_size_aware
  - private_progressive_separable_rate
  - shared_packet_plain_greedy
  - cts_style_static
  - vb_lora_style_static
  - share_style_online
  - dreamcache_feature_cache
  - hyperlora_upstream
  - stateless_amortizer
  - per_concept_lora
  - exact_append_only_quantized
  - exact_future_trace_packets
postlock_execution_required:
  - independent_fifo
  - independent_lru
  - independent_lrua
  - private_progressive_size_aware
  - private_progressive_separable_rate
  - shared_packet_plain_greedy
  - cts_style_static
  - vb_lora_style_static
  - share_style_online
  - dreamcache_feature_cache
  - hyperlora_upstream
  - stateless_amortizer
  - per_concept_lora
  - exact_append_only_quantized
  - exact_future_trace_packets
sana_primary_required:
  - independent_fifo
  - independent_lru
  - independent_lrua
  - private_progressive_size_aware
  - private_progressive_separable_rate
  - shared_packet_plain_greedy
  - cts_style_static
  - vb_lora_style_static
  - share_style_online
  - dreamcache_feature_cache
  - per_concept_lora
primary_roles:
  representation: [independent_fifo, independent_lru, private_progressive_size_aware, private_progressive_separable_rate, cts_style_static, vb_lora_style_static, share_style_online, dreamcache_feature_cache]
  allocator: [independent_fifo, independent_lru, independent_lrua, private_progressive_size_aware, private_progressive_separable_rate, shared_packet_plain_greedy, share_style_online]
  optimization_free_tradeoff: [per_concept_lora, dreamcache_feature_cache]
upper_reference_only: [exact_append_only_quantized, exact_future_trace_packets]
secondary_only: [hyperlora_upstream, stateless_amortizer]
contextual_literature_citation_keys: [sinelora_delta_aaai2026]
backbone_resolution:
  sole_primary: sana_1_5_1_6b
  allow_primary_backbone_fallback: false
  sdxl_native_evidence: contextual_only
  block_claim_if_required_sana_fidelity_fails: true
prelock_shared_input:
  permitted_bundle_kind: synthetic_protocol
  require_provider_neutral_schema: true
  require_exact_ledger_roundtrip: true
  forbid_learned_ratemem_dictionary: true
  forbid_validation_metrics: true
  forbid_final_trace: true
search_policy:
  maximum_trials_per_method: 24
  maximum_gpu_hours_per_method: 48.0
  split: validation
  prelock_mode: frozen_policy_only
  require_postlock_receipts_before_selection: true
literature_disposition:
  require_every_catalog_entry_classified: true
  contextual_or_incompatible_never_blocks_primary_lock: true
```

This list is exactly the `controls` set in `configs/baselines/literature-classification.yaml`. Closest works that cannot support a matched lifecycle claim remain visibly classified there; they are never silently promoted, omitted, or represented by a stub adapter.

- [ ] **Step 2: Write failing registry, protocol-identity, audit, and pre-lock-boundary tests**

```python
def test_required_ids_equal_the_companion_catalog_controls(requirements: Requirements, catalog: BaselineCatalog) -> None:
    assert len(requirements.runnable_registry) == 15
    assert set(requirements.runnable_registry) == {control.id for control in catalog.controls}


def test_sinelora_delta_is_contextual_sd3_literature_not_a_runnable_control(
    requirements: Requirements, catalog: BaselineCatalog,
) -> None:
    record = next(row for row in catalog.literature if row.citation_key == "sinelora_delta_aaai2026")
    assert record.comparison_class == "contextual_only"
    assert record.port_mode == "citation_only_sd3_medium"
    assert requirements.contextual_literature_citation_keys == ["sinelora_delta_aaai2026"]
    assert "sine_lora_delta_sdxl" not in requirements.runnable_registry
    assert all(control.id != "sine_lora_delta_sdxl" for control in catalog.controls)


def test_evaluation_reexports_the_one_canonical_protocol() -> None:
    from ratemem.baselines.protocol import BaselineAdapter as CanonicalBaselineAdapter
    from ratemem.evaluation.baselines import BaselineAdapter

    assert BaselineAdapter is CanonicalBaselineAdapter


def test_prelock_audit_rejects_real_dictionary_or_validation_outcomes(valid_audit_inputs: AuditInputs) -> None:
    valid_audit_inputs.synthetic_provider_report.bundle_kind = "ratemem_learned_dictionary"
    valid_audit_inputs.synthetic_provider_report.outcome_rows = [{"request_weighted_identity": 0.7}]
    with pytest.raises(BaselineLockError, match="pre-lock evidence boundary"):
        freeze_baseline_lock(valid_audit_inputs)


def test_baseline_lock_requires_source_fidelity_compliance_and_frozen_search_policy(valid_audit_inputs: AuditInputs) -> None:
    valid_audit_inputs.fidelity_reports.pop("dreamcache_feature_cache")
    with pytest.raises(BaselineLockError, match="fidelity receipt"):
        freeze_baseline_lock(valid_audit_inputs)


def test_sdxl_fidelity_cannot_replace_a_required_sana_comparator(valid_audit_inputs: AuditInputs) -> None:
    valid_audit_inputs.fidelity_reports["share_style_online"] = faithful_report(backbone="sdxl_1_0")
    with pytest.raises(BaselineLockError, match="required SANA fidelity"):
        freeze_baseline_lock(valid_audit_inputs)


def test_strongest_control_selector_uses_only_postlock_validation_rows() -> None:
    rows = validation_rows(
        ("private_progressive_size_aware", 0.61, 0.72),
        ("share_style_online", 0.64, 0.71),
        ("dreamcache_feature_cache", 0.66, 0.60),
    )
    selected = select_strongest_eligible_control(rows, endpoint="identity", constraint="prompt", minimum_constraint=0.70)
    assert selected == "share_style_online"
```

Add authorization-boundary cases that reject `split="validation"`, a final-trace path or hash, a model-selection/output-metric field, an unpinned source revision, a dirty-diff mismatch, a second workspace candidate, pilot scope, cap other than USD 28.00, and `known + pending + new > 27.00`.

- [ ] **Step 3: Run the focused tests and verify both missing-module failures**

Run:

```bash
uv run pytest tests/unit/evaluation/test_baselines.py tests/unit/evaluation/test_baseline_fidelity_authorization.py tests/contract/evaluation/test_baseline_coverage.py tests/contract/evaluation/test_baseline_fidelity_boundary.py -q
```

Expected: collection fails because `ratemem.evaluation.baselines` and `ratemem.evaluation.compute` do not exist.

- [ ] **Step 4: Implement only the scientific registry consumer and canonical protocol handoff**

Use the shared aliases and records below; do not implement any concrete baseline adapter in this plan.

```python
from ratemem.baselines.protocol import (
    BaselineAdapter as BaselineAdapter,
    CausalEventView as CausalEventView,
    EventReceipt as EventReceipt,
    ExactByteLedger as ExactByteLedger,
    FrozenComparisonContract as FrozenComparisonContract,
    MethodSnapshot as MethodSnapshot,
    ProbeResult as ProbeResult,
)
from ratemem.evaluation.types import Sha256


class BaselineLockEntry(BaseModel):
    method_id: str
    source_inventory_record_sha256: Sha256
    fidelity_report_sha256: Sha256
    compliance_report_sha256: Sha256
    state_ledger_test_sha256: Sha256
    supported_backbones: set[Literal["sana_1_5_1_6b", "sdxl_1_0"]]
    sana_primary_eligible: bool
    disposition: Literal["faithful", "incompatible", "secondary_only", "upper_reference_only"]


class BaselineAuditReceipt(BaseModel):
    schema_version: Literal["1.0"]
    status: Literal["pass"]
    requirements_sha256: Sha256
    catalog_sha256: Sha256
    backbone_plan_sha256: Sha256
    source_inventory_sha256: Sha256
    shared_input_schema_sha256: Sha256
    synthetic_provider_report_sha256: Sha256
    search_policy_sha256: Sha256
    method_entries: list[BaselineLockEntry]
    receipt_sha256: Sha256
```

The canonical types live only in `src/ratemem/baselines/protocol.py`, created by `2026-08-24-ratemem-matched-baselines.md`; `src/ratemem/evaluation/baselines.py` re-exports those exact objects and declares them in `__all__`. `BaselineAdapter` has attributes `method_id: str` and `role: Literal["causal", "upper_reference", "latency_control"]`, plus exact methods `initialize(contract: FrozenComparisonContract) -> None`, `apply_event(event: LifecycleEvent, view: CausalEventView) -> EventReceipt`, `copy_snapshot() -> MethodSnapshot`, `score_probe(snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult`, `export_online_state() -> bytes`, `import_online_state(payload: bytes) -> None`, `state_ledger() -> ExactByteLedger`, and `close() -> None`. Do not redefine any of these Pydantic models or the protocol under `ratemem.evaluation`; the companion plan alone supplies concrete independent/progressive/shared/external/feature-cache/LoRA/oracle adapters.

Implement `load_requirements(path: Path) -> Requirements`, `validate_prelock_handoff(inputs: AuditInputs) -> BaselineAuditReceipt`, `freeze_baseline_lock(inputs: AuditInputs, output: Path) -> BaselineLock`, and `select_strongest_eligible_control(rows: DataFrame, endpoint: str, constraint: str, minimum_constraint: float) -> str`. The pre-lock validator accepts only the provider-neutral shared-input JSON Schema, a schema-validated synthetic provider-contract report, and a frozen search policy with budgets but no execution ledger or outcome fields.

- [ ] **Step 5: Implement the one-workspace `baseline_fidelity` authorization before executing the companion plan**

Create the narrow policy:

```yaml
# configs/scientific/baseline-fidelity-compute-policy.yaml
schema_version: "1.0"
authorization_scope: baseline_fidelity
provider: modal
allowed_input_roles: [held_in, dedicated_calibration]
forbidden_input_roles: [validation, final_test]
forbid_model_selection: true
forbid_claim_quality_metrics: true
forbid_learned_ratemem_dictionary: true
require_method_cpu_gate: false
require_dataset_lock: true
require_baseline_requirements: true
require_comparator_catalog: true
require_fidelity_policy: true
require_source_inventory: true
require_clean_commit_and_diff: true
workspace_selection: explicit_operator_file
profile_prefix: ratemem-scientific-
automatic_workspace_discovery: false
automatic_workspace_reuse: false
automatic_workspace_rotation: false
automatic_workspace_fallback: false
outer_workspace_usage_budget_usd: "28.00"
internal_reservation_limit_usd: "27.00"
aggregate_ledger: /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl
reservation_formula: known_usage_plus_all_pending_worst_case_plus_new_phase_bound
one_phase_per_authorization: true
one_launch_per_reservation: true
require_reconciliation_before_next_phase: true
```

Implement this exact record in `src/ratemem/evaluation/compute.py`:

```python
from ratemem.evaluation.types import GitCommit, PhaseId, ScientificProfile, Sha256


class BaselineFidelityAuthorization(BaseModel):
    schema_version: Literal["1.0"]
    scope: Literal["baseline_fidelity"]
    phase_id: PhaseId
    workspace_id: str
    explicit_profile: ScientificProfile
    dataset_lock_sha256: Sha256
    baseline_requirements_sha256: Sha256
    comparator_catalog_sha256: Sha256
    fidelity_policy_sha256: Sha256
    source_inventory_sha256: Sha256
    source_revision: GitCommit
    source_archive_sha256: Sha256
    git_commit: GitCommit
    clean_diff_sha256: Sha256
    input_role: Literal["held_in", "dedicated_calibration"]
    input_manifest_sha256: Sha256
    job_spec_sha256: Sha256
    workspace_snapshot_sha256: Sha256
    issued_at_utc: AwareDatetime
    expires_at_utc: AwareDatetime
```

Implement `authorize_baseline_fidelity(selection: WorkspaceSelection, snapshot: WorkspaceSnapshot, phase: BaselineFidelityPhaseRequest, bindings: BaselineFidelityBindings, policy: BaselineFidelityPolicy) -> BaselineFidelityAuthorization`, `require_baseline_fidelity_permit(authorization_path: Path, reservation_path: Path, expected_phase_id: str, expected_workspace_id: str, launch_receipt_path: Path) -> ConsumedPermit`, and the workspace snapshot, append-only reservation, one-shot consumption, and reconciliation primitives shared later by Task 9. Authorization recursively rejects validation/final paths or hashes, final-envelope references, selection fields, scientific endpoint names, unpinned commits, mismatched diff/source hashes, and engineering-pilot records. The shared ledger admission is atomic and includes every pending `baseline_fidelity` or `scientific` reservation on the selected workspace.

Expose and test this command for each companion fidelity job specification:

```bash
uv run ratemem-eval compute attest-workspace \
  --policy configs/scientific/baseline-fidelity-compute-policy.yaml \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
  --output /home/ubuntu/.config/ratemem/scientific-workspace-snapshot.json
uv run ratemem-eval compute authorize-baseline-fidelity \
  --policy configs/scientific/baseline-fidelity-compute-policy.yaml \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --workspace-snapshot /home/ubuntu/.config/ratemem/scientific-workspace-snapshot.json \
  --phase-request artifacts/scientific/baselines/fidelity/phase-request.json \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --requirements configs/scientific/baseline-requirements.yaml \
  --catalog configs/baselines/literature-classification.yaml \
  --fidelity-policy configs/baselines/fidelity-policy.yaml \
  --source-inventory artifacts/scientific/baselines/source-inventory.json \
  --clean-diff-receipt artifacts/scientific/baselines/fidelity/clean-diff-receipt.json \
  --output artifacts/scientific/baselines/fidelity/authorization.json
uv run ratemem-eval compute reserve-baseline-fidelity \
  --authorization artifacts/scientific/baselines/fidelity/authorization.json \
  --phase-bound artifacts/scientific/baselines/fidelity/phase-cost-bound.json \
  --workspace-snapshot /home/ubuntu/.config/ratemem/scientific-workspace-snapshot.json \
  --ledger /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl \
  --output artifacts/scientific/baselines/fidelity/reservation.json
uv run ratemem-eval compute reconcile-baseline-fidelity \
  --authorization artifacts/scientific/baselines/fidelity/authorization.json \
  --reservation artifacts/scientific/baselines/fidelity/reservation.json \
  --launch-receipt artifacts/scientific/baselines/fidelity/launch-receipt.json \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
  --ledger /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl \
  --output artifacts/scientific/baselines/fidelity/reconciliation.json
```

Expected attestation matches `^PASS scientific-workspace attestation: workspace=[a-zA-Z0-9_-]+ outer_cap=28.00 known_usage=[0-9]+\.[0-9]{2}$`. Authorization success matches `^PASS baseline-fidelity authorization: phase=[a-z0-9_-]+ workspace=[a-zA-Z0-9_-]+ authorization=[0-9a-f]{64}$`; it does not reserve cost and the authorization contains no reservation hash. The reservation command hashes that completed authorization, atomically appends a record referencing `authorization_sha256` to the shared ledger, writes `reservation.json`, and prints `^PASS baseline-fidelity reservation: known=[0-9]+\.[0-9]{2} pending=[0-9]+\.[0-9]{2} new=[0-9]+\.[0-9]{2} total=[0-9]+\.[0-9]{2} <= 27.00$`. `require_baseline_fidelity_permit` consumes both files once and writes the shown `launch-receipt.json`; after the companion job, reconciliation verifies the same workspace/cap and prints `^PASS baseline-fidelity reconciliation: workspace=[a-zA-Z0-9_-]+ metered_delta=[0-9]+\.[0-9]{2} pending_remaining=[0-9]+\.[0-9]{2}$`. Add tests proving the authorization/reservation digests are acyclic, the reservation hash changes if authorization bytes change, and second consumption fails before provider invocation. Any missing hash, forbidden input, workspace mismatch, cap mismatch, pending-cost omission, or nonzero provider invocation before permit consumption exits 2 with `BLOCKED baseline-fidelity: <stable_reason_code>` and creates no output for that transition. Run the focused tests, generate `schemas/scientific-baseline-fidelity-authorization.schema.json`, then commit only the registry/protocol consumer and authorization foundation.

At this point pause this task and execute every task in `docs/superpowers/plans/2026-08-24-ratemem-matched-baselines.md`. Do not locally substitute adapter classes or weaken a failed companion fidelity case.

- [ ] **Step 6: Validate the matched-baselines handoff without admitting validation outcomes**

Require these exact companion inputs:

- `artifacts/scientific/baselines/source-inventory.json`
- `artifacts/scientific/baselines/fidelity/<method_id>/sana_1_5_1_6b/report.json`
- `artifacts/scientific/baselines/compliance/<method_id>/sana_1_5_1_6b/report.json`
- `configs/baselines/policy-search.yaml`
- `schemas/ratemem-shared-input-bundle-v1.schema.json`
- `artifacts/scientific/baselines/compliance/synthetic-provider-contract.json`

Before audit, assert that the shared-input input is a schema rather than a learned bundle, the synthetic report proves only protocol/byte roundtrip, the search input contains budgets and search-space rules but no run outcomes, every paid fidelity report binds a consumed/reconciled `baseline_fidelity` permit, and no artifact contains a validation or final-trace hash. Then run the companion-owned command unchanged:

```bash
uv run ratemem-baselines audit freeze \
  --requirements configs/scientific/baseline-requirements.yaml \
  --catalog configs/baselines/literature-classification.yaml \
  --backbones configs/baselines/backbones.yaml \
  --source-inventory artifacts/scientific/baselines/source-inventory.json \
  --fidelity-dir artifacts/scientific/baselines/fidelity \
  --compliance-dir artifacts/scientific/baselines/compliance \
  --search-policy configs/baselines/policy-search.yaml \
  --shared-input-schema schemas/ratemem-shared-input-bundle-v1.schema.json \
  --synthetic-provider-report artifacts/scientific/baselines/compliance/synthetic-provider-contract.json \
  --output-plan artifacts/scientific/baselines/backbone-plan.json \
  --output-receipt artifacts/scientific/baselines/audit-receipt.json \
  --output-lock configs/scientific/baseline-lock.yaml
```

Expected stdout matches `^PASS baseline-lock: backbone=sana_1_5_1_6b lock=[0-9a-f]{64} receipt=[0-9a-f]{64}$`. There is no automatic primary-backbone fallback: any required SANA fidelity miss exits 2 with a line matching `^BLOCKED baseline-lock: required_sana_control_unfaithful:[a-z0-9_.-]+$`, where the suffix is the sorted dot-joined failed IDs; other missing, stub, unlicensed, policy-over-budget, future-access, outcome-bearing, or hash-mismatched inputs emit their stable reason code. Every blocked audit creates none of the plan, receipt, or lock. Optional SDXL receipts live only under `artifacts/scientific/baselines/contextual/sdxl_1_0/` and cannot satisfy requirements; SineLoRA-Delta remains citation-only SD3 Medium literature and has no adapter or fidelity receipt.

- [ ] **Step 7: Verify the emitted plan/receipt/lock and make sealing fail closed**

Require these exact outputs:

- `artifacts/scientific/baselines/backbone-plan.json`
- `artifacts/scientific/baselines/audit-receipt.json`
- `configs/scientific/baseline-lock.yaml`

Generate `schemas/scientific-baseline-lock.schema.json`, run the four Task 8 tests plus the companion plan's fidelity/ledger/audit tests, and recompute every referenced SHA-256. `freeze_baseline_lock` must reject a missing fidelity or compliance report, absent source/license record, failed state roundtrip, search-execution or outcome evidence before lock, a learned shared-input bundle, missing paid-fidelity reconciliation, or mismatch between the backbone plan and every primary comparator. Only the companion audit command may create the baseline lock.

```bash
uv run ratemem-eval baselines verify-handoff \
  --requirements configs/scientific/baseline-requirements.yaml \
  --backbone-plan artifacts/scientific/baselines/backbone-plan.json \
  --audit-receipt artifacts/scientific/baselines/audit-receipt.json \
  --baseline-lock configs/scientific/baseline-lock.yaml
uv run pytest tests/unit/evaluation/test_baselines.py tests/unit/evaluation/test_baseline_fidelity_authorization.py tests/contract/evaluation/test_baseline_coverage.py tests/contract/evaluation/test_baseline_fidelity_boundary.py tests/baselines tests/contract/baselines -q
```

Expected: first stdout matches `^PASS baseline handoff: lock=[0-9a-f]{64} audit=[0-9a-f]{64}$`; all tests pass. Commit the registry and verified companion handoff in separate commits so policy review precedes immutable evidence.

- [ ] **Step 8: Seal the evaluation lock only from the verified baseline receipt**

```bash
uv run ratemem-eval lock evaluation \
  --policy configs/scientific/evaluation-policy.yaml \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --baseline-audit-receipt artifacts/scientific/baselines/audit-receipt.json \
  --trace-dir configs/scientific/traces \
  --evaluator-inventory artifacts/scientific/calibration/evaluators.json \
  --byte-ledger artifacts/scientific/calibration/independent-cache-ledger.json \
  --margin-record artifacts/scientific/calibration/margins.json \
  --power-record configs/scientific/required-units.json \
  --approvals artifacts/scientific/calibration/evaluation-approvals.json \
  --output configs/scientific/evaluation-lock.yaml
```

Expected: `^PASS evaluation-lock: [0-9a-f]{64}$`. A missing/failed companion receipt, nonidentical baseline-lock hash, mutable revision, pre-lock validation outcome, or real learned shared-input hash exits 2 and leaves the evaluation lock absent. The evaluation lock binds the comparator catalog, pre-lock audit, synthetic protocol receipt, and search-policy budget, but deliberately does not bind later validation results. Actual RateMem shared inputs and baseline train/tune/search receipts are frozen in Task 10 before replay/selection/final opening.

```bash
git add configs/scientific/baseline-requirements.yaml configs/scientific/baseline-fidelity-compute-policy.yaml src/ratemem/evaluation/baselines.py src/ratemem/evaluation/compute.py schemas/scientific-baseline-lock.schema.json schemas/scientific-baseline-fidelity-authorization.schema.json tests/unit/evaluation/test_baselines.py tests/unit/evaluation/test_baseline_fidelity_authorization.py tests/contract/evaluation/test_baseline_coverage.py tests/contract/evaluation/test_baseline_fidelity_boundary.py
git commit -m "feat(eval): define baseline registry and fidelity boundary"
git add artifacts/scientific/baselines/backbone-plan.json artifacts/scientific/baselines/audit-receipt.json configs/scientific/baseline-lock.yaml
git commit -m "science: freeze matched baseline fidelity audit"
git add configs/scientific/evaluation-lock.yaml
git commit -m "science: sign off evaluation lock"
```

### Task 9: Authorize each paid scientific phase independently

**Files:**
- Create: `configs/scientific/compute-policy.yaml`
- Modify: `src/ratemem/evaluation/compute.py`
- Create: `schemas/scientific-compute-authorization.schema.json`
- Create: `schemas/scientific-cost-reservation.schema.json`
- Create: `schemas/scientific-cost-reconciliation.schema.json`
- Test: `tests/unit/evaluation/test_compute_authorization.py`
- Test: `tests/contract/evaluation/test_paid_launch_guard.py`

- [ ] **Step 1: Add the scientific-only compute policy**

```yaml
# configs/scientific/compute-policy.yaml
schema_version: "1.0"
authorization_scope: scientific
forbidden_authorization_scopes: [engineering_pilot_only]
provider: modal
workspace_selection: explicit_operator_file
profile_prefix: ratemem-scientific-
automatic_workspace_discovery: false
automatic_workspace_reuse: false
automatic_workspace_rotation: false
automatic_workspace_fallback: false
outer_workspace_usage_budget_usd: "28.00"
internal_reservation_limit_usd: "27.00"
reservation_formula: known_usage_plus_pending_worst_case_plus_new_phase_bound
workspace_attestation_max_age_seconds: 900
reservation_max_age_seconds: 900
one_phase_per_authorization: true
one_launch_per_reservation: true
require_reconciliation_before_next_phase: true
require_dataset_lock: true
require_baseline_lock: true
require_evaluation_lock: true
require_method_lock: true
require_method_cpu_gate: true
credential_policy:
  interactive_non_echoing_authentication: true
  named_non_global_profile: true
  profile_must_be_explicit_on_every_provider_call: true
  forbid_repository_credentials: true
  forbid_command_argument_credentials: true
  forbid_environment_dumps: true
  forbid_raw_provider_config_capture: true
  forbid_credentials_in_logs_and_artifacts: true
```

- [ ] **Step 2: Write the failing scope and workspace-selection tests**

```python
def test_engineering_pilot_authorization_cannot_authorize_scientific_compute() -> None:
    pilot = authorization(scope="engineering_pilot_only", workspace_id="ws-a")
    with pytest.raises(ScientificComputeDenied, match="engineering-pilot authorization"):
        verify_scientific_authorization(pilot, phase_request("meta_train_seed_17"), POLICY)


def test_workspace_must_be_explicit_and_match_fresh_cap_evidence() -> None:
    selection = workspace_selection(workspace_id="ws-selected", explicit_profile="ratemem-scientific-study-a")
    snapshot = workspace_snapshot(workspace_id="ws-other", outer_budget_usd="28.00")
    with pytest.raises(ScientificComputeDenied, match="explicit workspace mismatch"):
        authorize_scientific_phase(selection, snapshot, phase_request("meta_train_seed_17"), LOCKS, CPU_GATE, POLICY)


@pytest.mark.parametrize("outer_cap", ["27.99", "28.01", "100.00"])
def test_outer_workspace_cap_must_be_exactly_28_usd(outer_cap: str) -> None:
    with pytest.raises(ScientificComputeDenied, match="USD 28.00 outer cap"):
        authorize_scientific_phase(SELECTION, workspace_snapshot(outer_budget_usd=outer_cap), PHASE, LOCKS, CPU_GATE, POLICY)


def test_full_scientific_authorization_requires_current_zero_provider_cpu_gate() -> None:
    with pytest.raises(ScientificComputeDenied, match="learned-method CPU gate"):
        authorize_scientific_phase(SELECTION, SNAPSHOT, PHASE, LOCKS, None, POLICY)
    stale = method_cpu_gate(git_commit="0" * 40)
    with pytest.raises(ScientificComputeDenied, match="learned-method CPU gate"):
        authorize_scientific_phase(SELECTION, SNAPSHOT, PHASE, LOCKS, stale, POLICY)
    non_cpu = method_cpu_gate(provider_invocations=1)
    with pytest.raises(ScientificComputeDenied, match="learned-method CPU gate"):
        authorize_scientific_phase(SELECTION, SNAPSHOT, PHASE, LOCKS, non_cpu, POLICY)
```

- [ ] **Step 3: Write the failing reservation and credential tests**

```python
def test_reservation_includes_known_pending_and_new_cost() -> None:
    accepted = reserve_scientific_cost(
        AUTHORIZATION, known_usage=Decimal("10.00"), pending_worst_case=Decimal("8.00"),
        new_phase_bound=Decimal("9.00"), ledger=EMPTY_LEDGER,
    )
    assert accepted.reserved_total_usd == Decimal("27.00")
    with pytest.raises(ScientificComputeDenied, match="internal USD 27.00 limit"):
        reserve_scientific_cost(
            AUTHORIZATION, known_usage=Decimal("10.00"), pending_worst_case=Decimal("8.01"),
            new_phase_bound=Decimal("9.00"), ledger=EMPTY_LEDGER,
        )


def test_credential_shaped_fields_are_rejected_recursively() -> None:
    with pytest.raises(CredentialMaterialDetected):
        validate_credential_free_payload(
            {**VALID_AUTH, "metadata": {"modal_" "token_secret": "secret-value"}}
        )
```

- [ ] **Step 4: Run the authorization tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_compute_authorization.py -q`

Expected: collection fails because `ratemem.evaluation.compute` does not exist.

- [ ] **Step 5: Implement the scope-bound authorization records**

```python
from ratemem.evaluation.types import GitCommit, PhaseId, ScientificProfile, Sha256


class WorkspaceSelection(BaseModel):
    workspace_id: str
    explicit_profile: ScientificProfile
    selected_by: str
    selected_at_utc: AwareDatetime
    selection_file_sha256: Sha256


class ScientificComputeAuthorization(BaseModel):
    schema_version: Literal["1.0"]
    authorization_id: str
    scope: Literal["scientific"]
    provider: Literal["modal"]
    phase_id: PhaseId
    workspace_id: str
    explicit_profile: ScientificProfile
    outer_workspace_usage_budget_usd: Decimal
    known_usage_usd: Decimal
    workspace_snapshot_sha256: Sha256
    budget_evidence_sha256: Sha256
    rates_snapshot_sha256: Sha256
    dataset_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    method_lock_sha256: Sha256
    method_cpu_gate_sha256: Sha256
    git_commit: GitCommit
    issued_at_utc: AwareDatetime
    expires_at_utc: AwareDatetime
    operator_approval_sha256: Sha256
    credential_scan_sha256: Sha256


class WorkspaceSnapshot(BaseModel):
    workspace_id: str
    explicit_profile: ScientificProfile
    outer_workspace_usage_budget_usd: Decimal
    known_metered_usage_usd: Decimal
    billing_snapshot_sha256: Sha256
    budget_evidence_sha256: Sha256
    rates_snapshot_sha256: Sha256
    captured_at_utc: AwareDatetime


class PaidPhaseRequest(BaseModel):
    phase_id: PhaseId
    gpu_sku: str
    gpu_count: Literal[1]
    cpu_cores: PositiveInt
    memory_gib: PositiveInt
    timeout_seconds: PositiveInt
    retries: Literal[0]
    detached: Literal[False]
    new_phase_bound_usd: Decimal
    request_sha256: Sha256


class ScientificLocks(BaseModel):
    dataset_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    method_lock_sha256: Sha256
    method_cpu_gate_sha256: Sha256


class MethodCpuGateReceipt(BaseModel):
    schema_version: Literal["1.0"]
    status: Literal["pass"]
    method_lock_sha256: Sha256
    git_commit: GitCommit
    clean_diff_sha256: Sha256
    test_command_sha256: Sha256
    provider_invocations: Literal[0]
    created_at_utc: AwareDatetime
```

Parse `ComputePolicy` from `configs/scientific/compute-policy.yaml` with exact keys and Decimal amounts. Extend Task 8's shared workspace/reservation primitives; do not replace them or start a second cost ledger. Implement `capture_selected_workspace_snapshot(selection: WorkspaceSelection, budget_evidence_path: Path, policy: ComputePolicy) -> WorkspaceSnapshot`, `authorize_scientific_phase(selection: WorkspaceSelection, snapshot: WorkspaceSnapshot, phase: PaidPhaseRequest, locks: ScientificLocks, method_cpu_gate: MethodCpuGateReceipt, policy: ComputePolicy) -> ScientificComputeAuthorization`, `verify_scientific_authorization(authorization: object, phase: PaidPhaseRequest, locks: ScientificLocks, method_cpu_gate: MethodCpuGateReceipt, policy: ComputePolicy) -> ScientificComputeAuthorization`, and `validate_credential_free_payload(payload: Mapping[str, object]) -> None`. Require a caller-supplied selection file, exact workspace-ID/profile agreement, an explicitly named `ratemem-scientific-*` non-global profile on every provider query, fresh usage/cap/rate evidence, exact outer cap `Decimal("28.00")`, current dataset/baseline/evaluation/method-lock hashes, the current CPU-gate file hash, `status="pass"`, its current clean 40-hex commit and empty-diff hash, its exact test-command hash, `provider_invocations == 0`, and `scope == "scientific"`. Do not import or accept the pilot plan's `ModalPilotAuthorization`, `configs/pilot/modal-budget.json`, workspace permit, profile, or reservation ledger. The `baseline_fidelity` scope defined in Task 8 deliberately does not require the learned-method lock or CPU gate and cannot be widened into this scope.

- [ ] **Step 6: Implement the append-only scientific reservation ledger**

```python
class ScientificCostReservation(BaseModel):
    schema_version: Literal["1.0"]
    reservation_id: str
    authorization_sha256: Sha256
    phase_id: PhaseId
    workspace_id: str
    known_usage_usd: Decimal
    pending_worst_case_usd: Decimal
    new_phase_bound_usd: Decimal
    reserved_total_usd: Decimal
    internal_limit_usd: Decimal = Field(default=Decimal("27.00"))
    previous_ledger_entry_sha256: Sha256 | None
    created_at_utc: AwareDatetime
    expires_at_utc: AwareDatetime
    consumed: bool


class ConsumedPermit(BaseModel):
    authorization_sha256: Sha256
    reservation_sha256: Sha256
    phase_id: PhaseId
    workspace_id: str
    consumed_at_utc: AwareDatetime
    launch_receipt_sha256: Sha256
```

Add a model validator that rejects any `internal_limit_usd` other than `Decimal("27.00")`. Implement `ScientificCostLedger(path: Path, internal_limit_usd: Decimal)`, with methods `pending_worst_case(workspace_id: str) -> Decimal`, `head_sha256() -> str | None`, and `append(reservation: ScientificCostReservation) -> None`; each method verifies the full hash chain while holding the ledger lock. Implement `reserve_scientific_cost(authorization: ScientificComputeAuthorization, known_usage: Decimal, pending_worst_case: Decimal, new_phase_bound: Decimal, ledger: ScientificCostLedger) -> ScientificCostReservation`. Under an exclusive file lock, derive pending worst-case from every unreconciled entry, verify it equals the supplied attested pending value, verify the provider's known usage has not decreased, require the exact equation `known + pending + new <= 27.00`, append a hash-chained record, and never reinterpret credits as reduced metered usage.

- [ ] **Step 7: Implement a one-use paid-launch guard**

Implement `require_scientific_compute_permit(authorization_path: Path, reservation_path: Path, dataset_lock_path: Path, baseline_lock_path: Path, evaluation_lock_path: Path, method_lock_path: Path, method_cpu_gate_path: Path, expected_phase_id: str, expected_workspace_id: str, launch_receipt_path: Path) -> ConsumedPermit`. Revalidate freshness, all five lock/gate hashes, CPU-gate status/current clean commit/current empty-diff hash/exact test-command hash/zero provider invocations, workspace, profile, cap, ledger head, reservation equation, phase, and scope immediately before provider invocation. Atomically create the launch receipt with `O_CREAT | O_EXCL`; a second use, phase mismatch, workspace mismatch, stale evidence, reconciled-cost gap, stale CPU receipt, or engineering-pilot scope raises `ScientificComputeDenied` before GPU allocation.

- [ ] **Step 8: Add the explicit workspace-attestation command**

```bash
uv run ratemem-eval compute attest-workspace \
  --policy configs/scientific/compute-policy.yaml \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
  --output /home/ubuntu/.config/ratemem/scientific-workspace-snapshot.json
```

Expected without the explicit selection, named scientific profile, or fresh evidence: exit 2 with `BLOCKED scientific-compute: explicit workspace selection and USD 28.00 cap evidence required`. The implementation queries billing/rates only for the selected workspace using the profile in the selection file; it never lists, chooses, activates globally, reuses, or rotates workspaces. With valid evidence, stdout matches `^PASS scientific-workspace attestation: workspace=[a-zA-Z0-9_-]+ outer_cap=28.00 known_usage=[0-9]+\.[0-9]{2}$`.

- [ ] **Step 9: Add the explicit phase-authorization command**

```bash
uv run ratemem-method verify-cpu-receipt \
  --method-lock configs/method/ratemem-training-lock.yaml \
  --receipt artifacts/method/cpu-gate.json
uv run ratemem-eval compute authorize \
  --policy configs/scientific/compute-policy.yaml \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --workspace-snapshot /home/ubuntu/.config/ratemem/scientific-workspace-snapshot.json \
  --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
  --phase-request artifacts/scientific/compute/phase-request.json \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --method-lock configs/method/ratemem-training-lock.yaml \
  --method-cpu-gate artifacts/method/cpu-gate.json \
  --output artifacts/scientific/compute/authorization.json
```

Expected: the first command prints `PASS RateMem CPU receipt: <64hex>`. Missing, stale, wrong-commit, dirty-diff, wrong-command, non-pass, or nonzero-provider CPU evidence exits 2 before workspace authorization. Without exact workspace/profile agreement and current dataset/baseline/evaluation/method/CPU-gate hashes, the second command exits 2 with `BLOCKED scientific-compute: workspace, lock, or learned-method CPU-gate attestation mismatch`. The command never lists available workspaces or chooses one. With verified inputs, stdout matches `^PASS scientific-compute authorization: phase=[a-z0-9_-]+ workspace=[a-zA-Z0-9_-]+ authorization=[0-9a-f]{64}$`.

- [ ] **Step 10: Add the reservation command**

```bash
uv run ratemem-eval compute reserve \
  --authorization artifacts/scientific/compute/authorization.json \
  --phase-bound artifacts/scientific/compute/phase-cost-bound.json \
  --ledger /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl \
  --output artifacts/scientific/compute/reservation.json
```

Expected: stdout matches `^PASS scientific-compute reservation: known=[0-9]+\.[0-9]{2} pending=[0-9]+\.[0-9]{2} new=[0-9]+\.[0-9]{2} total=[0-9]+\.[0-9]{2} <= 27.00$`. If the sum exceeds 27.00, exit 2 and do not append or write a reservation.

- [ ] **Step 11: Add the post-phase reconciliation command**

```bash
uv run ratemem-eval compute reconcile \
  --authorization artifacts/scientific/compute/authorization.json \
  --reservation artifacts/scientific/compute/reservation.json \
  --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
  --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
  --ledger /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl \
  --output artifacts/scientific/compute/reconciliation.json
```

Implement `reconcile_scientific_cost(authorization: ScientificComputeAuthorization, reservation: ScientificCostReservation, fresh_snapshot: WorkspaceSnapshot, ledger: ScientificCostLedger) -> ScientificCostReconciliation`. It requires the same explicitly selected workspace/profile and USD 28.00 cap, records the metered-usage delta and provider call/attempt IDs, clears only that reservation's pending bound, and appends a hash-chained reconciliation. Expected stdout matches `^PASS scientific-compute reconciliation: workspace=[a-zA-Z0-9_-]+ metered_delta=[0-9]+\.[0-9]{2} pending_remaining=[0-9]+\.[0-9]{2}$`. A new phase authorization is blocked while the selected workspace has an unreconciled consumed reservation.

Use this exact record:

```python
class ScientificCostReconciliation(BaseModel):
    schema_version: Literal["1.0"]
    authorization_sha256: Sha256
    reservation_sha256: Sha256
    workspace_id: str
    usage_before_usd: Decimal
    usage_after_usd: Decimal
    metered_delta_usd: Decimal
    pending_remaining_usd: Decimal
    provider_call_ids: list[str]
    reconciled_at_utc: AwareDatetime
    previous_ledger_entry_sha256: Sha256
```

- [ ] **Step 12: Add the paid-launch contract test**

The contract test parses `src/ratemem/evaluation/`, future `src/ratemem/scientific/`, and `scripts/` with `ast`/shell-token parsing. Any Modal `.remote`, `.spawn`, deploy, map, detached, or provider-launch call in a scientific path must be immediately dominated by `require_scientific_compute_permit` or the narrower Task 8 fidelity guard; it rejects imports of pilot authorization types, automatic workspace-list iteration, default/global profiles, fallback workspace arrays, raw token/config queries, environment dumps, and a full-scientific guard call that omits the current method lock or CPU receipt.

Run: `uv run pytest tests/unit/evaluation/test_compute_authorization.py tests/contract/evaluation/test_paid_launch_guard.py -q`

Expected: all tests pass; injected unguarded launch, pilot permit, second workspace candidate, stale cap evidence, and credential-bearing fixture cases all fail closed.

- [ ] **Step 13: Generate schemas and commit the authorization layer**

Run: `uv run ratemem-eval compute schema --authorization-output schemas/scientific-compute-authorization.schema.json --reservation-output schemas/scientific-cost-reservation.schema.json --reconciliation-output schemas/scientific-cost-reconciliation.schema.json`

Expected: exit 0 and all three committed schemas byte-match their Pydantic-generated forms.

```bash
git add configs/scientific/compute-policy.yaml src/ratemem/evaluation/compute.py schemas/scientific-compute-authorization.schema.json schemas/scientific-cost-reservation.schema.json schemas/scientific-cost-reconciliation.schema.json tests/unit/evaluation/test_compute_authorization.py tests/contract/evaluation/test_paid_launch_guard.py
git commit -m "feat(eval): authorize scientific paid compute explicitly"
```

### Task 10: Replay immutable lifecycle protocols without probe side effects

**Files:**
- Create: `src/ratemem/evaluation/replay.py`
- Create: `schemas/scientific-event-result.schema.json`
- Create: `schemas/scientific-comparative-execution-freeze.schema.json`
- Test: `tests/unit/evaluation/test_replay.py`
- Test: `tests/integration/evaluation/test_replay_protocols.py`
- Test: `tests/contract/evaluation/test_comparative_execution_freeze.py`
- Test: `tests/contract/evaluation/test_ratemem_canonical_protocol.py`

- [ ] **Step 1: Write failing replay tests for immutable probes and paired traffic**

```python
def test_probe_scores_a_copy_without_changing_usage_bytes_or_state_digest(method: FakeMethod, trace: Trace) -> None:
    before = method.inspect_state()
    result = replay_probe(method, trace.first_probe, evaluator=FakeEvaluator())
    after = method.inspect_state()
    assert result.update_usage is False
    assert (after.usage_sha256, after.serialized_bytes, after.state_sha256) == (
        before.usage_sha256, before.serialized_bytes, before.state_sha256,
    )


def test_all_methods_receive_identical_operational_events_prompts_and_noise(trace: Trace) -> None:
    receipts = replay_methods(trace, [RecordingMethod("a"), RecordingMethod("b")], CONTRACT)
    assert receipts["a"].input_commitment_sha256 == receipts["b"].input_commitment_sha256


def test_no_pressure_independent_cache_has_exactly_zero_fixed_seed_drift(no_pressure_trace: Trace) -> None:
    rows = replay_method(no_pressure_trace, IndependentCodeCacheFixture(), CONTRACT)
    assert {row.identity_state_sha256 for row in rows if row.handle == "h_fixed"} == {rows[0].identity_state_sha256}


def test_budget_pressure_reports_rejection_eviction_and_stale_handle(trace: Trace) -> None:
    rows = replay_method(trace, TinyBudgetMethod(), CONTRACT)
    assert {row.outcome for row in rows} >= {"rejected", "evicted", "stale_handle"}


def test_representation_and_allocator_ablations_hold_the_other_side_fixed(ablation_lock: AblationLock) -> None:
    assert all(cell.allocator_sha256 == ablation_lock.primary_allocator_sha256 for cell in ablation_lock.representation_cells)
    assert all(cell.candidate_stream_sha256 == ablation_lock.primary_candidate_stream_sha256 for cell in ablation_lock.allocator_cells)


from ratemem.method.adapter import RateMemAdapter


def test_ratemem_adapter_runs_through_the_canonical_protocol(ratemem_adapter: RateMemAdapter, trace: Trace) -> None:
    from ratemem.baselines.protocol import BaselineAdapter
    from ratemem.evaluation.baselines import BaselineAdapter as EvaluationBaselineAdapter

    assert EvaluationBaselineAdapter is BaselineAdapter
    assert isinstance(ratemem_adapter, BaselineAdapter)
    artifact = replay_method(trace, ratemem_adapter, CONTRACT)
    assert artifact.method_id == "ratemem_v1"
    assert all(row.online_state_bytes <= CONTRACT.byte_budget for row in artifact.events)
```

- [ ] **Step 2: Run the replay tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_replay.py -q`

Expected: collection fails because `ratemem.evaluation.replay` does not exist.

- [ ] **Step 3: Import the sole frozen comparison contract and implement only event-result records**

```python
from ratemem.baselines.protocol import (
    BaselineAdapter,
    CausalEventView,
    EventReceipt,
    FrozenComparisonContract,
    ProbeResult,
)


class EventResult(BaseModel):
    artifact_id: str
    method_id: str
    trace_id: str
    protocol: Literal["no_pressure", "budget_pressure", "autonomous_lookup"]
    event_index: NonNegativeInt
    event_kind: Literal["create", "update", "read", "delete", "probe"]
    handle: str | None
    prompt_id: str | None
    generation_seed: int | None
    outcome: Literal["created", "updated", "read", "deleted", "probed", "rejected", "evicted", "stale_handle"]
    online_state_bytes: NonNegativeInt
    state_component_bytes: dict[str, NonNegativeInt]
    shared_trained_bytes: NonNegativeInt
    usage_sha256_before: str
    usage_sha256_after: str
    state_sha256_before: str
    state_sha256_after: str
    latency_ms: NonNegativeFloat
    peak_memory_bytes: NonNegativeInt
```

Require `sum(state_component_bytes.values()) == online_state_bytes <= byte_budget` after every mutable event. Copy the canonical ledger's component map whose exact keys are `base_codes`, `packet_payloads`, `packet_hashes`, `incidences_gains`, `feature_cache`, `optional_tokens`, `handles`, `usage_age`, `reference_counts`, `controller_state`, `allocator_state`, `checksums`, and `alignment`.

- [ ] **Step 4: Implement the three replay protocols**

Implement `replay_method(trace: Trace, method: BaselineAdapter, contract: FrozenComparisonContract) -> ReplayArtifact`, `replay_methods(trace: Trace, methods: Sequence[BaselineAdapter], contract_factory: Callable[[str], FrozenComparisonContract]) -> dict[str, ReplayArtifact]`, and `replay_probe(method: BaselineAdapter, probe: ProbeEvent, evaluator: LockedEvaluator) -> ProbeResult`. Build `CausalEventView(trace.events, current_index)` for each operational call and pass it to `apply_event`; never add fields or constructors to the canonical protocol from the evaluation package.

For no-pressure, compute a method-specific sufficient budget before replay and assert no forced eviction. For budget-pressure, use the exact locked `25pct`, `50pct`, or `75pct` bytes. For autonomous lookup, remove the handle from read/update inputs and record `allocate`, `update`, or `reject` plus truth labels. Deep-copy or immutable-snapshot every probe, call the method with `update_usage=false`, and compare usage digest, state digest, byte ledger, reference counts, and handles before/after; any change raises `ProbeMutationError` and invalidates the artifact.

- [ ] **Step 5: Add deterministic replay and stale-handle integration tests**

Run the same trace twice and compare canonical event rows within declared evaluator numeric tolerance. Delete a handle, probe it, and require `stale_handle`; verify deletion collateral rows list only explicitly allocator-affected concepts and that unrelated decoded-code hashes remain fixed.

Run: `uv run pytest tests/unit/evaluation/test_replay.py tests/integration/evaluation/test_replay_protocols.py tests/contract/evaluation/test_ratemem_canonical_protocol.py -q && uv run mypy src/ratemem/baselines/protocol.py src/ratemem/evaluation/baselines.py src/ratemem/evaluation/replay.py src/ratemem/method/adapter.py`

Expected: all tests and mypy pass; the RateMem adapter is a runtime `BaselineAdapter`, returns the canonical receipt/ledger models, and any probe-mutating fixture fails with `ProbeMutationError`.

- [ ] **Step 6: Invoke the authorized learned training producer for every locked seed**

After issuing one Task 9 authorization/reservation per seed, run the learned-plan-owned producer; never assemble checkpoint or training receipts by hand.

```bash
for training_seed in 17 29 43; do
  uv run ratemem-method train-scientific \
    --method-lock configs/method/ratemem-training-lock.yaml \
    --method-cpu-gate artifacts/method/cpu-gate.json \
    --dataset-lock configs/scientific/dataset-lock.yaml \
    --baseline-lock configs/scientific/baseline-lock.yaml \
    --evaluation-lock configs/scientific/evaluation-lock.yaml \
    --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
    --trace-dir configs/scientific/traces \
    --split train \
    --training-seed "$training_seed" \
    --compute-authorization "artifacts/scientific/method/training/seed-${training_seed}/compute/authorization.json" \
    --cost-reservation "artifacts/scientific/method/training/seed-${training_seed}/compute/reservation.json" \
    --launch-receipt "artifacts/scientific/method/training/seed-${training_seed}/compute/launch-receipt.json" \
    --checkpoint-output "artifacts/scientific/method/training/seed-${training_seed}/checkpoint.safetensors" \
    --checkpoint-manifest-output "artifacts/scientific/method/training/seed-${training_seed}/checkpoint.manifest.json" \
    --receipt-output "artifacts/scientific/method/training/seed-${training_seed}/attempt-receipt.json"
  uv run ratemem-eval compute reconcile \
    --authorization "artifacts/scientific/method/training/seed-${training_seed}/compute/authorization.json" \
    --reservation "artifacts/scientific/method/training/seed-${training_seed}/compute/reservation.json" \
    --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
    --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
    --ledger /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl \
    --output "artifacts/scientific/method/training/seed-${training_seed}/compute/reconciliation.json"
  uv run ratemem-method finalize-phase \
    --kind training \
    --attempt "artifacts/scientific/method/training/seed-${training_seed}/attempt-receipt.json" \
    --authorization "artifacts/scientific/method/training/seed-${training_seed}/compute/authorization.json" \
    --reservation "artifacts/scientific/method/training/seed-${training_seed}/compute/reservation.json" \
    --reconciliation "artifacts/scientific/method/training/seed-${training_seed}/compute/reconciliation.json" \
    --output "artifacts/scientific/method/training/seed-${training_seed}/receipt.json"
done
```

Expected: three training lines matching `^PASS RateMem scientific training: seed=(17|29|43) checkpoint=[0-9a-f]{64} receipt=[0-9a-f]{64}$`, followed for each seed by reconciliation and final-receipt PASS lines. Each immutable attempt binds the current method/CPU/dataset/baseline/evaluation hashes, consumed permit, train-only trace hashes, checkpoint/manifest hashes, and provider attempt IDs. Only the sibling `receipt.json` emitted by `finalize-phase` after reconciliation is accepted downstream; failure stops the sequential loop before the next seed authorization.

- [ ] **Step 7: Invoke the learned producer for one real shared-input bundle per seed**

```bash
for training_seed in 17 29 43; do
  uv run ratemem-method materialize-shared-inputs \
    --method-lock configs/method/ratemem-training-lock.yaml \
    --method-cpu-gate artifacts/method/cpu-gate.json \
    --dataset-lock configs/scientific/dataset-lock.yaml \
    --baseline-lock configs/scientific/baseline-lock.yaml \
    --evaluation-lock configs/scientific/evaluation-lock.yaml \
    --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
    --checkpoint "artifacts/scientific/method/training/seed-${training_seed}/checkpoint.safetensors" \
    --checkpoint-manifest "artifacts/scientific/method/training/seed-${training_seed}/checkpoint.manifest.json" \
    --training-receipt "artifacts/scientific/method/training/seed-${training_seed}/receipt.json" \
    --trace-dir configs/scientific/traces \
    --splits train,validation \
    --schema schemas/ratemem-shared-input-bundle-v1.schema.json \
    --compute-authorization "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/authorization.json" \
    --cost-reservation "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/reservation.json" \
    --launch-receipt "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/launch-receipt.json" \
    --output "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}" \
    --receipt-output "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/attempt-receipt.json"
  uv run ratemem-eval compute reconcile \
    --authorization "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/authorization.json" \
    --reservation "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/reservation.json" \
    --workspace-selection /home/ubuntu/.config/ratemem/scientific-workspace-selection.json \
    --budget-evidence /home/ubuntu/.config/ratemem/scientific-workspace-budget-evidence.png \
    --ledger /home/ubuntu/.local/state/ratemem/scientific-cost-ledger.jsonl \
    --output "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/reconciliation.json"
  uv run ratemem-method finalize-phase \
    --kind materialization \
    --attempt "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/attempt-receipt.json" \
    --authorization "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/authorization.json" \
    --reservation "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/reservation.json" \
    --reconciliation "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/compute/reconciliation.json" \
    --output "artifacts/scientific/baselines/execution/shared-input/seed-${training_seed}/receipt.json"
done
```

Expected: three materialization lines matching `^PASS real shared-inputs: seed=(17|29|43) candidate_stream=[0-9a-f]{64} receipt=[0-9a-f]{64}$`, followed for each seed by reconciliation and final-receipt PASS lines. The producer owns all tensor/file hashes and binds amortizer, basis, codec dictionary, target codes, candidate stream, traces, checkpoint manifest, finalized training receipt, consumed permit, and output schema. Baseline search accepts only the final `receipt.json`; validation or freezing rejects copied, edited, duplicate-provider, attempt-only, unreconciled, or final-test materialization.

- [ ] **Step 8: Invoke the companion baseline search CLI once per registered method**

```bash
for method_id in independent_fifo independent_lru independent_lrua private_progressive_size_aware private_progressive_separable_rate shared_packet_plain_greedy cts_style_static vb_lora_style_static share_style_online dreamcache_feature_cache hyperlora_upstream stateless_amortizer per_concept_lora exact_append_only_quantized exact_future_trace_packets; do
  uv run ratemem-baselines search run \
    --method-id "$method_id" \
    --baseline-lock configs/scientific/baseline-lock.yaml \
    --evaluation-lock configs/scientific/evaluation-lock.yaml \
    --policy configs/baselines/policy-search.yaml \
    --shared-input-dir artifacts/scientific/baselines/execution/shared-input \
    --validation-trace-dir configs/scientific/traces \
    --compute-authorization "artifacts/scientific/baselines/execution/search/${method_id}/compute/authorization.json" \
    --cost-reservation "artifacts/scientific/baselines/execution/search/${method_id}/compute/reservation.json" \
    --launch-receipt "artifacts/scientific/baselines/execution/search/${method_id}/compute/launch-receipt.json" \
    --ledger-output "artifacts/scientific/baselines/execution/search/${method_id}/ledger.jsonl" \
    --receipt-output "artifacts/scientific/baselines/execution/search/${method_id}/receipt.json"
done
```

Expected: one `^PASS baseline-search: method=[a-z0-9_-]+ trials=[0-9]+ gpu_hours=[0-9]+\.[0-9]+ ledger=[0-9a-f]{64}$` line per exact `postlock_execution_required` ID. A method with no tunable parameters emits a producer-signed zero-trial receipt rather than a hand-written file. Contextual-only SDXL evidence is not searched. The CLI enforces validation-only access, append-before-evaluate ledgers, the common bundle hashes, and locked trial/GPU-hour limits. Reconcile every consumed paid reservation into the method's `compute/reconciliation.json`; a missing required ID or permit/reconciliation/hash mismatch is fatal.

- [ ] **Step 9: Freeze only producer-generated shared-input, training, and search receipts**

Then run:

```bash
uv run ratemem-eval baselines freeze-execution \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --method-lock configs/method/ratemem-training-lock.yaml \
  --method-cpu-gate artifacts/method/cpu-gate.json \
  --shared-input-bundle-dir artifacts/scientific/baselines/execution/shared-input \
  --train-receipts-dir artifacts/scientific/method/training \
  --search-policy configs/baselines/policy-search.yaml \
  --search-ledgers-dir artifacts/scientific/baselines/execution/search \
  --compute-reconciliations-dir artifacts/scientific/baselines/execution/reconciliations \
  --output artifacts/scientific/freeze/comparative-execution-freeze.json
```

Expected: `^PASS comparative-execution-freeze: methods=15 receipt=[0-9a-f]{64}$`. The freezer verifies the exact producer CLI IDs/versions, canonical command hashes, schemas, checkpoints, bundles, append-only ledgers, consumed launch receipts, and reconciliations; it never accepts a raw tensor, checkpoint, JSON, or JSONL file without its producer receipt. Contextual-only evidence is forbidden from this SANA execution freeze; in particular, the SineLoRA-Delta SD3 Medium citation produces no adapter, search ledger, or method receipt. Missing required methods, shared-input/schema mismatch, search above the locked trials/GPU-hours, non-validation search, missing paid authorization/reconciliation, stale CPU receipt, or any final-trace access exits 2 and creates no freeze. The receipt binds actual bundle/train/search artifact hashes, not selected hyperparameters or headline results; selection consumes it in Task 16.

- [ ] **Step 10: Add the validation replay command**

```bash
uv run ratemem-eval replay validation \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --trace-dir configs/scientific/traces \
  --methods configs/scientific/baseline-lock.yaml \
  --execution-freeze artifacts/scientific/freeze/comparative-execution-freeze.json \
  --output artifacts/scientific/validation
```

Expected: one artifact directory per `(method, training_seed, trace, budget, request_regime, protocol)` and stdout matches `^PASS replay: paired input commitment=[0-9a-f]{64}$`. A method exceeding the byte cap or mutating a probe exits 2 and marks that attempt invalid.

- [ ] **Step 11: Generate both schemas and commit**

Run: `uv run ratemem-eval replay schema --output schemas/scientific-event-result.schema.json && uv run ratemem-eval baselines execution-schema --output schemas/scientific-comparative-execution-freeze.schema.json && uv run pytest tests/contract/evaluation/test_comparative_execution_freeze.py -q`

```bash
git add src/ratemem/evaluation/replay.py schemas/scientific-event-result.schema.json schemas/scientific-comparative-execution-freeze.schema.json tests/unit/evaluation/test_replay.py tests/integration/evaluation/test_replay_protocols.py tests/contract/evaluation/test_comparative_execution_freeze.py tests/contract/evaluation/test_ratemem_canonical_protocol.py
git commit -m "feat(eval): replay immutable lifecycle protocols"
```

### Task 11: Compute locked lifecycle, lookup, byte, latency, and deletion metrics

**Files:**
- Create: `src/ratemem/evaluation/metrics.py`
- Test: `tests/unit/evaluation/test_metrics.py`
- Test: `tests/contract/evaluation/test_metric_lock.py`

- [ ] **Step 1: Write failing exact-formula tests on a hand-calculated lifecycle**

```python
def test_lifecycle_formulas_match_hand_calculation() -> None:
    rows = probe_rows(
        # event, handle, request_weight, identity, prompt, acquisition_identity
        (1, "a", 3.0, 0.8, 0.7, 0.8),
        (2, "a", 3.0, 0.6, 0.8, 0.8),
        (2, "b", 1.0, 0.4, 0.6, 0.4),
    )
    result = aggregate_lifecycle(rows, utility_weights={"identity": 0.5, "prompt": 0.5})
    assert result.request_weighted_identity == pytest.approx((3 * 0.7 + 1 * 0.4) / 4)
    assert result.request_weighted_prompt == pytest.approx((3 * 0.75 + 1 * 0.6) / 4)
    assert result.maximum_active_degradation == pytest.approx(0.2)
    assert result.active_state_drift == pytest.approx(-0.1)


def test_conditional_diversity_is_missing_below_locked_fidelity_floor() -> None:
    assert conditional_diversity(diversity=0.9, identity=0.39, prompt=0.8, identity_floor=0.4, prompt_floor=0.7) is None


def test_state_byte_total_includes_every_locked_component() -> None:
    component_bytes = {name: 1 for name in REQUIRED_ONLINE_COMPONENTS}
    ledger = ExactByteLedger(
        serializer_id="ratemem-baseline-cbor-v1",
        online_state_bytes=len(component_bytes),
        online_state_sha256="a" * 64,
        component_bytes=component_bytes,
        shared_trained_bytes=12,
        external_support_bytes=20,
    )
    assert ledger.online_state_bytes == sum(ledger.component_bytes.values())
    assert amortized_accounted_bytes(ledger, active_set_size=4) == Decimal(len(component_bytes) + 3)
```

- [ ] **Step 2: Run the metric tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_metrics.py -q`

Expected: collection fails because `ratemem.evaluation.metrics` does not exist.

- [ ] **Step 3: Implement the formula registry rather than evaluating YAML expressions**

```python
FORMULAS: dict[str, Callable[[MetricInputs], float | None]] = {
    "identity_mean_v1": identity_mean,
    "prompt_mean_v1": prompt_mean,
    "request_weighted_identity_v1": request_weighted_identity,
    "equal_weight_identity_prompt_v1": equal_weight_identity_prompt,
    "normalized_event_trapezoid_v1": normalized_event_trapezoid,
    "acquisition_delta_v1": acquisition_delta,
    "maximum_acquisition_drop_v1": maximum_acquisition_drop,
    "future_oracle_utility_gap_v1": future_oracle_utility_gap,
    "lookup_risk_coverage_v1": lookup_risk_coverage,
    "thresholded_conditional_diversity_v1": conditional_diversity,
}


def resolve_formula(formula_id: str) -> Callable[[MetricInputs], float | None]:
    try:
        return FORMULAS[formula_id]
    except KeyError as error:
        raise UnknownLockedFormula(formula_id) from error
```

Never execute an arbitrary expression from YAML. Add acquisition quality immediately after each create/update; average active quality; signed active-state drift from the latest acquisition; retention AUC normalized by active lifespan; maximum active degradation; request-weighted identity/prompt/utility; offline-oracle regret; insertion rejection; stale-handle rate; allocation precision/recall; lookup risk-coverage/AURC; similar-concept confusion; deletion collateral damage; and thresholded diversity.

- [ ] **Step 4: Consume the canonical byte ledger and report amortized shared overhead**

Import `ExactByteLedger` from `ratemem.baselines.protocol`; do not define an evaluation-side ledger model. Implement `amortized_accounted_bytes(ledger: ExactByteLedger, active_set_size: int) -> Decimal` as `online_state_bytes + shared_trained_bytes / active_set_size` using Decimal precision. Report `external_support_bytes` separately and never include it as free online state. The protocol/ledger owner rejects negative values, unknown components, serializer mismatch, or `sum(component_bytes.values()) != online_state_bytes`; metrics tests exercise that canonical validation rather than a second schema.

- [ ] **Step 5: Implement pinned latency and peak memory aggregation**

Discard exactly the locked warm-up count, then report p50/p95 insert and read latency, peak allocated/reserved accelerator memory, CPU RSS, energy kWh, GPU SKU, driver/runtime, batch, resolution, sampler, steps, and synchronization policy. `validate_latency_context` compares every context field to `evaluation-lock.yaml`; mismatched hardware or generation settings remain efficiency-only and cannot enter a matched table.

- [ ] **Step 6: Contract-test every locked formula ID and prohibited small-sample FID**

The contract test loads `evaluation-policy.yaml`, resolves every formula ID, and asserts no unregistered formula. It also asserts per-concept FID raises `UnsupportedMetricError`; aggregate KID/precision-recall require minimum sample/domain conditions explicitly stored in the evaluation lock.

Run: `uv run pytest tests/unit/evaluation/test_metrics.py tests/contract/evaluation/test_metric_lock.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit the metric layer**

```bash
git add src/ratemem/evaluation/metrics.py tests/unit/evaluation/test_metrics.py tests/contract/evaluation/test_metric_lock.py
git commit -m "feat(eval): compute locked lifecycle metrics"
```

### Task 12: Validate immutable result artifacts before aggregation

**Files:**
- Create: `src/ratemem/evaluation/artifacts.py`
- Create: `schemas/scientific-result-artifact.schema.json`
- Test: `tests/unit/evaluation/test_artifacts.py`
- Test: `tests/contract/evaluation/test_result_artifact_schema.py`

- [ ] **Step 1: Write failing artifact provenance and tamper tests**

```python
def test_valid_artifact_binds_code_locks_trace_checkpoint_and_rows(tmp_path: Path) -> None:
    artifact = write_fixture_artifact(tmp_path)
    validated = validate_result_artifact(artifact)
    assert validated.schema_status == "valid"
    assert validated.manifest.dataset_lock_sha256 == "d" * 64
    assert validated.manifest.evaluation_lock_sha256 == "e" * 64
    assert validated.manifest.trace_payload_sha256 == file_sha256(artifact / "events.parquet")


def test_changed_metric_file_invalidates_artifact(tmp_path: Path) -> None:
    artifact = write_fixture_artifact(tmp_path)
    (artifact / "metrics.json").write_text("{}")
    with pytest.raises(ArtifactChecksumError, match="metrics.json"):
        validate_result_artifact(artifact)


def test_dirty_diff_hash_and_exit_status_are_mandatory(valid_manifest: dict) -> None:
    del valid_manifest["diff_sha256"]
    with pytest.raises(ValidationError):
        ScientificResultManifest.model_validate(valid_manifest)
```

- [ ] **Step 2: Run the artifact tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_artifacts.py -q`

Expected: collection fails because `ratemem.evaluation.artifacts` does not exist.

- [ ] **Step 3: Define the scientific result manifest**

```python
from ratemem.evaluation.types import GitCommit, Sha256


class ScientificResultManifest(BaseModel):
    schema_version: Literal["1.0"]
    artifact_id: str
    created_at_utc: AwareDatetime
    git_commit: GitCommit
    diff_sha256: Sha256
    config_sha256: Sha256
    backbone_revision: str
    diffusers_revision: str
    peft_revision: str
    container_digest: str
    trainable_checkpoint_sha256: Sha256
    dataset_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    method_lock_sha256: Sha256
    method_cpu_gate_sha256: Sha256
    comparative_execution_freeze_sha256: Sha256
    trace_manifest_sha256: Sha256
    trace_payload_sha256: Sha256
    method_id: str
    training_seed: int
    generation_seed_commitment: Sha256
    gpu_sku: str
    hardware_sha256: Sha256
    price_snapshot_sha256: Sha256
    timeout_seconds: PositiveInt
    attempt_ids: list[str]
    peak_memory_bytes: NonNegativeInt
    step_time_p50_ms: NonNegativeFloat
    step_time_p95_ms: NonNegativeFloat
    pending_cost_bound_usd: NonNegativeFloat
    reconciled_cost_usd: NonNegativeFloat
    paid_compute: bool
    scientific_compute_authorization_sha256: Sha256 | None
    scientific_cost_reservation_sha256: Sha256 | None
    scientific_launch_receipt_sha256: Sha256 | None
    scientific_cost_reconciliation_sha256: Sha256 | None
    exit_status: Literal["success", "failed", "invalid"]
    files: list[ChecksummedFile]
```

Extend, rather than duplicate, the pilot attempt fields. Store immutable base checkpoints by revision/hash, not by copying multi-gigabyte files. Require `events.parquet`, `probes.parquet`, `metrics.json`, `byte-ledger.json`, `latency.json`, and `validation.json` in every successful scientific artifact.

For `paid_compute=true`, require all four scientific authorization/reservation/receipt/reconciliation hashes, verify them against Task 9's schemas and phase/workspace, and reject pilot-scope permits. For `paid_compute=false`, require all four fields to be null. Cost fields must reconcile to the same scientific ledger entry; credentials, raw provider config, and environment dumps invalidate the artifact.

- [ ] **Step 4: Implement atomic artifact finalization and read-only validation**

Implement `begin_result_artifact(root: Path, identity: ArtifactIdentity) -> ArtifactWriter`, `ArtifactWriter.finalize(manifest_fields: Mapping[str, object]) -> Path`, `validate_result_artifact(path: Path) -> ValidatedArtifact`, and `load_validated_artifacts(index_path: Path) -> list[ValidatedArtifact]`. Write to `<artifact_id>.incomplete`, fsync files, compute checksums, validate schema, atomically rename to `<artifact_id>`, and then set files read-only. Aggregators accept only `ValidatedArtifact`, never arbitrary CSV/JSON paths.

- [ ] **Step 5: Generate schema and add artifact indexing command**

```bash
uv run ratemem-eval artifacts schema --output schemas/scientific-result-artifact.schema.json
uv run ratemem-eval artifacts index \
  --root artifacts/scientific/validation \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --output artifacts/scientific/validation/artifact-index.json
```

Expected: stdout matches `^PASS artifact-index: [0-9]+ valid, 0 invalid, index=[0-9a-f]{64}$`. Any invalid attempt is listed with a reason and makes the command exit 2; it is never silently dropped.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/unit/evaluation/test_artifacts.py tests/contract/evaluation/test_result_artifact_schema.py -q`

Expected: all tests pass, including checksum tamper detection.

```bash
git add src/ratemem/evaluation/artifacts.py schemas/scientific-result-artifact.schema.json tests/unit/evaluation/test_artifacts.py tests/contract/evaluation/test_result_artifact_schema.py
git commit -m "feat(eval): validate immutable result artifacts"
```

### Task 13: Perform paired hierarchical inference and Holm correction

**Files:**
- Modify: `src/ratemem/evaluation/statistics.py`
- Create: `schemas/scientific-claim-statistics.schema.json`
- Test: `tests/unit/evaluation/test_statistics.py`
- Test: `tests/integration/evaluation/test_statistics_artifacts.py`

- [ ] **Step 1: Write failing tests for paired inference units and nested observations**

```python
def test_constant_paired_episode_gain_has_degenerate_positive_ci() -> None:
    rows = paired_rows(training_seeds=3, episodes=8, concepts=4, prompts=3, treatment_gain=0.1)
    result = paired_hierarchical_bootstrap(
        rows, method_id="ratemem", comparator_id="share_style_online",
        inference_unit="deployment_episode", hierarchy=("training_seed", "episode_id"),
        n_resamples=2_000, seed=271828,
    )
    assert result.estimate == pytest.approx(0.1)
    assert result.ci_low == pytest.approx(0.1)
    assert result.ci_high == pytest.approx(0.1)
    assert result.n_inference_units == 24


def test_prompts_and_images_do_not_inflate_inference_unit_count() -> None:
    rows = paired_rows(training_seeds=3, episodes=5, concepts=2, prompts=9, images_per_prompt=4)
    result = collapse_to_inference_units(rows, claim_id="shared_packet_representation")
    assert len(result) == 15


def test_holm_adjustment_matches_known_family() -> None:
    assert holm_adjust([0.01, 0.03, 0.04]) == pytest.approx([0.03, 0.06, 0.06])
```

- [ ] **Step 2: Run the statistics tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_statistics.py -q`

Expected: collection fails because `ratemem.evaluation.statistics` does not exist.

- [ ] **Step 3: Collapse paired rows at each claim's declared inference unit**

Implement `collapse_to_inference_units(rows: DataFrame, claim: ClaimLock) -> DataFrame`. Pair on dataset, protocol, budget, request regime, training seed, trace order, concept/pair as applicable, prompt template, and generation seed before aggregation. Use these locked outer units:

| Claim | Outer inference unit | Nested observations |
|---|---|---|
| `shared_packet_representation` | deployment episode | concepts, prompt templates, generated images |
| `causal_packet_allocator` | deployment episode | event positions, concepts, prompt templates |
| `optimization_free_tradeoff` | held-out concept | prompts and generated images |
| `autonomous_lookup` | concept-conditioned lookup episode | lookup events |
| `optional_composition` | prespecified concept pair | prompts and generated images |
| `optional_augmentation` | independently drawn class/support split | classes and test examples |

Prompts/images are averaged inside their parent unit and are never counted as independent replicates. Reject an unpaired row rather than imputing it.

- [ ] **Step 4: Implement deterministic hierarchical bootstrap and mixed-effects sensitivity**

Implement `paired_hierarchical_bootstrap(rows: DataFrame, method_id: str, comparator_id: str, inference_unit: str, hierarchy: Sequence[str], n_resamples: int, seed: int) -> BootstrapResult`. Resample training seeds with replacement, then paired inference units within selected seeds; preserve every method/comparator pair and aggregate nested prompts before resampling. Return percentile 95% CI, effect size, Monte Carlo standard error, and unit counts. Add `fit_mixed_effects_sensitivity` with fixed treatment effect and random intercepts for training seed, concept, trace order, and prompt template; label it sensitivity analysis, not a replacement for the locked primary bootstrap.

- [ ] **Step 5: Implement Holm families and non-inferiority transforms**

`holm_adjust` sorts raw p-values, applies `min(1, (m-rank+1)*p)`, and enforces monotonic adjusted values. Represent higher-is-better superiority as paired `method - comparator`; lower-is-better as `comparator - method`; non-inferiority passes when the transformed CI lower bound is greater than `-margin`. Group secondary comparisons by the exact `multiplicity_family` in the evaluation lock.

- [ ] **Step 6: Revalidate the frozen CI-width/power record during analysis**

Load `configs/scientific/required-units.json` through the Task 5 schema and verify that the final artifact index contains at least the locked number of paired inference units in every primary cell, without recomputing or lowering the target after results are visible. Retain the planning command for reproducibility:

```bash
uv run ratemem-eval stats plan-units \
  --calibration-record artifacts/scientific/calibration/calibration-record.json \
  --maximum-half-width 0.02 \
  --minimum-effect 0.03 \
  --alpha 0.05 \
  --power 0.80 \
  --minimum-units 12 \
  --simulation-seed 314159 \
  --output configs/scientific/required-units.json
```

Expected: stdout matches `^PASS power-plan: final deployment episodes=[0-9]+; target_half_width=0.02; power=0.80$`, with the calibration artifact hash recorded.

- [ ] **Step 7: Test artifact-only analysis and commit**

Run: `uv run pytest tests/unit/evaluation/test_statistics.py tests/integration/evaluation/test_statistics_artifacts.py -q`

Expected: all tests pass; direct DataFrame paths not wrapped by validated artifacts raise `UnvalidatedEvidenceError` in scientific mode.

```bash
git add src/ratemem/evaluation/statistics.py schemas/scientific-claim-statistics.schema.json tests/unit/evaluation/test_statistics.py tests/integration/evaluation/test_statistics_artifacts.py
git commit -m "feat(eval): add paired hierarchical inference"
```

### Task 14: Export and ingest a blinded paired human study

**Files:**
- Create: `configs/scientific/human-study-policy.yaml`
- Create: `src/ratemem/evaluation/human_study.py`
- Create: `schemas/scientific-human-study-export.schema.json`
- Test: `tests/unit/evaluation/test_human_study.py`
- Test: `tests/contract/evaluation/test_human_study_blinding.py`

- [ ] **Step 1: Prespecify pairing, questions, balance, and exclusions**

```yaml
# configs/scientific/human-study-policy.yaml
schema_version: "1.0"
design: paired_blinded_balanced_incomplete_block_v1
comparison: strongest_prespecified_eligible_control
pair_keys: [dataset_id, concept_token, prompt_id, generation_seed]
questions:
  - {id: identity_fidelity, choices: [left, tie, right]}
  - {id: prompt_alignment, choices: [left, tie, right]}
minimum_ratings_per_pair: 5
maximum_pairs_per_participant: 40
left_right_balance_tolerance: 1
attention_checks_per_participant: 2
exclusion_rules:
  incomplete_assignment: exclude
  duplicate_submission: keep_first_completed
  failed_attention_checks_at_least: 2
  minimum_median_seconds_per_pair: 2.0
participant_identifier: salted_sha256
collect_free_text: false
collect_direct_identifiers: false
analysis_unit: concept
```

The evaluation lock stores the final pair count from the power calculation, study policy hash, instruction hash, and strongest comparator ID before export.

- [ ] **Step 2: Write failing deterministic blinding and balance tests**

```python
def test_export_pairs_exact_prompt_seed_and_hides_method_labels(tmp_path: Path) -> None:
    export = build_study_export(validated_images(), LOCK, STUDY_POLICY, output_dir=tmp_path)
    assert all(pair.left.prompt_id == pair.right.prompt_id for pair in export.pairs)
    assert all(pair.left.generation_seed == pair.right.generation_seed for pair in export.pairs)
    public_bytes = b"".join(path.read_bytes() for path in export.public_files)
    assert b"ratemem" not in public_bytes.lower()
    assert b"share" not in public_bytes.lower()


def test_left_right_assignment_is_balanced_per_method_and_concept(tmp_path: Path) -> None:
    export = build_study_export(validated_images(), LOCK, STUDY_POLICY, output_dir=tmp_path)
    counts = export.assignment_counts()
    assert all(abs(left - right) <= 1 for left, right in counts.values())


def test_unblinding_requires_frozen_response_hash(tmp_path: Path) -> None:
    export = build_study_export(validated_images(), LOCK, STUDY_POLICY, output_dir=tmp_path)
    with pytest.raises(StudyNotFrozenError):
        import_responses(tmp_path / "responses.csv", export.blinding_key, frozen_response_sha256=None)
```

- [ ] **Step 3: Run the study tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_human_study.py -q`

Expected: collection fails because `ratemem.evaluation.human_study` does not exist.

- [ ] **Step 4: Implement the public export and encrypted blinding key**

Define `StudyPair` with `pair_id`, `concept_token`, `prompt_text`, `left_asset`, `right_asset`, and question IDs; public asset names are random UUIDs and contain no method/checkpoint tokens. Define a private `BlindingRecord` mapping `(pair_id, side)` to `method_id`, source artifact ID, image SHA-256, dataset, concept, prompt, and seed. Encrypt the canonical private mapping with the same X25519 envelope primitive as Task 6 under a separate study recipient key.

Implement `build_study_export(images: Sequence[ValidatedImage], lock: EvaluationLock, policy: HumanStudyPolicy, output_dir: Path) -> HumanStudyExport`. Use a deterministic block randomization seed committed in the evaluation lock, balance left/right within method and concept, ensure a participant never receives both orientations of a pair, copy assets without recompression, and checksum every public file.

- [ ] **Step 5: Implement response validation, freeze, exclusion, and paired rows**

Require response columns `study_id`, `assignment_id`, `participant_hash`, `pair_id`, `identity_fidelity`, `prompt_alignment`, `elapsed_seconds`, `completed_at_utc`, and `attention_check_passed`. `freeze_responses` canonicalizes rows and writes a SHA-256 receipt before unblinding. `import_responses` applies only the locked exclusions, emits an inclusion audit with reason counts, joins the private key after hash verification, maps left/tie/right to `1/0.5/0`, and collapses ratings to paired concept-level rows for Task 13.

- [ ] **Step 6: Add export/import commands and contract scan**

```bash
uv run ratemem-eval human-study export \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --artifact-index artifacts/scientific/final/artifact-index.json \
  --recipient configs/scientific/human-study-recipient.pem \
  --output artifacts/scientific/human-study/export
uv run ratemem-eval human-study freeze-responses \
  --export-manifest artifacts/scientific/human-study/export/manifest.json \
  --responses artifacts/scientific/human-study/raw-responses.csv \
  --output artifacts/scientific/human-study/response-freeze.json
uv run ratemem-eval human-study import \
  --response-freeze artifacts/scientific/human-study/response-freeze.json \
  --private-key /home/ubuntu/.config/ratemem/human-study-x25519.key \
  --output artifacts/scientific/human-study/analysis
```

Expected export: `PASS human-study export: labels_hidden=true balance_tolerance<=1`. Expected import: `PASS human-study import: response hash verified; exclusions locked`. The contract test recursively scans public files for method IDs, artifact IDs, checkpoint paths, and EXIF metadata.

- [ ] **Step 7: Generate schema, run tests, and commit**

Run: `uv run ratemem-eval human-study schema --output schemas/scientific-human-study-export.schema.json && uv run pytest tests/unit/evaluation/test_human_study.py tests/contract/evaluation/test_human_study_blinding.py -q`

Expected: all tests pass.

```bash
git add configs/scientific/human-study-policy.yaml src/ratemem/evaluation/human_study.py schemas/scientific-human-study-export.schema.json tests/unit/evaluation/test_human_study.py tests/contract/evaluation/test_human_study_blinding.py
git commit -m "feat(eval): export blinded paired human study"
```

### Task 15: Publish artifact-driven tables, curves, statistics, and qualitative selections

**Files:**
- Create: `src/ratemem/evaluation/publish.py`
- Create: `schemas/paper-release.schema.json`
- Test: `tests/unit/evaluation/test_publish.py`
- Test: `tests/integration/evaluation/test_paper_release.py`

- [ ] **Step 1: Write failing release-schema and source-lineage tests**

```python
def test_release_contains_only_values_recomputed_from_valid_artifacts(tmp_path: Path) -> None:
    release = publish_paper_release(valid_artifact_index(), valid_claim_stats(), valid_gates(), RELEASE_ID, tmp_path)
    validate_paper_release(release)
    main = pd.read_csv(release / "tables/main_lifecycle.csv")
    assert list(main.columns) == MAIN_LIFECYCLE_COLUMNS
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in main.artifact_ids_sha256)


def test_unvalidated_or_failed_artifact_blocks_release(tmp_path: Path) -> None:
    index = valid_artifact_index()
    index.artifacts[0].schema_status = "invalid"
    with pytest.raises(PaperReleaseError, match="invalid source artifact"):
        publish_paper_release(index, valid_claim_stats(), valid_gates(), RELEASE_ID, tmp_path)


def test_evaluated_failed_gate_is_published_with_negative_disposition(tmp_path: Path) -> None:
    gates = valid_gates().with_status("shared_packet_representation", "fail")
    release = publish_paper_release(valid_artifact_index(), valid_claim_stats(), gates, RELEASE_ID, tmp_path)
    manifest = json.loads((release / "artifact_manifest.json").read_text())
    rows = pd.read_csv(release / "tables/gates.csv")
    assert manifest["paper_disposition"] == "benchmark_or_negative_systems_study"
    assert set(rows.status) <= {"pass", "fail", "blocked"}
    assert "fail" in set(rows.status)


def test_blocked_gate_prevents_publication(tmp_path: Path) -> None:
    gates = valid_gates().with_status("shared_packet_representation", "blocked")
    with pytest.raises(PaperReleaseError, match="blocked scientific prerequisite"):
        publish_paper_release(valid_artifact_index(), valid_claim_stats(), gates, RELEASE_ID, tmp_path)


def test_qualitative_rule_includes_locked_failures_not_manual_choices(tmp_path: Path) -> None:
    selection = select_qualitative_panels(validated_images(), rule=locked_rule("median_plus_worst_decile", count=12))
    assert any(panel.failure_case for panel in selection.panels)
    assert len(selection.panels) == 12
```

- [ ] **Step 2: Run the publisher tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_publish.py -q`

Expected: collection fails because `ratemem.evaluation.publish` does not exist.

- [ ] **Step 3: Implement exact normalized paper tables**

Write these required CSVs with exactly these columns and stable sort keys:

```python
MAIN_LIFECYCLE_COLUMNS = [
    "claim_id", "dataset_id", "protocol", "method_id", "comparator_id",
    "budget_label", "budget_bytes", "request_regime", "endpoint", "estimate",
    "ci_low", "ci_high", "margin", "adjusted_p_value", "n_training_seeds",
    "n_inference_units", "gate_status", "artifact_ids_sha256",
]
BASELINE_COMPLIANCE_COLUMNS = [
    "method_id", "citation_key", "source_revision", "backbone", "status",
    "faithfulness_report_sha256", "search_trials", "gpu_hours",
    "state_ledger_pass", "eligible_claims",
]
EFFICIENCY_COLUMNS = [
    "dataset_id", "method_id", "budget_label", "state_bytes", "shared_weight_bytes",
    "amortized_shared_bytes", "insert_latency_p50_ms", "insert_latency_p95_ms",
    "read_latency_p50_ms", "read_latency_p95_ms", "peak_memory_bytes",
    "energy_kwh", "hardware_id", "artifact_ids_sha256",
]
GATE_COLUMNS = [
    "gate_id", "required", "claim_id", "dataset_id", "budget_label",
    "request_regime", "comparator_id", "estimate", "ci_low", "ci_high",
    "margin", "status", "reason_code", "evidence_sha256",
]
```

Validate every `status` as exactly `pass`, `fail`, or `blocked`; never serialize a boolean gate. A `fail` is an evaluated scientific result and remains in the table. A `blocked` row means required evidence was missing and blocks release creation.

Write `curves/quality_bytes.csv` columns `dataset_id,method_id,budget_bytes,request_regime,identity_estimate,identity_ci_low,identity_ci_high,prompt_estimate,prompt_ci_low,prompt_ci_high,utility_estimate,utility_ci_low,utility_ci_high`; write `curves/oracle_regret.csv` columns `dataset_id,method_id,budget_bytes,request_regime,event_index,regret_mean,regret_ci_low,regret_ci_high`; and write `curves/quality_wallclock_energy.csv` columns `dataset_id,method_id,quality_endpoint,quality_estimate,wall_clock_seconds,energy_kwh,optimization_steps,search_gpu_hours,hardware_id,artifact_ids_sha256`. The latter preserves each method's validation-tuned step count rather than forcing equal optimization steps.

- [ ] **Step 4: Implement primary-statistics and human-study JSON**

`stats/primary_claims.json` contains `schema_version`, `release_id`, `evaluation_lock_sha256`, `dataset_lock_sha256`, and `claims`; each claim record contains `claim_id`, `endpoint`, `inference_unit`, `method_id`, `comparator_id`, `cells`, `overall_status`, `multiplicity_family`, and `holm_adjusted`. `human_study/summary.json` contains frozen-response hash, policy hash, included/excluded counts, exclusion reasons, concept-unit paired effects/CIs, and no participant-level rows.

- [ ] **Step 5: Implement preregistered qualitative selection**

`qualitative/selection_manifest.json` contains `schema_version`, `rule_id`, `lock_sha256`, and `panels`. Each panel contains `panel_id`, `dataset_id`, anonymous `concept_token`, `prompt_id`, `seed`, `methods` (`method_id`, relative `image_path`, image SHA-256, source artifact ID), `failure_case`, `selection_score`, and `selected_rank`. Select the locked median cases plus the worst identity-decile failures with paired prompts/seeds; reject a hand-supplied image list.

- [ ] **Step 6: Implement the checksummed release manifest**

`artifact_manifest.json` contains `schema_version`, exact `release_id`, top-level `paper_disposition`, source artifact entries (`artifact_id`, source path, SHA-256, schema status), file entries (relative path, SHA-256, media type, row count), dataset/evaluation/baseline lock hashes, and `gates_sha256`. `paper_disposition` is exactly one of `algorithm_superiority_supported`, `theorem_free_empirical_system`, `benchmark_or_negative_systems_study`, or `negative_result`, derived from evaluated gate records rather than a caller argument. Validate that all numeric cells trace to a set of source artifact IDs whose combined canonical SHA-256 equals `artifact_ids_sha256` and that `tables/gates.csv` is a required checksummed file.

- [ ] **Step 7: Publish the concrete submission release**

```bash
uv run ratemem-eval publish paper \
  --release-id cvpr2027-submission-v1 \
  --artifact-index artifacts/scientific/final/artifact-index.json \
  --claim-statistics artifacts/scientific/final/claim-statistics.json \
  --gates artifacts/scientific/final/gates.json \
  --human-study artifacts/scientific/human-study/analysis/summary.json \
  --output artifacts/paper/cvpr2027-submission-v1
```

Expected: stdout matches `^PASS paper-release cvpr2027-submission-v1: all rows artifact-backed; manifest=[0-9a-f]{64}$`. A scientifically failed gate is published with its disposition; a missing/blocked gate, changed CSV byte, or row without lineage exits 2 and leaves the target directory absent.

- [ ] **Step 8: Generate schema, run tests, and commit**

Run: `uv run ratemem-eval publish schema --output schemas/paper-release.schema.json && uv run pytest tests/unit/evaluation/test_publish.py tests/integration/evaluation/test_paper_release.py -q`

Expected: all tests pass and the synthetic release validates recursively.

```bash
git add src/ratemem/evaluation/publish.py schemas/paper-release.schema.json tests/unit/evaluation/test_publish.py tests/integration/evaluation/test_paper_release.py
git commit -m "feat(eval): publish artifact-backed paper release"
```

### Task 16: Encode the scientific falsification gates and one-time final run

**Files:**
- Create: `src/ratemem/evaluation/gates.py`
- Create: `schemas/scientific-gates.schema.json`
- Create: `schemas/scientific-allocator-guarantee-evidence.schema.json`
- Test: `tests/unit/evaluation/test_gates.py`
- Test: `tests/integration/evaluation/test_final_evaluation.py`

- [ ] **Step 1: Write failing gate truth-table tests**

```python
def test_shared_packet_gate_requires_both_regimes_at_50_and_nonnegative_other_budgets() -> None:
    evidence = packet_evidence(
        at_50={"uniform": ci(0.02, 0.01, 0.03), "zipf": ci(0.03, 0.01, 0.05)},
        other_points={"25pct": 0.0, "75pct": 0.01}, prompt_noninferior=True,
    )
    assert evaluate_shared_packet_gate(evidence).status == GateStatus.PASS
    evidence.at_50["zipf"] = ci(0.01, -0.01, 0.03)
    assert evaluate_shared_packet_gate(evidence).reason_code == "CI_NOT_POSITIVE_ZIPF_50PCT"


def test_empirical_allocator_gate_does_not_depend_on_theorem_status() -> None:
    empirical = allocator_evidence(utility_positive=True, regret_lower=True, quality_noninferior=True)
    theorem = allocator_guarantee_evidence(certified=False)
    assert evaluate_allocator_gate(empirical).status == GateStatus.PASS
    assert evaluate_allocator_guarantee_gate(theorem).status == GateStatus.FAIL


def test_allocator_guarantee_rejects_full_candidate_pool_scope() -> None:
    boundary = allocator_boundary_evidence(
        primary_endpoint="certified_reduced_set_approximation_ratio",
        fixture_id="four_concepts_eight_packets_each_v1",
        proposal_count=32,
        prescreen_input_count=32,
        allocator_input_count=24,
        deterministic_tie_break="lexicographically_larger_packet_id_wins",
    )
    reduced = allocator_guarantee_evidence(
        certified=True,
        comparator_source="exact_reduced_set_optimum",
        ground_set_scope="causal_singleton_density_prescreen_C_t_max24",
        boundary=boundary,
    )
    assert evaluate_allocator_guarantee_gate(reduced).status == GateStatus.PASS
    full_pool = allocator_guarantee_evidence(
        certified=True,
        comparator_source="exact_reduced_set_optimum",
        ground_set_scope="full_G_t",
        boundary=boundary,
    )
    result = evaluate_allocator_guarantee_gate(full_pool)
    assert result.status == GateStatus.FAIL
    assert result.reason_code == "ALLOCATOR_GROUND_SET_SCOPE_MISMATCH"


@pytest.mark.parametrize(
    ("field", "value", "reason_code"),
    [
        ("proposal_count", 31, "ALLOCATOR_BOUNDARY_COUNT_MISMATCH"),
        ("prescreen_input_count", 31, "ALLOCATOR_BOUNDARY_COUNT_MISMATCH"),
        ("allocator_input_count", 25, "ALLOCATOR_BOUNDARY_COUNT_MISMATCH"),
        (
            "deterministic_tie_break",
            "lexicographically_smaller_packet_id_wins",
            "ALLOCATOR_TIE_BREAK_MISMATCH",
        ),
    ],
)
def test_allocator_guarantee_requires_the_locked_32_to_24_boundary(
    field: str, value: object, reason_code: str
) -> None:
    boundary = allocator_boundary_evidence(
        primary_endpoint="certified_reduced_set_approximation_ratio",
        fixture_id="four_concepts_eight_packets_each_v1",
        proposal_count=32,
        prescreen_input_count=32,
        allocator_input_count=24,
        deterministic_tie_break="lexicographically_larger_packet_id_wins",
    ).model_copy(update={field: value})
    evidence = allocator_guarantee_evidence(
        certified=True,
        comparator_source="exact_reduced_set_optimum",
        ground_set_scope="causal_singleton_density_prescreen_C_t_max24",
        boundary=boundary,
    )

    result = evaluate_allocator_guarantee_gate(evidence)

    assert result.status == GateStatus.FAIL
    assert result.reason_code == reason_code


def test_empirical_allocator_gate_fails_only_on_empirical_conditions() -> None:
    evidence = allocator_evidence(utility_positive=True, regret_lower=True, quality_noninferior=False)
    result = evaluate_allocator_gate(evidence)
    assert result.status == GateStatus.FAIL
    assert result.reason_code == "ALLOCATOR_QUALITY_INFERIOR"


def test_failed_theorem_selects_theorem_free_empirical_framing() -> None:
    gates = gate_results(causal_packet_allocator="pass", allocator_guarantee="fail")
    assert paper_disposition(gates) == "theorem_free_empirical_system"


def test_scale_gate_requires_three_seeds_and_two_locked_datasets() -> None:
    result = evaluate_scale_gate(seed_counts={"dreambench_plus_plus": 3, "controlled_post_checkpoint": 2})
    assert result.status == GateStatus.FAIL
    assert result.reason_code == "INSUFFICIENT_SEEDS_CONTROLLED_POST_CHECKPOINT"


def test_missing_evidence_is_blocked_never_passed() -> None:
    assert evaluate_amortizer_gate(None).status == GateStatus.BLOCKED
```

- [ ] **Step 2: Run gate tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_gates.py -q`

Expected: collection fails because `ratemem.evaluation.gates` does not exist.

- [ ] **Step 3: Implement exact core gate predicates**

```python
class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"


from ratemem.evaluation.types import Sha256


class AllocatorBoundaryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    primary_endpoint: Literal["certified_reduced_set_approximation_ratio"]
    fixture_id: Literal["four_concepts_eight_packets_each_v1"]
    proposal_count: NonNegativeInt
    prescreen_input_count: NonNegativeInt
    allocator_input_count: NonNegativeInt
    deterministic_tie_break: str


class AllocatorGuaranteeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    certified: bool
    comparator_source: str
    ground_set_scope: str
    boundary: AllocatorBoundaryEvidence
    controller_boundary_receipt_sha256: Sha256
    proof_artifact_sha256: Sha256
    reduced_set_instance_certificate_sha256: Sha256


class GateResult(BaseModel):
    gate_id: str
    required: bool
    claim_id: str
    status: GateStatus
    reason_code: str
    evidence_sha256: Sha256 | None
    cells: list[GateCell]
```

Implement these predicates exactly:

1. `amortizer`: held-out calibration-disjoint quality CI lower bound meets the locked floor under the locked evaluator revision.
2. `nonseparability`: at least one locked cohort has a packet with dependents from at least two distinct concepts, and the no-sharing paired quality-at-bytes frontier CI is positive rather than only saving metadata bytes.
3. `shared_packet_representation`: against the strongest eligible progressive/shared-subspace/feature-cache control, request-weighted identity paired CI lower bound is positive at `50pct` for both `uniform` and `zipf`; prompt is non-inferior; `25pct` and `75pct` point estimates are non-negative.
4. `causal_packet_allocator`: an independent empirical gate against the strongest eligible causal policy in the same cells; request-weighted utility and oracle-regret improvement CIs must be positive and active quality must be non-inferior. It never reads or conditions on `allocator_guarantee`.
5. `allocator_guarantee`: evidence must declare
   `ground_set_scope=causal_singleton_density_prescreen_C_t_max24` and compare only with the exact
   optimum on that same reduced `C_t`. That scope removes individually infeasible bundles, sorts
   the rest by descending exact singleton density, and retains the highest-density 24 with locked
   ties for which the lexicographically larger packet ID wins. Its schema-validated boundary
   evidence must record `primary_endpoint=certified_reduced_set_approximation_ratio`,
   `fixture_id=four_concepts_eight_packets_each_v1`, `proposal_count=32`,
   `prescreen_input_count=32`, `allocator_input_count=24`, and
   `deterministic_tie_break=lexicographically_larger_packet_id_wins`. A missing field is
   `blocked`; a count mismatch fails with `ALLOCATOR_BOUNDARY_COUNT_MISMATCH`; and any other tie
   rule fails with `ALLOCATOR_TIE_BREAK_MISMATCH`. Hash the validated
   `AllocatorGuaranteeEvidence` into `GateResult.evidence_sha256`; do not reconstruct these counts
   from prose or accept a hand-written gate row. The controller-boundary receipt must be produced by
   the learned-method plan's `four_concept_32_bundle_case`, bind its exact test-command and clean
   implementation revision, and record the proposal, prescreen-call, and allocator-call counts that
   populate the three evidence fields. The mechanically checked proof artifact must match the exact
   locked binary-rational surrogate and assumptions; every exhaustive/random reduced-set instance
   must be feasible; and exact utilities must pass the cross-multiplied rational lower bound
   `6321205588285576 / 10**16`, strictly below `1 - 1/e`, with no additive epsilon. The gate confers
   no ratio against full `G_t`; full-pool pre-screen loss remains empirical. Otherwise the
   theoretical claim is disabled.
6. `optimization_free_tradeoff`: identity and prompt are within frozen non-inferiority margins against matched-backbone `per_concept_lora` and the eligible `dreamcache_feature_cache` control, and the insertion-latency advantage CI exceeds the locked threshold.
7. `autonomous_lookup`: AURC is lower than nearest-key and learned-novelty controls under equal active state.
8. `scale`: the empirical nonseparability, shared-packet, and causal-allocator gates replicate with at least three independent training seeds on DreamBench++ and the controlled post-checkpoint set without consulting theorem status; multi-shot claims additionally require the locked eligible CustomConcept101 cohort. Any scaled theorem language is controlled separately by `allocator_guarantee`.

- [ ] **Step 4: Implement claim survival semantics**

Return both `paper_disposition` and exit status. Failure of `nonseparability` sets `paper_disposition=benchmark_or_negative_systems_study`. When `causal_packet_allocator=pass` and `allocator_guarantee=fail`, preserve the empirical allocator result and set `paper_disposition=theorem_free_empirical_system`; the theorem result cannot change the empirical gate status. Failure of core shared-packet or empirical allocator gates blocks an algorithm-superiority claim. Optional failures exclude that optional section and do not rewrite a core result. A missing artifact is `blocked`, never `fail` or `pass`.

- [ ] **Step 5: Freeze strongest-control and method selection from validation only**

```bash
uv run ratemem-eval baselines select \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --execution-freeze artifacts/scientific/freeze/comparative-execution-freeze.json \
  --validation-index artifacts/scientific/validation/artifact-index.json \
  --validation-statistics artifacts/scientific/validation/claim-statistics.json \
  --output artifacts/scientific/freeze/selected-configuration.json
```

Expected: `^PASS validation-selection: configuration=[0-9a-f]{64}$`. Missing actual shared-input/train/search receipts, a search-budget violation, unreconciled paid phase, stale learned CPU gate, any final-trace reference, or a validation artifact absent from the comparative execution freeze exits 2 and creates no selection.

- [ ] **Step 6: Create and sign the final freeze before opening the trace**

```bash
uv run ratemem-eval final freeze \
  --dataset-lock configs/scientific/dataset-lock.yaml \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml \
  --execution-freeze artifacts/scientific/freeze/comparative-execution-freeze.json \
  --final-envelope configs/scientific/traces/final-test-envelope.json \
  --selected-config artifacts/scientific/freeze/selected-configuration.json \
  --compute-authorization artifacts/scientific/compute/authorization.json \
  --cost-reservation artifacts/scientific/compute/reservation.json \
  --approver-private-key /home/ubuntu/.config/ratemem/final-approval-ed25519.key \
  --output artifacts/scientific/freeze/final-evaluation-permit.json
```

Expected with a dirty worktree, unsigned approval, mutable selection, missing margin, stale/missing comparative execution freeze, unbound search receipt, or validation artifact newer than the selection freeze: exit 2. When frozen, stdout matches `^PASS final-freeze: permit=[0-9a-f]{64}; final trace remains unopened$`.

- [ ] **Step 7: Execute the one-time final replay through a stream**

```bash
uv run ratemem-eval final run \
  --permit artifacts/scientific/freeze/final-evaluation-permit.json \
  --envelope configs/scientific/traces/final-test-envelope.json \
  --private-key /home/ubuntu/.config/ratemem/final-trace-x25519.key \
  --compute-authorization artifacts/scientific/compute/authorization.json \
  --cost-reservation artifacts/scientific/compute/reservation.json \
  --launch-receipt artifacts/scientific/compute/final-launch-receipt.json \
  --ledger artifacts/scientific/final-evaluation/final-open-ledger.json \
  --output artifacts/scientific/final
```

Expected: the scientific compute permit is consumed before any paid provider call, then the final-open ledger is atomically written before decryption; progress identifies only trace hashes, never event payloads; completion stdout matches `^PASS final-evaluation: trace opened once; artifacts=[0-9]+$`. Any pilot-scope permit or second invocation exits 2 before a new allocation; the final trace remains one-use even if the first invocation failed after opening.

- [ ] **Step 8: Compute gates and release the decrypted manifest for reproducibility**

After the one-time run, write the complete plaintext final trace manifest/payload into the checksummed final artifact release (not a training path), then run:

```bash
uv run ratemem-eval gates evaluate \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --artifact-index artifacts/scientific/final/artifact-index.json \
  --statistics artifacts/scientific/final/claim-statistics.json \
  --output artifacts/scientific/final/gates.json
```

Expected: one explicit `pass`, `fail`, or `blocked` row per prespecified gate and `paper_disposition` printed. The command exits 0 when every required gate was evaluated to `pass` or `fail`, and exits 2 when any required gate is `blocked`; failed scientific hypotheses remain valid evaluated outputs and are not deleted.

- [ ] **Step 9: Generate schemas, run tests, and commit code only**

Run: `uv run ratemem-eval gates schema --output schemas/scientific-gates.schema.json && uv run ratemem-eval gates evidence-schema --output schemas/scientific-allocator-guarantee-evidence.schema.json && uv run pytest tests/unit/evaluation/test_gates.py tests/integration/evaluation/test_final_evaluation.py -q`

Expected: both schemas are generated from the exact Pydantic models; the allocator evidence schema
requires the endpoint, fixture, three counts, deterministic tie rule, controller-boundary receipt,
proof, and reduced-set certificate. All tests pass, including atomic second-open rejection and
every gate boundary.

```bash
git add src/ratemem/evaluation/gates.py schemas/scientific-gates.schema.json schemas/scientific-allocator-guarantee-evidence.schema.json tests/unit/evaluation/test_gates.py tests/integration/evaluation/test_final_evaluation.py
git commit -m "feat(eval): enforce scientific falsification gates"
```

Never commit the private keys. Commit/publicly archive the final permit, one-time ledger, released final trace, statistics, and gates only after checking their disclosure content.

### Task 17: Gate the optional category-augmentation track behind core evidence

**Files:**
- Create: `configs/scientific/augmentation-policy.yaml`
- Create: `src/ratemem/evaluation/augmentation.py`
- Create: `schemas/scientific-augmentation-lock.schema.json`
- Test: `tests/unit/evaluation/test_augmentation.py`
- Test: `tests/integration/evaluation/test_augmentation_gate.py`

- [ ] **Step 1: Prespecify eligible datasets, class-disjointness, and controls**

```yaml
# configs/scientific/augmentation-policy.yaml
schema_version: "1.0"
run_requires_core_gates:
  - nonseparability
  - shared_packet_representation
  - causal_packet_allocator
  - scale
candidate_datasets:
  - {id: oxford_flowers102, standard_split_allowed: false}
  - {id: cub200_2011, standard_split_allowed: false}
  - {id: nabirds, standard_split_allowed: false}
  - {id: animal_faces_eligible, standard_split_allowed: false}
required_controls:
  - real_only
  - random_oversampling
  - standard_image_augmentation
  - class_name_only_frozen_generator
primary_endpoint: held_out_accuracy
required_secondary_endpoint: worst_class_accuracy
inference_unit: independent_class_support_split
confidence_level: 0.95
gate_rule: positive_paired_ci_on_both_endpoints_for_every_locked_dataset
```

The separate augmentation lock includes only sources with verified licenses/provenance and pins the preselected primary dataset set before any generated-example comparison. If no eligible set is locked, the optional track remains absent rather than changing datasets after seeing results.

- [ ] **Step 2: Write failing prerequisite, class split, and all-dataset tests**

```python
def test_augmentation_cannot_start_before_all_core_gates_pass() -> None:
    gates = core_gates(causal_packet_allocator="fail")
    with pytest.raises(OptionalTrackBlocked, match="causal_packet_allocator"):
        authorize_augmentation(gates, AUGMENTATION_POLICY)


def test_standard_oxford_split_is_rejected_when_classes_overlap() -> None:
    with pytest.raises(ClassSplitLeakageError):
        validate_class_disjoint_split(train_classes={1, 2}, validation_classes={2, 3}, test_classes={4})


def test_gate_requires_positive_mean_and_worst_class_ci_on_every_dataset() -> None:
    evidence = augmentation_evidence(oxford=(0.03, 0.01), cub=(0.02, -0.01))
    result = evaluate_augmentation_gate(evidence)
    assert result.status == GateStatus.FAIL
    assert result.reason_code == "WORST_CLASS_CI_NOT_POSITIVE_CUB200_2011"
```

- [ ] **Step 3: Run augmentation tests and verify failure**

Run: `uv run pytest tests/unit/evaluation/test_augmentation.py -q`

Expected: collection fails because `ratemem.evaluation.augmentation` does not exist.

- [ ] **Step 4: Implement the separate augmentation lock and authorization**

Define `AugmentationLock` with dataset revisions/licenses, duplicate-report hash, class-disjoint train/validation/test class IDs and hashes, frozen classifier architecture/revision, support sizes, split seeds, generated images per class, all four control configurations, training budgets, primary/secondary endpoints, bootstrap settings, and approvals. `authorize_augmentation` loads the artifact-backed core gate file and refuses to create the lock or launch generation unless every prerequisite is `pass`.

- [ ] **Step 5: Implement matched control execution and inference rows**

Define `AugmentationAdapter` operations `fit(split, seed)`, `augment(train_set, class_id, count, seed)`, and `evaluate(test_set)`. Use the same classifier initialization, optimizer steps, early-stopping rule, image count, and split across real-only, oversampling, standard augmentation, class-name-only frozen generator, and RateMem augmentation. Emit paired rows at the independent class/support split, with per-class accuracy retained for the worst-class estimand.

- [ ] **Step 6: Implement the all-dataset optional gate**

For each dataset, compare RateMem to the strongest validation-selected required control. Require the paired 95% CI lower bound to be positive for held-out mean accuracy and worst-class accuracy on every locked primary dataset. Apply Holm correction to secondary classifiers/support sizes. A failure produces `exclude_optional_augmentation`; it can never be averaged away by a favorable dataset.

- [ ] **Step 7: Add commands and verify no bypass**

```bash
uv run ratemem-eval augmentation lock \
  --core-gates artifacts/scientific/final/gates.json \
  --policy configs/scientific/augmentation-policy.yaml \
  --dataset-inventory artifacts/scientific/augmentation/dataset-inventory.json \
  --split-manifests artifacts/scientific/augmentation/splits \
  --output configs/scientific/augmentation-lock.yaml
uv run ratemem-eval augmentation run \
  --lock configs/scientific/augmentation-lock.yaml \
  --output artifacts/scientific/augmentation
```

Expected in the test fixture before core passes: exit 2 with `BLOCKED optional-augmentation: core gate causal_packet_allocator is not pass`. Expected after a valid lock/run: `PASS optional-augmentation evaluation complete`; the subsequent gate result may still be pass or fail.

- [ ] **Step 8: Generate schema, run tests, and commit**

Run: `uv run ratemem-eval augmentation schema --output schemas/scientific-augmentation-lock.schema.json && uv run pytest tests/unit/evaluation/test_augmentation.py tests/integration/evaluation/test_augmentation_gate.py -q`

Expected: all tests pass, including rejection of the standard non-class-disjoint Oxford Flowers split.

```bash
git add configs/scientific/augmentation-policy.yaml src/ratemem/evaluation/augmentation.py schemas/scientific-augmentation-lock.schema.json tests/unit/evaluation/test_augmentation.py tests/integration/evaluation/test_augmentation_gate.py
git commit -m "feat(eval): gate optional category augmentation"
```

### Task 18: Wire the full synthetic scientific pipeline into CI and document the operator sequence

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `tests/integration/evaluation/test_scientific_pipeline.py`
- Create: `docs/scientific-evaluation.md`

- [ ] **Step 1: Write the failing end-to-end synthetic test**

The test uses only `tests/fixtures/scientific/`, generates a dataset lock/card, runs the duplicate audit, builds disjoint traces, seals a synthetic final trace, validates a synthetic provider-contract baseline audit, freezes synthetic baseline/evaluation locks, creates and validates a zero-provider learned CPU receipt, creates an explicit synthetic workspace selection with a USD 28.00 cap, reserves known plus pending plus new cost at exactly USD 27.00, rejects pilot scope and full-scientific authorization without the CPU receipt, freezes synthetic shared-input/train/search receipts, consumes the scientific permit once, opens the final trace once, replays two deterministic fake methods, validates artifacts, bootstraps paired rows, exports/imports a blinded mini-study, evaluates gates, and publishes a paper release. Assert:

```python
def test_synthetic_pipeline_is_reproducible_and_artifact_only(tmp_path: Path) -> None:
    first = run_synthetic_scientific_pipeline(tmp_path / "first")
    second = run_synthetic_scientific_pipeline(tmp_path / "second")
    assert first.semantic_release_sha256 == second.semantic_release_sha256
    assert first.dataset_lock_id == second.dataset_lock_id
    assert first.trace_commitments == second.trace_commitments
    assert first.final_open_count == second.final_open_count == 1
    assert first.scientific_compute_permit_consumptions == second.scientific_compute_permit_consumptions == 1
    assert first.engineering_pilot_permit_accepted is False
    assert first.method_cpu_gate_validated
    assert first.comparative_execution_freeze_validated
    assert first.reserved_total_usd == Decimal("27.00")
    assert first.pending_cost_after_reconciliation_usd == Decimal("0.00")
    assert first.paper_release_valid
    assert first.unvalidated_input_count == 0
```

- [ ] **Step 2: Run the integration test and verify it initially fails at the first unwired command**

Run: `uv run pytest tests/integration/evaluation/test_scientific_pipeline.py -q -x`

Expected: FAIL with the first missing CLI registration under `ratemem-eval`.

- [ ] **Step 3: Register every subcommand in the Typer root**

Mount named sub-apps `data`, `traces`, `lock`, `baselines`, `compute`, `replay`, `artifacts`, `stats`, `human-study`, `gates`, `augmentation`, `final`, and `publish`. Each command catches only declared domain errors, prints `PASS`, `FAIL`, or `BLOCKED`, uses exit 2 for invalid/blocked input, and never catches unexpected exceptions as successful evidence.

- [ ] **Step 4: Add the CI scientific contract job**

Add a job that runs, in order:

```bash
uv sync --all-extras --frozen
uv run ruff check src/ratemem/evaluation tests/unit/evaluation tests/contract/evaluation tests/integration/evaluation
uv run mypy src/ratemem/evaluation
uv run pytest tests/unit/evaluation tests/contract/evaluation -q
uv run pytest tests/integration/evaluation/test_scientific_pipeline.py -q
git diff --exit-code -- schemas
```

Expected: all commands exit 0; schema drift or a non-deterministic synthetic release fails the job.

- [ ] **Step 5: Document the exact scientific operator sequence**

In `docs/scientific-evaluation.md`, list these phases and the corresponding commands from this plan: core plan; SANA/pilot plan; scientific Tasks 1--7; Task 8 registry plus narrow baseline-fidelity authorization; matched-baselines implementation/audit; baseline/evaluation sealing; learned-method CPU gate; Task 9 full authorization; real shared-input/train/search receipt freeze; validation replay and immediate cost reconciliation; strongest-control selection; final freeze; one-time final stream replay and immediate cost reconciliation; human study; statistics/Holm; gates; optional augmentation authorization; paper release. State explicitly that no comparative result is valid before dataset/evaluation locks, no full paid authorization exists without the current zero-provider learned CPU receipt, no model-selection code can open final events, no paid scientific launch may use an engineering-pilot authorization, and no manuscript number may bypass `artifact_manifest.json`.

- [ ] **Step 6: Run the full scientific suite and inspect CLI help**

Run: `uv run pytest tests/unit/evaluation tests/contract/evaluation tests/integration/evaluation -q && uv run ratemem-eval --help`

Expected: all tests pass; help lists all thirteen sub-apps and does not expose a command that writes final-test plaintext to a caller-selected repository path or launches paid scientific compute without an explicit permit.

- [ ] **Step 7: Run repository-wide checks**

Run: `uv run ruff check src tests && uv run mypy src/ratemem && uv run pytest -q`

Expected: Ruff/mypy exit 0 and the complete repository test suite passes.

- [ ] **Step 8: Commit CI and operator documentation**

```bash
git add .github/workflows/ci.yml src/ratemem/evaluation/cli.py tests/integration/evaluation/test_scientific_pipeline.py docs/scientific-evaluation.md
git commit -m "test(eval): verify locked scientific pipeline end to end"
```

## Final acceptance checklist

- [ ] `dataset-lock.yaml` and the generated data card bind exact revisions, licenses, concept units, counts, statistics, annotations, immutable pools, global duplicate components, and contamination disclosures.
- [ ] Train, validation, and final-test concept pools, trace IDs, prompt templates, and seed namespaces are pairwise disjoint; the final payload was committed encrypted before comparative development.
- [ ] The final-test opening ledger proves a single signed stream opening after dataset, baseline, evaluation, learned CPU, comparative-execution, configuration, and margin freeze.
- [ ] `evaluation-lock.yaml` binds evaluator/preprocessing revisions, formulas, margins with calibration provenance, exact byte budgets, both request regimes, trace hashes, generation/latency settings, power/CI target, human-study rules, and Holm families.
- [ ] Every paid phase names one operator-selected workspace, verifies its exact USD 28.00 outer cap, reserves known plus all pending plus new cost at no more than USD 27.00 in one aggregate ledger, consumes one scope-bound permit, reconciles immediately afterward, and contains no credential material; pre-lock `baseline_fidelity` cannot access validation/final inputs, full `scientific` requires the current zero-provider learned CPU receipt, and engineering-pilot authorization or workspace reuse/rotation is rejected.
- [ ] Every primary closest-work control passes fidelity/state-ledger checks on SANA-1.5; any failure blocks the affected primary claim and evaluation lock, while SDXL-native numbers remain contextual-only.
- [ ] Probes leave usage, bytes, handles, references, and state digests unchanged; all methods receive paired prompt/support/noise/event commitments.
- [ ] Storage, latency, lifecycle, lookup, deletion, diversity, and oracle-regret results come from validated artifacts with no per-concept small-sample FID.
- [ ] Paired hierarchical inference counts deployment episodes/concepts/pairs/splits as declared and treats prompts/images as nested observations.
- [ ] The blinded human export contains no method labels or metadata leakage and is unblinded only after response freeze.
- [ ] Core gates emit explicit pass/fail/blocked evidence; failed hypotheses change the paper disposition rather than disappearing.
- [ ] Optional augmentation cannot launch before core gates pass and must win on mean and worst-class accuracy on every preselected locked dataset.
- [ ] `artifacts/paper/cvpr2027-submission-v1/artifact_manifest.json` recursively checksums every table, curve, statistic, human summary, qualitative panel, and source artifact used by the manuscript.
