# RateMem-DiT CVPR Manuscript, Figures, and Overleaf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a fully rewritten, evidence-gated English CVPR manuscript, reproducible vector and qualitative figures, an eight-page anonymous submission PDF, and a verified copy in the local Overleaf instance without modifying the original 2020 project.

**Architecture:** The paper source is a clean `paper/` tree built on the pinned official CVPR 2026 release while CVPR 2027 materials are unavailable; a currency check blocks submission when a newer official kit appears. A fail-closed paper-release reader accepts only schema-valid, checksummed scientific artifacts and generates every empirical macro, table, curve, and qualitative panel into a disposable build tree. Compliance and visual-QA receipts bind the exact PDF, after which an authenticated sync command downloads and verifies the existing Overleaf project as a backup, imports the new paper as a separate project, recompiles it, and proves the original project contents did not change.

**Tech Stack:** Python 3.11, `uv`, JSON Schema, PyYAML, Matplotlib, Pillow, Pybtex, Requests, official CVPR 2026 LaTeX (`CVPR2026-v1(latex)`), pdfLaTeX/latexmk from the local Overleaf 6.2.2 image, TikZ/PGF, Poppler, qpdf, pytest.

---

## Execution boundary

Follow `2026-08-24-ratemem-master-execution.md`: execute the core-memory and free SANA/pilot work,
seal scientific-evaluation Tasks 1--7, implement and audit
`2026-08-24-ratemem-matched-baselines.md`, seal scientific Task 8, and then execute
`2026-08-24-ratemem-learned-method-training.md` before comparative replay.
The resulting prerequisites are `pyproject.toml`, `uv.lock`, `src/ratemem/`,
`configs/scientific/dataset-lock.yaml`, `configs/scientific/evaluation-lock.yaml`,
`configs/scientific/baseline-lock.yaml`, and `schemas/paper-release.schema.json`. Draft-mode
manuscript work may begin before a scientific release exists, but `make -C paper submission` must
remain impossible until `artifacts/paper/cvpr2027-submission-v1/artifact_manifest.json` and every
referenced file validate.

The historical source at `/home/ubuntu/memory-metagan-original/` and local Overleaf project `6a8b44fb070db27221ef64a0` (`Memory-MetaGAN: A Memory-based Few-shot GAN`) are read-only provenance. Do not copy the historical `cvpr.sty`, patch `memory-based-metagan.tex`, reuse its unverified bibliography, or reuse its raster architecture figures. The new project tells the RateMem-DiT story from a blank source tree.

This is a complete English rewrite, not a translation or incremental revision of the 2020 paper. Treat every checkbox as one 2--5 minute action. When a checkbox says to repeat over sources, queries, panels, or PDF pages, complete and record exactly one item, then repeat that same checkbox for the next item; never batch the repeated items into one unchecked action.

As of 2026-08-24, the latest published official kit is `CVPR2026-v1(latex)` at commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`; CVPR 2026 limits the paper body, including figures and tables, to eight pages and permits additional reference-only pages. The final submission gate must query the official author-kit releases and CVPR site again, because CVPR 2027 rules supersede this fallback immediately when published.

## Locked file map

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Add the `paper` dependency extra and `ratemem-paper` entry point. |
| `src/ratemem/paper/release.py` | Validate paper-release schema, paths, checksums, row counts, locks, and required gates. |
| `src/ratemem/paper/render.py` | Generate TeX macros/tables and vector result plots only from a validated release. |
| `src/ratemem/paper/qualitative.py` | Validate the preregistered selection manifest and compose unretouched qualitative panels. |
| `src/ratemem/paper/literature.py` | Enforce primary-source evidence and bibliography coverage. |
| `src/ratemem/paper/compliance.py` | Check source/PDF anonymity, page count, links, prompt injection, fonts, metadata, and provenance. |
| `src/ratemem/paper/overleaf.py` | Authenticate, back up, import as a new project, compile, and compare canonical project manifests. |
| `src/ratemem/paper/cli.py` | Expose deterministic `template`, `release`, `figures`, `audit`, and `overleaf` commands. |
| `paper/main.tex`, `paper/preamble.tex` | Official review-mode entry point and minimal package/macro definitions. |
| `paper/sections/*.tex` | Complete new English main-paper prose, one scientific responsibility per file. |
| `paper/supplement.tex`, `paper/supplement/*.tex` | Proof, protocol, baseline, reproducibility, and extended qualitative material. |
| `paper/cvpr.sty`, `paper/ieeenat_fullname.bst` | Byte-verified, unmodified official CVPR release files. |
| `paper/template/CVPR_TEMPLATE_SOURCE.json` | Release tag, commit, source URLs, date checked, and SHA-256 pins. |
| `paper/config/cvpr-policy.yaml` | Machine-readable current page/anonymity/ethics/link policy and target-year recheck rule. |
| `paper/related_work/required_sources.yaml` | Complete closest-work and baseline citation-key inventory. |
| `paper/related_work/evidence.yaml` | Human-verified claims, primary-source locators, source hashes, and BibTeX metadata. |
| `paper/claims.yaml` | Claim-to-section, artifact-cell, gate, comparator, and permitted wording map. |
| `paper/figures/vector/*.tex` | TikZ sources for overview, byte ledger, and lifecycle timeline. |
| `paper/generated/` | Disposable generated macros/tables/plots/panels; never hand edited. |
| `paper/build/` | PDFs, logs, rendered QA pages, and signed-off compliance receipts. |
| `schemas/paper-release.schema.json` | Scientific publisher contract owned by the evaluation plan and consumed read-only here. |
| `artifacts/paper/cvpr2027-submission-v1/` | Exact validated scientific release consumed by submission mode. |
| `tests/unit/paper/`, `tests/contract/paper/`, `tests/integration/paper/` | Focused release, render, literature, compliance, and Overleaf safety tests. |

### Task 1: Pin the current official CVPR kit and paper toolchain

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.gitignore`
- Create: `src/ratemem/paper/__init__.py`
- Create: `src/ratemem/paper/cli.py`
- Create: `paper/template/CVPR_TEMPLATE_SOURCE.json`
- Create: `paper/config/cvpr-policy.yaml`
- Create: `paper/cvpr.sty`
- Create: `paper/ieeenat_fullname.bst`
- Test: `tests/contract/paper/test_cvpr_template.py`

- [ ] **Step 1: Add the pinned paper dependencies and CLI**

Merge this extra and entry point into the existing `pyproject.toml`; preserve all existing SANA, core, and scientific extras:

```toml
[project.optional-dependencies]
paper = [
  "jsonschema==4.25.1",
  "matplotlib==3.10.5",
  "Pillow==11.3.0",
  "pybtex==0.25.1",
  "pypdf==6.0.0",
  "PyYAML==6.0.2",
  "requests==2.32.5",
]

[project.scripts]
ratemem-paper = "ratemem.paper.cli:main"
```

Add these repository-local outputs to `.gitignore` without weakening the scientific artifact rules:

```gitignore
paper/build/
paper/generated/
paper/figures/rendered/
paper/ratemem-overleaf.zip
paper/*.aux
paper/*.bbl
paper/*.blg
paper/*.fdb_latexmk
paper/*.fls
paper/*.log
```

Run: `uv lock && uv sync --all-extras --frozen`

Expected: exit 0 and `uv run python -c 'import jsonschema, matplotlib, PIL, pybtex, pypdf, requests, yaml'` exits 0.

- [ ] **Step 2: Write the failing official-template contract test**

```python
# tests/contract/paper/test_cvpr_template.py
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[3]
EXPECTED = {
    "cvpr.sty": "2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153",
    "ieeenat_fullname.bst": "e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cvpr_2026_release_is_pinned_without_style_edits() -> None:
    source = json.loads((ROOT / "paper/template/CVPR_TEMPLATE_SOURCE.json").read_text())
    assert source["release_tag"] == "CVPR2026-v1(latex)"
    assert source["commit"] == "12909ae437f6dbc7435069cfdb4ca44c18e6a02f"
    assert source["target_venue"] == "CVPR 2027"
    assert source["fallback_reason"] == "CVPR 2027 author kit was not published on 2026-08-24"
    for name, expected in EXPECTED.items():
        assert source["files"][name]["sha256"] == expected
        assert sha256(ROOT / "paper" / name) == expected
```

- [ ] **Step 3: Run the contract test and verify the expected failure**

Run: `uv run pytest tests/contract/paper/test_cvpr_template.py -q`

Expected: FAIL because `paper/template/CVPR_TEMPLATE_SOURCE.json` and the official style files do not exist.

- [ ] **Step 4: Vendor the exact official release files and source lock**

Download only the two files used by the manuscript from the immutable commit:

```bash
mkdir -p paper/template paper/config
curl --fail --silent --show-error --location \
  https://raw.githubusercontent.com/cvpr-org/author-kit/12909ae437f6dbc7435069cfdb4ca44c18e6a02f/cvpr.sty \
  --output paper/cvpr.sty
curl --fail --silent --show-error --location \
  https://raw.githubusercontent.com/cvpr-org/author-kit/12909ae437f6dbc7435069cfdb4ca44c18e6a02f/ieeenat_fullname.bst \
  --output paper/ieeenat_fullname.bst
sha256sum paper/cvpr.sty paper/ieeenat_fullname.bst
```

Expected hashes, in order:

```text
2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153  paper/cvpr.sty
e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc  paper/ieeenat_fullname.bst
```

Create the exact source lock:

```json
{
  "schema_version": "1.0",
  "target_venue": "CVPR 2027",
  "fallback_template": "CVPR 2026",
  "fallback_reason": "CVPR 2027 author kit was not published on 2026-08-24",
  "release_tag": "CVPR2026-v1(latex)",
  "commit": "12909ae437f6dbc7435069cfdb4ca44c18e6a02f",
  "checked_at_utc": "2026-08-24T00:00:00Z",
  "repository": "https://github.com/cvpr-org/author-kit",
  "release_url": "https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1(latex)",
  "author_guidelines": "https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines",
  "files": {
    "cvpr.sty": {
      "sha256": "2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153"
    },
    "ieeenat_fullname.bst": {
      "sha256": "e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc"
    }
  }
}
```

- [ ] **Step 5: Encode the current CVPR submission policy**

```yaml
# paper/config/cvpr-policy.yaml
schema_version: "1.0"
target_venue: CVPR 2027
active_fallback_year: 2026
official_guidelines: https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines
official_author_kit: https://github.com/cvpr-org/author-kit
body_page_limit: 8
references_may_follow_body: true
review_mode: double_blind
external_content_links_allowed: false
prompt_injection_allowed: false
personal_or_human_subject_data_requires_disclosure: true
submission_pdf_must_be_self_contained: true
currency_rule: block_when_a_newer_official_cvpr_release_or_target_year_guideline_exists
```

Implement `ratemem-paper template verify` so it checks the local hashes, queries GitHub releases, probes the official target-year author-guideline URL, and exits 2 when an official CVPR 2027 kit or guideline appears. Network failure is a blocked submission check, not a pass.

Run: `uv run ratemem-paper template verify --source paper/template/CVPR_TEMPLATE_SOURCE.json --policy paper/config/cvpr-policy.yaml`

Expected on 2026-08-24: `PASS template: CVPR2026-v1(latex) is the latest published kit; CVPR 2027 rules not detected`.

- [ ] **Step 6: Run the tests and commit the immutable template boundary**

Run: `uv run pytest tests/contract/paper/test_cvpr_template.py -q`

Expected: `1 passed`.

```bash
git add pyproject.toml uv.lock .gitignore src/ratemem/paper paper/template paper/config paper/cvpr.sty paper/ieeenat_fullname.bst tests/contract/paper/test_cvpr_template.py
git commit -m "build(paper): pin official CVPR author kit"
```

### Task 2: Build the fail-closed scientific release reader

**Files:**
- Create: `src/ratemem/paper/release.py`
- Create: `tests/unit/paper/test_release.py`
- Create: `tests/contract/paper/test_release_schema.py`

- [ ] **Step 1: Write failing tests for path, checksum, row-count, and gate validation**

Use a temporary synthetic release created by the test itself; mark its manifest `fixture_only: true` so no scientific command can publish or consume it outside tests.

```python
# tests/unit/paper/test_release.py
from pathlib import Path

import pytest

from ratemem.paper.release import PaperReleaseError, load_paper_release


def test_valid_release_is_loaded(valid_paper_release: Path, paper_schema: Path) -> None:
    release = load_paper_release(valid_paper_release, paper_schema, allow_fixture=True)
    assert release.release_id == "fixture-release"
    assert release.required_gate_ids == {
        "amortizer",
        "nonseparability",
        "shared_packet_representation",
        "causal_packet_allocator",
        "allocator_guarantee",
        "optimization_free_tradeoff",
        "scale",
    }


def test_tampered_file_is_rejected(valid_paper_release: Path, paper_schema: Path) -> None:
    target = valid_paper_release / "tables/main_lifecycle.csv"
    target.write_text(target.read_text() + "tampered\n")
    with pytest.raises(PaperReleaseError, match="checksum mismatch"):
        load_paper_release(valid_paper_release, paper_schema, allow_fixture=True)


def test_evaluated_failed_gate_is_loaded_for_negative_framing(valid_paper_release: Path, paper_schema: Path) -> None:
    gates = valid_paper_release / "tables/gates.csv"
    gates.write_text(gates.read_text().replace(",pass,passed,", ",fail,ci_crosses_zero,"))
    refresh_manifest_hash(valid_paper_release, "tables/gates.csv")
    release = load_paper_release(valid_paper_release, paper_schema, allow_fixture=True)
    assert release.paper_disposition == "benchmark_or_negative_systems_study"


def test_blocked_required_gate_blocks_submission(valid_paper_release: Path, paper_schema: Path) -> None:
    gates = valid_paper_release / "tables/gates.csv"
    gates.write_text(gates.read_text().replace(",pass,passed,", ",blocked,missing_artifact,"))
    refresh_manifest_hash(valid_paper_release, "tables/gates.csv")
    with pytest.raises(PaperReleaseError, match="required gate blocked"):
        load_paper_release(valid_paper_release, paper_schema, allow_fixture=True)


def test_parent_traversal_is_rejected(valid_paper_release: Path, paper_schema: Path) -> None:
    rewrite_manifest_path(valid_paper_release, "tables/main_lifecycle.csv", "../outside.csv")
    with pytest.raises(PaperReleaseError, match="unsafe release path"):
        load_paper_release(valid_paper_release, paper_schema, allow_fixture=True)
```

- [ ] **Step 2: Run the release tests and verify the expected failure**

Run: `uv run pytest tests/unit/paper/test_release.py tests/contract/paper/test_release_schema.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'ratemem.paper.release'`.

- [ ] **Step 3: Implement the immutable release model and exact headers**

```python
# src/ratemem/paper/release.py
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REQUIRED_FILES = {
    "tables/main_lifecycle.csv",
    "tables/baseline_compliance.csv",
    "tables/efficiency.csv",
    "tables/gates.csv",
    "curves/quality_bytes.csv",
    "curves/oracle_regret.csv",
    "curves/quality_wallclock_energy.csv",
    "stats/primary_claims.json",
    "human_study/summary.json",
    "qualitative/selection_manifest.json",
}

CSV_HEADERS = {
    "tables/main_lifecycle.csv": "claim_id,dataset_id,protocol,method_id,comparator_id,budget_label,budget_bytes,request_regime,endpoint,estimate,ci_low,ci_high,margin,adjusted_p_value,n_training_seeds,n_inference_units,gate_status,artifact_ids_sha256".split(","),
    "tables/baseline_compliance.csv": "method_id,citation_key,source_revision,backbone,status,faithfulness_report_sha256,search_trials,gpu_hours,state_ledger_pass,eligible_claims".split(","),
    "tables/efficiency.csv": "dataset_id,method_id,budget_label,state_bytes,shared_weight_bytes,amortized_shared_bytes,insert_latency_p50_ms,insert_latency_p95_ms,read_latency_p50_ms,read_latency_p95_ms,peak_memory_bytes,energy_kwh,hardware_id,artifact_ids_sha256".split(","),
    "tables/gates.csv": "gate_id,required,claim_id,dataset_id,budget_label,request_regime,comparator_id,estimate,ci_low,ci_high,margin,status,reason_code,evidence_sha256".split(","),
    "curves/quality_bytes.csv": "dataset_id,method_id,budget_bytes,request_regime,identity_estimate,identity_ci_low,identity_ci_high,prompt_estimate,prompt_ci_low,prompt_ci_high,utility_estimate,utility_ci_low,utility_ci_high".split(","),
    "curves/oracle_regret.csv": "dataset_id,method_id,budget_bytes,request_regime,event_index,regret_mean,regret_ci_low,regret_ci_high".split(","),
    "curves/quality_wallclock_energy.csv": "dataset_id,method_id,quality_endpoint,quality_estimate,wall_clock_seconds,energy_kwh,optimization_steps,search_gpu_hours,hardware_id,artifact_ids_sha256".split(","),
}


class PaperReleaseError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PaperRelease:
    root: Path
    manifest: dict[str, Any]
    rows: dict[str, list[dict[str, str]]]

    @property
    def release_id(self) -> str:
        return str(self.manifest["release_id"])

    @property
    def required_gate_ids(self) -> set[str]:
        return {
            row["gate_id"]
            for row in self.rows["tables/gates.csv"]
            if row["required"].lower() == "true"
        }

    @property
    def paper_disposition(self) -> str:
        return str(self.manifest["paper_disposition"])


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise PaperReleaseError(f"unsafe release path: {relative}")
    return candidate


def load_paper_release(root: Path, schema_path: Path, *, allow_fixture: bool = False) -> PaperRelease:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.path))
    if errors:
        raise PaperReleaseError(f"manifest schema failure: {errors[0].message}")
    if manifest.get("fixture_only") and not allow_fixture:
        raise PaperReleaseError("fixture release cannot be used for a paper build")
    if root.name != manifest["release_id"] and not allow_fixture:
        raise PaperReleaseError("release directory and release_id differ")
    declared = {item["path"]: item for item in manifest["files"]}
    missing = REQUIRED_FILES - declared.keys()
    if missing:
        raise PaperReleaseError(f"missing paper files: {sorted(missing)}")
    rows: dict[str, list[dict[str, str]]] = {}
    for relative, record in declared.items():
        path = _safe_child(root, relative)
        if file_sha256(path) != record["sha256"]:
            raise PaperReleaseError(f"checksum mismatch: {relative}")
        if relative.endswith(".csv"):
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != CSV_HEADERS[relative]:
                    raise PaperReleaseError(f"header mismatch: {relative}")
                rows[relative] = list(reader)
            if len(rows[relative]) != record["row_count"]:
                raise PaperReleaseError(f"row count mismatch: {relative}")
    bad_sources = [item["artifact_id"] for item in manifest["source_artifacts"] if item["schema_status"] != "valid"]
    if bad_sources:
        raise PaperReleaseError(f"invalid source artifacts: {bad_sources}")
    invalid_status = [row for row in rows["tables/gates.csv"] if row["status"] not in {"pass", "fail", "blocked"}]
    if invalid_status:
        raise PaperReleaseError(f"invalid gate status: {invalid_status[0]['gate_id']}")
    blocked = [row for row in rows["tables/gates.csv"] if row["required"].lower() == "true" and row["status"] == "blocked"]
    if blocked:
        raise PaperReleaseError(f"required gate blocked: {blocked[0]['gate_id']}")
    if manifest.get("paper_disposition") not in {
        "algorithm_superiority_supported", "theorem_free_empirical_system",
        "benchmark_or_negative_systems_study", "negative_result",
    }:
        raise PaperReleaseError("missing or invalid paper_disposition")
    return PaperRelease(root=root, manifest=manifest, rows=rows)
```

- [ ] **Step 4: Validate the non-CSV contracts and locked identifiers**

Extend `load_paper_release` to require:

```python
LOCKED_CLAIM_IDS = {
    "shared_packet_representation",
    "causal_packet_allocator",
    "allocator_guarantee",
    "optimization_free_tradeoff",
}
LOCKED_BUDGET_LABELS = {"25pct", "50pct", "75pct"}
LOCKED_REQUEST_REGIMES = {"uniform", "zipf"}
SHARED_PACKET_ENDPOINT = "request_weighted_identity"
ZIPF_EXPONENT = 1.2
```

Parse `stats/primary_claims.json` and require `schema_version`, matching `release_id`, both lock hashes, all locked claims, inference units, Holm status, and complete primary cells. Parse `qualitative/selection_manifest.json` and require `schema_version`, `rule_id`, `lock_sha256`, and `panels`. Verify every `image_path` is safe, appears in `artifact_manifest.files`, and matches `image_sha256`. Verify `artifact_manifest.locks` exactly match `configs/scientific/dataset-lock.yaml` and `configs/scientific/evaluation-lock.yaml` by SHA-256.

- [ ] **Step 5: Generate and contract-test the publisher schema**

The scientific publisher owns `schemas/paper-release.schema.json`. This test must load its valid and invalid examples and must not regenerate or weaken it from the paper package.

Run: `uv run pytest tests/unit/paper/test_release.py tests/contract/paper/test_release_schema.py -q`

Expected: all release checks pass; tampered paths/files/row counts/locks/identifiers and blocked or
malformed required gates fail with a field-specific message, while evaluated failed hypotheses load
with their negative `paper_disposition` intact.

- [ ] **Step 6: Commit the read-only release boundary**

```bash
git add src/ratemem/paper/release.py tests/unit/paper/test_release.py tests/contract/paper/test_release_schema.py
git commit -m "feat(paper): validate scientific paper releases"
```

### Task 3: Generate empirical macros and tables without hand-entered numbers

**Files:**
- Create: `src/ratemem/paper/render.py`
- Create: `tests/unit/paper/test_render.py`
- Create: `paper/claims.yaml`

- [ ] **Step 1: Define the falsifiable claim registry**

```yaml
# paper/claims.yaml
schema_version: "1.0"
claims:
  shared_packet_representation:
    section: results
    gate_id: shared_packet_representation
    endpoint: request_weighted_identity
    comparator_source: strongest_eligible_control
    required_cells:
      - {budget_label: 50pct, request_regime: uniform}
      - {budget_label: 50pct, request_regime: zipf}
    supporting_cells:
      - {budget_label: 25pct, request_regime: uniform}
      - {budget_label: 25pct, request_regime: zipf}
      - {budget_label: 75pct, request_regime: uniform}
      - {budget_label: 75pct, request_regime: zipf}
    positive_wording_requires_gate_pass: true
  causal_packet_allocator:
    section: results
    gate_id: causal_packet_allocator
    endpoint: request_weighted_utility
    comparator_source: strongest_eligible_control
    required_cells:
      - {budget_label: 50pct, request_regime: uniform}
      - {budget_label: 50pct, request_regime: zipf}
    positive_wording_requires_gate_pass: true
  allocator_guarantee:
    section: method
    gate_id: allocator_guarantee
    endpoint: certified_reduced_set_approximation_ratio
    comparator_source: exact_reduced_set_optimum
    ground_set_scope: causal_singleton_density_prescreen_C_t_max24
    positive_wording_requires_gate_pass: true
  optimization_free_tradeoff:
    section: results
    gate_id: optimization_free_tradeoff
    endpoint: identity
    comparator_source: strongest_eligible_control
    positive_wording_requires_gate_pass: true
```

- [ ] **Step 2: Write failing macro/table rendering tests**

```python
# tests/unit/paper/test_render.py
from pathlib import Path

import yaml

from ratemem.paper.release import load_paper_release
from ratemem.paper.render import build_generated_release


def test_allocator_claim_registry_locks_reduced_ground_set() -> None:
    registry = yaml.safe_load(Path("paper/claims.yaml").read_text(encoding="utf-8"))
    scientific = yaml.safe_load(
        Path("configs/scientific/evaluation-policy.yaml").read_text(encoding="utf-8")
    )
    claim = registry["claims"]["allocator_guarantee"]
    scientific_claim = scientific["claims"]["allocator_guarantee"]
    assert (
        claim["endpoint"]
        == scientific_claim["primary_endpoint"]
        == "certified_reduced_set_approximation_ratio"
    )
    assert scientific_claim["required_controls"] == [claim["comparator_source"]]
    assert claim["comparator_source"] == "exact_reduced_set_optimum"
    assert (
        claim["ground_set_scope"]
        == scientific_claim["ground_set_scope"]
        == "causal_singleton_density_prescreen_C_t_max24"
    )


def test_submission_macros_come_from_validated_cells(
    valid_paper_release: Path, paper_schema: Path, tmp_path: Path
) -> None:
    release = load_paper_release(valid_paper_release, paper_schema, allow_fixture=True)
    build_generated_release(release, Path("paper/claims.yaml"), tmp_path)
    macros = (tmp_path / "results_macros.tex").read_text()
    assert r"\PaperResultsAvailabletrue" in macros
    assert r"\newcommand{\PaperReleaseId}{fixture-release}" in macros
    assert r"\newcommand{\SharedPacketUniformEstimate}{1.23}" in macros
    assert r"\newcommand{\SharedPacketUniformCILow}{0.40}" in macros
    assert r"\newcommand{\SharedPacketUniformCIHigh}{2.06}" in macros
    assert "source_artifact" not in macros


def test_draft_mode_contains_no_empirical_number(tmp_path: Path) -> None:
    build_generated_release(None, Path("paper/claims.yaml"), tmp_path)
    macros = (tmp_path / "results_macros.tex").read_text()
    assert macros == "\\newif\\ifPaperResultsAvailable\n\\PaperResultsAvailablefalse\n"
    assert not (tmp_path / "tables/main_lifecycle.tex").exists()
```

- [ ] **Step 3: Run the renderer tests and verify the expected failure**

Run: `uv run pytest tests/unit/paper/test_render.py -q`

Expected: collection fails because `ratemem.paper.render` does not exist.

- [ ] **Step 4: Implement deterministic TeX escaping and decimal formatting**

```python
# src/ratemem/paper/render.py
from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import yaml

from ratemem.paper.release import PaperRelease, file_sha256

TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(value: str) -> str:
    return "".join(TEX_ESCAPES.get(character, character) for character in value)


def decimal_2(value: str) -> str:
    return str((Decimal(value) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{tex_escape(value)}}}" + "\n"


def _cell(release: PaperRelease, claim_id: str, endpoint: str, regime: str) -> dict[str, str]:
    matches = [
        row
        for row in release.rows["tables/main_lifecycle.csv"]
        if row["claim_id"] == claim_id
        and row["endpoint"] == endpoint
        and row["budget_label"] == "50pct"
        and row["request_regime"] == regime
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one locked primary cell for {claim_id}/{regime}")
    return matches[0]


def build_generated_release(release: PaperRelease | None, claims_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir():
        if child.is_file():
            child.unlink()
    if release is None:
        (output / "results_macros.tex").write_text(
            "\\newif\\ifPaperResultsAvailable\n\\PaperResultsAvailablefalse\n",
            encoding="utf-8",
        )
        return
    claims = yaml.safe_load(claims_path.read_text(encoding="utf-8"))
    uniform = _cell(release, "shared_packet_representation", "request_weighted_identity", "uniform")
    zipf = _cell(release, "shared_packet_representation", "request_weighted_identity", "zipf")
    lines = ["\\newif\\ifPaperResultsAvailable\n", "\\PaperResultsAvailabletrue\n"]
    lines.append(_macro("PaperReleaseId", release.release_id))
    for prefix, row in (("SharedPacketUniform", uniform), ("SharedPacketZipf", zipf)):
        lines.append(_macro(prefix + "Estimate", decimal_2(row["estimate"])))
        lines.append(_macro(prefix + "CILow", decimal_2(row["ci_low"])))
        lines.append(_macro(prefix + "CIHigh", decimal_2(row["ci_high"])))
        lines.append(_macro(prefix + "Comparator", row["comparator_id"]))
    (output / "results_macros.tex").write_text("".join(lines), encoding="utf-8")
    render_main_lifecycle_table(release, output / "tables/main_lifecycle.tex")
    render_baseline_table(release, output / "tables/baseline_compliance.tex")
    render_efficiency_table(release, output / "tables/efficiency.tex")
    provenance = {
        "schema_version": "1.0",
        "release_id": release.release_id,
        "artifact_manifest_sha256": file_sha256(release.root / "artifact_manifest.json"),
        "claims_sha256": file_sha256(claims_path),
        "generated_files": sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
```

Implement the three table renderers with fixed column order, `booktabs`, escaped text, and `Decimal` formatting. They must not accept a manually supplied caption, method value, or metric value. Write into a temporary sibling directory and atomically replace `paper/generated/` only after every output succeeds; never leave a partial generated release.

- [ ] **Step 5: Add gate-derived result sentences**

Generate these macros from the two validated 50% cells and their actual strongest comparator:

```tex
\newcommand{\PaperSharedPacketResultSentence}{At the locked 50\% state budget, RateMem-DiT changes request-weighted identity by \SharedPacketUniformEstimate\ percentage points under uniform requests (95\% CI: \SharedPacketUniformCILow, \SharedPacketUniformCIHigh) and by \SharedPacketZipfEstimate\ points under Zipf requests (95\% CI: \SharedPacketZipfCILow, \SharedPacketZipfCIHigh) relative to the strongest eligible controls.}
```

The renderer may use the verb `improves` only when the gate row has `status=pass`, both required CI
lower bounds are positive, the prompt margin passes, and supporting 25%/75% point estimates are
nonnegative. An evaluated `status=fail` is not a build error: generate a claim-specific sentence such
as `The preregistered comparison did not establish an improvement ...`, retain the estimate and CI,
and select the section/abstract/conclusion wording required by `paper_disposition`. A failed
`allocator_guarantee` disables theorem macros and emits the theorem-free empirical framing. Even a
passed row enables theorem macros only when the claim registry and evidence both bind
`comparator_source=exact_reduced_set_optimum` and
`ground_set_scope=causal_singleton_density_prescreen_C_t_max24`; missing or full-`G_t` scope is
malformed evidence, not a theorem pass. Only a `blocked` or missing gate exits 2. Tests must cover
all four allowed dispositions and prove that a failed gate cannot render positive verbs, while a
passed gate cannot render the negative template.

- [ ] **Step 6: Run focused tests and commit the only numeric ingress**

Run: `uv run pytest tests/unit/paper/test_release.py tests/unit/paper/test_render.py -q`

Expected: all tests pass, including checksum failure, blocked-gate rejection, and evaluated-failure
negative-framing cases; synthetic fixture numbers occur only under `tests/` and temporary test output.

```bash
git add src/ratemem/paper/render.py paper/claims.yaml tests/unit/paper/test_render.py
git commit -m "feat(paper): generate results from validated artifacts"
```

### Task 4: Create a clean review-mode manuscript and complete English introduction

**Files:**
- Create: `paper/Makefile`
- Create: `paper/latexmkrc`
- Create: `paper/main.tex`
- Create: `paper/preamble.tex`
- Create: `paper/sections/abstract.tex`
- Create: `paper/sections/introduction.tex`
- Create: `paper/sections/problem.tex`
- Create: `paper/sections/related_work.tex`
- Create: `paper/sections/method.tex`
- Create: `paper/sections/training.tex`
- Create: `paper/sections/evaluation.tex`
- Create: `paper/sections/results.tex`
- Create: `paper/sections/ethics.tex`
- Create: `paper/sections/limitations.tex`
- Create: `paper/sections/conclusion.tex`
- Create: `paper/README.md`
- Test: `tests/contract/paper/test_manuscript_source.py`

- [ ] **Step 1: Write a source-level test that rejects the historical manuscript**

```python
# tests/contract/paper/test_manuscript_source.py
from pathlib import Path

ROOT = Path(__file__).parents[3]
PAPER = ROOT / "paper"
OLD_PHRASES = (
    "Memory-MetaGAN",
    "Experiments (to do)",
    "Despit the rapid advance",
    "Gaussian mixed sampling",
    "Vggface",
    "lipsum",
)


def manuscript_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(PAPER.rglob("*.tex")))


def test_manuscript_is_a_new_english_review_source() -> None:
    text = manuscript_text()
    assert all(phrase not in text for phrase in OLD_PHRASES)
    assert not any("\u4e00" <= character <= "\u9fff" for character in text)
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert r"\usepackage[review]{cvpr}" in main
    assert r"\cvprfinalcopy" not in main
    assert r"\input{generated/results_macros}" in main
    assert r"\zlabel{paper:last-main-page}" in main
```

- [ ] **Step 2: Run the source test and verify the expected failure**

Run: `uv run pytest tests/contract/paper/test_manuscript_source.py -q`

Expected: FAIL because `paper/main.tex` and the new section sources do not exist.

- [ ] **Step 3: Add the deterministic draft and submission build targets**

```make
# paper/Makefile
SHELL := /bin/bash
ROOT := $(abspath ..)
TEX_IMAGE := ayakaleaf-overleaf-pro@sha256:9ba19de5f95dcf6c063725f9cf1c00a86327a866d26d65167ee85b2201e3e996
RELEASE := artifacts/paper/cvpr2027-submission-v1
TEX_RUN := docker run --rm --network none --user $$(id -u):$$(id -g) \
  --volume "$(ROOT):/workspace" --workdir /workspace/paper \
  --entrypoint /bin/bash $(TEX_IMAGE) -lc

.PHONY: draft submission supplement clean audit

draft:
	cd "$(ROOT)" && uv run ratemem-paper release build \
	  --mode draft --claims paper/claims.yaml --output paper/generated
	@mkdir -p build
	$(TEX_RUN) 'latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex'

submission:
	@test -n "$(PAPER_ID)" || { echo 'BLOCKED submission: PAPER_ID is required'; exit 2; }
	cd "$(ROOT)" && uv run ratemem-paper release build \
	  --mode submission --release "$(RELEASE)" --schema schemas/paper-release.schema.json \
	  --claims paper/claims.yaml --paper-id "$(PAPER_ID)" --output paper/generated
	@mkdir -p build
	$(TEX_RUN) 'latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex'
	cd "$(ROOT)" && uv run ratemem-paper audit submission \
	  --source paper --pdf paper/build/main.pdf --policy paper/config/cvpr-policy.yaml

supplement:
	@test -f generated/results_macros.tex || { echo 'BLOCKED supplement: generate a release first'; exit 2; }
	@mkdir -p build
	$(TEX_RUN) 'latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build supplement.tex'

clean:
	$(TEX_RUN) 'latexmk -C -outdir=build main.tex supplement.tex'
```

```perl
# paper/latexmkrc
$pdf_mode = 1;
$bibtex_use = 2;
$max_repeat = 5;
$recorder = 1;
```

The release command writes `paper/generated/build_config.tex`. Draft mode defines `\PaperTemplateYear` as `2026`, `\paperID` as `DRAFT`, and no author data. Submission mode requires a decimal OpenReview paper ID, records it in provenance, and refuses `DRAFT`, zeroes, asterisks, or missing input.

- [ ] **Step 4: Add the official review entry point and minimal preamble**

```tex
% paper/main.tex
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
\input{preamble}
\input{generated/build_config}
\input{generated/results_macros}

\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=cvprblue]{hyperref}

\def\confName{CVPR}
\def\confYear{\PaperTemplateYear}
\title{RateMem-DiT: Shared Progressive Adapter Packets for Bounded Optimization-Free Personalization}
\author{Anonymous CVPR submission}

\begin{document}
\maketitle
\input{sections/abstract}
\input{sections/introduction}
\input{sections/problem}
\input{sections/related_work}
\input{sections/method}
\input{sections/training}
\input{sections/evaluation}
\input{sections/results}
\input{sections/limitations}
\input{sections/conclusion}
\zlabel{paper:last-main-page}
{
  \small
  \bibliographystyle{ieeenat_fullname}
  \bibliography{references}
}
\end{document}
```

```tex
% paper/preamble.tex
\usepackage{amsmath,amssymb,mathtools}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage{xspace}
\usepackage{zref-abspage,zref-user}
\usepackage{algorithm}
\usepackage[noend]{algpseudocode}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,backgrounds,calc,fit,matrix,positioning,shapes.geometric}

\newcommand{\method}{RateMem-DiT\xspace}
\newcommand{\codeTarget}{c^{\star}}
\newcommand{\codeDecoded}{\widehat{c}}
\DeclareMathOperator{\bits}{bits}
\DeclareMathOperator*{\argmax}{arg\,max}
\sisetup{detect-all,group-separator={,},group-minimum-digits=4}
```

Do not load the duplicate algorithm packages, obsolete `times`, `epsfig`, `utf8x`, or caption overrides present in the 2020 source.

- [ ] **Step 5: Write the complete evidence-neutral abstract**

```tex
% paper/sections/abstract.tex
\begin{abstract}
Personalized text-to-image systems usually treat each subject's state as an
independent adapter or feature record.  This abstraction hides a deployment
constraint: a service must create, update, read, and delete many personalized
states while their total serialized footprint remains below a fixed byte
budget.  We formulate this setting as a causal lifecycle trace and introduce
\method, which predicts adapter coefficients without per-subject optimization,
stores a mandatory base code for each admitted concept, and shares immutable
progressive enhancement packets across concepts.  Packet admission is
nonseparable because one payload can improve several active concepts while its
bytes are paid once.  A request-aware allocator optimizes a locked monotone
submodular coverage surrogate under the residual packet budget; its certified
variant operates on a causal deterministically pre-screened candidate set and is
evaluated against exact reduced-set small-instance optima.  We pair this method
with byte-exact serialization, immutable operational traces, read-only scoring
probes, matched independent-code, shared-subspace, and feature-cache controls,
and artifact-derived statistical reporting.  \ifPaperResultsAvailable
\PaperSharedPacketResultSentence\fi
\end{abstract}
```

This is a genuine draft abstract: it states the problem, mechanism, and evaluation contract, but makes no superiority claim unless the validated release emits the final sentence.

- [ ] **Step 6: Write the complete new introduction**

Use this exact argument order and preserve the novelty boundary:

```tex
% paper/sections/introduction.tex
\section{Introduction}
\label{sec:introduction}

Subject-driven diffusion models can bind a small reference set to a new visual
concept, but deployment changes the unit of concern.  A personalization service
does not hold one adapter in isolation: it receives a stream of creations,
updates, reads, and deletions for many concepts under finite storage and latency.
Storing one independently generated adapter per active concept makes quality
approximately separable across users, yet it also makes capacity grow with the
active set.  Replacing those records with a conventional cache controls the
number of residents but does not exploit reusable structure among their adapter
codes.

We study optimization-free personalization under a hard budget on the exact
serialized online state.  The budget includes base codes, enhancement payloads,
hashes, concept--packet incidences and gains, handles, usage metadata, checksums,
alignment, and allocator state.  Operational reads update request statistics;
evaluation probes act on copied snapshots and cannot influence later admission
or eviction.  This separation turns a vague continual-personalization narrative
into a replayable systems problem with explicit state and traffic semantics.

\method represents every admitted concept by a compact base code and optional
immutable enhancement packets.  A packet is content addressed and may be
referenced by several concepts.  Retaining it can therefore improve several
decoded adapters for one payload cost, whereas removing it can degrade several
concepts.  This cross-concept incidence is the source of nonseparability; the
support-to-adapter predictor, low-rank atom basis, quantizer, and ordinary cache
bookkeeping are established components rather than claimed inventions.

Allocation separates whole-base admission from a fixed-cohort packet problem.
For the latter, we define a calibrated nonnegative coverage utility and select
immutable packet bundles under their exact modular byte costs.  Enumeration of
small seeds followed by exact marginal-density greedy is tested against brute
force on small instances.  The guarantee applies to one causal snapshot and its
locked surrogate.  It does not cover future-aware regret, whole-concept
admission, switching costs, or an unconstrained perceptual objective.

Our evaluation uses the same amortizer, adapter basis, backbone, prompts, noise,
and lifecycle trace for the proposed method and matched controls.  The decisive
comparisons include private progressive codes, online shared-subspace
compression, feature caching, and causal replacement or rate-allocation
policies under the same byte ledger.  Every empirical statement in the paper is
generated from a schema-valid, checksummed release; a failed scientific gate
blocks the algorithmic framing rather than being rewritten as a favorable
result.

Our contributions are:
\begin{itemize}
  \item a byte-bounded lifecycle formulation for optimization-free
        personalization with immutable read traffic and non-mutating probes;
  \item a progressive adapter memory in which immutable enhancement packets can
        serve multiple active concepts, coupling their rate allocation;
  \item a causal fixed-cohort packet allocator with an explicitly delimited
        coverage guarantee and byte-exact implementation tests; and
  \item an artifact-locked benchmark that separates representation, allocation,
        acquisition quality, autonomous lookup, latency, and deletion behavior.
\end{itemize}
```

- [ ] **Step 7: State the lifecycle problem and byte ledger precisely**

```tex
% paper/sections/problem.tex
\section{Bounded Personalization Lifecycle}
\label{sec:problem}
Let \(\mathcal{E}=(e_1,\ldots,e_T)\) be a serialized trace whose events are
\textsc{Create}, \textsc{Update}, \textsc{Read}, \textsc{Delete}, or
\textsc{Probe}.  A creation maps a support set and description to an opaque
handle; an update is labeled with an existing handle; and a read combines a
handle, prompt, and generation seed.  A probe evaluates a copied snapshot with
usage updates disabled.  Foundation-model and controller parameters are frozen
at deployment, so creation and update perform forward computation and bounded
state mutation but no per-concept gradient descent.

At every event, the mutable state \(M_t\) must satisfy
\begin{equation}
  \bits(M_t) \le B.
  \label{eq:hard-budget}
\end{equation}
The left-hand side is the length of the canonical serialized representation,
not a tensor-shape estimate.  Shared trained weights and frozen codec
dictionaries are reported separately and amortized at each active-set size.
This convention prevents unbounded handle tables, hidden support-image stores,
or uncounted controller metadata from becoming free capacity.
```

- [ ] **Step 8: Add nonempty evidence-neutral section boundaries and build**

Create the remaining section files with these exact contents so the intermediate manuscript is valid without unsupported detail:

```tex
% paper/sections/related_work.tex
\section{Related Work}\label{sec:related}
We distinguish amortized personalization, memory and feature caching, and shared
adapter compression before stating the narrower packet-allocation hypothesis.
```

```tex
% paper/sections/method.tex
\section{RateMem-DiT}\label{sec:method}
The method combines a shared amortizer with immutable progressive packets and a
causal byte-constrained allocator.
```

```tex
% paper/sections/training.tex
\section{Sequential Meta-Training}\label{sec:training}
Training uses concept-disjoint lifecycle segments and never opens the final trace.
```

```tex
% paper/sections/evaluation.tex
\section{Evaluation Protocol}\label{sec:evaluation}
All methods replay the same immutable traffic under the same exact byte ledger.
```

```tex
% paper/sections/results.tex
\section{Results}\label{sec:results}
\ifPaperResultsAvailable\PaperSharedPacketResultSentence\fi
```

```tex
% paper/sections/ethics.tex
\section{Ethical Considerations}
Identity personalization requires explicit privacy, consent, and misuse review.
```

```tex
% paper/sections/limitations.tex
\section{Limitations}
The scientific claims remain conditional on amortizer quality, useful packet
reuse, surrogate calibration, and controlled contamination evidence.
```

```tex
% paper/sections/conclusion.tex
\section{Conclusion}
RateMem-DiT makes online personalization state and its lifecycle auditable under
a hard serialized-byte budget.
```

Then run:

```bash
make -C paper draft
uv run pytest tests/contract/paper/test_manuscript_source.py -q
```

Expected: `paper/build/main.pdf` is produced, the source test passes, and the log contains no undefined citation/reference, missing file, or fatal LaTeX error.

```bash
git add paper/Makefile paper/latexmkrc paper/main.tex paper/preamble.tex paper/sections paper/README.md tests/contract/paper/test_manuscript_source.py
git commit -m "docs(paper): replace legacy manuscript with CVPR draft"
```

### Task 5: Build primary-source related-work and bibliography evidence

**Files:**
- Create: `paper/related_work/required_sources.yaml`
- Create: `paper/related_work/evidence.yaml`
- Create: `paper/references.bib`
- Modify: `paper/sections/related_work.tex`
- Create: `src/ratemem/paper/literature.py`
- Test: `tests/unit/paper/test_literature.py`
- Test: `tests/contract/paper/test_citations.py`

- [ ] **Step 1: Encode every required closest-work family and canonical source**

```yaml
# paper/related_work/required_sources.yaml
schema_version: "1.0"
sources:
  - {citation_key: hyperlora_cvpr2025, role: closest_amortizer, primary_url: "https://openaccess.thecvf.com/content/CVPR2025/html/Li_HyperLoRA_Parameter-Efficient_Adaptive_Generation_for_Portrait_Synthesis_CVPR_2025_paper.html"}
  - {citation_key: hyperdreambooth, role: amortized_personalization, primary_url: "https://arxiv.org/abs/2307.06949"}
  - {citation_key: vsm_diffusion_neurips2023, role: memory_personalization, primary_url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/17826a22eb8b58494dfdfca61e772c39-Abstract-Conference.html"}
  - {citation_key: dreamcache_cvpr2025, role: feature_cache, primary_url: "https://openaccess.thecvf.com/content/CVPR2025/html/Aiello_DreamCache_Finetuning-Free_Lightweight_Personalized_Image_Generation_via_Feature_Caching_CVPR_2025_paper.html"}
  - {citation_key: compress_then_serve_icml2025, role: shared_adapter_compression, primary_url: "https://proceedings.mlr.press/v267/gabrielsson25a.html"}
  - {citation_key: vb_lora_neurips2024, role: sparse_shared_bank, primary_url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e0d38c676d5855bcfab7f6d29d20ad9-Abstract-Conference.html"}
  - {citation_key: share_eccv2026, role: online_shared_subspace, primary_url: "https://arxiv.org/abs/2602.06043"}
  - {citation_key: moblora_acl2026, role: shared_bases_continual, primary_url: "https://aclanthology.org/2026.acl-long.481/"}
  - {citation_key: cf_star_2026, role: sparse_residual_compression, primary_url: "https://doi.org/10.1007/s40747-026-02238-y"}
  - {citation_key: rqt, role: progressive_residual_quantization, primary_url: "https://aclanthology.org/2025.findings-acl.554/"}
  - {citation_key: sinelora_delta, role: quantized_adapter_compression, primary_url: "https://ojs.aaai.org/index.php/AAAI/article/view/39279"}
  - {citation_key: loraquant, role: mixed_precision_adapter_quantization, primary_url: "https://arxiv.org/abs/2510.26690"}
  - {citation_key: ada_neurips2022, role: adapter_consolidation, primary_url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html"}
  - {citation_key: ella_icml2013, role: online_sparse_task_dictionary, primary_url: "https://proceedings.mlr.press/v28/ruvolo13.html"}
```

Import the remaining baseline citation keys from `configs/scientific/baseline-requirements.yaml` and their immutable primary URLs/revisions from `configs/scientific/baseline-lock.yaml`; the command must fail if either file contains a required method without a real source record.

- [ ] **Step 2: Write failing evidence and citation-coverage tests**

```python
# tests/contract/paper/test_citations.py
from pathlib import Path

from ratemem.paper.literature import audit_literature


def test_every_citation_and_related_work_claim_has_primary_evidence() -> None:
    report = audit_literature(
        paper_root=Path("paper"),
        required_path=Path("paper/related_work/required_sources.yaml"),
        evidence_path=Path("paper/related_work/evidence.yaml"),
        bibliography_path=Path("paper/references.bib"),
        baseline_requirements=Path("configs/scientific/baseline-requirements.yaml"),
        baseline_lock=Path("configs/scientific/baseline-lock.yaml"),
    )
    assert report.missing_required == set()
    assert report.citations_without_evidence == set()
    assert report.evidence_without_bibtex == set()
    assert report.duplicate_titles == set()
    assert report.unresolved_locators == set()
```

Each `evidence.yaml` entry must contain a citation key, title, authors, venue, year, canonical identifier, primary URL, immutable source SHA-256, UTC verification date, exact page/section/figure locators, conservative paraphrases, and structured BibTeX fields. Verbatim support excerpts are optional and limited to 25 words per source.

- [ ] **Step 3: Run the evidence tests and verify the expected failure**

Run: `uv run pytest tests/unit/paper/test_literature.py tests/contract/paper/test_citations.py -q`

Expected: collection fails because `ratemem.paper.literature` does not exist.

- [ ] **Step 4: Implement the literature auditor and deterministic BibTeX export**

```python
# essential validation in src/ratemem/paper/literature.py
def validate_evidence_entry(entry: dict[str, object]) -> None:
    required = {
        "citation_key", "title", "authors", "venue", "year",
        "canonical_identifier", "primary_url", "source_sha256",
        "verified_at_utc", "claims", "bibtex",
    }
    missing = required - entry.keys()
    if missing:
        raise LiteratureError(f"missing evidence fields: {sorted(missing)}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(entry["source_sha256"])):
        raise LiteratureError("source_sha256 must bind the reviewed primary source")
    for claim in entry["claims"]:
        if not claim["locator"] or not claim["paraphrase"]:
            raise LiteratureError("every claim needs a source locator and paraphrase")
        if len(claim.get("support_excerpt", "").split()) > 25:
            raise LiteratureError("support excerpt exceeds 25 words")
```

Use Pybtex to parse the generated `paper/references.bib`, normalize DOI/arXiv/venue metadata, reject duplicate DOI/title records, reject missing authors/titles/years, and byte-compare a second export. The LaTeX scanner must parse all `\cite`, `\citep`, and `\citet` keys and reject wildcard citation.

- [ ] **Step 5: Verify one cited primary source and repeat until the evidence ledger is complete**

For the next unchecked item in `required_sources.yaml` or cited baseline, download the official HTML/PDF, record its SHA-256, inspect one relevant method/experiment section, and enter only claims supported by an exact locator. Repeat this checkbox in 2--5 minute locator-sized passes until every intended paraphrase is evidenced. Prefer peer-reviewed versions over preprints when both exist. Do not cite search-result summaries, project-page marketing text, or a title that cannot be matched to a primary paper.

Run:

```bash
uv run ratemem-paper literature audit \
  --paper paper \
  --required paper/related_work/required_sources.yaml \
  --evidence paper/related_work/evidence.yaml \
  --bibliography paper/references.bib \
  --baseline-requirements configs/scientific/baseline-requirements.yaml \
  --baseline-lock configs/scientific/baseline-lock.yaml
```

Expected: `PASS literature: every citation has primary-source evidence; duplicate and unresolved counts are zero`.

- [ ] **Step 6: Rewrite related work around occupied components and the narrow boundary**

```tex
% paper/sections/related_work.tex
\section{Related Work}
\label{sec:related}

\paragraph{Amortized personalization.}
HyperDreamBooth predicts compact personalized weights and optionally refines
them, while HyperLoRA predicts coefficients over a low-dimensional LoRA basis
from subject evidence~\cite{hyperdreambooth,hyperlora_cvpr2025}.  We use the
same broad support-to-adapter pattern to remove per-request optimization; that
amortizer is shared with the independent-code controls and is not our claimed
contribution.

\paragraph{Memory and feature caching.}
VSM-Diffusion combines few-shot diffusion with episodic and semantic memory,
uncertainty, consolidation, and replacement~\cite{vsm_diffusion_neurips2023}.
DreamCache instead stores lightweight intermediate features for tuning-free
personalization~\cite{dreamcache_cvpr2025}.  These systems motivate matched
memory and feature-cache controls, but neither comparison permits hidden state:
all payloads, keys, handles, and metadata enter the same serialized byte ledger.

\paragraph{Shared and compressed adapters.}
Compress then Serve factors collections of adapters into shared bases and
adapter-specific coefficients~\cite{compress_then_serve_icml2025}; VB-LoRA uses
sparse mixtures over a shared vector bank~\cite{vb_lora_neurips2024}; and Share
continually updates a shared LoRA subspace while reprojecting prior task
coefficients~\cite{share_eccv2026}.  MoBLoRA, CF-STAR, RQT, SineLoRA-Delta, and
LoRAQuant further occupy shared bases, sparse residuals, progressive coding,
and mixed-precision compression~\cite{moblora_acl2026,cf_star_2026,rqt,sinelora_delta,loraquant}.
Online sparse task dictionaries and adapter consolidation predate this setting
as well~\cite{ella_icml2013,ada_neurips2022}.

\paragraph{Boundary of this work.}
Our hypothesis is not that adapter prediction, quantization, sparse banks,
residual coding, or cache replacement is new.  It is that immutable enhancement
packets with multi-concept incidences, coupled to a causal byte-constrained
allocator, improve the request-weighted quality--storage frontier over the
strongest matched separable, shared-subspace, and feature-cache alternatives.
The experiments isolate representation by fixing the allocator and isolate
allocation by fixing the packet stream.
```

- [ ] **Step 7: Run one literature-refresh query and repeat at both required freezes**

Run the next predefined primary-source query for online shared LoRA subspaces, progressive adapter compression, byte-bounded personalization memory, or submodular shared-packet allocation. Record its database, exact query string, date range, candidate titles, disposition, and source hashes in `paper/related_work/literature-audit.json`; repeat once per query at experiment freeze and again immediately before submission. The audit command fails if any required query is absent or if the completed audit is older than 30 days in submission mode.

Run: `uv run ratemem-paper literature freshness --audit paper/related_work/literature-audit.json --max-age-days 30`

Expected: `PASS literature freshness: audit is within 30 days and every included candidate is evidenced`.

- [ ] **Step 8: Run BibTeX and citation checks, then commit**

Run:

```bash
make -C paper draft
uv run pytest tests/unit/paper/test_literature.py tests/contract/paper/test_citations.py -q
```

Expected: the PDF builds with zero undefined citations; all citation tests pass.

```bash
git add paper/related_work paper/references.bib paper/sections/related_work.tex src/ratemem/paper/literature.py tests/unit/paper/test_literature.py tests/contract/paper/test_citations.py
git commit -m "docs(paper): ground related work in primary evidence"
```

### Task 6: Write the method, allocator boundary, and sequential training sections

**Files:**
- Modify: `paper/sections/method.tex`
- Modify: `paper/sections/training.tex`
- Create: `paper/supplement/theorem.tex`
- Test: `tests/contract/paper/test_method_claims.py`

- [ ] **Step 1: Write a failing contract test for the theorem boundary**

```python
# tests/contract/paper/test_method_claims.py
from pathlib import Path


def test_method_contains_exact_budget_and_theorem_qualifiers() -> None:
    text = Path("paper/sections/method.tex").read_text(encoding="utf-8")
    assert r"\sum_{p\in X}c_{t,p}\le b_t" in text
    assert r"1-1/e" in text
    assert r"X\subseteq C_t" in text
    assert "no approximation guarantee relative to the full" in text
    assert "fixed admitted cohort" in text
    assert "future-aware" in text
    assert "not a competitive or dynamic-regret guarantee" in text
    assert r"\ifAllocatorGuaranteeValidated" in text


def test_method_does_not_claim_borrowed_components() -> None:
    text = Path("paper/sections/method.tex").read_text(encoding="utf-8")
    forbidden = (
        "we invent LoRA",
        "our novel hypernetwork",
        "our novel quantizer",
        "our novel LRU",
        "guarantees low future regret",
    )
    assert all(phrase not in text for phrase in forbidden)
```

- [ ] **Step 2: Run the method test and verify the expected failure**

Run: `uv run pytest tests/contract/paper/test_method_claims.py -q`

Expected: FAIL because the new method section has not yet been written.

- [ ] **Step 3: Write the frozen adapter execution and progressive packet representation**

```tex
% first half of paper/sections/method.tex
\section{RateMem-DiT}
\label{sec:method}

\subsection{Amortized dynamic adapters}
Given an unordered support set \(S_i\) and concept description \(d_i\), a
support encoder predicts a target code
\begin{equation}
  c_i^\star = F_\psi(S_i,d_i).
  \label{eq:amortizer}
\end{equation}
The frozen denoiser executes this code through low-rank atoms.  For an adapted
linear projection,
\begin{equation}
  y = Wx + \sum_{m=1}^{A}\alpha_m B_m A_m x,
  \label{eq:dynamic-atom}
\end{equation}
where the backbone weight \(W\) is frozen and the low-rank sum is accumulated
without materializing a dense weight update.  The same amortizer and atom basis
are used by every independent-code and packet-memory comparison.

\subsection{Progressive shared-packet state}
Each admitted concept \(i\) owns a mandatory base code \(q_i^0\).  Optional
immutable packets refine the decoded adapter:
\begin{equation}
  \widehat c_i(M)=D(q_i^0)+
  \sum_{p\in\mathcal{P}(M)}m_{i,p}\widehat a_{i,p}e_p.
  \label{eq:packet-decode}
\end{equation}
Here \(e_p\) is an immutable content-addressed address into the frozen learned
dictionary, \(m_{i,p}\) is an incidence bit, and \(\widehat a_{i,p}\) is a
signed-int16 concept-specific gain.  Packets are reused only when their complete
dictionary-revision/group/stage/entry payloads match exactly.  The method does
not perform bounded-error packet matching or mutate/version dictionary payloads;
a concept update creates a new immutable all-incidence bundle while preserving
the other concepts' incidences byte-for-byte.

The exact online-state cost is
\begin{equation}
\begin{split}
  \bits(M)={}&\sum_{i\in\mathcal A}
    \bits(q_i^0,h_i,\mathrm{meta}_i)\\
  &+\sum_{p\in\mathcal P(M)}\bits(e_p,\mathrm{hash}_p)\\
  &+\sum_{(i,p):m_{i,p}=1}\bits(i,p,\widehat a_{i,p}).
\end{split}
\label{eq:byte-ledger}
\end{equation}
One packet variable can benefit several concepts while its payload bytes are
charged once.  This coupling distinguishes the method from assigning an
independent bit rate to each concept.
```

- [ ] **Step 4: Write the causal allocator and exact proof boundary**

```tex
% second half of paper/sections/method.tex
\subsection{Causal fixed-cohort packet allocation}
An outer size-aware policy first selects a fixed admitted cohort and reserves
all mandatory base and metadata bytes.  At event \(t\), the inner allocator has
packet capacity
\begin{equation}
  b_t=B-\sum_{i\in\mathcal A_t}\bits(q_i^0,h_i,\mathrm{meta}_i).
\end{equation}
Its finite ground set \(G_t\) contains resident packets and packets proposed by
the current event.  A ground item is an immutable bundle: one payload and hash
plus a prespecified list \(A_{t,p}\) of nonnegative incidences and gains.  Its
modular cost is
\begin{equation}
  c_{t,p}=\bits(e_p,\mathrm{hash}_p)+
  \sum_{i\in A_{t,p}}\bits(i,p,\widehat a_{i,p}).
\end{equation}

Request weights \(\omega_{t,i}\) depend only on operational reads in the past.
A frozen calibration model provides nonnegative group weights
\(\beta_{t,i,g}\) and packet gains \(v_{t,i,g,p}\), with zero gain when the
bundle contains no incidence for concept \(i\).  We optimize
\begin{equation}
F_t(X)=\sum_{i\in\mathcal A_t}\omega_{t,i}
\sum_g\beta_{t,i,g}
\min\!\left\{1,\sum_{p\in X}v_{t,i,g,p}\right\}.
\label{eq:coverage}
\end{equation}
This nonnegative concave-over-modular coverage function is normalized,
monotone, and submodular in packet bundles.

Before certified enumeration, a causal deterministic pre-screen removes
individually infeasible packets, sorts the remainder by descending exact
singleton marginal density with lexicographically larger packet IDs winning
exact ties, and retains the highest-density 24.  We denote this fixed reduced
ground set by \(C_t\).
The pre-screen has no approximation guarantee relative to the full \(G_t\), so
its full-pool loss is an empirical quantity.

The certified allocator enumerates feasible seed sets of at most three packets
from \(C_t\) and completes each seed by exact marginal-density greedy.  Lazy
evaluation is permitted only when it returns the identical sequence.  For a
fixed admitted cohort, fixed reduced set, and exact value oracle, the target
statement is
\begin{equation}
  F_t(X_t)\ge(1-1/e)
  \max_{X\subseteq C_t:\,\sum_{p\in X}c_{t,p}\le b_t}F_t(X).
  \label{eq:snapshot-guarantee}
\end{equation}
\ifAllocatorGuaranteeValidated
The proof and exhaustive implementation check are provided in the supplementary
material.\else
Equation~\eqref{eq:snapshot-guarantee} is treated as a prespecified validation
target and not as an empirical claim in this draft.\fi
The statement is causal because the candidate pool, reduced set, costs, gains,
and weights use only current or past information.  It applies only to the fixed
admitted cohort, fixed reduced set, and locked coverage surrogate; it is neither
a full-pool guarantee nor a competitive or dynamic-regret guarantee against a
future-aware trace oracle.  Whole-base admission, switching costs, hysteresis,
optional incidence removal, and perceptual quality remain empirical outer-policy
choices.
```

The release renderer must define `\newif\ifAllocatorGuaranteeValidated` and set it true only when the required `allocator_guarantee` gate passes and its evidence hash resolves to the exact proof/test artifact. Draft mode sets it false.

- [ ] **Step 5: Add the exact allocator pseudocode**

```tex
\begin{algorithm}[t]
\caption{Certified fixed-cohort packet allocation}
\label{alg:allocator}
\begin{algorithmic}[1]
\Require Ground set \(G_t\), modular costs \(c_{t,p}\), capacity \(b_t\), value oracle \(F_t\)
\State \(C_t\gets\) the at-most-24 individually feasible packets with largest exact
  \(F_t(\{p\})/c_{t,p}\), breaking ties by larger packet ID
\State \(X^\star\gets\varnothing\)
\ForAll{\(S\subseteq C_t\) with \(|S|\le3\) and \(\sum_{p\in S}c_{t,p}\le b_t\)}
  \State \(X\gets S\)
  \While{a feasible packet remains}
    \State choose feasible \(p\in C_t\setminus X\) maximizing
      \((F_t(X\cup\{p\})-F_t(X))/c_{t,p}\)
    \State \(X\gets X\cup\{p\}\)
  \EndWhile
  \If{\(F_t(X)>F_t(X^\star)\)} \State \(X^\star\gets X\) \EndIf
\EndFor
\State \Return \(X^\star\)
\end{algorithmic}
\end{algorithm}
```

- [ ] **Step 6: Write the bounded sequential training contract**

```tex
% paper/sections/training.tex
\section{Sequential Meta-Training}
\label{sec:training}
Training segments are sampled from a versioned lifecycle generator and contain
creations, operational reads, labeled updates, deletions, and capacity overflow.
Concept pools, trace identifiers, and seed namespaces are disjoint across train,
validation, and test.  The encrypted final-test payload is unavailable to
training and model selection until the evaluation freeze is signed.

Each query uses one sampled flow-matching timestep and one transformer
forward/backward pass.  Full denoising is validation-only under a frozen prompt
and seed cap.  Memory transitions are functional and detached at truncated
boundaries; frozen text embeddings, VAE latents, and support features are
precomputed.  The codec is trained with a declared soft-quantization schedule
but evaluated with deterministic hard packets and canonical serialization.

The training objective combines one-step flow prediction, support-to-code
supervision, decoded-code distortion, request-weighted lifecycle utility, and
calibration of the nonnegative packet-gain surrogate.  Representation ablations
reuse the allocator; allocator ablations reuse the identical candidate packet
stream.  Architecture, margins, evaluators, and stopping rules are selected
without opening the final trace.
```

- [ ] **Step 7: Write and mechanically bind the supplementary proof**

`paper/supplement/theorem.tex` must enumerate the exact assumptions, define the deterministic causal pre-screen and reduced set `C_t`, explicitly disclaim any approximation ratio against full `G_t`, prove normalization/monotonicity/submodularity of Eq.~\eqref{eq:coverage}, state the modular-cost prerequisite induced by bundling incidences, cite the standard knapsack-submodular result, and map each mathematical premise to a named test artifact. It must also include the squared-error counterexample showing why arbitrary reconstruction gain is not automatically submodular and a statement that optional incidence selection would create a fixed-charge cost.

The paper-release renderer writes `generated/theorem_evidence.tex` containing only the validated proof artifact ID, exact-small-instance test count, minimum observed approximation ratio, and hashes. No proof statistic is typed into `method.tex` or `theorem.tex`.

- [ ] **Step 8: Run contract/build tests and commit**

Run:

```bash
uv run pytest tests/contract/paper/test_method_claims.py -q
make -C paper draft
```

Expected: the test passes; equations and algorithm compile with no undefined references or overfull algorithm box.

```bash
git add paper/sections/method.tex paper/sections/training.tex paper/supplement/theorem.tex tests/contract/paper/test_method_claims.py
git commit -m "docs(paper): specify shared packets and certified allocator"
```

### Task 7: Write the locked evaluation, gated results, ethics, limitations, and conclusion

**Files:**
- Modify: `paper/main.tex`
- Modify: `paper/sections/evaluation.tex`
- Modify: `paper/sections/results.tex`
- Modify: `paper/sections/ethics.tex`
- Modify: `paper/sections/limitations.tex`
- Modify: `paper/sections/conclusion.tex`
- Test: `tests/contract/paper/test_claim_gating.py`

- [ ] **Step 1: Write failing tests that prevent unsupported empirical prose**

```python
# tests/contract/paper/test_claim_gating.py
import re
from pathlib import Path


def strip_latex_commands(text: str) -> str:
    without_commands = re.sub(r"\\[A-Za-z@]+(?:\{[^{}]*\})?", "", text)
    return re.sub(r"%.*", "", without_commands)


def test_results_use_generated_values_and_sentences_only() -> None:
    text = Path("paper/sections/results.tex").read_text(encoding="utf-8")
    assert r"\ifPaperResultsAvailable" in text
    assert r"\PaperSharedPacketResultSentence" in text
    assert r"\input{generated/tables/main_lifecycle}" in text
    assert not any(character.isdigit() for character in strip_latex_commands(text))


def test_limitations_cover_all_design_risks() -> None:
    text = Path("paper/sections/limitations.tex").read_text(encoding="utf-8").lower()
    required = (
        "amortizer",
        "packet reuse",
        "surrogate",
        "contamination",
        "not machine unlearning",
        "backbone",
        "pilot budget",
    )
    assert all(term in text for term in required)


def test_ethics_covers_identity_data_and_misuse() -> None:
    text = Path("paper/sections/ethics.tex").read_text(encoding="utf-8").lower()
    assert all(term in text for term in ("consent", "privacy", "impersonation", "human study", "deletion"))
```

- [ ] **Step 2: Run the claim-gating tests and verify the expected failure**

Run: `uv run pytest tests/contract/paper/test_claim_gating.py -q`

Expected: FAIL because the evaluation, result, ethics, and limitation sections are not complete.

- [ ] **Step 3: Write the immutable evaluation protocol**

```tex
% paper/sections/evaluation.tex
\section{Evaluation Protocol}
\label{sec:evaluation}
\paragraph{Locks and replay.}
Dataset and evaluation locks pin source revisions and rights, globally deduplicated
concept pools, evaluator weights and preprocessing, prompts, generation settings,
byte budgets, request distributions, margins, inference units, and multiplicity
correction before comparative runs.  Every method replays the same serialized
events, support draws, prompts, and noise seeds.  Operational reads update usage;
all scoring uses a copied snapshot with usage updates disabled.

\paragraph{Protocols.}
The no-pressure protocol isolates acquisition quality and unintended state drift.
The budget-pressure protocol measures request-weighted active quality, the
quality--bytes frontier, rejection, eviction regret, stale-handle behavior, and
deletion collateral damage.  Autonomous exemplar lookup removes exact handles
and reports risk--coverage separately, so recognition error cannot contaminate
the primary lifecycle comparison.

\paragraph{Controls.}
All compatible methods share the backbone, sampler, guidance, resolution,
supports, prompts, seeds, amortizer, and static atom basis.  Required controls
include independent FIFO/LRU/LRUA code caches, a private progressive codec with
size-aware and separable rate policies, matched shared-subspace compression,
feature caching, per-concept optimization, and a future-aware oracle reported
only as an upper reference.  SANA-1.5 is the sole primary backbone in this route;
if a matched-required SANA control cannot be implemented faithfully, the affected
primary claim is blocked.  SDXL-native evidence remains contextual and cannot
replace a missing SANA comparison.

\paragraph{Metrics and inference.}
Primary endpoints are request-weighted identity for packet sharing and
request-weighted utility for allocation, subject to locked prompt or active-quality
non-inferiority margins.  We also report active-state drift, retention area,
maximum degradation, exact state components, insertion/read latency, peak memory,
energy, oracle regret, lookup calibration, and deletion effects.  Independently
sampled deployment episodes are the primary inference units; prompts and images
remain nested observations.  Paired hierarchical confidence intervals use at
least three independent training seeds, with Holm correction for secondary
families.  A blinded paired human study complements an identity evaluator that
is not used for training or filtering.
```

- [ ] **Step 4: Wire results exclusively through generated artifacts**

```tex
% paper/sections/results.tex
\section{Results}
\label{sec:results}
\ifPaperResultsAvailable
\PaperSharedPacketResultSentence
\input{generated/tables/main_lifecycle}
\input{generated/tables/baseline_compliance}

\paragraph{Representation and allocation.}
\PaperRepresentationAnalysis
\PaperAllocatorAnalysis

\paragraph{Efficiency.}
\PaperEfficiencyAnalysis
\input{generated/tables/efficiency}

\paragraph{Failure analysis.}
\PaperFailureAnalysis
\else
This development build specifies the locked analyses but intentionally contains
no empirical values or comparative conclusions.
\fi
```

`\PaperRepresentationAnalysis`, `\PaperAllocatorAnalysis`, `\PaperEfficiencyAnalysis`, and `\PaperFailureAnalysis` are complete sentences emitted by `render.py` from validated rows and gate outcomes. In submission mode, the false branch is a hard error before LaTeX runs. The development sentence is omitted from the packaged submission because `\ifPaperResultsAvailable` is true.

- [ ] **Step 5: Write the ethics and data-responsibility section**

```tex
% paper/sections/ethics.tex
\section{Ethical Considerations}
Personalization can increase the ease of identity imitation, non-consensual
editing, harassment, and deceptive impersonation.  A smaller or faster state
representation does not remove those risks and may lower deployment barriers.
We therefore separate technical storage claims from claims of safe use, document
dataset licenses and collection/consent status, replace real identity names with
anonymous tokens, and exclude withdrawn or unaudited identity sources from the
scientific cohort.  The data card records filtering, demographic and domain
limitations, and any applicable institutional review or consent process.

The paired human study is blinded and uses an approved protocol, minimal retained
metadata, and aggregate reporting.  Generated examples are screened only under
the preregistered rule; screening is not used to hide method failures.  Deployment
would additionally require access controls, abuse monitoring, provenance or
watermarking appropriate to the application, and a process for data-subject
requests.  A \textsc{Delete} event reclaims the handle's active online record and
unreferenced packets, but it is not machine unlearning: shared directions used by
other concepts and information in the foundation model remain.
```

- [ ] **Step 6: Write candid limitations and conclusion**

```tex
% paper/sections/limitations.tex
\section{Limitations}
The amortizer may be the dominant bottleneck, especially when public support data
cannot match proprietary identity supervision.  Useful packet reuse is an
empirical premise: visually distinct concepts may share too little residual
structure for the added incidence metadata to pay off.  The calibrated coverage
surrogate may also disagree with full-generation identity or prompt quality; its
calibration error and downstream outcomes must both be reported.

Closest baselines may not all port faithfully to SANA; because this work has no
reviewed SDXL RateMem implementation, a failed required SANA port blocks the
affected primary claim rather than triggering a backbone switch.
Foundation-model and evaluator pretraining make absolute unseen-concept claims
difficult; the strongest claim therefore requires a controlled post-checkpoint
cohort and an explicit contamination audit.  Deletion manages active online state
and is not machine unlearning.  The capped engineering pilot budget can establish
compatibility and cost, not the scientific result.  Finally, the snapshot theorem
is only relative to the deterministically reduced candidate set; it does not
cover full-pool pre-screen loss, outer admission, switching, or future-aware trace
regret.
```

```tex
% paper/sections/conclusion.tex
\section{Conclusion}
We formulate personalization memory as a causal lifecycle under an exact
serialized-byte budget and develop a progressive representation in which one
immutable packet can refine several concept adapters.  This coupling makes
packet allocation nonseparable while preserving auditable state transitions and
a narrowly stated fixed-cohort coverage guarantee.  The accompanying benchmark
tests the representation and allocator against matched independent, shared, and
feature-cache controls.  \ifPaperResultsAvailable\PaperConclusionEvidence\fi
```

Add `\input{sections/ethics}` between limitations and conclusion in `paper/main.tex`. `\PaperConclusionEvidence` is generated and may summarize only gates that passed.

- [ ] **Step 7: Run the gating tests and commit the complete main-paper prose**

Run:

```bash
uv run pytest tests/contract/paper/test_claim_gating.py -q
make -C paper draft
```

Expected: all tests pass; the draft PDF contains no empirical number, unsupported positive comparison, or missing section.

```bash
git add paper/main.tex paper/sections/evaluation.tex paper/sections/results.tex paper/sections/ethics.tex paper/sections/limitations.tex paper/sections/conclusion.tex tests/contract/paper/test_claim_gating.py
git commit -m "docs(paper): complete gated CVPR narrative"
```

### Task 8: Draw the architecture and lifecycle figures as true vector graphics

**Files:**
- Create: `paper/figures/vector/styles.tex`
- Create: `paper/figures/vector/overview.tex`
- Create: `paper/figures/vector/memory_ledger.tex`
- Create: `paper/figures/vector/lifecycle_timeline.tex`
- Create: `src/ratemem/paper/figures.py`
- Test: `tests/integration/paper/test_vector_figures.py`

- [ ] **Step 1: Write a failing vector-output test**

```python
# tests/integration/paper/test_vector_figures.py
import subprocess
from pathlib import Path

import pytest

from ratemem.paper.figures import render_vector_figures


@pytest.mark.parametrize("name", ["overview", "memory_ledger", "lifecycle_timeline"])
def test_architecture_figure_is_pdf_vector(name: str, tmp_path: Path) -> None:
    output = render_vector_figures(Path("paper/figures/vector"), tmp_path)
    pdf = output / f"{name}.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-")
    images = subprocess.run(
        ["pdfimages", "-list", str(pdf)], check=True, text=True, capture_output=True
    ).stdout.splitlines()[2:]
    assert images == []
    fonts = subprocess.run(
        ["pdffonts", str(pdf)], check=True, text=True, capture_output=True
    ).stdout
    assert " no " not in fonts.lower()
```

- [ ] **Step 2: Run the figure test and verify the expected failure**

Run: `uv run pytest tests/integration/paper/test_vector_figures.py -q`

Expected: collection fails because `ratemem.paper.figures` and the TikZ sources do not exist.

- [ ] **Step 3: Define one restrained, color-blind-safe TikZ vocabulary**

```tex
% paper/figures/vector/styles.tex
\definecolor{rmblue}{HTML}{3569B7}
\definecolor{rmorange}{HTML}{D97A22}
\definecolor{rmgreen}{HTML}{3C8D6B}
\definecolor{rmgray}{HTML}{667085}
\definecolor{rmlight}{HTML}{F2F4F7}
\tikzset{
  >=Latex,
  block/.style={draw=rmgray, rounded corners=1.5pt, fill=rmlight,
    minimum height=5.5mm, align=center, font=\scriptsize, inner sep=3pt},
  base/.style={block, draw=rmblue, fill=rmblue!10},
  packet/.style={block, draw=rmorange, fill=rmorange!12},
  event/.style={block, draw=rmgreen, fill=rmgreen!10},
  flow/.style={->, semithick, draw=rmgray},
  shared/.style={->, semithick, draw=rmorange},
  probe/.style={->, semithick, dashed, draw=rmgreen},
  bytebox/.style={draw=rmgray, rounded corners=2pt, inner sep=4pt},
}
```

- [ ] **Step 4: Draw the private-versus-shared overview**

```tex
% paper/figures/vector/overview.tex
\documentclass[tikz,border=2pt]{standalone}
\usepackage{amsmath}
\input{styles}
\begin{document}
\begin{tikzpicture}[node distance=3.5mm and 5mm]
  \node[font=\bfseries\small] (private-title) {Private progressive codes};
  \node[base, below=of private-title] (pb1) {concept A\\base};
  \node[packet, right=of pb1] (pp1) {private\\packet};
  \node[base, below=of pb1] (pb2) {concept B\\base};
  \node[packet, right=of pb2] (pp2) {private\\packet};
  \draw[flow] (pb1)--(pp1); \draw[flow] (pb2)--(pp2);

  \node[font=\bfseries\small, right=24mm of private-title] (shared-title) {RateMem-DiT};
  \node[base, below=of shared-title] (sb1) {concept A\\base};
  \node[base, below=of sb1] (sb2) {concept B\\base};
  \node[packet, right=10mm of $(sb1)!0.5!(sb2)$] (sp) {immutable shared\\packet payload};
  \draw[shared] (sb1.east) to[bend left=10] node[above, font=\tiny] {incidence + gain} (sp.west);
  \draw[shared] (sb2.east) to[bend right=10] node[below, font=\tiny] {incidence + gain} (sp.west);
  \node[bytebox, fit=(sb1)(sb2)(sp), label={[font=\tiny]below:one payload cost, two concept benefits}] {};
\end{tikzpicture}
\end{document}
```

The main caption must explain that colors encode roles but arrow geometry and labels remain sufficient in grayscale. It must not say packets are always reusable; reuse is the falsifiable hypothesis.

- [ ] **Step 5: Draw the byte ledger and lifecycle timeline**

Use these complete semantic elements in `memory_ledger.tex`: a base-record region containing opaque handle, base code, usage/age, checksum, and alignment; a packet region containing payload, hash, and reference count; an incidence region containing concept ID, packet ID, and quantized gain; a separate dashed region for shared trained weights excluded from online `B`; and a bracket labeled `canonical serialized bytes <= B` spanning only mutable online regions.

Use these complete elements in `lifecycle_timeline.tex`:

```tex
\node[event] (create) {\textsc{Create}\\mutates state};
\node[event, right=of create] (read1) {\textsc{Read}\\updates usage};
\node[event, right=of read1] (update) {\textsc{Update}\\mutates state};
\node[event, right=of update] (probe) {\textsc{Probe}\\copied snapshot};
\node[event, right=of probe] (delete) {\textsc{Delete}\\reclaims record};
\draw[flow] (create)--(read1)--(update);
\draw[probe] (update)--node[above, font=\tiny] {no usage or byte change} (probe);
\draw[flow] (probe)--(delete);
```

Add a lower state-hash line that changes at create/read/update/delete and remains identical across the probe. This makes non-mutating evaluation visually testable.

- [ ] **Step 6: Implement isolated rendering with the pinned Overleaf image**

`render_vector_figures(source, output)` runs one container per source with network disabled, a read-only source mount, and a writable temporary output mount. It invokes:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=/output paper/figures/vector/overview.tex
```

Repeat for `memory_ledger.tex` and `lifecycle_timeline.tex`, then copy only the three PDFs into `paper/figures/rendered/`. Record source and PDF SHA-256 values in `paper/figures/rendered/vector-provenance.json`.

- [ ] **Step 7: Include all three diagrams in the manuscript**

Insert the overview as the first two-column figure after the introduction, the byte ledger beside Eq.~\eqref{eq:byte-ledger}, and the timeline in the evaluation section. Use these captions:

```tex
\caption{Private progressive storage pays for one enhancement payload per concept (left). RateMem-DiT stores immutable content-addressed packets once and records compact incidences and gains for each dependent concept (right). Whether this sharing improves the quality--byte frontier is evaluated rather than assumed.}
```

```tex
\caption{Exact online-state ledger. Every mutable record inside the bracket contributes to budget \(B\); frozen shared weights are reported separately and amortized by active-set size.}
```

```tex
\caption{Serialized lifecycle replay. Operational reads update usage, whereas scoring probes operate on copied snapshots and leave both usage and serialized bytes unchanged.}
```

- [ ] **Step 8: Run vector checks and commit**

Run:

```bash
uv run ratemem-paper figures vector --source paper/figures/vector --output paper/figures/rendered
uv run pytest tests/integration/paper/test_vector_figures.py -q
make -C paper draft
```

Expected: three PDFs render; `pdfimages -list` reports no embedded raster image for any of them; all fonts are embedded; the main paper compiles.

```bash
git add paper/figures/vector src/ratemem/paper/figures.py tests/integration/paper/test_vector_figures.py paper/sections
git commit -m "docs(paper): add vector RateMem diagrams"
```

### Task 9: Generate result curves and preregistered qualitative panels

**Files:**
- Modify: `src/ratemem/paper/figures.py`
- Create: `src/ratemem/paper/qualitative.py`
- Modify: `src/ratemem/paper/render.py`
- Modify: `paper/sections/results.tex`
- Create: `tests/unit/paper/test_result_figures.py`
- Create: `tests/unit/paper/test_qualitative.py`

- [ ] **Step 1: Write failing tests for artifact-only plots and image hashes**

```python
# tests/unit/paper/test_result_figures.py
def test_result_plots_are_pdf_and_bound_to_release(valid_release, tmp_path):
    outputs = render_result_plots(valid_release, tmp_path)
    assert {path.name for path in outputs} == {
        "quality_bytes.pdf",
        "oracle_regret.pdf",
        "quality_wallclock_energy.pdf",
    }
    assert all(path.read_bytes().startswith(b"%PDF-") for path in outputs)
    provenance = json.loads((tmp_path / "plot-provenance.json").read_text())
    assert provenance["release_id"] == valid_release.release_id
    assert set(provenance["input_sha256"]) == {
        "curves/quality_bytes.csv",
        "curves/oracle_regret.csv",
        "curves/quality_wallclock_energy.csv",
    }
```

```python
# tests/unit/paper/test_qualitative.py
def test_qualitative_requires_locked_hashes_and_a_failure(valid_release, tmp_path):
    output = render_qualitative(valid_release, tmp_path)
    assert output.name == "qualitative_grid.pdf"
    assert output.read_bytes().startswith(b"%PDF-")


def test_tampered_qualitative_image_is_rejected(valid_release, tmp_path):
    image = next(valid_release.root.glob("qualitative/images/*"))
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(QualitativeError, match="image checksum mismatch"):
        render_qualitative(valid_release, tmp_path)


def test_grid_without_declared_failure_case_is_rejected(valid_release, tmp_path):
    set_all_failure_flags(valid_release, False)
    refresh_release_manifest(valid_release)
    with pytest.raises(QualitativeError, match="at least one locked failure case"):
        render_qualitative(valid_release, tmp_path)
```

- [ ] **Step 2: Run the plot and qualitative tests and verify failure**

Run: `uv run pytest tests/unit/paper/test_result_figures.py tests/unit/paper/test_qualitative.py -q`

Expected: FAIL because plot and qualitative renderers do not exist.

- [ ] **Step 3: Render accessible vector curves from the exact CSVs**

Set Matplotlib once:

```python
MATPLOTLIB_STYLE = {
    "font.family": "serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}
METHOD_STYLES = {
    "ratemem": {"color": "#3569B7", "marker": "o", "linestyle": "-"},
    "strongest_control": {"color": "#D97A22", "marker": "s", "linestyle": "--"},
    "offline_oracle": {"color": "#3C8D6B", "marker": "^", "linestyle": ":"},
}
```

`quality_bytes.pdf` shows identity/prompt/utility with confidence bands over exact bytes for uniform and Zipf requests. `oracle_regret.pdf` shows event-index regret with confidence bands, never a smoothed or reordered trace. `quality_wallclock_energy.pdf` plots the locked quality endpoint against wall-clock and energy, annotating optimization steps and search GPU-hours so methods are not forced into equal-step comparisons. Use marker and line style as redundant encodings; order methods by the locked baseline registry, not final performance.

Reject NaN/Inf, negative byte/time/energy values, duplicate keys, missing locked methods, inconsistent hardware IDs within a latency panel, or curve rows not listed in `artifact_manifest.json`.

- [ ] **Step 4: Validate the exact qualitative-selection contract**

Parse `qualitative/selection_manifest.json` with this shape:

```json
{
  "schema_version": "1.0",
  "rule_id": "fixture_failure_stratified_v1",
  "lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "panels": [
    {
      "panel_id": "fixture-panel-001",
      "dataset_id": "fixture-dataset",
      "concept_token": "concept-001",
      "prompt_id": "fixture-prompt-001",
      "seed": 17,
      "methods": [
        {
          "method_id": "ratemem",
          "image_path": "qualitative/images/fixture-panel-001-ratemem.png",
          "image_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          "artifact_id": "fixture-attempt-001"
        }
      ],
      "failure_case": true,
      "selection_score": 0.0,
      "selected_rank": 1
    }
  ]
}
```

The example lives only in test construction; the scientific release contains real values. Validate `rule_id` and `lock_sha256` against the evaluation lock; ensure panel IDs, prompts, seeds, and method order are unique and prespecified; verify every image and source artifact hash; require at least one declared failure; and reject panels selected after unblinding or with a rank outside the locked rule.

- [ ] **Step 5: Compose without retouching or favorable reselection**

Load each source image in read-only mode, verify its decoded dimensions/mode and SHA-256, and place it on a Matplotlib grid with vector labels. The only permitted display operations are colorspace conversion to sRGB, uniform scale-to-fit, and neutral padding recorded in `qualitative-provenance.json`. Do not denoise, sharpen, inpaint, alter faces, replace backgrounds, change prompts/seeds, or crop differently by method. Show the reference/support column, prompt token, all locked methods, and a visible `failure case` tag where declared.

- [ ] **Step 6: Wire generated figures into results**

Add these exact includes under `\ifPaperResultsAvailable`:

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{generated/plots/quality_bytes.pdf}
  \caption{Request-weighted personalization quality against exact serialized online-state bytes. Points and intervals are generated from the locked paired release; method order and primary budget cells were fixed before final evaluation.}
  \label{fig:quality-bytes}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{generated/qualitative/qualitative_grid.pdf}
  \caption{Qualitative cases selected by the preregistered manifest, including declared failures. Prompts, seeds, method order, and source-image hashes are fixed by the paper release; images receive no method-specific editing.}
  \label{fig:qualitative}
\end{figure*}
```

Place eviction-regret and quality-versus-wall-clock/energy plots in the supplement unless one replaces, rather than adds to, a weaker main-paper panel under the section budget.

- [ ] **Step 7: Run artifact-tamper and render tests, then commit**

Run:

```bash
uv run pytest tests/unit/paper/test_result_figures.py tests/unit/paper/test_qualitative.py -q
uv run ratemem-paper release build \
  --mode submission \
  --release artifacts/paper/cvpr2027-submission-v1 \
  --schema schemas/paper-release.schema.json \
  --claims paper/claims.yaml \
  --paper-id 12345 \
  --output paper/generated
```

Expected during fixture development: tests pass. The real command exits 2 until a valid release exists; after publication it writes the three plot PDFs, qualitative PDF, TeX tables/macros, and provenance without reading any path outside the release and locked configs. `12345` is used only for a local non-submission integration build; the real submission command receives the assigned OpenReview ID.

```bash
git add src/ratemem/paper/figures.py src/ratemem/paper/qualitative.py src/ratemem/paper/render.py paper/sections/results.tex tests/unit/paper/test_result_figures.py tests/unit/paper/test_qualitative.py
git commit -m "feat(paper): render artifact-locked result figures"
```

### Task 10: Build a separate anonymous supplement and reproducibility record

**Files:**
- Create: `paper/supplement.tex`
- Modify: `paper/supplement/theorem.tex`
- Create: `paper/supplement/reproducibility.tex`
- Create: `paper/supplement/data_and_evaluators.tex`
- Create: `paper/supplement/baselines.tex`
- Create: `paper/supplement/additional_results.tex`
- Create: `paper/supplement/qualitative.tex`
- Test: `tests/contract/paper/test_supplement.py`

- [ ] **Step 1: Write a failing supplement-isolation test**

```python
# tests/contract/paper/test_supplement.py
from pathlib import Path


def test_supplement_is_separate_and_anonymous() -> None:
    main = Path("paper/main.tex").read_text()
    supplement = Path("paper/supplement.tex").read_text()
    assert "supplement.tex" not in main
    assert r"\usepackage[review]{cvpr}" in supplement
    assert r"\input{generated/results_macros}" in supplement
    assert "Anonymous CVPR submission" in supplement
    assert r"\input{supplement/reproducibility}" in supplement
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run: `uv run pytest tests/contract/paper/test_supplement.py -q`

Expected: FAIL because the supplement entry point is absent.

- [ ] **Step 3: Create the standalone review-mode supplement**

```tex
% paper/supplement.tex
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
\input{preamble}
\input{generated/build_config}
\input{generated/results_macros}
\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=cvprblue]{hyperref}
\def\confName{CVPR}
\def\confYear{\PaperTemplateYear}
\title{RateMem-DiT: Supplementary Material}
\author{Anonymous CVPR submission}
\begin{document}
\maketitle
\appendix
\input{supplement/theorem}
\input{supplement/reproducibility}
\input{supplement/data_and_evaluators}
\input{supplement/baselines}
\ifPaperResultsAvailable
\input{supplement/additional_results}
\input{supplement/qualitative}
\fi
{\small\bibliographystyle{ieeenat_fullname}\bibliography{references}}
\end{document}
```

- [ ] **Step 4: Write the exact reproducibility inventory**

`reproducibility.tex` must name and explain the immutable revision/hash for the backbone, Diffusers/PEFT/container, dataset and evaluation locks, public trace manifests, encrypted final trace, baseline lock, exact serializer, generation settings, seeds, hardware, warm-up, power calculation, human-study blinding, statistical code, and paper-release manifest. It must state that operational reads mutate usage and probes do not, that support images are not free state, and that frozen weights are referenced rather than copied into attempt artifacts.

Use generated macros for every revision/hash/count. In draft mode, describe the required record types without showing a fabricated value. In submission mode, missing macros block release generation.

- [ ] **Step 5: Populate data, baseline, and extended-result appendices**

`data_and_evaluators.tex` renders license/consent, source revision, concept unit, split/deduplication, evaluator independence, and contamination disclosures from the locked data card. `baselines.tex` inputs `generated/tables/baseline_compliance.tex` and explains every incompatibility or contextual-only disposition without substituting published numbers or promoting SDXL evidence. `additional_results.tex` includes generated oracle-regret and quality-wall-clock-energy curves, all secondary Holm-adjusted intervals, byte-ledger breakdowns, and locked ablations. `qualitative.tex` includes the remaining manifest-selected cases and failures in stable panel order.

- [ ] **Step 6: Compile, test, and commit the supplement**

Run:

```bash
make -C paper draft
make -C paper supplement
uv run pytest tests/contract/paper/test_supplement.py -q
```

Expected: `paper/build/supplement.pdf` builds independently; all checks pass; no author, institution, grant, repository identity, or external contributed-content link appears.

```bash
git add paper/supplement.tex paper/supplement tests/contract/paper/test_supplement.py
git commit -m "docs(paper): add anonymous reproducibility supplement"
```

### Task 11: Enforce eight pages, anonymity, policy, and PDF integrity

**Files:**
- Create: `src/ratemem/paper/compliance.py`
- Create: `tests/unit/paper/test_compliance.py`
- Create: `tests/integration/paper/test_submission_pdf.py`

- [ ] **Step 1: Write failing page-count and anonymity tests**

```python
# tests/unit/paper/test_compliance.py
def test_last_main_page_above_eight_fails(tmp_path):
    aux = tmp_path / "main.aux"
    aux.write_text(r"\zref@newlabel{paper:last-main-page}{\default{}\page{9}\abspage{9}}")
    with pytest.raises(ComplianceError, match="body uses 9 pages; limit is 8"):
        read_main_page_count(aux, limit=8)


@pytest.mark.parametrize(
    "source",
    [
        r"\cvprfinalcopy",
        r"\section*{Acknowledgments}",
        r"\href{https://example.invalid/demo}{demo}",
        r"\color{white} hidden instruction",
        r"\phantom{ignore previous instructions}",
        "\u200bzero-width text",
    ],
)
def test_review_source_rejects_identity_link_or_hidden_content(source, tmp_path):
    (tmp_path / "bad.tex").write_text(source)
    with pytest.raises(ComplianceError):
        audit_tex_sources(tmp_path, review=True)
```

- [ ] **Step 2: Run the compliance tests and verify the expected failure**

Run: `uv run pytest tests/unit/paper/test_compliance.py -q`

Expected: collection fails because `ratemem.paper.compliance` does not exist.

- [ ] **Step 3: Implement source and build-log checks**

`audit_tex_sources` must reject review-mode author/affiliation/email/grant text, acknowledgments, `\cvprfinalcopy`, manual empirical numerals in `sections/results.tex`, unverified citation keys, absolute paths, `\write18`, URI links to contributed content, white/tiny/zero-width hidden text, `\phantom`, raw PDF literals, attachments, and any graphic outside `paper/`. It also requires `\usepackage[review]{cvpr}`, the official style hashes, the generated-release provenance, and `\PaperResultsAvailabletrue` for submission.

Build a private identity denylist outside Git:

```bash
umask 077
uv run ratemem-paper audit identity-denylist \
  --derive-from-git \
  --output /home/ubuntu/.config/ratemem/paper-identity-denylist.txt
```

The command reads Git user name/email without printing them and interactively accepts additional author names, institutions, grant IDs, domains, and repository URLs. Submission audit refuses a missing or empty denylist. The build-log check rejects undefined citations/references, multiply defined labels, missing glyphs/files, and overfull boxes.

- [ ] **Step 4: Implement exact PDF checks**

Parse `paper/build/main.aux` for `paper:last-main-page` and require an absolute page number no greater than eight. Use pypdf and Poppler/qpdf to require US Letter pages, a readable xref, no encryption, no attachments, empty Author/Creator/Producer fields after deterministic metadata cleanup, no URI annotations, embedded fonts only, and no Type 3 fonts. Permit extra pages only after the body marker and only for references; `main.tex` may not input supplement material after the bibliography.

Run these low-level commands inside the audit and save their output:

```bash
qpdf --check paper/build/main.pdf
pdfinfo paper/build/main.pdf
pdffonts paper/build/main.pdf
pdftotext -layout paper/build/main.pdf paper/build/main.txt
```

The high-level command prints only `PASS submission PDF: body<=8, review anonymous, references-only tail, fonts embedded, no external URI or hidden content` when every check succeeds.

- [ ] **Step 5: Add target-year policy and artifact regeneration checks**

Submission audit must rerun `template verify`, rebuild `paper/generated/` into a temporary directory from `artifacts/paper/cvpr2027-submission-v1/`, and byte-compare every generated TeX/PDF/provenance file with the tree used for the PDF. It must also rebuild the bibliography from `evidence.yaml`, rerun literature freshness, and verify the qualitative source hashes. Any mismatch proves a hand edit or stale artifact and exits 2.

- [ ] **Step 6: Run compliance tests and commit**

Run:

```bash
uv run pytest tests/unit/paper/test_compliance.py tests/integration/paper/test_submission_pdf.py -q
make -C paper draft
```

Expected: unit fixtures cover 8-page pass/9-page fail, bad font/metadata/link/hidden-text failures, and clean PDF pass; draft compile remains allowed without a scientific release.

```bash
git add src/ratemem/paper/compliance.py tests/unit/paper/test_compliance.py tests/integration/paper/test_submission_pdf.py
git commit -m "test(paper): enforce CVPR submission policy"
```

### Task 12: Render and sign off every PDF page visually

**Files:**
- Modify: `src/ratemem/paper/compliance.py`
- Create: `tests/unit/paper/test_visual_qa.py`

- [ ] **Step 1: Write a failing visual-QA receipt test**

```python
# tests/unit/paper/test_visual_qa.py
def test_visual_receipt_is_bound_to_exact_pdf(clean_pdf, tmp_path):
    receipt = render_visual_qa(clean_pdf, tmp_path, dpi=180)
    assert receipt["pdf_sha256"] == file_sha256(clean_pdf)
    assert len(receipt["page_images"]) == receipt["page_count"]
    assert (tmp_path / "contact-sheet.png").is_file()
    with pytest.raises(ComplianceError, match="visual review not signed off"):
        require_visual_signoff(receipt)
```

- [ ] **Step 2: Render deterministic page images and a contact sheet**

Implement `ratemem-paper audit render-pages` using `pdftoppm -png -r 180`; hash each page PNG and use Pillow only to assemble a labeled contact sheet. Write `paper/build/qa/render-manifest.json` with the PDF hash, page count, media boxes, page-image hashes, and renderer version.

Run:

```bash
uv run ratemem-paper audit render-pages \
  --pdf paper/build/main.pdf \
  --output paper/build/qa \
  --dpi 180
```

Expected: one `page-N.png` per PDF page, `contact-sheet.png`, and `render-manifest.json`; the command prints `PASS render: every PDF page rasterized at 180 dpi`.

- [ ] **Step 3: Inspect one main or reference page at readable scale and repeat**

Open `/home/ubuntu/.config/superpowers/worktrees/Memory-GAN/ratemem-implementation/paper/build/qa/contact-sheet.png`, then inspect the next unchecked `page-N.png` at full resolution. Record that page number and check title/anonymity, column order, crop and margin bounds, equation and algorithm breaks, table legibility, vector-label size, color/grayscale distinction, citation/back-reference rendering, no overlapping floats, no clipped text, no blank/duplicate page, qualitative image/prompt alignment, visible failure labels, and references beginning only after the main body. Repeat this checkbox once per page. Fix the source and repeat compilation/rendering from page 1 after any defect.

- [ ] **Step 4: Record an exact, hash-bound visual signoff**

After inspection, run:

```bash
uv run ratemem-paper audit sign-visual \
  --manifest paper/build/qa/render-manifest.json \
  --check margins \
  --check typography \
  --check equations \
  --check tables \
  --check vector-figures \
  --check qualitative-integrity \
  --check references \
  --output paper/build/qa/visual-qa.json
```

The command requires interactive confirmation for every page and check, records no personal reviewer identity in the review artifact, and writes a receipt bound to the PDF and page hashes. It cannot accept `--yes` or a partial page range.

- [ ] **Step 5: Require the receipt in submission audit and commit tests**

Run: `uv run pytest tests/unit/paper/test_visual_qa.py -q`

Expected: all tests pass; changing one PDF byte or page image invalidates the signoff.

```bash
git add src/ratemem/paper/compliance.py tests/unit/paper/test_visual_qa.py
git commit -m "test(paper): require hash-bound visual PDF review"
```

### Task 13: Back up the original Overleaf project and import a verified new copy

**Files:**
- Create: `src/ratemem/paper/overleaf.py`
- Create: `tests/unit/paper/test_overleaf.py`
- Create: `tests/integration/paper/test_overleaf_package.py`

- [ ] **Step 1: Write failing safety tests before any live request**

```python
# tests/unit/paper/test_overleaf.py
SOURCE_ID = "6a8b44fb070db27221ef64a0"


def test_sync_backs_up_before_import_and_never_mutates_source(fake_overleaf, paper_tree, tmp_path):
    receipt = sync_as_new_project(
        session=fake_overleaf,
        base_url="http://127.0.0.1:8380",
        source_project_id=SOURCE_ID,
        expected_source_name="Memory-MetaGAN: A Memory-based Few-shot GAN",
        paper_root=paper_tree,
        backup_root=tmp_path / "backups",
        new_project_name="RateMem-DiT CVPR 2027 Draft",
    )
    assert receipt.backup_verified_at < receipt.import_started_at
    assert receipt.new_project_id != SOURCE_ID
    mutating = {"POST", "PUT", "PATCH", "DELETE"}
    assert all(
        not (request.method in mutating and SOURCE_ID in request.path)
        for request in fake_overleaf.requests
    )
    assert fake_overleaf.canonical_manifest(SOURCE_ID) == receipt.source_manifest_before
    assert receipt.source_manifest_before == receipt.source_manifest_after


def test_sync_refuses_non_loopback_without_explicit_remote_flag(paper_tree, tmp_path):
    with pytest.raises(OverleafSafetyError, match="loopback"):
        sync_as_new_project(base_url="https://latex.example.org", paper_root=paper_tree, backup_root=tmp_path)


@pytest.mark.parametrize(
    "failure_point",
    ["backup_download", "package_verify", "new_project_upload", "new_project_compile", "pdf_download"],
)
def test_every_failure_path_leaves_source_manifest_unchanged(
    fake_overleaf, paper_tree, tmp_path, failure_point
):
    before = fake_overleaf.canonical_manifest(SOURCE_ID)
    fake_overleaf.fail_at(failure_point)
    with pytest.raises(OverleafSyncError):
        sync_as_new_project(
            session=fake_overleaf,
            base_url="http://127.0.0.1:8380",
            source_project_id=SOURCE_ID,
            expected_source_name="Memory-MetaGAN: A Memory-based Few-shot GAN",
            paper_root=paper_tree,
            backup_root=tmp_path / "backups",
            new_project_name="RateMem-DiT CVPR 2027 Draft",
        )
    assert fake_overleaf.canonical_manifest(SOURCE_ID) == before
```

- [ ] **Step 2: Run the Overleaf tests and verify the expected failure**

Run: `uv run pytest tests/unit/paper/test_overleaf.py tests/integration/paper/test_overleaf_package.py -q`

Expected: collection fails because `ratemem.paper.overleaf` does not exist.

- [ ] **Step 3: Implement non-echoing local authentication**

Use `requests.Session`. GET `/login`, parse the hidden `_csrf` value, prompt for email with `input`, prompt for password with `getpass.getpass`, and POST `/login` with the same session. Reject a redirect back to `/login`. Never accept password/token arguments, environment variables, repository files, logs, or receipts; never print response cookies.

- [ ] **Step 4: Download and verify the live original before import**

GET `/Project/6a8b44fb070db27221ef64a0/download/zip`. Create a new mode-0700 directory under `/home/ubuntu/overleaf-backups/ratemem-dit/` using an RFC3339 UTC timestamp and `exist_ok=False`. Save `original-project.zip`, verify `ZipFile.testzip()` returns `None`, and write `original-manifest.json` with the source ID/name, download time, zip SHA-256, and sorted per-entry hashes/sizes. Also hash `/home/ubuntu/memory-metagan-original/.project-sync-state` and record its local archival relationship without modifying that directory.

Abort before creating a package or POSTing anything if download, title/ID check, zip validation, manifest write/fsync, or backup reread fails.

- [ ] **Step 5: Package an allowlisted, self-contained paper root**

The zip root must contain `main.tex`, `supplement.tex`, `preamble.tex`, `latexmkrc`, `cvpr.sty`, `ieeenat_fullname.bst`, `references.bib`, `sections/**/*.tex`, `supplement/**/*.tex`, `figures/rendered/*.pdf`, and `generated/**/*`. Exclude `.git`, scripts, raw artifacts, credentials, logs, aux files, QA page PNGs, and the built submission PDF. Write `OVERLEAF_PACKAGE_MANIFEST.json` inside the archive with sorted per-file hashes and the validated scientific release ID/hash. Reject symlinks, absolute paths, parent traversal, files above the configured size limit, and any generated-tree mismatch.

- [ ] **Step 6: Import only as a new project and compile it through Overleaf**

POST `/project/new/upload` with multipart field `qqfile`, form field `name=RateMem-DiT CVPR 2027 Draft.zip`, and the CSRF token. This supported endpoint creates a new project. Centralize every HTTP call behind a request guard whose mutating allowlist contains only login, this exact new-project upload path, and compile requests whose project ID equals the newly returned ID. Reject every `POST`, `PUT`, `PATCH`, or `DELETE` whose URL contains the source ID, plus every project upload/delete/archive/rename endpoint and every direct Mongo operation. Require JSON `success: true` and a new 24-hex project ID distinct from the source.

POST the path constructed as `f"/project/{new_project_id}/compile"` with JSON:

```json
{
  "rootResourcePath": "main.tex",
  "compiler": "pdflatex",
  "check": "error",
  "stopOnFirstError": true
}
```

Require compile status `success`, download the path constructed as `f"/download/project/{new_project_id}/build/{build_id}/output/output.pdf"`, run the PDF compliance checks, and save it in the backup receipt directory. Redownload the new project zip and compare its canonical content manifest with the uploaded package.

- [ ] **Step 7: Prove the original remained unchanged and write the receipt**

Redownload the source project, compute per-entry hashes, and require an exact match with `original-manifest.json`. Write `sync-receipt.json` outside the repository with source/new IDs, names, backup/package/imported/PDF hashes, API paths/methods excluding login, compile build ID, timestamps, and `source_unchanged: true`. If comparison fails, report the discrepancy and stop; do not attempt an automatic rollback.

- [ ] **Step 8: Run tests, then perform the live backup-first import**

Run tests first:

```bash
uv run pytest tests/unit/paper/test_overleaf.py tests/integration/paper/test_overleaf_package.py -q
```

Expected: fake-server tests prove backup-before-import ordering and that no mutating request targets the source ID.

After the submission PDF and visual receipt pass:

```bash
umask 077
uv run ratemem-paper overleaf sync-new \
  --base-url http://127.0.0.1:8380 \
  --source-project-id 6a8b44fb070db27221ef64a0 \
  --expected-source-name "Memory-MetaGAN: A Memory-based Few-shot GAN" \
  --paper paper \
  --release artifacts/paper/cvpr2027-submission-v1 \
  --visual-receipt paper/build/qa/visual-qa.json \
  --backup-root /home/ubuntu/overleaf-backups/ratemem-dit \
  --new-project-name "RateMem-DiT CVPR 2027 Draft"
```

Expected final lines:

```text
PASS backup: original project zip and canonical manifest verified
PASS import: separate RateMem-DiT project created and compiled
PASS source: original project canonical manifest unchanged
```

The command then prints the new local project URL. The original project remains available at `http://127.0.0.1:8380/project/6a8b44fb070db27221ef64a0`.

- [ ] **Step 9: Commit the tested sync implementation, not receipts or credentials**

```bash
git add src/ratemem/paper/overleaf.py tests/unit/paper/test_overleaf.py tests/integration/paper/test_overleaf_package.py
git commit -m "feat(paper): add backup-first Overleaf import"
```

### Task 14: Run the final reproducible submission gate

**Files:**
- Modify: `paper/README.md`
- Test: all paper tests and full project regression suite

- [ ] **Step 1: Regenerate from a clean generated/build tree**

Run:

```bash
make -C paper clean
make -C paper submission PAPER_ID="$PAPER_ID"
make -C paper supplement
```

Expected: both PDFs rebuild from the validated release; submission audit exits 0. Set `PAPER_ID` in the invoking environment to the assigned OpenReview ID; the command refuses an absent value.

- [ ] **Step 2: Run focused and full regression tests**

Run:

```bash
uv run pytest tests/unit/paper tests/contract/paper tests/integration/paper -q
uv run pytest -q
uv run ruff check src/ratemem/paper tests/unit/paper tests/contract/paper tests/integration/paper
uv run mypy src/ratemem/paper
```

Expected: all tests pass, Ruff emits no findings, and mypy reports success.

- [ ] **Step 3: Re-run live currency, evidence, and artifact regeneration checks**

Run:

```bash
uv run ratemem-paper template verify --source paper/template/CVPR_TEMPLATE_SOURCE.json --policy paper/config/cvpr-policy.yaml
uv run ratemem-paper literature freshness --audit paper/related_work/literature-audit.json --max-age-days 30
uv run ratemem-paper audit submission --source paper --pdf paper/build/main.pdf --policy paper/config/cvpr-policy.yaml
```

Expected: three PASS results. If CVPR 2027 material exists, stop, vendor the new official files, update the source/policy lock and `\confYear`, and rerun every paper test before proceeding.

- [ ] **Step 4: Perform final visual QA on both PDFs**

Render and inspect `main.pdf` and `supplement.pdf` independently at 180 dpi. Record separate hash-bound receipts. Expected: every page is legible at 100% zoom, diagrams remain vector-sharp, qualitative grids match the locked manifest, and the main-body marker is at or before page eight.

- [ ] **Step 5: Create a self-contained submission archive and verify it**

Run `ratemem-paper package submission` to create `paper/build/ratemem-cvpr2027-submission.zip` containing the main PDF, supplement PDF, anonymous LaTeX source package, release provenance, template lock, literature audit, and both visual receipts. The archive excludes raw identity data, credentials, private final-trace keys, author information, and the original manuscript.

Run: `uv run ratemem-paper package verify --archive paper/build/ratemem-cvpr2027-submission.zip`

Expected: `PASS package: hashes, anonymity, eight-page body, references-only tail, artifact regeneration, and visual receipts verified`.

- [ ] **Step 6: Commit the final documentation after all gates pass**

```bash
git add paper/README.md
git commit -m "docs(paper): document CVPR submission workflow"
```

Do not commit generated result files, PDFs, paper-release artifacts, visual receipts, Overleaf backups, sync receipts, credentials, or author identity data.
