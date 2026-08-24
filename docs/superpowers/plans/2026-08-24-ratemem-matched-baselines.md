# RateMem Matched Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the byte-exact, causally matched baselines and fidelity audit needed to falsify RateMem on the same frozen SANA-1.5 backbone, amortizer, adapter basis, candidate stream, lifecycle traces, and search budget, while keeping incompatible and different-backbone prior work out of the primary comparison.

**Architecture:** Every runnable method implements one typed `BaselineAdapter` lifecycle protocol and exports canonical online-state bytes that the host, rather than the method, measures. This plan freezes a provider-neutral shared-input schema for the SANA route and verifies native controls against locked synthetic bundles; after the baseline/evaluation locks, the learned workflow materializes one real SANA bundle that RateMem and matched controls share. Original repositories run out of process through a strict JSONL bridge and must pass source, license, state-roundtrip, and locked fidelity audits. The primary backbone is immutably `sana_1_5_1_6b`: if any matched-required SANA control is not faithful, the affected primary comparison and claim are blocked. SDXL-native implementations and published numbers remain contextual-only and can never satisfy, replace, or trigger a fallback for a missing SANA comparator; promoting SDXL would require a separately reviewed future RateMem-SDXL extension outside this plan.

**Tech Stack:** Python 3.11, `uv`, PyTorch 2.8, Diffusers 0.35.1, NumPy 2.2.6, SciPy 1.16.1, Pydantic 2.11.7, canonical CBOR, safetensors, PyYAML 6.0.2, JSON Schema with `jsonschema==4.25.1`, Typer, pytest, Hypothesis, Ruff, mypy

---

## Execution boundary and handoff order

This is a companion to, not a replacement for, the core, SANA/pilot, and scientific-evaluation plans. Execute it in this order:

1. Complete `docs/superpowers/plans/2026-08-24-ratemem-core-memory.md` and freeze its state, codec, allocator, and lifecycle interfaces.
2. Complete the code-facing tasks in `docs/superpowers/plans/2026-08-24-ratemem-sana-modal-pilot.md`; no engineering-pilot artifact is scientific evidence.
3. Complete scientific-evaluation Tasks 1--7 and Task 8 Steps 1--5 so the dataset, traces, evaluators, margins, budgets, registry types, `BaselineAdapter` re-export, and baseline requirements are fixed. At this boundary, `configs/scientific/evaluation-lock.yaml` must still be blocked because no baseline fidelity lock exists.
4. Execute this plan. CPU implementation and fidelity work uses locked synthetic shared-input bundles and does not require learned weights. If a real-checkpoint structural fidelity case requires paid GPU time, consume only the scientific plan's narrow pre-lock `baseline_fidelity` permit: it is bound to the dataset lock, comparator catalog, fidelity-policy hash, and held-in/calibration inputs; it cannot open validation/final traces, select a method, train RateMem, tune a baseline, or compute a claim metric. It uses one explicitly selected workspace whose outer budget is verified at USD 28.00, shares the project-wide `known usage + pending worst case + new reservation <= USD 27.00` ledger, reconciles immediately, and forbids workspace rotation or fallback. Then this plan audits source, implementation, state/ledger, synthetic-provider, and structural fidelity evidence, verifies that every matched-required SANA control is faithful, and produces the inputs consumed by scientific Task 8's sealing command. A missing or unfaithful required SANA control blocks the applicable primary claim; no SDXL artifact is consulted as a substitute.
5. Return to scientific Task 8 Steps 6--7 to seal `configs/scientific/baseline-lock.yaml` and then `configs/scientific/evaluation-lock.yaml`. The baseline lock freezes implementations, source/fidelity receipts, exact-byte rules, the fixed SANA primary-backbone decision, the shared-input schema, and search policy/budgets; it contains no learned weight hash, real candidate-stream hash, tuned configuration, validation score, or search outcome.
6. Full scientific authorization remains scientific Task 9 and requires both locks plus the learned-method CPU gate. Only after that authorization may the workflow train RateMem/dictionaries, materialize one real `SharedInputBundle`, train/tune baselines under the frozen search policy, emit search receipts, freeze selected configurations, and replay the final trace. A learned-method manifest whose amortizer, basis, codec, or candidate-stream hashes differ from its matched controls is rejected at that later freeze.

This plan does not activate credentials, choose a paid workspace, launch Modal, run a scientific GPU job, enter a number into a paper table, or copy external source into the repository. GPU fidelity cases are emitted as signed job specifications for the separately authorized scientific-compute workflow. All commands in this plan are safe to run locally unless they explicitly say they consume an already authorized scientific artifact.

## File responsibility map

| Path | Responsibility |
|---|---|
| `configs/baselines/literature-classification.yaml` | Prespecified matched-required, contextual-only, and incompatible dispositions; never inferred from results. |
| `configs/baselines/backbones.yaml` | Fixed SANA primary checkpoint and shared-input requirements, plus non-promotable contextual SDXL evidence identity. |
| `configs/baselines/policy-search.yaml` | Equal validation-only trial/GPU-hour budget and method search spaces. |
| `configs/baselines/fidelity-policy.yaml` | Source, algebraic, state-roundtrip, backbone, and output-fidelity case definitions. |
| `src/ratemem/baselines/catalog.py` | Literature/control catalog parsing and primary-eligibility rules. |
| `src/ratemem/baselines/protocol.py` | Common comparison contract, event receipt, snapshot, probe result, adapter protocol, and causal event view. |
| `src/ratemem/baselines/ledger.py` | Canonical tensor/state encoding and host-computed exact byte ledger. |
| `src/ratemem/baselines/shared_inputs.py` | Immutable amortizer-code, basis, and candidate-packet bundles shared by RateMem and matched controls. |
| `src/ratemem/baselines/backbones.py` | SANA primary runner binding and a separate contextual-only SDXL evidence contract that cannot enter matched replay. |
| `src/ratemem/baselines/independent.py` | Uncompressed-code FIFO, LRU, and LRUA caches. |
| `src/ratemem/baselines/private_progressive.py` | Private progressive codec with causal size-aware and exact separable-rate policies. |
| `src/ratemem/baselines/shared_greedy.py` | Same-packet-stream plain marginal-density greedy allocator. |
| `src/ratemem/baselines/static_shared.py` | Frozen Compress-then-Serve-style and VB-LoRA-style code representations. |
| `src/ratemem/baselines/online_share.py` | Mutable SHARE-style shared subspace with reprojection and exact online-basis accounting. |
| `src/ratemem/baselines/feature_cache.py` | DreamCache-style feature-state lifecycle bridge. |
| `src/ratemem/baselines/stateless.py` | Explicitly nondeployable stateless amortizer control with external-support accounting. |
| `src/ratemem/baselines/lora_reference.py` | Per-concept optimization LoRA reference on the fixed SANA primary backbone. |
| `src/ratemem/baselines/oracles.py` | Exact append-only quantized-code and full-future packet upper references. |
| `src/ratemem/baselines/external_jsonl.py` | Sandboxed, schema-checked JSONL subprocess adapter for original implementations. |
| `src/ratemem/baselines/fidelity.py` | Immutable source/revision/license inventory, SANA structural-fidelity cases, frozen search policy, later search-ledger validation, audit, and fixed-primary gate. |
| `src/ratemem/baselines/replay.py` | Paired trace runner and per-event comparator artifacts. |
| `src/ratemem/baselines/cli.py` | `ratemem-baselines` inventory, fidelity, audit, shared-input, and replay commands. |
| `external_baselines/*/runner.py` | Thin JSONL wrappers around pinned, out-of-tree upstream checkouts; no upstream code is vendored. |
| `schemas/ratemem-baseline-*.schema.json` | Generated catalog, receipt, ledger, source, fidelity, and audit schemas. |
| `tests/baselines/` | Fast native algorithm, causal-access, byte-ledger, protocol, oracle, and SANA primary-gate tests. |
| `tests/contract/baselines/` | External-process, real-backbone opt-in, fidelity-evidence, and paired-replay contracts. |
| `tests/fixtures/baselines/` | Synthetic codes, local fake Git repositories, fake external worker, traces, and non-publication artifacts. |
| `docs/baselines.md` | Operator sequence, method disposition table, byte rules, fidelity rules, and failure semantics. |

### Task 1: Freeze the comparator catalog instead of promising every cited method

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `configs/baselines/literature-classification.yaml`
- Create: `src/ratemem/baselines/__init__.py`
- Create: `src/ratemem/baselines/catalog.py`
- Create: `schemas/ratemem-baseline-catalog-v1.schema.json`
- Test: `tests/baselines/test_catalog.py`

- [ ] **Step 1: Verify the predecessor environment and preserve its pins**

Run:

```bash
test "$(cat .python-version)" = "3.11.13"
uv run python - <<'PY'
from importlib.metadata import version

assert version("jsonschema") == "4.25.1"
assert version("numpy") == "2.2.6"
assert version("pydantic") == "2.11.7"
assert version("scipy") == "1.16.1"
print("PASS baseline dependency preflight")
PY
```

Expected: `PASS baseline dependency preflight`. If a predecessor is incomplete, stop rather than replacing its dependency groups. Add only this console script to the existing project table, then run `uv lock` without removing any dependency or extra:

```toml
[project.scripts]
ratemem-baselines = "ratemem.baselines.cli:main"
```

- [ ] **Step 2: Write the failing catalog tests**

```python
# tests/baselines/test_catalog.py
from pathlib import Path

import pytest

from ratemem.baselines.catalog import (
    CatalogError,
    ComparisonClass,
    REQUIRED_CONTROL_IDS,
    load_catalog,
)


CATALOG = Path("configs/baselines/literature-classification.yaml")


def test_every_required_control_is_present_once_and_has_an_implementation_mode() -> None:
    catalog = load_catalog(CATALOG)
    assert set(catalog.control_ids) == REQUIRED_CONTROL_IDS
    assert len(catalog.control_ids) == len(set(catalog.control_ids))
    assert all(control.implementation_mode in {"native", "external_jsonl"} for control in catalog.controls)


def test_literature_is_explicitly_partitioned_and_never_auto_promoted() -> None:
    catalog = load_catalog(CATALOG)
    assert len(catalog.literature) >= 24
    assert {item.comparison_class for item in catalog.literature} == set(ComparisonClass)
    assert all(item.reason_code and item.allowed_claims for item in catalog.literature)
    assert not any(item.primary_table for item in catalog.literature if item.comparison_class != ComparisonClass.MATCHED_REQUIRED)


def test_incompatible_or_contextual_method_cannot_be_selected_for_primary_table() -> None:
    catalog = load_catalog(CATALOG)
    for citation_key in ("vsm_diffusion_neurips2023", "moblora_acl2026", "rqt_acl2025", "sinelora_delta_aaai2026"):
        with pytest.raises(CatalogError, match="not eligible for primary matched table"):
            catalog.require_primary_eligible(citation_key)


def test_non_sana_sinelora_delta_is_contextual_only_and_has_no_claimed_port() -> None:
    catalog = load_catalog(CATALOG)
    item = next(row for row in catalog.literature if row.citation_key == "sinelora_delta_aaai2026")
    assert item.comparison_class == ComparisonClass.CONTEXTUAL_ONLY
    assert item.primary_table is False
    assert item.port_mode == "citation_only_sd3_medium"
    assert "sine_lora_delta_sdxl" not in catalog.control_ids
```

- [ ] **Step 3: Run the catalog test and verify the missing-module failure**

Run: `uv run pytest tests/baselines/test_catalog.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'ratemem.baselines'`.

- [ ] **Step 4: Add the complete prespecified control and literature disposition**

Create `configs/baselines/literature-classification.yaml` exactly as follows. `matched_required` means this project must supply a faithful SANA control or block the applicable primary comparison and claim; it does not mean the upstream implementation can be copied verbatim. `contextual_only` means only source-attributed discussion or a separate appendix row is allowed. SDXL-native work is contextual-only because this route has no reviewed SDXL RateMem implementation. `incompatible` means the method addresses a different state, task, or lifecycle and cannot enter the primary matched table.

```yaml
schema_version: "1.0"
controls:
  - {id: independent_fifo, implementation_mode: native, family: independent_code, primary_table: true, roles: [representation, allocator]}
  - {id: independent_lru, implementation_mode: native, family: independent_code, primary_table: true, roles: [representation, allocator]}
  - {id: independent_lrua, implementation_mode: native, family: independent_code, primary_table: true, roles: [allocator]}
  - {id: private_progressive_size_aware, implementation_mode: native, family: private_progressive, primary_table: true, roles: [representation, allocator]}
  - {id: private_progressive_separable_rate, implementation_mode: native, family: private_progressive, primary_table: true, roles: [representation, allocator]}
  - {id: shared_packet_plain_greedy, implementation_mode: native, family: shared_packet, primary_table: true, roles: [allocator]}
  - {id: cts_style_static, implementation_mode: native, family: static_shared_code, primary_table: true, roles: [representation]}
  - {id: vb_lora_style_static, implementation_mode: native, family: static_shared_code, primary_table: true, roles: [representation]}
  - {id: share_style_online, implementation_mode: native, family: online_shared_subspace, primary_table: true, roles: [representation, allocator]}
  - {id: dreamcache_feature_cache, implementation_mode: external_jsonl, family: feature_cache, primary_table: true, roles: [representation, optimization_free]}
  - {id: hyperlora_upstream, implementation_mode: external_jsonl, family: stateless_amortizer, primary_table: false, roles: [eligible_portrait_acquisition]}
  - {id: stateless_amortizer, implementation_mode: native, family: stateless_amortizer, primary_table: false, roles: [latency_upper_reference]}
  - {id: per_concept_lora, implementation_mode: native, family: per_concept_optimization, primary_table: true, roles: [optimization_free_tradeoff]}
  - {id: exact_append_only_quantized, implementation_mode: native, family: append_only_oracle, primary_table: false, roles: [upper_reference]}
  - {id: exact_future_trace_packets, implementation_mode: native, family: future_trace_oracle, primary_table: false, roles: [upper_reference]}
literature:
  - {citation_key: hyperlora_cvpr2025, title: "HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis", comparison_class: matched_required, port_mode: external_native_task, primary_table: false, allowed_claims: [eligible_portrait_acquisition], reason_code: portrait_only_external_control}
  - {citation_key: dreamcache_cvpr2025, title: "DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching", comparison_class: matched_required, port_mode: external_jsonl_bridge, primary_table: true, allowed_claims: [shared_representation, optimization_free_tradeoff], reason_code: closest_feature_cache}
  - {citation_key: share_eccv2026, title: "Shared LoRA Subspaces for almost Strict Continual Learning", comparison_class: matched_required, port_mode: clean_room_matrix_control_plus_external_fidelity, primary_table: true, allowed_claims: [shared_representation, causal_allocator], reason_code: closest_online_shared_subspace}
  - {citation_key: compress_then_serve_icml2025, title: "Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead", comparison_class: matched_required, port_mode: clean_room_code_space_control, primary_table: true, allowed_claims: [shared_representation], reason_code: closest_shared_basis_compression}
  - {citation_key: vb_lora_neurips2024, title: "VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks", comparison_class: matched_required, port_mode: clean_room_code_space_control, primary_table: true, allowed_claims: [shared_representation], reason_code: closest_shared_vector_bank}
  - {citation_key: lora_iclr2022, title: "LoRA: Low-Rank Adaptation of Large Language Models", comparison_class: matched_required, port_mode: native_diffusers_reference, primary_table: true, allowed_claims: [optimization_free_tradeoff], reason_code: standard_per_concept_reference}
  - {citation_key: sinelora_delta_aaai2026, title: "SineLoRA-Delta: Sine-Activated Delta Compression", comparison_class: contextual_only, port_mode: citation_only_sd3_medium, primary_table: false, allowed_claims: [secondary_rate_distortion], reason_code: sd3_medium_not_sana_and_no_auditable_upstream_implementation}
  - {citation_key: hyperdreambooth_cvpr2024, title: "HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [related_amortized_personalization], reason_code: different_adapter_target_and_training_contract}
  - {citation_key: lofa_2024, title: "LoFA: Any Subject in Any Style at Any Time", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [related_optimization_free_personalization], reason_code: no_locked_lifecycle_port}
  - {citation_key: textual_inversion_iclr2023, title: "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [optimization_reference], reason_code: token_state_not_matched_adapter_state}
  - {citation_key: dreambooth_cvpr2023, title: "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [optimization_reference], reason_code: full_model_state_not_primary_adapter_control}
  - {citation_key: custom_diffusion_cvpr2023, title: "Multi-Concept Customization of Text-to-Image Diffusion", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [optimization_reference, composition_context], reason_code: different_trainable_state}
  - {citation_key: ip_adapter_2023, title: "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [image_conditioning_context], reason_code: stateless_conditioning_not_persistent_memory}
  - {citation_key: photomaker_cvpr2024, title: "PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [portrait_context], reason_code: human_identity_only}
  - {citation_key: instantid_2024, title: "InstantID: Zero-shot Identity-Preserving Generation in Seconds", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [portrait_context], reason_code: human_identity_conditioning_not_adapter_memory}
  - {citation_key: blip_diffusion_cvpr2023, title: "BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [optimization_free_context], reason_code: different_conditioning_state}
  - {citation_key: cf_star_2026, title: "CF-STAR: Highly Compressible Adapters for Model Merging via Centralized Task Vectors", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [compression_context], reason_code: model_merging_and_noncausal_centering}
  - {citation_key: rqt_acl2025, title: "RQT: Hierarchical Residual Quantization for Multi-Model Compression", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [compression_context], reason_code: full_model_llm_tree_not_online_adapter_memory}
  - {citation_key: loraquant_2025, title: "LoRAQuant: Mixed-Precision Quantization of LoRA to Ultra-Low Bits", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [compression_context], reason_code: llm_post_training_quantization_without_lifecycle}
  - {citation_key: moblora_acl2026, title: "MoBLoRA: Mixture of Basis LoRA for Continual Multimodal Instruction Tuning", comparison_class: contextual_only, port_mode: citation_only, primary_table: false, allowed_claims: [continual_shared_basis_context], reason_code: multimodal_llm_task}
  - {citation_key: vsm_diffusion_neurips2023, title: "Few-Shot Diffusion Models with a Visual Concept Memory", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [episodic_memory_context], reason_code: different_memory_object_and_training_semantics}
  - {citation_key: continual_diffusion_clora_2023, title: "Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [continual_learning_context], reason_code: sequential_gradient_training_not_bounded_code_storage}
  - {citation_key: mining_your_own_secrets_2024, title: "Mining Your Own Secrets: Diffusion Classifier Scores for Continual Personalization", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [continual_learning_context], reason_code: different_interference_claim}
  - {citation_key: conceptguard_2025, title: "ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting Control", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [continual_learning_context], reason_code: different_interference_claim}
  - {citation_key: concept_neuron_selection_2025, title: "Continual Personalization for Text-to-Image Models with Concept Neuron Selection", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [continual_learning_context], reason_code: different_parameter_isolation_claim}
  - {citation_key: ada_adapter_consolidation_2025, title: "ADA: Adaptive Adapter Consolidation for Continual Learning", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [consolidation_context], reason_code: fixed_pool_task_adapter_semantics}
  - {citation_key: autolora_2024, title: "AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [adapter_routing_context], reason_code: rank_selection_not_memory_allocation}
  - {citation_key: mod_adapter_2025, title: "Mod-Adapter: Enabling Scalable Multi-Concept Customization", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [composition_context], reason_code: secondary_composition_task}
  - {citation_key: loraverse_2025, title: "LoRAverse: Consistent and Controllable Multi-Concept Image Generation", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [composition_context], reason_code: secondary_composition_task}
  - {citation_key: cached_multi_lora_2025, title: "Cached Multi-LoRA Composition for Multi-Concept Image Generation", comparison_class: incompatible, port_mode: no_primary_port, primary_table: false, allowed_claims: [composition_context], reason_code: composition_cache_not_capacity_allocator}
```

- [ ] **Step 5: Implement strict catalog models and generate the schema**

```python
# src/ratemem/baselines/catalog.py
from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CatalogError(ValueError):
    pass


class ComparisonClass(StrEnum):
    MATCHED_REQUIRED = "matched_required"
    CONTEXTUAL_ONLY = "contextual_only"
    INCOMPATIBLE = "incompatible"


class ControlEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    implementation_mode: Literal["native", "external_jsonl"]
    family: str
    primary_table: bool
    roles: Sequence[str]


class LiteratureEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation_key: str
    title: str
    comparison_class: ComparisonClass
    port_mode: str
    primary_table: bool
    allowed_claims: Sequence[str]
    reason_code: str

    @model_validator(mode="after")
    def reject_unmatched_primary(self) -> "LiteratureEntry":
        if self.primary_table and self.comparison_class is not ComparisonClass.MATCHED_REQUIRED:
            raise ValueError("only matched_required literature can enter a primary table")
        return self


class BaselineCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"]
    controls: Sequence[ControlEntry]
    literature: Sequence[LiteratureEntry]

    @model_validator(mode="after")
    def reject_duplicates(self) -> "BaselineCatalog":
        for values, label in (
            ([item.id for item in self.controls], "control id"),
            ([item.citation_key for item in self.literature], "citation key"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label}")
        return self

    @property
    def control_ids(self) -> Sequence[str]:
        return tuple(item.id for item in self.controls)

    @property
    def primary_control_ids(self) -> Sequence[str]:
        return tuple(item.id for item in self.controls if item.primary_table)

    def require_primary_eligible(self, citation_key: str) -> LiteratureEntry:
        item = next(entry for entry in self.literature if entry.citation_key == citation_key)
        if not item.primary_table or item.comparison_class is not ComparisonClass.MATCHED_REQUIRED:
            raise CatalogError(f"{citation_key} is not eligible for primary matched table")
        return item


REQUIRED_CONTROL_IDS = {
    "independent_fifo", "independent_lru", "independent_lrua",
    "private_progressive_size_aware", "private_progressive_separable_rate",
    "shared_packet_plain_greedy", "cts_style_static", "vb_lora_style_static",
    "share_style_online", "dreamcache_feature_cache", "hyperlora_upstream",
    "stateless_amortizer", "per_concept_lora",
    "exact_append_only_quantized", "exact_future_trace_packets",
}


def load_catalog(path: Path) -> BaselineCatalog:
    return BaselineCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
```

Generate `schemas/ratemem-baseline-catalog-v1.schema.json` with `BaselineCatalog.model_json_schema()` using canonical sorted JSON and validate the YAML with `jsonschema.Draft202012Validator.check_schema` and `jsonschema.validate`.

- [ ] **Step 6: Run catalog/schema checks**

Run:

```bash
uv lock
uv sync --all-extras --frozen
uv run ratemem-baselines schema catalog --output schemas/ratemem-baseline-catalog-v1.schema.json
uv run pytest tests/baselines/test_catalog.py -q
uv run ruff check src/ratemem/baselines/catalog.py tests/baselines/test_catalog.py
uv run mypy src/ratemem/baselines/catalog.py
```

Expected: all tests pass; the schema is valid Draft 2020-12; Ruff and mypy exit 0.

- [ ] **Step 7: Commit the frozen comparator disposition**

```bash
git add pyproject.toml uv.lock configs/baselines/literature-classification.yaml src/ratemem/baselines/__init__.py src/ratemem/baselines/catalog.py schemas/ratemem-baseline-catalog-v1.schema.json tests/baselines/test_catalog.py
git commit -m "science: classify matched and contextual baselines"
```

### Task 2: Define the common lifecycle protocol and host-owned byte ledger

**Files:**
- Create: `src/ratemem/baselines/protocol.py`
- Create: `src/ratemem/baselines/ledger.py`
- Modify: `src/ratemem/evaluation/baselines.py`
- Modify: `src/ratemem/evaluation/replay.py`
- Create: `schemas/ratemem-baseline-event-receipt-v1.schema.json`
- Create: `schemas/ratemem-baseline-ledger-v1.schema.json`
- Test: `tests/baselines/test_protocol.py`
- Test: `tests/baselines/test_ledger.py`

- [ ] **Step 1: Write failing protocol, causality, and ledger tests**

```python
# tests/baselines/test_protocol.py
import pytest

from ratemem.baselines.protocol import CausalEventView, FutureAccessError
from tests.fixtures.baselines.contracts import make_contract
from tests.fixtures.baselines import events, hashes


def test_causal_view_exposes_current_and_past_but_never_future() -> None:
    view = CausalEventView(events.three_event_trace(), current_index=1)
    assert [event.event_index for event in view.history()] == [0, 1]
    with pytest.raises(FutureAccessError, match="causal adapter requested event 2"):
        view.at(2)


def test_contract_binds_all_paired_inputs_and_method_dependencies() -> None:
    contract = make_contract(backbone_id="sana_1_5_1_6b", hashes=hashes.complete())
    assert contract.amortizer_sha256 == hashes.AMORTIZER
    assert contract.adapter_basis_sha256 == hashes.BASIS
    assert contract.candidate_stream_sha256 == hashes.CANDIDATES
    assert contract.prompt_pool_sha256 == hashes.PROMPTS
    assert contract.support_pool_sha256 == hashes.SUPPORTS
    assert contract.noise_seed_manifest_sha256 == hashes.NOISE
```

```python
# tests/baselines/test_ledger.py
from ratemem.baselines.ledger import ONLINE_COMPONENT_NAMES, export_state, ledger_from_export
from tests.fixtures.baselines.states import complete_state


def test_host_computes_exact_total_from_canonical_export() -> None:
    blob = export_state(complete_state())
    ledger = ledger_from_export(blob, shared_trained_bytes=4096, external_support_bytes=0)
    assert ledger.online_state_bytes == len(blob)
    assert sum(ledger.component_bytes.values()) == ledger.online_state_bytes
    assert set(ledger.component_bytes) == set(ONLINE_COMPONENT_NAMES)


def test_roundtrip_restores_future_receipts_without_hidden_mutable_state(adapter_factory) -> None:
    original = adapter_factory()
    original.initialize(adapter_factory.contract)
    original.apply_event(adapter_factory.create_event, adapter_factory.create_view)
    exported = original.export_online_state()
    restored = adapter_factory()
    restored.initialize(adapter_factory.contract)
    restored.import_online_state(exported)
    assert restored.apply_event(
        adapter_factory.read_event, adapter_factory.read_view
    ) == original.apply_event(adapter_factory.read_event, adapter_factory.read_view)
```

- [ ] **Step 2: Run the tests and verify the missing protocol failure**

Run: `uv run pytest tests/baselines/test_protocol.py tests/baselines/test_ledger.py -q`

Expected: collection fails importing `ratemem.baselines.protocol` or `ratemem.baselines.ledger`.

- [ ] **Step 3: Implement the exact shared contract and receipts**

```python
# src/ratemem/baselines/protocol.py
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from ratemem.evaluation.traces import LifecycleEvent, ProbeEvent


Hash256 = str
PrimaryBackboneId = Literal["sana_1_5_1_6b"]


class FutureAccessError(RuntimeError):
    pass


class FrozenComparisonContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["1.0"] = "1.0"
    trace_id: str
    dataset_lock_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_lock_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_requirements_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    backbone_id: PrimaryBackboneId
    backbone_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_layout_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    amortizer_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_basis_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    codec_dictionary_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_stream_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_pool_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    support_pool_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    noise_seed_manifest_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    sampler_id: str
    scheduler_revision: str
    cfg_scale: float
    resolution: tuple[PositiveInt, PositiveInt]
    denoising_steps: PositiveInt
    byte_budget: PositiveInt
    request_regime: Literal["uniform", "zipf"]
    search_budget_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")


class ExactByteLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    serializer_id: Literal["ratemem-baseline-cbor-v1"]
    online_state_bytes: NonNegativeInt
    online_state_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    component_bytes: dict[str, NonNegativeInt]
    shared_trained_bytes: NonNegativeInt
    external_support_bytes: NonNegativeInt

    @model_validator(mode="after")
    def totals_match(self) -> "ExactByteLedger":
        if sum(self.component_bytes.values()) != self.online_state_bytes:
            raise ValueError("component bytes do not equal canonical online state bytes")
        return self


class EventReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method_id: str
    trace_id: str
    event_index: NonNegativeInt
    event_kind: Literal["create", "update", "read", "delete"]
    input_commitment_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    method_state_sha256_before: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    method_state_sha256_after: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_stream_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: Literal["created", "updated", "read", "deleted", "rejected", "evicted", "stale_handle"]
    affected_handles: Sequence[str]
    evicted_handles: Sequence[str]
    decoded_code_sha256: Hash256 | None
    generated_sample_sha256: Hash256 | None
    ledger: ExactByteLedger


class MethodSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method_id: str
    trace_id: str
    event_index: NonNegativeInt
    state_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    online_state_bytes: NonNegativeInt
    opaque_snapshot_token: str


class ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method_id: str
    trace_id: str
    probe_event_index: NonNegativeInt
    snapshot_state_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    input_commitment_sha256: Hash256 = Field(pattern=r"^[0-9a-f]{64}$")
    generated_sample_sha256: Hash256
    update_usage: Literal[False] = False


class CausalEventView:
    def __init__(self, events: Sequence[LifecycleEvent], current_index: int) -> None:
        self._events = tuple(events)
        self._current = current_index

    def __len__(self) -> int:
        return self._current + 1

    def __getitem__(self, index: int) -> LifecycleEvent:
        if index < 0:
            index += len(self)
        return self.at(index)

    def __iter__(self) -> Iterator[LifecycleEvent]:
        return iter(self._events[: self._current + 1])

    def at(self, index: int) -> LifecycleEvent:
        if index > self._current:
            raise FutureAccessError(f"causal adapter requested event {index}")
        return self._events[index]

    def history(self) -> Sequence[LifecycleEvent]:
        return self._events[: self._current + 1]


@runtime_checkable
class BaselineAdapter(Protocol):
    method_id: str
    role: Literal["causal", "upper_reference", "latency_control"]

    def initialize(self, contract: FrozenComparisonContract) -> None:
        raise NotImplementedError

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        raise NotImplementedError

    def copy_snapshot(self) -> MethodSnapshot:
        raise NotImplementedError

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        raise NotImplementedError

    def export_online_state(self) -> bytes:
        raise NotImplementedError

    def import_online_state(self, payload: bytes) -> None:
        raise NotImplementedError

    def state_ledger(self) -> ExactByteLedger:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
```

The future-trace oracle implements a separate `UpperReferenceAdapter` constructor that receives the full immutable event tuple. No object satisfying the causal factory can receive or retain that tuple. Re-export these exact types from `ratemem.evaluation.baselines`; change scientific replay to import this `FrozenComparisonContract` rather than defining a second class.

- [ ] **Step 4: Implement canonical state export and host-recomputed accounting**

`export_state` writes one canonical CBOR top-level map. Every component value is a length-framed list of records; tensor records contain dtype, shape, byte order, and contiguous payload. Alignment is explicit zero bytes in the `alignment` component. It never uses pickle or trusts a method-supplied byte count.

```python
# src/ratemem/baselines/ledger.py
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cbor2
import numpy as np
from numpy.typing import NDArray

from ratemem.baselines.protocol import ExactByteLedger


ONLINE_COMPONENT_NAMES = (
    "base_codes", "packet_payloads", "packet_hashes", "incidences_gains",
    "feature_cache", "optional_tokens", "handles", "usage_age",
    "reference_counts", "controller_state", "allocator_state", "checksums", "alignment",
)


@dataclass(frozen=True, slots=True)
class TensorRecord:
    name: str
    array: NDArray[np.generic]


def tensor_record(record: TensorRecord) -> dict[str, object]:
    array = np.ascontiguousarray(record.array)
    dtype = array.dtype.newbyteorder("<")
    little = array.astype(dtype, copy=False)
    return {"name": record.name, "dtype": dtype.str, "shape": list(array.shape), "data": little.tobytes()}


def component_blob(name: str, records: Sequence[object]) -> bytes:
    return cbor2.dumps({"component": name, "records": list(records)}, canonical=True)


def export_state(components: Mapping[str, Sequence[object]]) -> bytes:
    if set(components) != set(ONLINE_COMPONENT_NAMES):
        missing = sorted(set(ONLINE_COMPONENT_NAMES) - set(components))
        extra = sorted(set(components) - set(ONLINE_COMPONENT_NAMES))
        raise ValueError(f"state components mismatch missing={missing} extra={extra}")
    framed = {name: component_blob(name, components[name]) for name in ONLINE_COMPONENT_NAMES}
    return cbor2.dumps({"format": "ratemem-baseline-cbor-v1", "components": framed}, canonical=True)


def ledger_from_export(payload: bytes, shared_trained_bytes: int, external_support_bytes: int) -> ExactByteLedger:
    decoded = cbor2.loads(payload)
    framed: dict[str, bytes] = decoded["components"]
    envelope = len(payload) - sum(len(blob) for blob in framed.values())
    component_bytes = {name: len(framed[name]) for name in ONLINE_COMPONENT_NAMES}
    component_bytes["checksums"] += envelope
    return ExactByteLedger(
        serializer_id="ratemem-baseline-cbor-v1",
        online_state_bytes=len(payload),
        online_state_sha256=hashlib.sha256(payload).hexdigest(),
        component_bytes=component_bytes,
        shared_trained_bytes=shared_trained_bytes,
        external_support_bytes=external_support_bytes,
    )
```

Every adapter's `state_ledger()` must call `ledger_from_export(self.export_online_state(), shared_trained_bytes=self.shared_trained_bytes, external_support_bytes=self.external_support_bytes)`. After every mutable event, the runner asserts `online_state_bytes <= contract.byte_budget`. A baseline that cannot export and restore all state is ineligible; no manually estimated tensor size enters a matched table.

- [ ] **Step 5: Generate schemas and verify roundtrip/byte invariants**

Run:

```bash
uv run ratemem-baselines schema receipt --output schemas/ratemem-baseline-event-receipt-v1.schema.json
uv run ratemem-baselines schema ledger --output schemas/ratemem-baseline-ledger-v1.schema.json
uv run pytest tests/baselines/test_protocol.py tests/baselines/test_ledger.py -q
uv run ruff check src/ratemem/baselines/protocol.py src/ratemem/baselines/ledger.py tests/baselines/test_protocol.py tests/baselines/test_ledger.py
uv run mypy src/ratemem/baselines/protocol.py src/ratemem/baselines/ledger.py
```

Expected: all tests pass; mutating one tensor byte changes `online_state_sha256`; omitting any component raises `ValueError`; scientific replay imports the same contract class.

- [ ] **Step 6: Commit the protocol and exact ledger**

```bash
git add src/ratemem/baselines/protocol.py src/ratemem/baselines/ledger.py src/ratemem/evaluation/baselines.py src/ratemem/evaluation/replay.py schemas/ratemem-baseline-event-receipt-v1.schema.json schemas/ratemem-baseline-ledger-v1.schema.json tests/baselines/test_protocol.py tests/baselines/test_ledger.py
git commit -m "feat(baselines): define causal protocol and exact ledger"
```

### Task 3: Freeze a provider-neutral SANA shared-input schema and separate contextual backbone evidence

**Files:**
- Create: `configs/baselines/backbones.yaml`
- Create: `src/ratemem/baselines/shared_inputs.py`
- Create: `src/ratemem/baselines/backbones.py`
- Create: `schemas/ratemem-shared-input-bundle-v1.schema.json`
- Test: `tests/baselines/test_shared_inputs.py`
- Test: `tests/contract/baselines/test_backbone_binding.py`

- [ ] **Step 1: Write failing shared-stream and backbone tests**

```python
# tests/baselines/test_shared_inputs.py
from pathlib import Path

import pytest

from ratemem.baselines.shared_inputs import (
    CandidateAccessError,
    SharedInputReader,
    materialize_fixture_bundle,
)


def test_code_based_controls_read_identical_amortizer_basis_and_candidates(tmp_path: Path) -> None:
    bundle = materialize_fixture_bundle(tmp_path)
    assert bundle.manifest.backbone_id == "sana_1_5_1_6b"
    assert bundle.manifest.projection_count == 120
    assert bundle.manifest.code_dim == 480
    readers = [SharedInputReader(bundle, method_id=method_id) for method_id in (
        "independent_lru", "private_progressive_separable_rate",
        "shared_packet_plain_greedy", "cts_style_static", "vb_lora_style_static",
        "share_style_online",
    )]
    record_hashes = {reader.for_event(0, current_index=0).record_sha256 for reader in readers}
    assert record_hashes == {bundle.manifest.event_records[0].record_sha256}
    assert {reader.manifest.amortizer_sha256 for reader in readers} == {bundle.manifest.amortizer_sha256}
    assert {reader.manifest.adapter_basis_sha256 for reader in readers} == {bundle.manifest.adapter_basis_sha256}
    assert {reader.manifest.candidate_stream_sha256 for reader in readers} == {bundle.manifest.candidate_stream_sha256}


def test_reader_cannot_open_future_candidate_record(tmp_path: Path) -> None:
    bundle = materialize_fixture_bundle(tmp_path)
    reader = SharedInputReader(bundle, method_id="independent_fifo")
    with pytest.raises(CandidateAccessError, match="future shared input event 1"):
        reader.for_event(1, current_index=0)


def test_tampered_tensor_or_candidate_json_invalidates_bundle(tmp_path: Path) -> None:
    bundle = materialize_fixture_bundle(tmp_path)
    tensor_path = bundle.root / bundle.manifest.event_records[0].tensor_path
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="shared-input file hash mismatch"):
        SharedInputReader(bundle.root, method_id="independent_lru")
```

```python
# tests/contract/baselines/test_backbone_binding.py
from ratemem.baselines.backbones import load_backbone_policy


def test_sana_is_the_only_primary_and_sdxl_cannot_be_promoted() -> None:
    policy = load_backbone_policy("configs/baselines/backbones.yaml")
    sana = policy.backbones["sana_1_5_1_6b"]
    sdxl = policy.backbones["sdxl_1_0"]
    assert policy.primary_backbone == "sana_1_5_1_6b"
    assert policy.primary_backbone_is_fixed is True
    assert policy.contextual_backbones == ("sdxl_1_0",)
    assert policy.contextual_backbones_may_satisfy_primary_requirements is False
    assert sana.comparison_role == "primary"
    assert sana.revision == "b77948f2b4eed5c728e9b828ccff07f7427b43cc"
    assert sana.expected_projection_count == 120
    assert sana.code_dim == 480
    assert sdxl.comparison_role == "contextual_only"
    assert sdxl.primary_eligible is False
    assert sdxl.ratemem_extension_available is False
    assert sdxl.revision == "462165984030d82259a11f4367a4eed129e94a7b"


def test_feature_native_and_optimization_controls_are_explicit_exceptions(method_matrix) -> None:
    assert method_matrix["dreamcache_feature_cache"].shared_input_scope == "feature_native"
    assert method_matrix["per_concept_lora"].shared_input_scope == "optimization_native"
    assert method_matrix["stateless_amortizer"].shared_input_scope == "same_amortizer_recompute"
    assert all(
        row.shared_input_scope == "same_code_and_candidates"
        for row in method_matrix.values()
        if row.method_id in method_matrix.code_based_method_ids
    )
```

- [ ] **Step 2: Run the tests and verify the shared-input module is missing**

Run: `uv run pytest tests/baselines/test_shared_inputs.py tests/contract/baselines/test_backbone_binding.py -q`

Expected: collection fails importing `ratemem.baselines.shared_inputs`.

- [ ] **Step 3: Lock exact backbone identities and matching rules**

```yaml
# configs/baselines/backbones.yaml
schema_version: "1.0"
primary_backbone: sana_1_5_1_6b
primary_backbone_is_fixed: true
contextual_backbones: [sdxl_1_0]
contextual_backbones_may_satisfy_primary_requirements: false
future_contextual_promotion_requires_separately_reviewed_ratemem_extension: true
published_different_backbone_numbers_are_contextual: true
backbones:
  sana_1_5_1_6b:
    model_id: Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
    revision: b77948f2b4eed5c728e9b828ccff07f7427b43cc
    architecture: sana_transformer
    resolution: [1024, 1024]
    target_suffixes: [to_q, to_k, to_v]
    layout_lock_path: configs/scientific/layouts/sana_1_5_1_6b.json
    comparison_role: primary
    primary_eligible: true
    ratemem_extension_available: true
    expected_projection_count: 120
    code_dim: 480
  sdxl_1_0:
    model_id: stabilityai/stable-diffusion-xl-base-1.0
    revision: 462165984030d82259a11f4367a4eed129e94a7b
    architecture: sdxl_unet
    resolution: [1024, 1024]
    target_suffixes: [to_q, to_k, to_v]
    comparison_role: contextual_only
    primary_eligible: false
    ratemem_extension_available: false
shared_input_scopes:
  same_code_and_candidates:
    - independent_fifo
    - independent_lru
    - independent_lrua
    - private_progressive_size_aware
    - private_progressive_separable_rate
    - shared_packet_plain_greedy
    - cts_style_static
    - vb_lora_style_static
    - share_style_online
  same_amortizer_recompute: [stateless_amortizer]
  feature_native: [dreamcache_feature_cache]
  optimization_native: [per_concept_lora]
  upstream_native: [hyperlora_upstream]
  upper_reference: [exact_append_only_quantized, exact_future_trace_packets]
```

Before scientific training, `ratemem-baselines backbones lock-layout --backbone sana_1_5_1_6b` opens the pinned SANA configuration, enumerates its sorted q/k/v projection paths, records input/output widths and attention kind, and commits the generated layout. The command must observe the 120 projections already contract-tested by the pilot plan and the locked 480-dimensional RateMem code; either mismatch blocks the route. The SDXL commit is retained only as immutable provenance for contextual upstream reproduction. This plan does not create an SDXL RateMem layout, amortizer, basis, shared-input bundle, matched replay, or primary row. Any future SDXL RateMem extension needs a new design review, versioned contract, learned-method port, and scientific lock before SDXL evidence can be reconsidered for a matched table.

- [ ] **Step 4: Implement content-addressed shared input bundles**

```python
# src/ratemem/baselines/shared_inputs.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt
from safetensors.numpy import load_file, save_file

from ratemem.baselines.protocol import LifecycleEvent
from ratemem.evaluation.canonical import canonical_json_bytes, file_sha256


Sha256 = str
GitCommit = str
Float32 = NDArray[np.float32]
SignedInt16 = Annotated[int, Field(ge=-32768, le=32767)]


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SharedProviderMetadata:
    provider_id: str
    provider_revision_sha256: Sha256
    backbone_id: Literal["sana_1_5_1_6b"]
    backbone_revision: GitCommit
    adapter_layout_sha256: Sha256
    projection_count: Literal[120]
    code_dim: Literal[480]
    amortizer_sha256: Sha256
    adapter_basis_sha256: Sha256
    codec_dictionary_sha256: Sha256
    support_pool_sha256: Sha256
    incidence_gain_step: float

    def __post_init__(self) -> None:
        if (
            not self.provider_id
            or len(self.backbone_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.backbone_revision)
        ):
            raise ValueError("provider id and pinned backbone revision are required")
        for field in (
            "provider_revision_sha256", "adapter_layout_sha256", "amortizer_sha256",
            "adapter_basis_sha256", "codec_dictionary_sha256", "support_pool_sha256",
        ):
            _require_sha256(str(getattr(self, field)), field)
        if self.projection_count != 120 or self.code_dim != 480:
            raise ValueError("shared provider must expose the frozen SANA 120/480 layout")
        if not np.isfinite(self.incidence_gain_step) or self.incidence_gain_step <= 0.0:
            raise ValueError("incidence gain step must be finite and positive")


@dataclass(frozen=True, slots=True, order=True)
class ProviderPacketKey:
    dictionary_revision_sha256: Sha256
    group: int
    stage: int
    entry: int

    def __post_init__(self) -> None:
        _require_sha256(self.dictionary_revision_sha256, "dictionary_revision_sha256")
        if min(self.group, self.stage, self.entry) < 0:
            raise ValueError("packet address indices must be nonnegative")


@dataclass(frozen=True, slots=True)
class ProviderPacketCandidate:
    key: ProviderPacketKey
    packet_id: Sha256
    packet_payload: bytes
    gain_q: int

    def __post_init__(self) -> None:
        _require_sha256(self.packet_id, "packet_id")
        if hashlib.sha256(self.packet_payload).hexdigest() != self.packet_id:
            raise ValueError("provider packet content address does not match payload")
        if not -32768 <= self.gain_q <= 32767:
            raise ValueError("provider gain must fit signed int16")


@dataclass(frozen=True, slots=True)
class ProviderEventOutput:
    event_index: int
    handle: str
    target_code: Float32
    base_code: Float32
    quantizer_scales: Float32
    candidates: tuple[ProviderPacketCandidate, ...]

    def __post_init__(self) -> None:
        if self.event_index < 0 or not self.handle:
            raise ValueError("provider event identity is invalid")
        for name, value, shape in (
            ("target_code", self.target_code, (480,)),
            ("base_code", self.base_code, (480,)),
            ("quantizer_scales", self.quantizer_scales, (30,)),
        ):
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite float32 with shape {shape}")
        if len({candidate.key for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("provider event repeats a packet address")


@runtime_checkable
class SharedInputProvider(Protocol):
    def manifest_metadata(self) -> SharedProviderMetadata:
        raise NotImplementedError

    def record_for_event(self, event: LifecycleEvent) -> ProviderEventOutput:
        raise NotImplementedError


class CandidateAccessError(RuntimeError):
    pass


class CandidatePacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    packet_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    dictionary_revision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group: NonNegativeInt
    stage: NonNegativeInt
    entry: NonNegativeInt
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_bytes: NonNegativeInt
    incidence_bytes: NonNegativeInt
    dependent_handles: Sequence[str]
    gain_q_by_handle: dict[str, SignedInt16]


class SharedEventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_index: NonNegativeInt
    handle: str
    tensor_path: str
    tensor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_packets: Sequence[CandidatePacket]
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SharedInputManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"]
    split: Literal["train", "validation", "final"]
    trace_id: str
    backbone_id: Literal["sana_1_5_1_6b"]
    backbone_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_layout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_count: Literal[120]
    code_dim: Literal[480]
    amortizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_basis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    codec_dictionary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_pool_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_records: Sequence[SharedEventRecord]
    candidate_stream_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SharedInputBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    root: Path
    manifest: SharedInputManifest


class SharedInputReader:
    def __init__(self, root: Path | SharedInputBundle, method_id: str) -> None:
        self.root = root.root if isinstance(root, SharedInputBundle) else root
        manifest_path = self.root / "manifest.json"
        self.manifest = SharedInputManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        self.method_id = method_id
        for record in self.manifest.event_records:
            if file_sha256(self.root / record.tensor_path) != record.tensor_sha256:
                raise ValueError("shared-input file hash mismatch")

    def for_event(self, event_index: int, current_index: int) -> SharedEventRecord:
        if event_index > current_index:
            raise CandidateAccessError(f"future shared input event {event_index}")
        return next(record for record in self.manifest.event_records if record.event_index == event_index)
```

These are the sole canonical `SharedInputProvider`, `SharedProviderMetadata`, `ProviderPacketKey`,
`ProviderPacketCandidate`, and `ProviderEventOutput` definitions. The learned plan imports them; it
must not redefine, wrap, or subclass their data records. This baseline-owned module does not train
or load an amortizer, basis, or codec. `write_shared_input_bundle` accepts provider outputs, requires
every packet key's dictionary revision to equal the metadata codec dictionary hash, verifies each
payload content address and signed-int16 gain, saves `target_code`, `base_code`, quantizer scales,
and packet payloads in safetensors, and aggregates identical packet IDs seen so far into canonical
dependent-handle and `gain_q_by_handle` maps. It computes every record hash and then hashes the
ordered record hashes into `candidate_stream_sha256`. Pre-lock tests use only
`materialize_fixture_bundle` and bind its synthetic-provider hash into the audit receipt. After both
locks and scientific authorization, the learned workflow supplies the real provider exactly once.
Final-test materialization accepts a stream iterator, not a path, so model-selection code cannot
open the encrypted final trace.

- [ ] **Step 5: Implement the SANA runner and contextual-evidence gate**

```python
# src/ratemem/baselines/backbones.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor


class PrimaryBackboneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    architecture: Literal["sana_transformer"]
    resolution: tuple[int, int]
    target_suffixes: Sequence[Literal["to_q", "to_k", "to_v"]]
    layout_lock_path: Path
    comparison_role: Literal["primary"]
    primary_eligible: Literal[True]
    ratemem_extension_available: Literal[True]
    expected_projection_count: Literal[120]
    code_dim: Literal[480]


class ContextualBackboneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    architecture: Literal["sdxl_unet"]
    resolution: tuple[int, int]
    target_suffixes: Sequence[Literal["to_q", "to_k", "to_v"]]
    comparison_role: Literal["contextual_only"]
    primary_eligible: Literal[False]
    ratemem_extension_available: Literal[False]


BackboneSpec = Annotated[PrimaryBackboneSpec | ContextualBackboneSpec, Field(discriminator="comparison_role")]


class BackbonePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"]
    primary_backbone: Literal["sana_1_5_1_6b"]
    primary_backbone_is_fixed: Literal[True]
    contextual_backbones: tuple[Literal["sdxl_1_0"], ...]
    contextual_backbones_may_satisfy_primary_requirements: Literal[False]
    future_contextual_promotion_requires_separately_reviewed_ratemem_extension: Literal[True]
    published_different_backbone_numbers_are_contextual: Literal[True]
    backbones: dict[str, BackboneSpec]
    shared_input_scopes: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def validate_fixed_route(self) -> "BackbonePolicy":
        if set(self.backbones) != {"sana_1_5_1_6b", "sdxl_1_0"}:
            raise ValueError("unexpected backbone set")
        if self.contextual_backbones != ("sdxl_1_0",):
            raise ValueError("SDXL must be the sole contextual backbone")
        if self.backbones[self.primary_backbone].comparison_role != "primary":
            raise ValueError("SANA must be the primary backbone")
        if any(self.backbones[item].primary_eligible for item in self.contextual_backbones):
            raise ValueError("contextual backbone cannot be primary eligible")
        return self


def load_backbone_policy(path: str | Path) -> BackbonePolicy:
    return BackbonePolicy.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class BackboneRunner(Protocol):
    backbone_id: Literal["sana_1_5_1_6b"]
    spec: PrimaryBackboneSpec

    def install_code(self, code: Tensor) -> None:
        raise NotImplementedError

    def clear_code(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, seed: int, *, sampler_id: str, cfg_scale: float, steps: int) -> Tensor:
        raise NotImplementedError

    def one_step_latent(self, prompt: str, seed: int, timestep: int) -> Tensor:
        raise NotImplementedError
```

`SanaBackboneRunner` delegates code installation to `SanaDynamicAdapterBank` from the SANA plan. Structural pre-lock tests inject synthetic SANA coefficients and do not require learned amortizer/basis weights. During post-lock scientific work, every code-based matched method receives the same SANA runner spec, 120-projection layout hash, 480-dimensional-code contract, amortizer hash, basis hash, prompt, seed, sampler, CFG, resolution, and step count. There is deliberately no `SdxlBackboneRunner` implementing the RateMem code contract in this route. An SDXL-native upstream worker can emit a contextual artifact through `ExternalJsonlAdapter`, but the registry rejects that artifact from `FrozenComparisonContract`, primary fidelity satisfaction, selection, and paired replay.

- [ ] **Step 6: Generate the shared-input schema and run all CPU contracts**

Run:

```bash
uv run ratemem-baselines schema shared-input --output schemas/ratemem-shared-input-bundle-v1.schema.json
uv run pytest tests/baselines/test_shared_inputs.py tests/contract/baselines/test_backbone_binding.py -q
uv run ruff check src/ratemem/baselines/shared_inputs.py src/ratemem/baselines/backbones.py tests/baselines/test_shared_inputs.py tests/contract/baselines/test_backbone_binding.py
uv run mypy src/ratemem/baselines/shared_inputs.py src/ratemem/baselines/backbones.py
```

Expected: all tests pass; a future record or tampered tensor is rejected before an adapter sees it.

- [ ] **Step 7: Commit the matched input/backbone contract**

```bash
git add configs/baselines/backbones.yaml src/ratemem/baselines/shared_inputs.py src/ratemem/baselines/backbones.py schemas/ratemem-shared-input-bundle-v1.schema.json tests/baselines/test_shared_inputs.py tests/contract/baselines/test_backbone_binding.py
git commit -m "feat(baselines): bind shared inputs and backbones"
```

### Task 4: Implement independent uncompressed-code FIFO, LRU, and LRUA

**Files:**
- Create: `src/ratemem/baselines/independent.py`
- Test: `tests/baselines/test_independent.py`
- Test: `tests/contract/baselines/test_independent_budget.py`

- [ ] **Step 1: Write failing deterministic policy and accounting tests**

```python
# tests/baselines/test_independent.py
import pytest

from ratemem.baselines.independent import IndependentCodeCacheAdapter
from tests.fixtures.baselines.runtime import run_events


@pytest.mark.parametrize(
    ("policy", "expected_victim"),
    [("fifo", "h0"), ("lru", "h1"), ("lrua", "h1")],
)
def test_locked_policy_selects_expected_victim(policy: str, expected_victim: str, cache_fixture) -> None:
    adapter = IndependentCodeCacheAdapter(method_id=f"independent_{policy}", policy=policy, lrua_decay=0.99)
    receipts = run_events(adapter, cache_fixture.contract, cache_fixture.create_read_overflow_trace)
    overflow = receipts[-1]
    assert overflow.outcome == "created"
    assert overflow.evicted_handles == (expected_victim,)
    assert overflow.ledger.online_state_bytes <= cache_fixture.contract.byte_budget


def test_probe_never_refreshes_lru_or_lrua(cache_fixture) -> None:
    adapter = IndependentCodeCacheAdapter(method_id="independent_lru", policy="lru", lrua_decay=0.99)
    run_events(adapter, cache_fixture.contract, cache_fixture.prefix)
    before = adapter.export_online_state()
    snapshot = adapter.copy_snapshot()
    adapter.score_probe(snapshot, cache_fixture.probe)
    assert adapter.export_online_state() == before


def test_uncompressed_code_is_bf16_and_not_reencoded(cache_fixture) -> None:
    adapter = IndependentCodeCacheAdapter(method_id="independent_fifo", policy="fifo", lrua_decay=0.99)
    run_events(adapter, cache_fixture.contract, cache_fixture.single_create)
    state = adapter.inspect_state()
    assert state.codes["h0"].dtype == "bfloat16"
    assert state.codes["h0"].raw_sha256 == cache_fixture.target_code_bf16_sha256
```

```python
# tests/contract/baselines/test_independent_budget.py
from hypothesis import given, strategies as st

from ratemem.baselines.independent import IndependentCodeCacheAdapter
from tests.fixtures.baselines.runtime import random_trace, run_events


@given(seed=st.integers(min_value=0, max_value=2**31 - 1), budget=st.integers(min_value=256, max_value=4096))
def test_every_randomized_receipt_is_within_exact_budget(seed: int, budget: int, cache_fixture) -> None:
    contract = cache_fixture.contract.model_copy(update={"byte_budget": budget})
    adapter = IndependentCodeCacheAdapter("independent_lrua", "lrua", lrua_decay=0.99)
    receipts = run_events(adapter, contract, random_trace(seed, events=40))
    assert all(receipt.ledger.online_state_bytes <= budget for receipt in receipts)
```

- [ ] **Step 2: Run tests and verify the independent adapter is missing**

Run: `uv run pytest tests/baselines/test_independent.py tests/contract/baselines/test_independent_budget.py -q`

Expected: collection fails importing `ratemem.baselines.independent`.

- [ ] **Step 3: Implement one state engine with three deterministic victim scores**

```python
# src/ratemem/baselines/independent.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


Policy = Literal["fifo", "lru", "lrua"]


@dataclass(frozen=True, slots=True)
class CodeRecord:
    handle: str
    bf16_payload: bytes
    created_event: int
    last_read_event: int
    decayed_usage: float
    update_count: int


def age_usage(records: dict[str, CodeRecord], decay: float) -> dict[str, CodeRecord]:
    return {handle: replace(record, decayed_usage=record.decayed_usage * decay) for handle, record in records.items()}


def victim_key(record: CodeRecord, policy: Policy) -> tuple[float, int, int, str]:
    if policy == "fifo":
        return (float(record.created_event), record.created_event, record.last_read_event, record.handle)
    if policy == "lru":
        return (float(record.last_read_event), record.created_event, record.last_read_event, record.handle)
    return (record.decayed_usage, record.last_read_event, record.created_event, record.handle)
```

`IndependentCodeCacheAdapter` receives the event's target BF16 code from `SharedInputReader`. At every operational event it multiplies all LRUA usage values by the locked decay; a successful operational `READ` adds one only to the addressed record. `PROBE` operates on a copied snapshot and does neither. `UPDATE` atomically replaces the target code while preserving creation/read metadata. `DELETE` reclaims the complete record. If a create/update exceeds the cap, evict the minimum `victim_key` record excluding the currently updated handle until feasible; reject when even the incoming record alone cannot fit. Serialize code bytes, handles, usage/age, checksums, and controller parameters through Task 2's ledger.

- [ ] **Step 4: Run focused tests, static checks, and the hidden-state roundtrip**

Run:

```bash
uv run pytest tests/baselines/test_independent.py tests/contract/baselines/test_independent_budget.py tests/baselines/test_ledger.py -q
uv run ruff check src/ratemem/baselines/independent.py tests/baselines/test_independent.py tests/contract/baselines/test_independent_budget.py
uv run mypy src/ratemem/baselines/independent.py
```

Expected: all tests pass; every randomized event stays within the host-measured budget and export/import preserves the next receipt exactly.

- [ ] **Step 5: Commit the independent controls**

```bash
git add src/ratemem/baselines/independent.py tests/baselines/test_independent.py tests/contract/baselines/test_independent_budget.py
git commit -m "feat(baselines): add independent FIFO LRU and LRUA"
```

### Task 5: Implement private progressive coding with causal size-aware and exact separable-rate policies

**Files:**
- Create: `src/ratemem/baselines/private_progressive.py`
- Test: `tests/baselines/test_private_progressive.py`
- Test: `tests/contract/baselines/test_separable_rate_oracle.py`

- [ ] **Step 1: Write failing privacy, prefix, causality, and optimality tests**

```python
# tests/baselines/test_private_progressive.py
from ratemem.baselines.private_progressive import PrivateProgressiveAdapter
from tests.fixtures.baselines.runtime import run_events


def test_equal_payloads_are_stored_twice_for_distinct_concepts(progressive_fixture) -> None:
    adapter = PrivateProgressiveAdapter("private_progressive_size_aware", policy="size_aware")
    run_events(adapter, progressive_fixture.contract, progressive_fixture.two_equal_packet_creates)
    state = adapter.inspect_state()
    assert state.private_packets[("h0", 0)].payload == state.private_packets[("h1", 0)].payload
    assert state.private_packets[("h0", 0)].serialized_key != state.private_packets[("h1", 0)].serialized_key
    assert state.ledger.component_bytes["packet_payloads"] >= 2 * len(state.private_packets[("h0", 0)].payload)


def test_degradation_always_removes_a_suffix(progressive_fixture) -> None:
    adapter = PrivateProgressiveAdapter("private_progressive_size_aware", policy="size_aware")
    run_events(adapter, progressive_fixture.contract, progressive_fixture.overflow_trace)
    for prefixes in adapter.inspect_state().prefixes.values():
        assert prefixes == tuple(range(len(prefixes)))


def test_size_aware_uses_only_causal_request_weights(progressive_fixture) -> None:
    first = progressive_fixture.trace_with_future_reads(handle="h0")
    second = progressive_fixture.trace_with_future_reads(handle="h1")
    adapter_a = PrivateProgressiveAdapter("private_progressive_size_aware", policy="size_aware")
    adapter_b = PrivateProgressiveAdapter("private_progressive_size_aware", policy="size_aware")
    receipts_a = run_events(adapter_a, progressive_fixture.contract, first, stop_after=progressive_fixture.overflow_index)
    receipts_b = run_events(adapter_b, progressive_fixture.contract, second, stop_after=progressive_fixture.overflow_index)
    assert receipts_a == receipts_b
```

```python
# tests/contract/baselines/test_separable_rate_oracle.py
from hypothesis import given, strategies as st

from ratemem.baselines.private_progressive import RateChoice, exact_separable_allocation
from tests.fixtures.baselines.oracles import brute_force_rate_choices, rate_choice_instances


@given(instance=st.data())
def test_sparse_dynamic_program_matches_bruteforce(instance) -> None:
    options, budget = instance.draw(rate_choice_instances(max_concepts=6, max_prefixes=5, max_bytes=40))
    result = exact_separable_allocation(options, budget)
    expected = brute_force_rate_choices(options, budget)
    assert result.total_value == expected.total_value
    assert result.total_bytes <= budget
    assert result.prefix_by_handle == expected.prefix_by_handle
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run: `uv run pytest tests/baselines/test_private_progressive.py tests/contract/baselines/test_separable_rate_oracle.py -q`

Expected: collection fails importing `ratemem.baselines.private_progressive`.

- [ ] **Step 3: Implement exact separable prefix allocation**

```python
# src/ratemem/baselines/private_progressive.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateChoice:
    handle: str
    prefix_length: int
    serialized_bytes: int
    value: Decimal


@dataclass(frozen=True, slots=True)
class RateAllocation:
    prefix_by_handle: dict[str, int]
    total_bytes: int
    total_value: Decimal


def exact_separable_allocation(options: dict[str, Sequence[RateChoice]], budget: int) -> RateAllocation:
    frontier: dict[int, tuple[Decimal, Sequence[tuple[str, int]]]] = {0: (Decimal(0), ())}
    for handle in sorted(options):
        expanded: dict[int, tuple[Decimal, Sequence[tuple[str, int]]]] = {}
        for used, (value, choices) in frontier.items():
            for choice in options[handle]:
                candidate_bytes = used + choice.serialized_bytes
                if candidate_bytes > budget:
                    continue
                candidate = (value + choice.value, choices + ((handle, choice.prefix_length),))
                incumbent = expanded.get(candidate_bytes)
                if incumbent is None or candidate[0] > incumbent[0] or (
                    candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
                ):
                    expanded[candidate_bytes] = candidate
        best_value = Decimal("-Infinity")
        frontier = {}
        for used in sorted(expanded):
            value, choices = expanded[used]
            if value > best_value:
                frontier[used] = (value, choices)
                best_value = value
    used, (value, choices) = max(frontier.items(), key=lambda row: (row[1][0], -row[0], row[1][1]))
    return RateAllocation(dict(choices), used, value)
```

Each handle's options include rate zero (whole-concept eviction), base-only, and every legal private packet prefix. Bytes come from serializing that complete option with its private handle-qualified packet keys. Values use only nonnegative request weights derived from operational history through the current event and the locked distortion predictor. No option may contain a non-prefix packet set.

- [ ] **Step 4: Implement the causal size-aware adapter**

`PrivateProgressiveAdapter` starts from all requested prefixes and repeatedly chooses one legal action: remove the last private packet for one handle, or remove its base record at rate zero. For action `a`, compute

```python
loss_per_reclaimed_byte = request_weight[handle] * predicted_quality_loss(a) / exact_reclaimed_bytes(a)
```

Choose the smallest value, then tie-break by handle and resulting prefix length. Recompute the canonical state after each action because handle/length framing can change exact bytes. The separable-rate variant calls `exact_separable_allocation` once per mutation. Both variants use the same base/packet bytes from `SharedInputReader`; they differ only in policy. Reject the incoming record if no feasible state retains its base. Explicit `DELETE` removes its private packets; no packet survives or benefits another concept.

- [ ] **Step 5: Run tests, lint, types, and commit**

Run:

```bash
uv run pytest tests/baselines/test_private_progressive.py tests/contract/baselines/test_separable_rate_oracle.py tests/baselines/test_ledger.py -q
uv run ruff check src/ratemem/baselines/private_progressive.py tests/baselines/test_private_progressive.py tests/contract/baselines/test_separable_rate_oracle.py
uv run mypy src/ratemem/baselines/private_progressive.py
```

Expected: all tests pass; Hypothesis instances match brute force; no receipt exceeds its exact cap.

```bash
git add src/ratemem/baselines/private_progressive.py tests/baselines/test_private_progressive.py tests/contract/baselines/test_separable_rate_oracle.py
git commit -m "feat(baselines): add private progressive rate controls"
```

### Task 6: Implement plain shared-packet marginal-density greedy on the identical candidate stream

**Files:**
- Create: `src/ratemem/baselines/shared_greedy.py`
- Test: `tests/baselines/test_shared_greedy.py`

- [ ] **Step 1: Write failing same-stream and algorithm-isolation tests**

```python
# tests/baselines/test_shared_greedy.py
from ratemem.baselines.shared_greedy import plain_density_greedy
from tests.fixtures.baselines.objectives import GREEDY_TRAP, shared_oracle


def test_plain_greedy_uses_exact_marginal_density_without_seed_enumeration() -> None:
    result = plain_density_greedy(shared_oracle(GREEDY_TRAP), budget_bytes=GREEDY_TRAP.budget)
    assert result.selected_packet_ids == GREEDY_TRAP.plain_greedy_selection
    assert result.selected_packet_ids != GREEDY_TRAP.enumerated_allocator_selection


def test_plain_greedy_and_ratemem_fixture_bind_same_candidate_hash(shared_adapter_fixture) -> None:
    greedy = shared_adapter_fixture.make_plain_greedy()
    provider_manifest = shared_adapter_fixture.synthetic_provider_manifest
    assert greedy.contract.candidate_stream_sha256 == provider_manifest.candidate_stream_sha256
    assert greedy.contract.amortizer_sha256 == provider_manifest.amortizer_sha256
    assert greedy.contract.adapter_basis_sha256 == provider_manifest.adapter_basis_sha256


def test_one_payload_cost_benefits_every_declared_dependent(shared_adapter_fixture) -> None:
    adapter = shared_adapter_fixture.make_plain_greedy()
    shared_adapter_fixture.run_create_pair(adapter)
    state = adapter.inspect_state()
    packet = next(packet for packet in state.packets.values() if len(packet.dependent_handles) == 2)
    assert state.payload_occurrences(packet.payload_sha256) == 1
    assert set(packet.dependent_handles) == {"h0", "h1"}
```

- [ ] **Step 2: Run the test and verify the module is absent**

Run: `uv run pytest tests/baselines/test_shared_greedy.py -q`

Expected: collection fails importing `ratemem.baselines.shared_greedy`.

- [ ] **Step 3: Implement the deliberately weaker allocator**

```python
# src/ratemem/baselines/shared_greedy.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from ratemem.allocation.objective import CoverageOracle


@dataclass(frozen=True, slots=True)
class GreedyResult:
    selected_packet_ids: Sequence[str]
    total_cost: int
    objective_value: float


def plain_density_greedy(oracle: CoverageOracle, budget_bytes: int) -> GreedyResult:
    selected: Sequence[str] = ()
    remaining = set(oracle.bundles)
    used = 0
    while remaining:
        feasible = [
            packet_id
            for packet_id in remaining
            if used + oracle.bundles[packet_id].cost_bytes <= budget_bytes
        ]
        if not feasible:
            break
        packet_id = min(
            feasible,
            key=lambda item: (
                -(oracle.marginal(frozenset(selected), item) / oracle.bundles[item].cost_bytes),
                item,
            ),
        )
        if oracle.marginal(frozenset(selected), packet_id) <= 0:
            break
        selected += (packet_id,)
        remaining.remove(packet_id)
        used += oracle.bundles[packet_id].cost_bytes
    return GreedyResult(selected, used, oracle.value(frozenset(selected)))
```

Wrap this function in `SharedPacketGreedyAdapter`. Base admission, current request weights, serialized bundle costs, packet payloads, incidences, gains, and candidate order are exactly those provided to RateMem. The only deliberate differences are: no seed enumeration, no lazy shortcut, no switching penalty, and no hysteresis. Record these flags in every attempt manifest so it cannot be mistaken for the guarantee-bearing allocator.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/baselines/test_shared_greedy.py tests/allocation -q`

Expected: all tests pass; the greedy-trap fixture proves this is plain density greedy rather than an alias of RateMem's allocator.

```bash
git add src/ratemem/baselines/shared_greedy.py tests/baselines/test_shared_greedy.py
git commit -m "feat(baselines): add plain shared-packet greedy"
```

### Task 7: Implement frozen Compress-then-Serve-style and VB-LoRA-style shared-code controls

**Files:**
- Create: `src/ratemem/baselines/static_shared.py`
- Create: `schemas/ratemem-static-codebook-v1.schema.json`
- Test: `tests/baselines/test_static_shared.py`
- Test: `tests/contract/baselines/test_static_training_split.py`

- [ ] **Step 1: Write failing algebra, accounting, and train-only tests**

```python
# tests/baselines/test_static_shared.py
import numpy as np

from ratemem.baselines.static_shared import CtsCodebook, VbCodebook


def test_cts_projection_and_decode_match_locked_matrix_formula() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    codebook = CtsCodebook.from_fixture(group_bases=(basis,), quantization_bits=16)
    code = np.array([3.0, -2.0, 7.0], dtype=np.float32)
    encoded = codebook.encode(code)
    np.testing.assert_allclose(codebook.decode(encoded), np.array([3.0, -2.0, 0.0]), atol=1e-6)


def test_vb_topk_uses_indices_and_weights_from_frozen_bank() -> None:
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    codebook = VbCodebook.from_fixture(bank=bank, subvector_size=2, top_k=2, weight_bits=16)
    encoded = codebook.encode(np.array([0.75, 0.25], dtype=np.float32))
    assert encoded.indices == ((0, 1),)
    np.testing.assert_allclose(codebook.decode(encoded), np.array([0.75, 0.25]), atol=1e-3)


def test_shared_dictionary_is_reported_separately_not_hidden(static_adapter_fixture) -> None:
    adapter = static_adapter_fixture.make_cts()
    static_adapter_fixture.run_create(adapter)
    ledger = adapter.state_ledger()
    assert ledger.shared_trained_bytes == static_adapter_fixture.codebook_file.stat().st_size
    assert ledger.online_state_bytes == len(adapter.export_online_state())
    assert ledger.shared_trained_bytes not in ledger.component_bytes.values()
```

```python
# tests/contract/baselines/test_static_training_split.py
import pytest

from ratemem.baselines.static_shared import LeakageError, fit_static_codebook


def test_codebook_fit_rejects_validation_or_final_codes(code_corpus) -> None:
    for forbidden in (code_corpus.validation, code_corpus.final):
        with pytest.raises(LeakageError, match="static codebook accepts train split only"):
            fit_static_codebook(forbidden, family="cts_style_static")
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run: `uv run pytest tests/baselines/test_static_shared.py tests/contract/baselines/test_static_training_split.py -q`

Expected: collection fails importing `ratemem.baselines.static_shared`.

- [ ] **Step 3: Implement the two frozen representations and label them as style controls**

For `cts_style_static`, split target codes by the locked adapter groups. Fit a deterministic truncated SVD only on training target codes, store each right-singular basis once, and encode a concept by quantized coordinates `s_i = U_g c_{i,g}`. Decode with `U_g^T s_i`. For `vb_lora_style_static`, split codes into fixed-length subvectors, fit a deterministic training-only vector bank, select top-k bank rows by residual reduction, solve nonnegative least squares for mixture weights, and quantize indices/weights. These are clean-room code-space controls inspired by the cited representations; they are never described as reproducing the upstream LLM results.

```python
# src/ratemem/baselines/static_shared.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls


Float32 = NDArray[np.float32]


def truncated_basis(train_codes: Float32, rank: int) -> Float32:
    if train_codes.ndim != 2 or not 0 < rank <= min(train_codes.shape):
        raise ValueError("invalid static basis shape or rank")
    _, _, vh = np.linalg.svd(train_codes.astype(np.float64), full_matrices=False)
    basis = vh[:rank].astype(np.float32)
    for row in basis:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    return basis


def vb_encode_subvector(vector: Float32, bank: Float32, top_k: int) -> tuple[Sequence[int], Float32]:
    residual = vector.astype(np.float64)
    selected: list[int] = []
    for _ in range(top_k):
        scores = bank.astype(np.float64) @ residual
        index = min(range(len(scores)), key=lambda item: (-abs(float(scores[item])), item))
        if index not in selected:
            selected.append(index)
        design = bank[selected].astype(np.float64).T
        weights, _ = nnls(design, vector.astype(np.float64))
        residual = vector.astype(np.float64) - design @ weights
    return tuple(selected), weights.astype(np.float32)
```

The codebook artifact stores training-manifest hash, code-corpus hash, seed, group boundaries, rank/bank size/top-k/quantization parameters, tensor file hash, and codebook hash. It is immutable shared trained state. Per-concept coordinates, indices, weights, scales, handles, and controller metadata are online state and count exactly. A matched outer FIFO/LRU controller is selected in `policy-search.yaml` and then frozen for both static controls.

- [ ] **Step 4: Generate the codebook schema, run tests, and commit**

Run:

```bash
uv run ratemem-baselines schema static-codebook --output schemas/ratemem-static-codebook-v1.schema.json
uv run pytest tests/baselines/test_static_shared.py tests/contract/baselines/test_static_training_split.py -q
uv run ruff check src/ratemem/baselines/static_shared.py tests/baselines/test_static_shared.py tests/contract/baselines/test_static_training_split.py
uv run mypy src/ratemem/baselines/static_shared.py
```

Expected: all tests pass; validation/final leakage raises before a tensor is fitted.

```bash
git add src/ratemem/baselines/static_shared.py schemas/ratemem-static-codebook-v1.schema.json tests/baselines/test_static_shared.py tests/contract/baselines/test_static_training_split.py
git commit -m "feat(baselines): add static shared-code controls"
```

### Task 8: Implement the online SHARE-style subspace with mutable-basis accounting

**Files:**
- Create: `src/ratemem/baselines/online_share.py`
- Test: `tests/baselines/test_online_share.py`
- Test: `tests/contract/baselines/test_online_share_roundtrip.py`

- [ ] **Step 1: Write failing update, drift, and byte tests**

```python
# tests/baselines/test_online_share.py
import numpy as np

from ratemem.baselines.online_share import OnlineShareAdapter, update_subspace


def test_update_reprojects_reconstructed_old_codes_without_hidden_targets() -> None:
    basis = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    coefficients = {"h0": np.array([2.0], dtype=np.float32)}
    new_basis, new_coefficients = update_subspace(
        basis=basis,
        coefficients=coefficients,
        incoming=np.array([0.0, 3.0, 0.0], dtype=np.float32),
        incoming_handle="h1",
        rank=1,
    )
    assert set(new_coefficients) == {"h0", "h1"}
    assert new_basis.shape == (1, 3)


def test_mutable_basis_is_counted_as_online_state(online_share_fixture) -> None:
    adapter = OnlineShareAdapter(rank=online_share_fixture.rank)
    online_share_fixture.run_creates(adapter)
    ledger = adapter.state_ledger()
    assert ledger.component_bytes["allocator_state"] >= adapter.inspect_state().basis.nbytes
    assert ledger.online_state_bytes <= online_share_fixture.contract.byte_budget


def test_new_concept_records_reprojection_drift_for_existing_handles(online_share_fixture) -> None:
    adapter = OnlineShareAdapter(rank=online_share_fixture.rank)
    receipts = online_share_fixture.run_creates(adapter)
    assert "h0" in receipts[-1].affected_handles
    assert receipts[-1].method_state_sha256_before != receipts[-1].method_state_sha256_after
```

```python
# tests/contract/baselines/test_online_share_roundtrip.py
def test_export_import_preserves_basis_coefficients_usage_and_next_update(online_share_fixture) -> None:
    first = online_share_fixture.make_adapter()
    online_share_fixture.run_prefix(first)
    restored = online_share_fixture.make_adapter()
    restored.initialize(online_share_fixture.contract)
    restored.import_online_state(first.export_online_state())
    assert restored.apply_event(online_share_fixture.next_event, online_share_fixture.next_view) == first.apply_event(
        online_share_fixture.next_event, online_share_fixture.next_view
    )
```

- [ ] **Step 2: Run tests and verify the online SHARE module is absent**

Run: `uv run pytest tests/baselines/test_online_share.py tests/contract/baselines/test_online_share_roundtrip.py -q`

Expected: collection fails importing `ratemem.baselines.online_share`.

- [ ] **Step 3: Implement deterministic online subspace update**

```python
# src/ratemem/baselines/online_share.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Float32 = NDArray[np.float32]


def canonicalize_basis_signs(basis: Float32) -> Float32:
    result = basis.copy()
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    return result


def update_subspace(
    basis: Float32,
    coefficients: dict[str, Float32],
    incoming: Float32,
    incoming_handle: str,
    rank: int,
) -> tuple[Float32, dict[str, Float32]]:
    reconstructed = {handle: coef @ basis for handle, coef in coefficients.items()}
    reconstructed[incoming_handle] = incoming
    handles = sorted(reconstructed)
    matrix = np.stack([reconstructed[handle] for handle in handles]).astype(np.float64)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    new_basis = canonicalize_basis_signs(vh[: min(rank, len(vh))].astype(np.float32))
    new_coefficients = {
        handle: (reconstructed[handle] @ new_basis.T).astype(np.float32) for handle in handles
    }
    return new_basis, new_coefficients
```

The adapter never stores original target codes after projection. On `CREATE`/`UPDATE`, reconstruct current codes from the resident basis/coefficients, add or replace the incoming target, update the basis deterministically, and reproject all active handles. Store the mutable basis, all coefficients, quantization scales, drift hashes, handles, usage/age, and controller state in the online ledger. Under pressure, use the exact same locked outer policy as the static controls. `READ` never updates the subspace. `PROBE` uses a copied basis and cannot change usage. The fidelity suite in Task 14 compares this matrix update and drift semantics with the pinned SHARE diffusion source; the paper calls it `SHARE-style` unless that audit justifies a stronger name.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run pytest tests/baselines/test_online_share.py tests/contract/baselines/test_online_share_roundtrip.py tests/baselines/test_ledger.py -q
uv run ruff check src/ratemem/baselines/online_share.py tests/baselines/test_online_share.py tests/contract/baselines/test_online_share_roundtrip.py
uv run mypy src/ratemem/baselines/online_share.py
```

Expected: all tests pass; restoring an export reproduces the next update receipt exactly; the mutable basis is never reported as free shared trained state.

```bash
git add src/ratemem/baselines/online_share.py tests/baselines/test_online_share.py tests/contract/baselines/test_online_share_roundtrip.py
git commit -m "feat(baselines): add online shared-subspace control"
```

### Task 9: Implement DreamCache-style feature state and the stateless amortizer control

**Files:**
- Create: `src/ratemem/baselines/feature_cache.py`
- Create: `src/ratemem/baselines/stateless.py`
- Test: `tests/baselines/test_feature_cache.py`
- Test: `tests/baselines/test_stateless.py`

- [ ] **Step 1: Write failing feature-state lifecycle and role tests**

```python
# tests/baselines/test_feature_cache.py
from ratemem.baselines.feature_cache import FeatureCacheAdapter
from tests.fixtures.baselines.feature_backend import RecordingFeatureBackend
from tests.fixtures.baselines.runtime import run_events


def test_create_stores_features_but_not_support_images(feature_fixture) -> None:
    backend = RecordingFeatureBackend()
    adapter = FeatureCacheAdapter("dreamcache_feature_cache", backend=backend, policy="lru")
    run_events(adapter, feature_fixture.contract, feature_fixture.single_create)
    exported = adapter.export_online_state()
    state = adapter.inspect_state()
    assert state.handles == ("h0",)
    assert state.cached_features["h0"].shape == feature_fixture.expected_shape
    assert feature_fixture.raw_support_bytes not in exported
    assert adapter.state_ledger().component_bytes["feature_cache"] > 0


def test_update_replaces_feature_record_and_delete_reclaims_it(feature_fixture) -> None:
    adapter = FeatureCacheAdapter("dreamcache_feature_cache", RecordingFeatureBackend(), policy="lru")
    receipts = run_events(adapter, feature_fixture.contract, feature_fixture.create_update_delete)
    assert receipts[1].outcome == "updated"
    assert receipts[2].outcome == "deleted"
    assert adapter.inspect_state().handles == ()


def test_feature_probe_is_read_only(feature_fixture) -> None:
    adapter = FeatureCacheAdapter("dreamcache_feature_cache", RecordingFeatureBackend(), policy="lru")
    run_events(adapter, feature_fixture.contract, feature_fixture.single_create)
    before = adapter.export_online_state()
    adapter.score_probe(adapter.copy_snapshot(), feature_fixture.probe)
    assert adapter.export_online_state() == before
```

```python
# tests/baselines/test_stateless.py
import pytest

from ratemem.baselines.catalog import CatalogError
from ratemem.baselines.stateless import StatelessAmortizerAdapter


def test_stateless_control_reports_external_support_storage_and_cannot_enter_storage_frontier(stateless_fixture) -> None:
    adapter = StatelessAmortizerAdapter(stateless_fixture.support_provider, stateless_fixture.amortizer)
    stateless_fixture.run_read(adapter)
    ledger = adapter.state_ledger()
    assert adapter.role == "latency_control"
    assert ledger.external_support_bytes == stateless_fixture.support_provider.serialized_bytes
    assert ledger.online_state_bytes == len(adapter.export_online_state())
    with pytest.raises(CatalogError, match="latency control cannot enter storage frontier"):
        stateless_fixture.primary_selector.validate(adapter)
```

- [ ] **Step 2: Run tests and verify the modules are absent**

Run: `uv run pytest tests/baselines/test_feature_cache.py tests/baselines/test_stateless.py -q`

Expected: collection fails importing `ratemem.baselines.feature_cache` or `ratemem.baselines.stateless`.

- [ ] **Step 3: Define the feature backend and byte-exact adapter**

```python
# src/ratemem/baselines/feature_cache.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CachedFeature:
    tensor: NDArray[np.generic]
    tap_path: str
    injection_path: str
    encoding_timestep: int
    scale: float


class FeatureBackend(Protocol):
    backbone_id: str
    source_revision: str

    def encode_support(self, support_image_ids: Sequence[str], description_id: str) -> CachedFeature:
        raise NotImplementedError

    def generate(self, feature: CachedFeature, prompt_id: str, seed: int) -> Tensor:
        raise NotImplementedError

    def one_step_latent(self, feature: CachedFeature, prompt_id: str, seed: int, timestep: int) -> Tensor:
        raise NotImplementedError
```

`FeatureCacheAdapter` calls `encode_support` only on `CREATE`/`UPDATE`; it stores the canonical feature tensor, dtype/shape, tap/injection paths, encoding timestep, scale, handle, usage/age, reference metadata, and controller state. It never retains image bytes or support IDs after encoding. Under pressure it uses the same locked FIFO/LRU controller chosen for the feature-cache family, with all feature bytes counted. It passes exact prompt and generation seed to `FeatureBackend.generate`. Task 12 supplies the JSONL-backed DreamCache implementation; Task 14 must prove its feature/one-step outputs against the pinned upstream implementation on the locked fixture before this adapter becomes eligible.

- [ ] **Step 4: Implement the explicitly nondeployable stateless control**

```python
# src/ratemem/baselines/stateless.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from torch import Tensor


class RetainedSupportProvider(Protocol):
    serialized_bytes: int

    def support_for_handle(self, handle: str) -> Sequence[Tensor]:
        raise NotImplementedError

    def description_for_handle(self, handle: str) -> Tensor:
        raise NotImplementedError


class Amortizer(Protocol):
    checkpoint_sha256: str

    def __call__(self, support: Sequence[Tensor], description: Tensor) -> Tensor:
        raise NotImplementedError
```

`StatelessAmortizerAdapter` stores only the minimal handle table needed to address an external `RetainedSupportProvider`, recomputes the target code with the identical frozen amortizer on every `READ`, and installs it in the matched backbone runner. Its ledger reports support bytes in `external_support_bytes`, never as zero-cost state. Its role is `latency_control`; the selector rejects it from the byte-quality frontier and allocator claim. It remains useful for measuring the latency/quality cost of avoiding a learned online memory.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/baselines/test_feature_cache.py tests/baselines/test_stateless.py tests/baselines/test_ledger.py -q
uv run ruff check src/ratemem/baselines/feature_cache.py src/ratemem/baselines/stateless.py tests/baselines/test_feature_cache.py tests/baselines/test_stateless.py
uv run mypy src/ratemem/baselines/feature_cache.py src/ratemem/baselines/stateless.py
```

Expected: all tests pass; support bytes cannot disappear from the stateless control's disclosure and DreamCache state contains no raw support image.

```bash
git add src/ratemem/baselines/feature_cache.py src/ratemem/baselines/stateless.py tests/baselines/test_feature_cache.py tests/baselines/test_stateless.py
git commit -m "feat(baselines): add feature-cache and stateless controls"
```

### Task 10: Implement the matched per-concept LoRA optimization reference

**Files:**
- Create: `src/ratemem/baselines/lora_reference.py`
- Create: `configs/baselines/lora-reference.yaml`
- Test: `tests/baselines/test_lora_reference.py`
- Test: `tests/contract/baselines/test_lora_reference_contract.py`

- [ ] **Step 1: Write failing frozen-backbone, state, and event tests**

```python
# tests/baselines/test_lora_reference.py
from ratemem.baselines.lora_reference import LoRAOptimizationAdapter, load_lora_reference_config
from tests.fixtures.baselines.lora_trainer import RecordingLoRATrainer
from tests.fixtures.baselines.runtime import run_events


def test_create_optimizes_only_qkv_lora_and_exports_all_persistent_state(lora_fixture) -> None:
    trainer = RecordingLoRATrainer(lora_fixture.backbone)
    adapter = LoRAOptimizationAdapter(load_lora_reference_config(lora_fixture.config), trainer)
    run_events(adapter, lora_fixture.contract, lora_fixture.single_create)
    assert trainer.changed_parameter_names
    assert all(name.endswith(("to_q.lora_A", "to_q.lora_B", "to_k.lora_A", "to_k.lora_B", "to_v.lora_A", "to_v.lora_B")) for name in trainer.changed_parameter_names)
    assert trainer.frozen_parameter_hash_before == trainer.frozen_parameter_hash_after
    assert adapter.state_ledger().online_state_bytes == len(adapter.export_online_state())


def test_optimizer_state_is_discarded_and_not_needed_for_deterministic_read(lora_fixture) -> None:
    adapter = lora_fixture.make_adapter()
    run_events(adapter, lora_fixture.contract, lora_fixture.single_create)
    assert adapter.inspect_state().optimizer_state_present is False
    restored = lora_fixture.make_adapter()
    restored.initialize(lora_fixture.contract)
    restored.import_online_state(adapter.export_online_state())
    assert lora_fixture.read(adapter) == lora_fixture.read(restored)


def test_update_uses_only_new_evidence_under_the_locked_update_rule(lora_fixture) -> None:
    adapter = lora_fixture.make_adapter()
    run_events(adapter, lora_fixture.contract, lora_fixture.create_then_update)
    calls = adapter.trainer.calls
    assert calls[0].initial_state == "zero_lora"
    assert calls[1].initial_state == "resident_lora"
    assert calls[1].support_ids == lora_fixture.update_support_ids
```

```python
# tests/contract/baselines/test_lora_reference_contract.py
import pytest

from ratemem.baselines.lora_reference import PrimaryBackboneError, require_primary_lora_contract


def test_sana_reference_shares_generation_contract_with_ratemem(lora_sana_fixture) -> None:
    fixture = lora_sana_fixture
    assert fixture.lora_contract.backbone_id == "sana_1_5_1_6b"
    assert fixture.lora_contract.backbone_revision == fixture.ratemem_contract.backbone_revision
    assert fixture.lora_contract.sampler_id == fixture.ratemem_contract.sampler_id
    assert fixture.lora_contract.cfg_scale == fixture.ratemem_contract.cfg_scale
    assert fixture.lora_contract.resolution == fixture.ratemem_contract.resolution
    assert fixture.lora_contract.denoising_steps == fixture.ratemem_contract.denoising_steps
    assert fixture.lora_contract.prompt_pool_sha256 == fixture.ratemem_contract.prompt_pool_sha256
    assert fixture.lora_contract.noise_seed_manifest_sha256 == fixture.ratemem_contract.noise_seed_manifest_sha256


def test_sdxl_native_lora_contract_is_contextual_and_rejected_from_primary(contextual_sdxl_lora_contract) -> None:
    with pytest.raises(PrimaryBackboneError, match="primary comparisons require sana_1_5_1_6b"):
        require_primary_lora_contract(contextual_sdxl_lora_contract)
```

- [ ] **Step 2: Run tests and verify the reference is missing**

Run: `uv run pytest tests/baselines/test_lora_reference.py tests/contract/baselines/test_lora_reference_contract.py -q`

Expected: collection fails importing `ratemem.baselines.lora_reference`.

- [ ] **Step 3: Add an exact, validation-tuned LoRA contract**

```yaml
# configs/baselines/lora-reference.yaml
schema_version: "1.0"
method_id: per_concept_lora
target_suffixes: [to_q, to_k, to_v]
precision: bfloat16
gradient_checkpointing: true
optimizer: adamw
discard_optimizer_after_event: true
create_initial_state: zero_lora
update_initial_state: resident_lora
update_support_rule: new_evidence_only
search_space:
  rank: [2, 4, 8, 16]
  learning_rate: [0.00001, 0.00005, 0.0001]
  steps: [50, 100]
  prior_preservation_weight: [0.0]
search_selector:
  split: validation
  endpoint: request_weighted_identity
  prompt_constraint_source: evaluation_lock
```

There are exactly 24 grid cells, matching the locked maximum trials per method. The scientific search harness counts actual GPU time and may not exceed 48 GPU-hours. No final concept, final prompt, final seed, or final metric enters the search. If a cell fails, it remains a consumed trial with a failure artifact; it is not silently replaced.

- [ ] **Step 4: Implement trainable-state isolation and lifecycle semantics**

Define a `PerConceptLoRATrainer` protocol with `fit(backbone, support_ids, description_id, initial_lora, config, seed) -> LoRAState` and `generate(backbone, state, prompt_id, seed, contract) -> Tensor`. The production trainer accepts only `SanaBackboneRunner` and uses Diffusers/PEFT for static LoRA execution; `require_primary_lora_contract` rejects every non-SANA contract before model loading. It verifies all base parameter hashes before and after each fit. `CREATE` starts from zero LoRA. `UPDATE` resumes the resident LoRA and uses only new evidence, as frozen above, so no uncounted support archive is needed. Optimizer tensors are transient and reported in training peak memory but discarded after the event. Persistent LoRA safetensors bytes, per-tensor names/dtypes/shapes, handles, optional trigger tokens, usage/age, controller state, and checksums are canonical online state. Any SDXL-native LoRA result is ingested only as labeled contextual evidence through the external-evidence path and is never paired with RateMem.

Under a hard lifecycle cap, use the locked independent LRU outer controller and reject a LoRA that cannot fit alone. For the optimization-free acquisition claim, compare on held-out concepts without pressure and report insertion latency, training energy, and state bytes. Do not treat a contextual DreamBooth/Textual-Inversion published number as a matched result.

- [ ] **Step 5: Run synthetic training contracts and commit**

Run:

```bash
uv run pytest tests/baselines/test_lora_reference.py tests/contract/baselines/test_lora_reference_contract.py -q
uv run ruff check src/ratemem/baselines/lora_reference.py tests/baselines/test_lora_reference.py tests/contract/baselines/test_lora_reference_contract.py
uv run mypy src/ratemem/baselines/lora_reference.py
```

Expected: all tests pass; fake-backbone hashes prove isolation; export/import reproduces a fixed-seed read.

```bash
git add configs/baselines/lora-reference.yaml src/ratemem/baselines/lora_reference.py tests/baselines/test_lora_reference.py tests/contract/baselines/test_lora_reference_contract.py
git commit -m "feat(baselines): add per-concept LoRA reference"
```

### Task 11: Implement exact append-only and future-trace packet upper references

**Files:**
- Create: `src/ratemem/baselines/oracles.py`
- Create: `schemas/ratemem-baseline-oracle-certificate-v1.schema.json`
- Test: `tests/baselines/test_append_only_oracle.py`
- Test: `tests/baselines/test_future_trace_oracle.py`
- Test: `tests/contract/baselines/test_oracle_roles.py`

- [ ] **Step 1: Write failing exactness and role-separation tests**

```python
# tests/baselines/test_append_only_oracle.py
from ratemem.baselines.oracles import ExactAppendOnlyAdapter
from tests.fixtures.baselines.runtime import run_events


def test_append_only_never_evicts_or_changes_an_admitted_record(oracle_fixture) -> None:
    adapter = ExactAppendOnlyAdapter(oracle_fixture.teacher_codes, oracle_fixture.quantizers)
    receipts = run_events(adapter, oracle_fixture.contract, oracle_fixture.overflow_trace)
    assert all(not receipt.evicted_handles for receipt in receipts)
    assert receipts[-1].outcome == "rejected"
    assert adapter.inspect_state().record_hashes["h0"] == oracle_fixture.first_admitted_hash


def test_append_only_selects_minimum_distortion_feasible_quantizer(oracle_fixture) -> None:
    adapter = ExactAppendOnlyAdapter(oracle_fixture.teacher_codes, oracle_fixture.quantizers)
    run_events(adapter, oracle_fixture.contract, oracle_fixture.single_create)
    assert adapter.inspect_state().quantizer_id["h0"] == oracle_fixture.best_feasible_quantizer_id
```

```python
# tests/baselines/test_future_trace_oracle.py
from hypothesis import given

from ratemem.baselines.oracles import solve_future_trace
from tests.fixtures.baselines.oracles import brute_force_future_trace, future_trace_instances


@given(problem=future_trace_instances(max_events=5, max_handles=4, max_packets=6))
def test_future_milp_matches_exhaustive_optimum(problem) -> None:
    result = solve_future_trace(problem)
    expected = brute_force_future_trace(problem)
    assert result.status == "optimal"
    assert result.objective_integer == expected.objective_integer
    assert result.allocations == expected.allocations
    assert all(row.serialized_bytes <= problem.byte_budget for row in result.allocations)
```

```python
# tests/contract/baselines/test_oracle_roles.py
import pytest

from ratemem.baselines.oracles import FutureTracePacketAdapter
from ratemem.baselines.protocol import FutureAccessError


def test_future_trace_is_available_only_to_upper_reference_factory(oracle_fixture) -> None:
    with pytest.raises(FutureAccessError, match="full trace requires upper_reference role"):
        oracle_fixture.causal_factory.make(FutureTracePacketAdapter, full_trace=oracle_fixture.trace)
    adapter = oracle_fixture.upper_reference_factory.make(FutureTracePacketAdapter, full_trace=oracle_fixture.trace)
    assert adapter.role == "upper_reference"
```

- [ ] **Step 2: Run tests and verify the oracle module is absent**

Run: `uv run pytest tests/baselines/test_append_only_oracle.py tests/baselines/test_future_trace_oracle.py tests/contract/baselines/test_oracle_roles.py -q`

Expected: collection fails importing `ratemem.baselines.oracles`.

- [ ] **Step 3: Implement the append-only teacher-code reference**

For each `CREATE`, obtain the exact teacher adapter code from the locked training target, not the amortizer. Evaluate every prespecified deterministic quantizer by code-space squared error, discard options whose canonical record cannot fit the remaining bytes, and choose `(minimum distortion, minimum bytes, quantizer_id)` lexicographically. Once admitted, a record is immutable until explicit `DELETE`; `UPDATE` creates a new version only if both old and new records fit during atomic replacement, otherwise it rejects. It never evicts another concept. Because it has teacher-code access, its role is `upper_reference`, not a deployable baseline.

```python
from collections.abc import Sequence


def choose_append_option(options: Sequence[QuantizedTeacherCode], remaining_bytes: int) -> QuantizedTeacherCode | None:
    feasible = [option for option in options if option.serialized_bytes <= remaining_bytes]
    if not feasible:
        return None
    return min(feasible, key=lambda option: (option.squared_error, option.serialized_bytes, option.quantizer_id))
```

- [ ] **Step 4: Implement the full-future mixed-integer oracle and certificate**

`FutureTraceProblem` contains integer-scaled nonnegative utilities, exact base and bundle byte costs, bundle incidence sets, create/delete availability, and a finite full trace. Build variables `y[t,i]` for admitted base records, `x[t,p]` for packets, and continuous coverage `z[t,i,g]`. Add:

- exact capacity at every event: base bytes plus selected packet-bundle bytes do not exceed `B`;
- `y[t,i]=0` before create and after delete;
- once a live handle is evicted, `y[t,i] <= y[t-1,i]` until delete;
- `x[t,p] <= y[t,i]` for every dependent incidence and zero before packet proposal;
- `0 <= z[t,i,g] <= 1` and `z[t,i,g] <= sum_p v[t,i,g,p] x[t,p]`;
- the integer-scaled request-weighted coverage objective over the entire trace, with the locked switching term if it is nonzero.

```python
# src/ratemem/baselines/oracles.py (solver boundary)
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


class OracleNotOptimal(RuntimeError):
    pass


def solve_future_trace(problem: FutureTraceProblem) -> FutureTraceResult:
    model = problem.to_milp_arrays()
    result = milp(
        c=model.objective,
        integrality=model.integrality,
        bounds=Bounds(model.lower_bounds, model.upper_bounds),
        constraints=LinearConstraint(model.matrix, model.constraint_lower, model.constraint_upper),
        options={"mip_rel_gap": 0.0, "presolve": True, "disp": False},
    )
    if result.status != 0 or result.x is None:
        raise OracleNotOptimal(f"future trace oracle did not prove optimality: status={result.status}")
    decoded = problem.decode_solution(result.x)
    problem.verify_integer_solution(decoded)
    return decoded.with_solver_certificate(
        solver="scipy-highs-milp",
        status="optimal",
        objective_integer=problem.objective_integer(decoded),
    )
```

The evaluation lock bounds oracle problem size. A time-limited feasible solution, nonzero MIP gap, floating capacity violation, or failed integer recomputation is invalid rather than reported as exact. Exhaustive enumeration cross-checks every synthetic instance and a deterministic sample of locked reduced traces. The full scientific trace can report oracle regret only if it returns an optimal certificate; otherwise the oracle-dependent claim is blocked.

- [ ] **Step 5: Generate the certificate schema, run tests, and commit**

Run:

```bash
uv run ratemem-baselines schema oracle-certificate --output schemas/ratemem-baseline-oracle-certificate-v1.schema.json
uv run pytest tests/baselines/test_append_only_oracle.py tests/baselines/test_future_trace_oracle.py tests/contract/baselines/test_oracle_roles.py -q
uv run ruff check src/ratemem/baselines/oracles.py tests/baselines/test_append_only_oracle.py tests/baselines/test_future_trace_oracle.py tests/contract/baselines/test_oracle_roles.py
uv run mypy src/ratemem/baselines/oracles.py
```

Expected: all tests pass; every Hypothesis MILP matches exhaustive search and every oracle certificate states `optimal`.

```bash
git add src/ratemem/baselines/oracles.py schemas/ratemem-baseline-oracle-certificate-v1.schema.json tests/baselines/test_append_only_oracle.py tests/baselines/test_future_trace_oracle.py tests/contract/baselines/test_oracle_roles.py
git commit -m "feat(baselines): add exact upper references"
```

### Task 12: Add the strict ExternalJsonl adapter for pinned original implementations

**Files:**
- Create: `src/ratemem/baselines/external_jsonl.py`
- Create: `schemas/ratemem-external-baseline-message-v1.schema.json`
- Create: `external_baselines/hyperlora/runner.py`
- Create: `external_baselines/dreamcache/runner.py`
- Create: `external_baselines/share/runner.py`
- Test: `tests/baselines/test_external_jsonl.py`
- Test: `tests/contract/baselines/test_external_worker_protocol.py`
- Create: `tests/fixtures/baselines/external_worker.py`

- [ ] **Step 1: Write failing framing, timeout, state-export, and probe tests**

```python
# tests/baselines/test_external_jsonl.py
import sys
from collections.abc import Sequence

import pytest

from ratemem.baselines.external_jsonl import ExternalJsonlAdapter, ExternalProtocolError


def worker(mode: str) -> Sequence[str]:
    return (sys.executable, "tests/fixtures/baselines/external_worker.py", "--mode", mode)


def test_external_worker_roundtrips_state_and_host_recomputes_bytes(external_fixture) -> None:
    adapter = ExternalJsonlAdapter(external_fixture.manifest(command=worker("valid")))
    adapter.initialize(external_fixture.contract)
    receipt = adapter.apply_event(external_fixture.create, external_fixture.create_view)
    exported = adapter.export_online_state()
    assert receipt.ledger.online_state_bytes == len(exported)
    restored = ExternalJsonlAdapter(external_fixture.manifest(command=worker("valid")))
    restored.initialize(external_fixture.contract)
    restored.import_online_state(exported)
    assert restored.apply_event(external_fixture.read, external_fixture.read_view).decoded_code_sha256 == adapter.apply_event(
        external_fixture.read, external_fixture.read_view
    ).decoded_code_sha256


@pytest.mark.parametrize("mode", ["stdout_log", "extra_field", "wrong_index", "invalid_base64"])
def test_any_noncanonical_response_invalidates_worker(mode: str, external_fixture) -> None:
    adapter = ExternalJsonlAdapter(external_fixture.manifest(command=worker(mode)))
    with pytest.raises(ExternalProtocolError):
        adapter.initialize(external_fixture.contract)


def test_worker_timeout_is_failure_not_a_partial_result(external_fixture) -> None:
    adapter = ExternalJsonlAdapter(external_fixture.manifest(command=worker("hang"), timeout_seconds=0.1))
    with pytest.raises(ExternalProtocolError, match="deadline"):
        adapter.initialize(external_fixture.contract)
```

```python
# tests/contract/baselines/test_external_worker_protocol.py
def test_probe_uses_snapshot_token_and_does_not_change_export(external_adapter, external_fixture) -> None:
    external_adapter.initialize(external_fixture.contract)
    external_adapter.apply_event(external_fixture.create, external_fixture.create_view)
    before = external_adapter.export_online_state()
    result = external_adapter.score_probe(external_adapter.copy_snapshot(), external_fixture.probe)
    assert result.update_usage is False
    assert external_adapter.export_online_state() == before


def test_subprocess_uses_argv_without_shell_and_minimal_environment(external_adapter) -> None:
    launch = external_adapter.inspect_launch()
    assert launch.shell is False
    assert set(launch.environment) <= {"PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE"}


```

- [ ] **Step 2: Run tests and verify the JSONL adapter is missing**

Run: `uv run pytest tests/baselines/test_external_jsonl.py tests/contract/baselines/test_external_worker_protocol.py -q`

Expected: collection fails importing `ratemem.baselines.external_jsonl`.

- [ ] **Step 3: Implement one-request/one-response canonical JSONL framing**

The seven request operations are `initialize`, `event`, `snapshot`, `probe`, `export_state`, `import_state`, and `close`. Every request has `protocol_version="1.0"`, monotonic `request_id`, `method_id`, `trace_id`, and one operation payload. Every response echoes those identity fields, has `status` equal to `ok` or `error`, and contains exactly the schema fields for its operation. Standard output is protocol-only; diagnostic text goes to standard error and is redacted before artifact storage.

```python
# src/ratemem/baselines/external_jsonl.py (transport boundary)
from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ratemem.evaluation.canonical import canonical_json_bytes


class ExternalProtocolError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalWorkerManifest:
    method_id: str
    command: Sequence[str]
    checkout: Path
    source_revision: str
    environment: dict[str, str]
    timeout_seconds: float
    maximum_line_bytes: int = 16 * 1024 * 1024


def canonical_line(message: dict[str, Any]) -> bytes:
    return canonical_json_bytes(message) + b"\n"


def decode_state_export(response: dict[str, Any]) -> bytes:
    try:
        payload = base64.b64decode(response["state_cbor_base64"], validate=True)
    except (KeyError, ValueError) as error:
        raise ExternalProtocolError("invalid state export") from error
    if response["state_bytes"] != len(payload):
        raise ExternalProtocolError("worker state byte declaration disagrees with export")
    return payload
```

Launch with `subprocess.Popen(list(command), shell=False, cwd=checkout, env=allowlisted_environment, stdin=PIPE, stdout=PIPE, stderr=PIPE)`. Use a per-request deadline, reject lines above `maximum_line_bytes`, parse one UTF-8 JSON object, validate it against the operation schema, and reject additional standard-output bytes. On timeout or protocol error, terminate, then kill after a bounded grace period; do not return the last good receipt as a completed attempt. The host decodes `state_cbor_base64`, recomputes its SHA-256 and ledger, and overwrites any worker-reported total before constructing `EventReceipt`.

- [ ] **Step 4: Implement thin upstream runners without vendoring source**

Each method-specific file under `external_baselines/` imports only from the pinned checkout path supplied by its audited worker manifest and implements the operation loop below. Method-specific functions must translate the common contract into upstream inputs and translate upstream cache/adapter state into Task 2's canonical component map. They must not emit an estimate when upstream state cannot be exported.

```python
def serve(backend: UpstreamBackend) -> int:
    for raw in sys.stdin.buffer:
        request = validate_request(json.loads(raw))
        try:
            response = dispatch(backend, request)
        except Exception as error:
            response = error_response(request, type(error).__name__, str(error))
        sys.stdout.buffer.write(canonical_line(response))
        sys.stdout.buffer.flush()
        if request["operation"] == "close":
            return 0
    return 2
```

The HyperLoRA runner is eligible only for the locked portrait-acquisition cohort. The DreamCache runner exposes cached feature tensors and one-step latents. The SHARE runner exposes matrix/basis updates for fidelity cases, not published scores. SineLoRA-Delta has no auditable upstream implementation in the locked source catalog and was evaluated on Stable Diffusion 3 Medium rather than SANA, so it remains a contextual citation and deliberately has no runner or `BaselineAdapter` factory. A runner whose pinned source lacks an executable license, immutable revision, or exportable state remains disabled by Task 13's audit.

- [ ] **Step 5: Generate message schemas and run protocol tests**

Run:

```bash
uv run ratemem-baselines schema external-message --output schemas/ratemem-external-baseline-message-v1.schema.json
uv run pytest tests/baselines/test_external_jsonl.py tests/contract/baselines/test_external_worker_protocol.py -q
uv run ruff check src/ratemem/baselines/external_jsonl.py external_baselines tests/baselines/test_external_jsonl.py tests/contract/baselines/test_external_worker_protocol.py
uv run mypy src/ratemem/baselines/external_jsonl.py
```

Expected: all tests pass; malformed output, extra output, wrong identity, invalid state, and deadline expiration all fail closed.

- [ ] **Step 6: Commit the external adapter bridge**

```bash
git add src/ratemem/baselines/external_jsonl.py external_baselines schemas/ratemem-external-baseline-message-v1.schema.json tests/baselines/test_external_jsonl.py tests/contract/baselines/test_external_worker_protocol.py tests/fixtures/baselines/external_worker.py
git commit -m "feat(baselines): add audited external JSONL bridge"
```

### Task 13: Build an immutable source, revision, and license inventory

**Files:**
- Create: `configs/baselines/source-registry.yaml`
- Create: `src/ratemem/baselines/sources.py`
- Create: `schemas/ratemem-baseline-source-inventory-v1.schema.json`
- Test: `tests/baselines/test_sources.py`
- Create: `tests/fixtures/baselines/source-repositories.py`

- [ ] **Step 1: Write failing source immutability and license tests**

```python
# tests/baselines/test_sources.py
from pathlib import Path

import pytest

from ratemem.baselines.sources import SourceAuditError, inventory_source
from tests.fixtures.baselines.source_repositories import make_repository


def test_inventory_resolves_commit_archive_and_license_hash(tmp_path: Path) -> None:
    repository = make_repository(tmp_path, license_expression="Apache-2.0")
    record = inventory_source(repository.registry_entry, cache_dir=tmp_path / "cache")
    assert len(record.source_revision) == 40
    assert len(record.source_archive_sha256) == 64
    assert record.license_expression == "Apache-2.0"
    assert len(record.license_file_sha256) == 64


def test_required_external_source_without_license_is_not_executable(tmp_path: Path) -> None:
    repository = make_repository(tmp_path, license_expression=None)
    with pytest.raises(SourceAuditError, match="required external source has no auditable license"):
        inventory_source(repository.registry_entry, cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD", ""])
def test_sealed_source_record_rejects_mutable_revision(revision: str, source_record) -> None:
    with pytest.raises(ValueError):
        source_record.model_copy(update={"source_revision": revision}).validate_sealed()
```

- [ ] **Step 2: Run tests and verify the source module is missing**

Run: `uv run pytest tests/baselines/test_sources.py -q`

Expected: collection fails importing `ratemem.baselines.sources`.

- [ ] **Step 3: Add the exact source registry**

```yaml
# configs/baselines/source-registry.yaml
schema_version: "1.0"
sources:
  - source_id: hyperlora_upstream
    methods: [hyperlora_upstream]
    kind: git
    repository_url: https://github.com/bytedance/ComfyUI-HyperLoRA.git
    license_required_for_execution: true
  - source_id: dreamcache_upstream
    methods: [dreamcache_feature_cache]
    kind: git
    repository_url: https://github.com/Emanuele97x/DreamCache.git
    license_required_for_execution: true
  - source_id: share_upstream
    methods: [share_style_online]
    kind: git
    repository_url: https://github.com/ankit-vaidya19/Share.git
    license_required_for_execution: true
  - source_id: vb_lora_upstream
    methods: [vb_lora_style_static]
    kind: git
    repository_url: https://github.com/leo-yangli/VB-LoRA.git
    license_required_for_execution: false
  - source_id: compress_then_serve_paper
    methods: [cts_style_static]
    kind: paper
    canonical_url: https://proceedings.mlr.press/v267/gabrielsson25a.html
    license_required_for_execution: false
  - source_id: diffusers_upstream
    methods: [per_concept_lora]
    kind: installed_distribution
    distribution: diffusers
    locked_version: "0.35.1"
    license_required_for_execution: true
  - source_id: peft_upstream
    methods: [per_concept_lora]
    kind: installed_distribution
    distribution: peft
    locked_version: "0.17.1"
    license_required_for_execution: true
```

This registry intentionally contains no branch name or claimed commit. `inventory resolve` performs the one-time resolution, records the actual 40-hex commit, archives that commit, hashes it, locates license files/SPDX package metadata, and writes a sealed inventory. Paper-only entries authorize clean-room implementation from the cited algorithm, not execution of unlicensed source.

- [ ] **Step 4: Implement non-vendoring source resolution**

Use `subprocess.run` with argument arrays and `check=True`: `git ls-remote {repository_url} HEAD`, `git clone --filter=blob:none --no-checkout {repository_url} {cache_path}`, `git -C {cache_path} fetch {commit_sha}`, and `git -C {cache_path} archive --format=tar {commit_sha}`. Store checkouts only under ignored `data/cache/baselines/sources/{commit_sha}/`. Hash the archive stream and every detected license file. Accept an SPDX expression only from an upstream SPDX field, package metadata, or a separately signed operator declaration that itself enters the inventory; never infer a license from repository visibility. `NOASSERTION` blocks execution when `license_required_for_execution=true`.

The sealed `SourceRecord` fields are `source_id`, `kind`, canonical URL/distribution, 40-hex revision or exact installed wheel hash, archive/wheel SHA-256, license expression, license file paths and hashes, resolver tool versions, resolution timestamp, and record SHA-256. Re-running verification checks the local archive against the record and performs no network write.

- [ ] **Step 5: Generate schema, run tests, and commit**

Run:

```bash
uv run ratemem-baselines schema source-inventory --output schemas/ratemem-baseline-source-inventory-v1.schema.json
uv run pytest tests/baselines/test_sources.py -q
uv run ruff check src/ratemem/baselines/sources.py tests/baselines/test_sources.py tests/fixtures/baselines/source_repositories.py
uv run mypy src/ratemem/baselines/sources.py
```

Expected: all tests pass against local fixture repositories; no network access is needed in CI.

```bash
git add configs/baselines/source-registry.yaml src/ratemem/baselines/sources.py schemas/ratemem-baseline-source-inventory-v1.schema.json tests/baselines/test_sources.py tests/fixtures/baselines/source_repositories.py
git commit -m "feat(baselines): audit source revisions and licenses"
```

### Task 14: Lock fidelity cases, equal search budgets, and the SANA-only primary gate

**Files:**
- Create: `configs/baselines/fidelity-policy.yaml`
- Create: `configs/baselines/policy-search.yaml`
- Create: `src/ratemem/baselines/fidelity.py`
- Create: `schemas/ratemem-baseline-fidelity-report-v1.schema.json`
- Create: `schemas/ratemem-baseline-audit-receipt-v1.schema.json`
- Test: `tests/baselines/test_fidelity.py`
- Test: `tests/baselines/test_search_budget.py`
- Test: `tests/contract/baselines/test_primary_backbone_gate.py`
- Test: `tests/contract/baselines/test_required_evidence.py`

- [ ] **Step 1: Write failing fidelity, search, SANA-gate, and real-implementation tests**

```python
# tests/baselines/test_fidelity.py
import pytest

from ratemem.baselines.fidelity import FidelityAuditError, audit_fidelity_report


def test_fidelity_report_binds_source_contract_case_and_raw_outputs(fidelity_fixture) -> None:
    report = fidelity_fixture.passing_report()
    audited = audit_fidelity_report(report, fidelity_fixture.policy, fidelity_fixture.source_inventory)
    assert audited.status == "faithful"
    assert audited.raw_output_sha256 == fidelity_fixture.raw_output_sha256


def test_sana_required_set_exactly_matches_catalog_primary_controls(fidelity_fixture) -> None:
    assert tuple(fidelity_fixture.policy.sana_primary_required_methods) == tuple(
        fidelity_fixture.catalog.primary_control_ids
    )


@pytest.mark.parametrize("field", ["source_revision", "backbone_revision", "case_sha256", "raw_output_sha256"])
def test_changed_fidelity_provenance_fails(field: str, fidelity_fixture) -> None:
    report = fidelity_fixture.passing_report().model_copy(update={field: "0" * (40 if field.endswith("revision") else 64)})
    with pytest.raises(FidelityAuditError):
        audit_fidelity_report(report, fidelity_fixture.policy, fidelity_fixture.source_inventory)
```

```python
# tests/baselines/test_search_budget.py
import pytest

from ratemem.baselines.fidelity import SearchBudgetError, audit_search_ledger


def test_search_is_validation_only_and_within_equal_maximum(search_fixture) -> None:
    audited = audit_search_ledger(search_fixture.valid_ledger(), search_fixture.policy)
    assert audited.trials <= 24
    assert audited.gpu_hours <= 48.0
    assert audited.seen_splits == {"validation"}


def test_failed_trial_consumes_budget(search_fixture) -> None:
    ledger = search_fixture.ledger_with_failed_trial()
    assert audit_search_ledger(ledger, search_fixture.policy).trials == len(ledger.rows)


def test_contextual_sdxl_search_ledger_is_rejected(search_fixture) -> None:
    ledger = search_fixture.valid_ledger(backbone_id="sdxl_1_0")
    with pytest.raises(SearchBudgetError, match="search backbone must be sana_1_5_1_6b"):
        audit_search_ledger(ledger, search_fixture.policy)


@pytest.mark.parametrize("violation", ["twenty_fifth_trial", "final_split", "gpu_hour_overrun"])
def test_search_violation_blocks_method(violation: str, search_fixture) -> None:
    with pytest.raises(SearchBudgetError):
        audit_search_ledger(search_fixture.invalid_ledger(violation), search_fixture.policy)
```

```python
# tests/contract/baselines/test_primary_backbone_gate.py
import pytest

from ratemem.baselines.fidelity import BaselineAuditBlocked, resolve_primary_backbone_plan


def test_all_required_sana_controls_faithful_locks_sana_only(fidelity_matrix) -> None:
    plan = resolve_primary_backbone_plan(fidelity_matrix.all_faithful())
    assert plan.primary_backbone == "sana_1_5_1_6b"
    assert plan.contextual_backbones == ("sdxl_1_0",)
    assert plan.contextual_promotion_allowed is False


def test_one_required_sana_fidelity_failure_blocks_primary_claim(fidelity_matrix) -> None:
    matrix = fidelity_matrix.with_status("dreamcache_feature_cache", "sana_1_5_1_6b", "incompatible")
    with pytest.raises(BaselineAuditBlocked, match="required_sana_control_unfaithful:dreamcache_feature_cache"):
        resolve_primary_backbone_plan(matrix)


def test_faithful_sdxl_evidence_cannot_satisfy_missing_sana_control(fidelity_matrix) -> None:
    matrix = fidelity_matrix.with_status("dreamcache_feature_cache", "sana_1_5_1_6b", "incompatible")
    matrix = matrix.with_status("dreamcache_feature_cache", "sdxl_1_0", "faithful")
    with pytest.raises(BaselineAuditBlocked, match="required_sana_control_unfaithful:dreamcache_feature_cache"):
        resolve_primary_backbone_plan(matrix)


def test_contextual_sdxl_cannot_enter_primary_table(fidelity_matrix) -> None:
    plan = resolve_primary_backbone_plan(fidelity_matrix.all_faithful())
    with pytest.raises(BaselineAuditBlocked, match="contextual_backbone_not_primary_eligible:sdxl_1_0"):
        plan.require_primary("sdxl_1_0")
```

```python
# tests/contract/baselines/test_required_evidence.py
def test_every_primary_comparator_has_real_factory_fidelity_ledger_and_causality_evidence(audit_fixture) -> None:
    receipt = audit_fixture.run()
    assert receipt.primary_method_ids == audit_fixture.catalog.primary_control_ids
    assert set(receipt.primary_method_ids).isdisjoint(receipt.contextual_method_ids)
    for method in receipt.primary_methods:
        assert method.backbone_id == "sana_1_5_1_6b"
        assert method.backbone_revision == audit_fixture.sana_revision
        assert method.factory_importable is True
        assert method.concrete_factory_sha256
        assert method.source_record_sha256
        assert method.fidelity_report_sha256
        assert method.state_roundtrip_report_sha256
        assert method.byte_budget_report_sha256
        assert method.causal_access_report_sha256
        assert method.synthetic_provider_contract_report_sha256
        assert not any(token in method.factory_qualname.lower() for token in ("fake", "stub", "fixture"))
```

- [ ] **Step 2: Run tests and verify fidelity auditing is absent**

Run: `uv run pytest tests/baselines/test_fidelity.py tests/baselines/test_search_budget.py tests/contract/baselines/test_primary_backbone_gate.py tests/contract/baselines/test_required_evidence.py -q`

Expected: collection fails importing `ratemem.baselines.fidelity`.

- [ ] **Step 3: Freeze numerical and structural fidelity cases**

```yaml
# configs/baselines/fidelity-policy.yaml
schema_version: "1.0"
reports_are_algorithmic_contracts_not_published_metric_reproductions: true
common_required_cases:
  - {id: canonical_state_roundtrip, comparator: sha256_exact}
  - {id: event_receipt_identity, comparator: sha256_exact}
  - {id: exact_byte_budget_random_trace, comparator: boolean_true}
  - {id: probe_no_mutation, comparator: sha256_exact}
  - {id: causal_future_denial, comparator: exception_type_exact}
method_cases:
  independent_fifo: [{id: locked_fifo_victim, comparator: value_exact}]
  independent_lru: [{id: locked_lru_victim, comparator: value_exact}]
  independent_lrua: [{id: locked_lrua_decay_and_victim, comparator: float64_allclose, atol: 1.0e-12, rtol: 1.0e-12}]
  private_progressive_size_aware: [{id: prefix_and_size_action, comparator: value_exact}]
  private_progressive_separable_rate: [{id: exact_small_knapsack, comparator: value_exact}]
  shared_packet_plain_greedy: [{id: locked_density_order, comparator: value_exact}]
  cts_style_static: [{id: paper_formula_matrix_fixture, comparator: float32_allclose, atol: 1.0e-6, rtol: 1.0e-6}]
  vb_lora_style_static: [{id: vector_bank_topk_fixture, comparator: float32_allclose, atol: 1.0e-6, rtol: 1.0e-6}]
  share_style_online:
    - {id: upstream_matrix_update_fixture, comparator: float32_allclose, atol: 1.0e-6, rtol: 1.0e-6}
    - {id: reprojection_drift_fixture, comparator: float32_allclose, atol: 1.0e-6, rtol: 1.0e-6}
  dreamcache_feature_cache:
    - {id: upstream_preprocessed_feature, comparator: float32_allclose, atol: 1.0e-5, rtol: 1.0e-5}
    - {id: pinned_backbone_one_step_latent, comparator: bf16_allclose, atol: 0.02, rtol: 0.02}
    - {id: feature_tap_and_injection_paths, comparator: value_exact}
  per_concept_lora:
    - {id: peft_dense_delta_equivalence, comparator: float32_allclose, atol: 1.0e-5, rtol: 1.0e-5}
    - {id: frozen_backbone_hash, comparator: sha256_exact}
  hyperlora_upstream: [{id: upstream_portrait_coefficient_fixture, comparator: float32_allclose, atol: 1.0e-5, rtol: 1.0e-5}]
sana_primary_required_methods:
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
allowed_statuses: [faithful, incompatible, failed]
```

Every case has a canonical fixture/input hash and raw-output artifact. A required SANA report may use `incompatible` only when it includes a signed technical record with failing case ID, observed shape/API mismatch, source and backbone revisions, and attempted bridge revision; that status still blocks its primary claim. It cannot use an image-quality result to declare incompatibility. `failed` means the implementation was expected to work but did not meet the prelocked numerical/structural case. Contextual citation-only work has no fidelity case and cannot enter this policy.

- [ ] **Step 4: Freeze equal validation-only search budgets and concrete spaces**

```yaml
# configs/baselines/policy-search.yaml
schema_version: "1.0"
split: validation
primary_backbone: sana_1_5_1_6b
contextual_backbones_are_not_search_candidates: true
maximum_trials_per_method: 24
maximum_gpu_hours_per_method: 48.0
failed_trials_consume_budget: true
selector:
  endpoint: request_weighted_identity
  prompt_constraint_source: evaluation_lock
  tie_break: [lower_online_bytes, lower_insert_latency, method_id]
spaces:
  independent_fifo: {controller: [fifo]}
  independent_lru: {controller: [lru]}
  independent_lrua: {decay: [0.9, 0.95, 0.99, 0.995]}
  private_progressive_size_aware: {base_bits: [2, 4], packet_bits: [2, 4, 8], outer_policy: [fifo, lru], maximum_prefix: [2, 4]}
  private_progressive_separable_rate: {base_bits: [2, 4], packet_bits: [2, 4, 8], maximum_prefix: [2, 4, 8], request_weight_window: [16]}
  shared_packet_plain_greedy: {base_bits: [2, 4], packet_bits: [2, 4, 8], maximum_incidences: [2, 4, 8], request_weight_window: [16]}
  cts_style_static: {rank: [4, 8, 16, 32], coefficient_bits: [4, 8, 16], outer_policy: [fifo, lru]}
  vb_lora_style_static: {bank_size: [64, 128, 256], top_k: [1, 2, 4, 8], weight_bits: [4, 8]}
  share_style_online: {rank: [4, 8, 16, 32], coefficient_bits: [4, 8, 16], outer_policy: [fifo, lru]}
  dreamcache_feature_cache: {cache_dtype: [float16, bfloat16], controller: [fifo, lru], injection_scale: [0.5, 1.0, 1.5]}
  per_concept_lora: {config_path: [configs/baselines/lora-reference.yaml]}
```

When a Cartesian space exceeds 24 cells, enumerate a deterministic Latin-hypercube index set seeded by the policy hash. Smaller spaces run each distinct cell once; unused allowance is recorded, not converted into extra private tuning. GPU hours come only from validated attempt artifacts. This policy authorizes primary-method search only on `sana_1_5_1_6b`; it rejects an SDXL search ledger. Contextual SDXL reproduction uses the pinned upstream configuration, carries no matched validation selection, and cannot consume or replace a SANA method's search allowance. A future reviewed SDXL RateMem extension must define a separate policy and budget rather than reusing this lock.

- [ ] **Step 5: Implement audit records and the fail-closed SANA primary gate**

`FidelityReport` binds method ID, implementation entry point, concrete factory source hash, source-record hash/revision/license, backbone ID/revision, case-policy hash, each measured comparison with raw-output hashes, state-roundtrip/byte/causality/synthetic-provider contract report hashes, environment lock hash, status, and optional signed incompatibility record. The pre-lock audit hashes `configs/baselines/policy-search.yaml` but consumes no search result. `SearchLedger` validation is implemented now for later scientific use: after both locks and phase authorization, it stores every attempted method ID, SANA backbone ID/revision, configuration, split, real shared-input lock hashes, start/end timestamps, GPU SKU, GPU hours, exit status, and artifact hash before selected-configuration freeze. A row whose backbone is not `sana_1_5_1_6b` fails before its score is read.

```python
# src/ratemem/baselines/fidelity.py
from typing import Literal

from pydantic import BaseModel, ConfigDict


class BackbonePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    primary_backbone: Literal["sana_1_5_1_6b"] = "sana_1_5_1_6b"
    contextual_backbones: tuple[Literal["sdxl_1_0"], ...] = ("sdxl_1_0",)
    contextual_promotion_allowed: Literal[False] = False

    def require_primary(self, backbone_id: str) -> Literal["sana_1_5_1_6b"]:
        if backbone_id != self.primary_backbone:
            raise BaselineAuditBlocked(f"contextual_backbone_not_primary_eligible:{backbone_id}")
        return self.primary_backbone


def resolve_primary_backbone_plan(matrix: FidelityMatrix) -> BackbonePlan:
    missing = sorted(matrix.unfaithful_required("sana_1_5_1_6b"))
    if missing:
        method_ids = ".".join(missing)
        raise BaselineAuditBlocked(f"required_sana_control_unfaithful:{method_ids}")
    return BackbonePlan()
```

`unfaithful_required` examines every method in `sana_primary_required_methods` and returns any absent, `incompatible`, or `failed` SANA receipt. It cannot drop a failed method or consult an SDXL receipt. For code-based controls, the pre-lock audit requires the same locked synthetic provider/schema and exact factory contract, not learned hashes. DreamCache and per-concept LoRA require their explicit SANA-native-scope structural receipts. Published or reproduced different-backbone numbers never satisfy a missing report. Real amortizer, basis, codec dictionary, and candidate-stream equality is enforced later by Task 15 and the scientific selected-configuration freeze.

- [ ] **Step 6: Generate schemas and run the complete synthetic audit**

Run:

```bash
uv run ratemem-baselines schema fidelity-report --output schemas/ratemem-baseline-fidelity-report-v1.schema.json
uv run ratemem-baselines schema audit-receipt --output schemas/ratemem-baseline-audit-receipt-v1.schema.json
uv run pytest tests/baselines/test_fidelity.py tests/baselines/test_search_budget.py tests/contract/baselines/test_primary_backbone_gate.py tests/contract/baselines/test_required_evidence.py -q
uv run ruff check src/ratemem/baselines/fidelity.py tests/baselines/test_fidelity.py tests/baselines/test_search_budget.py tests/contract/baselines/test_primary_backbone_gate.py tests/contract/baselines/test_required_evidence.py
uv run mypy src/ratemem/baselines/fidelity.py
```

Expected: all tests pass; complete required SANA evidence yields the fixed SANA plan, while any missing or unfaithful required SANA control blocks the primary claim even when an SDXL-native report is faithful.

- [ ] **Step 7: Commit fidelity and search policy**

```bash
git add configs/baselines/fidelity-policy.yaml configs/baselines/policy-search.yaml src/ratemem/baselines/fidelity.py schemas/ratemem-baseline-fidelity-report-v1.schema.json schemas/ratemem-baseline-audit-receipt-v1.schema.json tests/baselines/test_fidelity.py tests/baselines/test_search_budget.py tests/contract/baselines/test_primary_backbone_gate.py tests/contract/baselines/test_required_evidence.py
git commit -m "feat(baselines): lock fidelity and SANA primary gate"
```

### Task 15: Produce paired trace outputs for every eligible method

**Files:**
- Create: `src/ratemem/baselines/replay.py`
- Create: `src/ratemem/baselines/registry.py`
- Create: `schemas/ratemem-baseline-paired-replay-v1.schema.json`
- Test: `tests/baselines/test_registry.py`
- Test: `tests/baselines/test_paired_replay.py`
- Test: `tests/contract/baselines/test_no_future_access.py`

- [ ] **Step 1: Write failing factory, pairing, budget, and future-access tests**

```python
# tests/baselines/test_registry.py
from ratemem.baselines.catalog import REQUIRED_CONTROL_IDS
from ratemem.baselines.registry import build_registry


def test_every_control_has_a_concrete_factory_or_audited_external_manifest(registry_fixture) -> None:
    registry = build_registry(registry_fixture.lock)
    assert set(registry.method_ids) == REQUIRED_CONTROL_IDS
    for method_id in registry.method_ids:
        entry = registry[method_id]
        assert entry.factory_importable
        assert entry.factory_sha256 == registry_fixture.lock.by_id(method_id).concrete_factory_sha256
```

```python
# tests/baselines/test_paired_replay.py
from ratemem.baselines.replay import replay_paired


def test_all_methods_receive_identical_trace_prompt_support_and_noise(paired_fixture) -> None:
    result = replay_paired(paired_fixture.trace, paired_fixture.adapters, paired_fixture.contracts)
    assert len({artifact.input_commitment_sha256 for artifact in result.artifacts.values()}) == 1
    for artifacts in result.artifacts.values():
        assert [row.event_index for row in artifacts.receipts] == paired_fixture.operational_event_indices


def test_every_mutable_receipt_respects_exact_budget(paired_fixture) -> None:
    result = replay_paired(paired_fixture.trace, paired_fixture.adapters, paired_fixture.contracts)
    for artifact in result.artifacts.values():
        assert all(row.ledger.online_state_bytes <= artifact.byte_budget for row in artifact.receipts)


def test_code_methods_bind_the_learned_method_shared_inputs(paired_fixture) -> None:
    result = replay_paired(paired_fixture.trace, paired_fixture.adapters, paired_fixture.contracts)
    expected = paired_fixture.learned_provider_manifest.candidate_stream_sha256
    assert {
        artifact.candidate_stream_sha256
        for method_id, artifact in result.artifacts.items()
        if method_id in paired_fixture.code_method_ids
    } == {expected}
```

```python
# tests/contract/baselines/test_no_future_access.py
def test_causal_methods_are_invariant_to_changed_future_suffix(paired_fixture) -> None:
    first = paired_fixture.trace_with_suffix("uniform")
    second = paired_fixture.trace_with_suffix("burst_h9")
    for method_id in paired_fixture.causal_method_ids:
        prefix_a = paired_fixture.replay_prefix(method_id, first)
        prefix_b = paired_fixture.replay_prefix(method_id, second)
        assert prefix_a == prefix_b


def test_only_future_oracle_receives_full_trace(paired_fixture) -> None:
    access = paired_fixture.access_audit()
    assert access["exact_future_trace_packets"].maximum_visible_event == paired_fixture.trace.events[-1].event_index
    assert all(
        row.maximum_visible_event == row.current_event
        for method_id, row in access.items()
        if method_id != "exact_future_trace_packets"
    )
```

- [ ] **Step 2: Run tests and verify registry/replay modules are absent**

Run: `uv run pytest tests/baselines/test_registry.py tests/baselines/test_paired_replay.py tests/contract/baselines/test_no_future_access.py -q`

Expected: collection fails importing `ratemem.baselines.registry` or `ratemem.baselines.replay`.

- [ ] **Step 3: Implement the lock-derived registry**

`build_registry` reads only the sealed baseline lock. Native factory paths are imported, source-hashed, instantiated, and checked against `BaselineAdapter`. External entries load the exact command, checkout revision, environment lock, and deadline from their SANA fidelity report. Upper references use a separate factory. Contextual and incompatible literature has no runnable registry entry. A method cannot register from `tests`, a notebook, an interactive module, or a class/function whose source hash differs from the lock.

- [ ] **Step 4: Implement paired replay and immutable artifacts**

```python
def replay_paired(
    trace: Trace,
    adapters: dict[str, BaselineAdapter],
    contracts: dict[str, FrozenComparisonContract],
) -> PairedReplay:
    assert len({contract.prompt_pool_sha256 for contract in contracts.values()}) == 1
    assert len({contract.support_pool_sha256 for contract in contracts.values()}) == 1
    assert len({contract.noise_seed_manifest_sha256 for contract in contracts.values()}) == 1
    artifacts: dict[str, MethodReplayArtifact] = {}
    for method_id in sorted(adapters):
        adapter = adapters[method_id]
        adapter.initialize(contracts[method_id])
        artifacts[method_id] = replay_one(trace, adapter, contracts[method_id])
        adapter.close()
    commitment = paired_input_commitment(trace, contracts)
    if {artifact.input_commitment_sha256 for artifact in artifacts.values()} != {commitment}:
        raise PairedReplayError("method input commitments differ")
    return PairedReplay(input_commitment_sha256=commitment, artifacts=artifacts)
```

`replay_one` passes a `CausalEventView` at each operational event, recomputes the host ledger and cap, writes the receipt, then services probes from copied snapshots. Each method output directory contains `manifest.json`, `events.jsonl`, `probes.jsonl`, `ledgers.jsonl`, `access-audit.json`, stderr digest, and checksums. The canonical path is `artifacts/scientific/baselines/replay/{baseline_lock_id}/{method_id}/{training_seed}/{trace_id}/{budget_id}/{request_regime}/`. No published metric or hand-entered number is accepted as an event row.

At post-lock scientific replay, the RateMem learned-method manifest is an input even though this plan does not implement RateMem. It must contain the same backbone/layout/amortizer/basis/codec/candidate hashes and the exact RateMem factory hash. A mismatch blocks the entire paired cell rather than regenerating baseline inputs. The pre-lock baseline audit never consumes this manifest.

- [ ] **Step 5: Generate schema, run paired contracts, and commit**

Run:

```bash
uv run ratemem-baselines schema paired-replay --output schemas/ratemem-baseline-paired-replay-v1.schema.json
uv run pytest tests/baselines/test_registry.py tests/baselines/test_paired_replay.py tests/contract/baselines/test_no_future_access.py -q
uv run ruff check src/ratemem/baselines/registry.py src/ratemem/baselines/replay.py tests/baselines/test_registry.py tests/baselines/test_paired_replay.py tests/contract/baselines/test_no_future_access.py
uv run mypy src/ratemem/baselines/registry.py src/ratemem/baselines/replay.py
```

Expected: all tests pass; every causal prefix is invariant to a changed future; all code-based methods bind the RateMem candidate-stream hash.

```bash
git add src/ratemem/baselines/registry.py src/ratemem/baselines/replay.py schemas/ratemem-baseline-paired-replay-v1.schema.json tests/baselines/test_registry.py tests/baselines/test_paired_replay.py tests/contract/baselines/test_no_future_access.py
git commit -m "feat(baselines): add paired lifecycle replay"
```

### Task 16: Wire the fail-closed audit handoff, CI, and operator documentation

**Files:**
- Create: `src/ratemem/baselines/cli.py`
- Create: `docs/baselines.md`
- Modify: `docs/scientific-evaluation.md`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/contract/baselines/test_audit_cli.py`
- Test: `tests/integration/baselines/test_synthetic_baseline_pipeline.py`

- [ ] **Step 1: Write failing CLI and end-to-end synthetic tests**

```python
# tests/contract/baselines/test_audit_cli.py
from typer.testing import CliRunner

from ratemem.baselines.cli import app


def test_missing_fidelity_report_blocks_without_writing_lock(audit_cli_fixture) -> None:
    result = CliRunner().invoke(app, audit_cli_fixture.args_without_dreamcache_report)
    assert result.exit_code == 2
    assert result.stdout == "BLOCKED baseline-lock: missing_fidelity_report:dreamcache_feature_cache:sana_1_5_1_6b\n"
    assert not audit_cli_fixture.output_lock.exists()


def test_complete_synthetic_evidence_writes_lock_plan_and_receipt(audit_cli_fixture) -> None:
    result = CliRunner().invoke(app, audit_cli_fixture.complete_args)
    assert result.exit_code == 0
    assert result.stdout.startswith("PASS baseline-lock: backbone=sana_1_5_1_6b lock=")
    assert audit_cli_fixture.output_lock.exists()
    assert audit_cli_fixture.output_plan.exists()
    assert audit_cli_fixture.output_receipt.exists()


def test_unfaithful_sana_control_blocks_even_with_contextual_sdxl_evidence(audit_cli_fixture) -> None:
    result = CliRunner().invoke(app, audit_cli_fixture.args_with_unfaithful_sana_and_faithful_sdxl)
    assert result.exit_code == 2
    assert result.stdout == "BLOCKED baseline-lock: required_sana_control_unfaithful:dreamcache_feature_cache\n"
    assert not audit_cli_fixture.output_lock.exists()
    assert not audit_cli_fixture.output_plan.exists()
    assert not audit_cli_fixture.output_receipt.exists()
```

```python
# tests/integration/baselines/test_synthetic_baseline_pipeline.py
def test_synthetic_pipeline_inventories_a_local_repo_audits_all_methods_and_replays_pair(tmp_path) -> None:
    result = run_synthetic_baseline_pipeline(tmp_path)
    assert result.source_inventory_valid
    assert result.required_sana_methods == result.audited_sana_methods
    assert result.backbone_plan.primary_backbone == "sana_1_5_1_6b"
    assert result.backbone_plan.contextual_backbones == ("sdxl_1_0",)
    assert result.primary_replay_backbone_ids == {"sana_1_5_1_6b"}
    assert "sinelora_delta_aaai2026" in result.contextual_literature_keys
    assert len(set(result.paired_input_commitments)) == 1
    assert all(row.online_state_bytes <= row.byte_budget for row in result.event_rows)
    assert result.contains_scientific_metric_values is False
```

- [ ] **Step 2: Run tests and verify the CLI is not wired**

Run: `uv run pytest tests/contract/baselines/test_audit_cli.py tests/integration/baselines/test_synthetic_baseline_pipeline.py -q`

Expected: collection fails importing `ratemem.baselines.cli`.

- [ ] **Step 3: Implement the exact audit handoff command**

Mount Typer groups `schema`, `sources`, `backbones`, `shared-inputs`, `fidelity`, `search`, `audit`, and `replay`. All domain failures print one stable `BLOCKED` reason to standard output and exit 2. Unexpected exceptions remain nonzero exceptions and cannot produce a lock. `audit freeze` validates every required source, factory, SANA structural-fidelity, state-roundtrip, exact-ledger, causal-access, synthetic-provider/schema, and native-scope record; hashes the search policy without reading outcomes; enforces the fixed SANA primary gate; and atomically writes all three outputs only after the complete audit passes. SDXL-native and other non-SANA publications appear only in the contextual catalog/source attribution; they are excluded before required-evidence satisfaction and cannot affect the gate.

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

Expected on complete genuine pre-lock evidence: standard output matches `^PASS baseline-lock: backbone=sana_1_5_1_6b lock=[0-9a-f]{64} receipt=[0-9a-f]{64}$`. Any missing, unlicensed, structurally unfaithful, over-budget, future-reading, source-hash, search-policy, factory, synthetic-provider, non-SANA-primary, or contextual-promotion mismatch exits 2 with a line matching `^BLOCKED baseline-lock: [a-z0-9_:.-]+$` and writes none of the plan, receipt, or lock. The emitted lock schema fixes `primary_backbone=sana_1_5_1_6b`, marks `sdxl_1_0` contextual-only, and rejects learned checkpoint hashes, real candidate-stream hashes, tuned hyperparameters, validation metrics, search-ledger paths, and selected method IDs.

- [ ] **Step 4: Document the real evidence sequence without launching paid work**

`docs/baselines.md` must give these operator phases and exact paths:

1. resolve sources into `artifacts/scientific/baselines/source-inventory.json` and manually review every executable license;
2. lock the 120-projection SANA q/k/v layout and 480-dimensional code contract; record the pinned SDXL identity only in the contextual source inventory, without creating an SDXL RateMem layout;
3. generate `artifacts/scientific/baselines/compliance/synthetic-provider-contract.json` from the provider-neutral schema and locked synthetic SANA bundle;
4. run CPU algorithm/state/byte/causal fidelity cases with synthetic codes and packets;
5. emit required SANA real-checkpoint structural fidelity job specifications; only the scientific Task 8 `baseline_fidelity` permit may execute them, bound to dataset lock, catalog, fidelity policy, and held-in/calibration inputs with no validation/final trace, learned training, tuning, or claim metric access; it must verify one selected workspace's USD 28.00 outer cap, reserve against the aggregate USD 27.00 ceiling, reconcile immediately, and never rotate/fallback;
6. ingest required outputs into `fidelity/{method_id}/sana_1_5_1_6b/report.json` and `compliance/{method_id}/sana_1_5_1_6b/report.json`; keep SDXL-native publications as contextual catalog/bibliography evidence only, and do not create an executable fidelity receipt without a separately reviewed future RateMem-SDXL extension;
7. run `audit freeze`, review the fixed SANA backbone plan, and let scientific Task 8 seal the baseline and evaluation locks; no learned hash, real input receipt, search outcome, selection, or SDXL substitution enters either baseline audit output;
8. pass the learned-method CPU gate, then let scientific Task 9 issue full phase authorization;
9. train SANA RateMem and frozen dictionaries, then materialize the real SANA target-code/candidate stream exactly once through `SharedInputProvider`;
10. train/tune all primary SANA baselines on validation under `policy-search.yaml`, store append-only ledgers under `search/{method_id}/ledger.jsonl`, verify same-input receipts, and freeze selected configurations;
11. replay the final trace only after that configuration freeze through the scientific workflow.

State explicitly that an upstream paper number cannot fill an absent artifact, a style control is labeled `-style`, a different-backbone row is contextual, and failure of any required SANA port blocks the affected primary comparison and claim. A faithful SDXL-native implementation remains contextual and provides no fallback.

- [ ] **Step 5: Add synthetic-only CI and run the complete local quality gate**

Add one CI job using Python 3.11.13 and `uv sync --all-extras --frozen`. It runs only fixture repositories, fake backbones, synthetic codes, and the fake external worker; it never downloads SANA/SDXL, opens final data, accesses credentials, or launches paid compute.

Run:

```bash
uv run pytest tests/baselines tests/contract/baselines tests/integration/baselines -q
uv run ruff check src/ratemem/baselines external_baselines tests/baselines tests/contract/baselines tests/integration/baselines
uv run mypy src/ratemem/baselines
uv run ratemem-baselines --help
git diff --check
```

Expected: all tests pass; Ruff, mypy, and `git diff --check` exit 0; help lists every group from Step 3.

- [ ] **Step 6: Commit the audit handoff and documentation**

```bash
git add src/ratemem/baselines/cli.py docs/baselines.md docs/scientific-evaluation.md .github/workflows/ci.yml tests/contract/baselines/test_audit_cli.py tests/integration/baselines/test_synthetic_baseline_pipeline.py
git commit -m "science: complete matched-baseline audit handoff"
```

## Final acceptance checklist

- [ ] Every primary comparator has an importable SANA implementation or audited SANA-capable external worker, immutable source/revision/license evidence, a locked SANA fidelity report, a canonical state roundtrip, and exact per-event byte receipts.
- [ ] Independent FIFO/LRU/LRUA, private progressive size-aware/separable-rate, plain shared density greedy, CTS-style, VB-LoRA-style, online SHARE-style, DreamCache-style, stateless amortizer, per-concept LoRA, append-only, and future-trace controls all have distinct tests and factory IDs.
- [ ] All code-based representation controls bind the same fixed SANA backbone, 120-projection layout, 480-dimensional code, amortizer, adapter basis, codec dictionary, and candidate stream as RateMem; explicit feature-native, optimization-native, and upper-reference exceptions are labeled and tested.
- [ ] Every causal method is prefix-invariant to a changed future trace; only `exact_future_trace_packets` receives future events and it cannot enter a causal comparison.
- [ ] Host-recomputed canonical bytes stay under the locked cap after every mutable event and include every listed state component; shared trained and external support bytes are separately disclosed.
- [ ] Any absent, incompatible, or failed required SANA fidelity case blocks the affected primary comparison and claim; no SDXL report, published number, or reproduction can satisfy it.
- [ ] Search uses validation only on SANA, at most 24 trials and 48 GPU-hours per primary method, rejects SDXL ledgers, and counts failed attempts.
- [ ] Paired artifacts have identical trace, support, prompt, and noise commitments and contain no published or hand-entered result values.
- [ ] Contextual and incompatible literature cannot be promoted into the primary table by registry edits after result inspection; SDXL remains contextual until a separately reviewed future RateMem-SDXL extension defines new contracts and locks.
- [ ] The baseline lock is created only by the complete audit command, fixes `sana_1_5_1_6b` as the sole primary backbone, and is ready for the scientific-evaluation plan's evaluation-lock seal.
