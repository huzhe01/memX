from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ratemem.evaluation.canonical import canonical_json_bytes, write_yaml_atomic
from ratemem.evaluation.dataset_lock import (
    load_inventory,
    seal_dataset_lock,
    write_dataset_lock_and_card,
)
from ratemem.evaluation.evaluation_lock import (
    AllocatorBoundaryLock,
    ApprovalLock,
    BootstrapLock,
    ClaimLock,
    EvaluationLockDraft,
    EvaluationLockError,
    EvaluatorLock,
    GenerationLock,
    HumanStudyLock,
    LatencyLock,
    MarginLock,
    PowerLock,
    WorkloadLock,
    derive_budget_cells,
    freeze_evaluation_lock,
    require_scientific_training_lock,
)


def _claim_locks() -> dict[str, ClaimLock]:
    return {
        "shared_packet_representation": ClaimLock(
            primary_endpoint="request_weighted_identity",
            inference_unit="deployment_episode",
            required_controls=[
                "private_progressive_size_aware",
                "private_progressive_separable_rate",
                "cts_style_static",
                "vb_lora_style_static",
                "share_style_online",
                "dreamcache_feature_cache",
            ],
            constraint_metric="prompt",
            pass_rule="positive_paired_ci_and_prompt_noninferiority",
        ),
        "causal_packet_allocator": ClaimLock(
            primary_endpoint="request_weighted_utility",
            inference_unit="deployment_episode",
            required_controls=[
                "independent_lrua",
                "private_progressive_size_aware",
                "private_progressive_separable_rate",
                "shared_packet_plain_greedy",
            ],
            constraint_metric="average_active_quality",
            pass_rule=(
                "positive_paired_ci_lower_regret_and_quality_noninferiority"
            ),
        ),
        "allocator_guarantee": ClaimLock(
            primary_endpoint="certified_reduced_set_approximation_ratio",
            inference_unit="allocator_instance",
            required_controls=["exact_reduced_set_optimum"],
            pass_rule="proof_and_exhaustive_reduced_set_instance_certificate",
            ground_set_scope="causal_singleton_density_prescreen_C_t_max24",
            allocator_boundary=AllocatorBoundaryLock(
                fixture_id="four_concepts_eight_packets_each_v1",
                proposal_count=32,
                prescreen_input_count=32,
                allocator_input_count=24,
                deterministic_tie_break=(
                    "lexicographically_larger_packet_id_wins"
                ),
            ),
        ),
        "optimization_free_tradeoff": ClaimLock(
            primary_endpoint="identity",
            inference_unit="concept",
            required_controls=["per_concept_lora", "dreamcache_feature_cache"],
            pass_rule="quality_noninferiority_and_insertion_latency_advantage",
        ),
        "autonomous_lookup": ClaimLock(
            primary_endpoint="lookup_aurc",
            inference_unit="concept_conditioned_lookup_episode",
            required_controls=["nearest_key_threshold", "learned_novelty"],
            pass_rule="lower_aurc",
        ),
    }


def valid_draft() -> EvaluationLockDraft:
    return EvaluationLockDraft(
        schema_version="1.0",
        policy_sha256="0" * 64,
        dataset_lock_sha256="1" * 64,
        baseline_lock_sha256="2" * 64,
        baseline_audit_receipt_sha256="3" * 64,
        comparator_catalog_sha256="4" * 64,
        shared_input_schema_sha256="5" * 64,
        synthetic_provider_report_sha256="6" * 64,
        search_policy_sha256="7" * 64,
        trace_manifest_sha256={
            "train": "8" * 64,
            "validation": "9" * 64,
            "final_test": "a" * 64,
        },
        evaluators=[
            EvaluatorLock(
                evaluator_id="training_identity_encoder",
                repository="https://example.org/training-encoder",
                revision="1" * 40,
                weights_sha256="b" * 64,
                preprocessing_id="training_identity_preprocess_v1",
                preprocessing_sha256="c" * 64,
                roles={"training_loss"},
            ),
            EvaluatorLock(
                evaluator_id="independent_identity_encoder",
                repository="https://example.org/identity-encoder",
                revision="2" * 40,
                weights_sha256="d" * 64,
                preprocessing_id="headline_identity_preprocess_v1",
                preprocessing_sha256="e" * 64,
                roles={"headline_identity"},
            ),
            EvaluatorLock(
                evaluator_id="independent_prompt_encoder",
                repository="https://example.org/prompt-encoder",
                revision="3" * 40,
                weights_sha256="f" * 64,
                preprocessing_id="headline_prompt_preprocess_v1",
                preprocessing_sha256="1" * 64,
                roles={"headline_prompt"},
            ),
            EvaluatorLock(
                evaluator_id="independent_diversity_encoder",
                repository="https://example.org/diversity-encoder",
                revision="sha256:" + "2" * 64,
                weights_sha256="3" * 64,
                preprocessing_id="diversity_preprocess_v1",
                preprocessing_sha256="4" * 64,
                roles={"diversity"},
            ),
        ],
        metric_formulas={
            "identity": "identity_mean_v1",
            "prompt": "prompt_mean_v1",
            "request_weighted_identity": "request_weighted_identity_v1",
            "request_weighted_utility": "equal_weight_identity_prompt_v1",
            "retention_auc": "normalized_event_trapezoid_v1",
            "active_state_drift": "acquisition_delta_v1",
            "maximum_active_degradation": "maximum_acquisition_drop_v1",
            "oracle_regret": "future_oracle_utility_gap_v1",
            "lookup_aurc": "lookup_risk_coverage_v1",
            "diversity": "thresholded_conditional_diversity_v1",
        },
        margins=[
            MarginLock(
                claim_id="shared_packet_representation",
                metric_id="prompt",
                value=0.01,
                direction="higher",
                source_kind="separate_calibration",
                source_reference="calibration_prompt_noninferiority_v1",
                calibration_pool_sha256="5" * 64,
                calibration_artifact_sha256="6" * 64,
                used_for_model_selection=False,
            )
        ],
        budget_cells=list(
            derive_budget_cells(
                reference_total_bytes=20_480,
                active_set_size=20,
                fractions=(0.25, 0.50, 0.75),
                ledger_sha256="7" * 64,
            )
        ),
        workload_distributions=[
            WorkloadLock(
                workload_id="uniform_v1",
                request_regime="uniform",
                exponent=None,
                update_delete_rate_multiplier=1.0,
            ),
            WorkloadLock(
                workload_id="zipf_1_2_v1",
                request_regime="zipf",
                exponent=1.2,
                update_delete_rate_multiplier=1.0,
            ),
        ],
        generation=GenerationLock(
            backbone_id="sana_1_5_1_6b",
            resolution=1024,
            sampler_id="flow_dpm_solver_v1",
            steps=20,
            guidance_scale=4.5,
            prompt_seed_pairing="strict",
            noise_seed_pairing="strict",
            generation_seeds=[101, 202, 303],
        ),
        latency=LatencyLock(
            hardware_id="ppu_zw810e_locked_queue",
            device_count=1,
            warmup_requests=5,
            measured_requests=30,
            batch_size=1,
            resolution=1024,
            sampler_id="flow_dpm_solver_v1",
            steps=20,
        ),
        human_study=HumanStudyLock(
            enabled=True,
            blinded=True,
            randomized_side=True,
            minimum_raters=3,
            attention_checks=True,
        ),
        claims=_claim_locks(),
        bootstrap=BootstrapLock(
            alpha=0.05,
            confidence_level=0.95,
            resamples=10_000,
            seed=271_828,
            multiplicity_method="holm",
            minimum_training_seeds=3,
            prompt_noise_pairing="strict",
            strongest_control_selector="validation_primary_endpoint_v1",
        ),
        power=PowerLock(
            required_units_record_sha256="8" * 64,
            target_power=0.80,
            two_sided_alpha=0.05,
            minimum_detectable_effect=0.03,
            required_deployment_episodes=24,
        ),
        required_postlock_receipts=[
            "ratemem_shared_input_bundle",
            "method_train_receipts",
            "method_search_ledgers",
            "search_budget_compliance",
        ],
        approvals=[
            ApprovalLock(
                approval_id="protocol_owner_approval_v1",
                approver_id="protocol_owner",
                role="protocol_owner",
                approved_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
                approved_policy_sha256="0" * 64,
                signed_record_sha256="9" * 64,
            ),
            ApprovalLock(
                approval_id="independent_review_approval_v1",
                approver_id="independent_reviewer",
                role="independent_reviewer",
                approved_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
                approved_policy_sha256="0" * 64,
                signed_record_sha256="a" * 64,
            ),
        ],
    )


def test_budget_bytes_are_derived_from_locked_independent_cache_ledger() -> None:
    budgets = derive_budget_cells(
        reference_total_bytes=20_480,
        active_set_size=20,
        fractions=(0.25, 0.5, 0.75),
        ledger_sha256="a" * 64,
    )
    assert [(cell.label, cell.bytes) for cell in budgets] == [
        ("25pct", 5_120),
        ("50pct", 10_240),
        ("75pct", 15_360),
    ]
    assert all(cell.independent_cache_ledger_sha256 == "a" * 64 for cell in budgets)


def test_freeze_rejects_mutable_evaluator_revision_and_uncalibrated_margin() -> None:
    draft = valid_draft()
    draft.evaluators[0] = draft.evaluators[0].model_copy(update={"revision": "main"})
    draft.margins[0] = draft.margins[0].model_copy(
        update={"calibration_artifact_sha256": None}
    )
    with pytest.raises(EvaluationLockError):
        freeze_evaluation_lock(draft)


def test_training_identity_representation_cannot_be_sole_headline_evaluator() -> None:
    draft = valid_draft()
    draft.evaluators = [
        EvaluatorLock(
            evaluator_id="training_identity_encoder",
            repository="https://example.org/training-encoder",
            revision="1" * 40,
            weights_sha256="b" * 64,
            preprocessing_id="training_identity_preprocess_v1",
            preprocessing_sha256="c" * 64,
            roles={"training_loss", "headline_identity"},
        )
    ]
    with pytest.raises(EvaluationLockError, match="independent headline identity evaluator"):
        freeze_evaluation_lock(draft)


@pytest.mark.parametrize("scope", [None, "full_G_t"])
def test_allocator_guarantee_lock_requires_reduced_ground_set_scope(
    scope: str | None,
) -> None:
    draft = valid_draft()
    claim = draft.claims["allocator_guarantee"]
    draft.claims["allocator_guarantee"] = claim.model_copy(
        update={"ground_set_scope": scope}
    )
    with pytest.raises(
        EvaluationLockError,
        match="allocator guarantee ground-set scope",
    ):
        freeze_evaluation_lock(draft)


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
    field: str,
    value: object,
) -> None:
    draft = valid_draft()
    claim = draft.claims["allocator_guarantee"]
    assert claim.allocator_boundary is not None
    changed_boundary = claim.allocator_boundary.model_copy(update={field: value})
    draft.claims["allocator_guarantee"] = claim.model_copy(
        update={"allocator_boundary": changed_boundary}
    )

    with pytest.raises(EvaluationLockError, match="allocator boundary"):
        freeze_evaluation_lock(draft)


def test_allocator_guarantee_lock_rejects_missing_controller_boundary() -> None:
    draft = valid_draft()
    claim = draft.claims["allocator_guarantee"]
    draft.claims["allocator_guarantee"] = claim.model_copy(
        update={"allocator_boundary": None}
    )

    with pytest.raises(EvaluationLockError, match="allocator boundary"):
        freeze_evaluation_lock(draft)


def test_freeze_rejects_allocator_boundary_on_other_claims() -> None:
    draft = valid_draft()
    boundary = draft.claims["allocator_guarantee"].allocator_boundary
    claim = draft.claims["autonomous_lookup"]
    draft.claims["autonomous_lookup"] = claim.model_copy(
        update={"allocator_boundary": boundary}
    )
    with pytest.raises(EvaluationLockError, match="only allocator_guarantee"):
        freeze_evaluation_lock(draft)


def test_lock_id_binds_every_claim_relevant_semantic() -> None:
    lock = freeze_evaluation_lock(
        valid_draft(),
        sealed_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
    )
    original = lock.model_dump(mode="json")
    mutations = []
    margin = {**original, "margins": [dict(original["margins"][0])]}
    margin["margins"][0]["value"] = 0.02
    mutations.append(margin)
    evaluator = {**original, "evaluators": [dict(row) for row in original["evaluators"]]}
    evaluator["evaluators"][1]["preprocessing_sha256"] = "b" * 64
    mutations.append(evaluator)
    trace = {**original, "trace_manifest_sha256": dict(original["trace_manifest_sha256"])}
    trace["trace_manifest_sha256"]["validation"] = "c" * 64
    mutations.append(trace)
    budget = {**original, "budget_cells": [dict(row) for row in original["budget_cells"]]}
    budget["budget_cells"][0]["bytes"] += 1
    mutations.append(budget)
    workload = {
        **original,
        "workload_distributions": [dict(row) for row in original["workload_distributions"]],
    }
    workload["workload_distributions"][1]["exponent"] = 1.3
    mutations.append(workload)
    generation = {**original, "generation": dict(original["generation"])}
    generation["generation"]["generation_seeds"] = [101, 202, 304]
    mutations.append(generation)

    for payload in mutations:
        payload.pop("lock_id")
        payload.pop("sealed_at_utc")
        assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != lock.lock_id


def test_scientific_training_requires_matching_untampered_locks(
    tmp_path: Path,
) -> None:
    root = tmp_path
    dataset = seal_dataset_lock(
        load_inventory(Path("tests/fixtures/scientific/source-inventory.json")),
        policy_path=Path("configs/scientific/dataset-policy.yaml"),
        mode="synthetic",
    )
    dataset_path = root / "dataset-lock.yaml"
    write_dataset_lock_and_card(dataset, dataset_path, root / "data-card.md")
    draft = valid_draft()
    draft.dataset_lock_sha256 = dataset.lock_id
    lock = freeze_evaluation_lock(
        draft,
        sealed_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
    )
    evaluation_path = root / "evaluation-lock.yaml"
    write_yaml_atomic(evaluation_path, lock.model_dump(mode="json"))

    require_scientific_training_lock(dataset_path, evaluation_path, "train")
    with pytest.raises(EvaluationLockError, match="only the train split"):
        require_scientific_training_lock(dataset_path, evaluation_path, "validation")

    tampered = lock.model_dump(mode="json")
    tampered["metric_formulas"]["identity"] = "changed_after_freeze"
    write_yaml_atomic(evaluation_path, tampered)
    with pytest.raises(EvaluationLockError, match="content hash changed"):
        require_scientific_training_lock(dataset_path, evaluation_path, "train")
