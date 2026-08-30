from __future__ import annotations

from pathlib import Path

from ratemem.evaluation.traces import (
    AllPools,
    TracePolicy,
    build_trace,
    build_trace_set,
)

POOLS = Path("tests/fixtures/scientific/concept-pools.json")
POLICY = Path("configs/scientific/trace-policy.yaml")


def test_trace_is_deterministic_and_read_probe_semantics_differ() -> None:
    pools = AllPools.load(POOLS).for_split("validation")
    policy = TracePolicy.load(POLICY)
    first = build_trace(
        split="validation",
        trace_index=3,
        pools=pools,
        policy=policy,
        event_count=40,
    )
    second = build_trace(
        split="validation",
        trace_index=3,
        pools=pools,
        policy=policy,
        event_count=40,
    )

    assert first.model_dump_json() == second.model_dump_json()
    assert all(event.update_usage for event in first.events if event.kind == "read")
    assert all(not event.update_usage for event in first.events if event.kind == "probe")


def test_train_validation_final_commitments_are_pairwise_disjoint() -> None:
    trace_sets = build_trace_set(
        AllPools.load(POOLS),
        TracePolicy.load(POLICY),
        counts={"train": 4, "validation": 3, "final_test": 3},
        event_count=40,
    )

    concepts = [set(value.concept_ids) for value in trace_sets.values()]
    trace_ids = [set(value.trace_ids) for value in trace_sets.values()]
    seeds = [set(value.generation_seeds) for value in trace_sets.values()]
    for collections in (concepts, trace_ids, seeds):
        for index, left in enumerate(collections):
            for right in collections[index + 1 :]:
                assert left.isdisjoint(right)


def test_update_is_labeled_and_delete_never_reuses_handle() -> None:
    trace = build_trace(
        split="validation",
        trace_index=1,
        pools=AllPools.load(POOLS).for_split("validation"),
        policy=TracePolicy.load(POLICY),
        event_count=100,
    )

    assert all(event.handle for event in trace.events if event.kind in {"update", "delete"})
    deleted = {
        event.handle for event in trace.events if event.kind == "delete"
    }
    created = {
        event.handle for event in trace.events if event.kind == "create"
    }
    assert len(created) == len(
        [event for event in trace.events if event.kind == "create"]
    )
    assert deleted <= created
