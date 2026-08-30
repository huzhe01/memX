from __future__ import annotations

from ratemem.baselines.protocol import BaselineAdapter, CausalEventView
from ratemem.method.adapter import RateMemAdapter
from tests.unit.method.test_adapter import (
    adapter_factory,
    comparison_contract,
    lifecycle_events,
)


def _execute_as_canonical(
    adapter: BaselineAdapter,
) -> tuple[str, bytes]:
    contract = comparison_contract()
    events = lifecycle_events()[:3]
    adapter.initialize(contract)
    for index, event in enumerate(events):
        adapter.apply_event(event, CausalEventView(events, current_index=index))
    snapshot = adapter.copy_snapshot()
    payload = adapter.export_online_state()
    assert adapter.state_ledger().online_state_sha256 == snapshot.state_sha256
    return snapshot.opaque_snapshot_token, payload


def test_concrete_ratemem_instantiates_and_roundtrips_as_canonical_adapter() -> None:
    adapter: RateMemAdapter = adapter_factory()
    token, payload = _execute_as_canonical(adapter)
    assert token.startswith("ratemem-snapshot-")
    restored: RateMemAdapter = adapter_factory()
    restored.initialize(comparison_contract())
    restored.import_online_state(payload)
    assert restored.export_online_state() == payload
