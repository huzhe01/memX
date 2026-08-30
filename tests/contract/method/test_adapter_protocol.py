from __future__ import annotations

from ratemem.baselines.protocol import BaselineAdapter, CausalEventView, ExactByteLedger
from tests.unit.method.test_adapter import (
    adapter_factory,
    comparison_contract,
    lifecycle_events,
)


def _require_canonical_adapter(value: BaselineAdapter) -> BaselineAdapter:
    return value


def test_ratemem_is_the_canonical_causal_adapter() -> None:
    adapter = adapter_factory()
    typed = _require_canonical_adapter(adapter)
    assert isinstance(typed, BaselineAdapter)
    assert typed.method_id == "ratemem_v1"
    assert typed.role == "causal"


def test_export_import_restores_identical_future_receipt() -> None:
    contract = comparison_contract()
    create_event, read_event, *_ = lifecycle_events()
    original = adapter_factory()
    original.initialize(contract)
    original.apply_event(create_event, CausalEventView((create_event,), 0))
    payload = original.export_online_state()
    ledger = original.state_ledger()
    assert isinstance(ledger, ExactByteLedger)
    assert ledger.online_state_bytes == len(payload)
    assert ledger.online_state_sha256 == original.copy_snapshot().state_sha256

    restored = adapter_factory()
    restored.initialize(contract)
    restored.import_online_state(payload)
    view = CausalEventView((create_event, read_event), current_index=1)
    assert restored.apply_event(read_event, view) == original.apply_event(
        read_event,
        view,
    )
