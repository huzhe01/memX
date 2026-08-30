from __future__ import annotations

import torch

from ratemem.training.functional_state import FunctionalMemoryState


def test_updates_are_out_of_place_and_boundary_detach_cuts_history() -> None:
    code = torch.randn(4, requires_grad=True)
    empty = FunctionalMemoryState()
    updated = empty.upsert("a", code, event_index=1)
    detached = updated.detach_boundary()
    assert "a" not in empty.codes
    assert updated.codes["a"] is code
    assert detached.codes["a"].grad_fn is None
    assert not detached.codes["a"].requires_grad


def test_update_requires_strictly_increasing_event_index() -> None:
    state = FunctionalMemoryState().upsert("a", torch.zeros(2), event_index=2)
    try:
        state.upsert("a", torch.ones(2), event_index=2)
    except ValueError as error:
        assert "forward" in str(error)
    else:
        raise AssertionError("same-event functional update was accepted")
