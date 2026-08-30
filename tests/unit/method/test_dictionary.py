from __future__ import annotations

import torch

from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary


def make_dictionary() -> GroupRVQDictionary:
    model = GroupRVQDictionary(group_count=2, group_size=4, stages=2, entries=3)
    with torch.no_grad():
        model.codebooks.copy_(
            torch.tensor(
                [
                    [
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
                    ],
                    [
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
                    ],
                ]
            )
        )
        model.normalize_codebooks_()
    return model


def test_hard_assignment_is_deterministic_and_residual_is_additive() -> None:
    model = make_dictionary()
    residual = torch.tensor(
        [[[2.0, 0.1, 0.0, 0.0], [0.0, 1.5, 1.4, 0.0]]]
    )
    first = model.hard_assign(residual)
    second = model.hard_assign(residual.clone())
    torch.testing.assert_close(first.reconstruction + first.residual, residual)
    assert torch.equal(first.indices, second.indices)
    torch.testing.assert_close(first.gains, second.gains)


def test_soft_and_straight_through_assignments_reach_gradients() -> None:
    for straight_through in (False, True):
        model = make_dictionary()
        residual = torch.randn(3, 2, 4, requires_grad=True)
        result = model.soft_assign(
            residual, temperature=0.5, straight_through=straight_through
        )
        result.reconstruction.square().mean().backward()
        assert residual.grad is not None
        assert model.codebooks.grad is not None


def test_freeze_normalizes_and_detaches_dictionary() -> None:
    model = make_dictionary()
    frozen = freeze_dictionary(model)
    torch.testing.assert_close(
        torch.linalg.vector_norm(frozen.codebooks, dim=-1),
        torch.ones(2, 2, 3),
    )
    assert not frozen.codebooks.requires_grad
    with torch.no_grad():
        model.codebooks[0, 0, 0, 1].add_(0.5)
    assert freeze_dictionary(model).revision_sha256 != frozen.revision_sha256
