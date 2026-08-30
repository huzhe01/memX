from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from ratemem.runtime.device import DeviceRuntime
from ratemem.runtime.distributed import (
    DistributedContext,
    RankEnvironment,
    all_reduce_gradients,
    distributed_session,
)


def distributed_values(**overrides: str) -> Mapping[str, str]:
    values = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "4",
    }
    values.update(overrides)
    return values


def cpu_runtime() -> DeviceRuntime:
    return DeviceRuntime(
        kind="cpu",
        device=torch.device("cpu"),
        distributed_backend="gloo",
        device_count=0,
        device_names=(),
        bf16_supported=False,
    )


def test_absent_rank_environment_is_single_process() -> None:
    ranks = RankEnvironment.from_mapping({}, visible_devices=0)

    assert ranks == RankEnvironment(rank=0, local_rank=0, world_size=1, local_world_size=1)
    assert ranks.node_count == 1


def test_partial_rank_environment_is_rejected() -> None:
    with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE"):
        RankEnvironment.from_mapping(
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "8"},
            visible_devices=8,
        )


@pytest.mark.parametrize("value", ["", "-1", "+1", "01", " 1", "1 ", "1.0"])
def test_rank_fields_require_canonical_decimal_integers(value: str) -> None:
    with pytest.raises(ValueError, match="canonical nonnegative decimal"):
        RankEnvironment.from_mapping(
            distributed_values(RANK=value),
            visible_devices=4,
        )


def test_world_size_must_be_divisible_by_local_world_size() -> None:
    with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE must divide WORLD_SIZE"):
        RankEnvironment.from_mapping(
            distributed_values(WORLD_SIZE="8", LOCAL_WORLD_SIZE="3"),
            visible_devices=4,
        )


def test_local_world_size_cannot_exceed_visible_devices() -> None:
    with pytest.raises(ValueError, match="visible device count"):
        RankEnvironment.from_mapping(distributed_values(), visible_devices=2)


def test_rank_and_local_rank_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="RANK is outside WORLD_SIZE"):
        RankEnvironment.from_mapping(
            distributed_values(RANK="8"),
            visible_devices=4,
        )
    with pytest.raises(ValueError, match="LOCAL_RANK is outside LOCAL_WORLD_SIZE"):
        RankEnvironment.from_mapping(
            distributed_values(LOCAL_RANK="4"),
            visible_devices=4,
        )


def test_single_process_cpu_session_does_not_create_process_group() -> None:
    assert not torch.distributed.is_initialized()
    ranks = RankEnvironment.from_mapping({}, visible_devices=0)

    with distributed_session(cpu_runtime(), ranks) as context:
        assert context.is_primary is True
        assert context.device == torch.device("cpu")
        assert not torch.distributed.is_initialized()

    assert not torch.distributed.is_initialized()


def test_rank_manifest_is_canonical() -> None:
    ranks = RankEnvironment.from_mapping(
        distributed_values(RANK="5", LOCAL_RANK="1"), visible_devices=4
    )

    assert ranks.as_manifest() == {
        "rank": 5,
        "local_rank": 1,
        "world_size": 8,
        "local_world_size": 4,
        "node_count": 2,
    }


def test_single_rank_gradient_reduction_validates_without_mutating() -> None:
    parameter = nn.Parameter(torch.tensor([2.0]))
    parameter.grad = torch.tensor([3.0])
    context = DistributedContext(
        runtime=cpu_runtime(),
        ranks=RankEnvironment.from_mapping({}, visible_devices=0),
    )

    all_reduce_gradients((parameter,), context)

    assert torch.equal(parameter.grad, torch.tensor([3.0]))
