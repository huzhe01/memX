from __future__ import annotations

import torch

from ratemem.method.dictionary import (
    GroupRVQDictionary,
    decode_packet_key,
    freeze_dictionary,
)


def test_same_dictionary_entry_has_one_exact_packet_for_many_concepts() -> None:
    torch.manual_seed(7)
    frozen = freeze_dictionary(
        GroupRVQDictionary(group_count=2, group_size=4, stages=2, entries=3)
    )
    first = frozen.packet(group=0, stage=0, entry=1)
    second = frozen.packet(group=0, stage=0, entry=1)
    other = frozen.packet(group=0, stage=0, entry=2)
    assert first.packet_id == second.packet_id
    assert first.payload == second.payload
    assert first.packet_id != other.packet_id
    assert decode_packet_key(first.payload) == (frozen.revision_sha256, 0, 0, 1)
    assert frozen.validate_packet(first) == (0, 0, 1)


def test_packet_cannot_cross_dictionary_revisions() -> None:
    torch.manual_seed(11)
    source = GroupRVQDictionary(group_count=1, group_size=4, stages=1, entries=2)
    first = freeze_dictionary(source)
    packet = first.packet(0, 0, 0)
    with torch.no_grad():
        source.codebooks[0, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    second = freeze_dictionary(source)
    assert first.revision_sha256 != second.revision_sha256
    try:
        second.validate_packet(packet)
    except ValueError as error:
        assert "another frozen dictionary" in str(error)
    else:
        raise AssertionError("cross-revision packet was accepted")
