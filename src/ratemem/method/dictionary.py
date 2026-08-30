"""Trainable grouped RVQ directions and immutable content-addressed packet keys."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ratemem.state.model import Packet
from ratemem.state.serialization import packet_from_payload

_PACKET_MAGIC = b"RTPKT001"
_PACKET_HEADER = struct.Struct("<8s32sHHH")


@dataclass(frozen=True, slots=True)
class RVQAssignment:
    indices: Tensor
    gains: Tensor
    probabilities: Tensor
    reconstruction: Tensor
    residual: Tensor


class GroupRVQDictionary(nn.Module):
    """One direction bank per code group and residual stage."""

    def __init__(
        self,
        group_count: int,
        group_size: int,
        stages: int,
        entries: int,
    ) -> None:
        super().__init__()
        values = (group_count, group_size, stages, entries)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("dictionary dimensions must be positive integers")
        if group_count > 0xFFFF or stages > 0xFFFF or entries > 0xFFFF:
            raise ValueError("packet-addressed dictionary dimensions must fit uint16")
        self.group_count = group_count
        self.group_size = group_size
        self.stages = stages
        self.entries = entries
        self.codebooks = nn.Parameter(
            torch.randn(group_count, stages, entries, group_size, dtype=torch.float32)
        )
        self.normalize_codebooks_()

    @torch.no_grad()
    def normalize_codebooks_(self) -> None:
        normalized = F.normalize(self.codebooks.float(), dim=-1, eps=1e-8)
        self.codebooks.copy_(normalized.to(dtype=self.codebooks.dtype))

    def _validate(self, residual: Tensor) -> None:
        if type(residual) is not Tensor:
            raise TypeError("residual must be an exact torch.Tensor")
        if residual.ndim != 3 or residual.shape[1:] != (
            self.group_count,
            self.group_size,
        ):
            raise ValueError("residual must have shape [batch, group_count, group_size]")
        if not residual.is_floating_point():
            raise TypeError("residual must have a floating dtype")
        if not bool(torch.isfinite(residual).all().item()):
            raise ValueError("residual must be finite")

    def _assign(
        self,
        residual: Tensor,
        temperature: float,
        straight_through: bool,
    ) -> RVQAssignment:
        self._validate(residual)
        if not isinstance(temperature, int | float) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if type(straight_through) is not bool:
            raise TypeError("straight_through must be an exact bool")
        remaining = residual.float()
        reconstruction = torch.zeros_like(remaining)
        index_rows: list[Tensor] = []
        gain_rows: list[Tensor] = []
        probability_rows: list[Tensor] = []
        for stage in range(self.stages):
            directions = F.normalize(
                self.codebooks[:, stage].float(), dim=-1, eps=1e-8
            )
            correlations = torch.einsum("bgd,ged->bge", remaining, directions)
            squared_error = remaining.square().sum(-1, keepdim=True) - correlations.square()
            soft = torch.softmax(-squared_error / float(temperature), dim=-1)
            indices = squared_error.argmin(dim=-1)
            hard = F.one_hot(indices, self.entries).to(dtype=soft.dtype)
            probabilities = hard + soft - soft.detach() if straight_through else soft
            selected = torch.einsum("bge,ged->bgd", probabilities, directions)
            gains = (remaining * selected).sum(dim=-1)
            contribution = gains.unsqueeze(-1) * selected
            remaining = remaining - contribution
            reconstruction = reconstruction + contribution
            index_rows.append(indices)
            gain_rows.append(gains)
            probability_rows.append(probabilities)
        return RVQAssignment(
            indices=torch.stack(index_rows, dim=2),
            gains=torch.stack(gain_rows, dim=2),
            probabilities=torch.stack(probability_rows, dim=2),
            reconstruction=reconstruction,
            residual=remaining,
        )

    def hard_assign(self, residual: Tensor) -> RVQAssignment:
        """Use deterministic lowest-index ties and a hard forward assignment."""

        return self._assign(residual, temperature=1.0, straight_through=True)

    def soft_assign(
        self,
        residual: Tensor,
        temperature: float,
        straight_through: bool,
    ) -> RVQAssignment:
        return self._assign(residual, temperature, straight_through)


@dataclass(frozen=True, slots=True)
class FrozenGroupRVQDictionary:
    codebooks: Tensor
    revision_sha256: str

    def __post_init__(self) -> None:
        expected_shape = 4
        if self.codebooks.ndim != expected_shape:
            raise ValueError("frozen dictionary must have four dimensions")
        if self.codebooks.device.type != "cpu" or self.codebooks.dtype is not torch.float32:
            raise ValueError("frozen dictionary must be CPU float32")
        if self.codebooks.requires_grad:
            raise ValueError("frozen dictionary cannot require gradients")
        if not self.codebooks.is_contiguous():
            raise ValueError("frozen dictionary must be contiguous")
        if len(self.revision_sha256) != 64:
            raise ValueError("frozen dictionary revision must be a sha256")

    def packet(self, group: int, stage: int, entry: int) -> Packet:
        group_count, stages, entries, _group_size = self.codebooks.shape
        if (
            type(group) is not int
            or type(stage) is not int
            or type(entry) is not int
            or not 0 <= group < group_count
            or not 0 <= stage < stages
            or not 0 <= entry < entries
        ):
            raise IndexError("dictionary packet index is out of range")
        payload = _PACKET_HEADER.pack(
            _PACKET_MAGIC,
            bytes.fromhex(self.revision_sha256),
            group,
            stage,
            entry,
        )
        return packet_from_payload(payload)

    def direction(self, group: int, stage: int, entry: int) -> Tensor:
        group_count, stages, entries, _group_size = self.codebooks.shape
        if not (
            0 <= group < group_count
            and 0 <= stage < stages
            and 0 <= entry < entries
        ):
            raise IndexError("dictionary direction index is out of range")
        return self.codebooks[group, stage, entry]

    def validate_packet(self, packet: Packet) -> tuple[int, int, int]:
        canonical = packet_from_payload(packet.payload)
        if canonical.packet_id != packet.packet_id:
            raise ValueError("packet content address does not match payload")
        revision, group, stage, entry = decode_packet_key(packet.payload)
        if revision != self.revision_sha256:
            raise ValueError("packet belongs to another frozen dictionary")
        self.direction(group, stage, entry)
        return group, stage, entry


def _dictionary_digest(codebooks: Tensor) -> str:
    dimensions = tuple(int(value) for value in codebooks.shape)
    shape = struct.pack("<IIII", *dimensions)
    payload = codebooks.numpy().astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(shape + payload).hexdigest()


def freeze_dictionary(dictionary: GroupRVQDictionary) -> FrozenGroupRVQDictionary:
    if type(dictionary) is not GroupRVQDictionary:
        raise TypeError("dictionary must be an exact GroupRVQDictionary")
    raw = dictionary.codebooks.detach().float()
    norms = torch.linalg.vector_norm(raw, dim=-1)
    if not bool(torch.isfinite(raw).all().item()) or bool((norms <= 1e-8).any().item()):
        raise ValueError("dictionary codewords must be finite and nonzero")
    codebooks = F.normalize(raw, dim=-1, eps=1e-8).cpu().contiguous().clone()
    codebooks.requires_grad_(False)
    return FrozenGroupRVQDictionary(
        codebooks=codebooks,
        revision_sha256=_dictionary_digest(codebooks),
    )


def decode_packet_key(payload: bytes) -> tuple[str, int, int, int]:
    if type(payload) is not bytes or len(payload) != _PACKET_HEADER.size:
        raise ValueError("packet payload has the wrong byte length")
    magic, revision, group, stage, entry = _PACKET_HEADER.unpack(payload)
    if magic != _PACKET_MAGIC:
        raise ValueError("unsupported packet payload version")
    return revision.hex(), group, stage, entry


__all__ = [
    "FrozenGroupRVQDictionary",
    "GroupRVQDictionary",
    "RVQAssignment",
    "decode_packet_key",
    "freeze_dictionary",
]
