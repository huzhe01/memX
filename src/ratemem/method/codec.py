"""Hard RateMem codec and a differentiable surrogate with identical deployed top-k."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer, decode_base_payload
from ratemem.method.dictionary import (
    FrozenGroupRVQDictionary,
    GroupRVQDictionary,
    freeze_dictionary,
)
from ratemem.state.model import Incidence, Packet


def quantize_gain(value: float, step: float) -> int:
    if not math.isfinite(value) or not math.isfinite(step) or step <= 0.0:
        raise ValueError("gain and step must be finite with positive step")
    return int(np.clip(np.rint(value / step), -32768, 32767))


def dequantize_gain(value: int, step: float) -> float:
    if (
        type(value) is not int
        or not -32768 <= value <= 32767
        or not math.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("gain_q must fit int16 and step must be finite and positive")
    return value * step


@dataclass(frozen=True, slots=True, order=True)
class PacketCandidateKey:
    group: int
    stage: int
    entry: int
    gain_q: int


def select_packet_topk(
    rows: Sequence[PacketCandidateKey],
    maximum_packets: int,
) -> tuple[PacketCandidateKey, ...]:
    if type(maximum_packets) is not int or maximum_packets < 1:
        raise ValueError("maximum_packets must be positive")
    if len({(row.group, row.stage) for row in rows}) != len(rows):
        raise ValueError("candidate stream repeats a group-stage position")
    if any(
        min(row.group, row.stage, row.entry) < 0
        or not -32768 <= row.gain_q <= 32767
        for row in rows
    ):
        raise ValueError("candidate indices or gain are out of range")
    ranked = sorted(
        rows,
        key=lambda row: (
            -(row.gain_q * row.gain_q),
            row.group,
            row.stage,
            row.entry,
        ),
    )
    return tuple(ranked[:maximum_packets])


@dataclass(frozen=True, slots=True)
class HardIncidence:
    incidence: Incidence
    packet: Packet
    group: int
    stage: int
    entry: int
    residual_reduction: float

    @property
    def key(self) -> PacketCandidateKey:
        return PacketCandidateKey(
            self.group,
            self.stage,
            self.entry,
            self.incidence.gain_q,
        )


@dataclass(frozen=True, slots=True)
class HardConceptEncoding:
    handle: str
    base_payload: bytes
    all_candidates: tuple[HardIncidence, ...]
    incidences: tuple[HardIncidence, ...]


class RateMemHardCodec:
    """The only deployment codec; ranking happens after int16 gain quantization."""

    def __init__(
        self,
        base_quantizer: BlockwiseBaseQuantizer,
        dictionary: FrozenGroupRVQDictionary,
        gain_step: float,
        maximum_packets: int,
    ) -> None:
        if type(base_quantizer) is not BlockwiseBaseQuantizer:
            raise TypeError("base_quantizer must be an exact BlockwiseBaseQuantizer")
        if type(dictionary) is not FrozenGroupRVQDictionary:
            raise TypeError("dictionary must be an exact FrozenGroupRVQDictionary")
        if base_quantizer.group_size != dictionary.codebooks.shape[-1]:
            raise ValueError("base and dictionary group sizes differ")
        candidate_count = dictionary.codebooks.shape[0] * dictionary.codebooks.shape[1]
        if type(maximum_packets) is not int or not 1 <= maximum_packets <= candidate_count:
            raise ValueError("maximum_packets exceeds the candidate count")
        if not math.isfinite(gain_step) or gain_step <= 0.0:
            raise ValueError("gain_step must be finite and positive")
        self.base_quantizer = base_quantizer
        self.dictionary = dictionary
        self.gain_step = gain_step
        self.maximum_packets = maximum_packets

    def encode(
        self,
        handle: str,
        code: NDArray[np.generic],
    ) -> HardConceptEncoding:
        if type(handle) is not str or not handle:
            raise ValueError("handle must be a nonempty exact string")
        flat = np.asarray(code, dtype=np.float32).reshape(-1)
        expected = self.dictionary.codebooks.shape[0] * self.dictionary.codebooks.shape[-1]
        if flat.shape != (expected,):
            raise ValueError(f"code must have exact width {expected}")
        base = self.base_quantizer.encode(flat)
        base_vector = base.decode()
        residual = torch.from_numpy(
            (flat - base_vector).reshape(
                1,
                self.dictionary.codebooks.shape[0],
                self.dictionary.codebooks.shape[-1],
            )
        )
        trainable = GroupRVQDictionary(
            self.dictionary.codebooks.shape[0],
            self.dictionary.codebooks.shape[-1],
            self.dictionary.codebooks.shape[1],
            self.dictionary.codebooks.shape[2],
        )
        with torch.no_grad():
            trainable.codebooks.copy_(self.dictionary.codebooks)
            assigned = trainable.hard_assign(residual)
        rows: list[HardIncidence] = []
        for group in range(trainable.group_count):
            for stage in range(trainable.stages):
                entry = int(assigned.indices[0, group, stage].item())
                gain_q = quantize_gain(
                    float(assigned.gains[0, group, stage].item()),
                    self.gain_step,
                )
                packet = self.dictionary.packet(group, stage, entry)
                rows.append(
                    HardIncidence(
                        incidence=Incidence(handle, packet.packet_id, gain_q),
                        packet=packet,
                        group=group,
                        stage=stage,
                        entry=entry,
                        residual_reduction=dequantize_gain(gain_q, self.gain_step) ** 2,
                    )
                )
        keys = select_packet_topk(tuple(row.key for row in rows), self.maximum_packets)
        by_key = {row.key: row for row in rows}
        selected = tuple(by_key[key] for key in keys)
        return HardConceptEncoding(handle, base.payload, tuple(rows), selected)

    def decode(
        self,
        base_payload: bytes,
        incidences: Sequence[HardIncidence],
    ) -> NDArray[np.float32]:
        if len(incidences) > self.maximum_packets:
            raise ValueError("decoded incidence count exceeds the codec maximum")
        output = decode_base_payload(base_payload).copy()
        expected = self.dictionary.codebooks.shape[0] * self.dictionary.codebooks.shape[-1]
        if output.shape != (expected,):
            raise ValueError("base payload width differs from the dictionary")
        if len({(row.group, row.stage) for row in incidences}) != len(incidences):
            raise ValueError("decoded incidences repeat a group-stage position")
        group_size = self.dictionary.codebooks.shape[-1]
        for row in incidences:
            packet_key = self.dictionary.validate_packet(row.packet)
            if packet_key != (row.group, row.stage, row.entry):
                raise ValueError("incidence metadata does not match packet payload")
            if row.incidence.packet_id != row.packet.packet_id:
                raise ValueError("incidence references another packet")
            start = row.group * group_size
            direction = self.dictionary.direction(row.group, row.stage, row.entry).numpy()
            output[start : start + group_size] += (
                dequantize_gain(row.incidence.gain_q, self.gain_step) * direction
            )
        return output


@dataclass(frozen=True, slots=True)
class DifferentiableEncoding:
    reconstruction: Tensor
    base_reconstruction: Tensor
    assignment_probabilities: Tensor
    hard_indices: Tensor
    quantized_gains: Tensor
    selected_mask: Tensor
    selected_keys: tuple[tuple[PacketCandidateKey, ...], ...]


@dataclass(frozen=True, slots=True)
class SoftHardAgreement:
    mean_code_error: float
    maximum_code_error: float
    assignment_disagreement: float
    topk_disagreement: float


def _ste_gain(values: Tensor, step: float) -> Tensor:
    hard = torch.round(values / step).clamp(-32768, 32767) * step
    return hard + values - values.detach()


def _hard_mask(
    keys: tuple[tuple[PacketCandidateKey, ...], ...],
    *,
    group_count: int,
    stages: int,
    device: torch.device,
) -> Tensor:
    mask = torch.zeros(
        len(keys), group_count, stages, dtype=torch.float32, device=device
    )
    for batch_index, selected in enumerate(keys):
        for row in selected:
            mask[batch_index, row.group, row.stage] = 1.0
    return mask


class RateMemDifferentiableCodec(nn.Module):
    def __init__(
        self,
        dictionary: GroupRVQDictionary,
        group_size: int,
        base_bits: int,
        gain_step: float,
        maximum_packets: int,
    ) -> None:
        super().__init__()
        if type(dictionary) is not GroupRVQDictionary:
            raise TypeError("dictionary must be an exact GroupRVQDictionary")
        if group_size != dictionary.group_size:
            raise ValueError("codec and dictionary group sizes differ")
        if base_bits not in {2, 4, 8}:
            raise ValueError("base_bits must be 2, 4, or 8")
        candidate_count = dictionary.group_count * dictionary.stages
        if type(maximum_packets) is not int or not 1 <= maximum_packets <= candidate_count:
            raise ValueError("maximum_packets exceeds the candidate count")
        if not math.isfinite(gain_step) or gain_step <= 0.0:
            raise ValueError("gain_step must be finite and positive")
        self.dictionary = dictionary
        self.group_size = group_size
        self.base_bits = base_bits
        self.gain_step = gain_step
        self.maximum_packets = maximum_packets

    def _actual_hard(
        self,
        code: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[tuple[PacketCandidateKey, ...], ...]]:
        hard_codec = RateMemHardCodec(
            BlockwiseBaseQuantizer(self.group_size, self.base_bits),
            freeze_dictionary(self.dictionary),
            self.gain_step,
            self.maximum_packets,
        )
        reconstructions: list[Tensor] = []
        bases: list[Tensor] = []
        selected_keys: list[tuple[PacketCandidateKey, ...]] = []
        rows = code.detach().float().cpu().numpy()
        for batch_index, row in enumerate(rows):
            encoded = hard_codec.encode(f"ste-{batch_index}", row)
            reconstructions.append(
                torch.from_numpy(
                    hard_codec.decode(encoded.base_payload, encoded.incidences)
                )
            )
            bases.append(torch.from_numpy(decode_base_payload(encoded.base_payload)))
            selected_keys.append(tuple(item.key for item in encoded.incidences))
        return (
            torch.stack(reconstructions).to(device=code.device),
            torch.stack(bases).to(device=code.device),
            tuple(selected_keys),
        )

    def forward(
        self,
        code: Tensor,
        *,
        temperature: float,
        mode: Literal["soft", "ste"],
    ) -> DifferentiableEncoding:
        if type(code) is not Tensor or code.ndim != 2:
            raise ValueError("code must be a rank-two exact tensor")
        expected_width = self.dictionary.group_count * self.group_size
        if code.shape[1] != expected_width:
            raise ValueError("code has the wrong width")
        if not code.is_floating_point() or not bool(torch.isfinite(code).all().item()):
            raise ValueError("code must be a finite floating tensor")
        if code.device != self.dictionary.codebooks.device:
            raise ValueError("code and dictionary must share one device")
        if mode not in ("soft", "ste"):
            raise ValueError("mode must be soft or ste")
        actual, hard_base, selected_keys = self._actual_hard(code)
        base = hard_base + code.float() - code.float().detach()
        residual = (code.float() - hard_base.detach()).reshape(
            code.shape[0], self.dictionary.group_count, self.group_size
        )
        assignment = self.dictionary.soft_assign(
            residual,
            temperature=temperature,
            straight_through=(mode == "ste"),
        )
        gains = _ste_gain(assignment.gains, self.gain_step)
        hard_selected = _hard_mask(
            selected_keys,
            group_count=self.dictionary.group_count,
            stages=self.dictionary.stages,
            device=code.device,
        )
        score = gains.square().to(torch.float64)
        candidate_count = score.shape[1] * score.shape[2]
        flat_tie_rank = torch.arange(
            candidate_count, device=score.device, dtype=score.dtype
        ).reshape(1, score.shape[1], score.shape[2])
        tie_unit = (self.gain_step * self.gain_step) / (2 * (candidate_count + 1))
        ranked_score = score + (candidate_count - 1 - flat_tie_rank) * tie_unit
        threshold = torch.topk(
            ranked_score.reshape(code.shape[0], -1),
            k=self.maximum_packets,
            dim=-1,
        ).values[:, -1].reshape(-1, 1, 1)
        soft_selected = torch.sigmoid(
            (ranked_score - threshold.detach()) / max(float(temperature), 1e-6)
        ).to(dtype=gains.dtype)
        selection = (
            hard_selected + soft_selected - soft_selected.detach()
            if mode == "ste"
            else soft_selected
        )
        directions = F.normalize(self.dictionary.codebooks.float(), dim=-1, eps=1e-8)
        surrogate = base.reshape(
            code.shape[0], self.dictionary.group_count, self.group_size
        )
        for stage in range(self.dictionary.stages):
            chosen = torch.einsum(
                "bge,ged->bgd",
                assignment.probabilities[:, :, stage],
                directions[:, stage],
            )
            surrogate = surrogate + (
                gains[:, :, stage].unsqueeze(-1)
                * selection[:, :, stage].unsqueeze(-1)
                * chosen
            )
        surrogate = surrogate.reshape_as(code)
        reconstruction = (
            actual + surrogate - surrogate.detach() if mode == "ste" else surrogate
        )
        hard_indices = torch.full(
            (code.shape[0], self.dictionary.group_count, self.dictionary.stages),
            -1,
            dtype=torch.long,
            device=code.device,
        )
        hard_gain_q = torch.zeros_like(hard_indices)
        for batch_index, selected in enumerate(selected_keys):
            for row in selected:
                hard_indices[batch_index, row.group, row.stage] = row.entry
                hard_gain_q[batch_index, row.group, row.stage] = row.gain_q
        return DifferentiableEncoding(
            reconstruction=reconstruction,
            base_reconstruction=base,
            assignment_probabilities=assignment.probabilities,
            hard_indices=hard_indices,
            quantized_gains=hard_gain_q,
            selected_mask=selection,
            selected_keys=selected_keys,
        )

    def hard_reference(self, code: Tensor) -> DifferentiableEncoding:
        return self.forward(code, temperature=1.0, mode="ste")


def _selected_position_sets(
    encoding: DifferentiableEncoding,
) -> tuple[frozenset[tuple[int, int]], ...]:
    flat = encoding.selected_mask.detach().reshape(encoding.selected_mask.shape[0], -1)
    count = len(encoding.selected_keys[0]) if encoding.selected_keys else 0
    indices = torch.topk(flat, k=count, dim=-1).indices.cpu().tolist()
    stages = encoding.selected_mask.shape[2]
    return tuple(
        frozenset((int(index) // stages, int(index) % stages) for index in row)
        for row in indices
    )


def measure_soft_hard_agreement(
    soft: DifferentiableEncoding,
    hard: DifferentiableEncoding,
) -> SoftHardAgreement:
    if soft.reconstruction.shape != hard.reconstruction.shape:
        raise ValueError("soft and hard reconstruction shapes differ")
    difference = (soft.reconstruction.detach().float() - hard.reconstruction.detach().float())
    squared = difference.square()
    hard_positions = tuple(
        frozenset((row.group, row.stage) for row in selected)
        for selected in hard.selected_keys
    )
    soft_positions = _selected_position_sets(soft)
    denominator = max(1, sum(len(row) for row in hard_positions))
    topk_disagreement = sum(
        len(left.symmetric_difference(right))
        for left, right in zip(soft_positions, hard_positions, strict=True)
    ) / denominator
    soft_assignments = soft.assignment_probabilities.detach().argmax(dim=-1)
    mismatch_count = 0
    observed_count = 0
    for batch_index, positions in enumerate(hard_positions):
        for group, stage in positions:
            observed_count += 1
            mismatch_count += int(
                soft_assignments[batch_index, group, stage].item()
                != hard.hard_indices[batch_index, group, stage].item()
            )
    return SoftHardAgreement(
        mean_code_error=float(squared.mean().item()),
        maximum_code_error=float(difference.abs().max().item()),
        assignment_disagreement=mismatch_count / max(1, observed_count),
        topk_disagreement=topk_disagreement,
    )


def enforce_agreement(
    report: SoftHardAgreement | None = None,
    *,
    mean_code_error: float | None = None,
    assignment_disagreement: float | None = None,
    topk_disagreement: float | None = None,
    maximum_mean_code_error: float,
    maximum_assignment_disagreement: float,
    maximum_topk_disagreement: float,
) -> None:
    if report is not None:
        if any(
            value is not None
            for value in (
                mean_code_error,
                assignment_disagreement,
                topk_disagreement,
            )
        ):
            raise ValueError("provide either an agreement report or explicit values")
        values = (
            report.mean_code_error,
            report.assignment_disagreement,
            report.topk_disagreement,
        )
    else:
        if (
            mean_code_error is None
            or assignment_disagreement is None
            or topk_disagreement is None
        ):
            raise ValueError("explicit agreement values are incomplete")
        values = (
            mean_code_error,
            assignment_disagreement,
            topk_disagreement,
        )
    thresholds = (
        maximum_mean_code_error,
        maximum_assignment_disagreement,
        maximum_topk_disagreement,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in (*values, *thresholds)):
        raise ValueError("agreement values and thresholds must be finite and nonnegative")
    if any(value > threshold for value, threshold in zip(values, thresholds, strict=True)):
        raise RuntimeError("soft-hard agreement release gate failed")


__all__ = [
    "DifferentiableEncoding",
    "HardConceptEncoding",
    "HardIncidence",
    "PacketCandidateKey",
    "RateMemDifferentiableCodec",
    "RateMemHardCodec",
    "SoftHardAgreement",
    "dequantize_gain",
    "enforce_agreement",
    "measure_soft_hard_agreement",
    "quantize_gain",
    "select_packet_topk",
]
