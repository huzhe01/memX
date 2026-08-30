"""Frozen CTS-style and VB-LoRA-style shared code representations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from scipy.optimize import nnls  # type: ignore[import-untyped]

from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.evaluation.types import Sha256

Float32 = NDArray[np.float32]
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class LeakageError(ValueError):
    """Raised before fitting shared state from a non-training corpus."""


def truncated_basis(train_codes: Float32, rank: int) -> Float32:
    if (
        train_codes.ndim != 2
        or train_codes.dtype != np.float32
        or not np.isfinite(train_codes).all()
        or not 0 < rank <= min(train_codes.shape)
    ):
        raise ValueError("invalid static basis shape, values, dtype, or rank")
    _u, _singular_values, vh = np.linalg.svd(
        train_codes.astype(np.float64),
        full_matrices=False,
    )
    basis = vh[:rank].astype(np.float32)
    for row in basis:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    return basis


def vb_encode_subvector(
    vector: Float32,
    bank: Float32,
    top_k: int,
) -> tuple[tuple[int, ...], Float32]:
    if (
        vector.ndim != 1
        or bank.ndim != 2
        or bank.shape[1] != vector.shape[0]
        or vector.dtype != np.float32
        or bank.dtype != np.float32
        or not np.isfinite(vector).all()
        or not np.isfinite(bank).all()
        or not 0 < top_k <= len(bank)
    ):
        raise ValueError("invalid VB subvector, bank, or top-k")
    target = vector.astype(np.float64)
    residual = target.copy()
    selected: list[int] = []
    for _ in range(top_k):
        scores = bank.astype(np.float64) @ residual
        candidates = [index for index in range(len(scores)) if index not in selected]
        index = min(candidates, key=lambda item: (-abs(float(scores[item])), item))
        selected.append(index)
        design = bank[selected].astype(np.float64).T
        weights, _residual_norm = nnls(design, target)
        residual = target - design @ weights
    return tuple(selected), weights.astype(np.float32)


@dataclass(frozen=True, slots=True)
class CtsEncoded:
    coordinates: tuple[Float32, ...]


@dataclass(frozen=True, slots=True)
class VbEncoded:
    indices: tuple[tuple[int, ...], ...]
    weights: tuple[Float32, ...]


class CtsCodebook:
    def __init__(self, group_bases: tuple[Float32, ...], quantization_bits: int) -> None:
        if quantization_bits not in {8, 16, 32}:
            raise ValueError("CTS coordinate precision must be 8, 16, or 32 bits")
        if not group_bases:
            raise ValueError("CTS requires at least one group basis")
        widths: list[int] = []
        owned: list[Float32] = []
        for basis in group_bases:
            if basis.ndim != 2 or basis.dtype != np.float32 or not np.isfinite(basis).all():
                raise ValueError("CTS bases must be finite float32 matrices")
            if basis.shape[0] < 1 or basis.shape[1] < 1:
                raise ValueError("CTS basis dimensions must be positive")
            widths.append(basis.shape[1])
            owned.append(basis.copy())
        self.group_bases = tuple(owned)
        self.group_widths = tuple(widths)
        self.quantization_bits = quantization_bits

    @classmethod
    def from_fixture(
        cls,
        *,
        group_bases: tuple[Float32, ...],
        quantization_bits: int,
    ) -> CtsCodebook:
        return cls(group_bases, quantization_bits)

    def _quantize(self, value: Float32) -> Float32:
        if self.quantization_bits == 16:
            return value.astype(np.float16).astype(np.float32)
        if self.quantization_bits == 8:
            scale = max(
                float(np.max(np.abs(value))) / 127.0,
                float(np.finfo(np.float32).eps),
            )
            return cast(
                Float32,
                (np.round(value / scale).clip(-127, 127) * scale).astype(np.float32),
            )
        return value.astype(np.float32)

    def encode(self, code: Float32) -> CtsEncoded:
        if code.dtype != np.float32 or code.shape != (sum(self.group_widths),):
            raise ValueError("CTS code has an invalid layout")
        coordinates: list[Float32] = []
        start = 0
        for basis, width in zip(self.group_bases, self.group_widths, strict=True):
            segment = code[start : start + width]
            coordinates.append(self._quantize((basis @ segment).astype(np.float32)))
            start += width
        return CtsEncoded(tuple(coordinates))

    def decode(self, encoded: CtsEncoded) -> Float32:
        if len(encoded.coordinates) != len(self.group_bases):
            raise ValueError("CTS encoded group count mismatch")
        segments = [
            (basis.T @ coordinate).astype(np.float32)
            for basis, coordinate in zip(
                self.group_bases,
                encoded.coordinates,
                strict=True,
            )
        ]
        return np.concatenate(segments).astype(np.float32)


class VbCodebook:
    def __init__(
        self,
        *,
        bank: Float32,
        subvector_size: int,
        top_k: int,
        weight_bits: int,
    ) -> None:
        if (
            bank.ndim != 2
            or bank.dtype != np.float32
            or not np.isfinite(bank).all()
            or bank.shape[1] != subvector_size
        ):
            raise ValueError("VB bank has an invalid layout")
        if not 0 < top_k <= len(bank):
            raise ValueError("VB top-k is outside the bank")
        if weight_bits not in {8, 16, 32}:
            raise ValueError("VB weight precision must be 8, 16, or 32 bits")
        self.bank = bank.copy()
        self.subvector_size = subvector_size
        self.top_k = top_k
        self.weight_bits = weight_bits

    @classmethod
    def from_fixture(
        cls,
        *,
        bank: Float32,
        subvector_size: int,
        top_k: int,
        weight_bits: int,
    ) -> VbCodebook:
        return cls(
            bank=bank,
            subvector_size=subvector_size,
            top_k=top_k,
            weight_bits=weight_bits,
        )

    def _quantize_weights(self, weights: Float32) -> Float32:
        if self.weight_bits == 16:
            return weights.astype(np.float16).astype(np.float32)
        if self.weight_bits == 8:
            scale = max(
                float(np.max(np.abs(weights))) / 255.0,
                float(np.finfo(np.float32).eps),
            )
            return cast(
                Float32,
                (np.round(weights / scale).clip(0, 255) * scale).astype(np.float32),
            )
        return weights.astype(np.float32)

    def encode(self, code: Float32) -> VbEncoded:
        if (
            code.ndim != 1
            or code.dtype != np.float32
            or len(code) % self.subvector_size
            or not np.isfinite(code).all()
        ):
            raise ValueError("VB code must be a divisible finite float32 vector")
        indices: list[tuple[int, ...]] = []
        weights: list[Float32] = []
        for start in range(0, len(code), self.subvector_size):
            selected, values = vb_encode_subvector(
                code[start : start + self.subvector_size],
                self.bank,
                self.top_k,
            )
            indices.append(selected)
            weights.append(self._quantize_weights(values))
        return VbEncoded(tuple(indices), tuple(weights))

    def decode(self, encoded: VbEncoded) -> Float32:
        if len(encoded.indices) != len(encoded.weights):
            raise ValueError("VB index and weight group counts differ")
        segments: list[Float32] = []
        for indices, weights in zip(encoded.indices, encoded.weights, strict=True):
            if len(indices) != len(weights):
                raise ValueError("VB index and weight counts differ")
            segment = weights.astype(np.float32) @ self.bank[list(indices)]
            segments.append(segment.astype(np.float32))
        return np.concatenate(segments).astype(np.float32)


@dataclass(frozen=True, slots=True)
class CodeCorpus:
    split: Literal["train", "validation", "final_test"]
    codes: Float32
    manifest_sha256: Sha256


class StaticCodebookArtifact(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    family: Literal["cts_style_static", "vb_lora_style_static"]
    training_manifest_sha256: Sha256
    code_corpus_sha256: Sha256
    seed: int = Field(ge=0)
    group_boundaries: tuple[PositiveInt, ...]
    rank: PositiveInt | None
    bank_size: PositiveInt | None
    top_k: PositiveInt | None
    quantization_bits: Literal[8, 16, 32]
    tensor_file: Path
    tensor_sha256: Sha256
    codebook_sha256: Sha256


def fit_static_codebook(
    corpus: CodeCorpus,
    *,
    family: Literal["cts_style_static", "vb_lora_style_static"],
    rank: int = 8,
    bank_size: int = 256,
    subvector_size: int = 16,
    top_k: int = 2,
    quantization_bits: int = 16,
) -> CtsCodebook | VbCodebook:
    """Fit deterministic shared state from train codes only."""

    if corpus.split != "train":
        raise LeakageError("static codebook accepts train split only")
    codes = corpus.codes
    if codes.ndim != 2 or codes.dtype != np.float32 or not np.isfinite(codes).all():
        raise ValueError("static code corpus must be a finite float32 matrix")
    if family == "cts_style_static":
        return CtsCodebook((truncated_basis(codes, rank),), quantization_bits)
    if len(codes[0]) % subvector_size:
        raise ValueError("VB corpus width must be divisible by subvector size")
    subvectors = codes.reshape(-1, subvector_size)
    if bank_size > len(subvectors):
        raise ValueError("VB bank size exceeds the training subvector count")
    keys = [
        hashlib.sha256(row.tobytes(order="C")).digest()
        for row in np.ascontiguousarray(subvectors)
    ]
    order = sorted(range(len(keys)), key=lambda index: (keys[index], index))
    bank = subvectors[order[:bank_size]].copy()
    return VbCodebook(
        bank=bank,
        subvector_size=subvector_size,
        top_k=top_k,
        weight_bits=quantization_bits,
    )


def codebook_semantic_sha256(codebook: CtsCodebook | VbCodebook) -> str:
    if isinstance(codebook, CtsCodebook):
        payload = {
            "family": "cts_style_static",
            "quantization_bits": codebook.quantization_bits,
            "bases": [
                {
                    "shape": list(basis.shape),
                    "sha256": hashlib.sha256(basis.tobytes(order="C")).hexdigest(),
                }
                for basis in codebook.group_bases
            ],
        }
    else:
        payload = {
            "family": "vb_lora_style_static",
            "subvector_size": codebook.subvector_size,
            "top_k": codebook.top_k,
            "weight_bits": codebook.weight_bits,
            "bank_sha256": hashlib.sha256(codebook.bank.tobytes(order="C")).hexdigest(),
        }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


__all__ = [
    "CodeCorpus",
    "CtsCodebook",
    "CtsEncoded",
    "LeakageError",
    "StaticCodebookArtifact",
    "VbCodebook",
    "VbEncoded",
    "codebook_semantic_sha256",
    "fit_static_codebook",
    "truncated_basis",
    "vb_encode_subvector",
]
