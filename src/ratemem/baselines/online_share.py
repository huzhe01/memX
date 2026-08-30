"""Deterministic online shared-subspace control."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float32 = NDArray[np.float32]


def canonicalize_basis_signs(basis: Float32) -> Float32:
    if basis.ndim != 2 or basis.dtype != np.float32 or not np.isfinite(basis).all():
        raise ValueError("basis must be a finite float32 matrix")
    result = basis.copy()
    for row in result:
        if not np.any(row):
            continue
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    return result


def update_subspace(
    basis: Float32,
    coefficients: dict[str, Float32],
    incoming: Float32,
    incoming_handle: str,
    rank: int,
) -> tuple[Float32, dict[str, Float32]]:
    """Reproject only reconstructed resident codes plus the current incoming target."""

    if not incoming_handle:
        raise ValueError("incoming handle must be non-empty")
    if type(rank) is not int or rank < 1:
        raise ValueError("online SHARE rank must be positive")
    if basis.ndim != 2 or basis.dtype != np.float32 or not np.isfinite(basis).all():
        raise ValueError("basis must be a finite float32 matrix")
    if incoming.ndim != 1 or incoming.dtype != np.float32 or not np.isfinite(incoming).all():
        raise ValueError("incoming code must be one finite float32 vector")
    if basis.shape[1] != incoming.shape[0]:
        raise ValueError("basis and incoming code dimensions differ")
    reconstructed: dict[str, Float32] = {}
    for handle, coefficient in coefficients.items():
        if not handle or coefficient.dtype != np.float32 or coefficient.shape != (basis.shape[0],):
            raise ValueError("resident online SHARE coefficient has an invalid layout")
        if not np.isfinite(coefficient).all():
            raise ValueError("resident online SHARE coefficient must be finite")
        reconstructed[handle] = (coefficient @ basis).astype(np.float32)
    reconstructed[incoming_handle] = incoming.copy()
    handles = sorted(reconstructed)
    matrix = np.stack([reconstructed[handle] for handle in handles]).astype(np.float64)
    _u, _singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    new_rank = min(rank, len(vh))
    new_basis = canonicalize_basis_signs(vh[:new_rank].astype(np.float32))
    new_coefficients = {
        handle: (reconstructed[handle] @ new_basis.T).astype(np.float32)
        for handle in handles
    }
    return new_basis, new_coefficients


def reconstruction_drift_sha256(
    prior_basis: Float32,
    prior_coefficients: dict[str, Float32],
    new_basis: Float32,
    new_coefficients: dict[str, Float32],
) -> dict[str, str]:
    """Hash per-handle drift without retaining any original target code."""

    import hashlib

    common = sorted(set(prior_coefficients) & set(new_coefficients))
    result: dict[str, str] = {}
    for handle in common:
        before = prior_coefficients[handle] @ prior_basis
        after = new_coefficients[handle] @ new_basis
        drift = np.ascontiguousarray(after - before, dtype=np.float32)
        result[handle] = hashlib.sha256(drift.tobytes(order="C")).hexdigest()
    return result


__all__ = [
    "canonicalize_basis_signs",
    "reconstruction_drift_sha256",
    "update_subspace",
]
