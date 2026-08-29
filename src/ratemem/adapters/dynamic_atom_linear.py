from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DynamicAtomLinear(nn.Module):
    """A frozen linear layer plus dynamically weighted low-rank atoms.

    Dynamic coefficients are deliberately transient: they are neither parameters nor
    buffers. Backward must run while the coefficient context is active so checkpoint
    recomputation cannot silently fall back to the frozen base path.
    """

    _coefficients: Tensor | None

    def __init__(self, base: nn.Linear, *, rank: int, atom_count: int) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if type(rank) is not int:
            raise TypeError("rank must be an int")
        if rank < 1:
            raise ValueError("rank must be positive")
        if type(atom_count) is not int:
            raise TypeError("atom_count must be an int")
        if atom_count < 1:
            raise ValueError("atom_count must be positive")

        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.atom_count = atom_count
        self.atom_down = nn.Parameter(
            base.weight.new_empty((atom_count, rank, base.in_features))
        )
        self.atom_up = nn.Parameter(
            base.weight.new_empty((atom_count, base.out_features, rank))
        )
        nn.init.normal_(self.atom_down, mean=0.0, std=0.01)
        nn.init.normal_(self.atom_up, mean=0.0, std=0.01)
        object.__setattr__(self, "_coefficients", None)

    @contextmanager
    def use_coefficients(self, coefficients: Tensor) -> Iterator[None]:
        if self._coefficients is not None:
            raise RuntimeError("coefficients are already active")
        if not isinstance(coefficients, Tensor):
            raise TypeError("coefficients must be a Tensor")
        if coefficients.ndim not in (1, 2):
            raise ValueError("coefficients must be 1D or 2D")
        if coefficients.shape[-1] != self.atom_count:
            raise ValueError(
                f"coefficient atom dimension must be {self.atom_count}"
            )
        object.__setattr__(self, "_coefficients", coefficients)
        try:
            yield
        finally:
            object.__setattr__(self, "_coefficients", None)

    def _validate_input(self, x: Tensor, coefficients: Tensor | None) -> None:
        if not isinstance(x, Tensor):
            raise TypeError("input must be a Tensor")
        if x.ndim < 1:
            raise ValueError("input must have at least one dimension")
        if x.shape[-1] != self.base.in_features:
            raise ValueError(
                f"input feature dimension must be {self.base.in_features}"
            )
        if coefficients is not None and coefficients.ndim == 2:
            if x.ndim < 2:
                raise ValueError(
                    "batched coefficients require input with a batch dimension"
                )
            coefficient_batch = coefficients.shape[0]
            input_batch = x.shape[0]
            if coefficient_batch != input_batch:
                raise ValueError(
                    f"coefficient batch {coefficient_batch} does not match "
                    f"input batch {input_batch}"
                )

    def _guard_backward_context(
        self, output: Tensor, coefficients: Tensor
    ) -> Tensor:
        if not output.requires_grad:
            return output

        def require_active_context(gradient: Tensor) -> Tensor:
            if self._coefficients is not coefficients:
                raise RuntimeError(
                    "coefficient context must remain active through backward"
                )
            return gradient

        output.register_hook(require_active_context)  # type: ignore[no-untyped-call]
        return output

    def forward(self, x: Tensor) -> Tensor:
        coefficients = self._coefficients
        self._validate_input(x, coefficients)
        output: Tensor = self.base(x)
        if coefficients is None:
            return output

        dynamic = torch.zeros_like(output)
        for atom_index in range(self.atom_count):
            low_rank = F.linear(x, self.atom_down[atom_index])
            atom_output = F.linear(low_rank, self.atom_up[atom_index])
            scale = (
                coefficients[atom_index]
                if coefficients.ndim == 1
                else coefficients[:, atom_index]
            )
            scale = scale.to(device=atom_output.device, dtype=atom_output.dtype)
            if scale.ndim == 1:
                scale = scale.reshape(
                    scale.shape[0], *([1] * (atom_output.ndim - 1))
                )
            dynamic = dynamic + atom_output * scale
        return self._guard_backward_context(output + dynamic, coefficients)
