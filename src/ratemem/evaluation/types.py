"""Named constrained identifiers shared by scientific record models."""

from typing import Annotated

from pydantic import StringConstraints

Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
GitCommit = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$"),
]
ConceptToken = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^<concept_[0-9]{6}>$"),
]
ScientificProfile = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^ratemem-scientific-[a-z0-9_-]+$",
    ),
]
PhaseId = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[a-z0-9_-]+$"),
]

__all__ = [
    "ConceptToken",
    "GitCommit",
    "PhaseId",
    "ScientificProfile",
    "Sha256",
]
