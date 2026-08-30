from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

_PATH_TYPE = type(Path())


def _fixture_pixels(identity: str, *, size: int = 32) -> list[tuple[int, int, int]]:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    pixels: list[tuple[int, int, int]] = []
    for y in range(size):
        for x in range(size):
            pixels.append(
                (
                    (digest[0] + x * 7 + y * 3 + digest[(x + y) % 32]) % 256,
                    (digest[1] + x * 5 + y * 11 + digest[(x * 3 + y) % 32]) % 256,
                    (digest[2] + x * 13 + y * 2 + digest[(x + y * 5) % 32]) % 256,
                )
            )
    return pixels


def write_fixture_image(path: Path, *, identity: str) -> None:
    if type(path) is not _PATH_TYPE:
        raise TypeError("fixture image path must be an exact Path")
    if type(identity) is not str or not identity:
        raise TypeError("fixture image identity must be a non-empty exact str")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32))
    image.putdata(_fixture_pixels(identity))
    image.save(path, format="PNG", optimize=False, compress_level=9)
