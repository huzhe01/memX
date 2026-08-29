from importlib.metadata import PackageNotFoundError, version

import pytest

EXPECTED_VERSIONS = {
    "accelerate": "1.10.1",
    "datasets": "4.1.1",
    "diffusers": "0.35.1",
    "huggingface-hub": "0.34.4",
    "modal": "1.5.4",
    "peft": "0.17.1",
    "safetensors": "0.6.2",
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "4.56.1",
}


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


@pytest.mark.parametrize(("distribution", "expected"), EXPECTED_VERSIONS.items())
def test_sana_dependency_has_exact_version(distribution: str, expected: str) -> None:
    assert _installed_version(distribution) == expected
