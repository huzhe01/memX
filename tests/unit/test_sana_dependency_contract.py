import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPOSITORY / "pyproject.toml"

EXPECTED_DEFAULT_VERSIONS = {
    "accelerate": "1.14.0",
    "cbor2": "6.1.4",
    "datasets": "5.0.1",
    "diffusers": "0.40.0",
    "filelock": "3.32.4",
    "huggingface-hub": "1.29.0",
    "jsonschema": "4.26.0",
    "peft": "0.20.0",
    "pillow": "12.3.0",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.16.1",
    "typer": "0.27.2",
}
EXPECTED_MODAL_VERSION = "1.5.4"
EXPECTED_PYTEST_VERSION = "9.0.3"


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _project_metadata() -> dict[str, object]:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)["project"]


def _exact_pins(requirements: list[str]) -> dict[str, str]:
    return dict(
        requirement.split("==", maxsplit=1) for requirement in requirements if "==" in requirement
    )


@pytest.mark.parametrize(("distribution", "expected"), EXPECTED_DEFAULT_VERSIONS.items())
def test_default_sana_dependency_has_exact_version(distribution: str, expected: str) -> None:
    assert _installed_version(distribution) == expected


def test_modal_is_an_exact_optional_dependency_not_a_default_dependency() -> None:
    project = _project_metadata()
    default_pins = _exact_pins(project["dependencies"])
    modal_pins = _exact_pins(project["optional-dependencies"]["modal"])

    assert "modal" not in default_pins
    assert modal_pins == {"modal": EXPECTED_MODAL_VERSION}


def test_default_dependency_metadata_uses_the_locked_versions() -> None:
    project = _project_metadata()
    default_pins = _exact_pins(project["dependencies"])
    assert {
        distribution: default_pins[distribution] for distribution in EXPECTED_DEFAULT_VERSIONS
    } == EXPECTED_DEFAULT_VERSIONS


def test_dev_test_runner_has_the_security_fixed_version() -> None:
    assert _installed_version("pytest") == EXPECTED_PYTEST_VERSION


@pytest.mark.skipif(
    _installed_version("modal") is None,
    reason="install the explicit Modal extra to exercise this runtime contract",
)
def test_installed_modal_extra_has_exact_version() -> None:
    assert _installed_version("modal") == EXPECTED_MODAL_VERSION


def test_pilot_entry_point_is_registered_with_its_target() -> None:
    project = _project_metadata()
    assert project["scripts"]["ratemem-pilot"] == "ratemem.pilot.cli:main"
    assert Path("src/ratemem/pilot/cli.py").is_file()
