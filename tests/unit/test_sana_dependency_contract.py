import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPOSITORY / "pyproject.toml"
SANA_PLAN_PATH = (
    REPOSITORY / "docs/superpowers/plans/2026-08-24-ratemem-sana-modal-pilot.md"
)

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
        requirement.split("==", maxsplit=1)
        for requirement in requirements
        if "==" in requirement
    )


@pytest.mark.parametrize(
    ("distribution", "expected"), EXPECTED_DEFAULT_VERSIONS.items()
)
def test_default_sana_dependency_has_exact_version(
    distribution: str, expected: str
) -> None:
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
        distribution: default_pins[distribution]
        for distribution in EXPECTED_DEFAULT_VERSIONS
    } == EXPECTED_DEFAULT_VERSIONS


def test_dev_test_runner_has_the_security_fixed_version() -> None:
    assert _installed_version("pytest") == EXPECTED_PYTEST_VERSION


@pytest.mark.skipif(
    _installed_version("modal") is None,
    reason="install the explicit Modal extra to exercise this runtime contract",
)
def test_installed_modal_extra_has_exact_version() -> None:
    assert _installed_version("modal") == EXPECTED_MODAL_VERSION


def test_pilot_entry_point_is_deferred_to_task_13() -> None:
    project = _project_metadata()
    assert "ratemem-pilot" not in project["scripts"]

    plan = SANA_PLAN_PATH.read_text(encoding="utf-8")
    task_one = plan.split("### Task 1:", maxsplit=1)[1].split(
        "### Task 2:", maxsplit=1
    )[0]
    task_thirteen = plan.split("### Task 13:", maxsplit=1)[1].split(
        "### Task 14:", maxsplit=1
    )[0]
    assert 'ratemem-pilot = "ratemem.pilot.cli:main"' not in task_one
    assert 'ratemem-pilot = "ratemem.pilot.cli:main"' in task_thirteen
    assert "uv run ratemem-pilot --help" in task_thirteen


def test_sana_plan_uses_the_locked_dependency_versions() -> None:
    plan = SANA_PLAN_PATH.read_text(encoding="utf-8")
    old_versions = {
        "1.10.1",
        "5.7.0",
        "4.1.1",
        "0.35.1",
        "3.19.1",
        "0.34.4",
        "4.25.1",
        "0.17.1",
        "11.3.0",
        "0.6.2",
        "2.8.0",
        "0.23.0",
        "4.56.1",
        "0.16.1",
    }
    for old_version in old_versions:
        assert old_version not in plan
    for expected in (
        *EXPECTED_DEFAULT_VERSIONS.values(),
        EXPECTED_MODAL_VERSION,
        EXPECTED_PYTEST_VERSION,
    ):
        assert expected in plan


def test_sana_plan_defines_an_offline_safe_hub_loading_contract() -> None:
    plan = SANA_PLAN_PATH.read_text(encoding="utf-8")
    task_four = plan.split("### Task 4:", maxsplit=1)[1].split(
        "### Task 5:", maxsplit=1
    )[0]
    assert "no network" in task_four.lower()
    assert "use_safetensors=True" in task_four
    assert "trust_remote_code=False" in task_four
    assert "custom_pipeline" in task_four
