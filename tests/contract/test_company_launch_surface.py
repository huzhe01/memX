from __future__ import annotations

import re
import stat
import subprocess
import tomllib
from pathlib import Path


def test_ppu_container_preserves_vendor_torch() -> None:
    dockerfile = Path("docker/Dockerfile.ppu").read_text(encoding="utf-8")

    assert "ARG PPU_BASE_IMAGE" in dockerfile
    assert "FROM ${PPU_BASE_IMAGE}" in dockerfile
    assert "PYTHONPATH=/workspace/memx/src" in dockerfile
    assert "pip install --no-cache-dir -r /tmp/memx/requirements/ppu.txt" in dockerfile
    assert "pip install --no-deps -e" not in dockerfile
    assert not re.search(r"(?:pip|uv).*(?:install|sync).*torch", dockerfile, re.IGNORECASE)


def test_ppu_requirements_are_exact_and_exclude_framework_wheels() -> None:
    lines = [
        line.strip()
        for line in Path("requirements/ppu.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert lines
    assert all(line.count("==") == 1 for line in lines)
    names = {line.split("==", maxsplit=1)[0].lower() for line in lines}
    assert "torch" not in names
    assert "torchvision" not in names
    assert "modal" not in names
    assert {"diffusers", "datasets", "transformers", "safetensors"} <= names


def test_project_registers_memx_entry_point() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert project["scripts"]["memx"] == "ratemem.experiment.cli:main"


def test_makefile_exposes_complete_operator_surface() -> None:
    source = Path("Makefile").read_text(encoding="utf-8")

    for target in ("bootstrap:", "data:", "smoke:", "train:", "evaluate:", "report:"):
        assert re.search(rf"^{re.escape(target)}", source, re.MULTILINE)
    for variable in ("DEVICE ?=", "WORLD_SIZE ?=", "DATA_ROOT ?=", "RUN_ROOT ?="):
        assert variable in source
    assert "modal" not in source.lower()


def test_make_help_documents_all_targets_and_variables() -> None:
    completed = subprocess.run(
        ["make", "help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for value in (
        "bootstrap",
        "data",
        "smoke",
        "train",
        "evaluate",
        "report",
        "DEVICE",
        "WORLD_SIZE",
        "DATA_ROOT",
        "RUN_ROOT",
    ):
        assert value in completed.stdout


def test_company_scripts_are_executable_valid_bash() -> None:
    for name in ("bootstrap.sh", "launch_train.sh", "run_memx.sh"):
        path = Path("scripts") / name
        source = path.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert stat.S_IMODE(path.stat().st_mode) == 0o755


def test_launcher_uses_generic_torchrun_without_hardware_or_provider_lock() -> None:
    source = Path("scripts/launch_train.sh").read_text(encoding="utf-8")

    for required in (
        "--nnodes",
        "--nproc-per-node",
        "--node-rank",
        "--master-addr",
        "--master-port",
        "-m ratemem.experiment.cli",
        '"${MEMX_MODE}"',
    ):
        assert required in source
    for forbidden in ("L40S", "H20", "ZW810E", "modal", "CUDA_VISIBLE_DEVICES="):
        assert forbidden not in source


def test_readme_is_clone_to_run_documentation_not_legacy_tensorflow_notes() -> None:
    source = Path("README.md").read_text(encoding="utf-8")

    for required in (
        "https://github.com/huzhe01/memX.git",
        "make bootstrap",
        "make data",
        "make smoke",
        "make train",
        "make evaluate",
        "make report",
        "RESUME=auto",
        "PPU-ZW810E",
        "PCCL",
        "publication_eligible=false",
    ):
        assert required in source
    assert "tensorflow-gpu       1.13.1" not in source


def test_ppu_runbook_has_one_eight_and_sixteen_card_gates() -> None:
    source = Path("docs/runbooks/company-ppu.md").read_text(encoding="utf-8")

    assert "WORLD_SIZE=1" in source
    assert "WORLD_SIZE=8" in source
    assert "WORLD_SIZE=16" in source
    assert "真实 ZW810E" in source
    assert "尚未验证" in source
