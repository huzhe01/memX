from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ratemem.pilot.config import SanaPilotConfig, SubjectsPilotConfig
from ratemem.pilot.data import (
    CACHE_TENSOR_SPECS,
    build_precomputed_cache,
    hydrate_locked_examples,
    load_precomputed_cache,
)
from ratemem.sana.components import hydrate_pinned_snapshots, load_pinned_components

SANA_CONFIG_PATH = Path("configs/pilot/sana-1.5-1.6b.json")
SUBJECTS_CONFIG_PATH = Path("configs/pilot/subjects200k-held-in.json")


@pytest.mark.cuda
@pytest.mark.real_sana
@pytest.mark.paid_modal
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("RATEMEM_RUN_REAL_PILOT_CACHE") != "1",
    reason="explicit paid CUDA real-pilot-cache opt-in is required",
)
def test_real_locked_subjects_build_and_strict_reload_six_tensors(
    tmp_path: Path,
) -> None:
    """End-to-end contract for an explicitly authorized CUDA/Modal environment."""

    sana = SanaPilotConfig.load(SANA_CONFIG_PATH)
    subjects = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)
    model_cache = Path(
        os.environ.get("RATEMEM_REAL_SANA_CACHE", "/cache/huggingface")
    )
    subjects_cache_text = os.environ.get("RATEMEM_REAL_SUBJECTS_HF_CACHE")
    if not subjects_cache_text:
        pytest.fail(
            "RATEMEM_REAL_SUBJECTS_HF_CACHE must name an explicit cache directory"
        )

    snapshots = hydrate_pinned_snapshots(sana, cache_dir=model_cache)
    components = load_pinned_components(
        sana,
        snapshots=snapshots,
        device=torch.device("cuda"),
    )
    examples = hydrate_locked_examples(
        subjects, cache_dir=Path(subjects_cache_text)
    )
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    output = parent / "cache"

    built = build_precomputed_cache(examples, components, output, sana, subjects)
    loaded = load_precomputed_cache(output, expected_receipt=built.receipt)

    assert loaded.receipt == built.receipt
    assert tuple(loaded.tensors) == tuple(CACHE_TENSOR_SPECS)
    for name, (shape, dtype) in CACHE_TENSOR_SPECS.items():
        tensor = loaded.tensors[name]
        assert tuple(tensor.shape) == shape
        assert tensor.dtype is dtype
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert not tensor.requires_grad
        assert not tensor.is_inference()
