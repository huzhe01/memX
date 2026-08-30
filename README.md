# memX / RateMem-DiT

memX is the reproducible implementation that supersedes the unpublished 2020
*Memory-MetaGAN* prototype. The new research direction studies byte-bounded, shared progressive
adapter memory for optimization-free image personalization on a frozen diffusion transformer.

The repository currently contains two deliberately separate execution surfaces:

- a locally verified, end-to-end orchestration fixture covering data preparation, distributed
  runtime policy, optimization, atomic checkpoint/resume, evaluation, and reporting; and
- the existing SANA-1.5 dynamic-adapter engineering code and guarded single-L40S Modal pilot.

The fixture emits `publication_eligible=false`. It proves that the pipeline runs; it is not a
CVPR result. Production RateMem/SANA training, matched baselines, and scientific datasets are being
implemented on top of this interface and remain subject to real PPU validation.

## Clone

Until the implementation branch is promoted to the default branch, clone it explicitly:

```bash
git clone --branch codex/ratemem-implementation \
  https://github.com/huzhe01/memX.git
cd memX
```

No access token belongs in the clone URL, repository, command history, or experiment artifact.

## Five-minute CPU verification

Install [uv](https://docs.astral.sh/uv/) 0.8.14 and run:

```bash
make bootstrap
make data DATA_ROOT=/tmp/memx-data
make smoke DEVICE=cpu DATA_ROOT=/tmp/memx-data RUN_ROOT=/tmp/memx-run
make evaluate DEVICE=cpu DATA_ROOT=/tmp/memx-data RUN_ROOT=/tmp/memx-run
make report RUN_ROOT=/tmp/memx-run
```

The smoke profile generates eight deterministic CC0 fixture concepts locally. It performs no
network download. Outputs include:

```text
/tmp/memx-run/
├── checkpoints/step-00000006/
├── metrics.jsonl
├── train-result.json
├── evaluation.json
└── report/
    ├── report.json
    ├── metrics.csv
    └── REPORT.md
```

## Resume an interrupted run

Checkpoints are published only after their tensor and state hashes validate. Continue the same run
with:

```bash
make train DEVICE=cpu DATA_ROOT=/tmp/memx-data RUN_ROOT=/tmp/memx-run RESUME=auto
```

A changed experiment config or prepared dataset hash blocks resume. `RESUME=never` never overwrites
an existing run.

## PPU-ZW810E container

Supply a company base image containing its compatible Python, PyTorch, torchvision, PPU SDK, and
PCCL. The memX image installs only non-framework Python dependencies, so it does not replace the
vendor torch build:

```bash
docker build \
  --build-arg PPU_BASE_IMAGE=registry.company/ppu-pytorch:approved \
  -f docker/Dockerfile.ppu \
  -t memx:ppu .
```

Inside the image:

```bash
make bootstrap DEVICE=ppu
make data DEVICE=ppu DATA_ROOT=/data/memx
make smoke DEVICE=ppu DATA_ROOT=/data/memx RUN_ROOT=/output/memx-smoke
make train DEVICE=ppu WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 \
  DATA_ROOT=/data/memx RUN_ROOT=/output/memx-train
make evaluate DEVICE=ppu WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 \
  DATA_ROOT=/data/memx RUN_ROOT=/output/memx-train
make report RUN_ROOT=/output/memx-train
```

If the vendor registers PCCL under a compatibility name, set it explicitly, for example
`RATEMEM_DIST_BACKEND=vendor_pccl`. An explicit PPU request never falls back to CPU.

The exact 1/8/16-card validation sequence is in
[`docs/runbooks/company-ppu.md`](docs/runbooks/company-ppu.md). Those hardware gates are not marked
passed until their commands run on real company PPU-ZW810E devices.

## Data policy

Raw datasets and frozen model weights are not stored in Git. Checked-in manifests bind the dataset
identity, immutable revision, SPDX license, disjoint train/validation/test concepts, and prepared
content hashes. Production manifests will support public sources and company mirrors through
`MEMX_DATA_MIRROR` and `MEMX_MODEL_MIRROR` without changing the locked identities.

The current runnable profile is:

```text
configs/data/smoke.yaml
configs/experiments/smoke.yaml
```

The earlier `configs/pilot/subjects200k-held-in.json` remains an eight-row engineering-only cache
contract and must not be used as publication evidence.

## Development verification

```bash
uv sync --all-extras --frozen
uv run ruff check src tests
uv run mypy src/ratemem
uv run pytest -q -m 'not paid_modal and not real_sana and not cuda and not ppu'
bash -n scripts/*.sh
git diff --check
```

The historical TensorFlow/GAN files remain at the repository root for provenance. New development
lives under `src/ratemem`, `configs`, `scripts`, `tests`, and `docs`.
