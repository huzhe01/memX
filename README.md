# memX / RateMem-DiT

memX is the reproducible implementation that supersedes the unpublished 2020
*Memory-MetaGAN* prototype. The new research direction studies byte-bounded, shared progressive
adapter memory for optimization-free image personalization on a frozen diffusion transformer.

The repository contains three deliberately separate execution surfaces:

- a locally verified, end-to-end orchestration fixture covering data preparation, distributed
  runtime policy, optimization, atomic checkpoint/resume, evaluation, and reporting; and
- a real SANA-1.5/Subjects200K RateMem path with pinned inputs, bounded sequential meta-training,
  distributed gradient averaging, checkpoint/restart, and held-out engineering evaluation; and
- the guarded historical single-L40S Modal pilot.

Both runnable paths emit `publication_eligible=false` until the real PPU gates, three locked seeds,
matched lifecycle baselines, and frozen scientific evaluation have completed. A successful training
job is evidence that the implementation ran, not a CVPR result by itself.

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
make data PROFILE=smoke DATA_ROOT=/data/memx
make smoke DEVICE=ppu DATA_ROOT=/data/memx RUN_ROOT=/output/memx-smoke
```

If the vendor registers PCCL under a compatibility name, set it explicitly, for example
`RATEMEM_DIST_BACKEND=vendor_pccl`. An explicit PPU request never falls back to CPU.

The exact smoke and real 1/8/16-card validation sequence is in
[`docs/runbooks/company-ppu.md`](docs/runbooks/company-ppu.md). Those hardware gates are not marked
passed until their commands run on real company PPU-ZW810E devices.

## Real SANA/RateMem training

Prepare the 32 pinned Subjects200K parquet shards (10,553,550,156 bytes) and both pinned model
snapshots once on shared storage:

```bash
make data PROFILE=subjects200k DATA_ROOT=/data/memx
make models MODEL_ROOT=/models/memx
```

An internal Hugging Face-compatible mirror can be selected without changing any locked identity:

```bash
export MEMX_HF_ENDPOINT=https://huggingface.company.example
```

Then launch the locked seed-17 run on eight visible cards:

```bash
make train PROFILE=sana-ratemem DEVICE=ppu \
  WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 \
  DATA_ROOT=/data/memx MODEL_ROOT=/models/memx \
  RUN_ROOT=/output/memx/seed-17

make evaluate PROFILE=sana-ratemem DEVICE=ppu \
  WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 \
  DATA_ROOT=/data/memx MODEL_ROOT=/models/memx \
  RUN_ROOT=/output/memx/seed-17
```

`RESUME=auto` restores model, optimizer and CPU RNG state after validating the experiment and
dataset identities. The other two locked runs use `PROFILE=sana-ratemem-seed29` and
`PROFILE=sana-ratemem-seed43`, with separate `RUN_ROOT` values. The engineering `evaluate` command
measures held-out-concept one-timestep flow MSE. Publication metrics and matched baselines run
through the separately frozen `ratemem-eval` protocol.

## Data policy

Raw datasets and frozen model weights are not stored in Git. Checked-in manifests bind the dataset
identity, immutable revision, SPDX license, disjoint concept partition, all 32 upstream shard sizes
and SHA-256 values, and prepared snapshot identity. `MEMX_HF_ENDPOINT` changes only transport;
downloaded bytes must still match the committed hashes.

The runnable profiles are:

```text
configs/data/smoke.yaml
configs/experiments/smoke.yaml
configs/data/subjects200k.yaml
configs/experiments/sana-ratemem.yaml
configs/experiments/sana-ratemem-seed29.yaml
configs/experiments/sana-ratemem-seed43.yaml
```

The earlier `configs/pilot/subjects200k-held-in.json` remains an eight-row engineering-only cache
contract and must not be used as publication evidence.

## Development verification

```bash
uv sync --all-extras --frozen
uv run --frozen ruff check src tests
uv run --frozen mypy src/ratemem
uv run --frozen python -m pytest -q -m 'not paid_modal and not real_sana and not cuda'
bash -n scripts/*.sh
git diff --check
```

The historical TensorFlow/GAN files remain at the repository root for provenance. New development
lives under `src/ratemem`, `configs`, `scripts`, `tests`, and `docs`.
