# memX company PPU runbook

This runbook brings up memX on real company PPU-ZW810E hardware in four gates. Passing the local
CPU suite or building the container is not evidence that a PPU gate passed. At the time this file
was written, every real ZW810E gate below is **尚未验证**.

## 1. Clone and build the approved image

```bash
git clone --branch codex/ratemem-implementation \
  https://github.com/huzhe01/memX.git
cd memX
docker build \
  --build-arg PPU_BASE_IMAGE=registry.company/ppu-pytorch:approved \
  -f docker/Dockerfile.ppu \
  -t memx:ppu .
```

The base image must provide Python 3.11 or 3.12, its vendor PyTorch and torchvision wheels, the PPU
driver userspace, and its registered collective backend. ZW810E vendor builds commonly expose the
card through PyTorch's CUDA-compatible API; memX intentionally uses that API and never installs a
replacement torch wheel. `requirements/ppu.txt` excludes torch and torchvision.

## 2. Prepare shared storage

Mount one data root read-only after preparation and one writable run root. For the offline gate:

```bash
make bootstrap DEVICE=ppu
make data DATA_ROOT=/data/memx PROFILE=smoke
```

For production datasets, download on a login/data worker rather than once per rank. A company mirror
may change transport location but may not change the locked revision or content hash.

Prepare the real data and model snapshots once on the shared mounts:

```bash
make data PROFILE=subjects200k DATA_ROOT=/data/memx
make models MODEL_ROOT=/models/memx
```

The data command verifies all 32 parquet files against their committed SHA-256 values. Set
`MEMX_HF_ENDPOINT` before both commands when using an internal Hugging Face-compatible mirror.
The locked training profiles are `sana-ratemem`, `sana-ratemem-seed29`, and
`sana-ratemem-seed43`; give each profile its own run root.

## 3. Gate A: one real PPU

```bash
make smoke DEVICE=ppu \
  DATA_ROOT=/data/memx \
  RUN_ROOT=/output/memx/gate-1

make train DEVICE=ppu WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 \
  PROFILE=sana-ratemem \
  DATA_ROOT=/data/memx \
  MODEL_ROOT=/models/memx \
  RUN_ROOT=/output/memx/gate-1-train
```

Acceptance requires runtime preflight, BF16 capability, forward/backward, optimizer update,
checkpoint round trip, evaluation, and report generation. The current fixture itself uses FP32 so
that its cross-device orchestration comparison remains stable; the runtime preflight still requires
BF16 before production PPU training.

## 4. Gate B: eight real PPUs on one node

```bash
make train DEVICE=ppu WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 \
  PROFILE=sana-ratemem \
  DATA_ROOT=/data/memx \
  MODEL_ROOT=/models/memx \
  RUN_ROOT=/output/memx/gate-8

make evaluate DEVICE=ppu WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 \
  PROFILE=sana-ratemem \
  DATA_ROOT=/data/memx \
  MODEL_ROOT=/models/memx \
  RUN_ROOT=/output/memx/gate-8
```

Acceptance requires collective initialization, disjoint rank sampling, equal optimizer-step
completion, one rank-zero checkpoint stream, a valid final checkpoint, and identical trained
parameters on all ranks. Inspect the collective names exposed by the vendor build:

```bash
python3 - <<'PY'
import torch
for name in ("pccl", "nccl", "flagcx"):
    print(name, torch.distributed.is_backend_available(name))
PY
```

If the CUDA-compatible image exposes `nccl`, or a vendor collective is registered under another
name, export that reviewed name before bootstrap and launch, for example
`RATEMEM_DIST_BACKEND=nccl`. Do not use the variable to name an unavailable backend or bypass
preflight.

## 5. Gate C: sixteen real PPUs

One 16-card node:

```bash
make train DEVICE=ppu WORLD_SIZE=16 LOCAL_WORLD_SIZE=16 \
  PROFILE=sana-ratemem \
  DATA_ROOT=/data/memx \
  MODEL_ROOT=/models/memx \
  RUN_ROOT=/output/memx/gate-16
```

Two 8-card nodes use the same values except for `NODE_RANK` and a reachable coordinator:

```bash
make train DEVICE=ppu WORLD_SIZE=16 LOCAL_WORLD_SIZE=8 NNODES=2 \
  PROFILE=sana-ratemem \
  NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
  DATA_ROOT=/data/memx MODEL_ROOT=/models/memx RUN_ROOT=/output/memx/gate-16

make train DEVICE=ppu WORLD_SIZE=16 LOCAL_WORLD_SIZE=8 NNODES=2 \
  PROFILE=sana-ratemem \
  NODE_RANK=1 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
  DATA_ROOT=/data/memx MODEL_ROOT=/models/memx RUN_ROOT=/output/memx/gate-16
```

Both nodes must see the same prepared data and run root. Schedule ranks according to the company's
ICN topology. Record device ordering, SDK/driver/PCCL versions, peak memory, per-step timing, and
node/rank mapping alongside the run.

## 6. Restart exercise

After a checkpoint boundary, stop the queue task and relaunch with exactly the same roots and locks:

```bash
make train DEVICE=ppu WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 RESUME=auto \
  PROFILE=sana-ratemem \
  DATA_ROOT=/data/memx \
  MODEL_ROOT=/models/memx \
  RUN_ROOT=/output/memx/gate-8
```

Changing the config, prepared dataset, rank topology contract, or checkpoint files must fail before
an optimizer update. Preserve failed staging directories for diagnosis.

## 7. Current evidence boundary

The smoke and real SANA/RateMem engineering profiles both record `publication_eligible=false`.
The latter is a real training path, but publication claims still require all three locked seeds,
matched baselines, lifecycle replay, frozen evaluation pools and a statistical release. Until this
runbook is executed on 真实 ZW810E hardware, the repository must say the PPU gates are 尚未验证.

## 8. Local release-one verification receipt

The provider-neutral release-one surface was verified on CPU on 2026-08-30. The following gates
completed from a clean temporary data/run root:

- `make bootstrap DEVICE=cpu` returned a passed Gloo preflight receipt.
- `make data`, `make smoke`, standalone `make train`, `make evaluate`, and `make report` completed
  in sequence.
- The deterministic fixture produced model SHA-256
  `262fdac1d6c77f79aba3f67dc6625787fba13cc32fd91ea9da665154ad33c5fb` and retained
  `publication_eligible=false`.
- Ruff passed, mypy reported no issues in 111 source files, and pytest reported
  `2049 passed, 1 skipped, 6 deselected`.
- Bash syntax, the frozen uv lock, Git whitespace, and credential scans across 342 repository files and
  the full reachable Git history passed.

This receipt covers orchestration only. No PPU-ZW810E, PCCL, throughput, memory, BF16 numerical,
multi-node, scientific dataset, baseline, or publication-result gate was executed locally.
