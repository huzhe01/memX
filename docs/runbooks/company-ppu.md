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
driver userspace, and PCCL. `requirements/ppu.txt` deliberately excludes torch and torchvision.

## 2. Prepare shared storage

Mount one data root read-only after preparation and one writable run root. For the offline gate:

```bash
make bootstrap DEVICE=ppu
make data DATA_ROOT=/data/memx PROFILE=smoke
```

For production datasets, download on a login/data worker rather than once per rank. A company mirror
may change transport location but may not change the locked revision or content hash.

## 3. Gate A: one real PPU

```bash
make smoke DEVICE=ppu \
  DATA_ROOT=/data/memx \
  RUN_ROOT=/output/memx/gate-1

make train DEVICE=ppu WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 \
  DATA_ROOT=/data/memx \
  RUN_ROOT=/output/memx/gate-1-train
```

Acceptance requires runtime preflight, BF16 capability, forward/backward, optimizer update,
checkpoint round trip, evaluation, and report generation. The current fixture itself uses FP32 so
that its cross-device orchestration comparison remains stable; the runtime preflight still requires
BF16 before production PPU training.

## 4. Gate B: eight real PPUs on one node

```bash
make train DEVICE=ppu WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 \
  DATA_ROOT=/data/memx \
  RUN_ROOT=/output/memx/gate-8

make evaluate DEVICE=ppu WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 \
  DATA_ROOT=/data/memx \
  RUN_ROOT=/output/memx/gate-8
```

Acceptance requires PCCL initialization, disjoint rank sampling, equal optimizer-step completion,
one rank-zero artifact stream, a valid final checkpoint, and identical model hashes observed by all
ranks. If PCCL is registered under another name, export the reviewed compatibility name:

```bash
export RATEMEM_DIST_BACKEND=vendor_pccl
```

Do not use this variable to name an unavailable backend or to bypass preflight.

## 5. Gate C: sixteen real PPUs

One 16-card node:

```bash
make train DEVICE=ppu WORLD_SIZE=16 LOCAL_WORLD_SIZE=16 \
  DATA_ROOT=/data/memx \
  RUN_ROOT=/output/memx/gate-16
```

Two 8-card nodes use the same values except for `NODE_RANK` and a reachable coordinator:

```bash
make train DEVICE=ppu WORLD_SIZE=16 LOCAL_WORLD_SIZE=8 NNODES=2 \
  NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
  DATA_ROOT=/data/memx RUN_ROOT=/output/memx/gate-16

make train DEVICE=ppu WORLD_SIZE=16 LOCAL_WORLD_SIZE=8 NNODES=2 \
  NODE_RANK=1 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
  DATA_ROOT=/data/memx RUN_ROOT=/output/memx/gate-16
```

Both nodes must see the same prepared data and run root. Schedule ranks according to the company's
ICN topology. Record device ordering, SDK/driver/PCCL versions, peak memory, per-step timing, and
node/rank mapping alongside the run.

## 6. Restart exercise

After a checkpoint boundary, stop the queue task and relaunch with exactly the same roots and locks:

```bash
make train DEVICE=ppu WORLD_SIZE=8 LOCAL_WORLD_SIZE=8 RESUME=auto \
  DATA_ROOT=/data/memx \
  RUN_ROOT=/output/memx/gate-8
```

Changing the config, prepared dataset, rank topology contract, or checkpoint files must fail before
an optimizer update. Preserve failed staging directories for diagnosis.

## 7. Current evidence boundary

The checked-in smoke profile is an orchestration test and always records
`publication_eligible=false`. Real RateMem/SANA result claims require the later locked production
configurations, datasets, matched baselines, three training seeds, lifecycle replay, and statistical
release. Until this runbook is executed on 真实 ZW810E hardware, the repository must say the PPU
gates are 尚未验证.
