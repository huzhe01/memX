# memX Company PPU Pipeline Design

**Status:** Approved for implementation

**Date:** 2026-08-30

**Target:** A fresh company machine can clone `huzhe01/memX`, prepare pinned data, run a
deterministic smoke experiment, launch resumable RateMem training on PPU-ZW810E, evaluate the
result, and render a machine-readable report without depending on Modal.

## 1. Scope and delivery order

The repository already contains well-tested byte accounting, lifecycle replay, SANA dynamic atom
adapters, support amortization, one-timestep flow training, secure checkpoint serialization, and a
strict single-L40S Modal engineering pilot. It does not yet contain a provider-neutral scientific
training application. The company pipeline is therefore an additive execution surface; the Modal
pilot remains unchanged evidence and the historical TensorFlow code remains provenance.

The work is split into four independently testable releases:

1. **Turnkey execution foundation:** runtime discovery, immutable run configuration, deterministic
   local fixture, data preparation, resumable training loop, evaluation/reporting, container,
   launch scripts, Make targets, and operator documentation.
2. **Production RateMem learner:** the learned base quantizer, shared packet dictionary, hard/STE
   codec, causal utility/controller, and SANA training session specified in the existing learned
   method plan.
3. **Scientific experiment suite:** frozen public data pools, matched baselines, lifecycle traces,
   metrics, statistics, ablations, and immutable result releases specified in the existing
   scientific and baseline plans.
4. **Paper release:** artifact-generated tables and figures, manuscript, supplement, and submission
   audit specified in the existing paper plan.

Release 1 must be genuinely executable before Release 2 begins. A successful fixture run proves
orchestration, not scientific quality. A PPU run is reported as validated only after it executes on
real ZW810E hardware.

## 2. User-facing contract

From the repository root, the stable operator surface is:

```bash
make bootstrap
make data DATA_ROOT=/data/memx PROFILE=smoke
make smoke DEVICE=cpu DATA_ROOT=/data/memx
make train DEVICE=ppu WORLD_SIZE=8 DATA_ROOT=/data/memx RUN_ROOT=/output/memx
make evaluate DEVICE=ppu WORLD_SIZE=8 DATA_ROOT=/data/memx RUN_ROOT=/output/memx
make report RUN_ROOT=/output/memx
```

`make bootstrap` never replaces the vendor PPU build of PyTorch. The PPU container begins with a
vendor image and installs the project plus pinned non-framework dependencies. Startup verifies the
actual torch version, accelerator count, collective backend, BF16 support, and a forward/backward
collective smoke before training.

All commands accept environment-variable overrides suitable for a company queue. They do not
assume Slurm, Kubernetes, or one internal scheduler. `scripts/launch_train.sh` maps the conventional
`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `WORLD_SIZE` variables to `torchrun`.

## 3. Runtime architecture

`ratemem.runtime` provides one immutable runtime description with these accelerator kinds:

- `cpu`: `torch.device("cpu")`, Gloo when distributed;
- `nvidia`: CUDA device API, NCCL by default;
- `ppu`: the vendor PyTorch CUDA-compatible device API, PCCL by default.

Detection never relies on a GPU marketing string alone. An operator may select `DEVICE=ppu`, and
the preflight must then observe accelerator availability, a positive device count, the requested
world size, and an available collective backend. `RATEMEM_DIST_BACKEND` is an explicit override for
vendor builds that register PCCL through a compatibility name. There is no silent CPU fallback.

The runtime record captures torch, driver/runtime, device name, device count, rank topology,
collective backend, BF16 capability, and relevant PPU SDK version variables without recording
credentials or the complete process environment.

## 4. Data architecture

Every dataset has a checked-in YAML manifest containing a schema version, logical name, immutable
upstream revision, license identifier, split roles, and source descriptors. A source descriptor is
either a Hugging Face dataset plus immutable revision or a regular file plus HTTPS URL, byte count,
and SHA-256. Downloading writes to a same-filesystem staging path, validates the complete payload,
and atomically publishes it. Existing files are accepted only after revalidation.

`MEMX_DATA_MIRROR` and `MEMX_MODEL_MIRROR` replace only the source base; they cannot alter expected
hashes or revisions. Raw datasets and frozen model weights are excluded from Git. A deterministic,
generated fixture is part of the code path so CI and `make smoke` can exercise the entire pipeline
without network access or redistribution ambiguity.

Prepared data uses a versioned episode index. Each record binds concept ID, support paths, query
path, prompts, source identity, and content hashes. Preparation rejects path traversal, duplicate
sample identities across locked splits, missing licenses, non-RGB images, and changed source
payloads.

## 5. Training architecture

Release 1 introduces a small deterministic fixture learner behind the same `Experiment` protocol
used by the production SANA learner. This is not a surrogate scientific method: it exists to verify
configuration, distributed sampling, optimization, checkpoint/resume, metrics, and artifact flow on
CPU and PPU before a multi-gigabyte checkpoint is loaded.

The production implementation plugs the existing SANA components into that protocol:

1. hydrate and verify pinned SANA/DINO snapshots;
2. prepare support features, text embeddings, and clean latents;
3. install dynamic q/k/v atoms and construct the support amortizer;
4. train the learned RateMem representation through bounded flow-matching steps;
5. synchronize gradients through DDP/PCCL;
6. write trainable-only checkpoints and immutable provenance;
7. materialize packet candidates and run lifecycle training/evaluation.

Configuration fixes seed, data manifest, model revisions, optimizer, steps, batch size, gradient
accumulation, precision, checkpoint cadence, metric cadence, output path, and experiment profile.
The canonical configuration hash is stored in every artifact.

## 6. Checkpoint, failure, and restart semantics

Only rank zero publishes checkpoints. Each checkpoint is a new directory containing trainable
tensors, optimizer/scaler/RNG state, a canonical JSON manifest, and checksums. Files are written to
an owned staging directory, synchronized, and atomically renamed. A `latest.json` pointer is updated
only after the checkpoint validates.

`RESUME=auto` loads the latest fully validated checkpoint. A changed configuration, source
revision, world-size contract, tensor topology, or dataset identity stops before mutation. Signals
finish the current optimizer boundary, save once, synchronize ranks, and exit nonzero with the
exact resume command. Corrupt or incomplete staging directories are ignored and preserved for
diagnosis; they are never treated as valid checkpoints.

## 7. Evaluation and reporting

Evaluation reads a frozen checkpoint and episode index without updating model or lifecycle usage.
Release 1 reports deterministic fixture loss, checkpoint identity, throughput, and exact artifact
hashes. Later releases add the locked personalization, prompt, diversity, byte, latency, energy,
lifecycle, oracle-regret, and hierarchical statistical endpoints.

`make report` reads only validated metric JSONL and manifests. It writes canonical JSON, CSV, and a
Markdown summary. It never inserts hand-entered experimental values and never labels fixture or
engineering results as publication eligible.

## 8. Verification gates

The repository has four explicit gates:

1. **CPU gate:** unit, contract, static-type, lint, offline data, train/resume/evaluate/report, and
   deterministic replay tests.
2. **Single-PPU gate:** runtime inventory, BF16 matrix operation, one forward/backward optimizer
   step, checkpoint round trip, and one collective rank.
3. **Eight-PPU gate:** PCCL all-reduce, disjoint sampler coverage, equal optimizer step, rank-zero
   artifact ownership, and resume.
4. **Sixteen-PPU gate:** topology receipt, scaling efficiency, peak memory, sustained step-time
   distribution, and failure/restart exercise.

The current development machine can certify only Gate 1. Gates 2--4 produce signed JSON receipts
on company hardware; documentation must distinguish an available command from an observed pass.

## 9. Repository acceptance criteria

Release 1 is accepted when:

- a clean clone exposes all six Make targets and `make help` documents every variable;
- `PROFILE=smoke DEVICE=cpu` completes offline from data preparation through report generation;
- training interruption followed by `RESUME=auto` produces the same final model and metrics as an
  uninterrupted fixture run;
- PPU selection fails clearly instead of falling back when PPU is absent;
- the PPU container preserves vendor torch and contains no credentials or raw datasets;
- the generic launcher emits a correct one-node or multi-node `torchrun` command;
- all existing 1,747 non-paid tests continue to pass;
- the README states which portions are runnable, which require external data/model downloads, and
  which real PPU gates remain unobserved; and
- the implementation and documentation are pushed to `huzhe01/memX` on the development branch.
