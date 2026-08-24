# RateMem-DiT Master Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. The companion
> plans contain the test-first implementation steps; this file owns only cross-plan ordering and
> release gates.

**Goal:** Execute the RateMem-DiT rewrite without spending compute before the engineering and
scientific contracts are testable, without letting later plans recreate earlier scaffolding, and
without allowing unverified measurements into the CVPR manuscript.

**Architecture:** Six companion plans share one Python 3.11 `uv` project. The core plan owns the
package and exact byte semantics. SANA integration owns the frozen-backbone adapter path and the
single engineering pilot. The scientific plan owns immutable data, trace, evaluator, authorization,
and release locks. The matched-baseline plan supplies real comparator implementations before the
baseline lock is sealed. The learned-method plan supplies the proposed codec, controller, and
training path. The paper plan reads only a checksummed scientific release and imports a new local
Overleaf project, leaving the 2020 source untouched.

**Tech Stack:** Python 3.11, `uv`, PyTorch, Diffusers/SANA-1.5, canonical CBOR, Pydantic, pytest,
Modal, CVPR LaTeX, Poppler, qpdf

---

## Fixed ownership

| Owner plan | Exclusive responsibility |
|---|---|
| `2026-08-24-ratemem-core-memory.md` | `.python-version`, base `pyproject.toml`, `uv.lock`, package skeleton, canonical state bytes, packet store, coverage oracle, snapshot allocator, lifecycle primitives |
| `2026-08-24-ratemem-sana-modal-pilot.md` | SANA q/k/v atom injection, support amortizer, flow step, trainable-only checkpoint, engineering-pilot artifacts and one-shot Modal runner |
| `2026-08-24-ratemem-scientific-evaluation.md` | Dataset/trace/evaluator/compute locks, replay protocol, metrics, statistics, falsification gates, paper release |
| `2026-08-24-ratemem-matched-baselines.md` | Executable matched controls, common adapter protocol, exact-byte ledger adapters, fidelity receipts |
| `2026-08-24-ratemem-learned-method-training.md` | Learned base/packet codec, immutable shared bundle proposal, causal controller, utility calibration, bounded sequential training |
| `2026-08-24-ratemem-cvpr-paper.md` | New English manuscript, vector figures, artifact-driven tables, PDF QA, new-project Overleaf import |

No companion plan may replace another plan's dependencies, CLI entry points, schemas, or tests.
Every additive edit to `pyproject.toml` is followed by `uv lock`, `uv sync --all-extras --frozen`,
and the predecessor smoke tests.

## Exact execution order

- [ ] **Gate 0: Preserve provenance and enter the isolated branch**

  Work only in the dedicated `codex/ratemem-implementation` worktree. Treat
  `/home/ubuntu/memory-metagan-original/`, local Overleaf project
  `6a8b44fb070db27221ef64a0`, and the legacy TensorFlow/GAN files as read-only provenance. Run
  `git status --short` before each phase and stop on overlapping user changes.

- [ ] **Gate 1: Complete the core-memory plan**

  Execute every task in `2026-08-24-ratemem-core-memory.md`. Required receipt: clean Ruff/mypy,
  all CPU core tests, randomized exact-byte lifecycle tests, exhaustive small-instance allocator
  checks, and the frozen core interface document. No provider authentication or paid compute is
  allowed in this phase.

- [ ] **Gate 2: Complete free SANA engineering work**

  Execute the SANA/Modal plan through all local tests, mock artifacts, static source contracts, and
  guarded runbook construction. Verify the pinned SANA layout and Subjects200K crop contract. Do
  not run the paid script until its workspace, cap, cost-reservation, and source-hash gates all pass.

- [ ] **Gate 3: Run at most one authorized engineering-pilot submission**

  Use exactly one operator-selected Modal workspace. Before any paid action, verify its hard
  Workspace usage budget is exactly USD 28.00 and that known metered use plus open worst-case
  reservations plus the new bound is at most USD 27.00. Submit exactly one synchronous `.remote()`
  invocation with `retries=0`, `max_containers=1`, one L40S, and no fan-out, deployment, detached
  execution, workspace rotation, or fallback. Modal may reschedule a container after infrastructure
  failure; record observed attempts and actual metered reconciliation, and rely on the verified USD
  28 Workspace budget as the hard outer stop. If billing lags, keep the reservation pending and do
  not launch again.

- [ ] **Gate 4: Build scientific prerequisites without opening the final test**

  Execute scientific-evaluation Tasks 1--7: canonical records, dataset inventory, immutable pools,
  leakage audit, visible traces, encrypted final trace, evaluator formulas, budgets, and power plan.
  The final-test payload remains encrypted. Synthetic fixtures may verify code but cannot satisfy a
  scientific lock.

- [ ] **Gate 5: Implement and audit every matched baseline**

  Execute `2026-08-24-ratemem-matched-baselines.md` against the frozen protocol from Gate 4. Every
  eligible method must implement the same provider-neutral event, shared-input, search-budget,
  adapter-site, and serialized-byte contracts. Pre-lock checks use immutable synthetic bundles; the
  real learned dictionary and shared candidate stream do not exist yet. Context-only methods remain
  clearly labeled and cannot serve as the strongest matched comparator. If a real-checkpoint
  fidelity contract cannot run on CPU, use
  only the narrow pre-lock `baseline_fidelity` permit owned by scientific Task 8: it is bound to the
  dataset lock, comparator catalog, fidelity policy, source hashes, held-in/calibration inputs, one
  explicit workspace, and one reconciled reservation. It cannot read validation/final traces,
  perform method selection, or emit claim metrics. Complete scientific Task 8 only after
  implementation, source-fidelity, synthetic ledger receipts, SANA-backbone binding, and frozen
  search-policy evidence exist for every required control. SDXL-native rows remain contextual because
  this route has no SDXL RateMem port; a failed matched-required SANA port blocks the primary claim
  rather than changing backbones. Do not require tuned validation outcomes or learned-method weights
  in the baseline lock.

- [ ] **Gate 6: Implement the learned method and synthetic release gate**

  Execute `2026-08-24-ratemem-learned-method-training.md` through its CPU synthetic integration
  gate. Freeze the learned dictionary identity before scientific replay. Demonstrate genuine shared
  packet reuse and a nonseparable synthetic advantage over the private/separable codec; if that
  contract fails, stop and revise the method rather than opening paid scientific compute.

- [ ] **Gate 7: Authorize each result-bearing paid scientific phase separately**

  Return to scientific-evaluation Task 9. A fresh authorization names one phase, one explicit
  workspace, exact lock hashes, one commit, one source diff, rates, and a worst-case bound. The
  engineering-pilot and pre-lock `baseline_fidelity` permits are invalid here. Never automatically
  discover, rotate, or fall back across the supplied workspaces. Reconcile a phase before reserving
  another one, and stop when the USD 27 admission inequality would fail or the hard USD 28 outer cap
  is reached. A later workspace can enter scope only through a separate explicit author decision,
  a new named non-global profile and selection file, fresh proof of its own USD 28 cap/current usage,
  and a distinct workspace partition in the append-only ledger; no running or failed phase may
  trigger that switch.

- [ ] **Gate 8: Train, replay, analyze, and attempt falsification**

  Execute the authorized learned training and baseline calibration phases, then scientific Tasks
  10--18. Materialize the real target-code/candidate stream once from the frozen learned checkpoint,
  train and tune every eligible baseline under the locked search policy, freeze those receipts, and
  replay all methods on identical immutable traces. Compute paired hierarchical statistics,
  apply multiplicity correction, run the prespecified human study if authorized, and evaluate the
  hard scientific gates. Consume the encrypted final trace only once after every prerequisite is
  frozen. Publish a paper release only if its schema, hashes, lineage, and gates validate. Negative
  results stay in the release; no threshold or method selection may be changed after final opening.

- [ ] **Gate 9: Write and render the paper from evidence**

  Execute `2026-08-24-ratemem-cvpr-paper.md`. Draft prose and vector method figures may start early,
  but empirical macros, tables, curves, qualitative panels, abstract claims, and conclusion wording
  must be generated from the validated release. Recheck the official target-year kit at build time,
  compile the main paper and supplement, render every PDF page to images, inspect every page, and
  record a PDF-bound QA receipt.

- [ ] **Gate 10: Back up and import as a new Overleaf project**

  Download and hash the original local project, import the rewritten source as a separate project,
  compile it there, download the result, and prove the original manifest is unchanged. Never
  overwrite the 2020 project. The final handoff includes source commit, lock hashes, artifact release,
  local PDF, imported project identifier, compile receipt, and remaining limitations.

## Global stop conditions

Stop immediately and preserve evidence when any of the following occurs:

- a credential value appears in a repository file, command argument, log, environment dump, or
  artifact;
- exact decoded state length exceeds the configured budget at any event;
- a source revision, license, checkpoint hash, dataset row identity, or evaluator formula cannot be
  sealed;
- a paid job lacks a fresh phase-specific authorization and worst-case reservation;
- billing is pending, the USD 27 admission inequality fails, or the selected Workspace budget is not
  visibly and exactly USD 28.00;
- a baseline uses a different event stream, search budget, model interface, or byte accounting;
- final-test data becomes visible before the one-time freeze permit;
- a paper number is not traceable to the checksummed scientific release;
- CVPR 2027 rules differ from the pinned fallback and the source has not been updated and re-audited.

## Master acceptance command

After all companion plans are complete, run from the worktree root:

```bash
uv sync --all-extras --frozen
uv run ruff check .
uv run mypy src/ratemem
uv run pytest -q
uv run ratemem-eval gates evaluate \
  --evaluation-lock configs/scientific/evaluation-lock.yaml \
  --artifact-index artifacts/scientific/final/artifact-index.json \
  --statistics artifacts/scientific/final/claim-statistics.json \
  --output artifacts/scientific/final/gates.json
test -n "${PAPER_ID:-}"
make -C paper submission PAPER_ID="$PAPER_ID"
```

Expected: every command exits 0; the scientific gate report and paper manifest name the same lock
hashes and commit; the PDF-bound visual-QA receipt covers every rendered page. This command cannot
replace the per-phase paid-compute authorizations, manual page inspection, or original-project
unchanged proof.
