# RateMem-DiT: Design Specification

**Status:** Revised draft for author approval

**Date:** 2026-08-24

**Target:** CVPR 2027, using the latest published CVPR format until the 2027 rules are released
**Supersedes:** The unpublished 2020 manuscript *Memory-MetaGAN: A Memory-based Few-shot GAN*

## 1. Research objective

The project will replace the incomplete TensorFlow/GAN prototype and rewrite the manuscript around this question:

> Can personalized adapter states be learned, stored, updated, and evicted jointly under a hard per-user byte budget, while preserving a better generation-quality--storage--latency frontier than independently generated adapters or feature caches?

The working title is **RateMem-DiT: Shared Progressive Adapter Packets for Bounded Optimization-Free Personalization**.

The intended contribution is a **capacity-coupled personalization memory**, not any ingredient below in isolation. The paper explicitly does not claim invention of:

- support-to-adapter or hypernetwork prediction;
- low-dimensional/shared LoRA bases;
- zero-step personalization;
- cosine retrieval, LRUA, LRU, or CRUD operations;
- episodic/semantic memory for diffusion;
- timestep/layer-dependent LoRA routing;
- Gaussian-mixture sampling; or
- multi-concept adapter composition.

Those mechanisms all have close prior art. Shared LoRA subspaces, sparse bank codes, residual compression, and mixed-bit adapter quantization are also prior art and are not contributions. The candidate contribution is instead the learned coupling among:

1. a frozen diffusion-transformer backbone;
2. an amortized support-to-adapter predictor;
3. a **hard total online-state budget in bytes**, not merely a fixed number of independent concept slots;
4. a progressive per-concept base representation plus optional enhancement packets;
5. immutable enhancement packets that may be referenced by and improve several concepts, so packet admission is nonseparable across concepts;
6. exact serialized-byte accounting for packets, incidence records, handles, and metadata;
7. a causal request-aware allocator that admits, shares, degrades, garbage-collects, or evicts packet state under nonstationary traces; and
8. a serialized personalization lifecycle benchmark with immutable read traffic and evaluation probes that cannot mutate memory state.

The algorithmic target is a shared-packet allocator with a formal guarantee for its locked surrogate objective. A separable per-concept variable-rate codec plus ordinary knapsack/LRU is only a baseline and is insufficient for a method contribution.

The decisive controls use the same amortized adapter predictor and frozen basis with: fixed-rate FIFO/LRU/LRUA caches; a progressive but unshared residual codec plus a causal size-aware cache; an offline multiple-choice-knapsack oracle; online Share-style shared-subspace compression under the same cap; and matched feature caching. If shared packets and the causal allocator do not improve the quality--bytes--request-utility frontier over the strongest of these controls, the algorithmic claim does not survive and the work is reframed as a benchmark/systems study.

All quantitative claims must be generated from validated artifacts. No result, comparison, quotation, or citation may be invented. Draft tables remain explicitly marked as placeholders until populated by scripts.

## 2. Novelty boundary and closest work

The design is intentionally scoped against the closest known systems:

- [HyperLoRA (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Li_HyperLoRA_Parameter-Efficient_Adaptive_Generation_for_Portrait_Synthesis_CVPR_2025_paper.html) already predicts coefficients over a low-dimensional LoRA basis without per-subject fine-tuning.
- HyperDreamBooth already predicts compact personalized weights and then optionally refines them.
- [VSM-Diffusion (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/17826a22eb8b58494dfdfca61e772c39-Abstract-Conference.html) already combines few-shot diffusion with episodic/semantic memory, uncertainty, consolidation, and least-used replacement.
- [DreamCache (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Aiello_DreamCache_Finetuning-Free_Lightweight_Personalized_Image_Generation_via_Feature_Caching_CVPR_2025_paper.html) caches lightweight features for tuning-free personalization.
- [Compress then Serve (ICML 2025)](https://proceedings.mlr.press/v267/gabrielsson25a.html) jointly compresses large LoRA collections into shared bases and adapter-specific scaling matrices.
- [VB-LoRA (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e0d38c676d5855bcfab7f6d29d20ad9-Abstract-Conference.html) reconstructs LoRA subvectors with sparse mixtures from a shared vector bank.
- [Share (ECCV 2026)](https://arxiv.org/abs/2602.06043) continually updates a shared LoRA subspace, reprojects old task coefficients, measures drift, and includes text-to-image adaptation and asynchronous serving.
- [MoBLoRA (ACL 2026)](https://aclanthology.org/2026.acl-long.481/) uses shared orthogonal bases, task mixing matrices, and residual consolidation for continual multimodal tuning.
- CF-STAR, RQT, SineLoRA-Delta, and LoRAQuant already cover sparse residuals, progressive or mixed-precision adapter compression, and explicit storage trade-offs.
- ADA already studies fixed-pool adapter consolidation, while standard online caching and rate allocation already cover separable request-weighted eviction/bit assignment.
- ConceptGuard, Continual Diffusion/C-LoRA, Mining Your Own Secrets, and Concept Neuron Selection study actual sequential parameter interference.
- AutoLoRA, Mod-Adapter, LoRAverse, and related work retrieve, route, or compose adapters.

RateMem therefore treats the amortizer, static adapter basis, progressive residual codec, quantizer, LRUA bookkeeping, and composition operators as borrowed infrastructure. Its falsifiable novelty hypothesis is:

> Jointly useful enhancement packets plus causal packet allocation improve request-weighted personalization quality at a fixed total state budget compared with separable progressive coding, shared-subspace compression, or cached features followed by independent replacement/rate policies.

The packet mechanism must be nonseparable: retaining one packet can benefit several active concepts, and removing it can degrade several concepts. The paper must include either a proved approximation/competitive/regret statement for the prespecified surrogate or sufficiently strong nonstationary-trace evidence to position the result as a systems algorithm. Merely applying a standard multiple-choice knapsack solver to independent concept bitstreams is a no-go.

This is a hypothesis, not a guaranteed priority claim. The literature search must be refreshed at experiment freeze and again before submission.

## 3. Problem definition

A deployment workload is a serialized trace

\[
\mathcal{E}=(e_1,\ldots,e_T),
\]

where each event is one of:

- `CREATE(S_t,d_t)`: insert a new concept from 1--10 references and return an opaque handle;
- `UPDATE(h_t,S_t)`: add evidence to a known active handle;
- `READ(h_t,p,z)`: generate for an active handle under prompt `p` and seed `z`;
- `DELETE(h_t)`: invalidate the handle and release its per-concept state; or
- `PROBE(snapshot,h_t,p,z)`: evaluate a copied state without changing usage statistics.

The total mutable personalization state is constrained by

\[
\operatorname{bits}(M) \le B.
\]

`B` includes per-concept base codes, enhancement-packet payloads, packet hashes/keys, concept-to-packet incidence and gains, any optional tokens, handles, usage and age metadata, allocator/controller state, checksums, and alignment overhead. Shared trained weights and frozen codec dictionaries are reported separately and amortized at each tested active-set size.

Normal generation uses an exact opaque handle so that naming/re-identification is not a hidden source of failure. Exemplar-based autonomous recognition is a separate secondary protocol. The handle table is part of `B` and cannot grow without bound.

At deployment, foundation-model and learned controller parameters are frozen. A `CREATE` or `UPDATE` performs only forward computation and bounded memory-state mutation; it does not run gradient descent for that user concept.

## 4. Method

### 4.1 Frozen backbone and amortized adapter target

The primary engineering target is the public SANA-1.5 1.6B 1024-pixel Diffusers checkpoint. Its denoiser, text encoder, and VAE remain frozen. A support encoder consumes an unordered set of image features plus a concept description and predicts an uncompressed target adapter code

\[
c_t^\star = F_{\psi}(S_t,d_t).
\]

The first public-data implementation uses a frozen visual encoder followed by a small Set Transformer. Support order is randomly permuted during training and tested explicitly. Predicting `c_t^star` is a HyperLoRA-style amortization component and is not claimed as novel.

For a clean causal comparison, every RateMem and independent-cache baseline receives the **same** `F_psi`, static adapter basis, training data, and optimization budget. A stateless predictor that recomputes from still-available support images is also reported as an oracle-latency/control condition, not as a deployable no-image-storage system.

### 4.2 Dynamic adapter execution

Ordinary PEFT LoRA is used for static baselines and checkpoint interchange only. It cannot express arbitrary per-example coefficients over several atoms. RateMem therefore uses a contract-tested `DynamicAtomLinear` wrapper on selected SANA projections:

\[
y = Wx + \sum_{m=1}^{A}\alpha_m B_m A_m x.
\]

The low-rank result is accumulated directly; a dense `Delta W` is never materialized. The first target set is the official SANA LoRA set `to_q`, `to_k`, and `to_v`. Output projections are an ablation only after contract tests pass.

For the 1.6B SANA configuration with 20 blocks, width 2240, and both self- and cross-attention, adapting q/k/v covers 120 projections. The atom parameter count is

\[
120 \times 4480 \times r \times A = 537{,}600rA.
\]

The pilot fixes `r=4` and `A=4`, or about 8.60M trainable atom parameters. Coefficients are initially constant across denoising timesteps and entities. Timestep-, layer-, spatial-, or entity-specific routing is outside the first pilot and cannot become a contribution without new evidence beyond existing routing work.

The adapter code has one coefficient per block, attention type, q/k/v projection, and atom:

\[
\alpha\in\mathbb{R}^{20\times2\times3\times A},
\qquad c=\operatorname{vec}(\alpha)\in\mathbb{R}^{d},
\qquad d=120A.
\]

Thus the pilot code has `d=480`. The support predictor emits FP32 logits; a learned per-projection positive scale followed by `tanh` bounds the coefficients, and execution casts them to BF16. The independent uncompressed-cache baseline serializes the resulting coefficient vector in BF16. RateMem operates on the normalized coefficient vector and uses explicitly versioned blockwise symmetric quantizers; every block stores its integer payload and scale. Scientific atom counts are selected only after dataset/evaluation lock and include larger prespecified values when the public-data amortizer needs more capacity. The static atom dictionary is shared trained state, not online memory, and is reported separately.

### 4.3 Progressive shared-packet memory

An independent cache stores one complete predicted code per concept. A separable progressive baseline stores a concept-specific base code and a private sequence of residual refinements. RateMem adds a packet layer that can deduplicate and share useful residual directions across concepts.

For active concept `i`, the reconstructed adapter code is

\[
\hat c_i(M)=D(q_i^0)
+\sum_{p\in\mathcal P(M)} m_{i,p}\,\hat a_{i,p}\,e_p,
\]

where `q_i^0` is a small mandatory base code, `e_p` is an immutable quantized enhancement packet stored once, `m_{i,p}` is a concept-to-packet incidence bit, and `a_i,p` is a compact quantized gain. The offline codec dictionary used to form candidate residual packets is frozen after meta-training; it is not mutable online state. A packet enters online memory only when at least one active concept references it.

Candidate packets are derived forward-only from the target-code residual. They are content-addressed by a canonical quantized payload hash. Exact payload matches are deduplicated; near matches may be shared only when a contract-tested error bound holds for every dependent concept. Packet payloads never mutate after admission. If a better packet is needed, the system adds a new version and atomically redirects chosen incidence records, preventing silent drift from in-place basis updates.

The serialized state obeys

\[
\sum_{i\in\mathcal A}
\operatorname{bits}(q_i^0,h_i,\text{metadata}_i)
+\sum_{p\in\mathcal P(M)}\operatorname{bits}(e_p,\text{hash}_p)
+\sum_{(i,p):m_{i,p}=1}\operatorname{bits}(i,p,\hat a_{i,p})
\le B.
\]

One retained packet may improve several concepts, while one removed packet may degrade several concepts. This makes packet admission nonseparable and distinguishes the proposed allocator from assigning an independent bit count to every concept. Whole-concept eviction is represented by removing its base record and all incidences; it is a fallback, not a separately novel operation.

The first implementation uses fixed-size code groups and residual-vector-quantization packets because they are auditable. Sparse residual, mixed-bit, CF-STAR/RQT/SineLoRA-style codecs and Share/VB-LoRA representations are required baselines, not novelty claims.

### 4.4 Causal packet allocation

Each create or update proposes one base record and a bounded set of enhancement packets/incidences. Allocation has two explicitly separated layers. An outer, non-theorem size-aware policy admits/rejects complete base records and reserves their exact bytes. Given the resulting fixed active cohort `A_t`, the theorem-bearing inner allocator receives residual packet capacity

\[
b_t=B-\sum_{i\in\mathcal A_t}
\operatorname{bits}(q_i^0,h_i,\text{metadata}_i).
\]

Its finite ground set `G_t` contains resident packets plus packets proposed by the current event. Ground item `p` is an immutable **packet bundle**: one payload/hash and one prespecified list `A_{t,p}` of nonnegative concept incidences/gains. Selecting `p` installs every incidence in that list; optional per-incidence choices are not allowed in the theorem variant. Its exact modular cost is

\[
c_{t,p}=\operatorname{bits}(e_p,\text{hash}_p)
+\sum_{i\in A_{t,p}}
\operatorname{bits}(i,p,\hat a_{i,p}).
\]

Request weights `omega_{t,i}` are nonnegative and measurable from operational reads through history `H_{t-1}` when allocation precedes event `t`; scoring probes and future reads never influence them. A calibration-checked predictor converts support uncertainty, code residuals, and bounded one-step diffusion probes into frozen nonnegative group weights `beta_{t,i,g}` and packet gains `v_{t,i,g,p}`, with `v=0` for incidences absent from `A_{t,p}`. The certified snapshot utility is

\[
F_t(X)=\sum_{i\in\mathcal A_t}\omega_{t,i}
\sum_g\beta_{t,i,g}
\min\left\{1,\sum_{p\in X}v_{t,i,g,p}\right\},
\qquad X\subseteq G_t.
\]

Because it is a nonnegative weighted sum of concave-over-modular coverage terms, `F_t` is normalized, monotone, and submodular. It remains nonseparable in packet variables because one packet can benefit several concepts for one payload cost.

The theorem-bearing allocator enumerates seed sets of up to three packets and completes each feasible seed by exact marginal-density greedy, using lazy evaluation only when it returns identical choices. Under the fixed-cohort, exact-value-oracle assumptions, it targets the standard per-snapshot guarantee

\[
F_t(X_t)\ge(1-1/e)
\max_{X\subseteq G_t:\sum_{p\in X}c_{t,p}\le b_t}F_t(X).
\]

This is causal because `G_t`, costs, gains, and weights use only current/past information, but it is **not** a competitive or dynamic-regret guarantee against a future-aware trace oracle. Whole-base admission/eviction, switching penalties, hysteresis, optional incidence dropping, and unconstrained learned distortion remain empirical outer-policy variants and are not covered by the theorem. The future-aware oracle is an upper reference. If exhaustive small-instance tests or the proof do not validate the exact implemented formulation, the theoretical claim is removed and the work falls back to a systems benchmark.

All proposals are ranked in code/packet space. Only the chosen allocation and at most one control allocation may receive a one-random-timestep denoising evaluation; these passes count against the segment-wide two-transformer-pass cap in Section 5. Full image generation is never inside an allocation decision.

The codec and distortion predictor are meta-trained with differentiable quantization surrogates and a declared temperature schedule, then evaluated with deterministic hard packets and byte-exact serialization. LRU/LRUA and size-aware caching are borrowed policies. Representation ablations fix the allocator; allocator ablations fix the encoded candidate packet stream.

### 4.5 Read, update, deletion, and autonomous lookup

Exact-handle `READ` reconstructs one code from its base record and currently resident packet incidences, then installs it in the dynamic adapter wrappers. `UPDATE` is explicitly labeled with its handle and does not require novelty detection. `DELETE` invalidates the handle, removes its base record/incidences, and garbage-collects only packets with no remaining dependents.

Deletion is an operational state-management guarantee, not machine unlearning. Shared packets used by another active concept remain resident, so deletion does not promise removal of a visual direction shared with other concepts or foundation parameters. Tests require stale-handle rejection, exact state reclamation, reference-count integrity, deterministic serialization, and no change to unrelated decoded codes beyond allocator actions explicitly triggered by the freed budget.

Autonomous exemplar lookup uses frozen-feature keys, sparse cosine addressing, and a calibrated update/allocate/reject risk curve. It is reported separately because false-update risk is irrelevant when an exact handle is supplied.

### 4.6 Secondary legacy components

The old paper's GMM idea is retained only as a supplementary ablation. If used, it models within-concept variation codes and never replaces the diffusion noise prior. Multi-concept generation is a stress test using simple adapter combination; a new composition router is not part of the core design. Downstream category augmentation is attempted only after the primary hard-budget claim passes.

These components are removed rather than promoted if they dilute the eight-page paper or fail their validation gates.

## 5. Sequential meta-training contract

Every training segment is a short lifecycle trace sampled from a frozen trace generator. It includes creates, operational reads, labeled updates, deletions, and overflow events. Event probabilities, request distributions, capacity budgets, and random seeds are stored in versioned trace manifests.

Train, validation, and test traces have disjoint concept pools, trace identifiers, and seed namespaces. Their manifests are hashed separately. The final-test manifest is generated and committed by the trace builder before comparative model development, but its event payload is kept unopened by training/model-selection code until all configuration and evaluation locks are signed off. Controller training may never replay or optimize against final test traces. After the one-time final evaluation, the complete manifest is released for reproducibility.

The bounded compute contract is:

- every training query uses one randomly sampled flow-matching timestep and one transformer forward/backward pass;
- no 18--20-step sampled image is retained in a training graph;
- full denoising is validation-only with a fixed prompt/seed cap;
- the pilot uses `K<=2`, sequence length 2, no more than two query passes per segment, and truncated BPTT length 2;
- memory updates are functional/out-of-place and state is detached at truncation boundaries;
- BF16 and activation checkpointing are enabled;
- frozen text embeddings, VAE latents, and support features are precomputed; and
- atom orthogonality is computed in factored form without constructing 2240-by-2240 products.

Scientific training uses strictly concept-disjoint meta-train, validation, and test sets. Architecture, loss-component, threshold, and margin selection stop before the final test is opened.

## 6. Dataset lock and leakage contract

No scientific result is produced until a versioned `dataset-lock.yaml` and data card specify exact revisions, licenses, concept units, counts, image statistics, captions/masks, and immutable support/query pools.

The provisional sources to audit are:

| Role | Provisional source | Intended use | Constraint |
|---|---|---|---|
| Engineering pilot | A small pinned subset of Subjects200K | Loader, one-step loss, tiny held-in fit | No scientific claim |
| Meta-training | Subjects200K (206,841 subject-consistent pairs, Apache-2.0 metadata) | Public pair-based amortizer training | Pair structure supports mainly 1-shot supervision |
| Multi-image training, if license passes | SynCD public multi-image groups | Support-set/update supervision | Must pin license and provenance before use |
| Primary 1-shot evaluation | DreamBench++ (150 references, 9 prompts each) | Human-aligned personalization and lifecycle traces | It is not a real held-out-query dataset |
| Multi-shot evaluation | Eligible CustomConcept101 concepts | Common 1/3/5-shot cohort | Only concepts with enough distinct images; maximum shot follows actual counts |
| Contamination audit | A post-checkpoint, rights-cleared multi-image concept set | Cleanest held-out evidence | Must be collected or identified before final training |
| Historical stress test | Omniglot | Cross-domain comparison with old paper | Not evidence for natural-image 1024px T2I |

Oxford Flowers, animal-face datasets, and NABirds/CUB are reserved for the optional category-augmentation track. Any use requires new class-disjoint splits; the standard Oxford Flowers split is not class-disjoint. Known overlap between CUB/ImageNet/Flickr-derived data and pretrained evaluators/backbones is disclosed rather than described as clean unseen data.

Before splitting, all sources are globally clustered for exact duplicates, crops, recompressions, burst/video neighbors, and frozen-feature near-duplicates. Entire clusters are assigned to one split. Validation/test concepts have immutable support and evaluation pools. No evaluation image or derivative, caption, mask, prompt, or label may influence training, early stopping, filtering, threshold selection, or hyperparameter search.

Prompt templates are split and frozen. Real identity names are replaced by anonymous tokens. Pretraining contamination of SANA, the support encoder, and evaluation encoders is disclosed; at least one post-checkpoint or otherwise independently controlled evaluation set is required for the strongest unseen-concept claim.

The representation used for `L_id` cannot also serve as the sole headline evaluator. Dataset lock precedes scientific architecture/loss selection, though checkpoint/API smoke tests may happen earlier.

A separate immutable `evaluation-lock.yaml` is required before any comparative validation or scientific training run. It pins evaluator names and immutable revisions, preprocessing, exact metric formulas, primary endpoints, byte budgets, workload distributions, trace hashes, generation settings, non-inferiority margins, multiplicity correction, and the target confidence-interval width/power calculation. Margins are set from published reliability evidence or a separate calibration set that is excluded from architecture/model selection; they are not selected after seeing comparative validation results. The strongest prespecified eligible control is the required comparator.

## 7. Immutable lifecycle evaluation

Every method replays the same serialized trace containing concept set, support draws, event order, operational read traffic, prompt templates, and generation seeds. Operational `READ` events update usage. All scoring uses a copied snapshot with `update_usage=false`; evaluation cannot influence eviction.

Three protocols are separated:

1. **No-pressure (`B` sufficient):** measure acquisition quality and unintended drift without forced eviction. Independent caches should have exactly zero state drift for fixed prompt/seed; RateMem must quantify any sharing-induced drift.
2. **Budget-pressure (`B` insufficient):** measure request-weighted active-set utility, quality--bytes frontier, eviction regret, insertion rejection, and stale-handle behavior. Evicted concepts are not reported as catastrophic forgetting.
3. **Autonomous lookup:** remove exact handles and measure update/allocate/reject calibration separately.

For identity score `I_{t,i}` and prompt score `P_{t,i}` after event `t`, report acquisition, average active quality, active-state drift, retention area under the event trace, and maximum active-concept degradation. Under pressure, report request-weighted utility and regret relative to an offline oracle that knows the fixed test trace but obeys the same byte budget.

Do not compute per-concept FID from a handful of references. Aggregate KID or precision/recall is allowed only where sample counts and domains justify it. Diversity is measured conditional on a prespecified minimum fidelity/alignment threshold so that identity failure cannot masquerade as diversity.

Other required estimands are allocation precision/recall, autonomous false-update risk--coverage, similar-concept confusion, bytes of every state component, insert/read latency, peak memory, and deletion collateral damage. Latency reports pin hardware, warm-up, batch, resolution, sampler, and step count.

The primary personalization comparison includes a blinded paired human study and at least one held-out automatic identity evaluator not used by training or filtering.

## 8. Hypotheses and statistical rules

Each claim has a preregistered calibration-chosen margin that is frozen before comparative validation and the one-time final test:

| Claim | Required control | Final pass rule |
|---|---|---|
| Shared packet representation | Same amortizer + unshared progressive codec; online Share/VB-LoRA-style compression; DreamCache-style features | At fixed `B` and prompt non-inferiority margin, the paired 95% CI for request-weighted identity gain is positive |
| Causal packet allocator | Same packet stream with LRU/LRUA, size-aware caching, separable rate allocation, and greedy shared-packet policies | Positive paired CI for request-weighted utility and lower oracle regret with active-quality non-inferiority |
| Allocator guarantee | Exact locked surrogate optimum on small instances and declared snapshot assumptions | Mechanically checked proof plus feasible outputs attaining at least the certified approximation factor on exhaustive/random small instances; otherwise no theoretical claim |
| Optimization-free trade-off | Best faithfully ported matched-backbone LoRA/DreamBooth configuration | Identity and prompt non-inferior within frozen margins, plus a prespecified insertion-latency advantage |
| Autonomous lookup | Nearest-key threshold and learned novelty controls | Better risk--coverage/AURC under the same active state |
| Optional composition | Naive adapter sum | Lower entity leakage with non-inferior per-entity fidelity |
| Optional augmentation | Real-only, oversampling, standard augmentation, class-name-only frozen generator | Positive accuracy CI over every preselected primary dataset, including worst-class results |

For the shared-packet and allocator claims, an independently sampled deployment episode is the inference unit. For the optimization-free trade-off, a held-out concept is the unit. For autonomous lookup, a concept-conditioned lookup episode is the unit. For optional composition, a prespecified concept pair is the unit; for optional augmentation, an independently drawn class/support split is the unit. Prompts and generated images are nested observations, not independent replicates. Use paired hierarchical bootstrap or a mixed-effects model with training seed, concept, trace order, and prompt-template effects. Scientific runs require at least three independent training seeds; the number of test traces is selected from a preregistered CI-width or power target rather than an arbitrary five-order ceiling.

The primary lifecycle grid contains three byte budgets corresponding to 25%, 50%, and 75% of the bytes needed by the independent uncompressed-code cache for the locked active-set size, crossed with uniform and one prespecified Zipf request distribution. The shared-packet claim must pass at the 50% budget under both request regimes and have non-negative point estimates at the other locked budgets; the allocator claim uses the same primary cells. The strongest claim additionally requires replication on DreamBench++ and the controlled post-checkpoint set. Results on any single favorable budget or request distribution cannot establish the claim.

Matched methods use paired prompts and noise seeds. Report effect sizes and 95% confidence intervals. Designate one primary endpoint per claim and correct families of secondary comparisons with Holm's procedure. The final test is executed once after filters, thresholds, margins, and component choices are frozen.

## 9. Baselines and causal ablations

The main table uses the same backbone, sampler, CFG, resolution, prompts, supports, and seeds. Every baseline is tuned on validation under an equal search budget, with quality plotted against wall-clock and energy rather than forcing equal step counts.

Required closest controls are:

- HyperLoRA-style coefficients plus independent `K`-slot FIFO, LRU, and LRUA caches;
- an unshared progressive residual codec with a causal multiple-choice-knapsack/size-aware cache and rate-zero eviction;
- Compress then Serve, VB-LoRA, Share, and MoBLoRA-style shared representations adapted to the same generated codes and byte ledger;
- CF-STAR, RQT, SineLoRA-Delta, and LoRAQuant-style compression at matched bytes;
- an offline packet/knapsack oracle with future trace access, reported only as an upper reference;
- an oracle append-only quantized code store at the same total bytes;
- a stateless support-to-adapter predictor;
- DreamCache-style cached features with the same lifecycle controller;
- VSM-Diffusion where task/backbone compatibility allows a faithful comparison;
- per-concept Textual Inversion, DreamBooth, and LoRA;
- reproducible optimization-free methods such as HyperLoRA, LoFA, and DreamCache;
- continual methods including C-LoRA, Mining Your Own Secrets, ConceptGuard, and CNS for contextual continual results; and
- relevant adapter retrieval/composition systems only for secondary composition tests.

For methods that cannot be ported faithfully to SANA, SDXL becomes the matched-backbone scientific comparison and SANA is reported as architecture-transfer evidence. Different-backbone published numbers appear only in a contextual appendix table and are never used to claim superiority.

Representation ablations fix the allocator; allocator ablations fix the candidate packet stream. Required ablations include packet group size, base-code rate, packet precision, exact-only versus bounded-error sharing, maximum incidences per concept, private progressive packets, no-sharing, greedy versus guarantee-bearing allocation, hysteresis/switch cost, budget `B`, support size, request skew/drift, update/delete frequency, and code-error versus one-step diffusion-error distortion.

Storage counts base codes, packet payloads and hashes, incidence/gain records, optional tokens, metadata, handles, reference counts, controller state, and allocator state. Shared learned overhead and offline training data/optimizer steps/GPU-hours are reported separately and amortized at each active-set size.

## 10. Falsification gates

### 10.1 Paid pilot probes

The first capped pilot may establish only:

- checkpoint/API compatibility and peak memory;
- `DynamicAtomLinear` numerical correctness and gradient flow;
- frozen-backbone integrity;
- insert/read/update/delete/evict state-machine correctness;
- shared-packet deduplication and allocator correctness on synthetic adapter codes;
- exact serialized byte accounting;
- p50/p95 step time and full-scale cost extrapolation;
- reduction of flow loss on a tiny held-in run; and
- reproduction of a fixed LoRA or explicit dense update in a toy contract test.

Tiny held-out images are exploratory. This pilot cannot establish CVPR-level personalization, memory-policy superiority, 5--50-concept lifecycle results, GMM benefit, composition, or downstream augmentation.

### 10.2 Scientific evidence gates

1. **Amortizer gate:** the shared support-to-adapter predictor passes the exact quality floor and evaluator revision recorded in `evaluation-lock.yaml` on calibration-disjoint held-out concepts.
2. **Nonseparability gate:** at least one locked cohort exhibits useful packet sharing across distinct concepts, and removing the sharing mechanism reduces the quality--bytes frontier rather than merely changing metadata overhead.
3. **Packet gate:** the shared-packet hypothesis in Section 8 passes at the locked 50% byte budget under both request regimes, with non-negative point estimates at the 25% and 75% budgets, against the strongest prespecified progressive-code/shared-subspace/feature-cache control.
4. **Allocator gate:** the causal-allocator hypothesis passes in the same two primary request regimes, with active-quality non-inferiority, against the strongest prespecified causal policy; its theoretical claim passes the locked proof and exhaustive-small-instance tests.
5. **Scale gate:** gates 2--4 are replicated with three training seeds on both DreamBench++ and the controlled post-checkpoint set. Multi-shot claims additionally require the locked eligible CustomConcept101 cohort.
6. **Optional gates:** composition, GMM, and augmentation enter the paper only if their exact locked Section 8 hypotheses pass on every preselected primary dataset.

A failed tiny run falsifies the current implementation/configuration, not the whole research hypothesis. If RateMem fails gate 2, the work is repositioned as a negative benchmark/systems study or redesigned; it is not presented as a new CVPR algorithm.

## 11. Modal pilot and hard cost controls

Only the first authorized Modal workspace is used initially. Credentials remain outside the repository and never appear in patches, command arguments, shell history, environment dumps, experiment metadata, paper artifacts, or logs.

Before token activation, the exact workspace and current billing-cycle usage must be verified and the Modal **workspace usage budget must be set to USD 28**. If that outer cap cannot be verified, no paid job launches. The internal launch target is USD 27 and includes pending worst-case attempts because reported billing can lag.

Execution rules:

- one named non-global profile;
- one GPU and one synchronous ephemeral invocation;
- `retries=0`, no map/fan-out, detached job, deployment, or automatic GPU fallback;
- one exact L40S by default; an OOM stops the phase and triggers explicit A100 re-budgeting;
- a timeout and conservative cost bound on every invocation;
- a rates snapshot and calculation of GPU, CPU, requested RAM, startup/download, and storage; and
- reconciliation after each phase before another launch.

Before every phase:

\[
\text{known usage} + \text{pending worst-case} + \text{new phase bound} \le 27.
\]

The provisional allocation is:

- setup/cache, one inference, and one backward pass: up to USD 2;
- 10 warm-up plus 20 measured training steps: up to USD 3;
- minimal held-in adapter/memory pilot: up to USD 16; and
- diagnosed rerun reserve: up to USD 6.

The measured p95 step time, not a guessed 200--500-step target, determines the remaining step cap. Jobs terminate on exceptions and never leave idle GPU services. No other supplied workspace is selected automatically.

Authentication uses an interactive/non-echoing Modal mechanism, a permission-restricted profile outside the repository, and explicit profile selection for every process. The public SANA checkpoint requires no Hugging Face token. W&B and unredacted environment/configuration dumps remain disabled. Repository changes, logs, and exports are scanned for Modal/HF/W&B credential patterns before publication.

## 12. Testing and artifacts

Mandatory unit/contract tests include:

- `alpha=0` exactly reproduces the frozen linear layer within declared numeric tolerance;
- dynamic output matches explicit `W + sum(alpha BA)` for batch-one and per-example batch coefficients;
- gradients reach code, controller, and atoms but never backbone parameters;
- no dense `Delta W` allocation, checked with peak-memory instrumentation;
- trainable-weight save/load equivalence;
- hard byte budget under randomized operation traces;
- canonical packet hashing, exact deduplication, incidence/reference-count integrity, and atomic packet redirection;
- stale-handle rejection, deletion state reclamation, and deterministic eviction/garbage collection;
- scoring probes leave usage and memory bytes unchanged;
- progressive decode is prefix-consistent and packet removal changes only declared dependent codes;
- allocator outputs are feasible and meet the certified approximation factor against brute-force optima on exhaustive/random tiny instances, while satisfying every mechanically testable theorem premise;
- soft/hard codec agreement on controlled cases;
- a tiny randomized SANA integration test on CPU;
- one real-checkpoint inference and one random-timestep backward; and
- deterministic replay within declared tolerances rather than cross-GPU bitwise identity.

Each attempt artifact contains configuration, git and diff hashes, immutable base-model revision, Diffusers/PEFT/container revisions, dataset-manifest hash, seed, GPU SKU, price snapshot, timeouts, Modal call/attempt IDs, peak memory, p50/p95 step time, exit status, pending cost bound, reconciled cost, metrics, and checksummed trainable checkpoint. The multi-gigabyte frozen checkpoint is referenced by immutable revision rather than copied into each artifact.

A result enters the manuscript only if its artifact directory passes schema validation and regenerates the exact table row or figure datum.

## 13. Manuscript and figure scope

The old manuscript is rewritten rather than patched. The main paper focuses on one story: **jointly learned personalization state under a hard byte budget**.

Provisional eight-page structure:

1. resource-bounded personalization motivation and precise lifecycle task;
2. closest work and why amortizer-plus-cache is insufficient;
3. progressive shared enhancement-packet memory;
4. causal packet allocator, guarantee, and training;
5. immutable workload, byte accounting, and statistical protocol;
6. quality--bytes frontier, allocation ablations, and efficiency;
7. limitations and conclusion.

Planned figures:

1. a vector overview contrasting private progressive codes with shared RateMem packets;
2. a byte-accurate memory diagram showing base records, shared packets, incidences, and lifecycle actions;
3. a serialized workload timeline distinguishing operational reads from read-only scoring probes;
4. real qualitative grids selected by a preregistered rule, including failures; and
5. request-weighted quality versus bytes/latency and eviction-regret curves generated directly from validated artifacts.

GMM, content lookup, multi-concept composition, and category augmentation move to supplementary material unless a result is strong enough to replace, rather than merely add to, a main-paper experiment. Architecture figures are vector graphics; qualitative result panels are never fabricated or beautified before real checkpoints exist.

## 14. Risks and decisions

- **Remaining novelty risk:** adapter subspaces, sparse/residual codecs, quantization, and online caching are established. The method survives only if cross-concept enhancement sharing is empirically nonseparable and the causal allocator adds a valid guarantee or a strong systems result.
- **Amortizer bottleneck:** a public-data support-to-adapter predictor may be much weaker than proprietary HyperLoRA. RateMem cannot hide this by comparing only to the frozen backbone.
- **SANA baseline portability:** if closest methods cannot be implemented faithfully, the primary matched table moves to SDXL.
- **Weak packet reuse:** visually distinct personalized concepts may share too few useful residual packets for meaningful savings. Failure of the nonseparability gate collapses the method contribution.
- **Surrogate mismatch:** a provable packet-utility surrogate may correlate poorly with generated identity/prompt quality. Both calibration error and real generation outcomes are reported.
- **Dataset contamination:** foundation models make absolute unseen status difficult to prove; the paper reports controlled post-checkpoint evidence and limitations.
- **Deletion scope:** only active online state is managed; no certified unlearning claim is made.
- **Compute insufficiency:** USD 27 is an engineering pilot, not a complete paper budget.
- **Target rules:** CVPR 2027 formatting, disclosure, and reproducibility requirements are rechecked when officially published.

## 15. Design-phase acceptance criteria

The design phase completes only when:

- the author approves the revised hard-budget research claim;
- a second novelty review finds no unaddressed fatal collision;
- the dataset-lock task precedes all scientific model selection;
- every principal claim maps to a frozen event trace, metric, baseline, test, and artifact;
- the USD 28 workspace cap and USD 27 internal bound are verifiable before credentials are activated; and
- the manuscript contains no claim that would require fabricated evidence.
