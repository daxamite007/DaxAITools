# nanoMoE v10 — Full Build Context (dax.md)

This document is the complete handoff for building nanoMoE **v10** on top of the
existing v9 codebase. Read it fully before writing any code. Read
`nanoMoE_v10_current_state.md` for decisions made / work completed, and
`nanoMoE_v10_index.md` for a map of the files.

v10 does **not** restart from scratch. The model, optimizer, data pipeline, and
evaluation harness from v9 are all reused. v10 adds **one new capability**:
Maximal Update Parametrization (µP) so that hyperparameters are tuned once on a
tiny proxy model and transferred — with math, not guesswork — to the full 322M
target.

---

## 1. Goal

Produce a 322M-total-parameter MoE that is **more robust than the v9
40×-double-Chinchilla run, for roughly the same compute cost**, by removing the
hyperparameter guesswork that dominated v9's failures.

Concretely:

- Tune learning rates, weight decay, and global batch size on a **10–30M
  parameter proxy** (the "toy run").
- Use **µP** to scale those exact values up to the 322M architecture so the full
  run starts at the right operating point on **step 0** — no warmup-phase
  flailing, no dead runs.
- Keep the v9 architecture, data, tokenizer, routing config, and two-optimizer
  (Muon + AdamW) setup unchanged. **µP is the only new experimental variable.**

### Why this is worth doing (grounded in v9 history)

v9's cost was not the 322M run itself — it was the **dead runs getting there**.
From `nanoMoE_v9_current_state.md`:

- **r1, r2, r3 — all DEAD.** Routing collapse during warmup from wrong
  loss weights / decay schedule.
- **r4-cont-BROKEN — DEAD.** An 8× LR spike at the resume boundary destroyed a
  trained checkpoint; steps 8001–14500 poisoned and deleted.

Every one of those was a **hyperparameter / schedule error** that µP is designed
to prevent. The proxy sweep costs a few dollars and a couple of hours; a single
dead 322M run costs ~$10–12 and days. **The entire ROI of v10 is making the
expensive run boring.**

### What "robustness" means here

Not "more tokens." The robustness gain comes from entering training at the
correct LR/WD/batch operating point, validated across widths, so the loss curve
is on its best trajectory from the first step rather than recovering from a bad
guess. If we also choose to extend the token budget, that is a *separate*
decision (see §6) and is explicitly **not** required to beat v9.

---

## 2. The proxy → target relationship (read this carefully)

> **"Copy the structure to the 322M" does NOT mean copying weights.**

The proxy is 128–256 wide; the target is 768 wide. Their weight tensors are
different shapes — you cannot and must not copy weights between them. Copying
weights would also defeat µP, whose entire premise is that the **full model is
re-initialized fresh** and only the *hyperparameters* transfer.

What actually transfers from proxy to target:

| Transfers (proxy → target) | Does **not** transfer |
|---|---|
| Architecture config (depth, experts, top_k, stride, head_dim, block_size) | Weights / checkpoints |
| Optimal `adamw_lr`, `muon_lr`, `weight_decay`, global batch size | Activations / optimizer state |
| The µP multiplier table (init scale, LR scale, output multiplier) | Anything learned by the proxy |

So the v10 pipeline is:

```
tiny proxy (fresh init) ──µP sweep──► optimal HPs ──µP scaling math──► 322M target (fresh init) ──train──► model
```

### 2.1 What must be held identical between proxy and target

µP transfers across **width only**. The standard µP theory (Tensor Programs V,
Yang & Hu 2022) is a *width* result. Depth-µP and expert-count transfer are
separate, more fragile research areas and are **out of scope**. Therefore the
proxy and target must match in **every dimension except width**:

| Dimension | Proxy | Target (322M) | Must match? |
|---|---|---|---|
| `n_layer` (depth) | 12 | 12 | **YES — identical** |
| `n_exp` (experts) | 8 | 8 | **YES — identical** |
| `top_k` | 2 | 2 | **YES — identical** |
| `stride` (MoE every N layers) | 2 | 2 | **YES — identical** |
| `head_dim` | 64 | 64 | **YES — identical** (see §2.2) |
| `block_size` | 4096 | 4096 | **YES** (seq len is not a µP axis; keep equal for fidelity) |
| routing / aux config | v9 values | v9 values | **YES — identical** (see §4.3) |
| `n_embd` (**width**) | 128 and 256 | 768 | **NO — this is the only thing that changes** |
| `n_head` | 2 (@128), 4 (@256) | 12 | scales *with* width to keep `head_dim=64` |

### 2.2 Width is `n_embd`; keep `head_dim` fixed at 64

Target is `n_embd=768`, `n_head=12` → `head_dim = 768/12 = 64`.

For the proxy, **hold `head_dim=64` and scale `n_head`** with width:

- width 128 → `n_head=2` (128/2 = 64)
- width 256 → `n_head=4` (256/4 = 64)

**Why this matters and saves work:** µP normally requires changing the attention
logit scale from `1/sqrt(head_dim)` to `1/head_dim` *because head_dim changes
with width*. If we keep `head_dim` **fixed** across proxy and target, that scale
factor is constant across all our models, so **the existing
`1/sqrt(head_dim)` scaling in `CausalSelfAttention` is fine and needs no
change** for transfer. This is the simpler, lower-risk path and is the v10
default. (If you instead held `n_head` fixed and scaled `head_dim`, you would
have to implement the µP attention-scaling change. Don't.)

---

## 3. µP: what actually has to be built

µP is **not** "make `n_embd` smaller and run a sweep." That is naive width
scaling and the optimal LR will *not* transfer. µP is a specific
re-parametrization of init scale, per-layer learning rate, and the output-layer
forward multiplier. The current `model.py`/`train.py` are **standard
parametrization** and must be extended.

### 3.1 The µP multiplier table (Adam-family params)

Let `m = width_target / width_base` be the width multiplier relative to the
width you *tuned at* (the base width — recommend 256, see §5). For the
Adam-optimized parameter groups:

| Param group | Init variance | Adam LR scaling | Forward multiplier |
|---|---|---|---|
| Input embedding (`wte`) | Θ(1) — constant (std≈0.02) | Θ(1) — fixed | Θ(1) |
| Hidden weights | Θ(1/fan_in) | Θ(1/m) | Θ(1) |
| Output / readout (`lm_head`) | μP zero- or 1/fan_in init | Θ(1/m) | **Θ(1/m)** |
| LayerNorm, biases | — | Θ(1) — fixed | — |

Notes on what the **existing code already gets right**:

- Embedding init is already constant `std=0.02` (µP-correct for the input layer).
- `use_switch_tfm_init=True` already gives hidden weights `std ∝ 1/sqrt(fan_in)`
  i.e. variance `Θ(1/fan_in)` — µP-correct for hidden layers.

So the **gaps to close** are:

1. The **output-layer (readout) treatment**: a `1/m` forward multiplier on the
   logits and µP readout init. The cleanest implementation is a `MuReadout`-style
   head (multiply logits by `1/width_mult`).
2. **Per-group LR scaling** by `1/m` for the hidden + readout groups when moving
   from base width to target width.
3. The **MoE expert tensors** (`MLPExperts.c_fc`, `c_proj`) are 3-D
   `nn.Parameter`s, **not** `nn.Linear`. The `mup` PyPI package will **not**
   auto-handle them — their init and LR scaling must be set manually. Same goes
   for the router `w_g` (a `nn.Linear` on AdamW). These are the MoE-specific
   parameters and they sit on AdamW, so standard Adam-µP rules apply once wired
   up by hand.

### 3.2 Weight tying conflict — OPEN DECISION (resolve before coding `model_mup.py`)

`model.py` ties `wte.weight = lm_head.weight`. µP wants the **input embedding**
(Θ(1) multiplier) and the **readout** (Θ(1/m) multiplier, different init)
treated *differently*. A single shared matrix cannot satisfy both cleanly.

Two options, both valid — pick one in the Current State doc:

- **(A) Keep tied** (preserves the 322M budget): apply the µP `1/m` logit
  multiplier at the output while sharing the matrix; treat the shared weight's
  init/LR as the input-embedding rule. Slightly non-standard but keeps params at
  322M.
- **(B) Untie** (cleanest µP, standard `mup` assumption): separate `lm_head`.
  Adds ~38.6M params (50264 × 768) → total ≈ **360M**, breaking the "322M"
  label and the v9 token-budget math (40 × params).

Recommendation: **(A) keep tied** unless we deliberately re-baseline the param
budget. Flagged as an open decision because it changes `model_mup.py`,
the param count, and the `40×` token target.

### 3.3 Implementation route

Two ways to implement µP; both require manual handling of the MoE tensors:

- **Manual µP** (recommended for this custom MoE): add µP flags/multipliers to a
  `model_mup.py` (or behind a `use_mup` flag in `model.py`), wire per-group LR
  multipliers into the optimizer construction in `train.py`. Most robust because
  you control the expert tensors and router explicitly.
- **`mup` package** (`pip install mup`): use `set_base_shapes`, `MuReadout`,
  `MuAdamW`. Faster for the `nn.Linear` parts, but you **still** hand-treat
  `MLPExperts` and verify it plays nicely with `torch.compile` and PyTorch
  2.11. Treat the package as a helper, not a turnkey solution.

### 3.4 Muon + µP — KNOWN UNCERTAINTY, do not hand-wave

µP's LR-transfer rules are derived for SGD and Adam. **Muon is neither.** Muon
already normalizes each update's spectral norm (scale ≈ `lr * sqrt(max_dim)`),
which gives it *some* built-in width-robustness — but this is **not** the same
as the Adam `1/m` rule, and applying the Adam rule blindly to the Muon group is
unjustified.

Safe procedure for the Muon group:

1. Sweep `muon_lr` and `adamw_lr` **jointly** on the proxy (don't assume the 3×
   ratio is optimal — it was a v9 choice, re-verify it).
2. Before committing the 768 run, **coord-check at an intermediate width (512)**
   and confirm the proposed `muon_lr` keeps activation coordinates stable. If
   512 looks wrong, the Muon LR scaling is off and 768 will be too.
3. Only after the 512 check passes do you launch 768.

This is the single most likely place v10 transfer can silently fail. Budget the
extra intermediate-width check; it is cheap insurance against another dead 322M
run.

---

## 4. What is reused unchanged from v9

### 4.1 Data — DONE, reuse as-is
All v9 data artifacts are **model-size agnostic** and already built. The proxy
sweeps and the target run both read the same binaries:

- `data/moe_v9_pretrain/train.bin` (~12.4B tokens), `val.bin` (~477M tokens)
- `meta.pkl` → `{'vocab_size': 50264}` (7 ChatML tokens)
- Do **not** re-run any data script. See `nanoMoE_v9_context.md` §5 / §8 and
  `nanoMoE_v9_current_state.md` "Data — COMPLETED".

### 4.2 Optimizer split — reuse `muon.py` as-is
`muon.py` is final. The param-split filter (Muon ← exactly-2D attention/dense-MLP
weights; AdamW ← embeddings, norms, router, 3-D experts, biases) is correct and
already in `train.py`. The three historical filter bugs (gate→router,
ndim>=2→ndim==2, embed→wte/wpe) are fixed. **µP changes the per-group LR
*multipliers*, not which params go to which optimizer.**

### 4.3 Routing / MoE loss config — HOLD FIXED, do not sweep
v9 validated these and saw **no routing collapse** through 8000 steps. Keep them
constant across proxy and target; do **not** put them in the µP sweep (routing
dynamics are sensitive and these weights are not part of the width-transfer
theory):

```
use_aux_loss=True   aux_loss_weight=0.02
use_router_z_loss=True   router_z_loss_weight=0.001
use_switch_tfm_init=True   router_use_full_prec=True
train_capacity=1.25   eval_capacity=2.0   min_capacity=4
```

### 4.4 Environment
Per current target pod: **Python 3.12, PyTorch 2.11+cu130, no flash-attn**
(SDPA handles attention; flash-attn fails to build on CUDA 13 — do not install).
Working dir: `/workspace/nanoMoE`. DDP via `torchrun --standalone`.

---

## 5. The proxy run (toy run) — sizing & sweep protocol

| Item | Value | Reason |
|---|---|---|
| Width(s) | **256 primary, 128 secondary** | Need ≥2 widths to coord-check transfer |
| `n_head` | 4 @256, 2 @128 | keeps `head_dim=64` (§2.2) |
| Everything else | identical to target | depth 12, experts 8, top_k 2, stride 2, block 4096 |
| Approx params | ~12M @128, ~25M @256 | inside the 10–30M target band |
| Tokens per sweep | **1–2B, exactly 1 epoch** | enough to see the loss trajectory; never repeat data (repetition shifts the curve and corrupts transfer) |
| Sweep variables | `adamw_lr`, `muon_lr`, `weight_decay`, global batch size | the µP-transferable knobs |
| Held fixed | routing/aux config (§4.3), architecture, dropout=0 | |

**Coord check first, then LR sweep:**

1. Implement µP. Run a **coordinate check**: forward a fixed batch at widths
   128 / 256 / 512 and confirm per-layer activation RMS is ~flat across width.
   If it drifts with width, µP is mis-wired — fix before sweeping. (This is the
   canonical µP sanity test; build `mup_coord_check.py`.)
2. Sweep `adamw_lr` × `muon_lr` on a small grid at width 256 (and spot-check at
   128) over 1–2B tokens each. Pick the LR pair with the best/most-stable val
   loss trajectory. The optimum should sit at roughly the **same coordinate** at
   128 and 256 — that agreement *is* the transfer guarantee.
3. Sweep `weight_decay` and global batch size more coarsely.

Estimated proxy cost: a handful of sub-billion-to-2B-token runs at ~12–25M
params is **hours and a few dollars**, consistent with v9's note that small
models hit the measurement threshold quickly.

---

## 6. The target run (322M) — sizing & cost (grounded, not optimistic)

Architecture (unchanged from v9): `n_layer=12, n_head=12, n_embd=768,
n_exp=8, top_k=2, stride=2, block_size=4096`, ~322M total / ~100–150M active.

### 6.1 Token budget — DECISION
The pasted "rough idea" suggested **30–50B tokens**. That conflicts with the
"not a bunch more cost" goal **and** with v9's measured throughput. Two paths:

- **Path 1 (recommended default): reuse the proven 12.88B budget.** Same as v9
  (40× double-Chinchilla). Robustness comes from µP-correct HPs, not more
  tokens. Lowest cost, directly comparable to v9, data already exists.
- **Path 2 (stretch): extend to ~25–40B tokens.** Requires building more data
  and 2–3× the compute/cost. Only justify this if a specific capability gap
  remains after Path 1. **Not required to beat v9.**

### 6.2 Cost / time — use v9's *measured* numbers, not the pasted estimate
The pasted doc assumed **444 TFLOPS / 40% MFU / FlashAttention / $0.50hr → ~$40**.
None of those match this project:

- 2× A5000 BF16 dense is ≈ **222 TFLOPS** (the 444 figure is the *2:4-sparsity*
  number; we don't use structured sparsity).
- v9 **measured ~22% MFU** on 2× A5000 (~9 s/step), and ~48% on a single
  better/unknown GPU (~4 s/step). Not 40% on the A5000.
- **No FlashAttention** here (SDPA only).
- v9 pod rate was **$0.61/hr**, not $0.50.

Grounded estimate for a **12.88B** target run (Path 1), at v9's ~1.05M
tokens/step (batch 1 × grad_accum 128 × block 4096 × 2 GPUs):

| Hardware (v9-measured) | s/step | tok/s | Time for 12.88B | Cost @ rate |
|---|---|---|---|---|
| 2× A5000, ~22% MFU | ~9 | ~116k | ~31 hr | ~$19 @ $0.61/hr |
| better pod, ~48% MFU | ~4 | ~262k | ~14 hr | rate unknown |

For a **40B** run (Path 2) multiply by ~3.1 → **~96 hr** on 2× A5000 (~$59),
i.e. clearly *more* than the pasted "$40", which is why Path 1 is the default.
Add proxy sweeps (~$2–5) and storage/egress buffer. **Realistic all-in for Path
1 ≈ $25–35; Path 2 ≈ $65–95.** Treat all figures as estimates, not guarantees —
rates and MFU vary by pod.

### 6.3 Resume-schedule lesson (carried from v9 — still applies)
If the target run is ever resumed past its original `lr_decay_iters`, you **must**
patch `iter_num=0` in the checkpoint and relaunch with a lower-peak schedule.
Never extend `lr_decay_iters` without patching `iter_num` — that is exactly what
killed r4-cont-BROKEN. See `nanoMoE_v9_current_state.md` "Resume LR schedule".

---

## 7. SFT and evaluation
Unchanged plan from v9. After the µP-tuned pretrain completes, run SFT
(`build_sft_v5.py`: Muon two-optimizer, weight_decay=0.1, ChatML, 500–1000
steps, LRs at the SFT ratio) and the eval harness (`eval_nanomoe.py`,
`compare_models.py`, `nla_test.py`, `final_eval.py`). Baselines to beat are in
`nanoMoE_v9_context.md` §9. µP does not change SFT mechanics; if you want, you
can also µP-tune SFT LRs on the proxy, but it is lower priority.

---

## 8. Out of scope for v10
- Depth-µP or expert-count transfer (width-only is the supported regime).
- LSUV data-driven init and curriculum-learning phases — these were raised as
  *alternative* from-scratch accelerators. They are **complementary, optional,
  and deferred**; µP is the v10 mechanism. Revisit only after µP transfer is
  validated.
- Forced-sparsity warm-up as a separate mechanism — v9's aux + router-z loss
  already prevents router collapse; no separate warm-up is added.
- Runtime tool harness, RestrictedPython sandbox, streaming inference (same as
  v9 §13).

---

## 9. Build sequence (v10)
```
1.  Resolve OPEN DECISIONS in nanoMoE_v10_current_state.md
      - weight tying: keep tied (A) vs untie (B)   [§3.2]
      - token budget: Path 1 (12.88B) vs Path 2     [§6.1]
2.  model_mup.py        — µP init + MuReadout output multiplier + manual
                          MoE-tensor/router µP handling   [§3.1–3.3]
3.  train.py (µP)       — per-group LR multipliers (1/m for hidden+readout),
                          --use_mup, --base_width, --width flags
4.  mup_coord_check.py  — verify activation RMS flat across widths 128/256/512
5.  Proxy sweep         — adamw_lr × muon_lr (+ wd, batch) @256, spot-check @128,
                          1–2B tokens each, 1 epoch                  [§5]
6.  mup_transfer.py     — compute scaled HPs for width 768 from proxy optimum
7.  Intermediate check  — coord-check + short run @512 to validate Muon scaling [§3.4]
8.  Target run @768     — launch 322M with transferred HPs (Path 1 budget)  [§6]
9.  SFT + eval          — build_sft_v5.py, eval_nanomoe.py, compare/nla/final  [§7]
10. Compare to v9 baselines (nanoMoE_v9_context.md §9)
```

---

*End of context document — nanoMoE v10 (dax.md)*
