# nanoMoE v10 — Context

This is the full build context for **nanoMoE v10**. Read it before writing code,
alongside `current_state.md` (what's done / next) and the codebase.

> **This is a fresh, from-scratch run.** The model is trained from random
> initialization. There is no donor model, no prior checkpoint, and no earlier
> run being continued. The data, the tokenized corpus, the model weights, and
> every training step are new. The *only* thing carried in is **code** — the
> model definition, training loop, optimizer, and data tooling. Nothing else
> transfers.

---

## 1. Goal

Train a **322M-total-parameter Mixture-of-Experts** language model from scratch,
cheaply and robustly, by removing hyperparameter guesswork. The mechanism is
**Maximal Update Parametrization (µP)**: tune learning rates, weight decay, and
batch size once on a tiny proxy model, then transfer those settings — with math,
not trial and error — up to the full 322M architecture so the real run starts at
the right operating point on step 0.

Robustness here comes from entering training at the correct LR/WD/batch point
(validated across widths), not from spending more compute. The proxy sweep costs
a few dollars; it buys a full-size run that doesn't have to flail its way to a
good configuration.

---

## 2. The proxy → target relationship

µP tunes a small model and transfers the *hyperparameters* (not weights) to the
big one. The target is re-initialized fresh; only the tuned settings move across.

```
tiny proxy (fresh init) ──µP sweep──► optimal HPs ──µP scaling──► 322M target (fresh init) ──train──► model
```

**Weights never transfer** (proxy width ≠ target width — different tensor
shapes). What transfers: the architecture config, the optimal `adamw_lr`,
`muon_lr`, `weight_decay`, global batch size, and the µP multiplier table.

### 2.1 µP transfers across WIDTH only

The proxy must match the target in **every dimension except width** (`n_embd`).
Depth, expert count, routing, and sequence length are held identical — µP's
guarantees only hold when width is the sole axis that changes.

| Dimension | Proxy | Target | Must match? |
|---|---|---|---|
| `n_layer` | 12 | 12 | **identical** |
| `n_exp` | 8 | 8 | **identical** |
| `top_k` | 2 | 2 | **identical** |
| `stride` (MoE every N layers) | 2 | 2 | **identical** |
| `head_dim` | 64 | 64 | **identical** (see 2.2) |
| `block_size` | 4096 | 4096 | **identical** |
| routing / aux-loss config | fixed | fixed | **identical** |
| `n_embd` (**width**) | 128 and 256 | 768 | the only thing that changes |
| `n_head` | 2 @128, 4 @256 | 12 | scales with width to keep head_dim=64 |

### 2.2 Width is `n_embd`; hold `head_dim` fixed at 64

Target: `n_embd=768`, `n_head=12` → `head_dim=64`. For the proxy, hold
`head_dim=64` and scale `n_head` (128→2 heads, 256→4 heads). Holding head_dim
fixed means the attention logit scale (`1/sqrt(head_dim)`) is constant across all
models, so **the standard attention scaling needs no µP change**. This is the
low-risk path and the default.

---

## 3. What µP requires in code

µP is a re-parametrization, not just shrinking `n_embd`. Standard init + a
sweep at small width will **not** transfer. The pieces:

### 3.1 Multiplier table (Adam-family groups), width multiplier `m = width/base_width`

| Param group | Init variance | Adam LR | Forward multiplier |
|---|---|---|---|
| Input embedding | Θ(1), const (std≈0.02) | Θ(1) | Θ(1) |
| Hidden weights | Θ(1/fan_in) | Θ(1/m) | Θ(1) |
| Output / readout | µP readout init | Θ(1/m) | **Θ(1/m)** |
| LayerNorm / biases | — | Θ(1) | — |

The model's embedding init (constant 0.02) and fan-in hidden init are already
µP-shaped. The gaps to build: a `1/m` **readout multiplier** on the logits, and
**per-group `1/m` LR scaling** for hidden + readout groups. The MoE expert
tensors are 3-D `nn.Parameter`s (not `nn.Linear`) and the router is a small
`nn.Linear`; both sit on AdamW and need their µP init/LR set **by hand** — no
package auto-handles the 3-D expert tensors.

### 3.2 Weight tying — OPEN DECISION
The model ties input embedding and output head. µP wants them treated
differently (input Θ(1) multiplier; readout Θ(1/m), different init). Either keep
tied + apply the `1/m` logit multiplier (preserves the 322M budget), or untie
(cleanest µP, +~38M params → ~360M). Resolve before writing the µP model.

### 3.3 Muon + µP — verify, don't assume
µP's LR-transfer math is derived for SGD/Adam. **Muon is neither** — it
normalizes each update's spectral norm, giving it its own width behavior. Do not
blindly apply the Adam `1/m` rule to the Muon group. Procedure: sweep `muon_lr`
and `adamw_lr` jointly on the proxy, then **coord-check at an intermediate width
(512)** before committing the 768 run. This is the most likely place transfer
can silently fail.

---

## 4. Architecture

Target (322M total, ~100–150M active per token):
`n_layer=12, n_head=12, n_embd=768, n_exp=8, top_k=2, stride=2, block_size=4096`,
vocab 50264 (base BPE + 7 ChatML special tokens), MoE every 2 layers (6 MoE
blocks). Dropout 0.

Proxy: identical except `n_embd ∈ {128, 256}`, `n_head ∈ {2, 4}` (head_dim 64).
~12M params @128, ~25M @256 — inside the 10–30M target band.

Routing config (held fixed across proxy and target; do **not** sweep):
`aux_loss_weight=0.02, router_z_loss_weight=0.001, switch_tfm_init=True,
router_full_prec=True, train_capacity=1.25, eval_capacity=2.0, min_capacity=4`.

---

## 5. Optimizer

Two-optimizer split:
- **Muon** ← exactly-2-D attention and dense-MLP weights.
- **AdamW** ← embeddings, norms, router, 3-D expert tensors, all biases.

µP scales the per-group learning rates by `1/m` for the hidden + readout groups
(see §3). Sweep `muon_lr` / `adamw_lr` jointly on the proxy; the Muon LR scaling
to the target is confirmed empirically at width 512 (§3.3).

---

## 6. Data

Tokenized corpus is assembled by `prepare_data_chunked.py` into bucket streams at
fixed ratios, drawn proportionally to a total-token budget. Each bucket folder is
looped forever, so download/keep size sets *repetition*, not the ratio.

| Bucket | Ratio | Source |
|---|---|---|
| LongForm | 40% | Cosmopedia |
| Chat | 28% | Hermes + Infinity-Instruct + Magpie (see §6.1) |
| Code | 25% | StarCoderData (Python) |
| Web | 5% | FineWeb (sample) |
| Synthetic | 2% | `generate_reasoning.py` (ChatML, 7 special tokens) |

Vocab/tokenizer: 50264, with 7 ChatML special tokens; `meta.pkl` written by the
data pipeline.

### 6.1 Chat filtering — two different mechanisms, by design

The chat bucket is quality-filtered, but **not** with a single filter, because no
single filter works across all instruction-data distributions:

- **Hermes → NLA proxy** (MiniLM embeddings + `nla_45k_proxy_embeddings.lgb`,
  tier1 ≥ 0.5 / tier2 ≥ 0.4). The proxy reliably selects genuine
  instruction-following on Hermes-style data (~23% pass).
- **Infinity-Instruct & Magpie → heuristics only.** The NLA proxy does **not**
  generalize to these: on Infinity it elevates terse FLAN-style classification
  stubs ("C", "yes", one-line answers) and lets leaked system-prompt boilerplate
  through, and on Magpie it rejects ~99% of otherwise-clean data. So these use
  transparent heuristics instead: strip leaked "You are an AI assistant…"
  boilerplate, drop responses under ~10 words, drop pure single-token/MCQ
  answers, and dedup. `filter_chat_data.py` implements both paths.

Within the chat bucket the source mix is ≈ Infinity 40% / Magpie 30% / Hermes 30%
(set by the yield pattern in `get_tiered_chat_stream()`; Hermes split tier1 2× /
tier2 1×). See `current_state.md` for the open question of full-run repetition.

### 6.2 Token budget
Target ≈ **13B tokens** (≈40× the 322M parameter count — roughly double the
Chinchilla-optimal point, where small models benefit from overtraining).
Extensible toward 30–40B given the large chat pool now available; the proxy
sweeps use 1–2B tokens each, exactly one epoch (no data repetition during a
sweep, which would distort the loss curve and break transfer).

---

## 7. Cost (estimate, to validate on the actual pod)

Hardware: 2× NVIDIA A5000 (48 GB total). For a ~13B-token run, project on the
order of **~30–50 GPU-hours** depending on achieved MFU (BF16, SDPA attention —
no flash-attn on CUDA 13). Proxy µP sweeps are a few hours and a few dollars
total. Treat all figures as projections; measure tokens/sec on the first hundred
steps and recompute. SFT and evaluation are additional but small.

---

## 8. Environment
Python 3.12, PyTorch 2.11+cu130, **no flash-attn** (SDPA handles attention).
Working dir `/workspace/nanoMoE`. DDP via `torchrun --standalone`.

---

## 9. Build sequence
```
1. model_mup.py        — µP init + readout 1/m multiplier + manual MoE-tensor/router µP
2. train.py (µP)       — per-group 1/m LR scaling; --use_mup, --base_width, --width
3. mup_coord_check.py  — verify activation RMS flat across widths 128/256/512
4. Proxy sweep         — adamw_lr × muon_lr (+ wd, batch) @256, spot-check @128; 1–2B tok, 1 epoch
5. mup_transfer.py     — scale optimal HPs to width 768
6. Width-512 check     — confirm Muon LR scaling before the full run (§3.3)
7. Target run @768     — 322M, ~13B tokens
8. SFT + evaluation
```

---

## 10. Out of scope
Depth-µP / expert-count transfer (width-only is the supported regime); LSUV and
curriculum-phase init (optional, deferred — µP is the v10 mechanism); a separate
forced-sparsity warm-up (the aux + router-z loss already prevents router
collapse).

*End of context.*
