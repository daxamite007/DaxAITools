# nanoMoE v10 — Index (file map & where to look)

A map of the repo for v10. v10 reuses the v9 codebase and adds a µP layer on
top. Start with the three docs, then the model/train/optimizer core.

- **`dax.md`** — the v10 **context / full plan** (µP strategy, proxy→target,
  cost, build sequence). Read this first.
- **`nanoMoE_v10_current_state.md`** — what's done, what's open, next steps.
- **`nanoMoE_v10_index.md`** — this file.
- `nanoMoE_v9_context.md` / `nanoMoE_v9_current_state.md` — inherited v9 detail
  (data ratios, tokenizer, NLA filter, baselines, run history). Still the source
  of truth for anything v10 reuses unchanged.

Working dir: `/workspace/nanoMoE`. Env: Python 3.12, PyTorch 2.11+cu130,
no flash-attn.

---

## Where do I look for…?

| I want to… | Look at |
|---|---|
| Understand the v10 µP plan / why | `dax.md` |
| Know what's built and what's blocked | `nanoMoE_v10_current_state.md` |
| Change the model architecture / forward pass | `model.py` |
| Add / debug µP | `model_mup.py` *(to create)*, `mup_coord_check.py` *(to create)* |
| Change the training loop, LR schedule, optimizer split | `train.py` |
| Understand the Muon optimizer | `muon.py` |
| Understand MoE routing / load-balancing loss | `model.py` → `Router`, `MOELayer` |
| Rebuild data (only if lost) | `restore_local_data_v9.py` → `prepare_data_chunked.py` → `safe_merge_v9.py` |
| Regenerate synthetic data | `generate_reasoning.py` (v2.1, ChatML) |
| Keep periodic checkpoints during a run | `keep_checkpoints.py` |
| Evaluate the trained model | `eval_nanomoe.py`, `final_eval.py`, `compare_models.py`, `nla_test.py`, `test_inference.py` |
| Inspect routing health mid-run | `inspect_routing.py` (use the 200-token prompt) |
| Run SFT after pretrain | `build_sft_v5.py` |
| Pass CLI overrides to train.py | `configurator.py` (exec'd by train.py) |
| Understand aux/router-z loss aggregation | `manager.py` (`MANAGER`) |

---

## Core model & training (verified against source)

### `model.py` — the GPT/MoE definition (everything in one file)
Key classes and where they are:
- `GPTConfig` (dataclass) — all architecture + MoE flags. **The width knob is
  `n_embd`; keep `head_dim = n_embd/n_head = 64` fixed for v10.**
- `GPT.__init__` / `GPT._init_weights` — init scheme. `wte` is constant `std=0.02`
  (µP-correct input embedding); `use_switch_tfm_init` gives hidden weights
  `std∝1/sqrt(fan_in)` (µP-correct hidden init). Residual `c_proj` get the
  GPT-2 `1/sqrt(2·n_layer)` scaling. **`wte.weight` is tied to `lm_head.weight`
  (line ~400) — this is the µP weight-tying conflict; see `dax.md` §3.2 / D1.**
- `CausalSelfAttention` — uses SDPA with default `1/sqrt(head_dim)` scaling.
  Because `head_dim` is held fixed in v10, **this needs no µP change.**
- `Router` — (noisy) top-k gating, aux loss, router-z loss, expert capacity.
  Router weight is `...mlp.router.w_g` (a `nn.Linear`, on AdamW).
- `MLPExperts` — experts as **3-D `nn.Parameter`s** (`c_fc`, `c_proj`) run with
  `bmm`. **Not `nn.Linear`** → the `mup` package can't auto-handle these; their
  µP init/LR must be set manually (`dax.md` §3.1).
- `MOELayer`, `MLP`, `Block` — block assembly; MoE placed every `stride` layers.
- `estimate_mfu` — MFU estimate (note: hardcoded A100 312 TFLOPS reference; for
  A5000 the absolute % differs, but it's fine as a relative health signal).

### `train.py` — training entry point (verified)
- Top: all default config; `configurator.py` applies `--flag=value` CLI
  overrides; `min_lr` auto-set to `adamw_lr/10` unless passed explicitly.
- Model init: `scratch` vs `resume`. **Resume only restores 6 keys** — pass MoE
  flags (`--n_exp=8` …) on the CLI when resuming.
- **Two-optimizer split** (~line 252): Muon ← exactly-2D attn/dense-MLP weights;
  AdamW ← embeddings/norms/router/3-D experts/biases. Asserts no overlap/gap.
- `get_lr` — cosine schedule w/ linear warmup; returns the AdamW LR.
  `muon_adamw_ratio = muon_lr/adamw_lr` scales Muon's LR by the same cosine
  factor each step (~line 362–375).
- **v10 will add here:** `--use_mup`, `--base_width`, `--width`, and per-group
  `1/m` LR multipliers for hidden + readout groups (`dax.md` §3.1).

### `muon.py` — Muon optimizer (verified, FINAL — reuse unchanged)
Newton-Schulz orthogonalized momentum; update spectral norm ≈ `lr·sqrt(max_dim)`;
decoupled weight decay (Moonlight). Header documents the 3 fixed filter bugs
(gate→router, ndim>=2→ndim==2, embed→wte/wpe). **This self-normalization is why
Muon's µP LR scaling must be checked empirically, not assumed** (`dax.md` §3.4).

### `manager.py` — `MANAGER` global (referenced by model.py)
Accumulates aux loss and router-z loss across MoE layers each forward; `train.py`
/ `model.py` aggregate and reset per step. Look here if MoE loss terms misbehave.

### `configurator.py` — CLI override shim (referenced by train.py)
`exec`'d inside train.py; turns `--key=value` args into globals. No real
argparse — keep that in mind when adding µP flags.

---

## Data pipeline (DONE — reuse, do not re-run)

### `prepare_data_chunked.py` — main data pipeline (verified header)
v9 tokenizer (7 ChatML tokens, vocab 50264), tiered 6-stream chat interleave,
`pyarrow_streamer` for synthetic, writes `meta.pkl`. Already run.

### `generate_reasoning.py` — synthetic generator v2.1 (verified header)
Multiprocessed Parquet, ChatML 7-token format, ~1M examples (~257M tok ≈ 2% of
budget). Already run → `./synthetic_pretrain_v2/`. (Note: the v9 *context* doc
calls the new generator `generate_pretrain_v2.py`; the actual file in the repo
is `generate_reasoning.py` carrying the v2.1 ChatML header — this file is the
current generator, not the superseded one.)

### Inherited data utilities (from v9 docs; not all directly inspected here)
- `filter_chat_data.py` — NLA pre-filter (Hermes/Orca/WizardLM → tier1/tier2).
- `safe_merge_v9.py` — merges chunks into `data/moe_v9_pretrain/`.
- `restore_local_data_v9.py` — re-downloads source data if the volume is lost.

---

## Utilities & evaluation (from v9 docs; reused unchanged in v10)

| File | Purpose | Notes |
|---|---|---|
| `keep_checkpoints.py` | Watches `out/ckpt.pt`, saves versioned copies, keeps last 5 | Verified. Run in a 2nd terminal during training. |
| `inspect_routing.py` | MoE routing diagnostics | Use the 200-token multi-topic prompt for representative results. |
| `eval_nanomoe.py` | lm-eval-style harness w/ v9 tokenizer | Benchmarks read above noise only after ~10k steps. |
| `final_eval.py` | Final benchmark runner | |
| `compare_models.py` | Side-by-side v1 vs v9/v10 + significance tests | Needs `scipy`. |
| `nla_test.py` | NLA preference discrimination + MCQ | Meaningful only post-SFT. Needs `scipy`. |
| `test_inference.py` | Inference smoke test | Induction-head section is high-variance; average 5–10 runs. |
| `build_sft_v5.py` | SFT (Muon + wd 0.1 + ChatML, 500–1000 steps) | Runs after pretrain. |

---

## New files to create for v10 (do not exist yet)

| File | Purpose | Spec |
|---|---|---|
| `model_mup.py` | µP variant of the model (readout `1/m` multiplier, µP init, manual MoE-tensor/router handling). *Or* a `use_mup` flag inside `model.py`. | `dax.md` §3.1–3.3 |
| `mup_coord_check.py` | Verify per-layer activation RMS is flat across widths 128/256/512 (the µP sanity test). | `dax.md` §5 step 1 |
| `mup_transfer.py` | Compute scaled HPs for width 768 from the proxy optimum. | `dax.md` §9 step 6 |
| (proxy launch configs) | `torchrun` commands / config for the 256 & 128 sweep runs. | `dax.md` §5 |

---

## Output / runtime artifacts
- `out/ckpt.pt` — latest checkpoint (overwritten each eval); `out/ckpt_NNNNN.pt`
  — versioned copies from `keep_checkpoints.py`.
- `./synthetic_pretrain_v2/` — generated synthetic Parquet (done).
- `data/moe_v9_pretrain/` — `train.bin`, `val.bin`, `meta.pkl` (done).
- `wandb/` — run logs (project `nanoMoE-v9`; consider a `nanoMoE-v10` project
  for the new runs).

---

*End of index — nanoMoE v10*
