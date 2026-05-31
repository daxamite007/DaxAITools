# nanoMoE v10 — Current Build State

Last updated: 2026-05-31 — **v10 just starting.** No v10 code written yet.
This doc captures DECISIONS MADE and WORK COMPLETED only. Read `dax.md`
(context) and `nanoMoE_v10_index.md` (file map) alongside it.

v10 = v9 codebase + Maximal Update Parametrization (µP) for cheap, transferable
hyperparameter tuning. See `dax.md` §1 for the rationale (kill the dead-run
tax that dominated v9 cost).

---

## Status at a glance

| Area | State |
|---|---|
| v9 model / data / muon.py / eval harness | ✅ Reused unchanged (see Inherited below) |
| µP design | 🟡 Specified in `dax.md`, not yet implemented |
| `model_mup.py` | ⬜ Not started |
| `train.py` µP flags | ⬜ Not started |
| `mup_coord_check.py` | ⬜ Not started |
| Proxy sweep | ⬜ Not started |
| Target 322M run | ⬜ Not started |
| **Open decisions** | 🔴 2 blocking — must resolve first (below) |

---

## 🔴 Open decisions — resolve BEFORE writing `model_mup.py`

### D1 — Weight tying (affects param count, token budget, and `model_mup.py`)
v9 ties `wte.weight = lm_head.weight`. µP wants input-embedding and readout
treated differently. See `dax.md` §3.2.
- **(A) Keep tied** — preserves 322M total + the `40×` token math. Recommended.
- **(B) Untie** — cleanest standard µP, but adds ~38.6M params → ~360M total,
  breaking the 322M label and the token budget.
- **DECISION:** _pending_ — default to **(A)** unless re-baselining the budget.

### D2 — Target token budget (affects cost and whether new data is built)
See `dax.md` §6.1.
- **Path 1 — 12.88B** (reuse v9 data, ~$25–35 all-in). Recommended default.
- **Path 2 — 25–40B** (build more data, ~$65–95, 2–3× compute).
- **DECISION:** _pending_ — default to **Path 1**.

---

## Decisions already locked (from `dax.md`)

| Decision | Value | Why |
|---|---|---|
| µP axis | **width (`n_embd`) only** | width-transfer is the supported µP regime |
| `head_dim` | **fixed at 64** across all widths | avoids needing the µP attention-scale change |
| Proxy widths | **256 primary, 128 secondary** | ≥2 widths required to coord-check transfer |
| Held identical proxy↔target | depth=12, n_exp=8, top_k=2, stride=2, block=4096, routing config | µP only holds when these are equal |
| Routing / aux config | **frozen at v9 values** (aux 0.02, z 0.001, switch init, full-prec router) | validated in v9, not part of width theory — do not sweep |
| Sweep variables | `adamw_lr`, `muon_lr`, `weight_decay`, global batch | the µP-transferable knobs |
| Implementation route | **manual µP** (MoE tensors need hand-handling regardless) | `mup` pkg won't touch 3-D `MLPExperts` params |
| Muon LR scaling | **verify empirically** (coord-check @512 before 768) | Adam-µP rule does not directly apply to Muon — see `dax.md` §3.4 |
| Token budget default | Path 1 (12.88B) | robustness from correct HPs, not more tokens |

---

## Inherited from v9 (reused as-is — do NOT rebuild)

### Data — COMPLETE, model-size agnostic
| Artifact | Location |
|---|---|
| train.bin (~12.4B tok) / val.bin (~477M tok) | `data/moe_v9_pretrain/` |
| meta.pkl → `{'vocab_size': 50264}` | `data/moe_v9_pretrain/meta.pkl` |
| Binary chunks (12.88B tok, 26 chunks) | `./moe_scaled_dataset_000.bin`–`025.bin` |
| NLA proxy model | `./nla_45k_proxy_embeddings.lgb` |

Do not re-run any data script. Recovery path (only if volume lost):
`restore_local_data_v9.py` → `prepare_data_chunked.py` → `safe_merge_v9.py`.

### Code — final / reused
| File | Status for v10 |
|---|---|
| `muon.py` | ✅ Final. Reused unchanged. Param-split filter correct; 3 historical bugs fixed. |
| `model.py` | ✅ Baseline. v10 adds µP via `model_mup.py` (or a `use_mup` flag). Do not break the standard path. |
| `train.py` | 🟡 Will gain `--use_mup`, `--base_width`, `--width`, per-group LR multipliers. Two-optimizer split stays. |
| `prepare_data_chunked.py` | ✅ Done + run. Not touched in v10. |
| `generate_reasoning.py` | ✅ v2.1 synthetic generator (ChatML, 7 tokens) — already run → `./synthetic_pretrain_v2/`. |
| `keep_checkpoints.py` | ✅ Reused. Run it in a second terminal during the target run. |
| eval harness (`eval_nanomoe.py`, `compare_models.py`, `nla_test.py`, `test_inference.py`, `inspect_routing.py`, `final_eval.py`) | ✅ Reused unchanged. |
| `build_sft_v5.py` | 🟡 Same v9 plan (Muon + wd 0.1 + ChatML + 500–1000 steps). Runs after pretrain. |

### Environment (target pod)
- **Python 3.12, PyTorch 2.11+cu130, no flash-attn** (SDPA only; flash-attn
  fails on CUDA 13 — do not install).
- Working dir: `/workspace/nanoMoE`. DDP: `torchrun --standalone --nproc_per_node=N`.

### Carried-over hard lessons (still binding in v10)
- **Routing health thresholds** (v9 state): step 500 ≤35%, 1000 ≤30%, 2000 ≤25%,
  4000+ ≤20%. Inspect with the 200-token multi-topic prompt, not a greeting.
- **Resume LR rule:** never extend `lr_decay_iters` without patching `iter_num=0`
  and lowering the peak LR. (Killed r4-cont-BROKEN.)
- **`n_exp` on resume:** MoE flags are not all restored from checkpoint — pass
  `--n_exp=8` etc. on the CLI when resuming or you initialize a dense model.
- **val loss is the primary health metric** at this scale; benchmarks only read
  above noise after ~10k steps.

---

## Work completed in v10 so far
- ✅ `dax.md`, this file, and `nanoMoE_v10_index.md` written.
- ⬜ Everything else.

---

## Immediate next steps
1. Resolve **D1** and **D2** above (record the choices in this file).
2. Write `model_mup.py` per `dax.md` §3.1–3.3 (µP init, readout `1/m`
   multiplier, manual `MLPExperts` + router µP handling).
3. Write `mup_coord_check.py`; confirm activation RMS is flat across widths
   128/256/512. **Do not sweep until the coord check passes.**
4. Proxy LR sweep at width 256 (spot-check 128), 1–2B tokens each, 1 epoch.
5. `mup_transfer.py` → scaled HPs for width 768.
6. Intermediate coord-check + short run at width 512 to validate the **Muon**
   LR scaling specifically (the one high-risk transfer; `dax.md` §3.4).
7. Launch the 322M target run (Path 1 budget) only after step 6 passes.

---

## Cost tracker (fill in as runs happen)
| Phase | Hardware | Tokens | Time | $ | Notes |
|---|---|---|---|---|---|
| Proxy coord-check | — | — | — | — | |
| Proxy LR sweep | — | ~1–2B × N | — | — | |
| Width-512 validation | — | short | — | — | |
| Target 322M (Path 1) | — | 12.88B | — | — | |
| SFT | — | — | — | — | |

Estimated all-in (Path 1): **~$25–35** (see `dax.md` §6.2). This is *not*
cheaper than v9's single good run alone — the savings are in **not paying for
dead runs** like v9's r1/r2/r3/r4-cont-BROKEN.
