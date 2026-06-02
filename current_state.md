# nanoMoE v10 — Current State

Fresh, from-scratch run. Training from random initialization — no donor model, no
prior checkpoint, no continuation. Only code is carried in. This doc tracks what's
done and what's next; read `context.md` for the full plan.

**Right now:** environment is up, all data is downloaded and prepared, synthetic
data is generated, chat data is filtered. **Next step: build the first tiny
(proxy) v10 model for the µP sweep.**

---

## Status at a glance

| Area | State |
|---|---|
| Environment (Py 3.12 / torch 2.11+cu130 / no flash-attn) | ✅ Done |
| Raw data downloaded (code / web / longform / chat sources) | ✅ Done |
| Synthetic data generated | ✅ Done |
| Chat data filtered (Hermes NLA; Infinity/Magpie heuristics) | ✅ Done |
| Tokenized corpus assembled (`prepare_data_chunked.py`) | ⬜ Pending stream update (below) |
| `model_mup.py` (µP model) | ⬜ Not started — **next** |
| `mup_coord_check.py` | ⬜ Not started |
| Proxy µP sweep | ⬜ Not started |
| 322M target run | ⬜ Not started |

---

## Completed

### Environment
Python 3.12, PyTorch 2.11+cu130, SDPA attention (no flash-attn). Working dir
`/workspace/nanoMoE`. 2× A5000.

### Data — downloaded & staged
| Bucket | Source | Location |
|---|---|---|
| LongForm | Cosmopedia | `./local_data/longform/` |
| Code | StarCoderData (Python) | `./local_data/code/` |
| Web | FineWeb (sample) | `./local_data/web/` |
| Synthetic | `generate_reasoning.py` output (ChatML, 7 special tokens) | `./synthetic_*/` |

### Chat data — filtered (results from this run)
| Source | Method | Input rows | Kept | Rate |
|---|---|---|---|---|
| OpenHermes-2.5 | NLA proxy (tier1 ≥0.5 / tier2 ≥0.4) | 1,001,551 | **233,271** (t1 163,802 / t2 69,469) | 23.3% |
| Infinity-Instruct (7M) | heuristic | 7,449,106 | **4,083,144** | 54.8% |
| Magpie-Pro-300K-Filtered | heuristic | 300,000 | **299,995** | 100% |

Total chat examples kept: **~4.62M**. Output dirs:
`./local_data/chat/hermes_tier1`, `hermes_tier2`, `./local_data/chat/infinity`,
`./local_data/chat/magpie`.

Notes on the chat filtering:
- The NLA proxy is used **only** for Hermes (validated in-distribution). Infinity
  and Magpie use heuristics — the proxy mis-ranks them (FLAN stubs on Infinity,
  near-total rejection on Magpie). See `context.md` §6.1.
- Infinity's heuristic drops were healthy: ~2.58M removed as **short** (these are
  the FLAN-style one-line/stub answers — the length check catches them, so the
  separate stub counter reads 0), ~787k as **duplicates**, ~2.7k as
  boilerplate-only. The kept 4.08M is the substantive, longer Infinity content.
- Infinity here is the **7M** subset. The heuristics removed its stub tier, so
  it's usable; the `Gen` subset (~1M, more conversational) remains an option if
  more dialogue-style data is wanted later.

---

## Locked decisions

| Decision | Value |
|---|---|
| Run type | **fresh, from random init** — nothing continued |
| Target architecture | `n_layer=12, n_head=12, n_embd=768, n_exp=8, top_k=2, stride=2, block_size=4096`, ~322M total |
| µP axis | width (`n_embd`) only; depth/experts/routing fixed |
| `head_dim` | fixed 64 across all widths (avoids the µP attention-scale change) |
| Proxy widths | 256 primary, 128 secondary (coord-check needs ≥2) |
| Optimizer | Muon (2-D attn/dense-MLP weights) + AdamW (embeddings/norms/router/3-D experts/biases) |
| Routing config | aux 0.02, router-z 0.001, switch init, full-prec router — fixed, not swept |
| Chat filtering | Hermes → NLA proxy; Infinity/Magpie → heuristics |
| Sweep variables | `adamw_lr`, `muon_lr`, `weight_decay`, global batch size |
| Muon LR transfer | verified empirically (coord-check @512 before 768) |

---

## Open decisions

1. **Weight tying for µP** (`context.md` §3.2): keep tied + `1/m` logit multiplier
   (stays 322M) vs untie (cleaner µP, ~360M). Resolve before `model_mup.py`.
2. **Final token budget**: ~13B baseline vs extend toward 30–40B (the chat pool
   now supports it). Decide before the full run, not the proxy.
3. **Full-run chat yield weights**: the kept sources are very uneven (Infinity
   4.08M, Magpie 300k, Hermes 233k). The mix ratio is enforced regardless, but at
   40/30/30 the small sources repeat ~12–15× at a 13B budget while Infinity
   repeats ~once. Not relevant to the proxy; decide for the full run (use
   `chat_mix_report.py` to set weights with real numbers).

---

## Immediate next steps
1. Resolve the weight-tying decision (open #1).
2. Write `model_mup.py` — µP init, readout `1/m` multiplier, manual µP for the
   3-D expert tensors and router (`context.md` §3.1).
3. Write `mup_coord_check.py`; confirm per-layer activation RMS is flat across
   widths 128 / 256 / 512. Do not sweep until this passes.
4. Update `get_tiered_chat_stream()` in `prepare_data_chunked.py` to read the new
   dirs (`hermes_tier1/2`, `infinity`, `magpie`), then run `prepare_data_chunked.py`
   to assemble the tokenized corpus.
5. Proxy µP sweep at width 256 (spot-check 128), 1–2B tokens each, 1 epoch.

---

## Health checks to watch (once training starts)
- **Routing balance** is the primary risk during early steps — watch the max
  expert load and aux loss; a healthy router spreads tokens rather than collapsing
  onto one expert.
- **Validation loss** is the main quality signal at this scale; downstream
  benchmarks only read above noise after several thousand steps.
- **Coord check before trusting transfer** — if activation RMS drifts with width,
  µP is mis-wired; fix before spending sweep compute.
