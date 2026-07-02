# DaxAITools — Repository Overview

A file-by-file map of **what is actually in this repository**. Two things live
here: a set of **planning/context docs** for the *nanoMoE v10* language-model
build, and the **`NLA_Filter/`** pipeline that produces the data-quality filter
those docs refer to.

> Note: the nanoMoE training code itself (`model.py`, `train.py`, `muon.py`,
> etc.) is **not** in this repo — it lives in the separate `/workspace/nanoMoE`
> working tree. The markdown docs below describe and plan that code; only the
> `NLA_Filter/` pipeline is present as runnable source here.

---

## Root files

| File | What it is |
|---|---|
| `OVERVIEW.md` | This file — the top-level map of the repository. |
| `context.md` | The **full build plan** for nanoMoE v10: the µP (Maximal Update Parametrization) strategy, proxy→target hyperparameter transfer, architecture, optimizer split (Muon + AdamW), data mix, cost estimate, and the ordered build sequence. Read this first for the *why*. (Formerly `dax.md`.) |
| `current_state.md` | The **living status doc**: what's done, what's blocked, locked decisions, open decisions, and immediate next steps. Read this for *where things stand right now*. (Formerly `nanoMoE_v10_current_state.md`.) |
| `nanoMoE_v10_index.md` | A **code map** for the nanoMoE v10 codebase — a "where do I look for X" table pointing at the model, training loop, optimizer, and data-pipeline files in the `/workspace/nanoMoE` tree. Useful once you're working in that codebase. |

---

## `NLA_Filter/` — the NLA proxy filter pipeline

Builds a fast, serverless-friendly quality filter for chat/instruction datasets
(OpenHermes, WizardLM, etc.). A heavy autoregressive critic sets a "golden"
quality threshold on a sample; a lightweight LightGBM + MiniLM proxy is then
trained to replicate those decisions at scale. See `NLA_Filter/README.md` for
the full write-up.

| File | What it does |
|---|---|
| `README.md` | Detailed description of the pipeline, its contents, and the 4-step workflow (extract → score → train → deploy). |
| `pull_wizard_sample.py` | **Step 1 — data extraction.** Reads `wizardlm_converted.jsonl` and pulls a reproducible random sample (500 rows, seed 42) into `wizard_500_sample.jsonl` for threshold discovery. |
| `discover_wizard_threshold.py` | **Step 2 — the heavy lifter.** Runs the sample through a large AR value critic (Qwen2.5-7B-Instruct + a custom value head), extracts the final-token hidden state, projects it, and computes the L2-normalized magnitude to find the quality "cliff" / threshold. Writes scored JSONL. |
| `train_proxy_embeddings.py` | **Step 3 — the fast proxy.** Embeds text with `all-MiniLM-L6-v2` (384-dim), appends structural heuristics (char length, word count), and trains a LightGBM classifier via 5-fold stratified CV to mimic the AR critic's scores instantly. Outputs the `.lgb` model. |
| `nla_45k_proxy_embeddings.lgb` | **Step 4 — the deployable artifact.** The compiled LightGBM proxy filter used to curate millions of rows cheaply before SFT. (This is the same model `context.md` §6.1 uses to filter the Hermes chat bucket.) |

---

## Workflow at a glance

```
NLA_Filter pipeline (build the filter)          nanoMoE v10 (use the filter)
────────────────────────────────────           ────────────────────────────
pull_wizard_sample.py                           context.md   → the plan
        │                                       current_state.md → status
        ▼                                       nanoMoE_v10_index.md → code map
discover_wizard_threshold.py  (AR critic)                 │
        │                                                 ▼
        ▼                                        filter chat data with
train_proxy_embeddings.py                        nla_45k_proxy_embeddings.lgb
        │                                                 │
        ▼                                                 ▼
nla_45k_proxy_embeddings.lgb  ──────────────────►  train the 322M MoE model
```
