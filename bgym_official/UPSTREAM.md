# Vendored from the BindingGYM repository

Source clone: `/home/guoj0f/repos/BindingGYM`
Upstream commit: `752c612a286720d98f31638ffbd8ea16911e2249` (2025-07-29T00:02:15+08:00)
Vendored on: 2026-08-25T17:00:57+03:00

## Why vendored rather than referenced

Two reasons, and the second is the operative one.

1. ibex-usage §1c-3 requires everything a task consumes to live physically inside the working
   branch, so that per-branch Ibex sync and branch isolation both hold.
2. **The point is control.** Having the official pipeline in-tree lets us slot new model
   experiments into a pipeline whose numbers we have already reproduced, instead of maintaining a
   parallel reimplementation and then arguing about which half is wrong. (user, 2026-08-25)

## What was copied

| Path | Purpose |
|---|---|
| `training/main.py` | fine-tuning entry point (`--model_type structure --mode inter --split cluster`) |
| `training/dataset.py` | `StructureDataset`: same-assay batching via `seed = index // batch_size` |
| `training/loss.py` | `listMLE` |
| `training/utils.py` | `DMS_file_for_LLM` |
| `training/DEMEmodel.py` | imported at module level by main.py; pulls torch_geometric + torch_scatter |
| `training/protein_mpnn_utils.py` | **modified** ProteinMPNN whose `forward(data)` takes a PyG batch — this is the fine-tuning model |
| `training/cache/v_48_020.pt` | pretrained ProteinMPNN weights (same file H3-DDG uses as its backbone) |
| `training/cache/BindingGYM_cluster.tsv` | MMseqs2 cluster table the inter-assay split is built from |
| `baselines/protein_mpnn/*` | zero-shot scoring; **vanilla** ProteinMPNN with the positional `forward(X, S, ...)` |
| `install.sh`, `BindingGYM.yml` | env recipe |

Not copied: `main-OHE*.py`, the ESM/EVE/SaProt/PIFold/PPIformer baselines, notebooks, and the
published `results/` csvs (those are read in place from the clone for reference values only).

## Local modifications

Any change we make to these files is recorded in `PATCHES.md` next to this file, so the delta
against upstream stays explicit.

## Data

`input/` is NOT vendored into git (376,446 rows + 22 structures). It is copied from this
branch's own `data/input/`, which passed the full audit recorded in the sibling project
(`bindingGYM-reproduce`, §3.1b: 0 unresolved mutations, 0 wild-type mismatches).
