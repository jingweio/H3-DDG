# Local modifications to the vendored BindingGYM code

Every change we make to `bgym_official/` is listed here, so the delta against the upstream
commit recorded in `UPSTREAM.md` stays explicit. Each edit is also marked `# LOCAL PATCH` in the
source.

## `training/main.py`

### 1. `--fold` (default `-1`, i.e. unchanged upstream behaviour)

Upstream loops all five folds inside one process. On Ibex that means one job whose walltime is
the sum of five folds — and the folds differ by ~7x in cost here, because the per-epoch cost is
dominated by evaluating the held-out fold and f0/f3 hold the large structures (KRAS_PICK3CG 915
residues, 4D5 1041, SARS2-RBD 791) while f1/f2 hold small ones (Z-domain 109-116, PSD95 120).
Measured per-epoch, A4500, batch 8: f0 21.9 min, f1 3.3, f2 3.1, f3 22.2, f4 8.5.

A single walltime covering all five is therefore either wasteful or fatal, and short walltimes
matter on this account: 7h jobs have started in ~11h while 16-18h jobs sat pending for 5 days.

**No merge step is needed.** Upstream already writes one `{DMS_id}_oof.csv` per assay, and under
the inter-assay split each assay is held out in exactly one fold, so five jobs sharing
`--tmp_path` produce the same file set as one job doing all five.

### 2. Per-fold log filename

`log = open(output_path + 'train.log','w')` truncates. With five jobs sharing `--tmp_path` they
would clobber one another's log, so the name becomes `train_fold{N}.log` whenever `--fold` is
given. Unchanged when `--fold` is omitted.

## Not modified

`dataset.py`, `loss.py`, `utils.py`, `DEMEmodel.py`, `protein_mpnn_utils.py`,
`baselines/protein_mpnn/*` — byte-identical to upstream. In particular the sampler
(`seed = index // batch_size`, same-assay batches), `listMLE`, the AdamW/OneCycleLR schedule,
`epochs = 100`, `patience = 3`, and the selection on the held-out fold's Spearman are all
untouched: this project reproduces their protocol, including its selection-on-test-fold property.

### 3. Cache `DMS_file_for_LLM`

`DMS_file_for_LLM` runs `.apply(eval)` on `wildtype_sequence`, `mutant` and `mutated_sequence`
— the last holds whole protein sequences, so this parses tens of MB of Python literals — and then
walks every row with scalar `df.loc[i, col]` access. Measured on the fold-2 smoke
(`50852061`): **~80 min** of startup before the first epoch, against ~6.7 min per epoch
afterwards.

Every job repeats it in full, because the loop preprocesses all 25 assays regardless of which
fold is being trained. With `--fold` splitting the run into five jobs that is ~6.7h of pure
repeat work.

The cache key is the source csv's `(mtime, size)` plus `focus`, so editing the data or switching
`--model_type` invalidates it instead of silently serving a stale frame. Writes go to a
pid-suffixed temp file then `os.replace`, since five concurrent jobs can race on the same key.
The frame produced is identical; no protocol behaviour changes.

### 4. `--resume`

Upstream has no checkpointing, so a job killed at walltime loses the entire fold. That is fatal
rather than merely wasteful here, because patience-3 early stopping requires the full held-out
fold to be scored **every** epoch. Measured on f3 (`50878825`, 142,905 rows, structures up to 1041
residues): **~1.6h per epoch**, so f0/f3 need 24–32h to reach a plausible stopping point — past any
walltime that schedules on this account (16–18h jobs have sat pending five days; 7h has started in
~11h). The f3 measurement job demonstrated the failure directly: it timed out having completed one
epoch, and kept nothing.

State is written after every epoch, to a pid-suffixed temp file then `os.replace`, so a kill
mid-write cannot corrupt the only copy. Saved:

| field | why it must be saved |
|---|---|
| `model`, `optimizer`, `scheduler` | the obvious three; `scheduler` matters because OneCycleLR's position in its cycle sets the LR |
| `best_model` | the weights the fold will emit |
| **`best_valid_pred`** | what the OOF is written from (`fold_valid['pred'] = best_valid_pred`). Restoring weights without it would finish the run and emit an empty prediction column |
| `best_valid_metric`, `not_improve_epochs` | the early-stopping state; without them a resumed run would restart the patience counter and train longer than an uninterrupted one |
| `rng_torch`, `rng_numpy`, `rng_cuda` | the model draws a fresh decoding order from the global torch RNG on every training forward (`if self.randn is None or self.training: randn = torch.randn(...)`). Replaying that stream from scratch puts a resumed run on a different trajectory. The same fix was needed, and verified, on the H3-DDG side of this repo |

One asymmetry handled explicitly: upstream's patience check sits at the **end** of the loop body,
so a run resumed after early stopping had already fired would train one more epoch before
noticing — and if that epoch beat `best`, it would change which weights the fold selects. Resume
therefore checks `not_improve_epochs >= patience` before entering the loop and goes straight to
output.

`--resume` defaults off, so behaviour without it is unchanged.
