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
