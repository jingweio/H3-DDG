# BGYM-CLIFF v1 — implementation spec

**Mutation-cliff / interaction-cliff audit of BindingGYM binding landscapes**
Base repo: `/home/guoj0f/repos/H3-DDG/.claude/worktrees/bindingGYM-cliff-design` (`<R>` below), branch `bindingGYM-cliff-design`.
Data: `$BINDINGGYM_INPUT=/home/guoj0f/share/BindingGYM/input`. CPU-only, local, no scheduler, no GPU.

---

## 0. Which design is the base, and why (the judges split 1-1-1)

| judge | best | sali | landscape | user-clusters |
|---|---|---|---|---|
| validity | landscape | 7 | **8** | 6 |
| feasibility | sali | **9** | 7 | 5 |
| answers-the-claim | user-clusters | 7 | 7 | **8** |
| total | — | 23 | 22 | 19 |

**Decision: the spine is `landscape`; the compute architecture is `sali`; `user-clusters` contributes a gated supplementary channel and the object-level deliverable.** Reasons, in order:

1. **`sali`'s 1-point aggregate lead is entirely computational, and computation is portable.** Its cheapness comes from three algorithms — wildcard bucketing, closed-form Gini mean difference, sampled high-distance moments. All three are grafted in below verbatim. Nothing is lost by not making it the spine.
2. **`landscape`'s two core assets are NOT portable and are demanded by two of the three judges' `must_keep` lists.** (a) The exact variogram `V(h)` over the whole `h` axis is the only statistic in any design that tests the user's literal sentence — "按照不同的 mutation 程度呈现相对 smooth 的变化" — whose subject is mutation degree as an axis; `sali` and `user-clusters` stop at `d=1,2,3` and `η²` respectively. (b) `σ_ε = 0.1243`, measured from 10,868 twice-measured KRAS site-pairs, is the only noise floor in the benchmark that is measured **for the quantity actually being tested** (ε, not y). The answers-judge and the validity-judge both require these as primary.
3. **`landscape` is the only design whose C2 verdict does not need the missing noise floor** (tail-quantile ratios are invariant under `c → λc`), which is the decisive property given that all 28 files have zero duplicate genotypes.
4. **`sali` contains one algebraic error at its C3 core**: `Cov(r_e, mean sibling r)` is *not* exactly zero under i.i.d. noise, because both residuals subtract the same in-sample `β̂_i`. Fixed here by cross-fitting `β̂` and by calibrating the slope against a surrogate null instead of against an analytic 0.
5. **`user-clusters` is right that the pair form cannot express the user's claim in fixed-depth libraries**, and right that its own O(n²) Ward is unaffordable on GB1 (34.5 GB condensed, 4.314e9 > int32). Resolution: the cluster channel runs **only** on `n ≤ 30,000` assays where the pair channel is powerless — which removes the memory problem entirely (peak 3.6 GB) and keeps the channel exactly where it adds information.

`landscape`'s own defects are fixed, not inherited: the exact all-pairs variogram (4.31e9 pairs on GB1, 11.9e9 total) is replaced by exact `h=1,2` + closed-form `V(∞)` + a seeded 2e7 random-pair sample for `h≥3`; `np.add.at` (5–20 M elem/s) is replaced by `np.bincount`; nulls never recompute a full variogram.

---

## 1. Verdict logic

All statistics are **per assay**. Nothing is ever pooled across assays. Aggregation is over **7 structure-independent families** (§6), with the explicit statement that a 5-of-7 sign test has binomial p = 0.2266 under a 0.5 null and is therefore a *generality* statement, never the evidence — the evidence is the per-family CIs and a cluster-df random-effects meta-analysis.

### 1.0 Definitions (frozen; `cliff/config.py`)

- **Canonical key** `K(v) = tuple(sorted((chain, seq_pos, aa_mut)))`. **The chain label is mandatory** — dropping it is what manufactured the fake Z-domain noise floor (§1.1 G3).
- **Hamming distance = mutation-set distance** (verified 508,962/508,962 rows: `mutated_sequence` reconstructs exactly, 0 indels, 0 identity mutations, 0 X-hits). So all pair work uses the cheap `mutant` column.
- **Nested pair** `(B, B∪{i})`: `|K_u Δ K_v| = 1`. **Same-site swap** `(B∪{i→a}, B∪{i→b})`: `|K_u Δ K_v| = 2` with identical `(chain,pos)`. **These two sets are never merged in any statistic.** 1,678,963 nested vs 2,602,669 same-site across 28 files; in GB1_1FCC 184,735 vs 861,874, so merging would mislabel 82% of pairs as epistasis.
- **Design matrix** `X ∈ {0,1}^{n×M}`, `M` = number of distinct observed `(chain, seq_pos, aa_mut)`; row nnz = `num_muts`.
- **Latent scale.** `y = g(φ)`, `φ = Xβ`, `g` monotone. Fit by alternation, 10 iterations, convergence `max|Δβ| < 1e-6`:
  1. `β = scipy.sparse.linalg.lsqr([1|X], z)`; `φ = [1|X]β`
  2. `g = sklearn.isotonic.IsotonicRegression(out_of_bounds='clip').fit(φ, y)`
  3. `z = g⁻¹(y)`, where `g⁻¹` is linear interpolation over the strictly-increasing hull of the PAV breakpoints, clipped to `[min φ, max φ]`.
  **Censored rows** enter step 1 through a Tobit E-step only: `z_i = φ_i − σ(φ_i)·pdf(a)/Φ(a)`, `a = (g⁻¹(L) − φ_i)/σ(φ_i)`. They are excluded from every pair statistic regardless.
- **Cross-fitting (mandatory).** 5 folds over variants, fixed seed 20260902. `φ_i^{oof}`, `z_i^{oof}`, `e_i = z_i^{oof} − φ_i^{oof}` are computed from a fit that never saw variant `i`. In-sample residuals are a sensitivity column only. Reason: in-sample LS residuals are shrunk and their tails distorted worst at rarely-sampled substitutions — exactly the positions most likely to yield a large deviation.
- **Level-dependent noise scale.** `σ̂(φ)` = `1.4826 × MAD(e)` within each of 20 equal-count `φ` bins, linearly interpolated. **MAD, never SD** (Z-ZSPA1-LL1 has sd 0.140 against range 4.871).
- **Cliff statistic.** For nested pair `p = (B, B∪{i})`:
  `ĉ_p = (e_{B∪i} − e_B) / sqrt(σ̂²(φ_B^{oof}) + σ̂²(φ_{B∪i}^{oof}))`.
  Note `e_{B∪i} − e_B = Δ_i(B) − β̂_i` on the latent scale — i.e. **background-relative epistasis**, the quantity Weinreich 2005 / Ferretti 2016 / Poelwijk 2016 already define. This must be written in the record: the cliff is the tail of a known quantity, not a new index.
- **PRIMARY NESTED SET `P_a`**: nested pairs with (a) `B ≠ ∅` (excludes the WT hub — 100% of nested edges in the 5 singles-only assays, 35.6% in CD19, 16.1% in SARS2-RBD, ~7% in KRAS, all sharing one measured `y_WT`); (b) neither endpoint at a detected censoring level; (c) finite `φ^{oof}` at both endpoints; (d) assay in tier PRIMARY or ARM.
- **Score quantum** `q_a` = modal spacing of sorted unique `y`. **Grid guard**: any `τ` whose absolute cut `< 3q_a` is dropped from the sweep (binds on SARS2-RBD `q=0.01`, CR9114-H3 `q=0.1`).
- **Sign convention — RESOLVED.** All 28 assays are **higher = better binding**. The `confounds` profiling agent asserted that logKd assays are inverted; that is wrong: the scale is `−log10(Kd)`, and CR9114-H1 goes germline 8.425 → fully matured 9.592, CR6261 7.000 → 9.507. Use higher = tighter = better throughout. Record this resolution as a KEY DECISION.
- **hYAP65 is log-transformed before anything else** (`log10`; strictly positive fold-change with WT exactly 1.000, min 0.00911, max 15.56). No other assay is transformed.
- **`σ_noise` registry with provenance** (`cliff/noise.py`), four provenance values and every headline number recomputed at `σ × {0.5, 1, 2}`:

| family | σ_y | σ_ε | provenance | source |
|---|---|---|---|---|
| KRAS (5 assays) | 0.148 | **0.1243** | `measured_replicate` (upper bound) | KRAS_RAF1_6VJJ vs RAF1-RBD_6VJJ, 10,868 shared site-pairs, r = 0.812, OLS slope 0.6445 removed by affine alignment, residual sd 0.1479 → 0.1479/√(1+0.6445²) |
| GB1 (2 assays) | 0.129 | — | `cross_study_contaminated` | 160 shared variants, r = 0.9789, residual sd 0.1826; **chain-C pos 2 WT differs (Q vs T) → two backgrounds, not two measurements** |
| all others | `σ̂(φ)` | — | `internal_residual` | assay's own cross-fitted residual MAD per φ-decile; conservative (attributes all epistasis to noise) |
| cross-check, all | 0.20 × MAD-scale | — | `stipulated` | imported from the GB1 ratio; never primary |

  **FORBIDDEN**: the Z-domain within-genotype SDs (0.1695 / 0.4105 / 0.3064 / 0.5091). They are a chain-key collision artefact — G3 proves it.

---

### 1.1 Stage G — gates and kill criteria (run FIRST, before any observed statistic is read)

Every gate has a hard consequence. `cliff/calibrate.py` + `tests/test_cliff_invariants.py`.

| id | check | pass condition | consequence of failure |
|---|---|---|---|
| **G0** | pre-flight benchmark on GB1_IgG-Fc_1FCC and CR9114_FluAH1_4FQI: wall + peak RSS of (i) pair enumeration, (ii) one `fit_latent`, (iii) one null replicate | enumeration ≤ 180 s / 4 GB; `fit_latent` ≤ 4 s; one replicate ≤ 20 s | switch that assay to the `n = 40,000` × 5-independent-subsample path; **gate**: between-subsample SE of SI must be ≤ 0.03, else the assay is INCONCLUSIVE |
| **G1** | parse audit, all 28 files | 508,962 rows = 508,962 unique canonical keys; 0 wt-letter mismatches across `mutant` / `mutant_pdb` / `wildtype_sequence` / PDB residue (1,173,273 instances); 0 X-hits; 0 `*`; token counts match per chain | **STOP.** The data is not what the profile describes; nothing downstream is interpretable |
| **G1b** | BH3 cross-assay join resolution (the two profiling agents disagreed: r = 0.1709 vs +0.592) | joining 1PQ1 chain B ↔ 3KZ0 chain C via `mutant_pdb` + WT-residue identity gives **518/518** shared, not 97/518 | record which number is right and why; naive `(pos,aa)` joins are banned repo-wide |
| **G2** | twin-assay byte identity | KRAS_SOS1_8BE4 vs KRAS_DARPinK27_5O2S: 19,227 shared keys, max\|Δ\| = 0.0 on the raw score **strings** | **STOP.** The de-duplication premise has changed |
| **G3** | chain-key integrity | with chain retained: 0 duplicate genotypes in Z-LL1/LL2/HL1/HL2 (45,476 / 5,583 / 2,904 / 600 all unique). With chain dropped: 847 / 59 / 650 / 38 duplicates reappear | if duplicates appear **with** chain, the within-genotype SDs are real and become the primary noise floor for those assays — report either way, never silently |
| **G4** | **null self-calibration.** Hold out 1 of 200 N1 surrogates, score it against the other 199 | `T(τ) = 1.00 ± 0.05` for all `τ`; the 200 empirical p-values are uniform (KS p > 0.05) | **STOP.** The surrogate machinery is biased; no observed number is readable |
| **G5** | **censoring positive control.** CR9114_FluAH3_4FQY (89.05% at exactly 6.000; 451,181/524,272 = 86.1% of nested edges are floor–floor with Δ ≡ 0) | unmasked `T(4) ≥ 5`; after floor masking + Tobit, `T(4)` inside the N2 95% band AND `|P_a|` collapses ≥ 10× (524,272 → ≤ 52,000; expected ≈ 41.7k) | **STOP.** The pipeline cannot tell a detection limit from a cliff |
| **G6** | **anti-smooth negative control.** Z-ZSPA1-LL1 (SI 1.398), Z-ZSPA1-LL2 (SI 1.001) | both C1-REFUTED; both `T(4)` inside the N2 band; cliff rate must NOT be monotone in density quintile | **STOP.** The pipeline is being fooled by selection-dependent library membership |
| **G7** | **scale-mixture discrimination.** Score both the tail statistics and the localisation statistics on 200 N2c surrogates (heteroscedastic scale mixture, marginal kurtosis matched to observed) | if N2c inflates `TR` / `T(τ)` (expected), then **`C2` alone is inadmissible and the conjunction `C2 ∧ C3-L` becomes mandatory**; if N2c leaves them at 1.00, `C2` alone is admissible. Localisation statistics (`β`, ICC, `R`) must stay at their null values under N2c | this gate *sets* the verdict rule and depends only on surrogates, so it is run before observed values are read. If localisation is ALSO inflated under N2c → **STOP**, the localisation axis has no discriminating power either |
| **G8** | **power & bias.** Inject synthetic cliffs into N1 at amplitude `a ∈ {2,3,4,6}·σ̂` × rate `π ∈ {0.001, 0.005, 0.02}`, 40 reps, on 6 representative assays (GB1_1FCC, GB1_2016, KRAS_RAF1-RBD, SARS2-RBD, 5A12_VEGF, CR9114_H1) | report recovered `σ̂` bias, recovered `π̂`, and detection power at the frozen thresholds | any assay with power < 0.50 at `(a = 4σ, π = 0.005)` is stamped **UNDERPOWERED** and reports INCONCLUSIVE whatever it shows |
| **G9** | **aggregate-rule FPR.** Run the entire verdict procedure end-to-end on 50 complete N1 surrogate datasets | family-level false-positive rate of the k-of-7 rule ≤ 0.10 | tighten `k` until ≤ 0.10 and **record the change before any observed value is inspected** |
| **G10** | **censoring-mask composition.** After masking, compare the (order × degree-decile × φ-decile) composition of observed `P_a` against N1/N2 surrogate `P_a` | max absolute bin-proportion difference ≤ 0.02 | the clamp replay in the null is mis-specified; fix or flag every claim from that assay as conditional |
| **G11** | **twin-structure control.** The KRAS score table is registered against two complexes (8BE4/SOS1 and 5O2S/DARPin) with byte-identical scores | at most ONE of the two interface localisations can be causal. If the cliff positions localise to *both* interfaces with similar OR, interface localisation is declared non-causal for KRAS | reported as a finding, not a stop |
| **G-UP** | **optional upstream SE arm.** Attempt per-variant SE / read counts: Starr 2020 (SARS2-RBD `delta_log10Ka` SE + `n_barcodes`), Olson 2014 (GB1 input/selected counts), Phillips 2021 (CR9114 replicate Kd fits) | if obtained, run the standard pair z-test `z = Δy/√(se_i²+se_j²)` and report calibrated significance | if not obtained, **every C3-N verdict is stamped `conditional`** and the record must say verbatim: "effect size relative to one contaminated replicate bound, not a calibrated significance" |
| **G-OPT** | **optional structural recovery for the hypercube arm.** Fetch 4FQI / 4FQY / 3GBN from RCSB (absent from `structures/`), define sides (antibody H+L vs HA), map the 16/11 heavy-chain somatic sites | mapping must validate at 100% with the per-token offset resolution (chain H offsets are `{−4, −1, 0}`, not a constant) | skip the structural half for the arm and say so; **never** substitute a constant offset |

---

### 1.2 C1 — the landscape is smooth in mutation degree

**Statistics** (`cliff/variogram.py`)

- `V(h) = (1/2N_h) Σ_{H(u,v)=h} (y_u−y_v)²`, `G(h) = (1/N_h) Σ |y_u−y_v|`.
  `h=1,2` **exact** from the cached bucketing pair index arrays, reduced with `np.bincount(H, weights=d2)` — **not** `np.add.at`.
  `h≥3` from **one** seeded sample of 2e7 uniformly random variant pairs per assay (Hamming computed from the P-length `int8` code vector by block XOR-nonzero-count); the same pass yields every `h` simultaneously and is cross-checked against the exact `h=1,2` (must agree within the sampling SE).
- `V(∞) = Var(y)·n/(n−1)` **closed form**. `GMD = [2/(n(n−1))] Σ_k (2k−n−1)·y_(k)` on sorted `y`, **closed form, O(n log n)**. No random-pair enumeration anywhere.
- `SI = G(1)/GMD`.
- `γ(1)` = Pearson over all `(i,{j})` with singles `i`,`j` and double `{i,j}` all observed, between `Δ_i(∅)=y_i−y_WT` and `Δ_i({j})=y_ij−y_j`; `γ(m)` = Pearson(`Δ_i(B)`, `β̂_i`) over backgrounds of size `m`. CI by 2,000 bootstraps over site pairs `(i,j)`.
- `r/s` (Szendro 2013) reported with both calibration nulls and the dimensionless position `pos_rs = (rs_obs − mean rs_N1)/(mean rs_N3 − mean rs_N1)`.

**Null** N1 (lower calibration), N3 (House-of-Cards, upper calibration only — never a hypothesis test).

**Decision, per assay.**
SUPPORTED iff `SI ≤ 0.50` **and** `V(1)/V(∞) ≤ 0.35` **and** `V(h)` non-decreasing over `h=1..4` **and** `γ(1) ≥ 0.60` with 95% CI lower bound > 0.45.
**REFUTED iff** `SI ≥ 0.80` **or** `V(1)/V(∞) ≥ 0.70` **or** `V(1) > V(2)` **or** `γ(1) ≤ 0.20` with CI upper < 0.45 **or** `pos_rs ≥ 0.70`.

**Aggregate.** C1 TRUE iff supported in ≥ 5 of 7 families. **C1 REFUTED iff refuted in ≥ 3 of 7 families.** Pre-declared refutations (must reproduce, else the implementation is wrong): Z-ZSPA1-LL1 1.398, Z-ZSPA1-LL2 1.001, Z-ZpA963-HL2 0.893, CXCR4 0.883, 5A12_Ang2 0.844.

---

### 1.3 C2 — a minority of sequence-near pairs jump beyond what a smooth landscape produces

**Statistics** (`cliff/stats_c2.py`), on `ĉ` over `P_a`, latent scale primary, raw scale secondary.

- **Tail ratio, threshold-free.** `TR1 = Q_.999(|ĉ|)/Q_.75(|ĉ|)` when `|P_a| ≥ 20,000`; `TR2 = Q_.99/Q_.75` when `2,000 ≤ |P_a| < 20,000`; **no tail-ratio verdict below 2,000** (mixture + localisation routes only). Gaussian references, exact: `Q_.75(|Z|)=1.1503`, `Q_.99=2.5758`, `Q_.999=3.2905`, so `TR1_gauss=2.8606`, `TR2_gauss=2.2393`. Inference is **surrogate-referenced** (rank among N1 replicates), never a naive bootstrap of an extreme quantile — the latter is not consistent.
  *Why this carries the design*: `TR` is invariant under `ĉ → λĉ`, so a wrong `σ_noise` cannot create or destroy the C2 signal. Only the noise **shape** can, and that is what N2/N2c test.
- **Mixture.** Zero-mean two-component Gaussian EM, 200 restarts, 100 iterations, closed-form M-step; report `π̂` (cliff mass), `ρ̂ = σ̂₂/σ̂₁` (jump amplification), `ΔBIC = BIC₂ − BIC₁`, and `Λ = 2(ℓ₂−ℓ₁)` calibrated against 200 N1 bootstraps (the LRT is non-regular; χ² is wrong).
- **Swept enrichment.** `T(τ) = P_obs(|ĉ|≥τ) / mean_b P_{N2,b}(|ĉ|≥τ)` for `τ ∈ {2,3,4,5,6,8}` in **two** unit systems (`σ̂`-standardised and MAD-standardised), with the grid guard. Empirical `p = (1+#{b: P_b ≥ P_obs})/(B+1)`, BH-FDR **over the 14 primary+arm assays**, never over 1.7e6 edges. CIs by block bootstrap over **mutated positions** (resample the position set; take all edges whose differing position is in the resample) — the edge bootstrap ignores the dominant dependence.
- **Zero mass reported separately.** The fraction with `ĉ` exactly 0 is its own column and is never folded into "small" (5 assays have coarse grids).
- **Decomposition, not a gate.** `T_N2b(τ)` against the additive+ridge-pairwise surrogate. If `T_N2` is significant and `T_N2b` is not, the finding is reported as **"the excess is first-order pairwise epistasis, not idiosyncratic"** — a decomposition of the claim, not a failure of it (a pairwise-epistasis cliff is still a cliff).

**Nulls.** N1 (primary calibration), N2 (heavy-tailed-noise alternative), N2b (ordinary pairwise epistasis), N2c (heteroscedastic scale mixture). Full definitions in §3.

**Decision, per assay.** SUPPORTED iff **all** of:
1. `TR` exceeds the 99.5th percentile of the N1 ensemble;
2. `T(τ) ≥ 2.0` with `q_BH < 0.05` for **≥ 4 consecutive τ in [3,8]**, in **both** unit systems;
3. `ΔBIC ≤ −10` with `π̂ ∈ [0.001, 0.05]`, CI lower > 0.0005, and `ρ̂ ≥ 3.0`;
4. **if G7 shows the tail is inflatable by heteroscedasticity** (expected): the assay also passes ≥ 1 localisation route (C3-L).

**REFUTED iff** `TR` below the 95th percentile of N1 **or** `max_{τ∈[3,8]} T(τ) < 1.5` with CI upper < 2.0 **or** `ΔBIC > −10` **or** `π̂` CI upper < 0.001.

**Aggregate.** C2 TRUE iff supported in ≥ 4 of 7 families (subject to G9 tightening). **C2 REFUTED iff supported in ≤ 1 of 7.** If C2 is refuted, the deliverable headline is the negative result: *"BindingGYM binding landscapes are additive-plus-monotone-link-plus-heteroscedastic-noise to within the resolution of the data; no cliff component is detectable."* That is publishable and must be written as the finding, not buried.

---

### 1.4 C3 — the jumps are real: not measurement noise, not an artefact

Three named sub-tests, each with its own verdict line. **`C3-L` is the discriminator against heteroscedastic noise** and is therefore the load-bearing one.

#### C3-N — measurement noise (`cliff/stats_c3.py`)
`ε_st = y(st) − y(s) − y(t) + y(WT)` for every double whose two singles are observed. Cliff site-pair iff `|ε| ≥ 3σ_ε = 0.373` (KRAS). Replication rate `R = P(|ε_b| ≥ 2σ_ε ∧ sign match | |ε_a| ≥ 3σ_ε)`, chance level from 10,000 site-pair-label permutations in assay b.
SUPPORTED iff `R ≥ 0.70` with permutation chance ≤ 0.10 **and** sign agreement ≥ 0.85. **REFUTED iff** `R ≤ 0.35` **or** sign agreement ≤ 0.60 (chance 0.50) **or** > 50% of catalogued cliffs have `|Δ| < 3σ`.
**Testable in exactly one family (KRAS).** Every C3-N verdict is stamped `conditional` unless G-UP completes.

#### C3-L — localisation / reproducibility (five routes, each with a hard feasibility gate)
| route | statistic | gate | feasible assays |
|---|---|---|---|
| **L1** sibling corroboration | `β_a` = HC3 OLS slope of `e` on the mean of node-disjoint siblings `S(e)={(B',B'∪{i}) : |B⊕B'|=1}`, computed on **cross-fitted** residuals and compared to its **N2 null distribution** (not to 0 — the analytic-zero claim is false) | ≥ 1,000 edges with `|S| ≥ 3` | CR9114_H1 (every edge has 15 siblings), CR6261, 5A12_VEGF, GB1_2016, Z-ZpA963_HL1; **report the count for SARS2-RBD and hYAP65 and drop if it fails** |
| **L2** site-pair ICC across backgrounds | one-way random-effects `ICC=(MSB−MSW)/(MSB+(k̄−1)MSW)` | ≥ 200 site-pairs with ≥ 2 backgrounds | deep libraries only |
| **L2′** site-pair ICC across amino-acid combinations | same estimator, grouping ε by site pair `(s,t)` over its different `(a,b)` substitutions | ≥ 5 aa-combinations per site pair | **GB1_1FCC: 91,845 doubles / C(55,2)=1,485 site pairs ≈ 62 per pair.** Not available in KRAS (1.5) or SARS2-RBD (0.45) |
| **L3** cross-measurement ε replication | as C3-N | an independent measurement of the same site-pair exists | KRAS twin only (10,868) |
| **L4** out-of-sample pairwise predictability | `dR²_oos = R²_oos([1\|X\|Z]) − R²_oos([1\|X])`, ridge, inner 5-fold CV over 12 log-spaced λ, outer 5-fold over variants; plus the top-1%-of-\|coef\| share of `dR²_oos` | **≥ 5 observations per `Z` column** — otherwise each interaction is seen once, `dR²_oos ≡ 0` by construction and would be misread as "no epistasis" | 5A12_VEGF (~1,300 cols / 29,981 rows ≈ 23), GB1_2016 (2,166 / 22,176 ≈ 10), Z-HL1 (240 / 2,904 ≈ 12), CR9114_H1 (120), CR6261 (55). **INFEASIBLE and declared so**: GB1_1FCC, all KRAS, SARS2-RBD, CD19 |
| **L5** 3D localisation of ε | AUROC of `(−min heavy-atom distance between sites s,t)` discriminating `|ε_st| ≥ 3σ` from `< 1σ`, null NS2 | ≥ 500 ε values with both sites structurally annotated | **GB1_1FCC (91,845, unbiased design)**, SARS2-RBD_6M0J, CD19 (weak, 478 doubles), KRAS ×4 |

SUPPORTED iff `β_a` exceeds the 99.5th percentile of its N2 null **and** (`ICC ≥ 0.30` with CI lower > 0.15 **or** `dR²_oos ≥ 0.02` with CI lower > 0.005 **or** `AUROC_L5 ≥ 0.60` with `p_NS2 < 0.01`), using whichever routes are feasible.
**REFUTED iff** `β_a` inside the N2 95% band **and** ICC CI upper < 0.15 **and** `dR²_oos` CI upper < 0.02 — i.e. the deviations do not recur, which is indistinguishable from heteroscedastic noise.

#### C3-A — artefact clauses (all must pass)
1. **Sampling depth.** REFUTED if `Spearman(per-position cliff rate, per-position pair count) > 0.40` while the best structural covariate is < 0.20.
2. **Density.** Cliff enrichment must appear in **both** the top and bottom neighbourhood-density quintile (5 bins by observed nested-degree). Monotone-in-density ⇒ sequencing-depth artefact.
3. **Floor invariance.** Verdict unchanged before/after floor masking (with G10's composition check).
4. **Scale invariance.** A verdict holding on raw but not on the latent scale is **discarded**, never reported as positive.

**Aggregate.** C3 TRUE iff C3-L supported in ≥ 3 of 7 families, C3-A clean everywhere, and C3-N supported in the one family where it is testable. **C3 REFUTED iff C3-L supported in ≤ 1 of 7.**

---

### 1.5 C4 — is "interaction cliff" the right name? (interpretation; does not gate C1–C3)

`cliff/stats_c4.py`. **43.7% of the 2,220 mutated positions are Levy `interior`** (buried in the monomer fold) — the classic source of large DMS jumps. Any interface test that lumps interior with surface answers the wrong question.

- **C4-S site level, burial-matched.** Poisson/binomial GLM: `cliff_count_p ~ offset(log n_p) + iface_p + rsa_iso_p + Levy_class_p + log n_p + |β̂_p|`. Interface flag = min heavy-atom distance to the opposite side < 5.0 Å (empirically justified: max min-heavy distance over all `ΔSASA>1` residues is 6.07 Å, so 6.0 Å is a near-exact superset; **Cβ–Cβ < 8 Å is banned** — 911 flagged but only 825/1,050 recall). Reported alongside `ΔSASA > 1 Å²` and Levy `∈ {core, support, rim}`. Null **NS1**. **Kill switch: REFUTED if `β_iface` loses significance when `rsa_iso` enters.** SUPPORTED iff burial-matched OR ≥ 1.5 with `p_NS1 < 0.01` in ≥ 4 of 7 eligible assays; REFUTED iff the OR CI covers 1 in ≥ 5 of 7 or OR < 1.0 in ≥ 3.
- **C4-P pair level (the end-to-end test).** L5 above, null NS2.
- **C4-I partner specificity, double-centered (the actual interaction test).** For family F with J positions × K partners: `Z_jk = logit(per-position cliff rate)`, `W_jk = −min heavy-atom distance from j to partner k`. Double-centre both (`Z̃ = Z − rowmean − colmean + grandmean`); `M_F = corr(Z̃, W̃)`. **Row-centering removes each position's partner-invariant propensity algebraically** — which is exactly the fold-destabilisation contribution, since the protein folds the same way against every partner. Null **NS3**. Plus `F_spec = Var(δ)/(Var(μ)+Var(δ))` from the two-way decomposition `ε^{(a)} = μ + δ^{(a)} + noise` with `Var(noise) = σ_ε² = 0.01545` subtracted, and `PSI_j = (#partners in which j is a cliff)/K`.
  **Fold-axis validation (required):** `Spearman(row mean of Z, rsa_iso)` must be > 0. If it is not, the fold interpretation of the partner-invariant component is unsupported and must be reported as such rather than assumed.
  Families: KRAS (J=163, K=4 after dropping the duplicate — the only adequately powered one), PSD95 (83×2, 1,577/1,577 shared, same-site channel only), BH3 (10×2, 518/518 after the −2 offset correction), 5A12 (9×2, 534 shared), CR9114-H1 vs H3 (65,093 shared — reported as **censoring-limited only**, H3 is 89% floored).
  **"Interaction cliff" LICENSED iff** `F_spec ≥ 0.40` (noise-corrected) **and** `p_NS3 < 0.05` in KRAS **and** cliff-position PSI stochastically below non-cliff PSI (one-sided Mann-Whitney p < 0.05). **REFUTED (⇒ the correct name is "stability cliff", and the record must say so) iff** `F_spec ≤ 0.15` **or** median cliff PSI ≥ 0.75 **or** G11 shows dual localisation.

---

### 1.6 C5 — why it matters (cliff-aware evaluation)

`cliff/stats_c5.py`. `PSA(τ) = mean over catalogued cliff edges of 1[sign(Δy)==sign(Δŷ)]` (ties in `Δŷ` count 0.5; chance exactly 0.5), plus `AUPSA` = mean over the τ sweep, reported against `PSA` on non-cliff nested edges and against per-assay Spearman on all rows. Three **CPU-only** models: **M1** the additive-isotonic fit; **M2** linear on `[BLOSUM62, Δhydrophobicity, Δvolume, rsa_iso, iface]`; **M3** MSA site-independent log-odds from `$BINDINGGYM_INPUT/msas/*.a2m`. The ProteinMPNN OOF arm is **out of scope** (`diagnostics/oof/` does not exist on this branch and regenerating it needs a GPU).

**Blind spot demonstrated iff** `PSA_cliff ≤ 0.60` for all of M1–M3 while per-assay Spearman ≥ 0.30, in ≥ 4 of 7 families.
**Practical-emptiness refutation (the fourth way this study returns negative):** if `PSA_cliff ≥ 0.75` for **M1** — a purely additive model gets cliff directions right — then C2 may be statistically true but is practically empty, and it must be reported that way.

---

## 2. Assay inclusion list

28 files = **12 PRIMARY + 2 ARM + 3 CONTROL + 11 EXCLUDED-with-role**. All numbers from the profile.

### PRIMARY (registry; the 7 families)
| # | DMS_id | n | nested / \|P_a\| | why in | caveats |
|---|---|---|---|---|---|
| 1 | **GB1_IgG-Fc_fitness_1FCC** | 92,891 | 184,735 / ~183,690 | **FLAGSHIP END-TO-END ASSAY**: complete single scan (1,045 = 55×19) ⇒ exact ε for all 91,845 doubles; SI 0.268; zero censoring; zero ties; design@iface 0.327 ≈ bg 0.321. Carries C1+C2+C3-L(L2′,L5)+C4-S in **one** landscape | L4 infeasible (1 obs/Z-column) |
| 2 | GB1_IgG-Fc_fitness_1FCC_2016 | 22,176 | 52,175 / ~52,149 | densest neighbourhoods (19.31 pairs/variant); L1/L2/L4 all feasible | 4 positions ⇒ short `h` axis, C4-S undefined; pos-2 WT is T not Q ⇒ **a different background from #1**, not a replicate |
| 3 | KRAS_RAF1_norfitness_6VJJ | 12,677 | 24,138 / ~22,950 | σ_ε anchor half; 10.02 pairs/variant | design@iface 0.238 vs bg 0.101 (**2.4× enriched**) ⇒ C4-S flagged |
| 4 | KRAS_RAF1-RBD_norfitness_6VJJ | 23,162 | 43,202 / ~40,377 | largest, cleanest KRAS; 0.102 ≈ 0.101 ⇒ C4-S eligible | **byte-identical per-residue structural annotation to #3** ⇒ one structural unit, not two |
| 5 | KRAS_RALGDS-RBD_norfitness_1LFD | 20,341 | 37,730 / ~35,186 | 0.091 ≈ 0.090 ⇒ C4-S eligible | narrow range (span 1.89) |
| 6 | KRAS_PICK3CG-RBD_norfitness_1HE8 | 19,203 | 35,395 / ~32,756 | skew −0.02 (most symmetric in the set) ⇒ symmetric criterion defensible | 0.098 ≈ 0.096 |
| 7 | KRAS_SOS1_norfitness_8BE4 | 19,425 | 35,915 / ~33,412 | the retained half of the duplicate pair | C4-S primary entry uses the **unbiased 5O2S annotation** (0.160 ≈ 0.158) with both reported; SOS1's own design@iface is 0.264 (2.4×). G11 |
| 8 | SARS2-RBD_ACE2_deltaKd_6M0J | 21,872 | 19,459 / ~11,000 | graded orders 1–10 over 194 positions ⇒ a real mutation-degree axis; 0.108 ≈ 0.108 ⇒ C4-S + C4-P eligible | **23.85% censored** at −4.84/−4.76 with spike fraction rising 0.004→1.000 in order ⇒ masking mandatory; hard 0.01 grid ⇒ TR2 regime |
| 9 | 5A12_VEGF_fitness_4ZFF | 29,981 | 22,064 / ~22,010 | smoothest landscape in the set (SI 0.250) ⇒ strongest single C1 datum; L1/L2/L4 feasible | **designed C4 NEGATIVE control**: 0/9 mutated positions within 6.4 Å of VEGF, ΔSASA 0.0 (they contact Ang2); WT at the 0.0 percentile ⇒ no WT-anchored normalisation |
| 10 | Z-domain_ZpA963_HL1_fitness_2M5A | 2,904 | 9,736 / ~9,712 | 70.9% of a 4⁶ space ⇒ near-complete neighbourhoods; L1/L2/L4 feasible | 6/6 interface ⇒ **out of C4-S** (unfalsifiable); two-sided; WT is the global max |
| 11 | hYAP65_peptide_FunctioncalScore_1JMQ | 18,407 | 29,695 / ~29,407 | orders to 21 on a 46-aa chain ⇒ the longest degree axis | **`log10` first** (ratio scale, WT = 1.000); 0.324 vs 0.238 (1.36×) ⇒ C4-S flagged |
| 12 | CD19_FMC63_Fitness_7URV | 3,886 | 4,540 / ~2,922 | 0.078 ≈ 0.078 ⇒ C4-S eligible; widest range (span 22.6) | singles+triples with a **doubles gap** (478) ⇒ additive baseline for triples unavailable, C4-P weak; 0.569 value-uniqueness + trimodality ⇒ **score provenance caveat, may be binned selection** |

### ARM — unregistered hypercubes, separate denominator, never folded into the primary count
| # | DMS_id | n | \|P_a\| | why | caveats |
|---|---|---|---|---|---|
| 13 | CR9114_FluAH1_logKd_4FQI | 65,094 | ~517,528 | 2^16 at 99.33% ⇒ **every variant's full 16-neighbour shell observed**; strongest L1/L2/L4 power in the benchmark | not in `BindingGYM.csv`; **no PDB in `structures/`** ⇒ structurally mute unless G-OPT completes; 2.57% floored at 7.000 |
| 14 | CR6261_FluAH1_logKd_3GBN | 1,887 | ~9,000 | 2^11 at 92.14%; independent replication of #13's biology | 11.34% floored **with WT on the floor**; small-n ⇒ likely UNDERPOWERED by G8 |

Rationale for the ARM: the repo's own convention says the 3 unregistered files must not be mixed into BindingGYM comparisons, and the profile says these two are the best landscapes in the collection. Both are honoured: primary analysis is the official registry (24 independent landscapes after de-duplication), the arm is reported with its own denominator (28 vs 25) and an explicit "not comparable to any published BindingGYM number".

### CONTROL — not data points
| # | DMS_id | role |
|---|---|---|
| 15 | CR9114_FluAH3_logKd_4FQY | **censoring positive control (G5)**. 58,361/65,535 = 89.05% at exactly 6.000; the median IS the floor; 86.1% of nested edges are floor–floor |
| 16 | Z-domain_ZSPA-1_LL1_fitness_1LP1 | **anti-smooth negative control (G6)**. SI 1.398, sd 0.140 / max 3.631 / skew +10.55, no WT row, selection-derived membership |
| 17 | Z-domain_ZSPA-1_LL2_fitness_1LP1 | **anti-smooth negative control (G6)**. SI 1.001 — neighbours as different as random pairs |

### EXCLUDED, with the role each retains
| # | DMS_id | out because | retained for |
|---|---|---|---|
| 18 | KRAS_DARPinK27_norfitness_5O2S | **duplicate score table**: 19,227 shared keys, byte-identical scores | its structural annotation only (the unbiased KRAS C4-S entry + G11) |
| 19 | ACE2_SARS2-RBD_enrich_6M17 | `max_mut = 1` ⇒ `P_a` empty by construction | same-site substitution-roughness reference (19,665 pairs, complete 19-aa scan, zero ties) |
| 20 | CXCR4_CXCL12_enrich_8U4O | `max_mut = 1`; SI 0.883 | same-site reference over 295 positions; **C1 negative control** showing the smoothness signal is not a metric artefact |
| 21 | HLA-A2_TAPBPR_meanscore_5WER | `max_mut = 1`; **0 nested edges** (no WT row); duplicated `DMS_score` column | same-site reference (29,558 pairs) |
| 22 | PSD95_CRIPT_1BE9 | `max_mut = 1` | **C4-I partner-specificity probe** (1,577/1,577 shared with #23, r = 0.4795 ⇒ ~52% of the site-level effect is partner-specific) |
| 23 | PSD95_Tm2F_1BE9 | `max_mut = 1` | as above |
| 24 | 4D5_HER2_fitness_1N8Z | **nested = 262 < 500 power gate**; 0.50 pairs/variant; no singles, no doubles; SI 0.778 | **CLUSTER CHANNEL ONLY** (n=2,080). If cluster coverage fails its own gate, "4D5 is structurally incapable of testing this hypothesis" IS the finding |
| 25 | Z-domain_ZpA963_HL2_fitness_2M5A | n=600, 0.06% of a 10⁶ space, SI 0.893 | reported exclusion |
| 26 | 5A12_Ang2_fitness_4ZFG | span 0.648, sd 0.0787 ⇒ sub-replicate resolution; SI 0.844 | off-target half of the 5A12 specificity probe (534 shared, r = −0.163) |
| 27 | BH3_Bcl-xL_normed_1PQ1 | n=518; 33% dead plateau; SI 0.648; 10/10 interface | BH3 partner-specificity probe (518/518 after the −2 correction, G1b) |
| 28 | BH3_Mcl-1_normed_3KZ0 | n=518; 20% dead plateau; SI 0.682; 10/10 interface | as above |

### Families for aggregation (K = 7 primary)
`F1` GB1 {1,2} · `F2` KRAS {3,4,5,6,7} · `F3` SARS2-RBD {8} · `F4` 5A12 {9} · `F5` Z-ZpA963 {10} · `F6` hYAP65 {11} · `F7` CD19 {12}. `F8` hypercube arm {13,14} reported separately.

**Stated as the headline limitation, in the record's §1 and §6, not in a footnote:** the effective number of independent biological systems is **3–5**, not 7 and certainly not 25 — five of the twelve primary assays are KRAS on four near-identical interfaces (two with byte-identical structural annotation), and 25 registered assays sit on 22 PDBs. The two best landscapes for this hypothesis are unregistered and structurally mute. No aggregate number may be presented in a form a reader could mistake for 25 independent replications.

---

## 3. Module layout

All new code under `<R>/cliff/`. Every numeric threshold lives in `config.py` and nowhere else.

```
<R>/cliff/
  __init__.py
  config.py          # PATHS, ASSAYS (tier/family/eligibility/caveats), THRESH (frozen), SEEDS, TAUS
  io_bgym.py         # loading, canonical keys, mutant↔mutant_pdb pairing, G1/G1b/G2/G3 audits
  pairs.py           # wildcard bucketing; random-pair sampler; npz cache
  variogram.py       # exact V(1),V(2); sampled V(h>=3); closed-form V(inf), GMD; gamma; r/s
  latent.py          # M0' alternating fit + Tobit + g/ginv; 5-fold cross-fit; sigma(phi)
  noise.py           # sigma registry with provenance; KRAS-twin epsilon; GB1 overlap; sensitivity
  nulls.py           # N1, N2, N2b, N2c, N3, NS1, NS2, NS3 + parallel ensemble driver
  stats_c2.py        # c_hat, TR1/TR2, mixture EM+LRT, T(tau) sweep, grid guard, BH, block bootstrap
  stats_c3.py        # sibling slope beta, ICC, dR2_oos + gate, epsilon table, R, density strata
  clusters.py        # gated y-blind Ward channel (n<=30000 only), rho cuts, LOO-MAD on residuals
  structure.py       # PDB parse, H strip, cKDTree, ShrakeRupley, RSA/Levy, mutation->residue map
  stats_c4.py        # GLM+NS1, AUROC+NS2, double-centered Mantel+NS3, F_spec, PSI, fold-axis check
  stats_c5.py        # PSA/AUPSA; M1 additive-isotonic, M2 physchem linear, M3 MSA site-independent
  calibrate.py       # G4 self-cal, G7 scale-mixture discrimination, G8 power grid, G9 rule FPR
  verdict.py         # applies THRESH; emits T1..T12; family + aggregate verdicts
  figstyle.py        # rcParams, Okabe-Ito palette, panel-letter helper
  figures.py         # F1..F7
  run_all.py         # CLI: --stage {0,1,...,8} --assays ... --nproc ...
<R>/tests/
  test_cliff_invariants.py   # G1,G1b,G2,G3,G10 as pytest assertions; closed-form checks
```

### Core signatures (the implementer writes these exactly)

```python
# ---------------- io_bgym.py ----------------
MUT_RE = re.compile(r'^([A-Z])(-?\d+)([A-Za-z]?)([A-Z])$')   # wt, resnum, icode, mut

@dataclass(frozen=True)
class Assay:
    dms_id: str; poi: str; pdb_file: str
    side0: tuple; side1: tuple                  # from data_splits/assay_chain_sides.tsv
    y: np.ndarray                               # float64, len n  (log10 applied for hYAP65)
    keys: list                                  # canonical keys, len n
    codes: np.ndarray                           # (n, P) int8 code vector, 0 = WT at that position
    col_index: dict                             # (chain, seq_pos, aa_mut) -> column of X
    pos_index: dict                             # (chain, seq_pos)         -> code-vector column
    pdb_key: list                               # per row, per token: (chain, resseq, icode)
    row_index: np.ndarray                       # int32, source-csv 0-based row number (primary key)
    n_muts: np.ndarray                          # int8
    wt_row: int | None
    censor_levels: tuple                        # detected floor/ceiling values
    censor_mask: np.ndarray                      # bool
    quantum: float                              # modal spacing of sorted unique y

def load_assay(dms_id: str, *, apply_transform: bool = True) -> Assay
    """usecols=['POI','DMS_score','mutant','mutant_pdb'] ALWAYS.
       Side effect of usecols: HLA-A2's duplicated DMS_score column disappears (verified identical)."""

def parse_pair_dicts(mutant: str, mutant_pdb: str) -> list[tuple]
    """ast.literal_eval both; join BY CHAIN KEY (their key order differs, e.g. 5A12_Ang2
       mutant={'H':..,'L':'','A':''} vs mutant_pdb={'A':'','H':..,'L':''}); colon-split each
       chain's value; zip the two lists positionally. Returns
       [(chain, seq_pos, wt_aa, mut_aa, resseq, icode), ...]. icode '' when absent.
       Verified 0 failures in 1,173,273 instances."""

def canonical_key(muts) -> tuple            # tuple(sorted((chain, seq_pos, mut_aa)))
def detect_censoring(y: np.ndarray) -> tuple    # levels with mass >= 0.005 at min/max, 1-dp string form
def audit_all(files: list[str]) -> pd.DataFrame # G1, G1b, G2, G3

# ---------------- pairs.py ----------------
def enumerate_nested(keys: list) -> tuple[np.ndarray, np.ndarray]
    """Wildcard bucketing. Hold {K} in a set; for each variant v and each i in range(len(K_v)),
       test membership of K_v \\ i. Returns (idx (m,2) int32 with column 0 = the smaller set,
       add_col (m,) int32 = X-column of the added substitution). Counted once per unordered pair."""

def enumerate_samesite(keys: list) -> tuple[np.ndarray, np.ndarray]
    """Bucket each variant under (K\\i, chain_i, pos_i); a bucket of size k contributes exactly
       C(k,2) pairs; every aa in a bucket is distinct so no same-aa correction and no dedup."""

def sample_random_pairs(n: int, n_draw: int, seed: int) -> np.ndarray   # (n_draw, 2) int32, i<j
def hamming_from_codes(codes: np.ndarray, idx: np.ndarray, block: int = 1_000_000) -> np.ndarray

def cache_pairs(assay: Assay) -> dict   # writes data/cliff_cache/pairs/{id}_{kind}.npz + md5

# ---------------- variogram.py ----------------
def gini_mean_difference(y: np.ndarray) -> float
    """Closed form: y=np.sort(y); i=np.arange(1,n+1); 2*((2*i-n-1)*y).sum()/(n*(n-1))."""

def variogram_exact(y, idx, h) -> tuple[int, float, float]        # (N_h, V_h, G_h) via np.bincount
def variogram_sampled(y, codes, samp_idx) -> pd.DataFrame          # all h in one pass
def gamma_background(assay: Assay) -> dict                         # gamma(1), gamma(m), bootstrap CI
def roughness_to_slope(X, y) -> dict                               # r, s, rs, R2_add

# ---------------- latent.py ----------------
@dataclass
class LatentFit:
    beta: np.ndarray; phi: np.ndarray; z: np.ndarray
    g_knots: tuple; sigma_of_phi: callable
    r2_link_gain: float; n_iter_used: int

def fit_latent(X, y, censor_mask, censor_levels, *, n_iter=10, n_bins=20) -> LatentFit
def ginv(g_knots, y, lo, hi) -> np.ndarray
def crossfit_latent(X, y, censor_mask, censor_levels, folds) -> dict
    """returns phi_oof, z_oof, e_oof, sigma_oof (all len n)."""

# ---------------- noise.py ----------------
def sigma_registry() -> pd.DataFrame        # DMS_id, sigma_y, sigma_eps, provenance, n_source, caveat
def kras_twin_epsilon() -> dict             # n_shared, r, slope, resid_sd, sigma_eps, per-pair table
def gb1_cross_study() -> dict

# ---------------- nulls.py ----------------
def surrogate_N1(fit: LatentFit, y, rng, *, clamp, quantum) -> np.ndarray
def surrogate_N2(fit, e, rng, strata) -> np.ndarray            # residual exchange within m x phi-decile
def surrogate_N2b(fit_pairwise, rng, *, clamp, quantum) -> np.ndarray
def surrogate_N2c(fit, rng, *, kurtosis_target, clamp, quantum) -> np.ndarray
def permute_NS1(pos_table, rng) -> np.ndarray                  # iface label within burial x aa x |beta| x depth
def permute_NS2(eps_table, rng) -> np.ndarray                  # cliff label within seqsep-decile x rsa-tertile
def permute_NS3(Z, rng) -> np.ndarray                          # partner label within row
def run_ensemble(dms_id, null, B, stat_fn, seed0, nproc) -> pd.DataFrame
```

**Null definitions (what each preserves / destroys — this is the part that makes the claims falsifiable):**

- **N1 — smooth parametric surrogate.** `z* = Xβ̂ + ε*`, `ε* ~ N(0, σ̂²(φ))`; `y* = ĝ(φ*)`; re-apply the assay's clamp (`y* ← max(y*, L)`) and its decimal grid; **refit M0′ and the cross-fit from scratch on `y*`**, then recompute every statistic.
  *Preserves*: the exact variant set ⇒ the exact pair graph, every node degree including the WT hub, the mutation-order composition, the per-(pos,aa) effects, the monotone measurement nonlinearity, level-dependent noise, censored mass, tie mass, and the additive-fit estimation error.
  *Destroys*: all background dependence — every trace of epistasis, hence every cliff.
- **N2 — non-parametric residual exchange.** Keep `β̂, φ, ĝ, σ̂` fixed; permute `e` among variants **within (mutation order × φ-decile) strata**.
  *Preserves*: the empirical residual marginal **exactly**, including any heavy tail of any origin.
  *Destroys*: the assignment of a residual to a genotype ⇒ all locality on the graph.
  **Declared limitation**: in singles+doubles libraries a double's marginal residual *is* `ε_ij` and the nested difference is also `ε_ij`, so N2 coincides with the data and has **zero power** there. That is precisely why C3-L in those assays is carried by L3/L5, not by N2.
- **N2b — additive + ridge pairwise + link.** `Z` columns only for site pairs co-observed ≥ 20 times; λ by 5-fold CV on held-out variants. *Preserves* everything N1 does **plus** all first-order pairwise epistasis. Used to **decompose**, not to gate.
- **N2c — heteroscedastic scale mixture.** `ε* ~ N(0, σ̂²(φ)·V)`, `V` from a two-point discrete mixture calibrated so the marginal residual kurtosis matches the observed marginal kurtosis exactly. *Preserves* everything N1 does and reproduces the observed tail **without any epistasis**. Its whole purpose is G7: it proves that tail-shape statistics have no power against heteroscedasticity, and therefore that localisation (C3-L) must be a mandatory conjunct.
- **N3 — free permutation of `y`.** House-of-Cards; **upper calibration only**, never a hypothesis test (rejecting it is trivial and uninformative).
- **NS1 / NS2 / NS3** — as in §1.5.

```python
# ---------------- stats_c2.py ----------------
def c_hat(e_oof, sigma_oof, nested_idx, add_col) -> np.ndarray
def tail_ratio(c, q_hi, q_lo) -> float
def mixture_two_component(c, n_restart=200, n_iter=100) -> dict     # pi, s1, s2, rho, dBIC, Lambda
def enrichment_sweep(c_obs, c_null: np.ndarray, taus, unit, quantum) -> pd.DataFrame
def block_bootstrap_positions(c, pos_of_pair, stat_fn, B=1000, seed=...) -> tuple

# ---------------- stats_c3.py ----------------
def sibling_slope(e_oof, nested_idx, add_col, keys, *, min_siblings=3) -> dict
def epsilon_table(assay: Assay) -> pd.DataFrame     # site-pair eps with exact additive baseline
def icc_oneway(values, groups) -> dict
def dr2_oos(X, Z, y, folds, lam_grid, *, min_obs_per_col=5) -> dict   # returns feasible=False if gated
def replication_rate(eps_a, eps_b, sigma_eps, B=10_000, seed=...) -> dict
def density_strata(cliff_flag, degree, n_bins=5) -> pd.DataFrame

# ---------------- clusters.py  (GATED SUPPLEMENTARY CHANNEL) ----------------
def cluster_channel(assay: Assay, e_oof, rho_targets=(1,1.5,2,3), *, n_max=30_000) -> pd.DataFrame
    """Ward on the binary substitution matrix (Euclidean^2 == Hamming, so Ward minimises
       within-cluster Hamming dispersion). Blocked BLAS Gram -> condensed float64 -> scipy
       linkage(method='ward'); NEVER scipy's internal single-threaded pdist.
       RUNS ONLY IF n <= 30000. Statistic is the leave-one-out robust z of the CROSS-FITTED
       ADDITIVE RESIDUAL e (not of z) -- this is the fix to user-clusters' defect: on z, a
       flagged variant conflates a large main effect with a background-dependent jump.
       t_i = (e_i - med_c^{-i}) / max(1.4826*MAD_c^{-i}, sigma_noise_a); clusters with n_c < 8
       are dropped and frac_variants_covered is reported.
       Coverage gate: >= 30 clusters with n_c >= 8 AND >= 40% of variants covered, else the
       assay is declared structurally incapable and that IS the reported finding.
       rho=1 anchor: Jaccard of the flagged set against the pair-channel nested result.
       Degenerate ends reported verbatim: rho->0 the statistic does not exist; rho->max it
       degenerates to a global outlier test on the marginal.
       This channel is order-mixed by construction and therefore mixes nested with same-site
       steps. It may only ADD an assay to the C2 count when the pair channel is powerless
       (4D5), never override it, and it may never support an epistasis-ORDER claim."""
```

Eligible for the cluster channel (all `n ≤ 30,000`, condensed ≤ 3.6 GB): 4D5 (2,080), Z-ZpA963_HL1 (2,904), CD19 (3,886), hYAP65 (18,407), GB1_2016 (22,176), 5A12_VEGF (29,981). **Never** GB1_1FCC (34.5 GB condensed, 4.314e9 > int32 and scipy's index width unvalidated at that size), CR9114 ×2 (17.2 GB each), Z-LL1 (8.3 GB) — all of which have ample pair-channel power anyway.

```python
# ---------------- structure.py ----------------
def annotate_structure(poi, pdb_path, side0, side1) -> pd.DataFrame
    """STRIP element == 'H' first (all 22 PDBs are protonated models; Biopython gives H a 1.2 A
       radius and a naive ShrakeRupley call distorts the buried tail 2.5x: 37 vs 15 residues
       with SASA<1 on 1PQ1). Per residue: min heavy-atom distance to the OPPOSITE SIDE
       (cKDTree per side + np.minimum.at over atoms), sasa_iso, sasa_cplx, dsasa, rsa_iso,
       rsa_cplx (Tien 2013 maxima, clipped to [0,1] -- 35/9,493 exceed 1.0, max 1.36),
       levy in {interior, surface, support, rim, core}. Residue key is the TRIPLE
       (chain, resseq, icode). 37.9 s for all 25 assays, one core."""

def map_mutations(assay: Assay, annot: pd.DataFrame) -> pd.DataFrame
    """LOOKUP through mutant_pdb, never alignment. Assert all four wt-letter sources agree.
       Constant seq->pdb offsets may be used ONLY to build background (non-mutated) position
       sets, ONLY on the 19 verified-clean chains, ONLY with a per-position identity assertion:
       they FAIL on 4ZFG-H/L and 4ZFF-H/L (168 mismatches of 219 for 4ZFG-H)."""
```

---

## 4. Pinned env

**New env `bgym-cliff-v1`.** The existing `bgym-cliff` (py 3.11.16, numpy 1.26.4 only, everything else missing) is left untouched — its numpy is outside `EXPECTED_NUMPY` and a versioned name keeps a later re-pin auditable. **`h3ddg-reproduce` is never modified.** Verified just now: `h3ddg-reproduce` = py 3.9.25 / numpy 1.22.4 / scipy 1.13.1 / pandas 1.5.3 / sklearn 1.2.1 / biopython 1.81 — the new env clones that set exactly, so the structural recipe and any fold-related artefact stay bit-comparable, plus matplotlib (absent from `h3ddg-reproduce`) for figures.

`<R>/local-records/bindingGYM-cliff/sh/build_env_bgym-cliff-v1.sh`:

```bash
#!/usr/bin/env bash
# Pinned analysis env for BGYM-CLIFF v1.  CPU-only.  Do NOT install into h3ddg-reproduce.
#
# Why each pin is this value:
#   python 3.9.25   : matches h3ddg-reproduce, the env every verified number in this repo came from
#   numpy  1.22.4   : in make_inter_assay_folds.py's EXPECTED_NUMPY.  We only READ
#                     data_splits/inter_assay_folds.tsv, never recompute GroupKFold -- np.argsort's
#                     unstable quicksort silently changes the split across numpy versions -- but
#                     staying inside EXPECTED means any accidental recompute is still correct.
#   scipy  1.13.1   : cKDTree, sparse lsqr, optimize.least_squares, cluster.hierarchy.  Last
#                     series supporting py3.9 (1.14 requires >=3.10).
#   pandas 1.5.3    : usecols de-duplication of HLA-A2's repeated DMS_score column is verified here
#   sklearn 1.2.1   : IsotonicRegression, Ridge, KFold.  In EXPECTED_SKLEARN.
#   biopython 1.81  : Bio.PDB.SASA.ShrakeRupley -- no DSSP, no freesasa needed
#   matplotlib 3.7.5: vector PDF + 600 dpi PNG.  Absent from h3ddg-reproduce, hence a new env.
# statsmodels is deliberately NOT a dependency: BH-FDR, HC3 SEs, the Poisson/binomial GLM by IRLS
# and the mixture EM are ~20 lines each on top of scipy.
set -euo pipefail
source /home/guoj0f/anaconda3/etc/profile.d/conda.sh
conda create -y -n bgym-cliff-v1 python=3.9.25
conda activate bgym-cliff-v1
pip install --no-cache-dir \
  numpy==1.22.4 scipy==1.13.1 pandas==1.5.3 scikit-learn==1.2.1 \
  biopython==1.81 matplotlib==3.7.5 pytest==7.4.4
python - <<'PY'
import sys, numpy, scipy, pandas, sklearn, Bio, matplotlib
got = (sys.version.split()[0], numpy.__version__, scipy.__version__,
       pandas.__version__, sklearn.__version__, Bio.__version__, matplotlib.__version__)
want = ('3.9.25','1.22.4','1.13.1','1.5.3','1.2.1','1.81','3.7.5')
assert got == want, f'env pin mismatch: {got} != {want}'
import scipy.linalg, scipy.spatial, scipy.sparse.linalg, scipy.optimize, scipy.cluster.hierarchy
from sklearn.isotonic import IsotonicRegression
from Bio.PDB.SASA import ShrakeRupley
print('[env] bgym-cliff-v1 OK', got)
PY
pip list --format=freeze > "$(dirname "$0")/env_bgym-cliff-v1_freeze.txt"
```

Every run script asserts the same tuple in its first three lines, prints `torch`-free provenance, and prints the git commit + dirty count, so a wrong env shows up in the log's first line:

```bash
export BINDINGGYM_INPUT=/home/guoj0f/share/BindingGYM/input
python -c "import sys,numpy,scipy,pandas,sklearn,Bio;assert (sys.version.split()[0],numpy.__version__,scipy.__version__,pandas.__version__,sklearn.__version__,Bio.__version__)==('3.9.25','1.22.4','1.13.1','1.5.3','1.2.1','1.81')"
echo "[synced_commit] $(git -C $R rev-parse HEAD)  [dirty] $(git -C $R status --porcelain | wc -l)"
echo "[data] $(ls $BINDINGGYM_INPUT/Binding_substitutions_DMS/*.csv | wc -l) DMS csv, $(ls $BINDINGGYM_INPUT/structures/*.pdb | wc -l) structures"
```

`env_bgym-cliff-v1_freeze.txt` is committed. The record's §4 lists the six versions on one line and flags the load-bearing pins with 🔴. **No GPU. `CUDA_VISIBLE_DEVICES` is never set.** Fold membership is read from `data_splits/inter_assay_folds.tsv` with `sep='\t', comment='#'` and never recomputed.

---

## 5. Runtime plan

Machine: 80 cores, 111 GB available. Anchors are measured, not guessed: usecols read of the 62 MB GB1 file 0.24 s / 16.3 MB; whole-benchmark pair enumeration 59.2 s / < 2 GB; `fit_latent` 10 iterations 1.33 s at n = 92,891 / M = 1,045; full structural annotation 37.9 s.

| stage | work | per-assay cost | wall | nproc | peak RSS |
|---|---|---|---|---|---|
| 0 | load 28 (usecols) + G1/G1b/G2/G3 + enumerate nested & same-site + 2e7 random-pair sample × 14 + **G0 benchmark** | 2–12 s | **~7 min** | 1→14 | 2 GB |
| 1 | structural annotation, 25 assays (+ G-OPT if PDBs fetchable) | 1.5 s | **40 s** | 1 | 1 GB |
| 2 | `fit_latent` + 5-fold cross-fit, 17 assays | 1.3 s / 7 s (GB1) | **2 min** | 17 | 17 × 0.4 GB |
| 3 | null ensembles: 4 nulls × 200 reps × 17 assays = 13,600 replicate-jobs. Per job = draw + clamp + round + refit + cross-fit + statistics on the **cached pair index arrays** (~6 s GB1, ~5 s CR9114-H1, ~1.5 s median) ⇒ **≈ 9.4 core-hours** | — | **~9 min** | **64** | 64 × 0.5 = 32 GB |
| 4 | G4 (reuses stage 3, free), G7 localisation on N2c (+3,400 jobs), G8 power grid (6 × 3 × 3 × 40 = 2,160 jobs), G9 rule FPR (50 full-pipeline surrogate datasets × 17) ⇒ ≈ 4 core-hours | — | **~5 min** | 64 | 32 GB |
| 5 | **cluster channel, 6 assays, n ≤ 30,000, SERIAL (2 concurrent max)** | 5–60 s, 0.2–3.6 GB | **~6 min** | 2 | 7 GB |
| 6 | C4: GLM + 10,000 NS1 × 7; NS2 × 4; NS3 × 10,000 on KRAS; PSD95/BH3/5A12 probes | — | **~5 min** | 8 | 4 GB |
| 7 | C5: three CPU models + PSA/AUPSA × 14 (MSA read from `msas/*.a2m`) | — | **~3 min** | 14 | 4 GB |
| 8 | verdict tables T1–T12 + figures F1–F7 | — | **~3 min** | 1 | 2 GB |

**Total ≈ 45 min wall, peak 32 GB.** Hard scheduling rule: **stage 5 never runs concurrently with stage 3 or 4.** Stage 3/4 workers are capped at 64 (not 80) so that 64 × 0.5 GB stays well inside 111 GB.

**What was removed to get here.** `landscape`'s exact all-pairs variogram (4.31e9 pairs on GB1, 11.9e9 total) is gone: `h=1,2` exact from cached indices via `np.bincount` (~1 s), `V(∞)` and `GMD` closed form, `h≥3` from one 2e7-pair seeded sample. Nulls **never** recompute a variogram beyond `h=1,2` on the cached index arrays — the un-priced `200 × 4.3e9 = 8.6e11` pair-operations in `landscape`'s null loop would have been infeasible. `user-clusters`' O(n²) Ward on GB1 (34.5–70 GB) is gone by the `n ≤ 30,000` gate.

**Cache — all under `<R>/data/cliff_cache/`, which `.gitignore:6` (`data/`) already ignores at any depth. Nothing large is ever committed.**

```
<R>/data/cliff_cache/
  keys/{DMS_id}.npz              # codes int8 (n,P), col_index, pos_index, row_index, n_muts, y
  pairs/{DMS_id}_nested.npz      # idx int32 (m,2), add_col int32     (4.28M pairs total = ~34 MB)
  pairs/{DMS_id}_samesite.npz
  randpairs/{DMS_id}_2e7_seed20260902.npz          # + .md5
  latent/{DMS_id}.npz            # beta, phi, z, e_oof, sigma_oof, folds, g_knots
  nulls/{DMS_id}_{N1|N2|N2b|N2c|N3}_B200_seed*.npz # STATISTIC VECTORS ONLY, never surrogate y
  eps/{DMS_id}.npz               # site-pair epsilon
  structure/{POI}_{side0}_{side1}.npz
  MANIFEST.json                  # md5 of every cache file + env tuple + git commit + seeds
```

**Every sampled set is materialised once, seeded, md5-fingerprinted in `MANIFEST.json`, and read-only downstream** — a random sample is exactly the kind of version-sensitive, experiment-defining artefact the repo's own convention requires to be pinned. Downstream code verifies the md5 before use and refuses to run on a mismatch.

---

## 6. Deliverables under `local-records/`

`local-records/` does not exist yet — create it. **Never** create `results/` or `data/` under it (`.gitignore:6-7` match at any depth and would silently drop the commit). Small artifacts go in `artifacts/`.

First, append to `<R>/.gitignore`:
```
# local (workstation-less) analysis run logs: same convention as ibex-/workstation-records --
# raw logs stay on disk, the substance is transcribed into the record md.
local-records/**/*.out
local-records/**/*.err
local-records/**/*.npz
```

```
<R>/local-records/bindingGYM-cliff/
  bgym_cliff_audit_{YYYYmmdd-HHMMSS}.md          # THE record (skeleton below)
  sh/
    build_env_bgym-cliff-v1.sh
    env_bgym-cliff-v1_freeze.txt
    run_bgym_cliff_{YYYYmmdd-HHMMSS}.sh          # nohup driver, stages 0-8, log -> same dir/*.out
  artifacts/
    T01_assay_manifest.csv
    T02_gates.csv
    T03_noise_registry.csv
    T04_smoothness_C1.csv
    T05_variogram.csv
    T06_cliff_tail_C2.csv
    T07_localisation_C3.csv
    T08_epsilon_replication.csv
    T09_structure_sites.csv
    T10_structure_pairs.csv
    T11_partner_specificity.csv
    T12_cliff_aware_eval.csv
    T13_sensitivity.csv
    T14_verdict_by_family.csv
    T15_cluster_channel.csv
    cliff_catalogue_{DMS_id}.csv.gz              # |c_hat| >= 4 only (~2% of P_a, ~20k rows total)
    F1_variogram_panel.pdf / .png
    F2_tail_survival_vs_nulls.pdf / .png
    F3_enrichment_sweep.pdf / .png
    F4_localisation.pdf / .png
    F5_structure.pdf / .png
    F6_gates_and_calibration.pdf / .png
    F7_cliff_blind_spot.pdf / .png
```

### Table columns (exact)

**T01 assay_manifest** — `DMS_id, filename, registered, tier{PRIMARY|ARM|CONTROL|EXCLUDED}, family_id, structure_cluster_id, exclusion_reason, n_rows, n_unique_keys, n_dup_keys(must be 0), poi, pdb_file, pdb_exists, side0_chains, side1_chains, scale_type, transform_applied, sign_convention, has_wt_row, wt_row_index, wt_value, wt_percentile, rho_depth_score, max_mut, mut_count_hist, n_positions, aa_per_pos_median, y_min, y_max, y_sd, y_mad, n_distinct_values, modal_decimals, quantum, floor_value, floor_frac, ceil_value, ceil_frac, modal_value_frac, n_nested, n_samesite, n_nested_wt_anchored, n_nested_censor_touching, n_primary_Pa, pairs_per_variant, wt_degree, max_degree, n_edges_ge3_siblings, mean_obs_per_pairwise_col, pairwise_feasible, design_iface_frac, bg_iface_frac, iface_bias_factor, eligible_C1, eligible_C2, eligible_C3L, eligible_C4S, eligible_C4P, eligible_C4I, eligible_cluster_channel, underpowered_G8`

**T02 gates** — `gate_id, gate_name, assay, statistic, expected, observed, tolerance, PASS/FAIL, consequence_if_fail, halts_study`

**T03 noise_registry** — `DMS_id, sigma_y, sigma_eps, provenance{measured_replicate|cross_study_contaminated|internal_residual|stipulated}, source_partner, n_source, r_source, slope_source, resid_sd_source, sigma_over_mad, caveat, upstream_SE_obtained, verdict_stamp{calibrated|conditional}`

**T04 smoothness_C1** — `DMS_id, SI, SI_lo95, SI_hi95, V1_over_Vinf, V_monotone_h1_h4, V_range_h90, gamma1, gamma1_lo95, gamma1_hi95, gamma_decay_json, r_rough, s_slope, rs, rs_N1_mean, rs_N3_mean, pos_rs, R2_add_raw, R2_add_latent, link_R2_gain, SI_N1_p975, SI_N3_p025, samesite_SI_reference, verdict_C1, failing_criterion`

**T05 variogram** — `DMS_id, h, N_h, exact_or_sampled, V_h, V_h_se, G_h, G_h_se, V_h_over_Vinf, G_h_over_GMD, V_h_N1_mean, V_h_N1_lo, V_h_N1_hi, V_h_N2_mean` + one row `h='random'` carrying closed-form `V(∞)` and `GMD`

**T06 cliff_tail_C2** — `DMS_id, scale{latent|raw}, unit{sigma|MAD}, n_Pa, frac_c_exact_zero, Q75, Q99, Q999, TR_used{TR1|TR2}, TR, TR_N1_p95, TR_N1_p995, TR_N2c_mean, kurtosis, pi_hat, pi_lo95, pi_hi95, sigma1, sigma2, rho_hat, dBIC, Lambda, Lambda_N1_p995, tau, tau_absolute, grid_guard_pass, n_cliff, rate_obs, rate_N1_mean, rate_N2_mean, rate_N2_p95, rate_N2b_mean, T_N2, T_N2_lo95, T_N2_hi95, T_N2b, p_perm_N2, q_BH, n_consecutive_tau_passing, verdict_C2, failing_criterion`

**T07 localisation_C3** — `DMS_id, route{L1..L5}, feasible, n_units, beta_sibling, se_hc3, beta_N2_p995, beta_in_N2_band, ICC, ICC_lo95, ICC_hi95, ICC_N2_mean, dR2_oos, dR2_lo95, dR2_hi95, top1pct_share, ridge_lambda, AUROC_L5, AUROC_lo95, p_NS2, depth_spearman, best_struct_covariate, density_q1_rate, density_q5_rate, density_monotone, floor_mask_invariant, latent_raw_consistent, verdict_C3L, verdict_C3A, failing_criterion`

**T08 epsilon_replication** — `assay_a, assay_b, relation{same_partner_diff_construct|same_library_diff_partner|same_interaction_diff_study|identical_score_table}, join_method{canonical_key|mutant_pdb_aligned|wt_alignment}, n_shared, sd_eps_a, sd_eps_b, pearson_raw, ols_slope, resid_sd_after_affine, sigma_eps, n_cliff_a_3sigma, R, R_chance_perm, perm_p, sign_agreement, F_spec, F_spec_noise_corrected, verdict_C3N, verdict_stamp`

**T09 structure_sites** — `DMS_id, chain, resseq, icode, seq_idx, wt_aa, levy_class, rsa_iso, rsa_cplx, dsasa, min_heavy_dist, cb_dist, is_iface_5A, is_iface_dsasa, n_variants_at_site, n_pairs_at_site, n_cliff_pairs, cliff_rate, beta_hat_abs, rsa_decile, aa_class, depth_tertile, OR_burial_matched, OR_lo95, OR_hi95, beta_iface_unadj, beta_iface_adj, p_wald, p_NS1, beta_iface_after_rsa, assay_permissible`

**T10 structure_pairs** — `DMS_id, site_s, site_t, aa_s, aa_t, seq_separation, d3d_min_heavy, levy_s, levy_t, both_iface, n_backgrounds, n_aa_combos, eps, eps_z, is_cliff_3sigma, ICC_sitepair, AUROC_contribution, p_NS2`

**T11 partner_specificity** — `family, chain, resseq, icode, wt_aa, K_partners, cliff_rate_p1..p4, min_heavy_dist_p1..p4, iface_flag_p1..p4, PSI, Z_doublecentered_p1..p4, rowmean_Z, rsa_iso, family_M_stat, family_p_NS3, foldaxis_spearman_rowmean_rsa, F_spec, F_spec_noise_corrected, MW_PSI_p, twin_structure_OR_8BE4, twin_structure_OR_5O2S, classification{interaction_cliff|stability_cliff|undetermined}`

**T12 cliff_aware_eval** — `DMS_id, model{M1_additive_isotonic|M2_physchem|M3_msa_site_indep}, tau, n_cliff_edges, PSA_cliff, PSA_lo95, PSA_hi95, PSA_noncliff, AUPSA, spearman_all_rows, rmse_all, rmse_cliff, n_pred_ties, verdict_blindspot, verdict_practical_emptiness`

**T13 sensitivity** — `DMS_id, knob{tau|sigma_mult|scale|floor_mask|crossfit|linkage|rho|min_cluster_size|include_arm|iface_def}, value, n_Pa, TR, T_N2, q_BH, dBIC, pi_hat, SI, beta_sibling, OR_iface, verdict_flips`

**T14 verdict_by_family** — `family_id, member_assays, n_eligible, C1_pos/C1_neg, C2_pos/C2_neg, C3L_pos/C3L_neg, C3N_result, C4S_result, C4I_result, C5_result, family_verdict_C1, family_verdict_C2, family_verdict_C3, all_three, meta_effect, meta_ci_lo, meta_ci_hi, notes` + footer rows: the k-of-7 counts, the **G9 empirical FPR of the rule**, the binomial p of the count (0.2266 at 5/7), and the aggregate SUPPORTED / REFUTED / INCONCLUSIVE call

**T15 cluster_channel** — `DMS_id, linkage, rho_target, K, n_clusters_ge8, frac_variants_covered, coverage_gate_pass, mean_within_radius, order_range_within_cluster, eta2_residual, s_rho, ari_ward_vs_average, ari_subsample_5seed, n_cliff, cliff_rate, T_N2, jaccard_vs_pair_channel_rho1, peak_RAM_GB, wall_s, adds_assay_to_C2_count`

**cliff_catalogue_{DMS_id}.csv.gz** — `pair_id, DMS_id, row_index_u, row_index_v, mutant_u, mutant_v, order_u, order_v, background_key, add_chain, add_seq_pos, add_resseq, add_icode, add_wt_aa, add_mut_aa, y_u, y_v, delta_y, delta_latent, beta_hat_add, c_hat, c_hat_MAD_unit, sigma_used, sigma_provenance, censor_class{floorfree|crossing|floorfloor}, wt_anchored, degree_u, degree_v, density_quintile, n_siblings, sibling_mean, sibling_z, tau_min_included, q_value, levy_class, min_heavy_dist, dsasa, rsa_iso, rsa_cplx, blosum62, d_hydrophobicity, d_volume, family, PSI, partners_cliff_in, verdict_flags`
Primary key `(DMS_id, row_index)` — identical to the repo's `id = f'{DMS_id}#{row_index}'` and to `entries.pkl`, so the catalogue joins directly to any OOF csv.

### Figures — publication-legible (`cliff/figstyle.py`)

Explicit rcParams, no seaborn: `savefig.dpi=600`, `pdf.fonttype=42`, `font.family='sans-serif'`, `font.sans-serif=['Helvetica','Arial','DejaVu Sans']`, `font.size=8`, `axes.labelsize=8`, `xtick.labelsize=7`, `ytick.labelsize=7`, `legend.fontsize=7`, `axes.titlesize=9`, `axes.linewidth=0.6`, `xtick.major.width=0.6`, `lines.linewidth=1.1`, `axes.spines.top=False`, `axes.spines.right=False`, `figure.constrained_layout.use=True`. Widths exactly 89 mm (single) / 183 mm (double). Panel letters bold 9 pt at axes-fraction (0.01, 0.99). Palette = Okabe–Ito 8-colour (colour-blind safe), one colour per family, consistent across all seven figures. Every shaded band is a real null envelope (5th–95th percentile), never a decorative ribbon. Both `.pdf` (vector) and `.png` (600 dpi) written.

- **F1 `variogram_panel`** — 12 small multiples (one per primary assay, grouped by family) + a separately outlined control row (CR9114-H3, Z-LL1, Z-LL2). x = Hamming distance `h` (integer ticks 1…max), y = `V(h)/V(∞)` on log scale; observed line with `N_h`-based SE, N1 ribbon (grey), N2 ribbon (blue), dashed line at 1.0, and the 0.35 / 0.70 decision lines annotated. This is the C1 figure and it shows the negative result (Z-domain above 1.0) on the same axes as the positive.
- **F2 `tail_survival_vs_nulls`** — per assay, log-y survival `P(|ĉ| ≥ τ)`: observed (solid) against **four** envelopes — N1, N2, N2b, **N2c**. N2c is the point of the figure: it shows visually that heteroscedasticity alone reproduces the tail, which is why localisation is a mandatory conjunct. τ axis marked with the sweep grid and the grid-guard cut. Inset per assay: the mixture fit with the `ĉ = 0` spike drawn as a separate marked stem.
- **F3 `enrichment_sweep`** — the robustness figure. Twin panels: `T(τ)` vs τ, one line per assay coloured by family, log-y, `T=1` and `T=2` reference lines, refutation region shaded — left in σ units, right in MAD units. This is what answers "you picked a magic threshold": the conclusion is read off the whole curve.
- **F4 `localisation`** — (a) `e_e` vs sibling mean with the fitted `β_a`, the **N2 null band** shaded (not an analytic zero), cliff edges highlighted, `β_a(τ)` as an inset; (b) KRAS twin hexbin of `ε_a` vs `ε_b` on the 10,868 shared site-pairs with the affine line, the ±3σ box, `r = 0.812` annotated, and the ROC with bootstrap band; (c) variance-decomposition bars (shared μ / partner-specific δ / noise) with `F_spec`; (d) GB1_1FCC site-pair ICC across amino-acid combinations.
- **F5 `structure`** — (a) burial-matched cliff rate by Levy class within `rsa_iso` deciles for the 7 permissible assays, NS1 bands; (b) forest of `AUROC(−d3d)` per assay against NS2; (c) the KRAS 163 × 4 double-centered heatmap with per-partner interface distance annotated, plus the NS3 permutation histogram and `M_obs`; (d) the twin-structure control (8BE4 OR vs 5O2S OR) and the 5A12_VEGF designed-negative panel.
- **F6 `gates_and_calibration`** — (a) CR9114-H3 before/after masking; (b) Z-LL1 cliff rate by density quintile; (c) **G8 power curves** (detection power vs injected amplitude, one line per rate, per assay) with the 0.50 underpower line; (d) G4 uniformity QQ of the 200 null p-values; (e) G9 aggregate-rule FPR against the 0.10 ceiling. This is the figure a hostile referee is shown first.
- **F7 `cliff_blind_spot`** — per-assay scatter of per-assay Spearman (x) against `PSA_cliff` (y) for M1–M3, with the 0.5 chance line, the 0.60 blind-spot line and the 0.75 practical-emptiness line; companion bars of AUPSA on cliff vs non-cliff edges.

### Record markdown skeleton (house convention)

`local-records/bindingGYM-cliff/bgym_cliff_audit_{YYYYmmdd-HHMMSS}.md`, timestamp from `date +%Y%m%d-%H%M%S`:

```markdown
# bindingGYM-cliff — mutation-cliff / interaction-cliff 审计（BindingGYM 24 独立 landscape + 2 hypercube 臂）

> created {YYYY-mm-dd HH:MM} ｜ **status: PLANNED**（judgement 标准已写死，尚未跑数）
> 关联：数据说明 `$BINDINGGYM_INPUT`；解析口径 `<R>/bindinggym.py:10-16`；指标口径 `<R>/bindinggym_metrics.py`

## 1. Goal / hypothesis
**用户原话（不改写、不弱化，2026-09-02）**：
> 「我之前发现了 mutation effect 中一个很有意思的现象：…（整块照抄）」

**拆成可判定的命题**：C1 smooth / C2 jumps / C3 real（各自的形式化 + 度量 + 否证线）
**判读标准（先写死，避免事后找解释）** ← §1.2–1.6 的表格整搬，跑数之前落笔
**不在本次范围内**：GPU 推断、模型训练、ProteinMPNN OOF 臂（`diagnostics/oof/` 不在本 branch）
**⚠️ 头号 limitation（写在这里，不写脚注）**：有效独立体系 3–5 个，不是 7 也不是 25

## 2. 数据层（已实测核实，非推测）
全量审计表 + 已知 gotcha（HLA-A2 重复列、X padding、Kabat icode、三个 unregistered 文件、
KRAS 同一列 score、Z-domain chain-key 伪重复、五个纯 single assay、五处 censoring）

## 3. Design & decision points（← 开跑前写；有 trade-off 的让用户拍板）
### 3.1 要新写的代码（`cliff/` 18 个模块表）
### 3.2 KEY DECISIONS（选项 + 理由 + 拍板人 + 日期）
  1. spine = landscape，compute = sali，cluster 作 gated 补充（judges 1-1-1 分裂，理由见 §0）（我的判断，待用户确认）
  2. 🔴 sign convention：全部 28 个 assay 是 higher = better（−log10(Kd) 越大越紧）；confounds agent 的反向说法是错的（我的判断，已用 CR9114-H1 germline 8.425 → matured 9.592 核实）
  3. 剔 KRAS_DARPinK27_5O2S 的 score，留其结构注释（我的判断）
  4. 25 registry 为主分析，CR9114-H1 + CR6261 单列 arm 与分母（我的判断，待确认）
  5. cluster channel 只跑 n ≤ 30,000，且统计量建在 cross-fitted residual 上（我的判断）
  6. C2 单独不成立：G7 若显示 N2c 能抬起尾巴，则强制 C2 ∧ C3-L（规则由 gate 决定，非事后）
### 3.3 口径（定义写死，后续所有数字都按它算）
### 3.4 尚未测量、故不预先承诺的量：墙钟、峰值 RSS、cliff 数量级、G8 power、G9 FPR

## 4. Run config（本地，无调度器）
机器 / pinned env 六个版本一行 + 🔴 load-bearing pin / commit + dirty / BINDINGGYM_INPUT +
指纹 / 启动脚本绝对路径 / PID / 起止 + elapsed / 峰值 RSS / 并行度 / 输出绝对路径 /
最小复现命令块（可直接粘贴）

## 5. Change log

## 6. Results
### 6.1 Headline   ### 6.2 逐 assay 明细（末行 **per-family mean** 加粗）
### 6.3 negative controls / gates（G1–G11、G8 power、G9 FPR）
### 6.4 口径说明（必须随结果一起引用）
### 6.5 残余不确定性（诚实记录）← §8 的四条整搬
### 6.6 下一步（按性价比排序，每条注明是否需要用户批准）

## 7. 更正史（原处已就地标注，此处汇总索引）
```

Commit after every update: `git add local-records/bindingGYM-cliff cliff tests .gitignore && git commit -m "local-records: bindingGYM-cliff/bgym_cliff_audit ({date})" && git push origin HEAD` — current branch only, never `master`, never `--force`, and scan for > 50 MB files before `add`.

---

## 7. Kill criteria (consolidated)

**Stop and report a negative result** — do not keep digging — the moment any of these fires:

1. **G1 / G2 / G3 fail** → the data is not what the profile describes. Stop; report the discrepancy.
2. **G4 fails** (N1 scored against N1 ≠ 1.00 ± 0.05, or the null p-values are non-uniform) → the surrogate machinery is biased; no observed number is readable. Stop.
3. **G5 fails** (CR9114-H3 still shows `T(4)` outside the N2 band after masking, or `|P_a|` does not collapse ≥ 10×) → the pipeline cannot distinguish a detection limit from a cliff. Stop.
4. **G6 fails** (Z-ZSPA1-LL1 or LL2 shows cliff enrichment) → the pipeline is fooled by selection-dependent membership. Stop.
5. **G7 fails in the bad direction** (localisation statistics are *also* inflated under N2c) → the design has no axis that separates cliffs from heteroscedastic noise. Stop and say so.
6. **G9 cannot be satisfied** (no `k` gives family-level FPR ≤ 0.10) → the aggregate rule is uncalibratable at this K. Report per-family only; no aggregate verdict.
7. **C1 refuted in ≥ 3 of 7 families** → report *"premise C1 is not a general property of BindingGYM landscapes"* with the per-family table. This is a real result: the four Z-domain assays already falsify it (SI 1.398 / 1.001 / 0.893), and the finding that smoothness is a property of well-sampled designed libraries rather than of mutation landscapes in general is worth stating on its own.
8. **C2 supported in ≤ 1 of 7 families** → report *"BindingGYM binding landscapes are additive + monotone link + heteroscedastic noise to within the resolution of the data; no cliff component is detectable."* Headline the negative.
9. **C3-L supported in ≤ 1 of 7 families** (β inside the N2 band, ICC upper < 0.15, dR²_oos upper < 0.02) → the deviations do not recur ⇒ **indistinguishable from heteroscedastic measurement noise**. Report as such; do not rename it a cliff.
10. **C5 practical-emptiness fires** (`PSA_cliff ≥ 0.75` for the purely additive M1) → C2 may be statistically true and practically empty. Report it that way even if C1–C3 all passed.
11. **Any assay stamped UNDERPOWERED by G8** reports INCONCLUSIVE whatever its numbers say, and never contributes to a family count.
12. **G-UP does not complete** → every C3-N verdict is stamped `conditional` and the record states verbatim: *"effect size relative to one contaminated replicate bound, not a calibrated significance."*

---

## 8. What is explicitly NOT claimed (must appear in the record's §6.5, in these words)

The judges found four defects present in **all three** candidate designs. None is fully solvable inside these files; each is named, bounded, and given a hard reporting rule.

1. **No measurement-replicate noise floor exists in BindingGYM.** All 28 files have zero duplicate canonical genotypes; the whole tree carries eight columns and no read counts, SEs, replicate or barcode fields. Every "this is not noise" statement traces to two contaminated anchors: the KRAS twin (different construct — full RAF1 vs RBD-only; different library — 63 vs 166 positions; normalisation slope 0.6445 removed by an affine correction that is right only if the construct effect is affine in ε) and the GB1 cross-study overlap (n = 160, chain-C position 2 is Q in one file and T in the other ⇒ two backgrounds, not two measurements). The Z-domain "within-genotype replicates" are a chain-key collision artefact and are forbidden. **Rule**: unless G-UP delivers upstream per-variant SEs, C3-N is reported as an effect size against one contaminated bound, never as a calibrated significance.
2. **The surrogate null's noise scale is estimated from data that contain the tail being tested.** `σ̂(φ)` is a bin-wise MAD of the very residuals whose excess tail is the hypothesis — partially self-calibrating, and biased toward absorbing the signal. Mitigations, all implemented and all reported: C2's verdict statistics are scale-free, so a globally mis-scaled σ cannot create the result; G8 measures the estimator's bias and the design's power by injection; the `σ × {0.5, 1, 2}` surface accompanies every headline number; and G7 quantifies exactly how much of the tail a pure heteroscedastic mixture can manufacture. **The circularity is bounded, not eliminated.**
3. **Effective independent sample size is 3–5 biological systems, not 7 families and not 25 assays.** Five of twelve primary assays are KRAS on four near-identical interfaces (KRAS_RAF1 and KRAS_RAF1-RBD have byte-identical per-residue structural annotation); 25 registered assays sit on 22 PDBs; the structural half has 7 eligible assays of which 5 are KRAS, against a background in which 43.7% of mutated positions are buried interior. The two best landscapes for this hypothesis (CR9114-H1, 2^16 at 99.33%; CR6261, 2^11 at 92.14%) are unregistered, have no PDB in `structures/`, and are therefore structurally mute unless G-OPT completes — so a positive C3-L rests partly on numbers incomparable to any published BindingGYM result. **The generality claim is a count over 7 correlated families with binomial p = 0.2266 at 5/7; the evidence is the per-family CIs.**
4. **End-to-end demonstration is possible in four families, and this is stated rather than stitched silently.** GB1_IgG-Fc_fitness_1FCC is the flagship: a complete 55 × 19 single scan gives an exact additive baseline for all 91,845 doubles, its 1,485 site pairs carry ≈ 62 amino-acid combinations each (so site-pair reproducibility is testable *within the assay*), it has no censoring and no ties, and its design interface fraction 0.327 matches its background 0.321 — so C1, C2, C3-L (routes L2′ and L5) and C4-S are all available in **one** landscape. SARS2-RBD_6M0J is the second (with 23.85% censoring masked), KRAS the third (the only family with a measured ε replicate), CD19 the fourth (weak, 478 doubles). In 5A12_VEGF and Z-ZpA963_HL1 the C3-L routes are strong but the interface contrast is undefined (0/9 and 6/6); in CD19 the C3-L routes are weak. **Where the chain is completed across assays rather than within one, the verdict table says so.**
5. **"Interaction" versus folding is separated statistically, not mechanistically.** Row-centering in the double-centered Mantel removes each position's partner-invariant propensity algebraically, and `F_spec` subtracts the measured ε noise — but a partner-invariant cliff could still be a genuine binding cliff on a shared epitope, and a partner-specific one could still be an expression or normalisation difference. There is no independent folding-stability measurement in these files. The fold axis is validated only indirectly, by requiring `Spearman(row mean, rsa_iso) > 0`; if that fails, the fold interpretation of the partner-invariant component is reported as unsupported. Calibrating against the published KRAS ddG_fold / ddG_bind decomposition (Weng et al. 2024) is named as the next step and is out of today's scope.