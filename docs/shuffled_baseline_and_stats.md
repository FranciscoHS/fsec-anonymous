# Shuffled-weights baseline & statistical tests (2026-08-20)

Results produced for the NeurIPS 2026 "Interpretability as a Science"
workshop submission. All fits use the canonical protocol: per-pair
threshold (`--per_pair_threshold`), exact-geodesic normalization
(`--exact`), headline cell `("1.0xT", 60.0)`, value `p_median`.

## 1. Shuffled-weights baseline (randomized-transformer control)

Script: `scripts/shuffled_weights_baseline.py`. Gemma-2-9B, layer 2;
the entries of every 2D weight matrix in the transformer blocks are
permuted within each matrix (294 matrices, 8.32B entries; embeddings and
layer norms untouched). Per permutation seed, the full pipeline reruns
on the shuffled model: DoM direction extraction (same prompt sets),
FineWeb anchors, canonical 2D sweep. 5 seeds x 20 tier-1 pairs = 100
fits. Run on RunPod RTX 4090, ~24 s/pair, ~70 min total.

Fits: `results/fits/fits_gemma_L2_shufseed{0..4}_thrpair_exact.pkl`
Sweeps: `results/sweeps_2d/*_shufseed{k}.pkl` (100 files)
Directions: `results/directions/dirs_gemma_L2_shufseed{k}.pkl`

Per-seed and pooled exponents. CI = direction-level cluster bootstrap
(5000 iterations, resample directions with replacement, rebuild pairs,
drop self-pairs, central 95%; same machinery as
`_bootstrap_pooled_mean_ci` in `scripts/plotting/plot_robustness_beeswarm.py`,
extended to record the median per iteration). Pooled CI treats each
seed as its own cluster group.

| | median p [95% CI] | mean p [95% CI] |
|---|---|---|
| seed 0 | 2.013 [1.94, 2.09] | 2.019 [1.96, 2.07] |
| seed 1 | 1.989 [1.94, 2.08] | 1.984 [1.95, 2.06] |
| seed 2 | 2.009 [1.93, 2.11] | 2.022 [1.96, 2.11] |
| seed 3 | 1.969 [1.92, 2.07] | 1.979 [1.94, 2.05] |
| seed 4 | 2.006 [1.96, 2.06] | 1.999 [1.96, 2.05] |
| **pooled (n=100)** | **2.001 [1.97, 2.03]** | **2.001 [1.98, 2.03]** |

Only 7% of the 100 fits exceed 2.1; 2% exceed 2.2. Trained Gemma's
contrastive set sits at median 2.31 / mean 2.41 (overlap-filtered),
far outside the pooled CI. Downstream L2 maxima on the shuffled model
were nearly identical across all pairs (~1160-1210), consistent with
no direction being privileged.

Caveats / notes:
- Tier-1 (20 semantic pairs) is a low-p subset of the *trained* model
  (median 2.055, range [1.54, 3.17]); the matched trained-vs-shuffled
  test on tier-1 alone is weak (Mann-Whitney one-sided p = 0.056;
  filter-passing 15 pairs only: p = 0.011). A matched full-pair-set
  comparison would need shuffled-model sweeps beyond tier-1
  (~3.5 h/seed for all 528 pairs on the 4090).
- Unmatched trained-filtered (n=398) vs shuffled (n=100):
  Mann-Whitney U one-sided p = 7.5e-29, rank-biserial 0.72;
  cluster-bootstrap median difference 0.31, 95% CI [0.22, 0.40].

## 2. Statistical test: contrastive vs shuffled-label control (trained Gemma)

Both sides trained Gemma-2-9B L2, existing fits, both overlap-filtered
at intra-pair |cos| < 0.1. The shuffled-label control is
`randomdiffavg` (`scripts/random_diff_directions.py --k_avg 30`): each
direction is the mean of 30 FineWeb activation differences — the
construction-identical non-semantic twin of a contrastive DoM
direction (same estimator, label structure destroyed).

| comparison | n | medians | U | one-sided p | rank-biserial |
|---|---|---|---|---|---|
| contrastive vs **randomdiffavg** | 398 vs 710 | 2.31 vs 2.02 | 215,952 | **1.2e-48** | 0.53 |
| contrastive vs randomdiff (k=1) | 398 vs 703 | 2.31 vs 2.09 | 191,191 | 2.3e-24 | 0.37 |

Dependence-robust confirmation (pair fits share directions, so
Mann-Whitney's independence assumption is violated): direction-level
cluster bootstrap of the median difference, contrastive minus
randomdiffavg: **delta median = 0.29, 95% CI [0.18, 0.41]**; 0/5000
bootstrap draws <= 0.

Recommended phrasing for the paper: report the Mann-Whitney U as the
test and the cluster-bootstrap CI as the robustness check; use
randomdiffavg as the primary control (construction-matched and the
more conservative of the two).

## Reproduction

```bash
# baseline sweeps (GPU; RunPod pod fsec-shuffle used 2026-08-20)
python scripts/shuffled_weights_baseline.py --target gemma --layer 2

# fits
for k in 0 1 2 3 4; do
  python scripts/fit_pairs.py --target gemma --layer 2 \
      --variant_suffix _shufseed$k --per_pair_threshold --exact
done
```

The Mann-Whitney / bootstrap numbers above were computed ad hoc from
the fits PKLs (see this file's git commit message for date/context);
the cluster bootstrap follows `_bootstrap_pooled_mean_ci` with
RandomState(0), 5000 iterations.
