---
id: 0018
slug: nacp-noise-aware-calibration
timestamp: 2026-04-13T17:20:00
git_tag: exp/0018
queue_entry: 3
status: no_change
val_loss: 0.0510
val_accuracy: 0.9863
best_so_far: 0.0505
---

## Hypothesis
Applying Noise-Aware Conformal Prediction (NACP) sample weights to isotonic calibration — which estimates the label noise transition matrix from validation data and down-weights samples whose labels are likely mislabeled — will improve log_loss, because the dataset's ~1.4% label noise distorts isotonic regression's piecewise-constant mapping (which treats every validation label as ground truth), and NACP's principled noise correction addresses the root cause rather than the symptom of calibration plateau at 0.051087.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Replaced Phase 2b QXGBoost (from exp 0017) with Phase 2b NACP: implemented per-class conformity scores (z-score of true-class probability within the class's prediction distribution), auxiliary features (entropy, max_proba, margin), proxy noise labels (conformity_z > 1.5 AND max_proba < 0.6, plus entropy_z > 2.0 at 0.5 weight), and logistic regression to predict noise probability from 7 features; (3) Updated Phase 3b to compare NACP-weighted isotonic vs unweighted isotonic (N=500 bins) and updated results logging notes from "QXGB-uncertainty-iso" to "NACP-iso". No changes to prepare.py.

## Result
- XGBoost: 30 trials, best=0.0531 | LightGBM: 5 trials, best=0.0548 | CatBoost: 25 trials, best=0.0531
- Preliminary ensemble (for NACP features): w=(0.46,0.05,0.49), val_loss=0.0525
- Proxy noise rate: 2.63% (3317/126000 samples — slightly above the ~1.4% estimated noise rate)
- NACP weight distribution: near-uniform (min=0.10, median=0.998, max=0.999) — only 4.3% of samples had weight < 0.5
- 3-model final weights: XGB=0.47, LGB=0.04, CAT=0.49
- **Unweighted isotonic: N_bins=500, val_loss=0.051049** ← wins
- **NACP-weighted isotonic: N_bins=500, val_loss=0.051225** ← loses by 0.000176
- **Final: 0.0510** — **no improvement** from previous best 0.0505

## What I learned
- **NACP weighting is near-uniform and cannot meaningfully shift isotonic calibration**: The NACP noise proxy (conformity_z > 1.5 AND max_proba < 0.6) identified only 2.63% of samples as likely noisy. After converting to weights (clip(1 - noise_prob, 0.1, 1.0)), 95.7% of samples received weight > 0.5 and the weight distribution was essentially flat (median=0.998, max=0.999). This near-uniform weighting cannot meaningfully shift isotonic's bin means, explaining the null result (0.051225 vs 0.051049, difference of 0.000176 — within noise). The NACP approach's theoretical promise of principled noise correction fails because the proxy noise signal is too weak and diffuse to distinguish noisy from clean labels at the precision needed to shift isotonic's piecewise-constant mapping.
- **Isotonic's built-in bin averaging already handles ~1.4% noise implicitly without explicit correction**: The NACP approach hypothesized that isotonic's piecewise-constant mapping treats every validation label as ground truth and would be distorted by noisy labels. But isotonic's equal-frequency binning averages each sample's label into bin means across 252 samples per bin (126K/500), diluting individual noisy labels by a factor of ~252x. At 1.4% noise rate, even the most anomalous samples contribute at most ~0.5% noise to their bin mean — far below the noise floor. NACP's explicit correction is unnecessary when isotonic's implicit regularization already achieves the same effect.
- **The proxy noise indicators (conformity + entropy thresholds) lack sufficient signal**: The NACP approach relied on proxy noise labels computed from model prediction characteristics rather than ground-truth noise labels (which are unavailable). The proxy noise rate (2.63%) slightly exceeded the KB's estimated noise rate (1.4%), suggesting the thresholds were too aggressive and generated false positives that contaminated the noise signal. A more conservative proxy threshold might produce a sparser noise signal, but the fundamental problem remains: without ground-truth noise labels, the proxy noise signal is a noisy estimate of a noisy estimate — too second-order to reliably shift isotonic.
- **The calibration axis is comprehensively closed**: JSD-weighting (exp 0013, essentially zero), QXGBoost-uncertainty-weighting (exp 0017, null), and NACP-noise-weighting (this experiment, null) all confirm that isotonic calibration on this dataset does not benefit from sample-specific guidance. The ~0.0010 improvement from isotonic is already maximal; explicit sample weighting (whether density-based, uncertainty-based, or noise-based) cannot improve on isotonic's built-in equal-frequency bin averaging. The calibration ceiling is confirmed. All three calibration refinement sub-axes are closed.

## KB entries consulted
- JSD-weighting provides essentially zero improvement over uniform isotonic weighting (+0.000005–0.000013, within noise); miscalibration is not concentrated in sparse feature regions (exp 0013) — **confirmed and extended: QXGBoost-uncertainty-weighting (exp 0017) and NACP-noise-weighting (this experiment) also provide zero improvement, confirming miscalibration is uniformly distributed and already handled by isotonic's built-in smoothing**
- **Calibration is not the bottleneck; base model quality is — comprehensively confirmed by exps 0013 (JSD neutral), 0012 (NA-FIR neutral), 0011 (warm-start regression), 0010 (bin regularization neutral beyond 500 bins)** — **extended: QXGBoost-uncertainty-weighting (exp 0017) and NACP-noise-weighting (this experiment) also neutral; calibration ceiling confirmed across 5 independent approaches**
- NACP noise-aware conformal prediction for isotonic calibration has **never been tried** (arxiv 2501.12749 — consumed exp 0018) — **axis now closed: NACP noise-weighting is near-uniform and cannot meaningfully shift isotonic's bin means; null result confirms isotonic already handles noise implicitly; calibration refinement axis comprehensively closed**
