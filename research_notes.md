# Research Notes

## exp 0005 · stagnation · query: "pseudo-label selection confidence variance calibration stacking"
- Paper: "A Confidence-Variance Theory for Pseudo-Label Selection in Semi-Supervised Learning" (arxiv 2601.11670v2) [consumed in exp 0005]
  Summary:
    - Fixed confidence thresholds for pseudo-labeling fail because deep networks are overconfident: high-confidence predictions can still be wrong while informative low-confidence samples near decision boundaries are discarded.
    - CoVar theory combines Maximum Confidence (MC) with Residual-Class Variance (RCV), where RCV measures how probability mass spreads across non-maximum classes — reliable pseudo-labels need both high MC and low RCV.
    - The influence of RCV grows as confidence grows, correcting overconfident-but-unstable predictions that fixed thresholds would accept as correct.
  Applicable idea: Applying CoVar-based pseudo-label selection (high MC + low RCV) on the 270K test rows instead of a fixed 0.95 confidence threshold may reduce val_loss because the current ensemble is systematically overconfident (temperatures ~0.94–0.95) and CoVar's variance criterion filters out the high-confidence-but-wrong predictions near class boundaries that a raw max-probability threshold would include.
- Paper: "Self Adaptive Threshold Pseudo-labeling and Unreliable Sample Contrastive Loss for Semi-supervised Image Classification" (arxiv 2407.03596v1)
  Summary:
    - Fixed or ad-hoc pseudo-labeling thresholds produce inferior performance and slow convergence because they cannot adapt to the evolving model quality during self-training iterations.
    - Self-adaptive per-class thresholds dynamically adjust to increase the number of reliable pseudo-labeled samples without relying on a pre-defined global cutoff.
    - Unreliable samples below threshold still carry discriminative signal — a contrastive loss on low-confidence samples recovers information that hard thresholding discards.
  Applicable idea: Iterative self-training with self-adaptive per-class confidence thresholds on the XGB+LGB+CatBoost ensemble's test predictions may reduce val_loss because the current pseudo-labeling direction has never been tried, and adaptive thresholds prevent the ensemble's evolving calibration from making a fixed threshold progressively more conservative or liberal.
- Paper: "Regularized boosting with an increasing coefficient magnitude stop criterion as meta-learner in hyperparameter optimization stacking ensemble" (arxiv 2402.01379v1) [consumed in exp 0007]
  Summary:
    - Standard stacking discards all HPO trial models except the best configuration, wasting the information in non-optimal hyperparameter configurations.
    - A regularized boosting meta-learner with an increasing coefficient magnitude stop criterion mitigates multicollinearity that afflicts naive stacking meta-learners (LR is a common choice but sensitive to correlated base predictions).
    - The Caruana method (average over best-subset predictions) handles multicollinearity well but lacks a learning procedure; boosting meta-learners combine learning with regularization.
  Applicable idea: Replacing the logistic regression meta-learner with a regularized boosting meta-learner that stops when coefficient magnitudes start increasing (early stopping on the stacking layer) may reduce val_loss because the persistent XGB-heavy weight pattern (0.65/0.10/0.25) suggests the current LR meta-learner cannot adequately handle the multicollinearity between the three GBDT members, and the exp 0005 stacking replication failures confirm this instability.

## exp 0003 · stagnation · query: "TabPFN categorical transformer ensemble diversity self-training tabular"
- Paper: "Tokenize features, enhancing tables: the FT-TABPFN model for tabular classification" (arxiv 2406.06891v1)
  Summary:
    - TabPFN is a 12-layer transformer pretrained on massive synthetic datasets for in-context tabular classification but is weak on categorical features
    - FT-TABPFN adds a Feature Tokenization layer that converts categorical features into learned token embeddings, enabling the transformer to natively process categoricals
    - FT-TabPFN significantly improves accuracy on tabular classification benchmarks compared to the original TabPFN
  Applicable idea: Fine-tuning FT-TABPFN as a 4th ensemble member alongside XGB+LGB+CatBoost may reduce val_loss because its transformer-with-tokenization architecture is structurally orthogonal to all three GBDTs, breaking the XGB-heavy weight pattern (0.65/0.10/0.25) that indicates correlated errors.
- Paper: "CAST: Cluster-Aware Self-Training for Tabular Data via Reliable Confidence" (arxiv 2310.06380v3) [consumed in exp 0004]
  Summary:
    - Standard self-training for tabular data is fragile because classifiers are overconfident in low-density regions, producing noisy pseudo-labels
    - CAST calibrates pseudo-label confidence by regularizing it based on local density in the labeled data (cluster assumption: nearby points share labels)
    - Achieves superior self-training performance across 21 tabular datasets at negligible computational overhead
  Applicable idea: Using CAST confidence calibration instead of a fixed 0.95 threshold for pseudo-labeling test data may reduce val_loss because the current ensemble's overconfident predictions near decision boundaries in low-density regions are the primary source of pseudo-label noise.
- Paper: "TabPFGen -- Tabular Data Generation with TabPFN" (arxiv 2406.05216v1)
  Summary:
    - Converts TabPFN (a discriminative in-context learner) into an energy-based generative model for tabular data without additional training
    - TabPFGen can sample synthetic tabular rows for data augmentation, class-balancing, and imputation
    - Demonstrates strong results on standard generative modeling tasks including data augmentation
  Applicable idea: Generating augmented training rows via TabPFGen for the minority class(es) may reduce log_loss because the best current approach (0.052421) still suffers from the same XGB-heavy weights across all 8+ experiments, and balanced synthetic data could help the weaker ensemble members (LGB, CatBoost) contribute more equally.

## exp 0001 · stagnation · query: "ensemble confidence calibration multi-class classification"
- Paper: "Confidence Calibration of Classifiers with Many Classes" (arxiv 2411.02988v2) [consumed in exp 0002]
- Paper: "GETS: Ensemble Temperature Scaling for Calibration in Graph Neural Networks" (arxiv 2410.09570v2) [consumed in exp 0003]
  Summary:
    - Transforms multiclass calibration into a single surrogate binary classification problem
    - Uses standard calibration methods (like temperature scaling) more efficiently on many-class problems
    - Significantly enhances existing calibration methods on image and text classification tasks
  Applicable idea: Post-hoc temperature scaling per-class on the 3-model ensemble predictions may reduce log_loss because well-calibrated probabilities improve log_loss directly, and the binary surrogate approach avoids the failure modes of direct multiclass calibration.

- Paper: "GETS: Ensemble Temperature Scaling for Calibration in Graph Neural Networks" (arxiv 2410.09570v2)
  Summary:
    - Combines input augmentation and model ensemble strategies within a joint calibration framework
    - Uses Graph Mixture of Experts architecture to select effective input combinations for calibration
    - Achieves 25% reduction in expected calibration error across 10 benchmark datasets
  Applicable idea: Joint calibration of the XGB+LGB+CatBoost ensemble outputs with per-class temperature scaling may outperform independent temperature scaling because the interaction between models and their prediction distributions is explicitly modeled.

- Paper: "TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023" (arxiv 2307.14338v2) [consumed in exp 0006, 0008]
  Summary:
    - Feed-forward network with custom k-Nearest-Neighbors component in the middle layers
    - Uses attention-like mechanism to retrieve and extract signal from nearest neighbors
    - Outperforms GBDT on the "GBDT-friendly" benchmark while achieving best average among tabular DL models
  Applicable idea: Adding TabR as a fourth diverse ensemble member may reduce log_loss because its retrieval-augmented architecture creates predictions that are structurally different from the 3 GBDTs, increasing ensemble diversity beyond what XGB-heavy weighting alone achieves.

---

## exp 0002 · stagnation · query: "multiclass calibration tabular pseudo-label selection"
- Paper: "Multiclass Local Calibration With the Jensen-Shannon Distance" (arxiv 2510.26566v1) [consumed in exp 0009]
  Summary:
    - Standard multiclass calibration lacks a notion of distance among inputs, causing proximity bias: predictions in sparse feature-space regions are systematically miscalibrated
    - Uses Jensen-Shannon distance to weight calibration training samples by local neighborhood density rather than treating all samples equally
    - Achieves state-of-the-art calibration on image and text benchmarks with only a small overhead compared to temperature scaling
  Applicable idea: Replacing naive temperature scaling with JSD-based local calibration on the XGB+LGB+CatBoost ensemble may reduce log_loss because the ensemble's miscalibration is concentrated in sparse feature regions, and local calibration addresses this without requiring an entirely new model.

- Paper: "A Confidence-Variance Theory for Pseudo-Label Selection in Semi-Supervised Learning" (arxiv 2601.11670v2) [consumed in exp 0005]
  Summary:
    - Fixed confidence thresholds are unreliable because deep networks are overconfident: high-confidence predictions can still be wrong while informative low-confidence samples are discarded
    - Proposes CoVar theory that combines Maximum Confidence (MC) with Residual-Class Variance (RCV), where RCV measures how probability mass distributes across non-maximum classes
    - Derives that reliable pseudo-labels require both high MC and low RCV; designs a threshold-free spectral-relaxation selection mechanism
  Applicable idea: Replacing fixed 0.95 confidence thresholds with CoVar-based pseudo-label selection (high confidence + low variance across non-maximum classes) when pseudo-labeling test data may reduce log_loss because the current ensemble's high-confidence predictions near decision boundaries may still be wrong, and filtering these out prevents noise injection.

- Paper: "How Ensemble Learning Balances Accuracy and Overfitting: A Bias-Variance Perspective on Tabular Data" (arxiv 2512.05469v1) [consumed in exp 0003]
  Summary:
    - Evaluates 9 ensemble methods on 4 tabular classification tasks, finding that ensembles consistently reduce variance without large generalization gaps
    - Gradient boosting ensembles (e.g., random forest, XGBoost, LightGBM) show the best accuracy-overfitting trade-off compared to linear and single decision tree baselines
    - The diversity of base learners is the primary driver of ensemble benefit; accuracy of individual learners is secondary
  Applicable idea: LightGBM DART mode (which applies dropout to trees, creating a pseudo-ensemble-of-subnetworks effect) may reduce log_loss because DART's regularization mechanism structurally decorrelates LightGBM from XGBoost/CatBoost, breaking the XGB-heavy pattern that indicates the three GBDTs are making correlated errors.


## exp 0004 · stagnation · query: "ROC-regularized isotonic calibration focal loss tabular pseudo-labeling"
- Paper: "Classifier Calibration with ROC-Regularized Isotonic Regression" (arxiv 2311.12436v1) [consumed in exp 0004]
  Summary:
    - Standard isotonic regression (IR) calibrates binary classifiers via monotone transformations but ignores class relationships, producing suboptimal rankings in multi-class settings
    - ROC-Regularized Isotonic Regression (ROC-IR) jointly optimizes class boundary ordering alongside calibration error, preserving ROC-AUC structure during calibration
    - For multi-class problems, ROC-IR avoids sacrificing ranking quality for calibration quality — both improve simultaneously rather than trading off
  Applicable idea: Applying ROC-Regularized Isotonic Regression to the XGB+LGB+CatBoost ensemble probabilities (instead of per-class temperature scaling) may reduce log_loss because the ensemble's overconfidence (temperatures ~0.94–0.95) combined with the JSD proximity bias means IR alone could distort the ranking structure, and ROC-IR preserves both calibration and ranking quality simultaneously.

- Paper: "Revisiting Self-Training with Regularized Pseudo-Labeling for Tabular Data" (arxiv 2302.14013v3)
  Summary:
    - Standard self-training methods (e.g., fixed confidence thresholds) are fragile on tabular data because gradient-boosting and neural classifiers behave differently on low-density regions versus high-density regions
    - Proposes a regularized pseudo-labeling approach tailored for tabular data, incorporating density-aware sample weighting and regularization on the student model's predictions
    - Achieves consistent gains across diverse tabular datasets with negligible computational overhead compared to standard self-training baselines
  Applicable idea: Applying regularized pseudo-labeling to the test set (instead of a fixed 0.95 confidence threshold) may reduce val_loss because the ensemble's predictions on low-density tabular feature regions are systematically noisier, and the density-aware weighting mechanism filters out high-confidence-but-wrong pseudo-labels that standard thresholding would include.

- Paper: "Improving Calibration by Relating Focal Loss, Temperature Scaling, and Properness" (arxiv 2408.11598v1) [consumed in exp 0004]
  Summary:
    - Cross-entropy trained classifiers tend to become overconfident on test data due to the generalization gap, and temperature scaling alone cannot fully address this because the root cause is in the training objective
    - Focal loss, while not a proper loss, acts as an implicit regularizer that produces better-calibrated test predictions when used during training without any post-hoc calibration step
    - Establishes a formal connection: focal loss + temperature scaling is optimal (minimizes ECE) when the model's confidence overestimation follows a specific power-law pattern
  Applicable idea: Re-training XGBoost with focal loss as the objective (instead of standard log-loss) may reduce val_loss because the persistent XGB-heavy weight pattern (0.65/0.10/0.25) across 8+ experiments indicates XGBoost is the most overconfident ensemble member, and focal loss's implicit regularization specifically targets the overconfidence that causes the ensemble's log_loss to plateau at ~0.052.
