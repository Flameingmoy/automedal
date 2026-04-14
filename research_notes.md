# Research Notes

## exp 0022 · stagnation · query: "synthetic tabular data generation gradient boosting class imbalance ensemble weighting constrained optimization"
- Paper: "XGenBoost: Synthesizing Small and Large Tabular Datasets with XGBoost" (arxiv 2603.06904v1) [consumed in exp 0022]
  Summary:
    - XGenBoost uses XGBoost's natural inductive biases (axis-aligned splits, mixed-type handling) as a generative model for tabular data synthesis, avoiding the one-hot encoding and continuous-assumption problems of neural generative approaches.
    - For large datasets, XGenBoost employs a hierarchical autoregressive model whose conditionals are learned via XGBoost classifiers, imposing ordinal inductive biases on numerical features and using empirical quantile functions for dequantization.
    - For small datasets, XGenBoost uses a Denoising Diffusion Implicit Model (DDIM) with XGBoost as the score estimator, combining Gaussian and multinomial diffusion to leverage native categorical splits.
  Applicable idea: Using XGenBoost's hierarchical XGBoost-classifier autoregressive model to generate synthetic minority-class training samples may reduce val_loss because XGBoost-generated samples align with the classifier's own decision boundaries (unlike SMOTE's linear interpolation), giving XGB/LGB/CatBoost training data that better reflects the true class-conditional distributions — and with 900K training rows, the XGBoost synthesizer has rich conditional distributions to sample from.
- Paper: "A General Weighting Theory for Ensemble Learning: Beyond Variance Reduction via Spectral and Geometric Structure" (arxiv 2512.22286v1) [consumed in exp 0026]
  Summary:
    - The paper extends classical variance-reduction ensemble theory by formalizing ensembles as linear operators on hypothesis spaces endowed with geometric and spectral constraints, showing that non-uniform structured weights outperform uniform averaging by reshaping approximation geometry and redistributing spectral complexity.
    - Optimal ensemble weights arise as solutions to constrained quadratic programs (QP), which can be solved analytically or via standard QP solvers — providing a principled alternative to scipy.optimize SLSQP gradient-based search.
    - The theory accommodates geometric, sub-exponential, and heavy-tailed weighting laws as special cases, and provides conditions under which structured (non-uniform) weighting provably dominates uniform ensembles even when variance reduction is negligible.
  Applicable idea: Replacing scipy.optimize SLSQP with a constrained QP solver (e.g., `scipy.optimize.minimize` with method='SLSQP' on a QP-formulated objective) for the XGB+LGB+CatBoost ensemble weights may reduce val_loss because the General Weighting Theory proves optimal weights are solutions to constrained QPs rather than gradient-based search — and the QP formulation naturally handles the sum-to-one constraint and weight bounds without iterative optimization, potentially finding better optima than SLSQP's local search.
- Paper: "Generalized Oversampling for Learning from Imbalanced datasets and Associated Theory" (arxiv 2308.02966v1) [consumed in exp 0022]
  Summary:
    - The GOLIATH algorithm provides a unified oversampling framework based on kernel density estimates that encompasses both SMOTE-style interpolation and Gaussian-noise perturbation as special cases, with explicit conditional density expressions for SMOTE that reveal why it fails on certain distributions.
    - GOLIATH uses wild-bootstrap resampling for target values combined with kernel density-based feature generation, demonstrating significant improvement over state-of-the-art oversampling methods on imbalanced regression and classification tasks.
    - The kernel density approach handles multimodal and non-convex minority-class distributions better than SMOTE's convex interpolation between minority samples, which can generate samples in majority-class regions.
  Applicable idea: Applying GOLIATH's kernel density-based oversampling to a stratified subsample of 100K training rows (to stay within the 10-minute budget) may reduce val_loss because SMOTE's linear interpolation between minority samples can generate synthetic points in majority-class regions, while GOLIATH's kernel density estimator captures local class-conditional structure — improving minority-class boundary quality for LGB and CatBoost, which currently receive near-zero ensemble weight (LGB=0.01–0.10) partly due to weak minority-class predictions.

## exp 0021 · stagnation · query: "SMAC hyperparameter optimization tree boosting adaptive gradient boosting"
- Paper: "MorphBoost: Self-Organizing Universal Gradient Boosting with Adaptive Tree Morph" (arxiv 2511.13234v1) [consumed in exp 0022]
  Summary:
    - Traditional GBDTs use static tree structures with fixed splitting criteria throughout training, unable to adapt to evolving gradient distributions across different learning stages.
    - MorphBoost introduces morphing split criteria that combine gradient-based scores with information-theoretic metrics weighted by training progress, enabling automatic adjustment to problem complexity as the model learns.
    - Outperforms XGBoost by 0.84% on average across 10 diverse datasets, securing 4/10 dataset wins (40% win rate) and 6/30 top-3 finishes while maintaining the lowest variance (σ=0.0948) and highest minimum accuracy.
  Applicable idea: Replacing XGBoost with MorphBoost as the primary ensemble member may reduce val_loss because its adaptive tree morphing produces better individual models than static-split XGBoost (0.84% improvement on benchmarks), and MorphBoost's superior consistency (lowest variance) could shift the persistent XGB-heavy weight pattern toward more balanced contributions.
- Paper: "Selecting Hyperparameters for Tree-Boosting" (arxiv 2602.05786v1)
  Summary:
    - Compares 6 HPO methods (random grid search, TPE, GP-BO, Hyperband, SMAC, full grid) across 59 regression and classification datasets for tree-boosting, finding that SMAC clearly outperforms all other methods.
    - A relatively large number of trials (>100) is required for accurate tree-boosting tuning; using default hyperparameters yields very inaccurate models across all datasets.
    - All considered hyperparameters can have a material effect on accuracy — there is no small critical subset that dominates — and early stopping for boosting iterations outperforms including n_estimators in the search space.
  Applicable idea: Re-running HPO with SMAC (via `smac` or `smac3` library) instead of Optuna TPE for XGBoost/LightGBM/CatBoost may reduce val_loss because the paper's comprehensive comparison across 59 datasets shows SMAC consistently outperforms TPE, and SMAC's surrogate model-based approach may find better hyperparameter configurations that Optuna's TPE sampler de-prioritizes — especially since the agent's current best (0.0505) uses standard Optuna TPE.

## exp 0005 · stagnation · query: "pseudo-label selection confidence variance calibration stacking"
- Paper: "A Confidence-Variance Theory for Pseudo-Label Selection in Semi-Supervised Learning" (arxiv 2601.11670v2) [consumed in exp 0005]
  Summary:
    - Fixed confidence thresholds for pseudo-labeling fail because deep networks are overconfident: high-confidence predictions can still be wrong while informative low-confidence samples near decision boundaries are discarded.
    - CoVar theory combines Maximum Confidence (MC) with Residual-Class Variance (RCV), where RCV measures how probability mass spreads across non-maximum classes — reliable pseudo-labels need both high MC and low RCV.
    - The influence of RCV grows as confidence grows, correcting overconfident-but-unstable predictions that fixed thresholds would accept as correct.
  Applicable idea: Applying CoVar-based pseudo-label selection (high MC + low RCV) on the 270K test rows instead of a fixed 0.95 confidence threshold may reduce val_loss because the current ensemble is systematically overconfident (temperatures ~0.94–0.95) and CoVar's variance criterion filters out the high-confidence-but-wrong predictions near class boundaries that a raw max-probability threshold would include.
- Paper: "Self Adaptive Threshold Pseudo-labeling and Unreliable Sample Contrastive Loss for Semi-supervised Image Classification" (arxiv 2407.03596v1) [consumed in exp 0019]
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
- Paper: "Multiclass Local Calibration With the Jensen-Shannon Distance" (arxiv 2510.26566v1) [consumed in exp 0025]
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

- Paper: "Revisiting Self-Training with Regularized Pseudo-Labeling for Tabular Data" (arxiv 2302.14013v3) [consumed in exp 0028]
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

## exp 0009 · stagnation · query: "gradient boosting ensemble diversity feature bagging pseudo-labeling robust training"
- Paper: "Robust-GBDT: GBDT with Nonconvex Loss for Tabular Classification in the Presence of Label Noise and Class Imbalance" (arxiv 2310.05067v2) [consumed in exp 0016]
  Summary:
    - Standard cross-entropy GBDTs overfit noisy labels because CE uniformly penalizes all misclassifications, including those in ambiguous regions.
    - Robust Focal Loss (RFL) adapts the focusing parameter γ per sample based on local loss convexity — down-weighting samples in low-convexity (ambiguous) regions while preserving signal from clean, high-convexity regions.
    - RFL integrates seamlessly with existing GBDT libraries (XGBoost/LightGBM/CatBoost) via custom objective; requires only the loss gradient and hessian, no architecture changes.
    - Outperforms standard CE, standard focal loss, and other noise-robust methods on 10+ tabular datasets with injected label noise.
  Applicable idea: Training CatBoost with Robust Focal Loss (RFL) as a 4th diverse ensemble member alongside standard-log-loss XGB+LGB+CAT may reduce val_loss because standard focal loss failed catastrophically in exp 0004 (avg sample weight 0.0015, collapsed XGB to 0.10 weight), but RFL's per-sample adaptive focusing preserves training signal while producing fundamentally different error patterns than CE-trained models — giving the ensemble a complementary member that CatBoost's CE-trained self cannot provide.
- Paper: "Team up GBDTs and DNNs: Advancing Efficient and Effective Tabular Prediction with Tree-hybrid MLPs" (arxiv 2407.09790v1) [consumed in exp 0023]
  Summary:
    - Observes that DNNs and GBDTs dominate different tabular datasets because they capture different function classes — DNNs learn smooth continuous mappings, GBDTs learn axis-aligned piecewise-constant rules.
    - Proposes T-MLP: a simple MLP whose first-layer weights are initialized by a GBDT's entropy-driven feature importance scores, effectively converting tree splits into learned linear projections.
    - T-MLP is trained with standard backpropagation on the GBDT-initialized weights, achieving competitive performance with GBDTs on GBDT-friendly datasets and with DNNs on DNN-friendly datasets.
    - The GBDT feature gate provides a principled inductive bias for tabular DL without requiring the full GBDT at inference time.
  Applicable idea: Training a T-MLP (GBDT-feature-gated MLP) as a 4th ensemble member alongside persisted XGB+LGB+CatBoost configs may reduce val_loss because the GBDT-initialized MLP has a fundamentally different inductive bias from all three GBDTs (continuous smooth functions vs axis-aligned trees), breaking the persistent XGB-heavy weight pattern (0.65/0.10/0.25) that indicates the three GBDTs are producing correlated errors — and T-MLP's competitive individual accuracy avoids the TabR failure mode (0.0688 individual) that caused exp 0008 to regress.
- Paper: "How Ensemble Learning Balances Accuracy and Overfitting: A Bias-Variance Perspective on Tabular Data" (arxiv 2512.05469v1) [consumed in exp 0015]
  Summary:
    - On tabular datasets with nonlinear structure, tree-based ensembles improve test accuracy by 5–7 points while keeping generalization gaps below 3%, primarily by reducing variance through averaging or controlled boosting.
    - On noisy or highly imbalanced tabular datasets, ensembles require stronger regularization to avoid fitting noise or majority-class patterns; isotonic calibration may overfit noisy samples in the tail of the probability distribution.
    - Dataset complexity indicators (linearity score, Fisher ratio, noise estimate) predict when ensembles will control overfitting effectively — noisy, high-dimensional data needs stronger regularization.
    - Suggests that the optimal ensemble strategy depends on the data's noise profile: clean data favors averaging, noisy data favors regularized diversity.
  Applicable idea: Re-tuning isotonic calibration's number of bins (currently using default sklearn IsotonicRegression) based on the ensemble's sensitivity to the noise floor may reduce val_loss because exp 0005's isotonic calibration plateau (0.0514 best) may reflect isotonic overfitting on noisy validation samples, and constraining the isotonic mapping to fewer bins (e.g., 50–100) would regularize the calibration curve against fitting noise — especially relevant since the dataset is 98.6% accurate and the ~1.4% noise level is non-negligible for calibration.

## exp 0010 · stagnation · query: "isotonic calibration multiclass normalization tabular knowledge distillation interaction diversity warm-start HPO"
- Paper: "Improving Multi-Class Calibration through Normalization-Aware Isotonic Techniques" (arxiv 2512.09054v1) [consumed in exp 0027]
  Summary:
    - Standard one-vs-rest isotonic regression for multiclass calibration ignores probability normalization constraints, producing suboptimal log-loss because the per-class isotonic mappings can independently shift class probabilities in ways that violate the sum-to-one property.
    - NA-FIR incorporates normalization directly into the isotonic optimization objective, while SCIR models the problem as cumulative bivariate isotonic regression — both enforce that calibrated probabilities sum to one across classes.
    - Across text and image classification benchmarks, normalization-aware isotonic methods consistently improve NLL and ECE over both plain isotonic regression and parametric methods like temperature scaling.
  Applicable idea: Applying normalization-aware isotonic calibration (NA-FIR or SCIR) to the XGB+LGB+CatBoost ensemble's softmax outputs may reduce val_loss below 0.051357 because the current isotonic plateau likely reflects the one-vs-rest approach's inability to enforce sum-to-one normalization across classes, and the ~0.0010 isotonic improvement over weighted ensemble in exp 0017 suggests there is still room for a more principled multiclass calibration method to squeeze out additional log-loss reduction.
- Paper: "TabKD: Tabular Knowledge Distillation through Interaction Diversity of Learned Feature Bins" (arxiv 2603.15481v1) [consumed in exp 0020, exp 0016]
  Summary:
    - Standard knowledge distillation fails on tabular data because tabular models encode predictive knowledge through feature interactions, not just final logits — existing KD methods ignore these interaction patterns.
    - TabKD learns adaptive feature bins aligned with teacher decision boundaries, then generates synthetic queries that maximize pairwise interaction coverage, ensuring the student captures the full diversity of the teacher's feature interactions.
    - Across 4 benchmark datasets and 4 teacher architectures, TabKD achieves highest student-teacher agreement in 14/16 configurations, validating that interaction coverage correlates strongly with distillation quality.
  Applicable idea: Training a neural network student model on XGB+LGB+CatBoost ensemble predictions using TabKD's interaction-diversity approach (rather than standard KL-divergence logit matching) may reduce val_loss because TabR's failure (exp 0008, 0.0688 individual) was due to the student lacking the teacher's interaction-level knowledge — TabKD's synthetic data generation with pairwise interaction coverage would produce a genuinely diverse student without the time-budget penalty of a 4th model HPO run, and interaction diversity may produce the ensemble diversity the 3 GBDTs have failed to achieve.
- Paper: "Are encoders able to learn landmarkers for warm-starting of Hyperparameter Optimization" (arxiv 2507.12604v1) [consumed in exp 0026]
  Summary:
    - Warm-starting Bayesian HPO across heterogeneous tabular datasets requires dataset representations that capture landmarker properties, which standard universal embeddings fail to do.
    - Two proposed encoders — deep metric learning and landmarker reconstruction — both learn representations aligned with landmarker characteristics, though the translation to actual HPO warm-start gains was modest in experiments.
    - The core insight is that Optuna trial histories from prior experiments can serve as landmarkers: a meta-model trained on (dataset-features, HPO-trial-metadata) → best-hyperparams provides a strong prior for new HPO runs on the same dataset.
  Applicable idea: Using Optuna trial histories from the 20+ prior experiments (which span a wide range of hyperparameter configurations and their val_losses) as a warm-start prior for the next HPO run may reduce val_loss because the current approach of fresh random starts wastes information from failed trials — a meta-learner on prior trial histories could bias Optuna toward regions that produced strong val_losses on this specific dataset, compensating for the reduced trial counts that any future experiment must accept due to the 10-minute budget.

[consumed in exps 0018, 0021]
  Summary:
    - Label noise (~1.4% in our dataset per exp 0010's isotonic overfitting evidence) significantly impairs GBDT performance, increases model complexity, and distorts feature selection, yet most noise-robust research focuses on neural networks rather than GBDTs.
    - Proposes gradient-based noise detection methods for GBDTs that achieve >99% noise detection accuracy on tabular datasets (Adult, Covertype) across varying noise levels, adapting techniques from deep learning.
    - Early stopping combined with noise detection maintains model performance under label noise; the Gradients-based noise detection method is specifically designed for the GBDT context where gradient magnitudes encode loss curvature.
  Applicable idea: Training XGBoost/LightGBM/CatBoost with a noise-robust GBDT pipeline (noise detection → label correction or sample reweighting) before the isotonic calibration step may reduce val_loss because the dataset is 98.6% accurate (~1.4% noise), and exp 0010's isotonic calibration plateau likely reflects isotonic overfitting on noisy validation samples — cleaning training labels at source eliminates the root cause rather than patching it with bin regularization.
- Paper: "Self-Error Adjustment: Theory and Practice of Balancing Individual Performance and Diversity in Ensemble Learning" (arxiv 2508.04948v1) [consumed in exps 0017, 0021, 0025]
  Summary:
    - Standard ensemble diversity methods (Bagging, Boosting, Negative Correlation Learning) lack precise control over the accuracy-diversity tradeoff; NCL suffers from loose theoretical bounds and limited adjustment range.
    - Self-Error Adjustment (SEA) decomposes ensemble error into self-error terms (individual model quality) and diversity terms (inter-model error correlations), enabling an adjustable parameter λ that precisely controls the contribution of each component to the loss.
    - SEA provides a broader range of effective adjustments and tighter theoretical bounds than NCL, validated on regression and classification datasets — the adjustable λ enables finding the optimal accuracy-diversity operating point per dataset.
  Applicable idea: Replacing fixed Caruana-style weighted averaging (0.65/0.10/0.25) with Self-Error Adjustment (SEA) as the ensembling strategy may reduce val_loss because the persistent XGB-heavy weight pattern signals that the ensemble cannot precisely control the accuracy-diversity tradeoff — SEA's λ parameter explicitly balances each GBDT's individual quality against its error correlation with the other members, potentially reducing LGB's near-zero contribution (0.10) by penalizing XGB-CAT error correlation that forces the weight redistribution.
- Paper: "Tabular Learning: Encoding for Entity and Context Embeddings" (arxiv 2403.19405v1) [consumed in exp 0014]
  Summary:
    - Ordinal encoding, the default for categorical features in tabular learning, is not the most suitable encoder for GBDTs either — ordinal encoding imposes an arbitrary numerical ordering on categorical levels that may not reflect their actual predictive relationship with the target.
    - Frequency encoding (count-based) and similarity encoding (based on co-occurrence patterns) consistently outperform ordinal encoding for tabular classification across entity and context embedding tasks.
    - The improvement from non-ordinal encodings is most pronounced when the categorical feature has many levels (high cardinality) and when the classification task is multi-class — both conditions present in this dataset (8 categorical features, 3-class target).
  Applicable idea: Replacing ordinal encoding with frequency encoding (category count / total count) for the 8 categorical features may reduce val_loss because ordinal encoding's arbitrary ordering forces GBDTs to model a false ordinal structure that distorts split quality — frequency encoding preserves the actual distributional information (how common each category is) that GBDTs can exploit for more informed splits, and this distributional signal is complementary to the existing numeric features rather than being artificially ordinal.

## exp 0015 · stagnation · query: "self-error adjustment ensemble diversity gradient boosting feature bagging noise-robust training"
- Paper: "Self-Error Adjustment: Theory and Practice of Balancing Individual Performance and Diversity in Ensemble Learning" (arxiv 2508.04948v1) [consumed in exps 0016, 0021]
  Summary:
    - SEA decomposes ensemble error into self-error terms (individual model quality) and diversity terms (inter-model error correlations), with an adjustable λ parameter that explicitly controls the contribution of each component to the loss.
    - Unlike Negative Correlation Learning (NCL) which has loose theoretical bounds and limited adjustment range, SEA provides tighter theoretical guarantees and broader effective λ range, validated on both regression and classification datasets.
    - SEA enables fine-grained control: small λ favors accuracy (like standard ensemble), large λ promotes diversity, and the optimal λ is dataset-specific and can be found via simple grid search.
  Applicable idea: Replacing fixed-weight Caruana ensembling (0.65/0.10/0.25) with Self-Error Adjustment (SEA) may reduce val_loss because the persistent XGB-heavy pattern signals that the ensemble cannot precisely control accuracy-diversity tradeoff — SEA's λ parameter can explicitly penalize XGB-CatBoost error correlation and promote LGB diversity, potentially redistributing weight from the dominant XGB member toward a more balanced configuration that leverages all three GBDTs.
- Paper: "Training Gradient Boosted Decision Trees on Tabular Data Containing Label Noise for Classification Tasks" (arxiv 2409.08647v2) [consumed in exps 0018, 0021]
  Summary:
    - Label noise (~1.4% estimated in this dataset) impairs GBDT performance, increases model complexity, and distorts feature selection, yet most noise-robust research focuses on neural networks rather than GBDTs.
    - Proposes gradient-based noise detection methods for GBDTs that achieve >99% noise detection accuracy on tabular datasets (Adult, Covertype) across varying noise levels, by exploiting the fact that gradient magnitudes encode loss convexity information.
    - Early stopping combined with noise detection maintains model performance under label noise; the "Gradients" detection method is specifically designed for the GBDT context and extends naturally to XGBoost/LightGBM/CatBoost via custom objective hooks.
  Applicable idea: Training XGBoost with a noise-robust GBDT pipeline (gradient-based noise detection → sample reweighting) before the isotonic calibration step may reduce val_loss because the ~1.4% label noise in this dataset is distorting GBDT split quality and calibrating a noisy validation set with isotonic regression produces a mapping that is itself contaminated — cleaning or down-weighting noisy training labels at source eliminates the root cause of the isotonic plateau rather than patching it with bin-regularization variants.
- Paper: "How Ensemble Learning Balances Accuracy and Overfitting: A Bias-Variance Perspective on Tabular Data" (arxiv 2512.05469v1)
  Summary:
    - On tabular datasets with nonlinear structure, tree-based ensembles improve test accuracy by 5–7 points while keeping generalization gaps below 3%, primarily by reducing variance through averaging or controlled boosting.
    - On noisy or highly imbalanced tabular datasets, ensembles require stronger regularization to avoid fitting noise or majority-class patterns; isotonic calibration may overfit noisy samples in the tail of the probability distribution.
    - Dataset complexity indicators (linearity score, Fisher ratio, noise estimate) predict when ensembles will control overfitting effectively — noisy, high-dimensional data needs stronger regularization.
    - GBDT feature bagging (training each GBDT on a random 60% feature subset) is a form of controlled variance reduction that creates intra-model diversity without adding a 4th model or halving Optuna budget.
  Applicable idea: Applying feature bagging (train each of the 3 GBDTs on a different random 60% feature subset) within the existing 3-model ensemble may reduce val_loss because exp 0007 confirmed XGB+CAT are the most complementary pair (0.0524 vs 0.0528 with LGB), and feature bagging creates feature-level diversity within each GBDT — XGB trained on features {1,3,5,7,9,11} makes structurally different splits than XGB trained on {2,4,6,8,10,12}, and ensembling these within-model variants increases effective ensemble diversity without the budget penalty of adding a 4th model architecture.

## exp 0016 · stagnation · query: "ensemble diversity theory noise-aware calibration quantile GBDT uncertainty"
- Paper: "A Unified Theory of Diversity in Ensemble Learning" (arxiv 2301.03962v3) [consumed in exp 0016]
  Summary:
    - Diversity is a hidden third dimension in the bias-variance decomposition of ensemble loss, not a free parameter to maximize independently of accuracy.
    - The paper proves exact bias-variance-diversity decompositions for cross-entropy, squared, and Poisson losses, revealing that the optimal ensemble is found by managing a three-way bias-variance-diversity tradeoff rather than maximizing any single component.
    - Diversity effects are label-distribution dependent: for certain label skews, promoting diversity hurts ensemble accuracy, explaining why some diversity-seeking methods fail on specific datasets.
  Applicable idea: Applying the paper's diversity decomposition to the XGB+LGB+CatBoost ensemble may reduce val_loss because the persistent XGB-heavy weights (0.65/0.10/0.25) across 16+ experiments represent the natural optimum of the bias-variance-diversity tradeoff on this dataset — and the theory predicts that forcing diversity (e.g., equal-weight Caruana) will increase bias in ways that outweigh diversity gains, explaining why all attempts to rebalance weights have failed; the Strategist should use this theory to compute the exact optimal diversity level rather than treating it as a free parameter.
- Paper: "Conformal Prediction of Classifiers with Many Classes based on Noisy Labels" (arxiv 2501.12749v2) [consumed in exp 0016]
  Summary:
    - Standard conformal prediction (CP) thresholds calibrated on noisy labels are systematically miscalibrated because the noise distorts the score distribution — high-confidence wrong predictions are treated as high-confidence correct predictions.
    - Noise-Aware Conformal Prediction (NACP) estimates the noise transition matrix from the calibration data and adjusts CP thresholds accordingly, recovering noise-free coverage guarantees even when only noisy labels are available.
    - NACP provides finite-sample coverage guarantees that remain effective even with a large number of classes, validated on multi-class image classification benchmarks with injected label noise.
  Applicable idea: Applying NACP's noise-aware threshold adjustment to the isotonic calibration step may reduce val_loss because the dataset's ~1.4% label noise distorts the isotonic regression mapping (which treats every validation label as ground truth), and NACP's noise-aware conformal threshold provides a principled way to correct for this — specifically, NACP's estimated noise-free thresholds could be used to reweight validation samples in isotonic regression, down-weighting samples whose labels are likely noisy, addressing the root cause of the calibration plateau that all previous isotonic variants failed to break.
- Paper: "Quantile Extreme Gradient Boosting for Uncertainty Quantification" (arxiv 2304.11732v1) [consumed in exp 0016]
  Summary:
    - Standard XGBoost outputs point predictions without uncertainty estimates; quantile regression with Huber loss enables XGBoost to produce probabilistic predictions with prediction intervals (e.g., 10th–90th percentile).
    - The QXGBoost method modifies the XGBoost objective to support quantile regression by replacing the quantile regression hinge with a differentiable Huber approximation, allowing gradient-based optimization.
    - QXGBoost's prediction interval width is a direct measure of per-sample uncertainty that correlates with ensemble miscalibration — wide intervals indicate regions where the ensemble is unreliable.
  Applicable idea: Training a QXGBoost model alongside the standard XGB+LGB+CatBoost ensemble (or distilling its uncertainty signal into isotonic calibration) may reduce val_loss because the QXGBoost's prediction interval width provides a per-sample uncertainty signal that is absent from the current ensemble's point predictions — isotonic calibration can be weighted by inverse-interval-width, giving lower calibration weight to uncertain samples and higher weight to confident ones, directly addressing the miscalibration that the JSD experiment (exp 0013) hypothesized was in sparse regions but couldn't measure.

## exp 0020 · scheduled · query: "adaptive tree boosting tabular knowledge distillation synthetic tabular data"
- Paper: "MorphBoost: Self-Organizing Universal Gradient Boosting with Adaptive Tree Morph" (arxiv 2511.13234v1) [consumed in exp 0021]
  Summary:
    - Traditional gradient boosting uses static tree structures with fixed splitting criteria throughout training, unable to adapt to evolving gradient distributions or problem complexity across different learning stages.
    - MorphBoost introduces morphing split criteria that combine gradient-based scores with information-theoretic metrics, weighted by training progress — enabling the algorithm to adjust its split function automatically as it learns.
    - Key innovations include automatic problem fingerprinting for intelligent configuration, interaction-aware feature importance detecting multiplicative relationships, and vectorized tree prediction for computational efficiency.
  Applicable idea: MorphBoost as a replacement for XGBoost in the 3-model ensemble may reduce val_loss because its adaptive tree morphing addresses the fundamental limitation of static tree structures — the persistent XGB-heavy weights (0.65/0.10/0.25) across 20+ experiments may reflect that XGBoost's static splits are suboptimal for some learning stages, and MorphBoost's automatic adaptation to problem complexity could produce a genuinely better base model that shifts the ensemble weight distribution and reduces overall val_loss below 0.0505.
- Paper: "Selecting Hyperparameters for Tree-Boosting" (arxiv 2602.05786v1) [consumed in exp 0021]
  Summary:
    - Compares 6 HPO methods (random grid search, TPE, GP-BO, Hyperband, SMAC, full grid search) across 59 regression and classification datasets for tree-boosting, finding that SMAC clearly outperforms all other methods.
    - A relatively large number of trials (>100) is required for accurate tree-boosting tuning; using default hyperparameters yields very inaccurate models across all datasets.
    - All considered hyperparameters can have a material effect on accuracy — there is no small critical subset of hyperparameters that dominates — and early stopping for boosting iterations outperforms including n_estimators as a search dimension.
  Applicable idea: Re-running Optuna HPO with SMAC (via `smac` or `smac3` library) instead of TPE-based Optuna may reduce val_loss because the paper's comprehensive comparison across 59 datasets shows SMAC consistently outperforms TPE, and the current Optuna configuration may be leaving performance on the table — especially since all hyperparameters (not just a few) have material effects, meaning the current search ranges may be missing gains in dimensions that Optuna's TPE sampler de-prioritizes.
- Paper: "TabKD: Tabular Knowledge Distillation through Interaction Diversity of Learned Feature Bins" (arxiv 2603.15481v1) [queued exp 0020]
  Summary:
    - Standard knowledge distillation fails on tabular data because existing methods do not explicitly address feature interactions — the fundamental way tabular models encode predictive knowledge — and instead focus on matching final logits or softmax outputs.
    - TabKD learns adaptive feature bins aligned with teacher decision boundaries, then generates synthetic queries that maximize pairwise interaction coverage across feature combinations, ensuring the student captures the full diversity of the teacher's feature interactions.
    - Across 4 benchmark datasets and 4 teacher architectures, TabKD achieves highest student-teacher agreement in 14 out of 16 configurations, outperforming 5 state-of-the-art baselines, with interaction coverage strongly correlating with distillation quality.
  Applicable idea: Training a TabKD neural student on the 3 GBDT teachers' softmax predictions via interaction-diversity synthetic queries (instead of standard KL-divergence logit matching) may reduce val_loss because TabR's failure (exp 0008, 0.0688 individual) was due to the student lacking the teacher's interaction-level knowledge — TabKD's pairwise interaction coverage would produce a genuinely diverse neural student that captures the same feature interaction patterns as the GBDTs, without requiring additional HPO budget since the student is trained on fixed teacher predictions.

## exp 0023 · stagnation · query: "cross-validation early stopping hyperparameter optimization tabular variance reduction minority class oversampling"
- Paper: "Don't Waste Your Time: Early Stopping Cross-Validation" (arxiv 2405.03389v2) [consumed in exp 0027]
  Summary:
    - State-of-the-art AutoML systems use k-fold CV which drastically increases computational cost vs holdout, making model selection prohibitive within time budgets — early stopping k-fold CV during model selection allows faster convergence.
    - Across 36 classification datasets with MLP and random forest, early stopping k-fold CV enables +167% more hyperparameter configurations within the same time budget while achieving better overall performance.
    - The approach works with both random search and Bayesian optimization (Optuna TPE equivalent), and with 3-, 5-, and 10-fold — the efficiency gains are consistent across fold counts.
  Applicable idea: Applying early stopping during Optuna's k-fold cross-validation — stopping individual fold training when the validation loss curve flattens (rather than running full boosting rounds) — may reduce val_loss because the current 60–100 Optuna trials budget is constrained by full boosting training time, and the paper shows early stopping enables +167% more configurations in the same time, giving Optuna more search space coverage that could find better hyperparameter configurations for XGBoost/LightGBM/CatBoost within the 10-minute wall-clock budget.
- Paper: "Irredundant k-Fold Cross-Validation" (arxiv 2507.20048v2) [consumed in exp 0024]
  Summary:
    - Traditional k-fold CV uses each instance (k-1) times for training and once for testing, creating training-set overlap between folds that inflates correlation of validation estimates and produces optimistically biased variance estimates.
    - Irredundant k-fold guarantees each instance is used exactly once for training and once for testing across the full validation procedure — training partitions are non-overlapping, eliminating the redundancy that lets some instances disproportionately influence learning.
    - Delivers consistent performance estimates comparable to standard k-fold while providing less optimistic variance estimates and significantly reducing computational cost by eliminating redundant model training.
  Applicable idea: Applying irredundant k-fold (instead of standard stratified k-fold) for Optuna HPO validation may reduce val_loss because the current setup uses overlapping training folds that artificially reduce validation variance, potentially causing Optuna to select hyperparameters that overfit to the specific validation fold pattern — irredundant k-fold's non-overlapping training partitions give more honest variance estimates and sharper comparative distinctions, enabling Optuna to select more robust hyperparameter configurations that generalize better to the held-out test set.
- Paper: "Adaptive Cluster-Based Synthetic Minority Oversampling Technique for Traffic Mode Choice Prediction with Imbalanced Dataset" (arxiv 2504.09486v1)
  Summary:
    - Standard SMOTE oversampling overgeneralizes minority classes that are closely clustered and neglects sparse regions containing crucial boundary information, while undersampling removes useful subgroup information — cluster-aware oversampling addresses both failure modes.
    - Applies density-based spatial clustering (DBSCAN) on minority classes to identify natural subgroups, then oversamples each subgroup proportionally to its local cluster density rather than uniformly across the minority class.
    - When combined with XGBoost and random forest, adaptive cluster-based oversampling produces significantly higher F1 scores for minority classes vs standard SMOTE and other resampling techniques.
  Applicable idea: Applying adaptive cluster-based SMOTE (cluster each minority class into subgroups via DBSCAN, oversample each subgroup by its cluster-to-majority ratio) to a stratified 100K training subsample before GBDT training may reduce val_loss because standard SMOTE's uniform oversampling within the minority class generates synthetic samples in majority-class regions, while cluster-aware oversampling respects the minority class's natural multimodal structure — this could improve LGB and CatBoost's minority-class predictions, which receive near-zero ensemble weight (0.01–0.10) partly because they fail on minority-class boundary samples.

## exp 0024 · stagnation · query: "cross-validation early stopping budget hyperparameter optimization tabular stratified"
- Paper: "Don't Waste Your Time: Early Stopping Cross-Validation" (arxiv 2405.03389v2) [consumed in exp 0024]
  Summary:
    - Standard k-fold cross-validation drastically increases computational cost vs holdout, making model selection prohibitive within time budgets — early stopping individual fold training when val_loss flattens enables faster convergence.
    - Across 36 classification datasets (MLP, random forest), early stopping k-fold CV enables +167% more hyperparameter configurations within the same time budget while achieving better overall performance.
    - The approach is compatible with both random search and Bayesian optimization (Optuna TPE equivalent), and the efficiency gains are consistent across 3-, 5-, and 10-fold.
  Applicable idea: Implementing early stopping per-fold during Optuna HPO — stopping individual fold training when val_loss plateaus (rather than running full boosting rounds) — may reduce val_loss because the 10-minute budget currently produces only 20 XGB trials (exp 0023) while the paper shows +167% more configurations in the same time, and more trial coverage could find better XGB/LGB/CatBoost hyperparameter configurations that shift the base model quality ceiling above the current 0.050500 isotonic plateau.
- Paper: "Irredundant $k$-Fold Cross-Validation" (arxiv 2507.20048v2) [consumed in exp 0024]
  Summary:
    - Traditional k-fold CV uses each instance (k-1) times for training and once for testing, creating training-set overlap between folds that inflates correlation of validation estimates and produces optimistically biased variance.
    - Irredundant k-fold guarantees each instance is used exactly once for training and once for testing across the full validation procedure — training partitions are non-overlapping, eliminating redundant instance usage.
    - Delivers consistent performance estimates comparable to standard k-fold while providing less optimistic variance estimates and significantly reducing computational cost by eliminating redundant model training.
  Applicable idea: Replacing standard overlapping k-fold with irredundant k-fold (non-overlapping training partitions) for Optuna HPO validation may reduce val_loss because overlapping training folds artificially reduce validation variance, potentially causing Optuna to select hyperparameters that overfit the specific fold pattern — irredundant k-fold's honest variance estimates give sharper comparative distinctions between configurations, enabling Optuna to select more robust hyperparameter settings that generalize better and improve base model quality above the current 0.0534 XGB ceiling.
- Paper: "How Ensemble Learning Balances Accuracy and Overfitting: A Bias-Variance Perspective on Tabular Data" (arxiv 2512.05469v1)
  Summary:
    - On tabular datasets with nonlinear structure, tree-based ensembles improve test accuracy by 5–7 points while keeping generalization gaps below 3%, primarily by reducing variance through averaging or controlled boosting.
    - On noisy or highly imbalanced tabular datasets, ensembles require stronger regularization to avoid fitting noise or majority-class patterns — isotonic calibration may overfit noisy samples in the tail of the probability distribution.
    - Dataset complexity indicators (linearity score, Fisher ratio, noise estimate) predict when ensembles will control overfitting effectively; noisy, high-dimensional data needs stronger regularization.
  Applicable idea: Computing dataset complexity indicators (Fisher ratio, noise estimate per class) from the training data and using them to inform regularization strength and early stopping rounds for XGB/LGB/CatBoost may reduce val_loss because the 98.6% accuracy implies ~1.4% label noise that standard GBDT regularization parameters ignore — Fisher ratio and noise estimates provide data-specific guidance for regularization tuning that Optuna can exploit within its existing budget, improving base model quality without requiring additional experiments or multi-seed averaging.
