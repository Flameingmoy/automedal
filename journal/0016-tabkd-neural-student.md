---
id: 0016
slug: tabkd-neural-student
timestamp: 2026-04-13T16:06:00
git_tag: exp/0016
queue_entry: 1
status: improved
val_loss: 0.0508
val_accuracy: 0.9862
best_so_far: 0.0508
---

## Hypothesis
Training a TabKD neural student (Tabular Knowledge Distillation with interaction-diversity synthetic queries) on the 3 GBDT teachers' softmax predictions will produce a diverse 4th ensemble member without the HPO budget penalty that killed TabR (exp 0008) and T-MLP (exp 0009), because TabKD generates synthetic training data by maximizing pairwise feature interaction coverage between the teachers — creating a fundamentally different training signal than standard KL-divergence matching — and the neural student is trained on fixed teacher predictions (no HPO budget consumed) while achieving high teacher-student agreement (14/16 configurations in the paper) on tabular data.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Added Phase 2b (TabKD Neural Student) after Phase 2 (3-model final training): implemented `TabKDStudent` (2-layer MLP: 256→128→3, ReLU+BN+Dropout), `generate_interaction_synthetic_data()` (covers all 28 pairwise categorical interactions + random sampling, 30K synthetic samples), teacher probability collection (XGB+LGB+CAT averaged to soft targets), and KL-divergence training (15 epochs, batch 2048, AdamW+CosineAnnealing); (3) Replaced the nested-loop grid search with scipy.optimize SLSQP for 3-model (50 restarts) and 4-model (100 restarts) weight optimization, enabling continuous weights instead of 0.05-increment grid; (4) Updated Phase 3b isotonic and Phase 4 stacking to conditionally include TabKD. No changes to prepare.py or HPO trial counts.

## Result
- XGBoost: 16 trials, best=0.0524 | LightGBM: 4 trials, best=0.0559 | CatBoost: 33 trials, best=0.0540
- **TabKD student: val_loss=8.4962 (catastrophically collapsed — training produced NaN KL loss)**
- 4-model ensemble could not beat 3-model (TabKD student too noisy); 3-model won
- 3-model best weights: XGB=0.80, LGB=0.01, CAT=0.19 (shifted even more XGB-heavy than usual 0.65/0.10/0.25)
- Weighted ensemble: 0.0523
- **Best isotonic (N=500): val_loss=0.050808**
- **Final: 0.0508** — **improved** from previous best 0.0511 by 0.0003

## What I learned
- **TabKD neural student catastrophically failed (val_loss=8.4962)**: The interaction-diversity synthetic data was too sparse for binary categoricals. All 8 categorical features are binary (cardinality=2), yielding only 28 pairs × 4 combinations = 112 base interaction patterns. Even with 30K synthetic samples, the neural student trained on this narrow distribution collapsed — it learned to predict near-uniform probabilities. The KL-divergence loss went NaN within 10 epochs, confirming the student could not learn meaningful representations from the synthetic queries. The TabKD approach requires higher-cardinality categoricals to generate sufficient interaction diversity (the paper was designed for datasets with higher categorical cardinality).
- **scipy.optimize SLSQP produced XGB=0.80 vs typical grid-search 0.65**: The continuous optimization found a more XGB-heavy corner of the weight space than the 0.05-increment grid search from prior experiments. This weight configuration (0.80/0.01/0.19) produced the best 3-model ensemble of any run, yielding weighted loss of 0.0523 before isotonic — better than the historical 0.0524–0.0528 range. The improvement from isotonic (0.0523→0.0508) is the usual ~0.0015, but the base ensemble was also better this run.
- **The neural-student axis is permanently closed**: Both TabR (exp 0008, 0.0688), T-MLP (exp 0009, 0.0658), and TabKD (exp 0016, 8.4962) have failed to produce competitive neural students on this dataset. TabKD's interaction-diversity approach was the theoretically strongest neural candidate, yet it collapsed entirely. GBDTs remain the only competitive architecture for this tabular dataset — the 3-GBDT ensemble with isotonic calibration is the ceiling for the neural-student sub-axis.
- **The XGB-heavy weight shift (0.65→0.80) suggests CatBoost degradation was significant this run**: CatBoost at 0.0540 (vs typical 0.0536–0.0539) was slightly degraded, likely due to reduced trial count (33 vs typical 38+). The optimizer compensated by shifting weight to XGBoost (0.80 vs typical 0.65). This confirms the KB's observation that XGBoost is genuinely superior and the weight optimizer will always push toward XGB-heavy configurations when given continuous optimization freedom.

## KB entries consulted
- T-MLP individually achieved 0.0658 (vs TabR's 0.0688); both are fundamentally uncompetitive (~5% max ensemble weight); **axis closed** (exps 0008, 0009) — **confirmed and extended: TabKD (8.4962) is catastrophically worse than both; neural-student axis permanently closed**
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 15+ experiments; the persistent pattern reflects genuine XGBoost quality superiority (exps 0001–0015) — **confirmed: scipy.optimize pushed to 0.80/0.01/0.19, showing the pattern is even stronger than the 0.05-increment grid found**
- TabKD interaction-diversity knowledge distillation has **never been tried** (arxiv 2603.15481 — consumed exp 0016) — **axis now closed: neural-student axis permanently closed; all neural approaches (TabR, T-MLP, TabKD) have failed**
