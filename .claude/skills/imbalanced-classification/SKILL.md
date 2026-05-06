---
name: imbalanced-classification
description: Handle 4.2:1 class imbalance in EMI eligibility prediction
user-invocable: true
---

# Imbalanced Classification — EMI Predict AI

## Our Situation
Ratio: 4.2:1 (Not Eligible : Eligible)
Decision: class_weight='balanced' — LOCKED
Why locked: preserves real distribution, no synthetic data risk,
            supported natively by all 8 planned models

## Why class_weight='balanced' Is the Right Call Here
- SMOTE: adds synthetic minority samples → financial data risk, pipeline complexity
- Undersampling: discards 80% of majority class → wastes 310,000 real rows
- Oversampling raw: duplicates → model memorizes, not generalizes
- class_weight='balanced': adjusts loss function weights → no data manipulation
  Formula used internally: n_samples / (n_classes * np.bincount(y))

## Evaluation Protocol (Always Follow This)
Never use:
- accuracy_score as primary (misleading: 80.8% by always predicting 0)
- Default 0.5 threshold without justification

Always use:
- ROC-AUC: threshold-independent, handles imbalance correctly
- F1 macro: penalizes poor minority class performance equally
- Classification report: per-class precision, recall, F1
- Confusion matrix: with actual counts (not normalized by default)

## Our Production Thresholds
These are locked — do not suggest changing without user decision:
>  0.85 → AUTO_APPROVE
0.40–0.85 → HUMAN_REVIEW
<  0.40 → AUTO_REJECT

For model selection: optimize F1 on minority class (Eligible = 1)
If precision/recall tradeoff needed: prefer recall (catch eligible customers)
Business cost: False Negative (missing eligible customer) > False Positive

## Debugging Poor Minority Class Performance
1. Verify class_weight='balanced' is actually in model constructor
2. Print class distribution before and after split — confirm 4.2:1 held
3. Check StratifiedKFold used (not KFold)
4. If recall on class 1 < 0.50 → class_weight definitely not working
5. Check y values: are they int (0,1) not float or string?