# Section 3 — Baseline Model

## What We Built
Two baseline classifiers for EMI eligibility prediction:

**1. Rule-Based Classifier** (`src/models/baseline_rules.py`)
Three hard business rules from EDA findings. No training — purely deterministic. Represents "what a credit officer does today manually."

**2. Logistic Regression Baseline** (`src/models/baseline_logistic.py`)
Standard statistical classifier with `class_weight='balanced'` and internal StandardScaler. Provides calibrated probabilities and interpretable coefficients.

---

## Results

Run `notebooks/03_baseline.py` to generate actual values. Template:

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Rule-Based | — | — | — | — |
| Logistic Regression | — | — | — | — |

See `reports/baseline_metrics.json` for the populated values after running.

---

## Key Decisions

| Decision | Value | Reason |
|---|---|---|
| Rule: credit_score floor | ≥ 650 | EDA top predictor; CIBIL "fair" threshold |
| Rule: expense_ratio ceiling | ≤ 0.60 | EDA key signal; >60% = financially stretched |
| Rule: salary floor | ≥ ₹20,000 | Minimum viable EMI capacity |
| class_weight | `'balanced'` | Locked — 4.2:1 imbalance, accuracy misleading |
| Primary metrics | ROC-AUC + F1 | Accuracy excluded — misleading on imbalanced data |
| Encoding | one-hot, drop_first=True | Avoids dummy variable trap in logistic regression |
| Train/test split | 80/20, stratified, seed=42 | Reproducibility + class distribution preserved |
| Evaluation threshold | 0.50 for both models | Same threshold = fair metric comparison |

---

## Why Baselines Matter for This Project

**The rule-based model** is your floor. It represents the institution's current manual process. Every complex model you build in Section 5 must beat it. If XGBoost can't outperform three simple rules, the data doesn't support ML — diagnose the features, not the algorithm.

**Logistic regression coefficients** are the most interpretable signal before going to black-box tree models. A negative coefficient on `expense_ratio` confirms the EDA finding. A large positive coefficient on `credit_score` confirms it's the dominant predictor. This guides what to engineer in Section 4.

**Confidence zones** tell the business team how many staff-hours the ML system saves. If 70% of applicants land in the auto-approve or auto-reject zones, the review team only manually handles 30% — that's the business case for the system.

---

## Files Created

| File | Purpose |
|---|---|
| `src/models/baseline_rules.py` | Rule-based classifier class |
| `src/models/baseline_logistic.py` | Logistic regression class with scaler |
| `notebooks/03_baseline.py` | Evaluation notebook — runs end-to-end |
| `docs/figures/03_confusion_matrices.png` | Side-by-side confusion matrices |
| `docs/figures/03_roc_curves.png` | Overlaid ROC curves |
| `docs/figures/03_lr_coefficients.png` | Top 15 LR feature coefficients |
| `docs/figures/03_confidence_zones.png` | Business zone distribution |
| `reports/baseline_metrics.json` | Structured metrics for Section 6 MLflow |

---

## Next Section
**Section 4 — Feature Engineering**
Guided by: LR coefficient chart (which features need transforming) + EDA signals (expense ratio variants, credit score binning, salary utilisation ratio).
