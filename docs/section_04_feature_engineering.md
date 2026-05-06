# Section 4 — Feature Engineering

## What We Built
A `FeatureEngineer` class (`src/features/feature_engineering.py`) that adds 21 engineered features to the cleaned dataset, transforming 32 original columns into 53 total features.

---

## The 21 New Features

### Group 1: Financial Ratios (8 features)
| Feature | Formula | Signal |
|---|---|---|
| `expense_ratio` | total_expenses / salary | Already proven strong in baseline |
| `emi_burden_ratio` | current_emi / salary | % of salary locked in existing EMIs |
| `requested_emi_monthly` | requested_amount / tenure | Actual monthly cost of new EMI |
| `requested_emi_ratio` | requested_emi_monthly / salary | How much of salary new EMI consumes |
| `total_emi_ratio` | (current_emi + requested_emi_monthly) / salary | Combined burden after new EMI granted |
| `disposable_income` | salary − total_expenses | Raw ₹ available (can be negative) |
| `savings_ratio` | bank_balance / salary | Months of salary in savings |
| `emergency_months` | emergency_fund / total_expenses | Months of expenses covered by emergency fund |

### Group 2: Credit Banding (2 features)
| Feature | Values | Signal |
|---|---|---|
| `credit_score_band` | Poor/Fair/Good/Excellent | Category label for display |
| `credit_score_ord` | 0/1/2/3 | Ordinal for tree models |

Thresholds: Poor (<600) / Fair (600–649) / Good (650–749) / Excellent (≥750)

### Group 3: Loan Capacity (2 features)
| Feature | Formula | Signal |
|---|---|---|
| `loan_to_income_ratio` | requested_amount / (salary × 12) | Annual salary multiples for loan |
| `loan_to_balance_ratio` | requested_amount / bank_balance | Can savings cover the loan? fillna=50 for ₹0 balance |

### Group 4: Interaction Features (2 features)
| Feature | Formula | Signal |
|---|---|---|
| `credit_x_income` | credit_score × salary / 1e7 | Joint effect: high score + high income |
| `employment_stability` | years_employed / (age − 18) | Proportion of working life employed |

### Group 5: Binary Flags (4 features)
| Feature | Logic | Signal |
|---|---|---|
| `has_emergency_fund` | emergency_fund > 0 | Any safety net at all |
| `is_renter` | house_type == 'Rented' | Renters carry extra expense burden |
| `high_credit` | credit_score ≥ 750 | Clean "excellent credit" flag |
| `low_expense_burden` | expense_ratio ≤ 0.50 | Financially comfortable |

### Group 6: Log-transforms (3 features)
| Feature | Formula | Why |
|---|---|---|
| `log_bank_balance` | log1p(bank_balance) | Reduces right skew for LR |
| `log_monthly_salary` | log1p(monthly_salary) | Reduces right skew for LR |
| `log_requested_amount` | log1p(requested_amount) | Reduces right skew for LR |

---

## Design Decisions

| Decision | Value | Reason |
|---|---|---|
| Fit/Transform pattern | sklearn-compatible class | Prevents leakage — fit on train only |
| Original features kept | All preserved | Let Section 5 models choose via importance |
| `loan_to_balance_ratio` fillna | 50.0 for zero balance | Business rule: no savings = high risk |
| `employment_stability` denominator | clip(min=1) | Prevents div/zero for 18-year-olds |
| `credit_score_band` | String column | For display only — use `credit_score_ord` for training |

---

## Files Created

| File | Purpose |
|---|---|
| `src/features/feature_engineering.py` | FeatureEngineer class |
| `notebooks/04_feature_engineering.py` | Validation + saving |
| `data/processed/train_features.csv` | Enriched training set (Section 5 input) |
| `data/processed/test_features.csv` | Enriched test set (Section 5 input) |
| `data/processed/feature_columns.json` | Canonical feature list (Section 5 + 9) |
| `models/feature_engineer.pkl` | Fitted transformer (Section 9 serving) |
| `docs/figures/04_feature_correlations.png` | Correlation ranking chart |
| `docs/figures/04_engineered_feature_kdes.png` | KDE distributions of top new features |

---

## Next Section
**Section 5 — Model Training**
- Load: `data/processed/train_features.csv`
- Models: RandomForest, XGBoost, LightGBM, ExtraTree (classifiers) + same 4 for regression
- Target to beat: ROC-AUC 0.9763 (Logistic Regression baseline from Section 3)
