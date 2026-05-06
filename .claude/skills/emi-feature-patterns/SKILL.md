---
name: emi-feature-patterns
description: Domain knowledge for EMI feature engineering — Indian lending context
user-invocable: true
---

# EMI Feature Patterns — Domain Knowledge

## Key EDA Findings (notebooks/02_eda.py — do not re-derive)
1. Credit score = strongest single predictor of eligibility
2. Expense ratio = key signal for EMI repayment capacity
3. Salary brackets show non-linear relationship with EMI eligibility
4. Bank balance = lagging indicator (point-in-time snapshot, not trend)
5. monthly_rent presence = significant financial burden signal
6. years_of_employment = income stability proxy

## High-Value Features to Engineer

### Priority 1 — Ratio Features (highest expected signal)
debt_to_income       = (existing_emi + monthly_rent) / monthly_income
expense_ratio        = monthly_expenses / monthly_income
emi_capacity         = (monthly_income - monthly_expenses - monthly_rent) * 0.40
emi_to_income        = existing_emi / monthly_income
savings_rate         = (monthly_income - monthly_expenses) / monthly_income

### Priority 2 — Interaction Features (from salary non-linearity finding)
credit_income_score  = credit_score * np.log1p(monthly_income)
income_stability     = monthly_income / (monthly_expenses + 1)
affordability_index  = emi_capacity / (existing_emi + 1)

### Priority 3 — Binary Stability Flags (from zero-heavy columns)
has_rent_burden      = (monthly_rent > 0).astype(int)
is_employed          = (years_of_employment > 0).astype(int)
is_stable_employed   = (years_of_employment > 2).astype(int)
has_existing_emi     = (existing_emi > 0).astype(int)

### Priority 4 — Credit Score Bands (capture non-linear effects)
300–579  → "poor"
580–669  → "fair"
670–739  → "good"
740–799  → "very_good"
800–850  → "exceptional"

## Features to Avoid
- Raw bank_balance without income normalization (scale too variable)
- Raw age as continuous (4-spike pattern → use 5-year bins instead)
- Raw monthly_income without log transform (right-skewed distribution)

## Validation Checklist for Every New Feature
- [ ] Denominator guarded: +1 or clip to prevent zero-division
- [ ] No NaN introduced: assert feature.isna().sum() == 0
- [ ] Correlation with target > 0.05 (else likely not useful)
- [ ] VIF < 10 (multicollinearity check against existing features)
- [ ] Unit test written and passing
- [ ] Added to sklearn Pipeline (not applied ad-hoc)