# EDA Interpretation Guide — EMI Predict AI
**For personal reference | Section 2 findings explained**

---

## How to Read This Guide

Each chart in `02_eda.py` was built to answer a specific question about the data. This guide explains what each chart shows, how to read it, and what the finding means for building our EMI prediction model.

---

## Chart 1 — Target Variable Analysis (3 panels)

**File:** `eda_01_target_analysis.png`

### Panel 1: EMI Eligibility Donut
**What it shows:** How many of our 387,287 applicants are eligible vs not eligible for EMI.

**How to read it:**
- The ring is split into two wedges: orange (Not Eligible) and teal (Eligible)
- The % labels inside the ring (80.8% / 19.2%) are the proportions
- The number in the center hole is the total applicant count

**What it means for the model:**
This is our most important finding before training anything. Out of every 5 applications, only 1 is eligible. This 4.2:1 imbalance means if our model just guessed "Not Eligible" for everyone, it would be 80.8% accurate — but completely useless. This is why we use `class_weight='balanced'` and measure ROC-AUC + F1, not accuracy.

---

### Panel 2: Max Monthly EMI Distribution (Histogram)
**What it shows:** The range and shape of the regression target — what EMI amounts are actually possible.

**How to read it:**
- X-axis: EMI amount in rupees
- Y-axis: How many applicants have that EMI capacity
- Orange dashed line = mean (₹6,461)
- Teal dashed line = median

**What it means for the model:**
The distribution is roughly bell-shaped but with a wide range (₹500–₹34,750). The mean ≈ median tells us it's not heavily skewed — good for regression. Section 5 will train a regression model to predict this value.

---

### Panel 3: Max EMI by Eligibility (KDE Density)
**What it shows:** Whether eligible and non-eligible applicants tend to have different EMI capacities.

**How to read it:**
- Two overlapping density curves (orange = Not Eligible, teal = Eligible)
- Where the curves separate = the feature has predictive power
- Where they overlap = the feature doesn't help discriminate

**What it means for the model:**
If the curves are separated, `max_monthly_emi` is useful for classification too. A visible gap between the means (shown by dotted vertical lines) confirms predictive signal.

---

## Chart 2 — EMI Scenario Analysis (4 panels)

**File:** `eda_02_scenario_analysis.png`

**What it shows:** Whether the type of EMI (Vehicle, Home Appliances, Education, etc.) affects approval rates.

### Panel 1: Applications per Scenario
Horizontal bar chart counting how many applications came in for each loan type. Tells you which loan types are most common.

### Panel 2: Approval Rate by Scenario
**Key panel.** The dashed vertical line at 19.2% is the overall average. Bars to the right = that loan type gets approved more than average; bars to the left = below average.

**Interpretation rule:** If all bars were at 19.2%, the loan type wouldn't matter. Bars that deviate significantly tell you the scenario is a predictor of eligibility.

### Panel 3: Average Max EMI per Scenario
Which loan types are associated with higher or lower EMI amounts? A "Vehicle EMI" applicant vs "Home Appliances EMI" applicant — do they have different financial profiles?

### Panel 4: Summary Table
Numeric reference for the 3 charts above. Use this to confirm exact values you see visually.

---

## Chart 3 — Key Financial Features vs EMI Eligibility

**File:** `eda_03_features_vs_target.png`

**What it shows:** For the 5 most important numerical features, how their distributions differ between eligible and non-eligible applicants.

### How to read each KDE panel:
- **Two curves** per panel: orange (Not Eligible, solid line) and teal (Eligible, dashed line)
- **Where curves separate** = the feature has discriminative power
- **Vertical dotted lines** = means for each class
- **p-value** in the corner: if p < 0.05, the separation is statistically real (not random chance)

### Feature-by-feature interpretation:

**Credit Score:** This is our top predictor. If the teal curve (Eligible) is shifted right (higher scores), it confirms that higher credit scores lead to more approvals. The gap between means directly tells you the credit score "threshold" for eligibility.

**Monthly Salary:** Higher salary → more likely eligible. Look at how far the teal mean sits above the orange mean. The wider the gap, the stronger the salary signal.

**Bank Balance:** A proxy for financial stability. Eligible applicants typically maintain higher balances.

**Age:** Check if age has any eligibility pattern. A U-shape or uniform distribution means age alone isn't a strong predictor.

**Years Employed:** Employment stability signal. New employees (0 years) are high-risk.

### Panel 6: Existing Loans vs Eligibility (Stacked Bar)
**How to read:** Each row is a category (Yes/No existing loans). The bar shows what % of that category was approved vs rejected. White text inside bars = the percentage.

**Interpretation:** If "No Loans" applicants have a higher approval rate than "Yes Loans", existing debt burden is penalizing new EMI requests.

---

## Chart 4 — Correlation Analysis

**File:** `eda_04_correlation.png`

### Left Panel: Correlation Matrix Heatmap
**What it shows:** How every feature relates to every other feature. The value in each cell is the Pearson correlation coefficient (r), ranging from -1 to +1.

**How to read the colors (coolwarm palette):**
| Color | Value | Meaning |
|-------|-------|---------|
| Deep Red | r close to +1 | Strong positive correlation — both features rise together |
| Light Pink/White | r near 0 | No linear relationship |
| Deep Blue | r close to -1 | Strong negative correlation — one rises, other falls |

**How to read the numbers:**
- r = +0.85 → very strong positive relationship
- r = +0.30 → weak positive relationship
- r = -0.60 → moderate negative relationship

**Why this matters for ML:**
- High correlation between TWO PREDICTORS (e.g., salary and bank_balance both = +0.7) = **multicollinearity** — could confuse linear models like Logistic Regression
- High correlation between a PREDICTOR and the TARGET (`emi_eligibility`) = that feature is important

**The lower triangle only:** The matrix is symmetric (correlation of A with B = B with A), so we only show the lower half to avoid redundancy.

---

### Right Panel: Features vs EMI Eligibility (Bar Chart)
**What it shows:** How strongly each feature is correlated with the target variable.

**How to read it:**
- **Teal (positive) bar** = higher value of this feature → more likely to be eligible
- **Orange (negative) bar** = higher value of this feature → less likely to be eligible
- **Longer bar** = stronger relationship

**Example reading:**
- If `credit_score` has the longest teal bar → credit score is the strongest positive predictor (higher score → more eligible) — confirms Chart 3 finding
- If `total_expenses` has a long orange bar → higher expenses → less likely eligible (logical: can't afford EMI if expenses are already high)

---

## Chart 5 — Categorical Features vs EMI Eligibility

**File:** `eda_05_categorical_analysis.png`

**What it shows:** For each categorical feature (gender, education, employment type etc.), which categories have higher vs lower approval rates.

**How to read each bar:**
- Each row = one category value (e.g., "Salaried", "Self-Employed", "Freelancer")
- The bar is split: orange portion = % Not Eligible, teal portion = % Eligible
- All bars add to 100%
- White text inside = the percentage for each segment

**How to find important patterns:**
1. If all rows have roughly the same split (all ~80/20) → this feature doesn't affect eligibility
2. If some rows have teal > 19.2% (our baseline) → those categories get approved more than average
3. The rows are sorted by Eligible % ascending → the most approved category is always at the top

**Example interpretation:**
- Education: If "Post-Graduate" has significantly more teal than "High School" → education level predicts eligibility
- Employment Type: If "Government" employees show higher approval than "Freelance" → employment stability matters

---

## Chart 6 — Financial Capacity Analysis

**File:** `eda_06_financial_analysis.png`

### Panel 1: Monthly Salary vs Max EMI (Scatter)
**What it shows:** The relationship between what someone earns and what EMI they can afford.

**How to read it:**
- Each dot = one applicant
- X-axis = monthly salary, Y-axis = max EMI they can afford
- Orange dots = Not Eligible, Teal dots = Eligible

**Pattern to look for:** A diagonal band rising left-to-right means "higher salary → higher EMI capacity" (expected). If teal dots cluster in the upper-right, higher earners are more eligible.

---

### Panel 2: Approval Rate by Salary Bracket
**What it shows:** Does EMI approval jump at certain salary levels?

**How to read it:**
- Each bar = one salary range (0-20k, 20-40k, etc.)
- Bar height = % of that bracket that got approved
- Dashed line = 19.2% overall average

**Pattern to look for:** A staircase pattern (each bracket higher than the previous) confirms salary is monotonically predictive. A flat pattern means salary bracket alone doesn't differentiate eligibility.

---

### Panel 3: Expense-to-Salary Ratio by Eligibility (KDE)
**What it shows:** The "expense ratio" = total monthly expenses ÷ monthly salary. This is our engineered signal.

**How to read it:**
- Ratio = 0.0 → person spends nothing (theoretical)
- Ratio = 0.5 → spends 50% of salary on existing expenses
- Ratio = 1.0 → spends 100% — no room for new EMI
- Ratio > 1.0 → spends more than they earn (danger zone)

**Pattern to look for:** If orange (Not Eligible) peaks at a higher ratio than teal (Eligible), expense ratio is a strong predictor — someone already spending 80% of income can't afford a new EMI.

The dotted vertical line at 0.5 is a potential decision threshold: people below it have more financial headroom.

---

### Panel 4: Credit Score vs Max EMI (Scatter)
**What it shows:** Do applicants with higher credit scores qualify for higher EMI amounts, or is it independent?

**Pattern:** If teal dots (Eligible) cluster at high credit scores AND high EMI amounts → credit score and EMI capacity are jointly important.

---

## Chart 7 — Numerical Feature Distributions

**File:** `eda_07_distributions.png`

**What it shows:** The distribution (shape) of each numerical feature after cleaning.

### How to read each histogram:
- X-axis = the feature value
- Y-axis = how many applicants have that value
- Red dashed = mean, dark dashed = median
- `skew=` in the title tells you the distribution shape

### Skewness guide:
| Skew value | Shape | Implication |
|-----------|-------|-------------|
| Near 0 | Symmetric bell curve | Normal-ish, most models handle well |
| +1 to +3 | Right-skewed (long tail right) | Many low values, few very high ones |
| > +3 | Heavily right-skewed | Consider log-transformation in Section 4 |
| Negative | Left-skewed | Rare in financial data |

### What the annotation boxes mean:
Features with "% = 0" boxes (emergency_fund, years_of_employment, monthly_rent) are **zero-heavy** — a large proportion of applicants have exactly zero for that feature. This is NOT a data error. These are legitimate zero values (e.g., 0 rent = owns home). Our model needs to handle this correctly — do NOT impute or remove these zeros.

### Key things to check per feature:
- **Age:** The 4-spike pattern is real institutional data — do not smooth
- **Credit Score:** Distribution shape tells us if low-credit applicants dominate
- **Monthly Salary:** Heavy right skew = a few very high earners pull the mean up

---

## Chart 8 — Class Imbalance Analysis

**File:** `eda_08_class_imbalance.png`

### Panel 1: Class Distribution (Bar)
**What it shows:** The visual proof of 4.2:1 imbalance. Not Eligible (orange) dominates.

**Why this matters:** Any ML model trained without correcting for this imbalance will be biased toward predicting "Not Eligible" because that's the easy path to high accuracy. A dumb model that always says "Not Eligible" would be 80.8% accurate — and would fail on every actual eligible applicant.

---

### Panel 2: Why Accuracy Alone is Misleading
**What it shows:** A side-by-side comparison of what a "naive" model (always predicts Not Eligible) achieves vs what we need our ML model to achieve.

**How to read it:**
- Orange bars = Naive model (predicts "Not Eligible" for everyone)
- Teal bars = Our ML target scores

| Metric | Naive Model | Our ML Target | Why |
|--------|-------------|---------------|-----|
| Accuracy | 80.8% | 85% | Naive looks good — this is the trap |
| F1-Score | 0% | 75% | Naive catches zero eligible applicants |
| Precision | 0% | 80% | Of predicted eligible, how many really are |
| Recall | 0% | 70% | Of all eligible, how many we correctly found |

**The key insight:** F1 is the harmonic mean of Precision and Recall. A model that doesn't identify any eligible applicants has F1 = 0%, no matter how high the accuracy. This is why **ROC-AUC and F1 are our primary metrics** — they penalize the model for ignoring the minority class.

**Business interpretation of Recall vs Precision:**
- High Recall = fewer eligible applicants wrongly rejected (revenue impact)
- High Precision = fewer ineligible applicants wrongly approved (risk/default impact)
- In Indian lending context, a false approval (lending to an ineligible person) is more costly than a false rejection → Precision matters slightly more

---

## Summary: What These 8 Charts Tell Us as a Team

| Finding | Chart | Model Implication |
|---------|-------|------------------|
| 4.2:1 class imbalance | 1, 8 | Use `class_weight='balanced'`, measure ROC-AUC + F1 |
| Credit score = top predictor | 3, 4 | Will be highest importance in any tree model |
| Expense ratio = strong signal | 6 | Engineer this feature explicitly in Section 4 |
| Salary + bank balance correlated | 4 | Watch for multicollinearity in Logistic Regression |
| Zero-heavy features | 7 | Do NOT impute — annotate as legitimate zeros |
| Loan type affects approval rate | 2 | Include `emi_scenario` as a feature |
| Employment stability matters | 3 | `years_of_employment` is a useful signal despite zeros |

---

*Generated for: EMI Predict AI — Section 2 EDA*
*Date: 2026-05-02*
