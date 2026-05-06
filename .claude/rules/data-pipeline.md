---
paths:
  - "src/data/**/*.py"
  - "scripts/preprocess*.py"
  - "scripts/load*.py"
---

# Data Pipeline Rules — EMI Predict AI

## The Cardinal Rule
data/raw/ is READ ONLY — always and forever.
Outputs go to data/processed/ only.
If writing to data/raw/, stop immediately — you are wrong.

## Preprocessing Reference
The 17-step pipeline in preprocess.py v4 is locked and complete.
New features go in src/models/feature_engineering.py — NOT in preprocess.py.
If preprocess.py must change: create preprocess_v5.py, never overwrite v4.

## Known Data Facts — Never "Fix" These
- monthly_rent zero-heavy: legitimate (living with parents, owned home)
- years_of_employment zero-heavy: legitimate (students, unemployed)
- Bank balance: filled with salary-bracket median — this was intentional
- Age 4-spike pattern: real data distribution, document but never smooth
- Class ratio 4.2:1: real world ratio — do NOT resample data files

## Split Protocol
- Always stratify on emi_eligibility column
- Test size: 20% — keep consistent across all experiments
- Split BEFORE any preprocessing — no exceptions, ever
- random_state=42 — always
- Verify split ratio after splitting: log actual class distribution

## Unlabeled High-Risk Data
- Location: data/processed/unlabeled_for_prediction.csv
- Rows: 17,488
- These rows get predictions in production — never labels
- Never include in training or evaluation sets
- Use case: Section 9 batch inference demonstration

## Data Loading Standards
- Always validate shape after loading
- Always log: rows, columns, memory usage, class distribution
- Never modify a DataFrame in place without explicit copy or comment
- Use dtype specifications when loading CSV (avoid pandas guessing)