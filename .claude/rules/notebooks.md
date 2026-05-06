---
paths:
  - "notebooks/**/*.py"
  - "notebooks/**/*.ipynb"
---

# Notebook Rules — EMI Predict AI

## Naming Convention (match section numbers exactly)
00_data_audit.py
01_data_cleaning.py
02_eda.py
03_baseline.py
04_feature_engineering.py
05_model_training.py
06_mlflow_experiments.py
07_airflow_pipeline.py
08_redis_feature_store.py
09_api_serving.py
10_monitoring.py

## Required Structure (every notebook)
Cell 1 — Header:
  # filename, purpose, section number, version, date

Cell 2 — Setup:
  imports + logging config + random seeds

Cell 3 — Load Data:
  always from data/processed/ — NEVER from data/raw/

Last Cell — Summary:
  Key findings from this section
  Decisions made and why
  What to watch for in next section

## Visualization Standards (Locked Palettes)
- Binary comparisons:  cmap='Dark2_r'
- Multi-category:      cmap='Paired_r'
- Histograms/dist:     cmap='Accent_r'
- Figure size default: (12, 6)
- Multi-panel figures: (16, 8)
- Always call:         plt.tight_layout() before savefig
- Save figures to:     docs/figures/{section}_{chart_name}.png

## Notebooks Are Not Production
- Notebooks = exploration + documentation only
- All production logic lives in src/
- If you write a reusable function in a notebook → move it to src/utils/
- Notebooks must be runnable top-to-bottom without errors before commit