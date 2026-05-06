---
name: debug-model
argument-hint: [model-name] [issue-description]
---

Debug ML issue — $ARGUMENTS:

1. Read the most recent file in logs/
2. Read the relevant model file in src/models/
3. Classify the issue type:
   - Poor ROC-AUC (<0.65)  → likely: class_weight missing or feature problem
   - NaN predictions        → likely: preprocessing gap, zero division
   - Memory error           → likely: wrong dtypes, too many features
   - Slow training          → likely: wrong n_jobs, too many CV folds
   - class_weight not set   → CRITICAL — fix this first always
4. First check ALWAYS: is class_weight='balanced' explicitly set?
   If not — this is the fix. Do it before anything else.
5. Check: are ROC-AUC and F1 both being evaluated?
6. Check: is data split BEFORE any preprocessing step?
7. Check: are stratified splits being used?
8. Implement fix with full explanation of root cause
9. Add assertion to catch this class of error earlier
10. Run: pytest tests/ -k "relevant_test_name" -v to verify
11. Append fix summary to docs/debugging_log.md