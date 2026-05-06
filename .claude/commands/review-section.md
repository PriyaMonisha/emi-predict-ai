---
name: review-section
argument-hint: [section-number]
---

Full quality review of Section $ARGUMENTS before marking complete:

1. Read CLAUDE.md — get expected file list for section $ARGUMENTS
2. For each source file:
   a. File exists in src/
   b. Header comment present: filename + purpose + version
   c. All functions have docstrings: what + args + returns + why
   d. No print() statements: grep -n "print(" file
   e. No hardcoded paths: grep -n '"data/' file
   f. Type hints on all function signatures
   g. No bare except: clauses
3. Run tests: pytest tests/ -v -k "section$ARGUMENTS" --tb=short
4. Check notebook exists in notebooks/
5. Check docs entry exists in docs/
6. ML checks (sections 3, 4, 5, 6 only):
   - class_weight='balanced' present on all classifiers
   - MLflow logging present in training code
   - ROC-AUC + F1 reported (not accuracy alone)
   - No fit_transform called on test/val data
   - random_state=42 on all models and splits
7. Report: ✅ PASS or ❌ FAIL for every single check
8. If all PASS → "Section $ARGUMENTS ready. Run /checkpoint to mark complete."
9. If any FAIL → "Fix these before proceeding: [exact list]"