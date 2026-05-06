---
name: run-tests
argument-hint: [section-number or all]
---

Run test suite for $ARGUMENTS:

1. Run fast unit tests first:
   pytest tests/unit/ -v --tb=short --timeout=30
2. If unit tests pass, run integration tests:
   pytest tests/integration/ -v --tb=short --timeout=120
3. Run coverage report:
   pytest tests/ --cov=src --cov-report=term-missing -q
4. Check coverage thresholds:
   - src/models/    → must be > 80%
   - src/data/      → must be > 90%
   - src/api/       → must be > 85%
5. If any test fails → report:
   - Which file
   - Which test function
   - Exact error message
   - Likely cause
   Do NOT auto-fix. Report and wait for confirmation.
6. Print final summary: X passed | Y failed | Z% coverage