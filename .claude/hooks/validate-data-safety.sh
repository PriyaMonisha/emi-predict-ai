#!/bin/bash
# filename: .claude/hooks/validate-data-safety.sh
# purpose:  Ensure data code never leaks test data into training
# version:  1.0

echo "🔍 Validating data pipeline safety..."

# ── Check fit_transform not used on test/val data ─────────────
FIT_ON_TEST=$(grep -rn "fit_transform" src/ --include="*.py" | \
    grep -i "test\|val\|valid")
if [ -n "$FIT_ON_TEST" ]; then
    echo "❌ CRITICAL: fit_transform possibly called on test/val data:"
    echo "$FIT_ON_TEST"
    echo "Rule: fit() on train only. transform() on test/val."
    exit 2
fi

# ── Check unlabeled file row count ───────────────────────────
UNLABELED="data/processed/unlabeled_for_prediction.csv"
if [ -f "$UNLABELED" ]; then
    ROWS=$(wc -l < "$UNLABELED")
    # Expect 17,489 lines (17,488 data rows + 1 header)
    if [ "$ROWS" -lt 17000 ] || [ "$ROWS" -gt 18000 ]; then
        echo "⚠️  WARNING: unlabeled_for_prediction.csv has $ROWS lines"
        echo "   Expected ~17,489 (17,488 rows + header)"
        echo "   Was this file modified accidentally?"
    fi
fi

echo "✅ Data safety checks passed"
exit 0