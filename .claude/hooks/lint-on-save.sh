#!/bin/bash
# filename: .claude/hooks/lint-on-save.sh
# purpose:  Auto-format Python files after Claude edits them
# version:  1.0

FILE="$1"

# Only process Python files outside notebooks/
if [[ "$FILE" == *.py ]] && [[ "$FILE" != notebooks/* ]]; then
    echo "🔧 Auto-formatting: $FILE"
    black "$FILE" --quiet
    ruff check "$FILE" --fix --quiet
    echo "   ✅ Done: $FILE"
fi