---
name: new-section
argument-hint: [section-number] [section-name]
---

Initialize Section $ARGUMENTS for EMI Predict AI:

1. Read CLAUDE.md — confirm previous section is marked complete
2. Check previous section has: src file + notebook + tests + docs entry
   If anything missing — STOP and report before proceeding
3. Create any new subdirectories needed for this section
4. Create notebook skeleton: notebooks/0N_sectionname.py with:
   - Header comment (filename, purpose, section, version)
   - Setup cell: imports + logging config
   - Load data cell: from data/processed/ only
   - Summary cell placeholder: findings + decisions + next steps
5. Create docs skeleton: docs/section_0N_name.md with section outline
6. Update CLAUDE.md:
   - Move previous section to Completed ✅
   - Set new section as In Progress 🔄
   - List all files to be created this section
7. Git commit: "section-N: initialize name"
8. Print: "Section N initialized. Files to create: [full list]"