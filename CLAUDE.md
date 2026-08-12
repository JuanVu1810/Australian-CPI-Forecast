# Claude Code Entry Point

Use this repo's shared AI context before reading broad project files:

1. `.ai/PROJECT_BRIEF.md`
2. `.ai/CODE_ROUTING.md`
3. `.ai/STATUS_RULES.md`

Default workflow:

- Start from the shared context above, then inspect only the files directly
  needed for the task.
- Prefer targeted reads over whole-repo exploration.
- Use `rg` or `rg --files` before opening broad files, generated data, reports,
  or notebooks.
- Do not reread full `README.md` or `PROJECT_ARCHITECTURE.md` unless the task
  needs public-facing wording, architecture rationale, or roadmap detail.
- Avoid `dataset/`, `data/`, `reports/`, notebooks, caches, and `.git/`
  internals unless the task explicitly requires them.
- Preserve the implemented/scaffolded/planned/deployed distinctions from
  `.ai/STATUS_RULES.md`.

Recommended Claude Code role:

- Architecture and documentation review.
- Portfolio wording and README/architecture polish.
- Notebook explanation and modelling narrative review.
- Second-pass review of Codex code changes.

When collaborating with Codex:

- Do not edit the same file at the same time.
- Prefer this cycle: Codex implements and runs tests, Claude reviews the diff,
  Codex applies final fixes.
- If Codex has just changed files, inspect the relevant diff before editing.
