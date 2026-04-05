# Commit Message Format Contract

## Purpose

This document defines the repository commit message format for substantive work.

Use this format so that milestones, issue plans, issue bodies, PRs, and commits stay aligned.

---

## Required Format

Use a subject line plus a structured body:

```text
feat(train | stage5-9): run paper-aligned smoke training

why:
- validate that the frozen stage5 paper-path can train end to end under a
- small smoke-only budget before main-run work, and record stability plus
- artifact completeness without confusing smoke with final results

what:
- add a dedicated stage5 smoke config, execute an L=5 smoke training run,
- record checkpoints, history, previews, and a smoke report, and attach
- post-run regular eval and blind eval evidence using the existing stage5
- paths
```

---

## Rules

- Keep the subject line short and specific.
- Put the functional area and milestone or stage in the subject when useful.
- Use `why:` to explain the reason for the change.
- Use `what:` to explain the concrete artifacts or behavior that changed.
- Prefer one logical change per commit.
- Do not use vague placeholders such as "update" or "misc" unless the change is genuinely trivial.
- The subject should be derivable from the approved issue plan, not invented ad hoc.

---

## Recommended Subject Pattern

```text
<type>(<area> | <milestone or stage>): <short imperative summary>
```

Examples:

- `feat(train | stage5-9): run paper-aligned smoke training`
- `fix(data | stage3-2): normalize input path handling`
- `docs(plan | m5): finalize milestone scope and exit criteria`

---

## Scope Guidance

Use these types when appropriate:

- `feat` for new behavior
- `fix` for bug fixes
- `docs` for documentation only
- `test` for test-only changes
- `refactor` for structure-preserving changes
- `chore` for tooling or maintenance work

If a change crosses multiple areas, either split it or make the scope explicit.

---

## Relation to Milestones, Issues, and PRs

Commit messages should stay consistent with:

- the milestone spec
- the approved issue plan draft
- the issue body
- the PR title
- the issue / PR body descriptions

If the commit message cannot be aligned with the current issue or milestone, the task scope is probably too broad.
