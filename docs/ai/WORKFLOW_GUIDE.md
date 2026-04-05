# WORKFLOW_GUIDE

## Purpose

This document explains how to use the repository as a real workflow.
It is not a design document.
It is an operational guide for humans and agents.

The goal is to turn a single milestone spec into a stable chain:

Milestone Spec -> Issue Plan Draft -> Issue Document -> Implementation -> PR -> Review -> Closeout

---

## Source of Truth

Use the repository as the source of truth.
Do not rely on:

- long chat history
- repeated manual prompt rewriting
- human copy-paste of project context

Instead, keep durable context in:

- `AGENTS.md`
- `docs/ai/PROJECT_CONTEXT.md`
- `docs/ai/WORKFLOW_GUIDE.md`
- `docs/plan/plan_stage.template.md`
- `docs/plan/issue_stage.template.md`
- `docs/contracts/commit_message_format.md`
- `docs/contracts/contract_freeze.template.md`
- `docs/review/code_review.template.md`
- `docs/handoff/milestone_closeout.template.md`
- `templates/PR_TEMPLATE.md`
- `templates/issue_report_template.md`
- `templates/milestone_review_report.template.md`
- `.github/ISSUE_TEMPLATE.md`
- `.github/pull_request_template.md`

## Command Run Log

Before running command groups that materially advance the task, append them to `docs/run_order.md`.

Logging format:

- write one comment line for the command group
- include an ISO 8601 timestamp for the group
- put one command per line underneath the comment
- start a new group whenever the intent changes

This log is a lightweight execution ledger, not a substitute for validation notes.

---

## Milestone Compilation Workflow

This is the required flow.

1. The author fills out a milestone spec in `docs/plan/plan_stage.template.md`.
2. Codex reads `docs/ai/PROJECT_CONTEXT.md` before doing any decomposition.
3. Codex generates an issue plan draft from the milestone spec.
4. A human reviews the issue plan draft for structure, dependency order, and coverage.
5. Only after approval does Codex expand the approved plan into issue bodies.
6. Implementation starts only after the issue body exists and the relevant contract is clear.
7. PRs, reviews, and closeout notes are produced after implementation.

Hard gates:

- milestone specs must not jump directly to code.
- issue plan draft is a required intermediate artifact.
- design / freeze work must appear before implementation work.
- validation / docs / closeout work must appear after implementation work.
- issue bodies must inherit the milestone spec terms rather than inventing new labels.

---

## Role Split

### Human owns
- project goal
- milestone design
- issue acceptance
- contract freeze
- evaluation meaning
- research / product direction
- final go / no-go decisions

### Implementer owns
- scoped implementation
- issue plan draft generation
- issue body expansion after approval
- narrow validation
- docs / config / test sync
- structured execution report

### Reviewer owns
- scope review
- contract review
- validation review
- semantic risk review
- structured approve / revise / stop recommendation

---

## GitHub Workflow

Use the docs in this repository as the canonical source for GitHub objects.

Recommended mapping:

- milestone spec -> GitHub milestone title and description
- issue plan draft -> review-only intermediate artifact
- issue body template -> GitHub issue body
- PR template -> GitHub pull request body
- closeout template -> milestone handoff / closeout note
- commit contract -> local commit message format

If GitHub credentials are available and the tool surface supports the action:

- create or update the GitHub milestone from the milestone spec
- create issues from the approved issue body template
- open PRs using the PR template
- keep branch names and titles aligned with the stage / issue identifiers

If GitHub credentials are not available, or the exposed tool surface does not include the required action:

- still draft the milestone, issue, and PR bodies locally
- stop and report that remote creation could not be completed

Milestones are not a substitute for planning docs.
They are the remote tracking mirror of the plan.

## Branch Merge Policy (Must Follow)

This repository uses a strict merge flow:

- All feature or topic branches must merge into `dev` first.
- Only `dev` is allowed to merge into `main`.
- Any PR opened from an auxiliary thread should still target `main` by default, but the actual code merge must respect the `dev -> main` path.

When in doubt, follow this sequence:

1. merge feature branch into `dev`
2. merge `dev` into `main`
3. push both branches

---

## Standard Lifecycle

### Phase 0 - Bootstrap

Use this when entering a new repository.

1. create starter kit structure
2. run a repository snapshot
3. generate `docs/snapshots/project_snapshot.template.md` or a project snapshot note
4. generate or rewrite project `AGENTS.md`
5. human fills `docs/ai/PROJECT_CONTEXT.md`

At the end of Phase 0, the repo should be understandable enough for scoped execution.

---

### Phase 1 - Planning

Use this before coding a new milestone.

1. write or update the milestone spec
2. derive the issue plan draft
3. human reviews scope and ordering
4. expand approved issue plan items into issue bodies
5. create or update relevant contract docs if execution is about to begin

At the end of Phase 1, the milestone should be executable in issue-sized chunks.

---

### Phase 2 - Freeze

Use this before a milestone enters active execution.

1. create `docs/contracts/contract_freeze_*.md`
2. lock:
   - milestone objective
   - stable interfaces
   - allowed change window
   - forbidden changes
   - stop conditions
3. human confirms the freeze

At the end of Phase 2, agents should know where execution freedom stops.

---

### Phase 3 - Execute

Use implementer for a single issue or a tightly scoped milestone task.

Recommended order:

1. read `AGENTS.md`
2. read `PROJECT_CONTEXT.md`
3. read the milestone spec and approved issue body
4. plan briefly
5. implement minimally
6. run narrow validation
7. sync docs/tests/config if needed
8. write a structured issue report or PR description

At the end of Phase 3, the task should be reviewable.

---

### Phase 4 - Review

Use reviewer after implementation, before merge or milestone close.

Reviewer should inspect:

- scope drift
- contract violations
- validation adequacy
- docs/config mismatch
- semantic correctness risk

Reviewer should produce:

- approve / revise / stop
- severity classification
- recommended next step

At the end of Phase 4, a human should be able to decide quickly.

---

### Phase 5 - Closeout

Use this when a milestone is done or paused.

1. update `docs/handoff/milestone_closeout.template.md` or a milestone closeout note
2. record:
   - completed work
   - missing work
   - key risks
   - next entry point
3. decide whether the milestone can be closed cleanly

At the end of Phase 5, the next stage should have a clear handoff entry.

---

## Success Criteria

This workflow is working if:

- agents read repo docs instead of asking for repeated context
- the first generated artifact after a milestone spec is an issue plan draft
- implementation reports become structured and predictable
- reviewer catches meaningful issues, not just style trivia
- humans spend less time doing copy-paste orchestration
- milestone transitions become easier to reason about

---

## Failure Signs

The workflow is not yet working well if:

- humans still rewrite long prompts every time
- agents ignore `AGENTS.md` or `PROJECT_CONTEXT.md`
- reviewer only repeats the implementer summary
- scope drift happens frequently
- contract docs exist but do not affect behavior
- docs are present but never referenced during execution

---

## How to Evolve Beyond V0

Only upgrade after repeated real use.

Good candidates for V1+:

- more reusable Skills
- nested `AGENTS.md`
- CI-triggered reviews
- project-type specializations
- richer repo-snapshot automation
- plan / issue drafting Skills
- milestone review automation

Do not expand just because it sounds nice.
Expand only after repeated friction shows what is truly reusable.
