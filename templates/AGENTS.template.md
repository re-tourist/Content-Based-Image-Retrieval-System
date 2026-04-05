# AGENTS.md

## Purpose

This file is the repository-level operating manual for AI coding agents.
It defines durable working rules for normal development work in this repository.
It is not a product README and not a task-specific prompt.

Agents should use this file together with:

- `docs/ai/PROJECT_CONTEXT.md`
- `docs/ai/WORKFLOW_GUIDE.md`
- `docs/plan/*`
- `docs/contracts/*`
- `docs/review/*`
- any nested `AGENTS.md` closer to the working directory

---

## Default Working Order

Follow this order unless a task explicitly requires otherwise:

1. read the relevant docs and entry files first
2. confirm the scope and affected files
3. make the smallest reasonable change
4. run the narrowest useful validation first
5. update docs/config/tests if behavior changed
6. summarize what changed, what was verified, and what remains uncertain

For multi-file, architecture-affecting, or ambiguous tasks:

- plan first
- then implement
- then review

---

## Repository Expectations

Always try to:

- preserve existing structure unless a task explicitly allows refactoring
- prefer small, reviewable diffs
- reuse existing patterns before introducing new abstractions
- keep behavior, config, and docs in sync
- report uncertainty instead of silently guessing

Do not:

- rewrite large areas without clear justification
- introduce unrelated cleanup into feature work
- silently change public behavior, contracts, or file layout
- assume missing project facts are true

## Response and Execution Preferences

- Default replies should be in Chinese unless the user explicitly asks for another language.
- Large-compute tasks should be migrated to the Linux server rather than run locally in this workspace.
- Before executing commands that materially advance the task, write them into `docs/run_order.md`.

---

## Rule Layer

- milestone spec first
- issue plan draft second
- issue body third
- implementation after approval

Use the canonical terms consistently:

- `milestone_id`
- `issue_id`
- `issue_type`
- `module`
- `depends_on`
- `priority`

Do not rename the same concept in different documents.

Command logging rule:

- create or update `docs/run_order.md` before running command groups that materially advance the task
- write a short comment line before each command group
- include an ISO 8601 timestamp for each command group
- keep one command per line under the comment so the execution order is easy to audit

---

## GitHub and Commit Conventions

Use these files as the source of truth:

- `docs/plan/plan_stage.template.md` for milestone planning
- `docs/plan/issue_stage.template.md` for issue plan drafts and issue bodies
- `docs/handoff/milestone_closeout.template.md` for milestone closeout
- `templates/PR_TEMPLATE.md` for PR descriptions
- `templates/issue_report_template.md` for issue execution reports
- `templates/milestone_review_report.template.md` for milestone review reports
- `docs/contracts/commit_message_format.md` for commit messages

Commit messages for substantive work should follow the repository contract:

- one short subject line
- a `why:` body explaining the rationale
- a `what:` body explaining the concrete changes
- use a stable scope such as stage, module, or milestone in the subject

If GitHub auth is missing or unavailable, stop and report that the remote action cannot be completed rather than pretending it was created.

---

## Preferred Validation Strategy

Prefer the smallest meaningful validation first:

1. file-level or module-level checks
2. focused unit tests
3. local integration checks
4. full test suite only when needed

If a requested change affects:

- public behavior
- config semantics
- data flow
- training / inference / evaluation outputs

then validation scope must expand accordingly.

---

## Done Definition

A task is not complete unless all of the following are true:

- the requested scope is implemented
- the changed code is internally consistent
- the most relevant validation was run, or the gap is explicitly reported
- related docs/config/tests were updated if needed
- the final report clearly states:
  - files changed
  - why they changed
  - what was verified
  - what remains uncertain or unverified

---

## Documentation Sync Rules

When behavior changes, consider whether these must be updated:

- `README.md`
- `docs/ai/`
- `docs/plan/`
- `docs/review/`
- `docs/contracts/`
- examples, scripts, config comments, or usage snippets

Do not assume code-only changes are sufficient.

---

## Reporting Format

When finishing a task, report in this structure:

1. Files changed
2. What changed in each file
3. Validation run
4. Risks / uncertainties
5. Suggested next step

Keep the report concise and factual.

---

## If the task is ambiguous

Do not invent hidden assumptions.
Instead:

- state the ambiguity
- propose the most conservative interpretation
- proceed only within clearly safe bounds
- stop and ask when the ambiguity affects architecture, evaluation, safety, or external behavior

---

## If the repository grows more complex

Move specialized rules out of this file and into:

- nested `AGENTS.md`
- `docs/review/code_review.md`
- `docs/contracts/*`
- reusable Skills under `.agents/skills/`

Keep this root file compact and durable.
