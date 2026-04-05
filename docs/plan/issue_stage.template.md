# Issue Plan and Issue Body Template

This file is a two-step template.
First produce an issue plan draft.
Then expand the approved plan into an issue body.
Do not skip the issue plan step.

## 1. Issue Plan Draft

Use this lightweight draft for human review of structure and dependencies.

- derived_from:
- issue_id:
- title:
- issue_type:
- module:
- goal:
- depends_on:
- priority: P0 | P1 | P2

Optional planning fields:

- reason_for_split:
- risk:
- expected_output:

Rules:

- Keep the draft short.
- Keep the draft aligned with the milestone spec.
- The draft is not the final issue body.
- The draft must be reviewed before expansion.

## 2. Issue Expansion Rules

Use the approved issue plan draft to generate the issue body below.
Keep the same canonical terms across milestone spec, issue plan, issue body, PR body, and review report.

### derived_from
- derived_from:

### issue_type
- issue_type:

### module
- module:

### depends_on
- depends_on:

### Background
- background:

### Goal
- goal:

### In Scope
- in_scope:

### Out of Scope
- out_of_scope:

### Inputs
- inputs:

### Deliverables
- deliverables:

### Constraints
- constraints:

### Acceptance
- acceptance:

### Suggested Branch
- suggested_branch:

### Validation
- validation:

### Report Format
- report_format:

## 3. Expansion Rules

- Use the milestone spec as the source of truth.
- Expand only after the issue plan draft is approved.
- Preserve issue_type, module, depends_on, and priority exactly.
- Keep one primary module per issue unless the milestone spec explicitly allows cross-module work.
- Put design / freeze issues before implementation issues.
- Put validation / docs / closeout issues after implementation issues.
