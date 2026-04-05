# Milestone Spec Template

This file is the canonical milestone spec used to derive issue plans.
Do not treat it as free text planning notes.
Every field below is part of the input contract for issue derivation.

## Meta

- milestone_id:
- title:
- type: freeze | feature | research | refactor | infra | evaluation | docs
- status:
- priority: P0 | P1 | P2

## Objective

- objective:

## Scope

### in_scope
- item 1
- item 2
- item 3

### out_of_scope
- item 1
- item 2
- item 3

## Constraints

### must_keep
- item 1
- item 2

### must_change
- item 1
- item 2

### must_not_change
- item 1
- item 2

## Resources

### available_inputs
- item 1
- item 2

### required_outputs
- item 1
- item 2

### non_goals
- item 1
- item 2

## Acceptance

### functional
- item 1
- item 2

### artifact
- item 1
- item 2

### validation
- item 1
- item 2

## Risk

- risk_1:
- risk_2:
- risk_3:

## Decomposition Policy

This section is the machine-readable policy layer used to derive an issue plan draft.

- preferred_split:
- max_issue_count:
- required_issue_types:
- optional_issue_types:
- dependency_rules:

Rules:

- `preferred_split` should describe how to break the milestone into issue-sized slices.
- `max_issue_count` is a hard upper bound unless human approval says otherwise.
- `required_issue_types` are the issue types that must appear if the milestone needs them.
- `optional_issue_types` are allowed but not required.
- `dependency_rules` must describe which issue types must precede others.

## Derivation Notes

- This spec is the only source for issue plan generation.
- Do not write issue bodies directly from this file until the issue plan draft has been reviewed.
- Keep milestone terminology stable so issue bodies can inherit it verbatim.
