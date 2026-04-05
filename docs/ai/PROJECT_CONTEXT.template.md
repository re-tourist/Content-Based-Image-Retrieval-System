# PROJECT_CONTEXT Template

Use this file as the starting scaffold for `docs/ai/PROJECT_CONTEXT.md`.
Do not treat it as a free-form notes file.
The same section names should be preserved so the rule layer stays stable.

## 1. Project Summary

- project_name:
- project_type: tooling / infra | application | library | research | coursework
- description:
- primary_stack:

## 2. Current Phase

- current_milestone:
- why_this_phase_exists:
- what_this_phase_should_prove:
- out_of_scope:

## 3. Supported Milestone Types

- freeze
- feature
- research
- refactor
- infra
- evaluation
- docs

Rules:

- type must be one of the supported values above.
- type selects the default issue skeleton used for decomposition.
- type is a policy signal, not a free-text label.

## 4. Default Skeletons by Milestone Type

### feature
- design / contract
- dataset / input
- model / core logic
- train / integration
- eval / output contract
- docs / validation / closeout

### research
- protocol freeze
- baseline reproduction
- method insertion
- experiment execution
- result analysis
- docs / closeout

### refactor
- boundary freeze
- interface migration
- internal refactor
- regression validation
- docs / closeout

### freeze
- contract freeze
- allowed-change definition
- validation gate
- closeout / handoff

### infra
- environment setup
- tooling update
- integration validation
- docs / closeout

### evaluation
- metric contract
- baseline comparison
- run execution
- analysis / report
- docs / closeout

### docs
- source review
- structure / template update
- example / guidance
- validation / consistency check
- closeout

## 5. Module Map

- data
- dataset
- model
- train
- eval
- infra
- docs
- experiment

## 6. Task Decomposition Policy

- milestone spec is the only upstream source for issue derivation.
- the first generated artifact must be an issue plan draft.
- issue plan drafts are for human review of structure, dependency, and coverage.
- each issue must have a single primary responsibility.
- design / freeze issues must come before implementation issues.
- validation / docs / closeout issues must come after implementation issues.

## 7. Priority Policy

- P0: blocking, correctness or contract failure, stop and escalate.
- P1: serious, must fix before milestone close.
- P2: non-blocking follow-up, can be scheduled later.

## 8. Issue Derivation Policy

- choose the default skeleton from milestone type.
- prune the skeleton by `in_scope` and `out_of_scope`.
- apply `must_keep`, `must_change`, and `must_not_change` as compatibility constraints.
- map `required_outputs` into output-oriented issues.
- split by module with one primary module per issue.
- assign dependency edges so design / freeze issues appear before implementation issues.
- emit a reviewable issue plan draft before any issue body is written.

## 9. Canonical Docs Map

- `docs/ai/PROJECT_CONTEXT.md`
- `docs/ai/WORKFLOW_GUIDE.md`
- `docs/plan/plan_stage.template.md`
- `docs/plan/issue_stage.template.md`
- `docs/contracts/commit_message_format.md`
- `.github/ISSUE_TEMPLATE.md`
- `.github/pull_request_template.md`
- `templates/PR_TEMPLATE.md`
- `templates/issue_report_template.md`
- `templates/milestone_review_report.template.md`
- `docs/review/code_review.template.md`
- `docs/handoff/milestone_closeout.template.md`

## 10. Stop Conditions

- dependency or API automation would be required
- the directory structure would need to change
- the milestone spec is missing and cannot be inferred safely
- the issue plan would exceed the allowed max issue count
- a validation step is missing for a P0 or P1 change
