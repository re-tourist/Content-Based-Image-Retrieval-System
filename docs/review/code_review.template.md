# code_review

## Purpose

This document defines the project-level review expectations.
It is not only about style.
Its primary purpose is to catch:

- scope drift
- contract violations
- missing validation
- doc/config mismatch
- misleading changes that look correct but are semantically wrong

---

## 1. Scope Review

Check whether the change stays within the intended scope.

Questions:

- Does the diff solve the requested issue or milestone objective?
- Did it introduce unrelated cleanup or opportunistic refactors?
- Did it touch files or modules that were not justified?
- Did it silently expand the milestone scope?
- Does it respect the approved issue plan draft?

Mark:

- [ ] in scope
- [ ] partly out of scope
- [ ] clearly out of scope

---

## 2. Milestone Spec and Issue Plan Review

Check whether the change still matches the upstream milestone spec and approved issue plan.

Questions:

- Does the issue body inherit the milestone terms verbatim where possible?
- Did the implementation follow the approved issue_type and module boundary?
- Did the dependency order remain intact?
- Did the change preserve the issue plan's priority ordering?
- Did the work stay within `in_scope` and `must_not_change` constraints?

Mark:

- [ ] spec and issue plan aligned
- [ ] possible mismatch
- [ ] clear mismatch

---

## 3. Contract / Interface Review

Check whether the change violates any frozen boundary.

Questions:

- Did it change public entrypoints?
- Did it change config semantics?
- Did it alter stable data flow or expected outputs?
- Did it break compatibility with documented behavior?
- Did it violate anything listed in `docs/contracts/`?

Mark:

- [ ] no contract issue found
- [ ] possible contract issue
- [ ] clear contract violation

---

## 4. Validation Review

Check whether the reported validation is adequate.

Questions:

- Were the most relevant checks actually run?
- Was validation proportional to the change impact?
- Were results reported honestly and specifically?
- Is there any unsupported claim like "works" without evidence?

Mark:

- [ ] validation adequate
- [ ] validation incomplete but acceptable
- [ ] validation insufficient

---

## 5. Documentation Sync Review

Check whether repository docs remain aligned with behavior.

Questions:

- Should `README.md` have changed?
- Should any file under `docs/` have changed?
- Did config comments / examples become stale?
- Did usage snippets, scripts, or instructions diverge from reality?

Mark:

- [ ] docs in sync
- [ ] minor doc sync missing
- [ ] significant doc sync missing

---

## 6. Semantic Correctness Review

This is the most important section for research / workflow-heavy repos.

Questions:

- Could this change pass tests but still be semantically wrong?
- Did it optimize the wrong target?
- Did it hide a deeper issue by making outputs merely look better?
- Does the implementation still reflect the intended meaning of the task?
- Is there any mismatch between metric improvement and actual goal satisfaction?

Mark:

- [ ] semantics look aligned
- [ ] semantics uncertain
- [ ] likely semantic mismatch

---

## 7. Risk Summary

Classify review findings:

### P0
Must stop. Merge should not proceed.

Examples:
- false validation claim
- clear contract violation
- dangerous infra / secret change
- result meaning is broken

### P1
Serious issue. Fix before merge.

Examples:
- missing validation
- significant doc mismatch
- out-of-scope architectural change
- risky dependency introduction

### P2
Improvement recommended but not blocking.

Examples:
- report clarity
- missing narrow tests
- over-complex implementation where simpler would suffice

---

## 8. Review Output Format

Review reports should follow this structure:

1. Overall verdict
   - approve / revise / stop
2. Scope status
3. Milestone spec and issue plan status
4. Contract status
5. Validation status
6. Documentation sync status
7. Semantic risk summary
8. Issue list by severity
9. Recommended next action

Keep the review concise and evidence-based.

---

## 9. Merge Guidance

Approve only if all of the following are true:

- the change is in scope
- no P0 / P1 issue remains
- validation is adequate for impact level
- docs/config are not misleading
- semantic intent still matches project goal

If unsure, prefer:

- "revise" over "approve"
- "stop and report" over silent acceptance
