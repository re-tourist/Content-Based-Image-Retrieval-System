# run_order

This file is the command execution ledger for the repository.

## Format

- Write one command group per section.
- Start each group with a comment line that explains why the commands are being run.
- Include an ISO 8601 timestamp in the comment line.
- Put one command per line under the comment.
- Add a blank line between command groups.

Before running command groups that materially advance the task, append them here first.

## Example

# [2026-04-04T10:00:00+08:00] inspect workflow docs
Get-Content AGENTS.md
Get-Content docs\ai\WORKFLOW_GUIDE.md

# [2026-04-04T10:05:00+08:00] verify milestone templates
Get-Content docs\plan\plan_stage.template.md
Get-Content docs\plan\issue_stage.template.md

# [2026-04-04T19:31:28+08:00] verify merged workflow docs and template coverage
git diff --check
git status --short
Get-ChildItem docs\contracts -File | Select-Object Name,Length | Sort-Object Name
Get-ChildItem docs\handoff -File | Select-Object Name,Length | Sort-Object Name
Get-ChildItem docs\plan -File | Select-Object Name,Length | Sort-Object Name
Get-ChildItem docs\review -File | Select-Object Name,Length | Sort-Object Name
Get-ChildItem docs\snapshots -File | Select-Object Name,Length | Sort-Object Name

# [2026-04-04T19:32:51+08:00] verify project snapshot file after adding the concrete snapshot
git diff --check
Get-ChildItem docs\snapshots -File | Select-Object Name,Length | Sort-Object Name

# [2026-04-04T19:33:15+08:00] verify snapshot link addition in project context
git diff --check

# [2026-04-04T19:40:00+08:00] remove codex-starter-kit after merging template content
Resolve-Path codex-starter-kit
Remove-Item -LiteralPath codex-starter-kit -Recurse -Force
git status --short

# [2026-04-05T21:16:51+08:00] commit and push milestone 5 starter-kit merge
git commit -m "docs(workflow | milestone5): merge starter-kit templates and closure records" -m "why:" -m "- align the repository with the starter-kit workflow layer now that GitHub tooling and milestone 5 closure work are available, and make the repository self-describing for future agent runs." -m "what:" -m "- add the missing workflow, contract, review, handoff, plan, snapshot, prompt, and Codex config docs; finalize the milestone 5 project context; and retain the canonical retrieval evaluation artifacts and scripts under version control."
git push

# [2026-04-05T21:16:51+08:00] create and close GitHub milestone 5 records
gh api -X POST repos/re-tourist/Content-Based-Image-Retrieval-System/milestones -f title='Milestone 5: TF-IDF / Inverted Index / Canonical Retrieval Evaluation Closure' -f description='Closes the sparse-retrieval line: TF-IDF statistics, method-specific inverted indexes, canonical gallery/query evaluation, and P@k / R@k / AP / mAP / PR data export.' --jq '.number'
gh issue create --repo re-tourist/Content-Based-Image-Retrieval-System --title 'Milestone 5 closeout: canonical retrieval evaluation closure' --milestone 'Milestone 5: TF-IDF / Inverted Index / Canonical Retrieval Evaluation Closure' --body 'Milestone 5 closes the traditional sparse-retrieval and canonical evaluation closure for the image retrieval project.'
gh issue close 35 --repo re-tourist/Content-Based-Image-Retrieval-System
gh api -X PATCH repos/re-tourist/Content-Based-Image-Retrieval-System/milestones/6 -f state=closed
gh pr edit 34 --repo re-tourist/Content-Based-Image-Retrieval-System --body '...updated milestone 5 summary and validation...'
gh pr ready 34 --repo re-tourist/Content-Based-Image-Retrieval-System

# [2026-04-05T21:17:08+08:00] commit the run_order audit of milestone 5 GitHub completion
git add docs/run_order.md
git commit -m "docs(run_order | audit): record milestone 5 GitHub completion" -m "why:" -m "- keep the repository execution ledger aligned with the milestone 5 GitHub closeout actions so the audit trail remains reproducible in-tree." -m "what:" -m "- add the commit, push, milestone, issue, milestone-close, PR-update, and PR-ready actions to docs/run_order.md." 
git push

# [2026-04-05T21:25:08+08:00] milestone 4 closeout audit and GitHub maintenance
gh api -X GET repos/re-tourist/Content-Based-Image-Retrieval-System/milestones --paginate
gh issue list --repo re-tourist/Content-Based-Image-Retrieval-System --milestone "Milestone 4: Feature Encoding and Codebook Wiring" --state all
gh issue list --repo re-tourist/Content-Based-Image-Retrieval-System --search "Issue 4" --state all

# [2026-04-05T21:25:51+08:00] close GitHub milestone for M4 feature encoding
gh api -X PATCH repos/re-tourist/Content-Based-Image-Retrieval-System/milestones/5 -f title='Milestone 4: Feature Encoding and Codebook Wiring' -f description='Closes Milestone 4 encoding foundation, codebook training, BoW encoding, encoded artifact persistence, and optional pipeline hook.' -f state=closed

# [2026-04-05T21:26:21+08:00] inspect PRs for milestone 4 closeout record
gh pr list --repo re-tourist/Content-Based-Image-Retrieval-System --search "stage4" --state all
gh pr list --repo re-tourist/Content-Based-Image-Retrieval-System --search "encoding" --state all

# [2026-04-05T21:32:10+08:00] update merge policy docs, ignore coursework, and merge branches
Get-Content docs\ai\WORKFLOW_GUIDE.md -TotalCount 220
Get-Content docs\ai\prompt\stage4\prompt_stage4-4.md -TotalCount 20
Get-Content .gitignore
git status --short
git diff -- docs/ai/prompt/stage4/prompt_stage4-4.md
git branch --show-current
git branch --all
git add docs/ai/WORKFLOW_GUIDE.md .gitignore docs/run_order.md docs/handoff/milestone_closeout_stage4.md docs/ai/prompt/stage4/prompt_stage4-4.md
git commit -m "docs(merge | policy): enforce dev->main flow and ignore coursework artifacts" -m "why:" -m "- align the repo workflow with the required merge policy and prevent coursework artifacts from leaking into version control." -m "what:" -m "- document the merge rule, add coursework/ to .gitignore, and record milestone 4 closeout artifacts." 
git checkout dev
git merge feat/scripts
git checkout main
git merge dev
git push origin dev
git push origin main

# [2026-04-05T21:34:55+08:00] sync dev with origin and retry push after rejection
git checkout dev
git pull
git push origin dev
git checkout main
git push origin main
