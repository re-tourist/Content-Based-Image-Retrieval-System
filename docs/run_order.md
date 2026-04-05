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
