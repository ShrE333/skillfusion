# Project
<one or two lines: what skillfusion does and what "done" means>

# Environment
- Isaac Sim + Isaac Lab run on a remote server. They cannot run in this
  workspace, so never try to launch them here.
- Always use headless mode. Camera/RTX rendering may not work (A100 GPU).

# Workflow
- Work only on branch dev. Do not create other branches. Never push to main.
- Do not add co-author lines or tool names to commit messages.
- Keep tests in tests/run_tests.sh: short smoke tests (under 10 minutes)
  that print clear PASS or FAIL lines.
- After pushing, run `git fetch origin results` and read results/summary.md.
  Wait until its "commit:" line matches your latest commit SHA (poll every
  60 seconds, up to 20 minutes). Then read results/run_tail.log, fix any
  failures, and push again.
- Commit small and often.
