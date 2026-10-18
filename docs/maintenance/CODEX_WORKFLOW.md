# Local Codex in VS Code and human-reviewed pull requests

Use Codex interactively in VS Code to investigate an issue, implement a focused
fix and add regression coverage. GitHub Actions runs CPU checks on the resulting
pull request. The contributor publishes the branch and PR; a maintainer decides
whether to merge. This workflow does not run Codex in GitHub Actions, deploy an
image or merge a PR automatically.

## Accounts and authentication

Sign in to the Codex extension with **Sign in with ChatGPT**, using an account
with Codex access. This local workflow does not require OPENAI_API_KEY or an
OpenAI Actions secret. ChatGPT plan limits apply; API-key authentication remains
an optional, separately billed alternative.

Create a GitHub account, verify its email and fork the repository. A fork is your
online copy; a clone is your local working copy. GitHub authentication authorizes
Git/PR operations and is separate from Codex authentication. GitHub CLI is useful
but optional; VS Code's GitHub sign-in or GitHub's website can also be used.

Official authentication reference: https://learn.chatgpt.com/docs/auth

## First local checkout

Clone your fork, open its root folder in VS Code, and keep the original repository
as a fetch remote named `upstream`. Substitute your GitHub username below:

```bash
git clone https://github.com/YOUR_USERNAME/vllm-radiance.git
cd vllm-radiance
git remote add upstream https://github.com/magiccodingman/vllm-radiance.git
git config user.name "Your commit name"
git config user.email "Your verified email or GitHub-provided noreply address"
git switch -c codex/issue-N-short-description
```

Read AGENTS.md, CONTRIBUTING.md and the testing policy before requesting changes.
Keep deployment paths, model locations and credentials out of commits.

## Ask Codex to investigate

Give Codex the issue URL or the observed behavior, reproduction steps and expected
result. For example:

> Read AGENTS.md, CONTRIBUTING.md and docs/maintenance/TESTING_POLICY.md. Investigate
> the supplied issue, identify a concrete root cause, implement the smallest
> justified correction and add a regression test. Preserve qualified ROCm/R9700
> defaults and exact pins. Run the selected CPU tests and report actual results,
> including failures. List the specific hardware gates still required as NOT RUN.
> Use the current topic branch and prepare a draft PR with the issue reference,
> explanation and evidence. Do not merge or deploy. Escalate genuine Tier-3
> decisions using AGENTS.md.

The `codex-fix` issue label can be used as a manual work queue. Applying it does
not start the VS Code extension. Start the task from the Codex chat in your editor.

## Run checks before publishing

Use Python 3.12 or newer. The GitHub job uses Python 3.12; its versions of PyYAML,
pytest and NumPy match the checked-in conformance lockfile. Install these in a
dedicated local environment. Set `PYTHONUTF8=1` on Windows when source decoding
requires it. Registered conformance/d7 tests also require `uv`; their commands use
the checked-in lockfile and fetch CPU dependencies when selected.

```bash
python -m pip install PyYAML==6.0.3 pytest==9.1.1 numpy==2.5.3 uv==0.13.1
git fetch upstream main
python tools/validate_control_plane.py
python tools/radiance.py test changed --base upstream/main --dry-run
python tools/radiance.py test changed --base upstream/main
```

`test changed` includes branch commits, staged/unstaged edits and new non-ignored
files. An explicit base avoids relying on the default `origin/main`, which in a
fork may lag behind the original. The command always includes baseline PR checks.
Run a newly added regression test explicitly if the registry does not select it.

On native Windows, the registry's existing `python-compile` command passes shell
globs to compileall without expansion. Inspect warnings such as `Can't list
'patch_*.py'` and explicitly compile the root patch/runtime files as well. The
Ubuntu Actions job expands these globs. A successful local exit alone does not
prove equivalent Linux or hardware coverage.

## Publish the branch and draft PR

Review `git diff` and `git status` first, including newly created files, then:

```bash
git add <reviewed-files>
git commit -m "fix: describe the concrete correction"
git push --set-upstream origin HEAD
```

Open a draft PR on GitHub with the original `magiccodingman/vllm-radiance:main`
as the base and your fork's topic branch as the head. With GitHub CLI, the
equivalent is:

```bash
gh pr create --repo magiccodingman/vllm-radiance --base main --head YOUR_USERNAME:YOUR_BRANCH --draft
```

Use the PR template to record the cause, fix, regression tests, exact verification
results and outstanding qualification. A PR is appropriate only when the change
has a demonstrated purpose; do not invent an issue reference.

## GitHub CPU checks and maintainer setup

`radiance-pr-checks.yml` runs on pull requests targeting main and checks the merge
candidate against the event's exact base SHA. It uses a read-only token, does not
persist checkout credentials and does not request secrets. It runs the baseline
and all automatic tests selected for affected areas; physical/candidate-image
gates retain their existing manual status. Actions are pinned to reviewed commit
SHAs. A new automatic test may require explicit additional CI dependencies.

The workflow must be accepted into the target repository for ongoing use. A
maintainer may need to approve Actions runs from first-time/fork contributors.
To test the configuration before upstream integration, open a PR from a topic
branch to main within your own fork and enable Actions there as necessary. Your
fork's CPU check does not configure checks on the original repository.

The repository owner must configure branch protection/rulesets: require a PR,
the **Radiance CPU checks** status, maintainer approval and appropriately restricted
bypass/merge permissions. If a named maintainer must approve every change, configure
CODEOWNERS and required code-owner review with that maintainer's agreement. Do not
grant Actions PR-write permission for this CPU-only workflow.

Preserve the existing GitHub/GitLab synchronization and GitLab publication policy;
review protections and mirror behavior on both hosts as described in
REPOSITORY_SYNC.md. Never copy upstream synchronization or publishing secrets to
a contributor fork. Do not attach unreviewed fork code to a production GPU host.

The old comment-only GitHub Docker workflow is retained as
`.github/workflows/publish-docker.yml.disabled`. Its former `.yml` suffix caused
GitHub to report an invalid workflow on pushes, despite publishing being disabled.

CPU success does not qualify ROCm, R9700 TP2, performance or production defaults.
Apply the existing registry and qualification policies before promotion. No GPU
purchase is required to contribute: missing physical evidence can be supplied by
the maintainer's laboratory before accepting changes that require it.
