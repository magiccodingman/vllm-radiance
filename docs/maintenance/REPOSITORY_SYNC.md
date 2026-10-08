# GitHub / GitLab main synchronization

GitLab project `lance-wright/vllm-radiance` keeps its protected-branch push
mirror to GitHub repository `magiccodingman/vllm-radiance`. That mirror must
have **Keep divergent refs** enabled so an incoming GitHub merge cannot be
replaced by older GitLab history.

The GitHub workflow `.github/workflows/sync-gitlab.yml` runs on pushes to
`main` (including merged PRs), or manually from Actions on `main`. It fetches
the current branch from both hosts, transfers commits using a normal
fast-forward-only push, and skips when GitLab already contains GitHub's
commits. It never force-pushes, deletes refs, copies PR discussions, or syncs
GitHub tags or other branches. Docker publishing remains on GitLab.

If independent commits land on both hosts, synchronization fails in GitHub
Actions. Fetch both histories, merge them on a review branch, integrate the
merge through the normal review process, and rerun **Sync main to GitLab**.
Do not resolve a divergence by force-pushing away either side's commits.
For routine work, merge changes on one host at a time and let synchronization
finish before starting another main-branch update on the other host.

## Credentials and maintenance

GitHub repository Actions secrets:

- `GITLAB_SYNC_SSH_KEY`: dedicated Ed25519 private deploy key.
- `GITLAB_SYNC_KNOWN_HOSTS`: verified GitLab SSH host key entry.

GitLab deploy key **GitHub Actions main sync - vllm-radiance** is private to
this project, has write access, and is explicitly allowed to push protected
`main`. Other existing protection rules are preserved. As with GitLab write
deploy keys generally, its project access also covers unprotected branches;
the workflow itself only pushes `main`.

Rotate the key by registering a replacement public key in this GitLab
project, granting it protected-main push access, updating the GitHub secret,
verifying a successful run, then revoking the old key. Verify host-key changes
through trusted administrative access before updating the known-hosts secret.
Never store private keys in Git, documentation, or issue comments.

To reuse this workflow for another repository, change both remote URLs and
the repository guard/concurrency group, create a separate project deploy key,
configure its protected-branch permission and the two Actions secrets, and
ensure any outgoing mirror preserves divergent refs. Both branches must
share Git history. Repository and runner access must be verified separately.
