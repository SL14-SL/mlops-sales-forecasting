# Template updates

This project was generated from the reusable MLOps project template.
Copier records the template source, version and project answers in
`.copier-answers.yml`.

Template updates modify the project repository only. They do not directly
change running infrastructure or production deployments.

## Check the current template version

```bash
grep -n \
  '_commit' \
  .copier-answers.yml
```

## Prepare an update

Start from a clean working tree:

```bash
git status
```

Create a dedicated update branch:

```bash
git switch \
  --create chore/template-vX.Y.Z
```

Replace `vX.Y.Z` with the target template release.

## Apply the update

Run Copier without adding it permanently to the project dependencies:

```bash
uvx \
  --from "copier>=9,<10" \
  copier update \
  --vcs-ref vX.Y.Z
```

For larger version jumps, prefer updating one released template version at a
time. This makes conflicts and behavioral changes easier to review.

## Review the result

Inspect all changed and newly generated files:

```bash
git status

git diff \
  --stat

git diff
```

Check for unresolved merge markers:

```bash
grep -RIn \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  -E '^(<<<<<<<|=======|>>>>>>>)' \
  .
```

A command with no output means that no conflict markers were found.

Project-specific behavior must be preserved when resolving conflicts,
particularly in:

- project training pipeline factories
- data ingestion and feature engineering
- model trainers and evaluators
- serving-release providers
- feedback and drift adapters
- project-specific configuration

## Validate the updated project

Synchronize dependencies:

```bash
make sync
```

Run all code-quality and test checks:

```bash
make check
```

Validate Docker Compose:

```bash
docker compose \
  --profile orchestration \
  --profile tracking \
  --profile monitoring \
  config \
  --quiet
```

Build the production image:

```bash
docker build \
  --tag template-update-test:local \
  .
```

Validate Terraform when infrastructure files changed:

```bash
make terraform-validate
```

## Commit and review

Commit the update only after all checks pass:

```bash
git add .

git commit \
  -m "chore: update project template to vX.Y.Z"
```

Push the branch and open a pull request:

```bash
git push \
  --set-upstream origin \
  chore/template-vX.Y.Z
```

The pull request should document:

- previous and new template version
- conflicts that required manual resolution
- relevant dependency or infrastructure changes
- completed validation steps
- required deployment or migration actions

## Deploying the update

Merge the update through the normal project review process. Use the regular
CI/CD deployment workflow after the merge.

A Copier update never deploys directly to production. The resulting Git
changes remain reviewable, testable and reversible through the project's
normal release process.