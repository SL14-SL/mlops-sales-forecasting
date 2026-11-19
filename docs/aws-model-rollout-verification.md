# Dev API release-switch and rollback verification

`scripts/verify_aws_model_rollout.py` is a manually invoked integration check
against the deployed dev environment. It uses Terraform outputs, a logged-in
AWS profile and the existing MLflow/API secrets. Tokens are never printed.
The model name must end in `-dev`.

## Scope

The script creates a tagged duplicate registered version using identical model
weights, copies the complete portable release into a new immutable directory
and validates the artifacts before cutover. It then aligns the champion alias
and S3 pointer, restarts the API service through the existing ECS refresh helper
and checks readiness plus an authenticated prediction. Finally, it calls the
existing model rollback operation and checks the original version and release.

This is a deployment-mechanics test. It does not run new training, establish a
quality improvement or exercise automatic promotion gates. The separate
[95-day simulation](aws-simulation.md) covers quality-driven promotion in an
isolated workspace.

## Run

Run from the project root, after AWS login and Terraform backend initialization.
Use the same dev state as the deployed workflows. The selected profile needs
permission to read the secrets, read/write release objects and update the ECS
API service. Check the current starting pointer first:

```bash
aws s3 cp \
  s3://mlops-sales-forecasting-dev-713877988523-eu-north-1/models/active_serving_release.json - \
  --profile mlops-terraform --region eu-north-1
```

The example below uses the release verified on November 18, 2026. Update
`--expected-release` after future legitimate promotions. A mismatch aborts
before any test version or release is created.

```bash
AWS_PROFILE=mlops-terraform AWS_REGION=eu-north-1 \
uv run --locked --extra aws python scripts/verify_aws_model_rollout.py \
  --expected-release release-09eb8fc0-8bda-4abe-9d80-b45f65929f99
```

The sample prediction is Store 1 on April 28, 2015 with `Promo=1`, matching the
verified Rossmann dev calendar. A different dataset or forecasting state may
require a different request; preflight inference must succeed before cutover.

## Effects and recovery

- The live dev API is restarted twice and can be interrupted during replacement.
  Let the verification finish and do not run another deployment, promotion,
  simulation or model operation concurrently.
- Every run retains an additional test model version and portable release,
  tagged as verification with identical weights. Existing models and original
  artifacts are not deleted. This creates a small amount of additional storage.
- Ordinary forward-path errors attempt to restore the original alias/release.
  Recovery is not guaranteed after process termination, lost credentials,
  concurrent changes or an AWS outage. Inspect the actual pointer, champion
  and API readiness before choosing recovery; do not blindly roll back again.
- A failed rollback may already have restored the original pointer and alias.
  If the API remains on the test release, use the lifecycle `refresh` operation
  to apply the current pointer, rather than toggling it with another rollback.
- A successful rollback retains history: the pointer's operation becomes
  `rollback` and its previous release identifies the retained test release.
  The pointer JSON is therefore not byte-for-byte identical to its initial state.

## Evidence and tests

The script writes `before.json`, `prepared.json`, `after-switch.json` and
`after-rollback.json` under `evidence/aws-model-rollback/<UTC timestamp>/` as the
stages complete. Incomplete runs may contain only some files. Do not commit
the complete evidence directory by default. See the selected screenshots in
[AWS verification](aws-verification.md).

Offline regression tests check MLflow 3 logged-model IDs, artifact-validation
failure, stale/concurrent starting state, real registry version creation and
restoration after activation, ECS-refresh and prediction-check failures. Tests
substitute the live ECS/HTTP calls; running the script above performs the real
service and inference checks.
