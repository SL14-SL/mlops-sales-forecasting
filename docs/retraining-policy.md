# Retraining Policy

## Purpose

The retraining policy decides whether new evidence justifies candidate
training. Scheduling and retraining are deliberately separate:

- Prefect determines when the policy is evaluated.
- The policy determines whether training should start.
- Champion/challenger evaluation determines whether a trained model is promoted.

This avoids both unconditional retraining and unconditional promotion.

## Evidence Model

The signal collector normalizes:

| Evidence | Meaning |
|---|---|
| Dataset version | Stable identity of currently available new data |
| New training rows | Rows from unprocessed ground-truth batches |
| Batch IDs | Unique identities used for deduplication |
| Data quality | Whether new ground truth is safe to use |
| Performance degradation | Persistent RMSE, MAE or bias breach |
| Feature drift | Persistent drift across recent windows |
| Scheduled refresh due | Maximum model age reached |
| Cooldown | Recent training blocks another run |
| Budget availability | Maximum row or workload constraint is respected |

The decision result contains an action, stable decision ID, trigger types,
human-readable reasons and the full evidence snapshot.

### Lifecycle monitoring evidence

The Streamlit lifecycle dashboard combines delayed ground truth with
rolling forecast metrics and records when retraining and final champion
promotion occurred.

<p align="center">
  <img
    src="images/streamlit_dashboard.png"
    width="75%"
    alt="Forecast performance and retraining lifecycle dashboard"
  >
</p>

<p align="center">
  <em>
    Rolling RMSE, MAE and bias across the simulated lifecycle,
    including the drift period, retraining event and final-refit
    champion promotion.
  </em>
</p>

## Decision Order

Blocking conditions are evaluated before positive triggers:

1. enough new training rows must exist;
2. data quality must pass;
3. workload budget must be available;
4. cooldown must be inactive;
5. at least one valid trigger must be active.

Valid triggers include:

- persistent performance degradation;
- persistent feature drift;
- scheduled model refresh.

If all gates pass, the action is `train_candidate`. Otherwise it is `skip`.

## Persistence Across Windows

A single noisy monitoring window should not trigger training. Performance and
feature drift must remain active across the configured number of consecutive
windows.

Typical reasons returned by the collector include:

- not enough recent windows;
- required monitoring columns are missing;
- thresholds were not breached consecutively;
- no matching prediction and ground-truth rows exist.

Missing evidence is reported explicitly instead of silently interpreted as
degradation.

## Ground-Truth Deduplication

Each ground-truth batch receives a deterministic ID. Retraining state records
the batch IDs consumed by a successful retraining lifecycle.

On the next evaluation:

- already processed batches remain valid historical evidence;
- their rows are not counted as new training rows;
- only unseen batch IDs contribute to the new-data threshold.

A skipped decision does not mark batches as processed. State is written only
at the appropriate successful lifecycle boundary.

## Cooldown

Cooldown prevents repeated training immediately after a previous run. Its
reference time is resolved from retraining state and, where appropriate, the
active release or last successful training metadata.

Malformed or missing state does not falsely activate cooldown. A model age
calculation uses timezone-aware UTC timestamps.

## Scheduled Refresh

Even when no drift or degradation is detected, a maximum model age can request
a scheduled candidate refresh. This is useful when monitoring evidence is
sparse or the model should periodically incorporate accumulated data.

Scheduled refresh does not guarantee deployment. The candidate still passes
normal comparison, final refit, release publication and verification.

## Decision Examples

### Insufficient new data

```json
{
  "action": "skip",
  "reasons": ["Insufficient new training rows: 0/500."],
  "trigger_types": []
}
```

### Cooldown active

```json
{
  "action": "skip",
  "reasons": ["Retraining cooldown is active."],
  "trigger_types": []
}
```

### Scheduled refresh

```json
{
  "action": "train_candidate",
  "reasons": ["Scheduled model refresh interval reached."],
  "trigger_types": ["scheduled_refresh"]
}
```

## Running the Policy

Evaluate the complete automatic retraining flow once:

```bash
make auto-retrain
```

Register the recurring Prefect deployment:

```bash
make prefect-setup
make prefect-worker
```

The deployment schedule should use an explicit timezone such as
`Europe/Berlin` when wall-clock execution time matters.

## Promotion Remains Independent

`train_candidate` means that sufficient evidence exists to spend resources on
training. It does not mean that the candidate is production-worthy.

Promotion still requires:

- safe champion loading;
- fair evaluation on common validation data;
- lower configured error than the champion;
- successful final refit;
- serving-release publication;
- post-deployment verification.

If champion evaluation cannot be completed safely, promotion is blocked rather
than treating the missing comparison as an automatic win.

## Controlled Retraining Evaluation

The lifecycle simulation supports matched comparisons between a static
champion and the adaptive retraining pipeline. Both runs use the same
scenario, drift parameters, initial champion and ground truth.

<p align="center">
  <img
    src="images/streamlit_retraining_comparison.png"
    width="75%"
    alt="Interactive comparison with and without model retraining"
  >
</p>

<p align="center">
  <em>
    Interactive Streamlit comparison of rolling forecast error for
    matched lifecycle runs with and without adaptive retraining.
  </em>
</p>

### Segment-level experiment result

The final offline evaluation separates overall performance from the
segments directly and indirectly affected by promotional drift.

<p align="center">
  <img
    src="images/promo_final_refit_comparison.png"
    width="100%"
    alt="Segment-level forecasting performance with and without retraining"
  >
</p>

<p align="center">
  <em>
    Controlled promo-effect decay experiment showing rolling forecast
    error and post-promotion RMSE by forecast segment.
  </em>
</p>

| Post-promotion segment | Without retraining | With final refit | Relative RMSE change |
|---|---:|---:|---:|
| All open stores | 916 | 893 | -2.4% |
| Promo stores | 1,032 | 969 | -6.1% |
| Non-promo stores | 825 | 836 | +1.4% |

Negative relative change indicates lower forecast error. The accepted
final-refit model improved the segment affected by promotional drift,
while performance for non-promo stores deteriorated slightly. This
result demonstrates why retraining decisions should be evaluated across
business-relevant segments rather than only through one aggregate metric.


## Related Documentation

- [Architecture](architecture.md)
- [Local development](local-development.md)
- [Serving releases](serving-releases.md)
- [Monitoring and SLOs](monitoring-and-slos.md)

