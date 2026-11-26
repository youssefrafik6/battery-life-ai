# BatteryLife AI · Technical Highlights

An end-to-end battery drain and state-of-charge simulation application
built from the 2026 COMAP research of Youssef Rafik and Lucas Carreno.

**Technologies** · Python, pandas, NumPy, scikit-learn, XGBoost, Pydantic,
FastAPI, Streamlit, Altair, Docker, pytest, Ruff, GitHub Actions.

## Bullets

- Built a reproducible telemetry pipeline that extracted 1,554 usable
  intervals from five Android device logs, with archive verification,
  charging detection, counter-reset checks, and interval feature engineering.
- Benchmarked five regressors with nested device holdouts and chronological
  validation. The device-held-out selection procedure achieved 4.38
  percentage points per hour MAE and 1.75 percentage points median
  trajectory MAE across 198 discharge sequences.
- Delivered a seven-page Streamlit dashboard and FastAPI service that
  share a continuous-time prediction engine, with multi-stage battery
  simulations, residual Monte Carlo ranges, and sequence bootstrap analysis.

## Explanation

Battery charge is a state variable. A supervised model estimates the
interval-average drain, and integrating that rate advances the charge.
Under constant drain, time to empty follows directly from starting charge
divided by predicted drain. For a schedule, integrate each segment and
solve for the crossing inside the first segment that reaches zero.

I used nested grouped validation because adjacent observations share
measurements and device history. Preprocessing is fitted inside each fold.
The separate chronological test asks whether earlier logs generalize to
later use on known devices. Random Forest won the deployment search,
Histogram Boosting had the lowest outer device MAE, and Ridge led the
chronological test. The distinction matters when explaining model choice.

For analyst interviews, emphasize diagnostics, coverage, and scenario
interpretation. For data-science interviews, emphasize split design and
uncertainty. For ML and AI engineering interviews, emphasize the shared
engine, artifact lifecycle, endpoint contracts and container verification.
For data engineering interviews, emphasize checksums, schemas,
counter deltas, reproducibility and split boundaries.

Future-usage features are supplied as scenario assumptions. Gauge
quantization and unmeasured workloads contribute to residual error.
With five devices, generalization remains uncertain. The validation
set supplied no eligible full-depletion trajectories, so hours-to-empty
accuracy remains unmeasured.
