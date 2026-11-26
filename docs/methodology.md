# Extension methodology

## Observations and deployment inputs

The archive checksum and five CSV paths are pinned. CSVs use semicolon
delimiters, decimal commas and day-first wall-clock times with unknown time
zone. Blank and malformed timestamps are removed. Duplicate timestamp groups
and out-of-range charge invalidate neighboring targets. A charging flag at
either endpoint or a gauge increase breaks a discharge sequence. Counter
decreases flag likely resets and gaps over two hours break the sequence.
Screen rates above 1 become missing. Recognized cellular labels are mapped
across English and Spanish. Unknown connections receive a missing category.
Temperature is taken from the beginning of the interval. Device IDs come
from archive filenames. The explorer uses telemetry fields only.

Valid zero-drop intervals remain because the displayed gauge is quantized.
Extreme target values are flagged and capped only in fitting. Quantile
clipping, imputation, scaling and device/network encoding are fitted on
each training split. Users specify future interval-average usage for
simulation. Actual logged counter differences are retrospective covariates
for validation. Evaluation therefore measures performance given realized
usage, with future behavior supplied by the user in a forecast.

## Evaluation populations

Outer leave-one-device-out scores use raw observed drain. Inner grouped CV
tunes each family and chooses a family by macro device MAE. The separate
selection-procedure row tests this strategy with the outer device unseen.
A final grouped search on all devices chooses the family to deploy, then
refits with all eligible intervals. Family comparisons have only five
held-out groups, so small differences carry substantial uncertainty.

Chronological validation uses roughly the first 60% of eligible intervals
per device for fitting, the next 20% for tuning, then refits on both and
scores the last 20%. Intervals spanning a split boundary are purged.
Within a held-out contiguous sequence of at least two intervals, exact
piecewise integration starts from the first observed charge, then proceeds
without resetting to measured charge. For time to empty, eligible sequences
must end at a recorded zero. The zero timestamp is a measurement bound for
the crossing. The report counts paired crossings and model forecasts that
remain above zero when an observed zero is reached.

## Uncertainty and interpretation

Residual Monte Carlo samples out-of-fold drain errors from the matching
connection pool when sample size allows. The selected deployment model's
pool is estimated from group-aware predictions and may reflect selection
on the same folds. Outer coverage diagnostics use only inner residuals
from that fold and concern drain. Schedule simulations use a common residual
percentile across segments to model persistent error. This is one dependence
assumption. The sequence bootstrap resamples entire discharge sequences
within each device and refits the fixed selected model and its preprocessing.
This reflects fitted-model variability conditional on the five devices.
Residuals mix gauge noise, omitted behavior, regime shifts, model error and
device differences. Time-to-empty intervals have unknown calibration, and
zero-drain draws produce infinite times. JSON represents unbounded values as
null. Sensitivity runs hold the outer scoring population fixed.

Ridge contributions decompose the linear predictor on the log-drain scale.
Tree and mean predictions can be contrasted with simple feature references.
These contrasts are associational and can evaluate unusual feature
combinations. Permutation importance measures MAE changes within held-out
devices. Device identity stays constant in each device fold, so this
permutation cannot estimate between-device effects.
