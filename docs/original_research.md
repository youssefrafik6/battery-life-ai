# Original COMAP research

**Modeling Smartphone Battery Drain**, 2026 COMAP MCM Problem A, team 2633026,
February 1, 2026. Original authors **Youssef Rafik and Lucas Carreno**.
See the [evaluation and simulation methodology](methodology.md) for the repository’s benchmark protocol.

## Objectives and data

The team estimated smartphone state of charge in continuous time, validated
discharge trajectories against logs, estimated time to empty in four usage
scenarios, and tested sensitivity and uncertainty. The report used five Android
traces from [Zenodo 13982080](https://doi.org/10.5281/zenodo.13982080). The traces
span about two days to several weeks, with a typical median spacing around ten
minutes. The report describes over 2,000 raw intervals and about 1,300 training
discharge intervals. Devices were OnePlus LE2113 and LE2123, POCO M2102J20SG,
and Samsung SM-A226B and SM-G991B.

## Equations and methods

- The governing equation is $dS/dt=-r(t)$, where $S$ is percent charge and $r$
  is percentage points per hour. The conceptual drain model adds a device
  baseline to terms for screen use, receive rate, send rate, cellular status,
  and temperature.
- Snapshots were parsed and sorted by time. Missing or invalid timestamps
  were rejected. Gauge rises were flagged and charging intervals excluded.
  Observed drain was $(S_k-S_{k+1})/\Delta t_h$. Screen fraction and network
  rates use cumulative counter differences divided by interval seconds.
  Receive and send units are MB/s. Temperature is in °C.
- The final settings were 300 seconds minimum duration, 30 pp/h maximum drain,
  and 0.99 feature winsorization. Appendix F also discusses 600 seconds as
  an illustrative threshold for reducing gauge quantization noise.
- The fitted predictor used Ridge on $\log(1+r)$ with device indicators. An
  inverse transformation $\exp(\hat y)-1$ returned predictions to pp/h. The
  report describes entire-device holdouts and a modest MAE improvement over
  the mean predictor, without giving fold-level drain MAE.
- Sequence validation initialized charge at the observed first value and used
  $S_{k+1}=\max(0,S_k-\hat r_k\Delta t_k)$. The report does not specify
  whether every summarized sequence prediction came from a held-out-device fit.
- A constant scenario gives $T=S_0/r$. The report used starting levels 100,
  75, 50 and 25 percent. It sampled fitted residuals, added them to the
  scenario rate, clipped the result to [0, 30] pp/h, and summarized the
  implied time-to-empty distribution. Small rates produce long upper tails.

## Historical results from the report

| At full charge | Rate (pp/h) | Deterministic (h) | Monte Carlo median (h) | P10–P90 (h) |
|---|---:|---:|---:|---:|
| Idle | 9.70 | 10.31 | 10.22 | 5.05–20.98 |
| Screen-heavy | 9.72 | 10.29 | 9.48 | 4.99–18.42 |
| Wi-Fi streaming | 12.72 | 7.86 | 8.10 | 4.39–13.54 |
| Mobile streaming | 14.48 | 6.90 | 7.07 | 4.03–10.65 |

| Sensitivity variant | Sequence MAE median / P90 / max (pp) | Idle / mobile streaming (h) |
|---|---:|---:|
| Final baseline | 4.82 / 14.68 / 27.81 | 10.31 / 6.90 |
| Minimum 600 seconds | 3.97 / 13.59 / 20.51 | 11.28 / 6.88 |
| Maximum drain 25 pp/h | 4.42 / 13.16 / 29.80 | 10.61 / 8.22 |
| Winsorization 0.95 | 3.87 / 13.44 / 42.55 | 13.30 / 5.03 |
| Remove mobile proxy | 4.66 / 14.91 / 27.81 | 10.01 / 7.71 |

The report kept the 300-second baseline after comparing scenario plausibility
and validation errors. Missing brightness, app-specific processor load, GPS,
and signal strength limit interpretation. Quantized charge, irregular gaps,
occasional resets, and correlated features can distort interval estimates. The
report discussed aging through an illustrative effective-capacity factor.
Its scenario comparisons describe fitted associations.
