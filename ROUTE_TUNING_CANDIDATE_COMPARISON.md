# Route Tuning Candidate Comparison

This file compares the original CMPD weekly schedule against two tuned candidates.

- `original_cmpd_weekly`: imported CMPD weekly route
- `tuned_cmpd_weekly`: v1, aggressive weak-point reallocation
- `tuned_cmpd_weekly_v2`: v2, weak-point reallocation with original hotspot-stop preservation

All runs use:

- 168 hourly steps per replicate
- 30 replicates
- seed sequence 7 through 500, stepping by 17
- `realized_crime = 1` for realized incidents
- `realized_crime = 0` for blocked attempts

## Summary Table

| Metric | Original | Tuned v1 | Tuned v2 |
|---|---:|---:|---:|
| Mean attempted incidents | 179.67 | 177.30 | 179.57 |
| Mean realized incidents | 164.50 | 158.30 | 162.33 |
| Mean blocked attempts | 15.17 | 19.00 | 17.23 |
| Patrol coverage ratio | 0.2105 | 0.3421 | 0.3947 |
| Hotspot coverage ratio | 1.0000 | 0.7500 | 1.0000 |
| Top hotspot visits | 8 | 6 | 8 |
| Station distribution MAE | 0.0489 | 0.0465 | 0.0489 |
| Centroid distribution MAE | 0.0324 | 0.0300 | 0.0311 |
| Hour distribution MAE | 0.0164 | 0.0170 | 0.0159 |

## Statistical Comparison Against Original

### Tuned v1

- Original realized rate: 0.9156
- Tuned v1 realized rate: 0.8928
- Difference: -0.0227
- p-value: 0.0000316
- Result: statistically lower realized-incident rate than original

### Tuned v2

- Original realized rate: 0.9156
- Tuned v2 realized rate: 0.9040
- Difference: -0.0116
- p-value: 0.0181
- Result: statistically lower realized-incident rate than original

## Interpretation

Tuned v1 is the strongest route if the only goal is reducing realized incidents. It produced the lowest realized incident rate and the highest blocked-attempt rate, but it reduced hotspot coverage from 1.00 to 0.75.

Tuned v2 is the stronger paper candidate if the route must preserve the original CMPD route's observed hotspot coverage. It still reduces realized incidents compared with the original route, while maintaining full hotspot coverage and increasing overall patrol coverage.

## Recommended Paper Framing

Use `tuned_cmpd_weekly_v2` as the defensible final route candidate because it improves the original CMPD route without sacrificing observed hotspot coverage.

Use `tuned_cmpd_weekly` as a sensitivity/upper-bound candidate showing that more aggressive route reallocation can produce larger simulated incident reductions, but at the cost of reduced hotspot coverage.

## Key Files

```text
Route_ML/candidate_routes/tuned_cmpd_weekly.geojson
Route_ML/candidate_routes/tuned_cmpd_weekly_summary.json
Route_ML/candidate_routes/tuned_cmpd_weekly_v2.geojson
Route_ML/candidate_routes/tuned_cmpd_weekly_v2_summary.json
Route_ML/outputs/original_vs_tuned_cmpd_weekly_comparison.json
Route_ML/outputs/original_vs_tuned_cmpd_weekly_v2_comparison.json
Route_ML/outputs/tuned_v1_vs_tuned_v2_comparison.json
results/tuned_cmpd_weekly/
results/tuned_cmpd_weekly_v2/
```
