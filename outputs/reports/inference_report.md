# Inference report (val split, 12 steps)

| mode | collision | success | heading err | corridor viol (m) | CPCG steps | QP status | QP cert | QP corr (m) | QP (s) | infer (s) |
|---|---|---|---|---|---|---|---|---|---|
| map_only | 0.1107 | 0.62 | 0.0 | 0.0000 | 0 | not_run | no | 0.0000 | 0.000 | 0.12 |
| safety_only | 0.0458 | 0.88 | 0.0 | 0.3483 | 0 | not_run | no | 0.0000 | 0.000 | 0.21 |
| cpcg | 0.0000 | 1.00 | 0.0 | 0.0001 | 11 | solved | yes | 0.1200 | 0.005 | 0.24 |
