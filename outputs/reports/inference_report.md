# Inference report (test split, 8 steps)

| mode | collision | success | heading err | corridor viol (m) | CPCG steps | QP status | QP cert | QP corr (m) | QP (s) | infer (s) |
|---|---|---|---|---|---|---|---|---|---|
| map_only | 0.0000 | 1.00 | 0.0 | 0.0000 | 0 | not_run | no | 0.0000 | 0.000 | 0.24 |
| safety_only | 0.0019 | 0.88 | 0.0 | 0.1591 | 0 | not_run | no | 0.0000 | 0.000 | 0.43 |
| cpcg | 0.0000 | 1.00 | 0.0 | 0.0669 | 4 | solved | yes | 0.0199 | 0.014 | 0.48 |
