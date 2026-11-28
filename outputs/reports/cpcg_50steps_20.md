# CPCG paired A/B/C/D experiment

- checkpoint: `outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt`
- dataset: real / test, steps = 50, seed = 0
- overrides: none

## Aggregate over scenes

| group | scenes | collision | collision-free | step viol mean/max (m) | final viol (m) | CPCG steps | QP solved | certified | QP corr (m) | s |
|---|---|---|---|---|---|---|---|---|---|---|
| cpcg_full | 20 | 0.0149 | 18 | 0.567 / 0.816 | 0.082 | 26.0 | 17 | 17 | 0.770 | 0.81 |
| cpcg_no_projection | 20 | 0.0247 | 15 | 0.567 / 0.816 | 0.378 | 26.0 | 0 | 0 | - | 0.77 |
| map_only | 20 | 0.0609 | 13 | 0.000 / 0.000 | 0.000 | 0.0 | 0 | 0 | - | 0.55 |
| safety_only | 20 | 0.0248 | 15 | 0.568 / 0.817 | 0.381 | 0.0 | 0 | 0 | - | 0.69 |

## Per scene

| scene | group | collision | min clearance (m) | step violation mean/max (m) | final violation (m) | CPCG steps | QP | QP correction (m) | cert |
|---|---|---|---|---|---|---|---|---|---|
| 0 | map_only | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 0 | safety_only | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 0 | cpcg_no_projection | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 0 | cpcg_full | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 0 | shared pack across B/C/D | True | | | | | | | |
| 1 | map_only | 0.0000 | 0.304 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 1 | safety_only | 0.0316 | -0.132 | 0.812 / 1.091 | 1.043 | 0 | not_run | 0.000 | False |
| 1 | cpcg_no_projection | 0.0308 | -0.130 | 0.809 / 1.086 | 1.037 | 26 | not_run | 0.000 | False |
| 1 | cpcg_full | 0.0308 | -0.130 | 0.809 / 1.086 | 1.037 | 26 | primal infeasible | 0.000 | False |
| 1 | shared pack across B/C/D | True | | | | | | | |
| 2 | map_only | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 2 | safety_only | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 2 | cpcg_no_projection | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 2 | cpcg_full | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 2 | shared pack across B/C/D | True | | | | | | | |
| 3 | map_only | 0.0000 | 1.047 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 3 | safety_only | 0.0000 | 0.972 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 3 | cpcg_no_projection | 0.0000 | 0.972 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 3 | cpcg_full | 0.0000 | 0.972 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 3 | shared pack across B/C/D | True | | | | | | | |
| 4 | map_only | 0.0000 | 1.604 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 4 | safety_only | 0.0000 | 0.834 | 0.194 / 0.416 | 0.379 | 0 | not_run | 0.000 | False |
| 4 | cpcg_no_projection | 0.0000 | 0.835 | 0.190 / 0.407 | 0.371 | 26 | not_run | 0.000 | False |
| 4 | cpcg_full | 0.0000 | 0.863 | 0.190 / 0.407 | 0.000 | 26 | solved | 0.510 | True |
| 4 | shared pack across B/C/D | True | | | | | | | |
| 5 | map_only | 0.0000 | 1.552 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 5 | safety_only | 0.0000 | 1.552 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 5 | cpcg_no_projection | 0.0000 | 1.552 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 5 | cpcg_full | 0.0000 | 1.552 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 5 | shared pack across B/C/D | True | | | | | | | |
| 6 | map_only | 0.0000 | 1.875 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 6 | safety_only | 0.0000 | 1.875 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 6 | cpcg_no_projection | 0.0000 | 1.875 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 6 | cpcg_full | 0.0000 | 1.875 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 6 | shared pack across B/C/D | True | | | | | | | |
| 7 | map_only | 0.0000 | 3.964 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 7 | safety_only | 0.0000 | 3.795 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 7 | cpcg_no_projection | 0.0000 | 3.795 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 7 | cpcg_full | 0.0000 | 3.795 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 7 | shared pack across B/C/D | True | | | | | | | |
| 8 | map_only | 0.3333 | -1.509 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 8 | safety_only | 0.1453 | -1.077 | 2.829 / 3.558 | 2.092 | 0 | not_run | 0.000 | False |
| 8 | cpcg_no_projection | 0.1451 | -1.074 | 2.826 / 3.558 | 2.085 | 26 | not_run | 0.000 | False |
| 8 | cpcg_full | 0.0000 | 0.503 | 2.826 / 3.558 | 0.000 | 26 | solved | 2.259 | True |
| 8 | shared pack across B/C/D | True | | | | | | | |
| 9 | map_only | 0.0000 | 0.375 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 9 | safety_only | 0.0000 | 0.375 | 0.601 / 0.601 | 0.601 | 0 | not_run | 0.000 | False |
| 9 | cpcg_no_projection | 0.0000 | 0.375 | 0.601 / 0.601 | 0.601 | 26 | not_run | 0.000 | False |
| 9 | cpcg_full | 0.0000 | 0.375 | 0.601 / 0.601 | 0.601 | 26 | primal infeasible | 0.000 | False |
| 9 | shared pack across B/C/D | True | | | | | | | |
| 10 | map_only | 0.2606 | -2.959 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 10 | safety_only | 0.2674 | -1.379 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 10 | cpcg_no_projection | 0.2674 | -1.379 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 10 | cpcg_full | 0.2674 | -1.379 | 0.000 / 0.000 | 0.000 | 26 | no_constraints | 0.000 | False |
| 10 | shared pack across B/C/D | True | | | | | | | |
| 11 | map_only | 0.1015 | -0.139 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 11 | safety_only | 0.0000 | 1.205 | 0.451 / 0.944 | 0.000 | 0 | not_run | 0.000 | False |
| 11 | cpcg_no_projection | 0.0000 | 1.205 | 0.451 / 0.944 | 0.000 | 26 | not_run | 0.000 | False |
| 11 | cpcg_full | 0.0000 | 1.205 | 0.451 / 0.944 | 0.000 | 26 | solved | 0.000 | True |
| 11 | shared pack across B/C/D | True | | | | | | | |
| 12 | map_only | 0.0000 | 0.408 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 12 | safety_only | 0.0000 | 0.616 | 0.055 / 0.136 | 0.000 | 0 | not_run | 0.000 | False |
| 12 | cpcg_no_projection | 0.0000 | 0.616 | 0.055 / 0.136 | 0.000 | 26 | not_run | 0.000 | False |
| 12 | cpcg_full | 0.0000 | 0.616 | 0.055 / 0.136 | 0.000 | 26 | solved | 0.000 | True |
| 12 | shared pack across B/C/D | True | | | | | | | |
| 13 | map_only | 0.0044 | -0.011 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 13 | safety_only | 0.0000 | 0.352 | 1.069 / 1.463 | 0.736 | 0 | not_run | 0.000 | False |
| 13 | cpcg_no_projection | 0.0000 | 0.359 | 1.063 / 1.463 | 0.724 | 26 | not_run | 0.000 | False |
| 13 | cpcg_full | 0.0000 | 0.673 | 1.063 / 1.463 | 0.000 | 26 | solved | 0.834 | True |
| 13 | shared pack across B/C/D | True | | | | | | | |
| 14 | map_only | 0.0000 | 0.666 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 14 | safety_only | 0.0000 | 0.666 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 14 | cpcg_no_projection | 0.0000 | 0.666 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 14 | cpcg_full | 0.0000 | 0.666 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 14 | shared pack across B/C/D | True | | | | | | | |
| 15 | map_only | 0.0000 | 4.856 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 15 | safety_only | 0.0000 | 4.856 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 15 | cpcg_no_projection | 0.0000 | 4.856 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 15 | cpcg_full | 0.0000 | 4.856 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 15 | shared pack across B/C/D | True | | | | | | | |
| 16 | map_only | 0.1705 | -1.634 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 16 | safety_only | 0.0068 | -0.034 | 2.781 / 3.980 | 1.657 | 0 | not_run | 0.000 | False |
| 16 | cpcg_no_projection | 0.0064 | -0.032 | 2.776 / 3.980 | 1.647 | 26 | not_run | 0.000 | False |
| 16 | cpcg_full | 0.0000 | 0.589 | 2.776 / 3.980 | 0.000 | 26 | solved | 2.086 | True |
| 16 | shared pack across B/C/D | True | | | | | | | |
| 17 | map_only | 0.0224 | -0.160 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 17 | safety_only | 0.0000 | 0.508 | 0.474 / 0.909 | 0.132 | 0 | not_run | 0.000 | False |
| 17 | cpcg_no_projection | 0.0000 | 0.511 | 0.472 / 0.909 | 0.129 | 26 | not_run | 0.000 | False |
| 17 | cpcg_full | 0.0000 | 0.640 | 0.472 / 0.909 | 0.000 | 26 | solved | 0.133 | True |
| 17 | shared pack across B/C/D | True | | | | | | | |
| 18 | map_only | 0.0000 | 6.204 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 18 | safety_only | 0.0000 | 6.204 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 18 | cpcg_no_projection | 0.0000 | 6.204 | 0.000 / 0.000 | 0.000 | 26 | not_run | 0.000 | False |
| 18 | cpcg_full | 0.0000 | 6.204 | 0.000 / 0.000 | 0.000 | 26 | solved | 0.000 | True |
| 18 | shared pack across B/C/D | True | | | | | | | |
| 19 | map_only | 0.3253 | -1.863 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 19 | safety_only | 0.0454 | -0.332 | 2.094 / 3.235 | 0.980 | 0 | not_run | 0.000 | False |
| 19 | cpcg_no_projection | 0.0446 | -0.325 | 2.091 / 3.235 | 0.975 | 26 | not_run | 0.000 | False |
| 19 | cpcg_full | 0.0000 | 0.528 | 2.091 / 3.235 | 0.000 | 26 | solved | 1.112 | True |
| 19 | shared pack across B/C/D | True | | | | | | | |
