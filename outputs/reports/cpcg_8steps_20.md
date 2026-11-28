# CPCG paired A/B/C/D experiment

- checkpoint: `outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt`
- dataset: real / test, steps = 8, seed = 0
- overrides: none

## Aggregate over scenes

| group | scenes | collision | collision-free | step viol mean/max (m) | final viol (m) | CPCG steps | QP solved | certified | QP corr (m) | s |
|---|---|---|---|---|---|---|---|---|---|---|
| cpcg_full | 20 | 0.0149 | 18 | 0.599 / 0.793 | 0.078 | 4.0 | 17 | 17 | 0.763 | 0.26 |
| cpcg_no_projection | 20 | 0.0244 | 15 | 0.599 / 0.793 | 0.365 | 4.0 | 0 | 0 | - | 0.22 |
| map_only | 20 | 0.0609 | 13 | 0.000 / 0.000 | 0.000 | 0.0 | 0 | 0 | - | 0.13 |
| safety_only | 20 | 0.0245 | 15 | 0.600 / 0.794 | 0.368 | 0.0 | 0 | 0 | - | 0.23 |

## Per scene

| scene | group | collision | min clearance (m) | step violation mean/max (m) | final violation (m) | CPCG steps | QP | QP correction (m) | cert |
|---|---|---|---|---|---|---|---|---|---|
| 0 | map_only | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 0 | safety_only | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 0 | cpcg_no_projection | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 0 | cpcg_full | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 0 | shared pack across B/C/D | True | | | | | | | |
| 1 | map_only | 0.0000 | 0.304 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 1 | safety_only | 0.0305 | -0.130 | 0.671 / 0.973 | 0.958 | 0 | not_run | 0.000 | False |
| 1 | cpcg_no_projection | 0.0298 | -0.128 | 0.669 / 0.968 | 0.953 | 4 | not_run | 0.000 | False |
| 1 | cpcg_full | 0.0298 | -0.128 | 0.669 / 0.968 | 0.953 | 4 | primal infeasible | 0.000 | False |
| 1 | shared pack across B/C/D | True | | | | | | | |
| 2 | map_only | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 2 | safety_only | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 2 | cpcg_no_projection | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 2 | cpcg_full | 0.0000 | 1.625 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 2 | shared pack across B/C/D | True | | | | | | | |
| 3 | map_only | 0.0000 | 1.047 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 3 | safety_only | 0.0000 | 0.980 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 3 | cpcg_no_projection | 0.0000 | 0.980 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 3 | cpcg_full | 0.0000 | 0.980 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 3 | shared pack across B/C/D | True | | | | | | | |
| 4 | map_only | 0.0000 | 1.604 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 4 | safety_only | 0.0000 | 0.831 | 0.185 / 0.479 | 0.462 | 0 | not_run | 0.000 | False |
| 4 | cpcg_no_projection | 0.0000 | 0.832 | 0.181 / 0.470 | 0.454 | 4 | not_run | 0.000 | False |
| 4 | cpcg_full | 0.0000 | 0.859 | 0.181 / 0.470 | 0.000 | 4 | solved | 0.611 | True |
| 4 | shared pack across B/C/D | True | | | | | | | |
| 5 | map_only | 0.0000 | 1.552 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 5 | safety_only | 0.0000 | 1.552 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 5 | cpcg_no_projection | 0.0000 | 1.552 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 5 | cpcg_full | 0.0000 | 1.552 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 5 | shared pack across B/C/D | True | | | | | | | |
| 6 | map_only | 0.0000 | 1.875 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 6 | safety_only | 0.0000 | 1.875 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 6 | cpcg_no_projection | 0.0000 | 1.875 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 6 | cpcg_full | 0.0000 | 1.875 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 6 | shared pack across B/C/D | True | | | | | | | |
| 7 | map_only | 0.0000 | 3.964 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 7 | safety_only | 0.0000 | 3.810 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 7 | cpcg_no_projection | 0.0000 | 3.810 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 7 | cpcg_full | 0.0000 | 3.810 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 7 | shared pack across B/C/D | True | | | | | | | |
| 8 | map_only | 0.3333 | -1.509 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 8 | safety_only | 0.1413 | -1.038 | 2.969 / 3.540 | 2.029 | 0 | not_run | 0.000 | False |
| 8 | cpcg_no_projection | 0.1411 | -1.031 | 2.965 / 3.540 | 2.022 | 4 | not_run | 0.000 | False |
| 8 | cpcg_full | 0.0000 | 0.512 | 2.965 / 3.540 | 0.000 | 4 | solved | 2.204 | True |
| 8 | shared pack across B/C/D | True | | | | | | | |
| 9 | map_only | 0.0000 | 0.375 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 9 | safety_only | 0.0000 | 0.375 | 0.601 / 0.601 | 0.601 | 0 | not_run | 0.000 | False |
| 9 | cpcg_no_projection | 0.0000 | 0.375 | 0.601 / 0.601 | 0.601 | 4 | not_run | 0.000 | False |
| 9 | cpcg_full | 0.0000 | 0.375 | 0.601 / 0.601 | 0.601 | 4 | primal infeasible | 0.000 | False |
| 9 | shared pack across B/C/D | True | | | | | | | |
| 10 | map_only | 0.2606 | -2.959 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 10 | safety_only | 0.2674 | -1.389 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 10 | cpcg_no_projection | 0.2674 | -1.389 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 10 | cpcg_full | 0.2674 | -1.389 | 0.000 / 0.000 | 0.000 | 4 | no_constraints | 0.000 | False |
| 10 | shared pack across B/C/D | True | | | | | | | |
| 11 | map_only | 0.1015 | -0.139 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 11 | safety_only | 0.0000 | 1.209 | 0.561 / 0.920 | 0.000 | 0 | not_run | 0.000 | False |
| 11 | cpcg_no_projection | 0.0000 | 1.209 | 0.560 / 0.916 | 0.000 | 4 | not_run | 0.000 | False |
| 11 | cpcg_full | 0.0000 | 1.209 | 0.560 / 0.916 | 0.000 | 4 | solved | 0.000 | True |
| 11 | shared pack across B/C/D | True | | | | | | | |
| 12 | map_only | 0.0000 | 0.408 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 12 | safety_only | 0.0000 | 0.615 | 0.067 / 0.129 | 0.000 | 0 | not_run | 0.000 | False |
| 12 | cpcg_no_projection | 0.0000 | 0.615 | 0.067 / 0.129 | 0.000 | 4 | not_run | 0.000 | False |
| 12 | cpcg_full | 0.0000 | 0.615 | 0.067 / 0.129 | 0.000 | 4 | solved | 0.000 | True |
| 12 | shared pack across B/C/D | True | | | | | | | |
| 13 | map_only | 0.0044 | -0.010 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 13 | safety_only | 0.0000 | 0.349 | 1.027 / 1.277 | 0.673 | 0 | not_run | 0.000 | False |
| 13 | cpcg_no_projection | 0.0000 | 0.357 | 1.020 / 1.277 | 0.660 | 4 | not_run | 0.000 | False |
| 13 | cpcg_full | 0.0000 | 0.625 | 1.020 / 1.277 | 0.000 | 4 | solved | 0.838 | True |
| 13 | shared pack across B/C/D | True | | | | | | | |
| 14 | map_only | 0.0000 | 0.666 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 14 | safety_only | 0.0000 | 0.666 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 14 | cpcg_no_projection | 0.0000 | 0.666 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 14 | cpcg_full | 0.0000 | 0.666 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 14 | shared pack across B/C/D | True | | | | | | | |
| 15 | map_only | 0.0000 | 4.856 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 15 | safety_only | 0.0000 | 4.856 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 15 | cpcg_no_projection | 0.0000 | 4.856 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 15 | cpcg_full | 0.0000 | 4.856 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 15 | shared pack across B/C/D | True | | | | | | | |
| 16 | map_only | 0.1705 | -1.634 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 16 | safety_only | 0.0014 | 0.023 | 2.981 / 3.895 | 1.470 | 0 | not_run | 0.000 | False |
| 16 | cpcg_no_projection | 0.0011 | 0.025 | 2.976 / 3.895 | 1.461 | 4 | not_run | 0.000 | False |
| 16 | cpcg_full | 0.0000 | 0.632 | 2.976 / 3.895 | 0.000 | 4 | solved | 1.958 | True |
| 16 | shared pack across B/C/D | True | | | | | | | |
| 17 | map_only | 0.0225 | -0.161 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 17 | safety_only | 0.0000 | 0.486 | 0.574 / 0.856 | 0.154 | 0 | not_run | 0.000 | False |
| 17 | cpcg_no_projection | 0.0000 | 0.488 | 0.572 / 0.856 | 0.152 | 4 | not_run | 0.000 | False |
| 17 | cpcg_full | 0.0000 | 0.640 | 0.572 / 0.856 | 0.000 | 4 | solved | 0.156 | True |
| 17 | shared pack across B/C/D | True | | | | | | | |
| 18 | map_only | 0.0000 | 6.204 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 18 | safety_only | 0.0000 | 6.204 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 18 | cpcg_no_projection | 0.0000 | 6.204 | 0.000 / 0.000 | 0.000 | 4 | not_run | 0.000 | False |
| 18 | cpcg_full | 0.0000 | 6.204 | 0.000 / 0.000 | 0.000 | 4 | solved | 0.000 | True |
| 18 | shared pack across B/C/D | True | | | | | | | |
| 19 | map_only | 0.3253 | -1.863 | 0.000 / 0.000 | 0.000 | 0 | not_run | 0.000 | False |
| 19 | safety_only | 0.0493 | -0.369 | 2.373 / 3.216 | 1.003 | 0 | not_run | 0.000 | False |
| 19 | cpcg_no_projection | 0.0488 | -0.366 | 2.370 / 3.216 | 0.998 | 4 | not_run | 0.000 | False |
| 19 | cpcg_full | 0.0000 | 0.526 | 2.370 / 3.216 | 0.000 | 4 | solved | 1.100 | True |
| 19 | shared pack across B/C/D | True | | | | | | | |
