# K-shot split report

- logs-dir: `/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/navsim_logs/trainval`
- Candidate pool: 978 logs / 573616 frames
- Stratification: city x turn(3) x speed(3) x accel(2) = 60 strata, guarantee=['city']

TV is the total variation distance from the candidate pool (0 is exact).

| split | logs | frames | TV(city) | TV(turn) | TV(speed) | TV(accel) |
|---|---|---|---|---|---|---|
| navsim_kshot_p001_seed0 | 10 | 4879 | 0.072 | 0.067 | 0.067 | 0.100 |
| navsim_kshot_p002_seed0 | 20 | 9061 | 0.097 | 0.033 | 0.033 | 0.050 |
| navsim_kshot_p005_seed0 | 49 | 29434 | 0.069 | 0.034 | 0.014 | 0.051 |
| navsim_kshot_p010_seed0 | 98 | 57205 | 0.026 | 0.017 | 0.017 | 0.000 |
| navsim_kshot_p100_seed0 | 978 | 573616 | 0.000 | 0.000 | 0.000 | 0.000 |
| navsim_kshot_p001_seed1 | 10 | 3251 | 0.072 | 0.067 | 0.067 | 0.100 |
| navsim_kshot_p002_seed1 | 20 | 9993 | 0.097 | 0.033 | 0.033 | 0.050 |
| navsim_kshot_p005_seed1 | 49 | 25691 | 0.069 | 0.034 | 0.014 | 0.051 |
| navsim_kshot_p010_seed1 | 98 | 52533 | 0.026 | 0.017 | 0.017 | 0.000 |
| navsim_kshot_p100_seed1 | 978 | 573616 | 0.000 | 0.000 | 0.000 | 0.000 |
| navsim_kshot_p001_seed2 | 10 | 3344 | 0.072 | 0.067 | 0.067 | 0.100 |
| navsim_kshot_p002_seed2 | 20 | 8497 | 0.097 | 0.033 | 0.033 | 0.050 |
| navsim_kshot_p005_seed2 | 49 | 31162 | 0.069 | 0.034 | 0.014 | 0.051 |
| navsim_kshot_p010_seed2 | 98 | 58744 | 0.026 | 0.017 | 0.017 | 0.000 |
| navsim_kshot_p100_seed2 | 978 | 573616 | 0.000 | 0.000 | 0.000 | 0.000 |
