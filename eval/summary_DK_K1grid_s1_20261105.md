## Sources

| model | kind | csv |
|---|---|---|
| nusc_s1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-nusc-tgt-23k_s1_e8_scorer/9258278/2026.11.04.19.05.39.csv |
| nusc_s1 | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-nusc-tgt-23k_s1_e8_medoid/9258279/2026.11.04.19.08.37.csv |
| nusc_s1 | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D2-K1-nusc-tgt-23k_s1_e8_scorer/9258280/2026.11.04.19.13.10.csv |
| nusc_s1 | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D2-K1-nusc-tgt-23k_s1_e8_scorer/9258281/2026.11.04.18.57.33.csv |
| nusc_s1 | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D2-K1-nusc-tgt-23k_s1_e8/9258282/2026.11.04.20.33.21.csv |
| womd_s1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-womd_s1_e7_scorer/9242233/2026.11.03.13.03.32.csv |
| womd_s1 | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-womd_s1_e7_medoid/9242234/2026.11.03.13.07.27.csv |
| womd_s1 | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D2-K1-womd_s1_e7_scorer/9242235/2026.11.03.13.07.24.csv |
| womd_s1 | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D2-K1-womd_s1_e7_scorer/9242236/2026.11.03.12.48.56.csv |
| womd_s1 | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D2-K1-womd_s1_e7/9242237/2026.11.03.14.21.57.csv |
| D3half_s1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D3-K1-nusc-womd-tgt-23k_s1_e6_scorer/9258284/2026.11.04.19.10.48.csv |
| D3half_s1 | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D3-K1-nusc-womd-tgt-23k_s1_e6_medoid/9258285/2026.11.04.19.10.48.csv |
| D3half_s1 | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D3-K1-nusc-womd-tgt-23k_s1_e6_scorer/9258286/2026.11.04.19.10.13.csv |
| D3half_s1 | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D3-K1-nusc-womd-tgt-23k_s1_e6_scorer/9258287/2026.11.04.18.51.38.csv |
| D3half_s1 | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D3-K1-nusc-womd-tgt-23k_s1_e6/9258288/2026.11.04.20.34.07.csv |
| D1K1_s1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_s1_e10_scorer/9103914/2026.10.23.01.10.15.csv |
| D1K1_s1 | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_s1_e10_medoid/9103918/2026.10.23.01.07.21.csv |
| D1K1_s1 | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D1-K1_s1_e10_scorer/9103920/2026.10.23.01.21.19.csv |
| D1K1_s1 | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D1-K1_s1_e10_scorer/9103924/2026.10.23.00.55.20.csv |
| D1K1_s1 | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D1-K1_s1_e10/9103925/2026.10.23.02.37.29.csv |

## 1. Headline (x100)

| model | v1 scorer | v1 medoid | v1 Δ(s−m) | v2 navtest scorer | v2 navtest medoid | v2 Δ(s−m) | navhard scorer combined (s1 / s2) | navhard medoid |
|---|---|---|---|---|---|---|---|---|
| nusc_s1 | 81.82 | 71.72 | 10.10 | 76.78 | — | — | 29.72 (64.04 / 46.91) | — |
| womd_s1 | 82.95 | 69.90 | 13.05 | 76.71 | — | — | 30.56 (62.67 / 48.20) | — |
| D3half_s1 | 82.35 | 73.23 | 9.11 | 76.44 | — | — | 33.16 (66.75 / 48.80) | — |
| D1K1_s1 | 78.12 | 69.30 | 8.82 | 70.90 | — | — | 33.58 (63.92 / 51.66) | — |

## 2. v1 navtest sub-metrics, scorer (x100)

| model | NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|---|
| nusc_s1 | 96.46 | 95.07 | 77.45 | 86.83 | 99.69 | 96.12 | 81.82 |
| womd_s1 | 96.86 | 95.17 | 76.91 | 89.96 | 99.76 | 95.64 | 82.95 |
| D3half_s1 | 96.80 | 96.29 | 74.48 | 89.46 | 99.81 | 96.62 | 82.35 |
| D1K1_s1 | 96.20 | 96.28 | 66.15 | 88.94 | 97.13 | 96.25 | 78.12 |

## 3. v2 navtest sub-metrics, scorer (x100)

| model | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---|---|---|---|---|---|---|---|---|
| nusc_s1 | 96.46 | 95.07 | 98.64 | 99.42 | 85.02 | 95.04 | 93.84 | 96.81 | 53.59 | 76.78 |
| womd_s1 | 96.87 | 95.17 | 98.28 | 99.63 | 84.12 | 95.08 | 92.87 | 97.42 | 52.46 | 76.71 |
| D3half_s1 | 96.79 | 96.29 | 99.00 | 99.61 | 80.71 | 94.54 | 93.21 | 97.82 | 48.52 | 76.44 |
| D1K1_s1 | 96.20 | 96.28 | 99.04 | 99.54 | 72.44 | 93.41 | 91.50 | 89.05 | 32.96 | 70.90 |

## 4. navhard two-stage sub-metrics, scorer (x100)

| model | stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nusc_s1 | s1 | 95.67 | 86.44 | 97.22 | 99.56 | 77.72 | 95.78 | 92.44 | 96.44 | 41.78 | 64.04 |
| nusc_s1 | s2 | 85.56 | 80.08 | 86.55 | 98.62 | 82.05 | 82.98 | 48.53 | 96.60 | 41.67 | 46.91 |
| womd_s1 | s1 | 95.33 | 85.11 | 96.89 | 100.00 | 78.27 | 94.89 | 91.33 | 97.56 | 45.33 | 62.67 |
| womd_s1 | s2 | 84.71 | 82.87 | 87.36 | 98.63 | 79.76 | 82.91 | 52.62 | 97.60 | 42.45 | 48.20 |
| D3half_s1 | s1 | 96.56 | 89.78 | 98.89 | 99.78 | 73.95 | 95.56 | 92.89 | 97.33 | 34.67 | 66.75 |
| D3half_s1 | s2 | 87.90 | 81.98 | 88.59 | 98.55 | 75.93 | 86.18 | 50.31 | 97.58 | 41.84 | 48.80 |
| D1K1_s1 | s1 | 96.67 | 92.67 | 98.11 | 99.78 | 66.60 | 96.44 | 89.11 | 89.11 | 17.78 | 63.92 |
| D1K1_s1 | s2 | 91.45 | 83.63 | 92.61 | 98.38 | 69.10 | 89.16 | 55.71 | 89.32 | 34.12 | 51.66 |

## 5. v1 oracle@64 (x100; capture = (scorer−mean64)/(oracle64−mean64))

| model | oracle64 | mean64 | scorer | medoid | capture | regret | pick=0 | pick=oracle | pick pctile | best of scorer top-1/4/8/16 | spearman(logit,score) | safe cand. | minADE64 / ADE scorer / ADE oracle (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nusc_s1 | 97.57 | 65.07 | 81.82 | 71.72 | 0.52 | 15.75 | 8.10 | 26.62 | 64.57 | 81.82 / 87.84 / 90.39 / 93.01 | 0.424 | 75.41 | 0.381 / 1.800 / 1.531 |
| womd_s1 | 98.00 | 62.39 | 82.95 | 69.90 | 0.58 | 15.05 | 7.72 | 26.30 | 61.49 | 82.95 / 89.21 / 91.75 / 94.00 | 0.390 | 71.21 | 0.500 / 1.778 / 2.744 |
| D3half_s1 | 97.09 | 66.74 | 82.35 | 73.23 | 0.51 | 14.75 | 6.69 | 24.34 | 63.95 | 82.35 / 88.22 / 90.45 / 92.59 | 0.384 | 77.91 | 0.396 / 2.063 / 1.493 |
| D1K1_s1 | 93.34 | 64.88 | 78.12 | 69.30 | 0.47 | 15.22 | 7.34 | 20.83 | 54.30 | 78.12 / 85.60 / 88.07 / 89.99 | 0.276 | 78.25 | 1.042 / 3.579 / 2.319 |

consistency: nusc_s1: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.003, n=12146; womd_s1: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.006, n=12146; D3half_s1: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.003, n=12146; D1K1_s1: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.005, n=12146

## 6. v1 gap to D1K1_s1: scorer part = Δcapture × (oracle64−mean64) of either model; candidate part = rest (x100)

| model | PDMS gap | scorer part | candidate part | oracle64 gap | mean64 gap |
|---|---|---|---|---|---|
| nusc_s1 | -3.70 | -1.63 … -1.43 | -2.27 … -2.07 | -4.22 | -0.19 |
| womd_s1 | -4.83 | -4.00 … -3.19 | -1.63 … -0.83 | -4.65 | 2.49 |
| D3half_s1 | -4.23 | -1.49 … -1.40 | -2.83 … -2.74 | -3.75 | -1.86 |

token→log map: 12146 tokens, 136 logs (from /gs/bs/tga-RLA/qdeng/RAP/metric_cache_v1_navtest)

## 7. paired nusc_s1 − womd_s1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | nusc_s1 | womd_s1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 81.82 | 82.95 | -1.13 | [-1.83, -0.38] | 39.7 | 26.5 | 33.8 |
| v1_medoid.score | 12146 | 71.72 | 69.90 | 1.82 | [0.59, 3.03] | 24.9 | 29.5 | 45.6 |
| v2_scorer.score | 12146 | 76.78 | 76.71 | 0.08 | [-0.62, 0.82] | 42.6 | 23.6 | 33.8 |
| oracle64.oracle64 | 12146 | 97.57 | 98.00 | -0.43 | [-0.71, -0.14] | 8.4 | 55.7 | 35.8 |
| oracle64.mean64 | 12146 | 65.07 | 62.39 | 2.68 | [2.06, 3.28] | 55.4 | 0.3 | 44.3 |
| oracle64.score_scorer | 12146 | 81.82 | 82.95 | -1.13 | [-1.83, -0.38] | 39.7 | 26.5 | 33.8 |
| oracle64.minADE64 | 12146 | 0.381 | 0.500 | -0.119 | [-0.138, -0.098] | 24.4 | 0.0 | 75.6 |

## 7. paired D3half_s1 − womd_s1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D3half_s1 | womd_s1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 82.35 | 82.95 | -0.60 | [-1.27, 0.14] | 24.3 | 24.1 | 51.6 |
| v1_medoid.score | 12146 | 73.23 | 69.90 | 3.33 | [1.93, 4.97] | 21.6 | 22.6 | 55.8 |
| v2_scorer.score | 12146 | 76.44 | 76.71 | -0.27 | [-0.96, 0.44] | 30.1 | 21.5 | 48.4 |
| oracle64.oracle64 | 12146 | 97.09 | 98.00 | -0.90 | [-1.19, -0.61] | 5.6 | 50.7 | 43.8 |
| oracle64.mean64 | 12146 | 66.74 | 62.39 | 4.35 | [3.21, 5.53] | 59.9 | 0.2 | 40.0 |
| oracle64.score_scorer | 12146 | 82.35 | 82.95 | -0.60 | [-1.27, 0.14] | 24.3 | 24.1 | 51.6 |
| oracle64.minADE64 | 12146 | 0.396 | 0.500 | -0.104 | [-0.117, -0.091] | 24.8 | 0.0 | 75.2 |

## 7. paired D3half_s1 − nusc_s1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D3half_s1 | nusc_s1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 82.35 | 81.82 | 0.53 | [-0.14, 1.29] | 22.2 | 23.8 | 54.0 |
| v1_medoid.score | 12146 | 73.23 | 71.72 | 1.51 | [-0.32, 3.49] | 26.1 | 20.7 | 53.2 |
| v2_scorer.score | 12146 | 76.44 | 76.78 | -0.35 | [-1.06, 0.36] | 24.9 | 22.4 | 52.6 |
| oracle64.oracle64 | 12146 | 97.09 | 97.57 | -0.47 | [-0.68, -0.29] | 17.7 | 45.8 | 36.6 |
| oracle64.mean64 | 12146 | 66.74 | 65.07 | 1.67 | [0.47, 2.89] | 50.2 | 0.1 | 49.6 |
| oracle64.score_scorer | 12146 | 82.35 | 81.82 | 0.53 | [-0.14, 1.29] | 22.2 | 23.8 | 54.0 |
| oracle64.minADE64 | 12146 | 0.396 | 0.381 | 0.015 | [0.001, 0.026] | 56.2 | 0.0 | 43.8 |

## 7. paired nusc_s1 − D1K1_s1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | nusc_s1 | D1K1_s1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 81.82 | 78.12 | 3.70 | [2.53, 4.79] | 63.6 | 18.6 | 17.7 |
| v1_medoid.score | 12146 | 71.72 | 69.30 | 2.42 | [0.25, 4.57] | 54.2 | 21.1 | 24.8 |
| v2_scorer.score | 12146 | 76.78 | 70.90 | 5.88 | [4.95, 6.86] | 64.2 | 17.3 | 18.5 |
| oracle64.oracle64 | 12146 | 97.57 | 93.34 | 4.22 | [3.70, 4.79] | 49.3 | 39.0 | 11.7 |
| oracle64.mean64 | 12146 | 65.07 | 64.88 | 0.19 | [-1.30, 1.60] | 50.2 | 0.1 | 49.8 |
| oracle64.score_scorer | 12146 | 81.82 | 78.12 | 3.70 | [2.53, 4.79] | 63.6 | 18.6 | 17.7 |
| oracle64.minADE64 | 12146 | 0.381 | 1.042 | -0.661 | [-0.793, -0.548] | 26.8 | 0.0 | 73.2 |

## 7. paired womd_s1 − D1K1_s1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | womd_s1 | D1K1_s1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 82.95 | 78.12 | 4.83 | [3.82, 5.73] | 64.1 | 18.9 | 17.0 |
| v1_medoid.score | 12146 | 69.90 | 69.30 | 0.60 | [-1.35, 2.47] | 58.2 | 21.8 | 20.1 |
| v2_scorer.score | 12146 | 76.71 | 70.90 | 5.80 | [4.84, 6.71] | 61.6 | 16.2 | 22.2 |
| oracle64.oracle64 | 12146 | 98.00 | 93.34 | 4.65 | [4.04, 5.25] | 54.8 | 41.1 | 4.1 |
| oracle64.mean64 | 12146 | 62.39 | 64.88 | -2.49 | [-4.03, -1.05] | 45.0 | 0.1 | 54.9 |
| oracle64.score_scorer | 12146 | 82.95 | 78.12 | 4.83 | [3.82, 5.73] | 64.1 | 18.9 | 17.0 |
| oracle64.minADE64 | 12146 | 0.500 | 1.042 | -0.542 | [-0.673, -0.428] | 48.7 | 0.0 | 51.3 |
