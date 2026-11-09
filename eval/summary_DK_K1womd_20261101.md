## Sources

| model | kind | csv |
|---|---|---|
| D2K1womd | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-womd_e16_scorer/9220138/2026.11.01.12.28.13.csv |
| D2K1womd | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-womd_e16_medoid/9220139/2026.11.01.12.28.13.csv |
| D2K1womd | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D2-K1-womd_e16_scorer/9220140/2026.11.01.12.31.33.csv |
| D2K1womd | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D2-K1-womd_e16_scorer/9220141/2026.11.01.12.09.35.csv |
| D2K1womd | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D2-K1-womd_e16/9220142/2026.11.01.13.43.17.csv |
| D2K1mix | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-mix_e13_scorer/9087030/2026.10.21.18.21.04.csv |
| D2K1mix | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-mix_e13_medoid/9087033/2026.10.21.18.17.25.csv |
| D2K1mix | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D2-K1-mix_e13_scorer/9087035/2026.10.21.18.24.48.csv |
| D2K1mix | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D2-K1-mix_e13_scorer/9087037/2026.10.21.18.05.10.csv |
| D2K1mix | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D2-K1-mix_e13/9087038/2026.10.21.19.36.48.csv |
| D3K1mix | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D3-K1-mix_e10_scorer/9210214/2026.10.31.17.04.59.csv |
| D3K1mix | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D3-K1-mix_e10_medoid/9210215/2026.10.31.17.04.59.csv |
| D3K1mix | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D3-K1-mix_e10_scorer/9210216/2026.10.31.17.07.52.csv |
| D3K1mix | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D3-K1-mix_e10_scorer/9210217/2026.10.31.16.49.30.csv |
| D3K1mix | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D3-K1-mix_e10/9210218/2026.10.31.18.22.02.csv |
| D1K1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_e16_scorer/9095481/2026.10.22.10.30.13.csv |
| D1K1 | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_e16_medoid/9095482/2026.10.22.10.30.13.csv |
| D1K1 | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D1-K1_e16_scorer/9095484/2026.10.22.10.33.36.csv |
| D1K1 | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D1-K1_e16_scorer/9095486/2026.10.22.10.15.53.csv |
| D1K1 | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D1-K1_e16/9095487/2026.10.22.11.51.57.csv |

## 1. Headline (x100)

| model | v1 scorer | v1 medoid | v1 Δ(s−m) | v2 navtest scorer | v2 navtest medoid | v2 Δ(s−m) | navhard scorer combined (s1 / s2) | navhard medoid |
|---|---|---|---|---|---|---|---|---|
| D2K1womd | 84.62 | 74.16 | 10.46 | 78.71 | — | — | 27.19 (59.44 / 44.35) | — |
| D2K1mix | 85.73 | 78.13 | 7.61 | 80.68 | — | — | 30.07 (65.07 / 45.75) | — |
| D3K1mix | 86.09 | 78.77 | 7.32 | 79.20 | — | — | 27.11 (62.61 / 44.14) | — |
| D1K1 | 80.36 | 69.97 | 10.38 | 73.70 | — | — | 28.05 (58.15 / 46.90) | — |

## 2. v1 navtest sub-metrics, scorer (x100)

| model | NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|---|
| D2K1womd | 96.22 | 95.39 | 80.03 | 90.66 | 99.93 | 94.90 | 84.62 |
| D2K1mix | 96.98 | 95.84 | 81.31 | 91.12 | 99.96 | 96.59 | 85.73 |
| D3K1mix | 97.72 | 96.09 | 79.10 | 93.48 | 99.98 | 95.59 | 86.09 |
| D1K1 | 96.52 | 94.76 | 72.80 | 89.26 | 97.11 | 95.30 | 80.36 |

## 3. v2 navtest sub-metrics, scorer (x100)

| model | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---|---|---|---|---|---|---|---|---|
| D2K1womd | 96.22 | 95.39 | 98.07 | 99.08 | 87.59 | 95.38 | 93.70 | 98.15 | 68.31 | 78.71 |
| D2K1mix | 96.97 | 95.84 | 98.66 | 99.36 | 87.91 | 96.03 | 94.08 | 98.19 | 72.06 | 80.68 |
| D3K1mix | 97.72 | 96.09 | 97.90 | 99.46 | 84.72 | 96.57 | 93.47 | 98.03 | 61.78 | 79.20 |
| D1K1 | 96.51 | 94.76 | 98.70 | 99.24 | 80.04 | 94.42 | 92.47 | 87.97 | 49.09 | 73.70 |

## 4. navhard two-stage sub-metrics, scorer (x100)

| model | stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D2K1womd | s1 | 93.67 | 81.78 | 95.33 | 99.33 | 81.74 | 92.00 | 92.67 | 97.56 | 56.44 | 59.44 |
| D2K1womd | s2 | 83.15 | 78.47 | 84.68 | 98.23 | 83.12 | 79.95 | 47.72 | 96.50 | 56.73 | 44.35 |
| D2K1mix | s1 | 95.22 | 87.11 | 96.67 | 99.56 | 81.73 | 94.00 | 93.11 | 97.56 | 60.00 | 65.07 |
| D2K1mix | s2 | 86.13 | 77.77 | 84.88 | 97.47 | 82.81 | 83.86 | 47.08 | 96.96 | 55.20 | 45.75 |
| D3K1mix | s1 | 95.00 | 87.11 | 94.56 | 99.56 | 79.97 | 94.44 | 90.22 | 96.67 | 44.89 | 62.61 |
| D3K1mix | s2 | 86.18 | 77.64 | 85.81 | 98.28 | 76.17 | 83.74 | 49.62 | 96.80 | 48.56 | 44.14 |
| D1K1 | s1 | 94.56 | 84.44 | 96.11 | 99.56 | 74.19 | 93.56 | 88.22 | 86.00 | 33.78 | 58.15 |
| D1K1 | s2 | 85.77 | 77.73 | 89.21 | 98.21 | 73.16 | 83.72 | 51.33 | 91.95 | 53.33 | 46.90 |

## 5. v1 oracle@64 (x100; capture = (scorer−mean64)/(oracle64−mean64))

| model | oracle64 | mean64 | scorer | medoid | capture | regret | pick=0 | pick=oracle | pick pctile | best of scorer top-1/4/8/16 | spearman(logit,score) | safe cand. | minADE64 / ADE scorer / ADE oracle (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D2K1womd | 98.09 | 68.13 | 84.62 | 74.16 | 0.55 | 13.47 | 8.12 | 30.32 | 64.17 | 84.62 / 89.36 / 91.40 / 93.61 | 0.412 | 76.68 | 0.282 / 1.406 / 2.112 |
| D2K1mix | 98.23 | 70.20 | 85.73 | 78.13 | 0.55 | 12.50 | 6.95 | 31.16 | 65.44 | 85.73 / 90.15 / 92.10 / 94.24 | 0.413 | 79.62 | 0.249 / 1.309 / 1.642 |
| D3K1mix | 97.80 | 70.38 | 86.09 | 78.77 | 0.57 | 11.71 | 5.93 | 29.61 | 65.08 | 86.09 / 90.26 / 92.10 / 93.77 | 0.385 | 80.62 | 0.276 / 1.446 / 1.520 |
| D1K1 | 93.99 | 63.83 | 80.36 | 69.97 | 0.55 | 13.63 | 8.32 | 26.49 | 60.60 | 80.36 / 85.63 / 87.74 / 89.70 | 0.378 | 76.04 | 0.852 / 2.104 / 1.991 |

consistency: D2K1womd: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.010, n=12146; D2K1mix: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.005, n=12146; D3K1mix: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.007, n=12146; D1K1: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.005, n=12146

## 6. v1 gap to D1K1: scorer part = Δcapture × (oracle64−mean64) of either model; candidate part = rest (x100)

| model | PDMS gap | scorer part | candidate part | oracle64 gap | mean64 gap |
|---|---|---|---|---|---|
| D2K1womd | -4.26 | -0.07 … -0.07 | -4.19 … -4.19 | -4.10 | -4.30 |
| D2K1mix | -5.37 | -0.18 … -0.17 | -5.20 … -5.19 | -4.24 | -6.37 |
| D3K1mix | -5.73 | -0.76 … -0.69 | -5.04 … -4.98 | -3.80 | -6.55 |

token→log map: 12146 tokens, 136 logs (from /gs/bs/tga-RLA/qdeng/RAP/metric_cache_v1_navtest)

## 7. paired D2K1womd − D1K1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D2K1womd | D1K1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 84.62 | 80.36 | 4.26 | [3.41, 5.19] | 57.9 | 25.7 | 16.4 |
| v1_medoid.score | 12146 | 74.16 | 69.97 | 4.19 | [3.01, 5.32] | 46.6 | 29.7 | 23.7 |
| v2_scorer.score | 12146 | 78.71 | 73.70 | 5.01 | [4.07, 5.98] | 57.1 | 24.3 | 18.6 |
| oracle64.oracle64 | 12146 | 98.09 | 93.99 | 4.10 | [3.56, 4.63] | 48.5 | 45.1 | 6.4 |
| oracle64.mean64 | 12146 | 68.13 | 63.83 | 4.30 | [3.39, 5.17] | 58.2 | 0.4 | 41.5 |
| oracle64.score_scorer | 12146 | 84.62 | 80.36 | 4.26 | [3.41, 5.19] | 57.9 | 25.7 | 16.4 |
| oracle64.minADE64 | 12146 | 0.282 | 0.852 | -0.570 | [-0.696, -0.462] | 29.7 | 0.0 | 70.3 |

## 7. paired D2K1womd − D2K1mix (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D2K1womd | D2K1mix | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 84.62 | 85.73 | -1.11 | [-1.76, -0.44] | 32.5 | 28.2 | 39.3 |
| v1_medoid.score | 12146 | 74.16 | 78.13 | -3.97 | [-5.36, -2.66] | 36.6 | 23.8 | 39.7 |
| v2_scorer.score | 12146 | 78.71 | 80.68 | -1.97 | [-2.66, -1.31] | 31.3 | 27.9 | 40.8 |
| oracle64.oracle64 | 12146 | 98.09 | 98.23 | -0.14 | [-0.40, 0.09] | 21.4 | 64.5 | 14.1 |
| oracle64.mean64 | 12146 | 68.13 | 70.20 | -2.07 | [-2.74, -1.43] | 45.5 | 0.3 | 54.2 |
| oracle64.score_scorer | 12146 | 84.62 | 85.73 | -1.11 | [-1.76, -0.44] | 32.5 | 28.2 | 39.3 |
| oracle64.minADE64 | 12146 | 0.282 | 0.249 | 0.033 | [0.026, 0.039] | 57.9 | 0.0 | 42.1 |

## 7. paired D3K1mix − D2K1womd (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D3K1mix | D2K1womd | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 86.09 | 84.62 | 1.47 | [0.75, 2.24] | 25.9 | 26.8 | 47.3 |
| v1_medoid.score | 12146 | 78.77 | 74.16 | 4.61 | [3.13, 6.22] | 20.1 | 20.3 | 59.6 |
| v2_scorer.score | 12146 | 79.20 | 78.71 | 0.49 | [-0.23, 1.23] | 25.7 | 25.7 | 48.5 |
| oracle64.oracle64 | 12146 | 97.80 | 98.09 | -0.30 | [-0.55, -0.04] | 11.2 | 63.2 | 25.6 |
| oracle64.mean64 | 12146 | 70.38 | 68.13 | 2.25 | [1.46, 3.08] | 52.1 | 0.4 | 47.5 |
| oracle64.score_scorer | 12146 | 86.09 | 84.62 | 1.47 | [0.75, 2.24] | 25.9 | 26.8 | 47.3 |
| oracle64.minADE64 | 12146 | 0.276 | 0.282 | -0.006 | [-0.015, 0.004] | 50.0 | 0.0 | 50.0 |
