## Sources

| model | kind | csv |
|---|---|---|
| D3K1mix | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D3-K1-mix_e10_scorer/9210214/2026.10.31.17.04.59.csv |
| D3K1mix | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D3-K1-mix_e10_medoid/9210215/2026.10.31.17.04.59.csv |
| D3K1mix | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D3-K1-mix_e10_scorer/9210216/2026.10.31.17.07.52.csv |
| D3K1mix | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D3-K1-mix_e10_scorer/9210217/2026.10.31.16.49.30.csv |
| D3K1mix | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D3-K1-mix_e10/9210218/2026.10.31.18.22.02.csv |
| D2K1mix | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-mix_e13_scorer/9087030/2026.10.21.18.21.04.csv |
| D2K1mix | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D2-K1-mix_e13_medoid/9087033/2026.10.21.18.17.25.csv |
| D2K1mix | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D2-K1-mix_e13_scorer/9087035/2026.10.21.18.24.48.csv |
| D2K1mix | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D2-K1-mix_e13_scorer/9087037/2026.10.21.18.05.10.csv |
| D2K1mix | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D2-K1-mix_e13/9087038/2026.10.21.19.36.48.csv |
| D1K1 | v1_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_e16_scorer/9095481/2026.10.22.10.30.13.csv |
| D1K1 | v1_medoid | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_navtest_D1-K1_e16_medoid/9095482/2026.10.22.10.30.13.csv |
| D1K1 | v2_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navtest_D1-K1_e16_scorer/9095484/2026.10.22.10.33.36.csv |
| D1K1 | navhard_scorer | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v2_navhard_two_stage_D1-K1_e16_scorer/9095486/2026.10.22.10.15.53.csv |
| D1K1 | oracle64 | /gs/bs/tga-RLA/qdeng/navsim_workspace/exp/eval_v1_oracle64_navtest_D1-K1_e16/9095487/2026.10.22.11.51.57.csv |

## 1. Headline (x100)

| model | v1 scorer | v1 medoid | v1 Δ(s−m) | v2 navtest scorer | v2 navtest medoid | v2 Δ(s−m) | navhard scorer combined (s1 / s2) | navhard medoid |
|---|---|---|---|---|---|---|---|---|
| D3K1mix | 86.09 | 78.77 | 7.32 | 79.20 | — | — | 27.11 (62.61 / 44.14) | — |
| D2K1mix | 85.73 | 78.13 | 7.61 | 80.68 | — | — | 30.07 (65.07 / 45.75) | — |
| D1K1 | 80.36 | 69.97 | 10.38 | 73.70 | — | — | 28.05 (58.15 / 46.90) | — |

## 2. v1 navtest sub-metrics, scorer (x100)

| model | NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|---|
| D3K1mix | 97.72 | 96.09 | 79.10 | 93.48 | 99.98 | 95.59 | 86.09 |
| D2K1mix | 96.98 | 95.84 | 81.31 | 91.12 | 99.96 | 96.59 | 85.73 |
| D1K1 | 96.52 | 94.76 | 72.80 | 89.26 | 97.11 | 95.30 | 80.36 |

## 3. v2 navtest sub-metrics, scorer (x100)

| model | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---|---|---|---|---|---|---|---|---|
| D3K1mix | 97.72 | 96.09 | 97.90 | 99.46 | 84.72 | 96.57 | 93.47 | 98.03 | 61.78 | 79.20 |
| D2K1mix | 96.97 | 95.84 | 98.66 | 99.36 | 87.91 | 96.03 | 94.08 | 98.19 | 72.06 | 80.68 |
| D1K1 | 96.51 | 94.76 | 98.70 | 99.24 | 80.04 | 94.42 | 92.47 | 87.97 | 49.09 | 73.70 |

## 4. navhard two-stage sub-metrics, scorer (x100)

| model | stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D3K1mix | s1 | 95.00 | 87.11 | 94.56 | 99.56 | 79.97 | 94.44 | 90.22 | 96.67 | 44.89 | 62.61 |
| D3K1mix | s2 | 86.18 | 77.64 | 85.81 | 98.28 | 76.17 | 83.74 | 49.62 | 96.80 | 48.56 | 44.14 |
| D2K1mix | s1 | 95.22 | 87.11 | 96.67 | 99.56 | 81.73 | 94.00 | 93.11 | 97.56 | 60.00 | 65.07 |
| D2K1mix | s2 | 86.13 | 77.77 | 84.88 | 97.47 | 82.81 | 83.86 | 47.08 | 96.96 | 55.20 | 45.75 |
| D1K1 | s1 | 94.56 | 84.44 | 96.11 | 99.56 | 74.19 | 93.56 | 88.22 | 86.00 | 33.78 | 58.15 |
| D1K1 | s2 | 85.77 | 77.73 | 89.21 | 98.21 | 73.16 | 83.72 | 51.33 | 91.95 | 53.33 | 46.90 |

## 5. v1 oracle@64 (x100; capture = (scorer−mean64)/(oracle64−mean64))

| model | oracle64 | mean64 | scorer | medoid | capture | regret | pick=0 | pick=oracle | pick pctile | best of scorer top-1/4/8/16 | spearman(logit,score) | safe cand. | minADE64 / ADE scorer / ADE oracle (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D3K1mix | 97.80 | 70.38 | 86.09 | 78.77 | 0.57 | 11.71 | 5.93 | 29.61 | 65.08 | 86.09 / 90.26 / 92.10 / 93.77 | 0.385 | 80.62 | 0.276 / 1.446 / 1.520 |
| D2K1mix | 98.23 | 70.20 | 85.73 | 78.13 | 0.55 | 12.50 | 6.95 | 31.16 | 65.44 | 85.73 / 90.15 / 92.10 / 94.24 | 0.413 | 79.62 | 0.249 / 1.309 / 1.642 |
| D1K1 | 93.99 | 63.83 | 80.36 | 69.97 | 0.55 | 13.63 | 8.32 | 26.49 | 60.60 | 80.36 / 85.63 / 87.74 / 89.70 | 0.378 | 76.04 | 0.852 / 2.104 / 1.991 |

consistency: D3K1mix: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.007, n=12146; D2K1mix: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.005, n=12146; D1K1: argmax(logit64)==idx_scorer 1.0000, npz vs csv max|Δ| 1.11e-16, constant-score tokens 0.005, n=12146

## 6. v1 gap to D2K1mix: scorer part = Δcapture × (oracle64−mean64) of either model; candidate part = rest (x100)

| model | PDMS gap | scorer part | candidate part | oracle64 gap | mean64 gap |
|---|---|---|---|---|---|
| D3K1mix | -0.36 | -0.53 … -0.52 | 0.16 … 0.17 | 0.44 | -0.18 |
| D1K1 | 5.37 | 0.17 … 0.18 | 5.19 … 5.20 | 4.24 | 6.37 |

token→log map: 12146 tokens, 136 logs (from /gs/bs/tga-RLA/qdeng/RAP/metric_cache_v1_navtest)

## 7. paired D3K1mix − D2K1mix (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D3K1mix | D2K1mix | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 86.09 | 85.73 | 0.36 | [-0.41, 1.16] | 25.4 | 25.8 | 48.8 |
| v1_medoid.score | 12146 | 78.77 | 78.13 | 0.65 | [-0.41, 1.70] | 20.7 | 23.9 | 55.4 |
| v2_scorer.score | 12146 | 79.20 | 80.68 | -1.48 | [-2.29, -0.69] | 23.7 | 24.7 | 51.6 |
| oracle64.oracle64 | 12146 | 97.80 | 98.23 | -0.44 | [-0.65, -0.24] | 15.0 | 61.5 | 23.5 |
| oracle64.mean64 | 12146 | 70.38 | 70.20 | 0.18 | [-0.34, 0.74] | 47.3 | 0.5 | 52.2 |
| oracle64.score_scorer | 12146 | 86.09 | 85.73 | 0.36 | [-0.41, 1.16] | 25.4 | 25.8 | 48.8 |
| oracle64.minADE64 | 12146 | 0.276 | 0.249 | 0.027 | [0.020, 0.034] | 58.9 | 0.0 | 41.1 |

## 7. paired D3K1mix − D1K1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D3K1mix | D1K1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 86.09 | 80.36 | 5.73 | [4.72, 6.79] | 49.8 | 23.2 | 27.0 |
| v1_medoid.score | 12146 | 78.77 | 69.97 | 8.80 | [7.31, 10.41] | 39.5 | 19.2 | 41.2 |
| v2_scorer.score | 12146 | 79.20 | 73.70 | 5.50 | [4.37, 6.57] | 49.8 | 21.5 | 28.7 |
| oracle64.oracle64 | 12146 | 97.80 | 93.99 | 3.80 | [3.35, 4.27] | 45.7 | 43.0 | 11.3 |
| oracle64.mean64 | 12146 | 70.38 | 63.83 | 6.55 | [5.47, 7.64] | 59.2 | 0.2 | 40.6 |
| oracle64.score_scorer | 12146 | 86.09 | 80.36 | 5.73 | [4.72, 6.79] | 49.8 | 23.2 | 27.0 |
| oracle64.minADE64 | 12146 | 0.276 | 0.852 | -0.576 | [-0.702, -0.467] | 32.6 | 0.0 | 67.4 |

## 7. paired D2K1mix − D1K1 (per-token mean; 95% CI by log-clustered bootstrap; minADE in m, others x100)

| metric | n | D2K1mix | D1K1 | Δ | 95% CI | Δ>0 % | tie % | Δ<0 % |
|---|---|---|---|---|---|---|---|---|
| v1_scorer.score | 12146 | 85.73 | 80.36 | 5.37 | [4.32, 6.36] | 58.0 | 23.7 | 18.3 |
| v1_medoid.score | 12146 | 78.13 | 69.97 | 8.15 | [6.70, 9.67] | 45.5 | 22.4 | 32.1 |
| v2_scorer.score | 12146 | 80.68 | 73.70 | 6.98 | [5.92, 8.04] | 58.6 | 22.3 | 19.1 |
| oracle64.oracle64 | 12146 | 98.23 | 93.99 | 4.24 | [3.74, 4.75] | 48.2 | 43.8 | 8.0 |
| oracle64.mean64 | 12146 | 70.20 | 63.83 | 6.37 | [5.25, 7.48] | 60.3 | 0.2 | 39.6 |
| oracle64.score_scorer | 12146 | 85.73 | 80.36 | 5.37 | [4.32, 6.36] | 58.0 | 23.7 | 18.3 |
| oracle64.minADE64 | 12146 | 0.249 | 0.852 | -0.603 | [-0.727, -0.494] | 27.6 | 0.0 | 72.4 |
